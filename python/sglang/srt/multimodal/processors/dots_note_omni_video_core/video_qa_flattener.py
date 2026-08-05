# ruff: noqa
"""
video_qa_flattener.py

把 pack_video.py 输出的 data_type='video' 数据「拍平」成 unified-sft 已识别的
顶层 image_K / audio_K 布局，让现有 _get_mm_id_and_and_mask 能直接接管。

输入(pack_video.py 输出):
    {
      "meta": {
        "max_fps": 2.0,
        "video_0": {"fps": 2.0, "image_0": <b64>, "image_1": <b64>, ...,
                    (optional) "audio_0": <b64_wav>, "audio_duration": 1.5,
                                "audio_sample_rate": 16000},
        "video_1": {...},
      },
      "conversations": [{"from": "user", "value": "<video_0> ... <video_1>"}, ...],
      "data_type": "video"
    }

输出(本类的 flatten() 返回):
    {
      "meta": {
        "max_fps": 2.0,
        "image_0": <b64>, "image_1": <b64>, ..., "image_n": <b64>,   # 跨所有 video 全局连续
        "audio_0": <b64>, ...                                         # 仅原 video_K 含非空 audio_0 才出现
      },
      "conversations": [{"from": "user", "value": "<ts0><image_0><ts1><image_1>...(<audio_0>)? ..."}, ...],
      "data_type": "mm"   # 关键: 改成 mm 让 unified-sft 走原 mm 分支
    }

策略:
  - RandomScale: fps_scale ∈ [lo, hi]; lo==hi 退化为固定。
  - 抽样后均匀 pick frame indices；时间戳按原 fps 映射到真实秒。
  - 音频处理: 自适应 — 哪个 video_K 自带非空 audio_0 就拍出 <audio_K> 标记 + meta["audio_K"]，
    没音轨的 video 完全不引入 audio 相关字段，无需调用方配置开关。
"""

import copy
import json
import random
import re
from typing import Optional, Tuple, List


_VIDEO_KEY_RE = re.compile(r'video_(\d+)')
_VIDEO_MARKER_RE = re.compile(r'<video_(\d+)>')


def _format_timestamp(sec: float, fmt: str = 'hms', seconds_decimals: int = 1) -> str:
    """文本时间戳渲染 (Chen et al., 2024b)。两种格式：
      - fmt='hms': HH:MM:SS.cc，e.g. 3.567 -> '00:00:03.57'（百分秒精度）
      - fmt='seconds': X.X seconds，e.g. 3.567 -> '3.6 seconds'（默认 1 位小数）

    训练时用两种格式做数据增强，模型学到对多种 timecode 表征的鲁棒性，
    用于 video grounding / dense captioning 等时间感知任务。

    HMS 用整数化 total_cs 反推 h/m/s/cc，避免浮点进位被 cap 成 .99 的边界 bug
    （e.g. 3.999 → '00:00:04.00' 而非 '00:00:03.99'）。
    """
    if sec < 0:
        sec = 0.0
    if fmt == 'seconds':
        return f'{sec:.{seconds_decimals}f} seconds'
    # 默认 hms
    total_cs = int(round(sec * 100))
    h  = total_cs // (3600 * 100)
    m  = (total_cs // (60   * 100)) % 60
    s  = (total_cs //         100 ) % 60
    cs =  total_cs %          100
    return f'{h:02d}:{m:02d}:{s:02d}.{cs:02d}'


def _sorted_image_keys(video_dict: dict) -> List[str]:
    """从一个 video_K dict 里捞出所有 image_t 键，按 t 升序。"""
    pairs = []
    for k in video_dict.keys():
        m = re.fullmatch(r'image_(\d+)', k)
        if m:
            pairs.append((int(m.group(1)), k))
    pairs.sort()
    return [k for _, k in pairs]


def _uniform_pick_indices(n_total: int, n_keep: int) -> List[int]:
    """从 [0, n_total) 均匀抽 n_keep 个索引，去重升序。"""
    if n_keep >= n_total:
        return list(range(n_total))
    if n_keep <= 1:
        return [n_total // 2]
    step = (n_total - 1) / (n_keep - 1)
    seen = sorted({round(i * step) for i in range(n_keep)})
    # 去重后可能少一个，用空位补满 (罕见)
    return seen[:n_keep]


class VideoQAFlattener:
    """把 data_type='video' 拍平成 unified-sft 的 mm 布局。

    可插拔: 主流程在 native_vlt.process_unified_sft 入口处先调 is_video_qa() 判定，
    再 flatten() 改写 data，下游 mm 路径完全不知道有 video 这回事。
    """

    def __init__(
        self,
        fps_scale_range: Tuple[float, float] = (1.0, 1.0),
        min_frames: int = 1,
        max_frames: Optional[int] = None,
        time_format: str = 'random',          # 'hms' | 'seconds' | 'random'
        seconds_decimals: int = 1,            # seconds 格式保留小数位数
        # ===== 音视频交错 (audio_interleave) 参数; 默认全关 = 旧"整段音频堆最后"行为 =====
        # 交错方案 (定稿): 音频按【帧时间戳 list 边界】切成 K 段, 每段 = [组内帧(带绝对时间戳) + 该组时间窗音频段]。
        #   - 单套时间戳: 只有帧带时间戳(绝对秒), 音频段不带独立戳(共用帧戳, 对齐 Qwen2.5-Omni audio-video 路线)。
        #   - K 选取 (B 方案): K_max=min(N_frames, audio_dur//ai_seg_min_sec);
        #       训练(logk): K=round(exp(uniform(0,log(K_max)))) 对数均匀; 切点=random.sample(N-1个帧间隙取K-1) 位置随机。
        #       评测(eval): K=round(sqrt(K_max)) 取中位; 切点均匀(可复现)。 whole: K=1 整段。
        #   - 无独立音频插戳 (音视频数据靠帧戳锚定, 不像纯音频那样 random-interval 插戳)。
        audio_interleave: bool = False,       # False(默认)=旧逻辑; True=帧list边界切段交错
        ai_seg_min_sec: float = 1.0,          # 每段最小时长(s): whisper 有效编码下限护栏 (防短段碎片); 控 K_max
        ai_k_mode: str = 'eval30',            # 'logk'(训练:logK随机) | 'eval30'(评测:sqrt(K_max)中位) | 'eval_ek'(评测:E[K]期望, 对齐旧评测口径) | 'whole'(K=1)
        ai_sample_rate: int = 16000,          # 音频采样率(切段wav重编码用)
    ):
        lo, hi = float(fps_scale_range[0]), float(fps_scale_range[1])
        assert 0 < lo <= hi <= 1.0, f'fps_scale_range must satisfy 0 < lo <= hi <= 1.0, got {fps_scale_range}'
        if min_frames < 1:
            min_frames = 1
        assert time_format in ('hms', 'seconds', 'random'), \
            f"time_format must be 'hms'/'seconds'/'random', got {time_format!r}"
        self.fps_scale_lo = lo
        self.fps_scale_hi = hi
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.time_format = time_format        # sample-level: 整条 sample 的所有时间戳同格式
        self.seconds_decimals = max(0, int(seconds_decimals))
        # 交错参数
        self.audio_interleave = bool(audio_interleave)
        self.ai_seg_min_sec = max(1e-6, float(ai_seg_min_sec))
        assert ai_k_mode in ('logk', 'eval30', 'eval_ek', 'whole'), \
            f"ai_k_mode must be 'logk'/'eval30'/'eval_ek'/'whole', got {ai_k_mode!r}"
        self.ai_k_mode = ai_k_mode
        self.ai_sample_rate = int(ai_sample_rate)

    # ---------- 判定 / 入口 ----------

    @staticmethod
    def is_video_qa(data: dict) -> bool:
        """判定是否走拍平: data_type == 'video' 且 meta 里至少有一个 video_K。

        兼容 pack_video.py 输出: meta 列在 parquet 中存为 JSON string 节省 schema 复杂度。
        本函数检测到 str 时自动 json.loads 并写回 data['meta']，让下游 flatten / process_unified_sft
        都按已解码的 dict 看待，无需改动后续代码。
        """
        if not isinstance(data, dict):
            return False
        if data.get('data_type') != 'video':
            return False
        meta = data.get('meta')
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
                data['meta'] = meta  # 写回 dict，下游 flatten() 直接用
            except Exception:
                return False
        if not isinstance(meta, dict):
            return False
        for k in meta.keys():
            if _VIDEO_KEY_RE.fullmatch(k):
                return True
        return False

    # ---------- 抽样 ----------

    def _sample_n_keep(self, n_total: int) -> int:
        if n_total <= 0:
            return 0
        if self.fps_scale_lo == self.fps_scale_hi:
            scale = self.fps_scale_lo
        else:
            scale = random.uniform(self.fps_scale_lo, self.fps_scale_hi)
        n_keep = max(self.min_frames, round(n_total * scale))
        if self.max_frames is not None and self.max_frames > 0:
            n_keep = min(n_keep, self.max_frames)
        return min(n_keep, n_total)

    def _subsample_one_video(self, video_dict: dict):
        """返回 (kept_b64_frames, kept_timestamps_sec, audio_b64_or_None)。
        音频自适应: 若 video_dict 里存在非空 audio_0 则带出，否则返回 None。
        """
        original_fps = float(video_dict.get('fps', 1.0)) or 1.0
        all_keys = _sorted_image_keys(video_dict)
        n_total = len(all_keys)
        n_keep = self._sample_n_keep(n_total)

        if n_keep == 0:
            kept_frames, kept_ts = [], []
        else:
            indices = _uniform_pick_indices(n_total, n_keep)
            kept_frames = [video_dict[all_keys[i]] for i in indices]
            kept_ts = [round(i / original_fps, 3) for i in indices]

        audio_b64 = video_dict.get('audio_0') or None
        return kept_frames, kept_ts, audio_b64

    # ---------- 音视频交错 (audio_interleave=True 时启用) ----------

    def _decode_wav_b64(self, audio_b64: str):
        """audio_0 (wav b64) -> (pcm_int16_1d_ndarray, sr, sampwidth, nchannels)。"""
        import io as _io
        import wave as _wave
        import base64 as _b64
        import numpy as _np
        raw = _b64.b64decode(audio_b64)
        with _wave.open(_io.BytesIO(raw), 'rb') as w:
            sr = w.getframerate()
            sw = w.getsampwidth()
            nch = w.getnchannels()
            n = w.getnframes()
            pcm = _np.frombuffer(w.readframes(n), dtype=_np.int16)
        if nch > 1:
            pcm = pcm.reshape(-1, nch).mean(axis=1).astype(_np.int16)
        return pcm, sr, sw, nch

    def _encode_wav_b64(self, pcm, sr: int):
        """pcm_int16_1d -> wav b64 (单声道16bit)。路径A: 切块后每块重新wav编码。"""
        import io as _io
        import wave as _wave
        import base64 as _b64
        buf = _io.BytesIO()
        with _wave.open(buf, 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm.tobytes())
        return _b64.b64encode(buf.getvalue()).decode('ascii')

    def _decide_group_bounds(self, n_frames: int, duration: float):
        """决定把 n 帧分成 K 组的【组边界】(=音频按帧时间戳边界切 K 段)。返回 group_bounds(长 K+1, 升序, [0]=0,[-1]=n)。

        K 选取 (B 方案, 段数与切点位置正交):
          K_max = min(n_frames, floor(duration / ai_seg_min_sec))   # 帧数 ∩ whisper最小段护栏
          'whole'  -> K=1 (整段)
          'logk'   -> 训练: K=round(exp(uniform(0,log(K_max)))) 对数均匀 (覆盖 1..K_max 全谱系粒度)
          'eval30' -> 评测: K=round(sqrt(K_max)) 取对数中位 (可复现)
        切点位置:
          logk -> random.sample(range(1,n), K-1) 随机选 K-1 个帧间隙 (位置随机, 破固定切法)
          其他 -> 均匀切点 round(i*n/K) (可复现)
        """
        import math as _math
        if n_frames <= 1 or duration <= 0:
            return [0, n_frames] if n_frames >= 1 else [0, 0]
        # K_max: 不超帧数(每组≥1帧), 且每段≥ai_seg_min_sec(防whisper短段碎片)
        k_max = min(n_frames, max(1, int(duration // self.ai_seg_min_sec)))
        if self.ai_k_mode == 'whole' or k_max <= 1:
            K = 1
        elif self.ai_k_mode == 'eval30':
            # 评测: 取对数中位 sqrt(K_max), 可复现
            K = max(1, min(k_max, int(round(_math.sqrt(k_max)))))
        elif self.ai_k_mode == 'eval_ek':
            # 评测: 取 logk 分布的期望 E[K]=(K_max-1)/ln(K_max), 更代表训练平均粒度, 可复现
            # (与旧评测 gpt.py _vi_build_interleaved 的 E[K] 口径一致; k_max>1 上方已保证, log 安全)
            K = max(1, min(k_max, int(round((k_max - 1) / _math.log(k_max)))))
        else:  # logk: 训练对数均匀
            K = max(1, min(k_max, int(round(_math.exp(random.uniform(0.0, _math.log(k_max)))))))
        if K <= 1:
            return [0, n_frames]
        # 切点位置: logk 随机选 K-1 个帧间隙; 其他均匀
        if self.ai_k_mode == 'logk':
            cuts = sorted(random.sample(range(1, n_frames), K - 1))
        else:
            cuts = sorted(set(round(i * n_frames / K) for i in range(1, K)))
            cuts = [c for c in cuts if 0 < c < n_frames]
        return [0] + cuts + [n_frames]

    def _build_interleaved_parts(self, kept_frames, kept_ts, audio_b64, video_dict,
                                 chosen_fmt, new_meta, next_img_idx, next_audio_idx):
        """把一个 video 的帧按【帧时间戳边界】分 K 组, 每组 [组内帧(带绝对时间戳) + 该组时间窗音频段] 交错。
        单套时间戳: 只有帧带时间戳(绝对秒), 音频段不带独立戳(共用帧戳)。
        音频段按帧时间戳边界切(snap到帧时刻), 首段含[0,..], 尾段含[..,duration]。
        返回 (parts_list, next_img_idx, next_audio_idx)。

        失败兜底: 音频解码异常 -> 回退为"帧交错但音频整段堆该video末尾", 不中断。
        """
        n = len(kept_frames)
        # 音频时长: 优先用 meta 的 audio_duration, 否则用末帧时间戳
        duration = float(video_dict.get('audio_duration', 0) or 0)
        if duration <= 0:
            duration = float(kept_ts[-1]) if kept_ts else 0.0
        if duration <= 0:
            duration = float(n)  # 极端兜底

        # 决定组边界 (K 段, 帧时间戳边界切; B 方案)
        group_bounds = self._decide_group_bounds(n, duration)
        K = len(group_bounds) - 1

        # 解码整段音频(一次), 失败则回退
        try:
            pcm, sr, _sw, _nch = self._decode_wav_b64(audio_b64)
        except Exception:
            # 回退: 帧交错布局照常, 音频整段作单块放该 video 末尾
            parts = []
            for b64, ts in zip(kept_frames, kept_ts):
                gk = f'image_{next_img_idx}'
                new_meta[gk] = b64
                parts.append(f'<{_format_timestamp(ts, fmt=chosen_fmt, seconds_decimals=self.seconds_decimals)}><{gk}>')
                next_img_idx += 1
            ak = f'audio_{next_audio_idx}'
            new_meta[ak] = audio_b64
            parts.append(f'<{ak}>')
            next_audio_idx += 1
            return parts, next_img_idx, next_audio_idx

        total_samples = len(pcm)
        parts = []
        for g in range(K):
            f0, f1 = group_bounds[g], group_bounds[g + 1]
            if f1 <= f0:
                continue
            # 该组帧的时间窗: [t_start, t_end)
            # t_start: 首组=0, 其余=该组首帧时间戳; t_end: 末组=duration, 其余=下组首帧时间戳
            t_start = 0.0 if g == 0 else float(kept_ts[f0])
            t_end = duration if g == K - 1 else float(kept_ts[group_bounds[g + 1]])
            if t_end <= t_start:
                t_end = t_start + (duration / max(K, 1))

            # 该组的帧(带时间戳)
            for fi in range(f0, f1):
                gk = f'image_{next_img_idx}'
                new_meta[gk] = kept_frames[fi]
                parts.append(f'<{_format_timestamp(kept_ts[fi], fmt=chosen_fmt, seconds_decimals=self.seconds_decimals)}><{gk}>')
                next_img_idx += 1

            # 该组的音频段: 按时间窗切 PCM(snap到样本)。音频段不带独立时间戳(共用上面帧的绝对戳)。
            s0 = max(0, int(round(t_start * sr)))
            s1 = min(total_samples, int(round(t_end * sr)))
            if s1 <= s0:
                continue  # 该窗无音频(罕见), 跳过该段
            seg = pcm[s0:s1]
            ak = f'audio_{next_audio_idx}'
            new_meta[ak] = self._encode_wav_b64(seg, sr)
            parts.append(f'<{ak}>')
            next_audio_idx += 1

        return parts, next_img_idx, next_audio_idx

    # ---------- 主入口 ----------

    def flatten(self, data: dict) -> dict:
        """返回新 data dict（深拷贝 conversations + 重建 meta），原 data 不动。

        时间戳格式（time_format='random' 时）：per-video-level 选定。
          - 每个原始 video 独立 random.choice(['hms','seconds'])
          - 同一 video 内的所有时间戳保持一致（视觉/上下文连贯）
          - 不同 video 之间互相独立 → 单 forward context 内大概率同时出现两种格式
        参考 Chen et al., 2024b textual-time augmentation：让模型在单个 context 内
        同时见到两种 timecode 表征，显式学到两者语义等价。
        """
        assert self.is_video_qa(data), 'flatten() called on non-video-qa data; call is_video_qa() first'

        old_meta = data['meta']
        # 收集所有 video_K 并按 K 升序
        video_pairs = []
        for k, v in old_meta.items():
            m = _VIDEO_KEY_RE.fullmatch(k)
            if m and isinstance(v, dict):
                video_pairs.append((int(m.group(1)), k, v))
        video_pairs.sort()

        new_meta = {k: v for k, v in old_meta.items() if not _VIDEO_KEY_RE.fullmatch(k)}
        # 全局递增计数器
        next_img_idx = 0
        next_audio_idx = 0
        # video_K -> 替换文本块
        replacement: dict = {}
        # 追踪冲突: 拍平后的 image_K 不能跟原 meta 已有的 image_K 撞
        for vk_int, vk_str, vd in video_pairs:
            # per-video-level: 每个原始 video 独立选格式（同 video 内一致）
            if self.time_format == 'random':
                chosen_fmt = random.choice(['hms', 'seconds'])
            else:
                chosen_fmt = self.time_format
            kept_frames, kept_ts, audio_b64 = self._subsample_one_video(vd)

            # 是否走交错: 开关开 且 该 video 有音频 且 有帧。否则走旧"整段堆最后"逻辑。
            use_interleave = (
                self.audio_interleave and audio_b64 is not None and len(kept_frames) > 0
            )

            if not use_interleave:
                # ===== 旧逻辑 (默认): 帧序列 + 整段音频堆最后 =====
                parts = []
                for b64, ts in zip(kept_frames, kept_ts):
                    global_key = f'image_{next_img_idx}'
                    if global_key in new_meta:
                        raise ValueError(
                            f"flatten conflict: '{global_key}' already exists in meta "
                            f"(possibly raw image marker collides with video frame). "
                            f"data_type=video should not contain top-level image_K markers."
                        )
                    new_meta[global_key] = b64
                    parts.append(f'<{_format_timestamp(ts, fmt=chosen_fmt, seconds_decimals=self.seconds_decimals)}><{global_key}>')
                    next_img_idx += 1
                if audio_b64 is not None:
                    global_audio_key = f'audio_{next_audio_idx}'
                    if global_audio_key in new_meta:
                        raise ValueError(f"flatten conflict: '{global_audio_key}' already in meta")
                    new_meta[global_audio_key] = audio_b64
                    parts.append(f'<{global_audio_key}>')
                    next_audio_idx += 1
                replacement[vk_int] = ''.join(parts)
            else:
                # ===== 交错逻辑: 按时段把帧分 K 组, 每组 帧 + 该时段音频块 交错 =====
                parts, next_img_idx, next_audio_idx = self._build_interleaved_parts(
                    kept_frames, kept_ts, audio_b64, vd, chosen_fmt,
                    new_meta, next_img_idx, next_audio_idx,
                )
                replacement[vk_int] = ''.join(parts)

        # 重写 conversations 中所有 <video_K>
        # parquet 反序列化后 conversations 列可能是 numpy.ndarray（不能直接做 `or []` 判断），统一转 list
        convs = data.get('conversations', [])
        if convs is None:
            convs = []
        if hasattr(convs, 'tolist'):
            convs = convs.tolist()
        new_convs = []
        for c in convs:
            # c 也可能是 numpy struct/dict-like，统一转 dict
            if hasattr(c, 'tolist') and not isinstance(c, dict):
                c = c.tolist()
            if not isinstance(c, dict):
                c = dict(c)
            value = c.get('value', '') or ''

            def _sub(match):
                k = int(match.group(1))
                # 不在 meta 里的引用 -> 删掉这个 marker (静默忽略)
                return replacement.get(k, '')

            new_value = _VIDEO_MARKER_RE.sub(_sub, value)
            nc = dict(c)
            nc['value'] = new_value
            new_convs.append(nc)

        new_data = {
            **{k: v for k, v in data.items() if k not in ('meta', 'conversations', 'data_type')},
            'meta': new_meta,
            'conversations': new_convs,
            'data_type': 'mm',  # 关键: 让 unified-sft 走 mm 分支
        }
        return new_data
