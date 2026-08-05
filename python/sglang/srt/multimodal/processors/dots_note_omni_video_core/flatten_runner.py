# ruff: noqa
"""flatten_runner.py — video flatten/interleave 策略执行 (core 层, 格式无关)。

承载「怎么把 nested video meta 拍平/交错」的全部**策略**逻辑 (决定抽哪些帧、时间戳、
分几组 K、音频怎么切片、随机格式), 从 template/format.py 挪来。template 层只负责把
本模块产出的 plan 映射成具体输出 schema (nested / flat / ...)。

核心产物 = plan (结构化中间表示), 与序列化无关:
    plan = {
      'seconds_decimals': int,
      'videos': [                       # 按 K 升序
        {'vk': int, 'fmt': 'hms'|'seconds', 'orig_fps': float,
         'emissions': [ {'kind':'frame','ts':float,'b64':str} | {'kind':'audio','b64':str}, ... ]},
        ...
      ],
      'passthrough_meta': {...},         # meta 里非 video_K 的顶层键 (v2 core 一般为空)
      'conversations': [ {from,value,...}, ... ],   # 原始对话 (numpy 已归一化), 供 <video_K> 替换
    }

build_plan 复用训练侧同源 VideoQAFlattener 的 RNG 消耗原语 (_subsample_one_video /
_decide_group_bounds / _decode_wav_b64 / _encode_wav_b64), 且 RNG 消耗顺序与 flatten() 逐一对齐:
    per video: random.choice(fmt) -> _subsample_one_video(scale) -> _decide_group_bounds(K,cuts)
因此 render_flat(build_plan(...)) 与训练侧 flatten() 输出**逐位一致** (见 A/B 测试)。

随机可复现: flatten 用全局 random 模块。build_plan 前用
  seed = sha1(global_seed | 'flatten' | record_key) 低 32 位 random.seed(),
  离线并行(多worker乱序)下每条产物只依赖其 record_key + global_seed。

参数 + 全局 seed 由入口 (convert.py) 通过环境变量注入 (env 跨 fork/spawn):
  DOTS_FLATTEN_SEED / _FPS_SCALE_LO / _HI / _MIN_FRAMES / _MAX_FRAMES
  DOTS_FLATTEN_TIME_FORMAT / _SECONDS_DECIMALS
  DOTS_FLATTEN_AUDIO_INTERLEAVE / _AI_SEG_MIN_SEC / _AI_K_MODE / _AI_SAMPLE_RATE
默认值全部对齐训练 yaml (dots2_video0.7_64k_thinkv2): fps[1,1]/min4/random/interleave logk。
"""
import hashlib
import os
import random
import re

from .video_qa_flattener import (
    VideoQAFlattener, _format_timestamp, _VIDEO_KEY_RE, _VIDEO_MARKER_RE,
)


def _env(k, default=None):
    v = os.environ.get(k, '')
    return v if v != '' else default


def build_flattener():
    """按 env 参数构造 flattener (默认对齐训练 yaml: fps[1,1]/min4/random/interleave logk)。"""
    max_frames = _env('DOTS_FLATTEN_MAX_FRAMES')
    max_frames = int(max_frames) if max_frames not in (None, 'null', 'none', 'None') else None
    return VideoQAFlattener(
        fps_scale_range=(float(_env('DOTS_FLATTEN_FPS_SCALE_LO', 1.0)),
                         float(_env('DOTS_FLATTEN_FPS_SCALE_HI', 1.0))),
        min_frames=int(_env('DOTS_FLATTEN_MIN_FRAMES', 4)),
        max_frames=max_frames,
        time_format=_env('DOTS_FLATTEN_TIME_FORMAT', 'random'),
        seconds_decimals=int(_env('DOTS_FLATTEN_SECONDS_DECIMALS', 1)),
        audio_interleave=_env('DOTS_FLATTEN_AUDIO_INTERLEAVE', '1') in ('1', 'true', 'True'),
        ai_seg_min_sec=float(_env('DOTS_FLATTEN_AI_SEG_MIN_SEC', 1.0)),
        ai_k_mode=_env('DOTS_FLATTEN_AI_K_MODE', 'logk'),  # logk(训练)/eval30(sqrt中位)/eval_ek(E[K]期望,对齐旧评测)/whole(K=1)
        ai_sample_rate=int(_env('DOTS_FLATTEN_AI_SAMPLE_RATE', 16000)),
    )


# flattener 构造无重依赖, 进程内复用一个实例即可 (参数从 env 一次读定)。
_FLATTENER = None


def get_flattener():
    global _FLATTENER
    if _FLATTENER is None:
        _FLATTENER = build_flattener()
    return _FLATTENER


def reset_flattener():
    """清缓存实例 (改 env 后重建时用; 测试也用)。"""
    global _FLATTENER
    _FLATTENER = None


def derive_seed(record_key):
    """确定性派生种子: sha1(global_seed | 'flatten' | record_key) 低 32 位。"""
    gseed = _env('DOTS_FLATTEN_SEED', '42')
    h = hashlib.sha1(f'{gseed}|flatten|{record_key}'.encode('utf-8', 'ignore')).hexdigest()
    return int(h[:8], 16)


def _normalize_convs(conversations):
    """numpy → list; 每条转 dict (与 flatten() 内部一致)。"""
    convs = conversations
    if convs is None:
        convs = []
    if hasattr(convs, 'tolist'):
        convs = convs.tolist()
    out = []
    for c in convs:
        if hasattr(c, 'tolist') and not isinstance(c, dict):
            c = c.tolist()
        if not isinstance(c, dict):
            c = dict(c)
        out.append(dict(c))
    return out


def _plan_interleaved_emissions(fl, kept_frames, kept_ts, audio_b64, video_dict):
    """镜像 VideoQAFlattener._build_interleaved_parts 的**决策**, 但产出结构化 emissions
    (不写 meta / 不拼 tag), 序列化交给上层。RNG 消耗顺序与原实现完全一致:
      _decide_group_bounds (可能 random.uniform + random.sample) -> 解码/切片/编码 (无 RNG)。
    失败兜底与原实现一致: 音频解码异常 -> [所有帧] + [整段音频] 单块。
    """
    n = len(kept_frames)
    duration = float(video_dict.get('audio_duration', 0) or 0)
    if duration <= 0:
        duration = float(kept_ts[-1]) if kept_ts else 0.0
    if duration <= 0:
        duration = float(n)

    group_bounds = fl._decide_group_bounds(n, duration)   # ← RNG (logk: uniform+sample)
    K = len(group_bounds) - 1

    try:
        pcm, sr, _sw, _nch = fl._decode_wav_b64(audio_b64)
    except Exception:
        em = [{'kind': 'frame', 'ts': ts, 'b64': b64} for b64, ts in zip(kept_frames, kept_ts)]
        em.append({'kind': 'audio', 'b64': audio_b64, 'dur': duration})  # v4: 段时长(整段兜底)
        return em

    total_samples = len(pcm)
    em = []
    for g in range(K):
        f0, f1 = group_bounds[g], group_bounds[g + 1]
        if f1 <= f0:
            continue
        t_start = 0.0 if g == 0 else float(kept_ts[f0])
        t_end = duration if g == K - 1 else float(kept_ts[group_bounds[g + 1]])
        if t_end <= t_start:
            t_end = t_start + (duration / max(K, 1))
        for fi in range(f0, f1):
            em.append({'kind': 'frame', 'ts': kept_ts[fi], 'b64': kept_frames[fi]})
        s0 = max(0, int(round(t_start * sr)))
        s1 = min(total_samples, int(round(t_end * sr)))
        if s1 <= s0:
            continue
        seg = pcm[s0:s1]
        em.append({'kind': 'audio', 'b64': fl._encode_wav_b64(seg, sr),
                   'dur': len(seg) / sr if sr else 0.0})  # v4: 该段真实时长, 供 token 按段算
    return em


def build_plan_with(fl, meta, conversations):
    """用给定 flattener 实例 (RNG 由调用方 seed) 构造 plan。镜像 flatten() 的循环+RNG顺序。"""
    old_meta = dict(meta) if meta else {}
    video_pairs = []
    for k, v in old_meta.items():
        m = _VIDEO_KEY_RE.fullmatch(k) if isinstance(k, str) else None
        if m and isinstance(v, dict):
            video_pairs.append((int(m.group(1)), k, v))
    video_pairs.sort()
    passthrough = {k: v for k, v in old_meta.items()
                   if not (isinstance(k, str) and _VIDEO_KEY_RE.fullmatch(k))}

    videos = []
    for vk_int, vk_str, vd in video_pairs:
        # 1) per-video 格式 (random 时 choice; RNG #1)
        if fl.time_format == 'random':
            chosen_fmt = random.choice(['hms', 'seconds'])
        else:
            chosen_fmt = fl.time_format
        # 2) 抽帧 (RNG #2: uniform, 仅 lo!=hi)
        kept_frames, kept_ts, audio_b64 = fl._subsample_one_video(vd)
        # 3) 交错 or 旧逻辑
        use_interleave = (fl.audio_interleave and audio_b64 is not None and len(kept_frames) > 0)
        if not use_interleave:
            emissions = [{'kind': 'frame', 'ts': ts, 'b64': b64}
                         for b64, ts in zip(kept_frames, kept_ts)]
            if audio_b64 is not None:
                emissions.append({'kind': 'audio', 'b64': audio_b64})
        else:
            emissions = _plan_interleaved_emissions(fl, kept_frames, kept_ts, audio_b64, vd)
        # 保留该 video 的音频超参 (供 nested 渲染写回 video 字典; flat 渲染不用)。
        # 训练侧 audio 采样率取全局 audio_data_args.sampling_rate, 这里的 audio_sample_rate/
        # audio_duration 仅作 video 字典内的信息性元数据 (parser 只读 fps+image_*+audio_<n>, 其余透传忽略)。
        audio_hparams = {}
        if 'audio_sample_rate' in vd:
            audio_hparams['audio_sample_rate'] = vd['audio_sample_rate']
        if 'audio_duration' in vd:
            audio_hparams['audio_duration'] = vd['audio_duration']

        videos.append({
            'vk': vk_int,
            'fmt': chosen_fmt,
            'orig_fps': float(vd.get('fps', 1.0)) or 1.0,
            'audio_hparams': audio_hparams,
            'emissions': emissions,
        })

    return {
        'seconds_decimals': fl.seconds_decimals,
        'videos': videos,
        'passthrough_meta': passthrough,
        'conversations': _normalize_convs(conversations),
    }


def build_plan(meta, conversations, record_key):
    """确定性 seed 后构造 plan (供 template 层任意 schema 渲染)。"""
    random.seed(derive_seed(record_key))
    return build_plan_with(get_flattener(), meta, conversations)


def render_flat(plan):
    """把 plan 渲染成训练侧 flat mm 布局 (顶层全局连续 image_K/audio_K + <ts><image_K>/<audio_K> 交错)。
    ★ 与训练侧 VideoQAFlattener.flatten() 输出逐位一致 (A/B 测试据此对拍)。
    """
    new_meta = dict(plan['passthrough_meta'])
    next_img = 0
    next_audio = 0
    replacement = {}
    dec = plan['seconds_decimals']
    for v in plan['videos']:
        parts = []
        for em in v['emissions']:
            if em['kind'] == 'frame':
                gk = f'image_{next_img}'
                new_meta[gk] = em['b64']
                parts.append(f"<{_format_timestamp(em['ts'], fmt=v['fmt'], seconds_decimals=dec)}><{gk}>")
                next_img += 1
            else:
                ak = f'audio_{next_audio}'
                new_meta[ak] = em['b64']
                parts.append(f'<{ak}>')
                next_audio += 1
        replacement[v['vk']] = ''.join(parts)

    new_convs = []
    for c in plan['conversations']:
        value = c.get('value', '') or ''

        def _sub(match):
            return replacement.get(int(match.group(1)), '')

        nc = dict(c)
        nc['value'] = _VIDEO_MARKER_RE.sub(_sub, value)
        new_convs.append(nc)

    return {'meta': new_meta, 'conversations': new_convs, 'data_type': 'mm'}
