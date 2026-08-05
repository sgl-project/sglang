# ruff: noqa
"""preprocess.py — video pack_algo=v2 端到端前处理核心 (格式无关引擎)。

从 response_generate_policy/v1.5 剥离而来, 函数体与源逐位一致 (防漂移):
  - pack_video.py: token 估算 / extract_frames(_v2) / extract_audio_wav / process_sample
  - pack.py:      _record_wants_audio / process_sample_video / inject_user_query_suffix / _normalize_*

think 与非think 两条路径都保留:
  - 非think: user_query_suffix='' (no-op), assistant 直接答案。
  - think:   user_query_suffix='<think_query>', 在算 token 前注入每轮 user query 末尾。

只做「前处理」: 单样本 → (nested_meta, conversations, vision_tokens, text_tokens)。
不含 bin-pack / merge / 全局重编号 (那些在 pack.py 里, 本引擎丢弃)。

纯算法常量/函数从 v2core import (single-source-of-truth, 与 pack_video.py 同款做法):
  compute_target_size / round_by_factor 等仍在本文件本地定义 (与 pack_video.py 一致),
  仅 V2_* 常量 + v2_ 算法函数从 v2core import。
"""
import base64
import io
import math
import os
# 多 worker 大并发场景下, 每个 worker 内部的线程池默认 = nproc, 必须在 import torch/torchcodec
# 之前 setdefault, 让所有内层库只开 1 线程。
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
           'GOMP_NUM_THREADS', 'OMP_THREAD_LIMIT', 'MKL_DOMAIN_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_DYNAMIC', 'FALSE')
os.environ.setdefault('OMP_NESTED', 'FALSE')

import hashlib
import re

import numpy as np
from PIL import Image
from torchcodec.decoders import VideoDecoder


PATCH_SIZE = 14
MERGE_SIZE = 2
ALIGN = PATCH_SIZE * MERGE_SIZE  # 28
DEFAULT_VIDEO_MIN_PIXELS = 100352   # 14*14*512   (yaml: video_min_pixels)
DEFAULT_VIDEO_MAX_PIXELS = 802816 # 602112   # 14*14*3072  (yaml: video_max_pixels)
DEFAULT_VIDEO_TOTAL_PIXELS = 175616000  # 14*14*896000 (yaml: video_total_pixels, 256k 刷取)

FRAME_FACTOR = 1          # nframes 取整因子; 同时也是 max_pixels 公式里的 temporal merge 因子
FPS_MIN_FRAMES = 4        # 每个 video 至少这么多帧
FPS_MAX_FRAMES = 1024 # 768      # 每个 video 至多这么多帧

DEFAULT_SYSTEM_PROMPT = 'You are a helpful assistant.'

# 训练侧的 special token 字符串 (来自 cybertron/config/const.py)
TOK_USER_START   = '<|user|>'
TOK_USER_END     = '<|endofuser|>'
TOK_ASSIST_START = '<|assistant|>'
TOK_ASSIST_END   = '<|endofassistant|>'
TOK_SYS_START    = '<|system|>'
TOK_SYS_END      = '<|endofsystem|>'
TOK_IM_START     = '<|img|>'
TOK_IM_END       = '<|endofimg|>'
TOK_IM_PAD       = '<|imgpad|>'
TOK_AUDIO_START  = '<|audio_comp_start|>'
TOK_AUDIO_END    = '<|audio_comp_end|>'
TOK_AUDIO_PAD    = '<|audio_comp_pad|>'

TIMESTAMP_TOKENS = 13
IMG_WRAP_TOKENS = 2
AUDIO_WRAP_TOKENS = 2
ROLE_WRAP_TOKENS = 2

# 默认音频参数
DEFAULT_AUDIO_SAMPLE_RATE = 16000
DEFAULT_AUDIO_SAMPLES_PER_TOKEN = 1280   # 160(mel hop) × 8 (2×2×2 conv strides 后再 //4 reshape)
DEFAULT_AUDIO_CHUNK_SEC = 30             # Whisper 标准 chunk
DEFAULT_AUDIO_LINEAR_TOKENS_PER_SEC = 0.0

# v2 纯算法常量/函数从独立无依赖模块 import (single-source-of-truth, 防漂移)。
from .v2core import (
    V2_FPS_CAP, V2_FPS_MIN, V2_PF_FLOOR, V2_PF_CEIL, V2_OVH,
    INTERLEAVE_SEG_MIN_SEC,
    v2_frame_hardcap, v2_solve_degrade, v2_split_visual_budget,
)


class SkipSample(Exception):
    pass


# ----- 真实 tokenizer 驱动的 token 计算 -----------------------------------

_TOKENIZER = None


def set_tokenizer(tokenizer):
    """Use the tokenizer already loaded by the serving model."""
    global _TOKENIZER
    if tokenizer is None:
        raise ValueError('dots video preprocessing requires a server tokenizer')
    _TOKENIZER = tokenizer


def get_tokenizer():
    if _TOKENIZER is None:
        raise RuntimeError(
            'dots video preprocessing tokenizer was not initialized; '
            'call set_tokenizer() with the server model tokenizer'
        )
    return _TOKENIZER


def tokenize_len(text, tok=None):
    """text → token 数 (不加 special tokens)"""
    if tok is None:
        tok = get_tokenizer()
    if not text:
        return 0
    return len(tok(text, add_special_tokens=False)['input_ids'])


def system_block_tokens(prompt=DEFAULT_SYSTEM_PROMPT, tok=None):
    """<|system|>{prompt}<|endofsystem|>\\n  整块 token 数"""
    return tokenize_len(f'{TOK_SYS_START}{prompt}{TOK_SYS_END}\n', tok=tok)


def round_by_factor(n, factor):
    return round(n / factor) * factor


def floor_by_factor(n, factor):
    return math.floor(n / factor) * factor


def ceil_by_factor(n, factor):
    return math.ceil(n / factor) * factor


def pick_nframes(duration, video_total_frames, max_fps,
                 min_frames=FPS_MIN_FRAMES, max_frames=FPS_MAX_FRAMES,
                 frame_factor=FRAME_FACTOR):
    """与 fetch_video 对齐的帧数选择"""
    nframes = round_by_factor(duration * max_fps, frame_factor)
    nframes = max(min_frames, min(nframes, max_frames))
    nframes = min(nframes, floor_by_factor(video_total_frames, frame_factor))
    nframes = max(frame_factor, nframes)
    return nframes


def pick_max_pixels(nframes, min_pixels, max_pixels, total_pixels,
                    frame_factor=FRAME_FACTOR):
    """与 fetch_video 对齐的 per-frame max_pixels 选择"""
    eff_max = max(
        min(max_pixels, int(total_pixels / nframes * frame_factor)),
        min_pixels
    )
    return eff_max


def compute_target_size(orig_h, orig_w, min_pixels, max_pixels):
    """对齐到 28 倍数; 约束在 [min_pixels, max_pixels]

    ★ v3 对齐新 dots3 训练侧 image_processor_native.smart_resize: min-branch 放大到 min_pixels 后,
      若仍超 max_pixels (极端长宽比小图), 再做一次 max_pixels floor-clamp (max_pixels 优先控 token 长度)。
      v2/pack 无此二次 clamp; v3 补上以与新 dots3 逐位一致。
    """
    h = max(ALIGN, round(orig_h / ALIGN) * ALIGN)
    w = max(ALIGN, round(orig_w / ALIGN) * ALIGN)
    if h * w > max_pixels:
        beta = math.sqrt(orig_h * orig_w / max_pixels)
        h = max(ALIGN, math.floor(orig_h / beta / ALIGN) * ALIGN)
        w = max(ALIGN, math.floor(orig_w / beta / ALIGN) * ALIGN)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / max(1, orig_h * orig_w))
        h = math.ceil(orig_h * beta / ALIGN) * ALIGN
        w = math.ceil(orig_w * beta / ALIGN) * ALIGN
        if h * w > max_pixels:  # max_pixels first to control the token length
            beta = math.sqrt(h * w / max_pixels)
            h = max(ALIGN, math.floor(h / beta / ALIGN) * ALIGN)
            w = max(ALIGN, math.floor(w / beta / ALIGN) * ALIGN)
    return int(h), int(w)


def patches_per_frame(h, w):
    """单帧经 ViT patch (14) + pixel_shuffle (merge=2) 后的 token 数: (h//28)*(w//28)"""
    return (h // ALIGN) * (w // ALIGN)


def frame_block_tokens(h, w):
    """单帧渲染成 '<ts><|img|><|imgpad|>×N<|endofimg|>' 的 token 数"""
    return TIMESTAMP_TOKENS + IMG_WRAP_TOKENS + patches_per_frame(h, w)


def video_segment_tokens(num_frames, h, w):
    """一个完整 video 段 (多帧串起来) 的 token 数"""
    return num_frames * frame_block_tokens(h, w)


def audio_block_tokens(duration_sec,
                       samples_per_token=DEFAULT_AUDIO_SAMPLES_PER_TOKEN,
                       chunk_sec=DEFAULT_AUDIO_CHUNK_SEC,
                       sr=DEFAULT_AUDIO_SAMPLE_RATE,
                       linear_tokens_per_sec=0.0):
    """一段音频渲染成 '<|audio_comp_start|><|audio_comp_pad|>×N<|audio_comp_end|>' 的 token 数。"""
    if duration_sec is None or duration_sec <= 0:
        return 0
    if linear_tokens_per_sec and linear_tokens_per_sec > 0:
        return AUDIO_WRAP_TOKENS + math.ceil(duration_sec * linear_tokens_per_sec)
    total_samples = int(duration_sec * sr)
    chunk_samples = chunk_sec * sr
    n_pad = 0
    pos = 0
    while pos < total_samples:
        L = min(chunk_samples, total_samples - pos)
        n_pad += (L - 1) // samples_per_token + 1
        pos += chunk_samples
    return AUDIO_WRAP_TOKENS + n_pad


def conversation_tokens(convs, video_meta, tok=None, audio_meta=None,
                        audio_samples_per_token=DEFAULT_AUDIO_SAMPLES_PER_TOKEN,
                        audio_chunk_sec=DEFAULT_AUDIO_CHUNK_SEC,
                        audio_sr=DEFAULT_AUDIO_SAMPLE_RATE,
                        audio_linear_tokens_per_sec=0.0):
    """计算 conversations 总 token 数 (不含 system)。
    video_meta: dict, video_K -> (num_frames, h, w)
    audio_meta: dict (可选), video_K -> audio_duration_sec 或 【每段时长列表 [d0,d1,...]】。
    ★ v4: audio 支持按段算 —— audio_meta[vk] 若为 list, 每段单独 audio_block_tokens 加总
      (每段各含 AUDIO_WRAP + 独立取整, 与训练侧对每个 audio 占位符分别 encode 对齐)。
      向后兼容: 标量(旧口径整段)仍按整段算一次。
    """
    if tok is None:
        tok = get_tokenizer()
    audio_meta = audio_meta or {}
    total = 0
    for c in convs:
        v = c.get('value', '') or ''
        total += ROLE_WRAP_TOKENS  # role_start + role_end
        parts = re.split(r'<video_(\d+)>', v)
        for i, seg in enumerate(parts):
            if i % 2 == 0:
                if seg:
                    total += tokenize_len(seg, tok=tok)
            else:
                vk = int(seg)
                if vk not in video_meta:
                    total += tokenize_len(f'<video_{vk}>', tok=tok)
                    continue
                n_frames, h, w = video_meta[vk]
                total += video_segment_tokens(n_frames, h, w)
                dur = audio_meta.get(vk)
                if dur:
                    seg_durs = dur if isinstance(dur, (list, tuple)) else [dur]
                    for d in seg_durs:
                        if d and d > 0:
                            total += audio_block_tokens(
                                d,
                                samples_per_token=audio_samples_per_token,
                                chunk_sec=audio_chunk_sec,
                                sr=audio_sr,
                                linear_tokens_per_sec=audio_linear_tokens_per_sec,
                            )
    return total


def extract_frames(video_bytes, max_fps,
                   min_pixels=DEFAULT_VIDEO_MIN_PIXELS,
                   max_pixels=DEFAULT_VIDEO_MAX_PIXELS,
                   total_pixels=DEFAULT_VIDEO_TOTAL_PIXELS,
                   min_frames=FPS_MIN_FRAMES,
                   max_frames=FPS_MAX_FRAMES,
                   frame_factor=FRAME_FACTOR,
                   jpeg_quality=85):
    """解码 mp4 bytes → 抽帧 → resize → 编码 jpeg (v1, 与 fetch_video 对齐)"""
    try:
        decoder = VideoDecoder(video_bytes, dimension_order='NHWC',
                               num_ffmpeg_threads=1, seek_mode='approximate')
    except TypeError:
        decoder = VideoDecoder(video_bytes, dimension_order='NHWC',
                               num_ffmpeg_threads=1)
    md = decoder.metadata
    duration = float(md.duration_seconds or 0)
    orig_h = int(md.height)
    orig_w = int(md.width)
    total_frames = int(getattr(md, 'num_frames', 0) or 0)
    if duration <= 0 or orig_h <= 0 or orig_w <= 0:
        raise ValueError(f'bad metadata: dur={duration} h={orig_h} w={orig_w}')
    if total_frames <= 0:
        avg_fps = float(getattr(md, 'average_fps', 0) or 0) or 25.0
        total_frames = max(1, int(duration * avg_fps))

    target_n = pick_nframes(
        duration, total_frames, max_fps,
        min_frames=min_frames, max_frames=max_frames, frame_factor=frame_factor,
    )
    actual_fps = round(target_n / max(duration, 1e-6), 4)

    if target_n == 1:
        indices = [0]
    else:
        step = (total_frames - 1) / (target_n - 1)
        indices = [int(round(i * step)) for i in range(target_n)]
    indices = sorted({max(0, min(idx, total_frames - 1)) for idx in indices})
    if not indices:
        indices = [0]

    eff_max_pixels = pick_max_pixels(
        len(indices), min_pixels, max_pixels, total_pixels,
        frame_factor=frame_factor,
    )
    target_h, target_w = compute_target_size(orig_h, orig_w, min_pixels, eff_max_pixels)

    try:
        batch = decoder.get_frames_at(indices=indices)
    except (IndexError, RuntimeError):
        safe = [i for i in indices if i < total_frames]
        while safe and safe[-1] > 0:
            try:
                batch = decoder.get_frames_at(indices=safe)
                break
            except (IndexError, RuntimeError):
                safe = safe[:-1]
        else:
            raise
    frames = batch.data  # (N, H, W, C) uint8

    avg_fps_orig = float(getattr(md, 'average_fps', 0) or 0) or (total_frames / max(duration, 1e-6))
    indices = indices[:len(frames)]
    timestamps = [round(idx / avg_fps_orig, 3) for idx in indices]
    actual_fps = round(len(frames) / max(duration, 1e-6), 4)

    results = []
    for i, frame in enumerate(frames):
        arr = frame.numpy() if hasattr(frame, 'numpy') else frame
        img = Image.fromarray(arr)
        if img.size != (target_w, target_h):
            img = img.resize((target_w, target_h), Image.BICUBIC)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=jpeg_quality)
        b64 = base64.b64encode(buf.getvalue()).decode('ascii')
        results.append((timestamps[i], b64))

    return results, duration, target_h, target_w, actual_fps


# ============================================================================
# pack_algo v2: 联合降级抽帧 (joint degradation)
# ============================================================================

def extract_frames_v2(video_bytes, seq_length, visual_budget,
                      pf_floor=V2_PF_FLOOR, pf_ceil=V2_PF_CEIL,
                      fps_cap=V2_FPS_CAP, fps_min=V2_FPS_MIN, ovh=V2_OVH,
                      jpeg_quality=85):
    """pack_algo v2 抽帧: 联合降级。与 v1 extract_frames 接口对齐的返回值。"""
    try:
        decoder = VideoDecoder(video_bytes, dimension_order='NHWC',
                               num_ffmpeg_threads=1, seek_mode='approximate')
    except TypeError:
        decoder = VideoDecoder(video_bytes, dimension_order='NHWC',
                               num_ffmpeg_threads=1)
    md = decoder.metadata
    duration = float(md.duration_seconds or 0)
    orig_h = int(md.height)
    orig_w = int(md.width)
    total_frames = int(getattr(md, 'num_frames', 0) or 0)
    if duration <= 0 or orig_h <= 0 or orig_w <= 0:
        raise ValueError(f'bad metadata: dur={duration} h={orig_h} w={orig_w}')
    avg_fps_orig = float(getattr(md, 'average_fps', 0) or 0) or 25.0
    if total_frames <= 0:
        total_frames = max(1, int(duration * avg_fps_orig))

    orig_h28 = max(ALIGN, round_by_factor(orig_h, ALIGN))
    orig_w28 = max(ALIGN, round_by_factor(orig_w, ALIGN))
    orig_max_pf = (orig_h28 // ALIGN) * (orig_w28 // ALIGN)

    nframes, target_fps, target_pf = v2_solve_degrade(
        visual_budget, duration, orig_max_pf, avg_fps_orig, seq_length,
        fps_cap=fps_cap, fps_min=fps_min, pf_floor=pf_floor, pf_ceil=pf_ceil, ovh=ovh,
        orig_h=orig_h, orig_w=orig_w,
    )

    eff_max_pixels = min(target_pf * ALIGN * ALIGN, orig_max_pf * ALIGN * ALIGN)
    target_h, target_w = compute_target_size(orig_h, orig_w, pf_floor * ALIGN * ALIGN, eff_max_pixels)

    nframes = max(FPS_MIN_FRAMES, min(nframes, total_frames))
    if nframes == 1:
        indices = [0]
    else:
        step = (total_frames - 1) / (nframes - 1)
        indices = [int(round(i * step)) for i in range(nframes)]
    indices = sorted({max(0, min(idx, total_frames - 1)) for idx in indices})
    if not indices:
        indices = [0]

    try:
        batch = decoder.get_frames_at(indices=indices)
    except (IndexError, RuntimeError):
        safe = [i for i in indices if i < total_frames]
        while safe and safe[-1] > 0:
            try:
                batch = decoder.get_frames_at(indices=safe)
                break
            except (IndexError, RuntimeError):
                safe = safe[:-1]
        else:
            raise
    frames = batch.data
    indices = indices[:len(frames)]
    timestamps = [round(idx / avg_fps_orig, 3) for idx in indices]
    actual_fps = round(len(frames) / max(duration, 1e-6), 4)

    results = []
    for i, frame in enumerate(frames):
        arr = frame.numpy() if hasattr(frame, 'numpy') else frame
        img = Image.fromarray(arr)
        if img.size != (target_w, target_h):
            img = img.resize((target_w, target_h), Image.BICUBIC)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=jpeg_quality)
        b64 = base64.b64encode(buf.getvalue()).decode('ascii')
        results.append((timestamps[i], b64))

    return results, duration, target_h, target_w, actual_fps


def extract_audio_wav(video_bytes, sample_rate=DEFAULT_AUDIO_SAMPLE_RATE,
                      max_duration_sec=None):
    """从 mp4 bytes 提取音轨 → 重采样单声道 → 16-bit PCM WAV。
    返回 (b64_wav_str, actual_duration_sec)。无音轨/失败 → (None, 0.0)。
    """
    from torchcodec.decoders import AudioDecoder
    try:
        dec = AudioDecoder(video_bytes, sample_rate=sample_rate, num_channels=1)
        samples = dec.get_all_samples()
    except Exception:
        return None, 0.0

    waveform = samples.data
    sr = int(samples.sample_rate)

    if waveform is None or waveform.numel() == 0:
        return None, 0.0

    if waveform.dim() == 2:
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)
    if max_duration_sec and max_duration_sec > 0:
        max_samples = int(max_duration_sec * sr)
        if waveform.shape[0] > max_samples:
            waveform = waveform[:max_samples]

    arr = waveform.numpy()
    arr = np.clip(arr, -1.0, 1.0)
    pcm16 = (arr * 32767.0).astype(np.int16)

    import wave
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
    wav_bytes = buf.getvalue()
    duration = float(pcm16.shape[0]) / sr if sr else 0.0
    b64 = base64.b64encode(wav_bytes).decode('ascii')
    return b64, duration


def find_video_keys_in_convs(convs):
    """从 conversations 文本中提取所有 <video_K>, 返回有序 K 列表"""
    seen = set()
    out = []
    text_all = ' '.join(c.get('value', '') for c in convs)
    for m in re.finditer(r'<video_(\d+)>', text_all):
        k = int(m.group(1))
        if k not in seen:
            seen.add(k)
            out.append(k)
    return sorted(out)


class _PrefilterAbort(Exception):
    """oversized 预过滤内部哨兵: 任何不确定情形抛出 → 静默退回原始完整解码路径。"""


def _make_video_decoder(video_bytes):
    """构造 VideoDecoder, 与 extract_frames 内部完全一致。"""
    try:
        return VideoDecoder(video_bytes, dimension_order='NHWC',
                            num_ffmpeg_threads=1, seek_mode='approximate')
    except TypeError:
        return VideoDecoder(video_bytes, dimension_order='NHWC',
                            num_ffmpeg_threads=1)


def _predict_video_segment(md, max_fps,
                           min_pixels=DEFAULT_VIDEO_MIN_PIXELS,
                           max_pixels=DEFAULT_VIDEO_MAX_PIXELS,
                           total_pixels=DEFAULT_VIDEO_TOTAL_PIXELS,
                           min_frames=FPS_MIN_FRAMES,
                           max_frames=FPS_MAX_FRAMES,
                           frame_factor=FRAME_FACTOR):
    """只用 metadata 预测一个 video 段的 (indices, target_h, target_w, vt_pred), 不解码任何帧。"""
    duration = float(md.duration_seconds or 0)
    orig_h = int(md.height)
    orig_w = int(md.width)
    total_frames = int(getattr(md, 'num_frames', 0) or 0)
    if duration <= 0 or orig_h <= 0 or orig_w <= 0:
        raise _PrefilterAbort(f'bad metadata: dur={duration} h={orig_h} w={orig_w}')
    if total_frames <= 0:
        avg_fps = float(getattr(md, 'average_fps', 0) or 0) or 25.0
        total_frames = max(1, int(duration * avg_fps))

    target_n = pick_nframes(
        duration, total_frames, max_fps,
        min_frames=min_frames, max_frames=max_frames, frame_factor=frame_factor,
    )
    if target_n == 1:
        indices = [0]
    else:
        step = (total_frames - 1) / (target_n - 1)
        indices = [int(round(i * step)) for i in range(target_n)]
    indices = sorted({max(0, min(idx, total_frames - 1)) for idx in indices})
    if not indices:
        indices = [0]

    eff_max_pixels = pick_max_pixels(
        len(indices), min_pixels, max_pixels, total_pixels, frame_factor=frame_factor,
    )
    target_h, target_w = compute_target_size(orig_h, orig_w, min_pixels, eff_max_pixels)
    vt_pred = video_segment_tokens(len(indices), target_h, target_w)
    return indices, target_h, target_w, vt_pred


def process_sample(sample, max_fps, process_audio=False,
                   min_pixels=DEFAULT_VIDEO_MIN_PIXELS,
                   max_pixels=DEFAULT_VIDEO_MAX_PIXELS,
                   total_pixels=DEFAULT_VIDEO_TOTAL_PIXELS,
                   jpeg_quality=85,
                   audio_sample_rate=DEFAULT_AUDIO_SAMPLE_RATE,
                   audio_max_duration_sec=None,
                   audio_samples_per_token=DEFAULT_AUDIO_SAMPLES_PER_TOKEN,
                   audio_chunk_sec=DEFAULT_AUDIO_CHUNK_SEC,
                   audio_linear_tokens_per_sec=0.0,
                   max_tok=None,
                   pack_algo='v1',          # 'v1'(默认,旧抽帧) | 'v2'(联合降级)
                   seq_length=None,         # v2 必需: 画面预算反解用
                   reserve_interleave=False,  # 为音频交错预留 worst-case token (独立开关)
                   audio_token_ratio_cap=0.4,  # audio token占比上界: audio_tok/seq>cap 则 drop (0=关闭, 仅v2, 默认0.4)
                   video_max_duration_sec=None):  # 视频时长上界(秒): 任一视频超此长度整条 drop (None/<=0=不过滤, 默认不过滤)
    """单个原始样本 → (nested_meta, conversations, vision_tokens, text_tokens)
    nested_meta 形如 {"video_0": {"image_0": b64, ..., (optional) "audio_0": b64_wav, "audio_duration": s}}
    """
    raw_meta = sample.get('meta')
    if raw_meta is None:
        raw_meta = {}
    elif hasattr(raw_meta, 'tolist'):  # numpy
        raw_meta = raw_meta.tolist()
    if not isinstance(raw_meta, dict):
        raw_meta = dict(raw_meta) if raw_meta else {}

    convs = sample.get('conversations')
    if convs is None:
        convs = []
    elif hasattr(convs, 'tolist'):  # numpy.ndarray
        convs = convs.tolist()
    convs = list(convs) if convs is not None else []
    if len(convs) == 0:
        raise SkipSample('empty_conversations')

    video_keys = find_video_keys_in_convs(convs)
    if not video_keys:
        raise SkipSample('no_video_marker_in_conversations')

    # === 视频时长过滤 (可选; 默认关闭) ===
    # probe 容器头读 duration (不解码帧), 任一视频 > video_max_duration_sec 则整条 drop。
    # 默认 None → 跳过。
    if video_max_duration_sec and video_max_duration_sec > 0:
        for vk in video_keys:
            vb64 = raw_meta.get(f'video_{vk}')
            if not vb64:
                continue
            try:
                _dur = float(_make_video_decoder(base64.b64decode(vb64)).metadata.duration_seconds or 0)
            except Exception:
                continue
            if _dur > video_max_duration_sec:
                raise SkipSample(f'video_too_long:video_{vk}:dur={_dur:.1f}>{video_max_duration_sec:.1f}')

    # === oversized metadata 预过滤 (bit-exact 无损加速; 仅 v1 适用, v2 跳过) ===
    if max_tok is not None and pack_algo != 'v2':
        try:
            probe = []  # [(decoder, last_idx)]
            vt_pred_total = 0
            for vk in video_keys:
                vb64 = raw_meta.get(f'video_{vk}')
                if not vb64:
                    raise _PrefilterAbort('no_video_data')
                vbytes = base64.b64decode(vb64)
                dec = _make_video_decoder(vbytes)
                indices, _h, _w, vt_pred = _predict_video_segment(
                    dec.metadata, max_fps,
                    min_pixels=min_pixels, max_pixels=max_pixels, total_pixels=total_pixels,
                )
                vt_pred_total += vt_pred
                probe.append((dec, indices[-1]))
            if vt_pred_total > max_tok:
                for dec, last_idx in probe:
                    dec.get_frames_at(indices=[last_idx])  # 失败抛 → 下方 except 退回
                raise SkipSample('oversized_prefilter')
        except SkipSample:
            raise
        except _PrefilterAbort:
            pass
        except Exception:
            pass

    new_meta = {}
    video_dims = {}  # vk -> (n_frames, h, w), 用于精确 token 计算
    audio_durations = {}  # vk -> duration_sec; 只有抽到非空音频时才有 entry

    if pack_algo == 'v2':
        if seq_length is None or seq_length <= 0:
            raise SkipSample('v2_requires_seq_length')

    # ---- v2 两遍式 (多视频预算正确) ----
    _v2_vbytes = {}        # vk -> 解码后的 video bytes (复用, 不重复 b64decode)
    _v2_audio_tok = {}     # vk -> 该 video 音频 token (含 reserve worst-case, 若开)
    _v2_vid_dur = {}       # vk -> 视频时长(秒), 用于按时长分配画面预算
    _v2_budget = {}        # vk -> 该 video 分到的画面 token 预算
    if pack_algo == 'v2':
        for vk in video_keys:
            vb64 = raw_meta.get(f'video_{vk}')
            if not vb64:
                raise SkipSample(f'no_video_data:video_{vk}')
            try:
                vbytes = base64.b64decode(vb64)
            except Exception as e:
                raise SkipSample(f'b64_decode_fail:{e}')
            _v2_vbytes[vk] = vbytes
            _v2_audio_tok[vk] = 0
            try:
                _dec = VideoDecoder(vbytes, dimension_order='NHWC', num_ffmpeg_threads=1)
                _dur = float(_dec.metadata.duration_seconds or 0)
            except Exception as e:
                raise SkipSample(f'v2_bad_video_meta:video_{vk}:{type(e).__name__}:{e}')
            if _dur <= 0:
                raise SkipSample(f'v2_bad_video_duration:video_{vk}:dur={_dur}')
            _v2_vid_dur[vk] = _dur
            if process_audio:
                wav_b64, audio_dur = extract_audio_wav(
                    vbytes, sample_rate=audio_sample_rate,
                    max_duration_sec=audio_max_duration_sec,
                )
                if wav_b64 and audio_dur > 0:
                    audio_durations[vk] = audio_dur
                    a_tok = audio_block_tokens(
                        audio_dur, samples_per_token=audio_samples_per_token,
                        chunk_sec=audio_chunk_sec, sr=audio_sample_rate,
                        linear_tokens_per_sec=audio_linear_tokens_per_sec,
                    )
                    if reserve_interleave:
                        _n_ub = max(1, int(_dur * V2_FPS_CAP))
                        _k_max = min(_n_ub, max(1, int(audio_dur // INTERLEAVE_SEG_MIN_SEC)))
                        a_tok += 3 * _k_max
                    _v2_audio_tok[vk] = a_tok
                    _v2_vbytes[vk] = (vbytes, wav_b64)
        _OVERHEAD_MARGIN = 64
        _tok = get_tokenizer()
        _v2_overhead = (system_block_tokens(tok=_tok)
                        + conversation_tokens(convs, {}, tok=_tok, audio_meta={})
                        + _OVERHEAD_MARGIN)
        _v2_audio_total = sum(_v2_audio_tok.values())
        if audio_token_ratio_cap and audio_token_ratio_cap > 0 and audio_durations:
            _pure_audio = sum(
                audio_block_tokens(
                    d, samples_per_token=audio_samples_per_token,
                    chunk_sec=audio_chunk_sec, sr=audio_sample_rate,
                    linear_tokens_per_sec=audio_linear_tokens_per_sec,
                )
                for d in audio_durations.values()
            )
            if _pure_audio > audio_token_ratio_cap * seq_length:
                raise SkipSample(
                    f'audio_token_ratio_exceed:{_pure_audio}/{seq_length}'
                    f'={_pure_audio/seq_length:.3f}>{audio_token_ratio_cap}')
        _v2_visual_total = max(V2_PF_FLOOR + V2_OVH,
                               seq_length - _v2_overhead - _v2_audio_total)
        _durs = [_v2_vid_dur[vk] for vk in video_keys]
        _budgets = v2_split_visual_budget(_durs, _v2_visual_total)
        _v2_budget = {vk: b for vk, b in zip(video_keys, _budgets)}

    for vk in video_keys:
        if pack_algo == 'v2':
            stored = _v2_vbytes[vk]
            vbytes = stored[0] if isinstance(stored, tuple) else stored
            wav_b64 = stored[1] if isinstance(stored, tuple) else None
        else:
            vb64 = raw_meta.get(f'video_{vk}')
            if not vb64:
                raise SkipSample(f'no_video_data:video_{vk}')
            try:
                vbytes = base64.b64decode(vb64)
            except Exception as e:
                raise SkipSample(f'b64_decode_fail:{e}')
            wav_b64 = None
        try:
            if pack_algo == 'v2':
                frames, duration, h, w, actual_fps = extract_frames_v2(
                    vbytes, seq_length, _v2_budget[vk],
                    jpeg_quality=jpeg_quality,
                )
            else:
                frames, duration, h, w, actual_fps = extract_frames(
                    vbytes, max_fps,
                    min_pixels=min_pixels, max_pixels=max_pixels,
                    total_pixels=total_pixels,
                    jpeg_quality=jpeg_quality
                )
        except Exception as e:
            raise SkipSample(f'mp4_decode_fail:{type(e).__name__}:{e}')

        vd = {'fps': actual_fps}    # 该 video 的真实采样 fps
        for idx, (ts, b64) in enumerate(frames):
            vd[f'image_{idx}'] = b64
        if pack_algo == 'v2':
            if wav_b64 is not None and vk in audio_durations:
                vd['audio_0'] = wav_b64
                vd['audio_duration'] = audio_durations[vk]
                vd['audio_sample_rate'] = audio_sample_rate
        elif process_audio:
            wav_b64, audio_dur = extract_audio_wav(
                vbytes,
                sample_rate=audio_sample_rate,
                max_duration_sec=audio_max_duration_sec,
            )
            if wav_b64 and audio_dur > 0:
                vd['audio_0'] = wav_b64
                vd['audio_duration'] = audio_dur
                vd['audio_sample_rate'] = audio_sample_rate
                audio_durations[vk] = audio_dur

        new_meta[f'video_{vk}'] = vd
        video_dims[vk] = (len(frames), h, w)

    # 真实 token 计算
    tok = get_tokenizer()
    total_vision = sum(video_segment_tokens(n, h, w) for (n, h, w) in video_dims.values())
    total_audio = sum(
        audio_block_tokens(
            d,
            samples_per_token=audio_samples_per_token,
            chunk_sec=audio_chunk_sec,
            sr=audio_sample_rate,
            linear_tokens_per_sec=audio_linear_tokens_per_sec,
        )
        for d in audio_durations.values()
    )
    total_vision += total_audio  # 把 audio 计入 vision 预算
    if reserve_interleave and audio_durations:
        interleave_reserve = 0
        for vk, d in audio_durations.items():
            n_frames = video_dims[vk][0] if vk in video_dims else 0
            k_max = min(n_frames, max(1, int(d // INTERLEAVE_SEG_MIN_SEC))) if n_frames > 0 else 1
            interleave_reserve += 3 * k_max
        total_vision += interleave_reserve
    total_text = conversation_tokens(
        convs, video_dims, tok=tok,
        audio_meta=audio_durations,
        audio_samples_per_token=audio_samples_per_token,
        audio_chunk_sec=audio_chunk_sec,
        audio_sr=audio_sample_rate,
        audio_linear_tokens_per_sec=audio_linear_tokens_per_sec,
    ) - total_vision

    return new_meta, convs, total_vision, total_text


# ============================================================================
# video 分支 dispatch (从 pack.py 复制, 音频门控 + audiovisual SkipSample + think注入)
# ============================================================================

# user-like 角色名 (与训练 role_map 对齐: human/user → user 侧)
_USER_ROLES = ('user', 'human')


def inject_user_query_suffix(convs, suffix):
    """think pack 策略: 给每一轮 user query 的 value 末尾追加后缀 (如 '<think_query>')。
    必须在算 token **之前**调用。空 suffix = 不动 (非think, 与原 pack 逻辑一致)。
    """
    if not suffix:
        return convs
    for c in convs:
        if isinstance(c, dict) and c.get('from') in _USER_ROLES:
            c['value'] = (c.get('value') or '') + suffix
    return convs


# 非think(no-think) 配对 pattern: 与 chat_template 的 enable_thinking=false 分支对齐。
#   user 尾 -> '<no_think>'; assistant 头 -> '<think>\n\n</think>\n\n'。
# 在算 token **之前**调用, 使 conversation_tokens 把这些 token 计入预算 (画面预算精确自适应)。
NOTHINK_USER_SUFFIX = '<no_think>'
NOTHINK_ASSISTANT_PREFIX = '<think>\n\n</think>\n\n'
_ASSISTANT_ROLES = ('assistant', 'gpt')


def has_think_pattern(convs):
    """检测 conversations 里(注入前的原始 value)是否含 <think>/</think> pattern。
    非think 数据若命中, 说明是"伪装成非think 的 think 数据"(如 question 内嵌 think 指令 /
    minimal 产物带了推理链), 应整条丢弃, 不注入 no-think pattern。
    """
    for c in convs:
        if not isinstance(c, dict):
            continue
        v = c.get('value') or ''
        if '<think>' in v or '</think>' in v:
            return True
    return False


def inject_nothink_pattern(convs):
    """给非think 数据注入 no-think 配对 pattern (幂等: 已注入的跳过)。"""
    for c in convs:
        if not isinstance(c, dict):
            continue
        role = c.get('from')
        v = c.get('value') or ''
        if role in _USER_ROLES:
            if not v.endswith(NOTHINK_USER_SUFFIX):
                c['value'] = v + NOTHINK_USER_SUFFIX
        elif role in _ASSISTANT_ROLES:
            if not v.lstrip().startswith('<think>'):
                c['value'] = NOTHINK_ASSISTANT_PREFIX + v
    return convs


def _normalize_meta(raw_meta):
    """parquet 反序列化 / numpy / dict 统一成 plain dict。"""
    import json
    if raw_meta is None:
        return {}
    if hasattr(raw_meta, 'tolist') and not isinstance(raw_meta, dict):
        raw_meta = raw_meta.tolist()
    if isinstance(raw_meta, str):
        try:
            raw_meta = json.loads(raw_meta)
        except Exception:
            return {}
    if not isinstance(raw_meta, dict):
        try:
            raw_meta = dict(raw_meta)
        except Exception:
            return {}
    return raw_meta


def _normalize_convs(raw_convs):
    """numpy.ndarray → list; 每条保留 {from, value, loss_mask?}。"""
    if raw_convs is None:
        return []
    if hasattr(raw_convs, 'tolist'):
        raw_convs = raw_convs.tolist()
    raw_convs = list(raw_convs)
    out = []
    lm_present = []
    for c in raw_convs:
        if isinstance(c, dict):
            o = {'from': c.get('from', 'user'), 'value': c.get('value', '') or ''}
            lm = c.get('loss_mask')
            if lm is not None:
                o['loss_mask'] = lm
                lm_present.append(True)
            else:
                lm_present.append(False)
            out.append(o)
    if any(lm_present) and not all(lm_present):
        raise ValueError(
            f'partial loss_mask in record (would corrupt parquet struct schema). '
            f'lm_present={lm_present}'
        )
    return out


def _record_wants_audio(sample, cfg):
    """按 audiovisual_type 决定该 record 是否抽取音频。

    三分类语义 (audiovisual_type):
      - 'mute'        -> 恒 False (原始视频无音轨)。
      - 'audiovisual' -> 恒 True  (QA 必须依赖音频)。
      - 'visual_only' -> 由 audio_ratio 比例控制 (per-record 确定性哈希)。
      - 缺字段 / 旧数据 -> 恒 False (当 mute 处理)。
    最终是否真抽还要 AND 全局 cfg['process_audio'] (见 process_sample_video)。
    """
    av_type = str(sample.get('audiovisual_type', '') or '')
    if av_type == 'mute' or av_type == '':
        return False
    if av_type == 'audiovisual':
        return True
    ratio = cfg.get('audio_ratio', 0.0) or 0.0
    if ratio <= 0:
        return False
    if ratio >= 1.0:
        return True
    rid = sample.get('id')
    if rid is None or (isinstance(rid, float) and math.isnan(rid)):
        convs = sample.get('conversations')
        if hasattr(convs, 'tolist'):
            convs = convs.tolist()
        rid = str(convs)[:512]
    key = f"{cfg.get('seed', 42)}|audio|{rid}"
    h = hashlib.sha1(key.encode('utf-8', 'ignore')).hexdigest()
    unit = int(h[:8], 16) / 0xFFFFFFFF
    return unit < ratio


def process_sample_video(sample, cfg):
    """video 分支: 直接转发到 process_sample。
    process_audio 最终生效 = 全局 cfg['process_audio'] AND 该 record 被抽中(record_wants_audio)。
    返回 (new_meta, convs, vt, tt)。
    """
    wants_audio = cfg['process_audio'] and _record_wants_audio(sample, cfg)
    audio_required = bool(cfg['process_audio']
                          and str(sample.get('audiovisual_type', '') or '') == 'audiovisual')
    # think pack 策略: process_sample 内部从 sample['conversations'] 取 convs 并自算 token,
    # 故在转发前就地注入写回 sample (在算 token 前, 使新增 token 计入预算)。
    #   - think: --user_query_suffix '<think_query>' 追加到 user 尾。
    #   - 非think(is_reasoning=false, env DOTS_TRAIN_IS_REASONING=='0'): 注入 no-think 配对 pattern
    #     (user 尾 '<no_think>' + assistant 头 '<think>\n\n</think>\n\n'), 与 chat_template 对齐。
    suffix = cfg.get('user_query_suffix') or ''
    is_nothink = os.environ.get('DOTS_TRAIN_IS_REASONING') == '0'
    if suffix or is_nothink:
        convs = sample.get('conversations')
        if hasattr(convs, 'tolist'):  # numpy.ndarray
            convs = convs.tolist()
        if convs is not None:
            convs = [dict(c) if isinstance(c, dict) else c for c in convs]
            if suffix:
                inject_user_query_suffix(convs, suffix)
            if is_nothink:
                # 非think 数据若已含 <think>/</think>(伪装成非think 的 think 数据) → 整条丢弃
                if has_think_pattern(convs):
                    raise SkipSample('nothink_but_has_think_pattern')
                inject_nothink_pattern(convs)
            sample = dict(sample)
            sample['conversations'] = convs
    new_meta, convs_out, vt, tt = process_sample(
        sample,
        max_fps=cfg['max_fps'],
        process_audio=wants_audio,
        min_pixels=cfg['video_min_pixels'],
        max_pixels=cfg['video_max_pixels'],
        total_pixels=cfg['video_total_pixels'],
        jpeg_quality=cfg['video_jpeg_quality'],
        audio_sample_rate=cfg.get('audio_sample_rate', DEFAULT_AUDIO_SAMPLE_RATE),
        audio_max_duration_sec=cfg.get('audio_max_duration_sec'),
        audio_samples_per_token=cfg.get('audio_samples_per_token', DEFAULT_AUDIO_SAMPLES_PER_TOKEN),
        audio_chunk_sec=cfg.get('audio_chunk_sec', DEFAULT_AUDIO_CHUNK_SEC),
        audio_linear_tokens_per_sec=cfg.get('audio_linear_tokens_per_sec', 0.0),
        max_tok=cfg.get('max_tok'),
        pack_algo=cfg.get('pack_algo', 'v1'),
        seq_length=cfg.get('seq_length'),
        reserve_interleave=cfg.get('reserve_interleave', False),
        audio_token_ratio_cap=cfg.get('audio_token_ratio_cap', 0.4),
        video_max_duration_sec=cfg.get('video_max_duration_sec'),
    )
    if audio_required:
        vids = [k for k in new_meta if isinstance(k, str) and k.startswith('video_')
                and isinstance(new_meta[k], dict)]
        for vk in vids:
            if 'audio_0' not in new_meta[vk]:
                raise SkipSample(f'audiovisual_no_audio:{vk}')
    return new_meta, convs_out, vt, tt
