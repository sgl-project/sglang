# ruff: noqa
"""pack_video_v2core.py — v2 联合降级的纯算法核心 (无 torchcodec/cv2 等重依赖)。

仅依赖标准库 math。供两方 import:
  - pack_video.py (训练侧 pack): from pack_video_v2core import *  (保持原符号不变)
  - dots3_eval gpt.py (评测侧): import pack_video_v2core  (评测机 torchcodec 加载失败也能用)

抽离原因: pack_video.py 顶层 `from torchcodec.decoders import VideoDecoder` 在评测机
(CUDA 库缺失) import 即崩; 而评测交错只需这里的纯算法(v2_solve_degrade/compute_target_size),
故把无重依赖的部分独立成本文件, 评测侧 import 它而非整个 pack_video。

⚠️ 本文件是 pack_video.py 中对应函数/常量的 single-source-of-truth:
   pack_video.py 通过 `from pack_video_v2core import (...)` 复用, 不再各留一份, 防漂移。
"""
import math

# ---- 视觉 patch 常量 (与 dots ViT patch14×merge2 = 28 对齐) ----
PATCH_SIZE = 14
MERGE_SIZE = 2
ALIGN = PATCH_SIZE * MERGE_SIZE  # 28

# ---- 帧/token 常量 ----
FPS_MIN_FRAMES = 4         # 每个 video 至少这么多帧
TIMESTAMP_TOKENS = 13      # hms 时间戳实测 token 数 (saved_tokenizer 实测; 原 11 是低估)
IMG_WRAP_TOKENS = 2        # <|img|> + <|endofimg|>

# ---- v2 联合降级常量 (定稿) ----
V2_FPS_CAP = 1.0
V2_FPS_MIN = 0.2
V2_PF_FLOOR = 128          # = video_min_pixels/784 ≈ 240p, 画质底线
V2_PF_CEIL = 1024          # = video_max_pixels/784, 预算全给分辨率不额外限制
V2_OVH = TIMESTAMP_TOKENS + IMG_WRAP_TOKENS   # 每帧固定开销 = 13 + 2 = 15

# ---- 音视频交错 (reserve_interleave) 常量 ----
# 训练侧音频按帧时间戳边界切 K 段, 每段最小时长护栏 (= flattener 的 ai_seg_min_sec, 防 whisper 短段碎片)。
# pack 侧据此算 worst-case K_max = min(N_frames, audio_dur // INTERLEAVE_SEG_MIN_SEC) 来预留交错预算。
INTERLEAVE_SEG_MIN_SEC = 1.0


def round_by_factor(n, factor):
    return round(n / factor) * factor


def floor_by_factor(n, factor):
    return math.floor(n / factor) * factor


def ceil_by_factor(n, factor):
    return math.ceil(n / factor) * factor


def compute_target_size(orig_h, orig_w, min_pixels, max_pixels):
    """对齐到 28 倍数; 约束在 [min_pixels, max_pixels]; 等比缩放保持长宽比。

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


def v2_frame_hardcap(seq_length, pf_floor=V2_PF_FLOOR, ovh=V2_OVH):
    """FRAME_HARDCAP 随 seq 自适应: ≈ 画面预算/每帧最低token, 取≥的最小2的幂, 下限1024。"""
    need = max(1, (seq_length - 2240) // (pf_floor + ovh))
    if need <= 1024:
        return 1024
    p = 1
    while p < need:
        p <<= 1
    return p


def real_patches_at(orig_h, orig_w, pf_cap, pf_floor=V2_PF_FLOOR):
    """给定 pf 上限, 算 compute_target_size 实际产出的真实 patch 数。

    ★ 关键: compute_target_size 把【小分辨率】视频放大到 min_pixels 时, 对齐 28 向上取整(ceil)后
      两维乘积可能【超过】pf_cap 对应的 patch 数 (如 320×240 → 280×392 = 140 > pf_floor 128)。
      抽象 pf 会低估这类视频的真实 token, 致训练误 drop / 评测 server 超长。此函数返回真实值。
    """
    eff_max_px = pf_cap * ALIGN * ALIGN
    th, tw = compute_target_size(orig_h, orig_w, pf_floor * ALIGN * ALIGN, eff_max_px)
    return (th // ALIGN) * (tw // ALIGN)


def v2_solve_degrade(visual_budget, duration, orig_max_pf, orig_fps,
                     seq_length,
                     fps_cap=V2_FPS_CAP, fps_min=V2_FPS_MIN,
                     pf_floor=V2_PF_FLOOR, pf_ceil=V2_PF_CEIL, ovh=V2_OVH,
                     orig_h=None, orig_w=None):
    """联合降级求解。返回 (nframes, target_fps, target_pf)。

    visual_budget: 画面可用 token 预算 (= seq - margin - sys - text - audio)。
    orig_max_pf:   原始视频单帧最大可用 pf (= 原始分辨率对应的 patch 数, 不放大上限)。
    orig_fps:      原始视频 fps (不超采上限)。
    orig_h/orig_w: 原始分辨率(可选)。给定时 usage 用【真实 patch】(compute_target_size 实际产出)
                   算预算, 修复小分辨率视频(如 320×240 放大对齐后 patch>pf_floor)被低估的 bug。
                   不给(None)则退回旧行为(用抽象 pf), 向后兼容。
    """
    fps_cap_eff = min(fps_cap, max(orig_fps, 1e-6))          # 不超采
    pf_ceil_eff = min(pf_ceil, max(orig_max_pf, pf_floor))   # 不放大超原图
    hardcap = v2_frame_hardcap(seq_length, pf_floor, ovh)
    _use_real = (orig_h is not None and orig_w is not None)

    def _patch_of(pf):
        # 给定目标 pf, 该帧真实 token 用的 patch 数: 有原图尺寸则按 compute_target_size 真实算
        if _use_real:
            return real_patches_at(orig_h, orig_w, max(pf_floor, int(round(pf))), pf_floor)
        return int(round(pf))

    def usage(r):
        fps = fps_min + r * (fps_cap_eff - fps_min)
        pf = pf_floor + r * (pf_ceil_eff - pf_floor)
        nf = max(FPS_MIN_FRAMES, min(int(round(duration * fps)), hardcap))
        return nf * (_patch_of(pf) + ovh), fps, pf, nf

    # r=1 能放下 -> 直接用满
    if usage(1.0)[0] <= visual_budget:
        _, fps, pf, nf = usage(1.0)
        return nf, fps, int(round(pf))
    # r=0 (最低配) 仍超 -> 突破 fps_min 继续降帧数, pf 锁 floor, 产物始终 ≤ 预算
    if usage(0.0)[0] > visual_budget:
        # 用真实 patch(地板配置下)算每帧成本, 防小分辨率低估
        _floor_cost = _patch_of(pf_floor) + ovh
        nf = max(FPS_MIN_FRAMES, min(visual_budget // _floor_cost, hardcap))
        fps = nf / max(duration, 1e-6)
        return nf, round(fps, 4), pf_floor
    # 二分最大可行 r
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if usage(mid)[0] <= visual_budget:
            lo = mid
        else:
            hi = mid
    _, fps, pf, nf = usage(lo)
    return nf, fps, int(round(pf))


# v2 画面预算固定开销 (text/sys/margin 粗留, 与 v2_frame_hardcap 同口径)
V2_OVERHEAD = 2240


def v2_split_visual_budget(durations, visual_total, pf_floor=V2_PF_FLOOR, ovh=V2_OVH):
    """把画面总预算 visual_total 在多个视频间按【时长比例】分配(带保底)。

    durations: list[float], 各视频时长(秒)。单视频时 len==1, 该视频拿全部。
    返回 list[int], 各视频分到的画面 token 预算。

    分配原则:
      - 每个视频保底 = FPS_MIN_FRAMES(4) × (pf_floor + ovh), 防短视频被比例压到抽不出帧;
      - 剩余预算按时长比例分(长视频多给)。
    单视频(len==1): 直接返回 [visual_total](保底逻辑不影响, 全给它)。
    注: 调用方(process_sample)已对"视频时长≤0"做 SkipSample fail-fast, 正常不会传入0时长;
        下方 tot_dur<=0 仅作纯函数防御性兜底(均分), 不作为业务降级路径。
    """
    n = len(durations)
    if n <= 1:
        return [max(pf_floor + ovh, int(visual_total))]
    floor_each = FPS_MIN_FRAMES * (pf_floor + ovh)
    floor_total = floor_each * n
    remain = max(0, int(visual_total) - floor_total)
    tot_dur = sum(max(0.0, d) for d in durations)
    out = []
    if tot_dur <= 0:
        share = remain // n
        for _ in range(n):
            out.append(floor_each + share)
    else:
        for d in durations:
            out.append(floor_each + int(remain * (max(0.0, d) / tot_dur)))
    return out


def v2_plan(seq_length, duration, orig_h, orig_w, orig_fps,
            audio_tok=0, reserve_tok=0, n_videos=1,
            pf_floor=V2_PF_FLOOR, pf_ceil=V2_PF_CEIL, ovh=V2_OVH, overhead=V2_OVERHEAD):
    """统一的"抽帧计划"(pack 和 eval 都调它, 口径完全一致)。

    输入: seq_length(真实序列长度) + 视频元信息 + 该样本音频/reserve 总 token + 视频数。
    输出: (nframes, target_h, target_w, target_pf)。

    画面总预算 = seq - overhead - audio_tok - reserve_tok; 多视频则均分到每个 video。
    单 video 调用传 n_videos=1 即可(eval 评测每条单视频用此)。
    """
    visual_total = max(pf_floor + ovh, seq_length - overhead - audio_tok - reserve_tok)
    per_video_budget = max(pf_floor + ovh, visual_total // max(1, n_videos))
    # 原始视频单帧最大可用 pf (28 对齐, 不放大上限)
    oh28 = max(ALIGN, round(orig_h / ALIGN) * ALIGN)
    ow28 = max(ALIGN, round(orig_w / ALIGN) * ALIGN)
    orig_max_pf = (oh28 // ALIGN) * (ow28 // ALIGN)
    nframes, target_fps, target_pf = v2_solve_degrade(
        per_video_budget, duration, orig_max_pf, orig_fps, seq_length,
        pf_floor=pf_floor, pf_ceil=pf_ceil, ovh=ovh,
        orig_h=orig_h, orig_w=orig_w,   # ★ 传真实尺寸, 让预算按真实 patch 算(修小分辨率低估)
    )
    eff_max_px = min(target_pf * ALIGN * ALIGN, orig_max_pf * ALIGN * ALIGN)
    target_h, target_w = compute_target_size(orig_h, orig_w, pf_floor * ALIGN * ALIGN, eff_max_px)
    return nframes, target_h, target_w, target_pf
