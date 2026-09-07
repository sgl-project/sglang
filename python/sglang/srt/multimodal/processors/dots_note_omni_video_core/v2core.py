"""Dependency-free token-budget algorithms for dots v2 video packing."""

import math

PATCH_SIZE = 14
MERGE_SIZE = 2
ALIGN = PATCH_SIZE * MERGE_SIZE

FPS_MIN_FRAMES = 4
TIMESTAMP_TOKENS = 13
IMG_WRAP_TOKENS = 2

V2_FPS_CAP = 1.0
V2_FPS_MIN = 0.2
V2_PF_FLOOR = 128
V2_PF_CEIL = 1024
V2_OVH = TIMESTAMP_TOKENS + IMG_WRAP_TOKENS

INTERLEAVE_SEG_MIN_SEC = 1.0


def compute_target_size(orig_h, orig_w, min_pixels, max_pixels):
    """Resize proportionally to aligned dimensions within the pixel budget."""
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
    """Return a power-of-two frame cap derived from the sequence budget."""
    need = max(1, (seq_length - 2240) // (pf_floor + ovh))
    if need <= 1024:
        return 1024
    p = 1
    while p < need:
        p <<= 1
    return p


def real_patches_at(orig_h, orig_w, pf_cap, pf_floor=V2_PF_FLOOR):
    """Return the actual patch count after aligned resizing."""
    eff_max_px = pf_cap * ALIGN * ALIGN
    th, tw = compute_target_size(orig_h, orig_w, pf_floor * ALIGN * ALIGN, eff_max_px)
    return (th // ALIGN) * (tw // ALIGN)


def v2_solve_degrade(
    visual_budget,
    duration,
    orig_max_pf,
    orig_fps,
    seq_length,
    fps_cap=V2_FPS_CAP,
    fps_min=V2_FPS_MIN,
    pf_floor=V2_PF_FLOOR,
    pf_ceil=V2_PF_CEIL,
    ovh=V2_OVH,
    orig_h=None,
    orig_w=None,
):
    """Jointly reduce frame rate and resolution to fit the visual budget."""
    fps_cap_eff = min(fps_cap, max(orig_fps, 1e-6))
    pf_ceil_eff = min(pf_ceil, max(orig_max_pf, pf_floor))
    hardcap = v2_frame_hardcap(seq_length, pf_floor, ovh)
    _use_real = orig_h is not None and orig_w is not None

    def _patch_of(pf):
        if _use_real:
            return real_patches_at(orig_h, orig_w, max(pf_floor, round(pf)), pf_floor)
        return round(pf)

    def usage(r):
        fps = fps_min + r * (fps_cap_eff - fps_min)
        pf = pf_floor + r * (pf_ceil_eff - pf_floor)
        nf = max(FPS_MIN_FRAMES, min(round(duration * fps), hardcap))
        return nf * (_patch_of(pf) + ovh), fps, pf, nf

    if usage(1.0)[0] <= visual_budget:
        _, fps, pf, nf = usage(1.0)
        return nf, fps, round(pf)
    if usage(0.0)[0] > visual_budget:
        _floor_cost = _patch_of(pf_floor) + ovh
        nf = max(FPS_MIN_FRAMES, min(visual_budget // _floor_cost, hardcap))
        fps = nf / max(duration, 1e-6)
        return nf, round(fps, 4), pf_floor
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if usage(mid)[0] <= visual_budget:
            lo = mid
        else:
            hi = mid
    _, fps, pf, nf = usage(lo)
    return nf, fps, round(pf)


def v2_split_visual_budget(durations, visual_total, pf_floor=V2_PF_FLOOR, ovh=V2_OVH):
    """Allocate a shared visual budget by duration with a per-video floor."""
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
