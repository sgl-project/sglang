# =========================
# File 2: LoRAFormatAdapter (e.g. sglang/multimodal_gen/runtime/pipelines_core/lora_format_adapter.py)
# =========================

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Dict, Iterable, Mapping, Optional

import torch
from diffusers.loaders import lora_conversion_utils as lcu

logger = logging.getLogger("LoRAFormatAdapter")


class LoRAFormat(str, Enum):
    """Supported external LoRA formats before normalization."""

    STANDARD = "standard"
    NON_DIFFUSERS_SD = "non-diffusers-sd"
    QWEN_IMAGE_STANDARD = "qwen-image-standard"
    XLABS_FLUX = "xlabs-ai"
    KOHYA_FLUX = "kohya-flux"
    WAN = "wan"


# ---------------------------------------------------------------------------
# Qwen-Image specific: fuse split Q/K/V LoRA into fused to_qkv/to_added_qkv
# ---------------------------------------------------------------------------


def apply_lora_alpha_scaling_inplace(
    sd: Dict[str, torch.Tensor],
    log: logging.Logger,
    *,
    remove_alpha: bool = True,
    atol: float = 1e-8,
) -> int:
    """
    Bake LoRA scaling (alpha / rank) into lora_B.weight in-place.

    This makes the pipeline correct even if runtime ignores `.alpha`.
    Returns: number of modules scaled.
    """
    scaled = 0
    keys = list(sd.keys())

    for kA in keys:
        if not kA.endswith(".lora_A.weight"):
            continue
        prefix = kA[: -len(".lora_A.weight")]
        kB = prefix + ".lora_B.weight"
        kAlpha = prefix + ".alpha"
        if kB not in sd or kAlpha not in sd:
            continue

        A = sd[kA]
        B = sd[kB]
        alpha_t = sd[kAlpha]

        # alpha may be scalar tensor or something similar
        try:
            alpha_val = float(alpha_t.reshape(-1)[0].item())
        except Exception:
            log.warning("[LoRAFormatAdapter] cannot parse alpha for %s", prefix)
            continue

        layout = _infer_lora_layout(A, B)
        if layout is None:
            log.warning("[LoRAFormatAdapter] cannot infer layout for %s", prefix)
            continue

        if layout == "BA":
            # A: (r, in), B: (out, r)
            r = int(A.shape[0])
        else:
            # "AB": A: (in, r), B: (r, out)
            r = int(B.shape[0])

        if r <= 0:
            continue
        scale = alpha_val / float(r)

        # If alpha is 0 or extremely small, skip (or keep)
        if abs(scale) <= atol:
            # still can zero out if desired; keep conservative
            continue

        # keep dtype/device consistent (avoid fp32 upcast surprises)
        scale_t = torch.tensor(scale, device=B.device, dtype=B.dtype)
        sd[kB] = B * scale_t

        if remove_alpha:
            sd.pop(kAlpha, None)

        scaled += 1

    if scaled > 0:
        log.info(
            "[LoRAFormatAdapter] baked alpha/rank into lora_B for %d modules", scaled
        )
    return scaled


def _infer_lora_layout(A: torch.Tensor, B: torch.Tensor) -> str | None:
    """
    Infer LoRA matmul layout from A/B shapes.

    Return:
      - "BA" if ΔW = B @ A is valid with (B: out×r, A: r×in)
      - "AB" if ΔW = A @ B is valid with (A: in×r, B: r×out)
      - None if neither works / unknown
    """
    if A.ndim != 2 or B.ndim != 2:
        return None

    ba_ok = B.shape[1] == A.shape[0]  # B(out,r) @ A(r,in)
    ab_ok = A.shape[1] == B.shape[0]  # A(in,r) @ B(r,out)

    if ba_ok and not ab_ok:
        return "BA"
    if ab_ok and not ba_ok:
        return "AB"
    if ba_ok and ab_ok:
        # Ambiguous (rare but possible if square-ish). Prefer the one that treats
        # the smaller dimension as rank.
        r_ba = int(A.shape[0])
        r_ab = int(A.shape[1])
        return "BA" if r_ba <= r_ab else "AB"
    return None


def _block_diag(mats: list[torch.Tensor]) -> torch.Tensor:
    """Dense block-diagonal concatenation for 2D tensors."""
    if not mats:
        raise ValueError("mats must be non-empty")
    if any(m.ndim != 2 for m in mats):
        raise ValueError("all mats must be 2D tensors")

    dtype = mats[0].dtype
    device = mats[0].device
    rows = sum(int(m.shape[0]) for m in mats)
    cols = sum(int(m.shape[1]) for m in mats)

    out = torch.zeros((rows, cols), dtype=dtype, device=device)
    r0 = 0
    c0 = 0
    for m in mats:
        r, c = int(m.shape[0]), int(m.shape[1])
        out[r0 : r0 + r, c0 : c0 + c] = m
        r0 += r
        c0 += c
    return out


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        return float("inf")
    try:
        return (a - b).abs().max().item()
    except Exception:
        return float("inf")


def _fuse_one_qkv_triplet(
    sd: Dict[str, torch.Tensor],
    prefix: str,
    q: str,
    k: str,
    v: str,
    fused: str,
    log: logging.Logger,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-3,
) -> tuple[bool, bool]:
    """
    Fuse one (q,k,v) triplet under a shared prefix into a fused projection.

    Returns:
      (fused_ok, used_blockdiag)
    """
    Aq_k = f"{prefix}.{q}.lora_A.weight"
    Ak_k = f"{prefix}.{k}.lora_A.weight"
    Av_k = f"{prefix}.{v}.lora_A.weight"
    Bq_k = f"{prefix}.{q}.lora_B.weight"
    Bk_k = f"{prefix}.{k}.lora_B.weight"
    Bv_k = f"{prefix}.{v}.lora_B.weight"

    need = (Aq_k, Ak_k, Av_k, Bq_k, Bk_k, Bv_k)
    if not all(x in sd for x in need):
        return False, False

    Aq, Ak, Av = sd[Aq_k], sd[Ak_k], sd[Av_k]
    Bq, Bk, Bv = sd[Bq_k], sd[Bk_k], sd[Bv_k]

    if any(t.ndim != 2 for t in (Aq, Ak, Av, Bq, Bk, Bv)):
        return False, False

    # Ensure all tensors are on the same device (avoid block_diag device mismatch).
    # Prefer Q's device/dtype.
    dev = Aq.device
    dtype = Aq.dtype
    if Ak.device != dev:
        Ak = Ak.to(dev)
    if Av.device != dev:
        Av = Av.to(dev)
    if Bq.device != dev:
        Bq = Bq.to(dev)
    if Bk.device != dev:
        Bk = Bk.to(dev)
    if Bv.device != dev:
        Bv = Bv.to(dev)

    # Keep consistent dtype where reasonable (avoid accidental fp32 expansion).
    if Ak.dtype != dtype:
        Ak = Ak.to(dtype)
    if Av.dtype != dtype:
        Av = Av.to(dtype)
    if Bq.dtype != dtype:
        Bq = Bq.to(dtype)
    if Bk.dtype != dtype:
        Bk = Bk.to(dtype)
    if Bv.dtype != dtype:
        Bv = Bv.to(dtype)

    layout = _infer_lora_layout(Aq, Bq)
    if layout is None:
        return False, False
    if _infer_lora_layout(Ak, Bk) != layout or _infer_lora_layout(Av, Bv) != layout:
        return False, False

    # Validate in/out feature dims are compatible
    if layout == "BA":
        # A: (r, in), B: (out, r)
        in_dim = int(Aq.shape[1])
        out_dim = int(Bq.shape[0])

        if int(Ak.shape[1]) != in_dim or int(Av.shape[1]) != in_dim:
            return False, False
        if int(Bk.shape[0]) != out_dim or int(Bv.shape[0]) != out_dim:
            return False, False

        shared_a = (
            Aq.shape == Ak.shape
            and Aq.shape == Av.shape
            and torch.allclose(Aq, Ak, atol=atol, rtol=rtol)
            and torch.allclose(Aq, Av, atol=atol, rtol=rtol)
        )

        if shared_a:
            # Exact fusion with shared A: concat B on out-dim (rows)
            A_fused = Aq
            B_fused = torch.cat([Bq, Bk, Bv], dim=0)  # (3*out, r)
            used_blockdiag = False
            log.info(
                "[LoRAFormatAdapter] QKV A shared at %s.%s, using shared-A fusion (layout=%s).",
                prefix,
                fused,
                layout,
            )
        else:
            # Exact fusion with mismatched A: block-diagonal B and stacked A (rank expands)
            A_fused = torch.cat([Aq, Ak, Av], dim=0)  # (r_sum, in)
            B_fused = _block_diag([Bq, Bk, Bv])  # (3*out, r_sum)
            used_blockdiag = True
            diff1 = _max_abs_diff(Aq, Ak)
            diff2 = _max_abs_diff(Aq, Av)
            log.warning(
                "[LoRAFormatAdapter] QKV LoRA A mismatch at %s.%s: max|Aq-Ak|=%s max|Aq-Av|=%s. "
                "Using block-diagonal exact fusion (layout=%s, rank expands).",
                prefix,
                fused,
                diff1,
                diff2,
                layout,
            )

    else:
        # layout == "AB"
        # A: (in, r), B: (r, out)
        in_dim = int(Aq.shape[0])
        out_dim = int(Bq.shape[1])

        if int(Ak.shape[0]) != in_dim or int(Av.shape[0]) != in_dim:
            return False, False
        if int(Bk.shape[1]) != out_dim or int(Bv.shape[1]) != out_dim:
            return False, False

        shared_a = (
            Aq.shape == Ak.shape
            and Aq.shape == Av.shape
            and torch.allclose(Aq, Ak, atol=atol, rtol=rtol)
            and torch.allclose(Aq, Av, atol=atol, rtol=rtol)
        )

        if shared_a:
            # Exact fusion with shared A: concat B on out-dim (cols)
            A_fused = Aq
            B_fused = torch.cat([Bq, Bk, Bv], dim=1)  # (r, 3*out)
            used_blockdiag = False
            log.info(
                "[LoRAFormatAdapter] QKV A shared at %s.%s, using shared-A fusion (layout=%s).",
                prefix,
                fused,
                layout,
            )
        else:
            # Exact fusion with mismatched A: block-diagonal B and concatenated A on rank-dim
            A_fused = torch.cat([Aq, Ak, Av], dim=1)  # (in, r_sum)
            B_fused = _block_diag([Bq, Bk, Bv])  # (r_sum, 3*out)
            used_blockdiag = True
            diff1 = _max_abs_diff(Aq, Ak)
            diff2 = _max_abs_diff(Aq, Av)
            log.warning(
                "[LoRAFormatAdapter] QKV LoRA A mismatch at %s.%s: max|Aq-Ak|=%s max|Aq-Av|=%s. "
                "Using block-diagonal exact fusion (layout=%s, rank expands).",
                prefix,
                fused,
                diff1,
                diff2,
                layout,
            )

    fused_A_k = f"{prefix}.{fused}.lora_A.weight"
    fused_B_k = f"{prefix}.{fused}.lora_B.weight"
    sd[fused_A_k] = A_fused
    sd[fused_B_k] = B_fused

    # Copy alpha from Q-proj if present (pipeline currently ignores alpha, but keep for completeness)
    alpha_q = f"{prefix}.{q}.alpha"
    if alpha_q in sd:
        sd[f"{prefix}.{fused}.alpha"] = sd[alpha_q]

    # Remove old split keys (avoid downstream collisions / confusion)
    for p in (q, k, v):
        sd.pop(f"{prefix}.{p}.lora_A.weight", None)
        sd.pop(f"{prefix}.{p}.lora_B.weight", None)
        sd.pop(f"{prefix}.{p}.alpha", None)

    return True, used_blockdiag


def fuse_qkv_like_keys_inplace(
    sd: Dict[str, torch.Tensor], log: logging.Logger
) -> dict[str, int]:
    """
    In-place fuse split q/k/v (and add_q/k/v) LoRA weights into fused projections.

    This is *exact*:
      - If Aq==Ak==Av (shared A), we keep A and concat B (standard fused-QKV constraint).
      - If A differs, we perform block-diagonal exact fusion (rank expands, typically 3r).

    Returns a small stats dict for logging/diagnostics.
    """
    patterns = [
        ("to_q", "to_k", "to_v", "to_qkv"),
        ("add_q_proj", "add_k_proj", "add_v_proj", "to_added_qkv"),
    ]

    fused = 0
    skipped = 0
    a_mismatch = 0
    for q, k, v, fused_name in patterns:
        # Support both "...attn.*" and "...attention.*"
        pat = re.compile(
            rf"^(?P<prefix>.+\.(?:attn|attention))\.{re.escape(q)}\.lora_A\.weight$"
        )

        prefixes = sorted(
            {
                m.group("prefix")
                for key in sd.keys()
                for m in [pat.match(key)]
                if m is not None
            }
        )

        for prefix in prefixes:
            # If already fused keys exist, skip
            if (
                f"{prefix}.{fused_name}.lora_A.weight" in sd
                or f"{prefix}.{fused_name}.lora_B.weight" in sd
            ):
                continue

            ok, used_blockdiag = _fuse_one_qkv_triplet(
                sd, prefix, q, k, v, fused_name, log
            )
            if ok:
                fused += 1
                if used_blockdiag:
                    a_mismatch += 1
            else:
                skipped += 1

    return {"fused": fused, "skipped": skipped, "A_mismatch": a_mismatch}


def maybe_fuse_qwen_image_qkv_lora_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, torch.Tensor]:
    """
    Fuse split Q/K/V LoRA weights into fused to_qkv / to_added_qkv for Qwen-Image style models.

    Expected split keys (after normalize_lora_state_dict):
      ...attn.to_q.lora_A.weight / ...attn.to_k... / ...attn.to_v...
      ...attn.add_q_proj.lora_A.weight / ...add_k_proj... / ...add_v_proj...

    Output fused keys:
      ...attn.to_qkv.lora_A.weight / ...attn.to_qkv.lora_B.weight
      ...attn.to_added_qkv.lora_A.weight / ...attn.to_added_qkv.lora_B.weight

    Fusion rule (EXACT, preserves Lightning LoRA semantics):
      - If Aq/Ak/Av are equal (shared-A), keep A and concatenate B on output dim.
      - If Aq/Ak/Av differ, perform block-diagonal exact fusion, expanding effective rank
        (typically from r to 3r).

    Notes:
      - We strip an optional "diffusion_model." prefix for robustness.
      - We remove the original split keys after fusing to avoid collisions downstream.
      - We copy alpha from Q-proj to fused proj if present (pipeline currently ignores alpha).
    """
    log = logger or globals()["logger"]
    if not state_dict:
        return dict(state_dict)

    # Canonicalize keys: strip optional "diffusion_model." prefix
    sd: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("diffusion_model."):
            k = k[len("diffusion_model.") :]
        sd[k] = v

    stats = fuse_qkv_like_keys_inplace(sd, log)

    if stats["fused"] > 0:
        sample = [k for k in sd.keys() if (".to_qkv." in k or ".to_added_qkv." in k)][
            :20
        ]
        log.info(
            "[LoRAFormatAdapter] fused qkv/add_qkv LoRA: fused=%d skipped=%d A_mismatch=%d, sample fused keys (<=20): %s",
            stats["fused"],
            stats["skipped"],
            stats["A_mismatch"],
            ", ".join(sample) if sample else "(none)",
        )

    return sd


# ---------------------------------------------------------------------------
# Fuse split FFN w1/w3 LoRA into fused w13 (SwiGLU gate+up projection)
# ---------------------------------------------------------------------------


def _fuse_one_w13_pair(
    sd: Dict[str, torch.Tensor],
    prefix: str,
    w1: str,
    w3: str,
    fused: str,
    log: logging.Logger,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-3,
) -> tuple[bool, bool]:
    """
    Fuse one (w1, w3) pair under a shared FFN prefix into fused projection (w13).

    Returns:
      (fused_ok, used_blockdiag)

    Exact fusion:
      - If A1 == A3: keep A, concat B along output dim.
      - Else: block-diagonal exact fusion (rank expands).
    """
    A1_k = f"{prefix}.{w1}.lora_A.weight"
    A3_k = f"{prefix}.{w3}.lora_A.weight"
    B1_k = f"{prefix}.{w1}.lora_B.weight"
    B3_k = f"{prefix}.{w3}.lora_B.weight"

    need = (A1_k, A3_k, B1_k, B3_k)
    if not all(k in sd for k in need):
        return False, False

    A1, A3 = sd[A1_k], sd[A3_k]
    B1, B3 = sd[B1_k], sd[B3_k]

    if any(t.ndim != 2 for t in (A1, A3, B1, B3)):
        return False, False

    # Align device/dtype (avoid block_diag mismatch / fp32 surprises)
    dev = A1.device
    dtype = A1.dtype
    if A3.device != dev:
        A3 = A3.to(dev)
    if B1.device != dev:
        B1 = B1.to(dev)
    if B3.device != dev:
        B3 = B3.to(dev)

    if A3.dtype != dtype:
        A3 = A3.to(dtype)
    if B1.dtype != dtype:
        B1 = B1.to(dtype)
    if B3.dtype != dtype:
        B3 = B3.to(dtype)

    layout = _infer_lora_layout(A1, B1)
    if layout is None:
        return False, False
    if _infer_lora_layout(A3, B3) != layout:
        return False, False

    # Validate in_dim matches
    if layout == "BA":
        # A: (r, in), B: (out, r)
        in_dim = int(A1.shape[1])
        if int(A3.shape[1]) != in_dim:
            return False, False
        # out dims can differ; w13 out = out1 + out3, so no strict equality required.

        shared_a = A1.shape == A3.shape and torch.allclose(A1, A3, atol=atol, rtol=rtol)

        if shared_a:
            A_fused = A1
            B_fused = torch.cat([B1, B3], dim=0)  # (out1+out3, r)
            used_blockdiag = False
            log.info(
                "[LoRAFormatAdapter] FFN w1/w3 A shared at %s.%s, using shared-A fusion (layout=%s).",
                prefix,
                fused,
                layout,
            )
        else:
            # Exact block-diag fusion: (B1@A1 ; B3@A3)
            A_fused = torch.cat([A1, A3], dim=0)  # (r1+r3, in)
            B_fused = _block_diag([B1, B3])  # (out1+out3, r1+r3)
            used_blockdiag = True
            diff = _max_abs_diff(A1, A3)
            log.warning(
                "[LoRAFormatAdapter] FFN w1/w3 LoRA A mismatch at %s.%s: max|A1-A3|=%s. "
                "Using block-diagonal exact fusion (layout=%s, rank expands).",
                prefix,
                fused,
                diff,
                layout,
            )

    else:
        # layout == "AB"
        # A: (in, r), B: (r, out)
        in_dim = int(A1.shape[0])
        if int(A3.shape[0]) != in_dim:
            return False, False

        shared_a = A1.shape == A3.shape and torch.allclose(A1, A3, atol=atol, rtol=rtol)

        if shared_a:
            A_fused = A1
            B_fused = torch.cat([B1, B3], dim=1)  # (r, out1+out3)
            used_blockdiag = False
            log.info(
                "[LoRAFormatAdapter] FFN w1/w3 A shared at %s.%s, using shared-A fusion (layout=%s).",
                prefix,
                fused,
                layout,
            )
        else:
            A_fused = torch.cat([A1, A3], dim=1)  # (in, r1+r3)
            B_fused = _block_diag([B1, B3])  # (r1+r3, out1+out3)
            used_blockdiag = True
            diff = _max_abs_diff(A1, A3)
            log.warning(
                "[LoRAFormatAdapter] FFN w1/w3 LoRA A mismatch at %s.%s: max|A1-A3|=%s. "
                "Using block-diagonal exact fusion (layout=%s, rank expands).",
                prefix,
                fused,
                diff,
                layout,
            )

    fused_A_k = f"{prefix}.{fused}.lora_A.weight"
    fused_B_k = f"{prefix}.{fused}.lora_B.weight"
    sd[fused_A_k] = A_fused
    sd[fused_B_k] = B_fused

    # Copy alpha if still present (often already baked & removed)
    alpha_1 = f"{prefix}.{w1}.alpha"
    if alpha_1 in sd:
        sd[f"{prefix}.{fused}.alpha"] = sd[alpha_1]

    # Remove old split keys
    for p in (w1, w3):
        sd.pop(f"{prefix}.{p}.lora_A.weight", None)
        sd.pop(f"{prefix}.{p}.lora_B.weight", None)
        sd.pop(f"{prefix}.{p}.alpha", None)

    return True, used_blockdiag


def fuse_w13_like_keys_inplace(
    sd: Dict[str, torch.Tensor], log: logging.Logger
) -> dict[str, int]:
    """
    In-place fuse split FFN w1/w3 LoRA into fused w13.

    Supports prefixes like:
      ...feed_forward.w1 / ...feed_forward.w3
      ...ffn.w1 / ...ffn.w3
      ...mlp.w1 / ...mlp.w3

    Returns stats for logging/diagnostics.
    """
    fused = 0
    skipped = 0
    a_mismatch = 0

    # Match both Z-Image naming ("feed_forward") and possible variants ("ffn"/"mlp")
    pat = re.compile(r"^(?P<prefix>.+\.(?:feed_forward|ffn|mlp))\.w1\.lora_A\.weight$")

    prefixes = sorted(
        {
            m.group("prefix")
            for key in sd.keys()
            for m in [pat.match(key)]
            if m is not None
        }
    )

    for prefix in prefixes:
        # If already fused exists, skip
        if f"{prefix}.w13.lora_A.weight" in sd or f"{prefix}.w13.lora_B.weight" in sd:
            continue

        ok, used_blockdiag = _fuse_one_w13_pair(sd, prefix, "w1", "w3", "w13", log)
        if ok:
            fused += 1
            if used_blockdiag:
                a_mismatch += 1
        else:
            skipped += 1

    return {"fused": fused, "skipped": skipped, "A_mismatch": a_mismatch}


def maybe_fuse_w13_lora_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, torch.Tensor]:
    """
    Fuse split FFN w1/w3 LoRA into fused w13 when target model uses fused w13.

    Input keys (after normalize_lora_state_dict):
      ...feed_forward.w1.lora_A.weight / ...w1.lora_B.weight
      ...feed_forward.w3.lora_A.weight / ...w3.lora_B.weight

    Output:
      ...feed_forward.w13.lora_A.weight / ...w13.lora_B.weight

    Notes:
      - Strips optional "diffusion_model." prefix.
      - Removes original split keys after fusing.
    """
    log = logger or globals()["logger"]
    if not state_dict:
        return dict(state_dict)

    # Canonicalize keys: strip optional "diffusion_model." prefix
    sd: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("diffusion_model."):
            k = k[len("diffusion_model.") :]
        sd[k] = v

    stats = fuse_w13_like_keys_inplace(sd, log)

    if stats["fused"] > 0:
        sample = [k for k in sd.keys() if ".w13." in k][:20]
        log.info(
            "[LoRAFormatAdapter] fused w1/w3->w13 LoRA: fused=%d skipped=%d A_mismatch=%d, sample fused keys (<=20): %s",
            stats["fused"],
            stats["skipped"],
            stats["A_mismatch"],
            ", ".join(sample) if sample else "(none)",
        )

    return sd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_keys(keys: Iterable[str], k: int = 20) -> list[str]:
    out = []
    for i, key in enumerate(keys):
        if i >= k:
            break
        out.append(key)
    return out


def _has_substring_key(keys: Iterable[str], substr: str) -> bool:
    return any(substr in k for k in keys)


def _has_prefix_key(keys: Iterable[str], prefix: str) -> bool:
    return any(k.startswith(prefix) for k in keys)


# ---------------------------------------------------------------------------
# Format-specific heuristics
# ---------------------------------------------------------------------------


def _looks_like_xlabs_flux_key(k: str) -> bool:
    """XLabs FLUX-style keys under double_blocks/single_blocks with lora down/up."""
    if not (k.endswith(".down.weight") or k.endswith(".up.weight")):
        return False

    if not k.startswith(
        (
            "double_blocks.",
            "single_blocks.",
            "diffusion_model.double_blocks",
            "diffusion_model.single_blocks",
        )
    ):
        return False

    return ".processor." in k or ".proj_lora" in k or ".qkv_lora" in k


def _looks_like_kohya_flux(state_dict: Mapping[str, torch.Tensor]) -> bool:
    """Kohya FLUX LoRA (flux_lora.py) under lora_unet_double/single_blocks_ prefixes."""
    if not state_dict:
        return False
    keys = state_dict.keys()
    return any(
        k.startswith("lora_unet_double_blocks_")
        or k.startswith("lora_unet_single_blocks_")
        for k in keys
    )


def _looks_like_non_diffusers_sd(state_dict: Mapping[str, torch.Tensor]) -> bool:
    """Classic non-diffusers SD LoRA (Kohya/A1111/sd-scripts)."""
    if not state_dict:
        return False
    keys = state_dict.keys()
    return all(
        k.startswith(("lora_unet_", "lora_te_", "lora_te1_", "lora_te2_")) for k in keys
    )


def _looks_like_wan_lora(state_dict: Mapping[str, torch.Tensor]) -> bool:
    """Wan2.2 distill LoRAs (Wan-AI / Wan2.2-Distill-Loras style)."""
    if not state_dict:
        return False

    for k in state_dict.keys():
        if not k.startswith("diffusion_model.blocks."):
            continue
        if ".lora_down" not in k and ".lora_up" not in k:
            continue
        if ".cross_attn." in k or ".self_attn." in k or ".ffn." in k or ".norm3." in k:
            return True

    return False


def _looks_like_qwen_image(state_dict: Mapping[str, torch.Tensor]) -> bool:
    keys = list(state_dict.keys())
    if not keys:
        return False
    return _has_prefix_key(keys, "transformer.transformer_blocks.") and (
        _has_substring_key(keys, ".lora.down.weight")
        or _has_substring_key(keys, ".lora.up.weight")
    )


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def detect_lora_format_from_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> LoRAFormat:
    """Classify LoRA format by key patterns only."""
    keys = list(state_dict.keys())
    if not keys:
        return LoRAFormat.STANDARD

    if _has_substring_key(keys, ".lora_A") or _has_substring_key(keys, ".lora_B"):
        return LoRAFormat.STANDARD

    if any(_looks_like_xlabs_flux_key(k) for k in keys):
        return LoRAFormat.XLABS_FLUX
    if _looks_like_kohya_flux(state_dict):
        return LoRAFormat.KOHYA_FLUX

    if _looks_like_wan_lora(state_dict):
        return LoRAFormat.WAN

    if _looks_like_qwen_image(state_dict):
        return LoRAFormat.STANDARD

    if _looks_like_non_diffusers_sd(state_dict):
        return LoRAFormat.NON_DIFFUSERS_SD

    if _has_substring_key(keys, ".lora.down") or _has_substring_key(keys, ".lora_up"):
        return LoRAFormat.NON_DIFFUSERS_SD

    return LoRAFormat.STANDARD


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------


def _convert_qwen_image_standard(
    state_dict: Mapping[str, torch.Tensor],
    log: logging.Logger,
) -> Dict[str, torch.Tensor]:
    """Qwen-Image: transformer.*.lora.down/up -> transformer_blocks.*.lora_A/B."""
    out: Dict[str, torch.Tensor] = {}

    for name, tensor in state_dict.items():
        new_name = name

        if new_name.startswith("transformer."):
            new_name = new_name[len("transformer.") :]

        if new_name.endswith(".lora.down.weight"):
            new_name = new_name.replace(".lora.down.weight", ".lora_A.weight")
        elif new_name.endswith(".lora.up.weight"):
            new_name = new_name.replace(".lora.up.weight", ".lora_B.weight")

        out[new_name] = tensor

    _ = _sample_keys(out.keys(), 20)
    return out


def _convert_non_diffusers_sd_simple(
    state_dict: Mapping[str, torch.Tensor],
    log: logging.Logger,
) -> Dict[str, torch.Tensor]:
    """Generic down/up -> A/B conversion for non-diffusers SD-like formats."""
    out: Dict[str, torch.Tensor] = {}

    for name, tensor in state_dict.items():
        new_name = name

        if "lora_down.weight" in new_name:
            new_name = new_name.replace("lora_down.weight", "lora_A.weight")
        elif "lora_up.weight" in new_name:
            new_name = new_name.replace("lora_up.weight", "lora_B.weight")
        elif new_name.endswith(".lora_down"):
            new_name = new_name.replace(".lora_down", ".lora_A")
        elif new_name.endswith(".lora_up"):
            new_name = new_name.replace(".lora_up", ".lora_B")

        out[new_name] = tensor

    sample = _sample_keys(out.keys(), 20)
    log.info(
        "[LoRAFormatAdapter] after NON_DIFFUSERS_SD simple conversion, "
        "sample keys (<=20): %s",
        ", ".join(sample),
    )
    return out


def _convert_with_diffusers_utils_if_available(
    state_dict: Mapping[str, torch.Tensor],
    log: logging.Logger,
) -> Optional[Dict[str, torch.Tensor]]:
    """Use diffusers.lora_conversion_utils if available."""
    try:
        if hasattr(lcu, "maybe_convert_state_dict"):
            converted = lcu.maybe_convert_state_dict(  # type: ignore[attr-defined]
                state_dict
            )
        else:
            converted = dict(state_dict)

        if not isinstance(converted, dict):
            converted = dict(converted)

        sample = _sample_keys(converted.keys(), 20)
        log.info(
            "[LoRAFormatAdapter] diffusers.lora_conversion_utils converted keys, "
            "sample keys (<=20): %s",
            ", ".join(sample),
        )
        return converted
    except Exception as exc:  # pragma: no cover
        log.warning(
            "[LoRAFormatAdapter] diffusers lora_conversion_utils failed, "
            "falling back to internal converters. Error: %s",
            exc,
        )
        return None


def _convert_via_diffusers_candidates(
    state_dict: Mapping[str, torch.Tensor],
    candidate_names: tuple[str, ...],
    log: logging.Logger,
    unavailable_warning: str,
    no_converter_warning: str,
    success_info: str,
    all_failed_warning: str,
) -> Dict[str, torch.Tensor]:
    """Try multiple named converters in lora_conversion_utils, use the first that works."""
    converters = [
        (n, getattr(lcu, n)) for n in candidate_names if callable(getattr(lcu, n, None))
    ]
    if not converters:
        log.warning(no_converter_warning)
        return dict(state_dict)

    last_err: Optional[Exception] = None

    for name, fn in converters:
        try:
            sd_copy = dict(state_dict)
            out = fn(sd_copy)
            if isinstance(out, tuple) and isinstance(out[0], dict):
                out = out[0]
            if not isinstance(out, dict):
                raise TypeError(f"Converter {name} returned {type(out)}")
            log.info(success_info.format(name=name))
            return out
        except Exception as exc:
            last_err = exc

    log.warning(all_failed_warning.format(last_err=last_err))
    return dict(state_dict)


def _convert_xlabs_ai_via_diffusers(
    state_dict: Mapping[str, torch.Tensor],
    log: logging.Logger,
) -> Dict[str, torch.Tensor]:
    """Convert XLabs FLUX LoRA via diffusers helpers."""
    return _convert_via_diffusers_candidates(
        state_dict,
        (
            "_convert_xlabs_flux_lora_to_diffusers",
            "convert_xlabs_lora_state_dict_to_diffusers",
            "convert_xlabs_lora_to_diffusers",
            "convert_xlabs_flux_lora_to_diffusers",
        ),
        log=log,
        unavailable_warning=(
            "[LoRAFormatAdapter] XLabs FLUX detected but diffusers is unavailable."
        ),
        no_converter_warning=(
            "[LoRAFormatAdapter] No XLabs FLUX converter found in diffusers."
        ),
        success_info="[LoRAFormatAdapter] Converted XLabs FLUX LoRA using {name}",
        all_failed_warning=(
            "[LoRAFormatAdapter] All XLabs FLUX converters failed; "
            "last error: {last_err}"
        ),
    )


def _convert_kohya_flux_via_diffusers(
    state_dict: Mapping[str, torch.Tensor],
    log: logging.Logger,
) -> Dict[str, torch.Tensor]:
    """Convert Kohya FLUX LoRA via diffusers helpers."""
    return _convert_via_diffusers_candidates(
        state_dict,
        (
            "_convert_kohya_flux_lora_to_diffusers",
            "convert_kohya_flux_lora_to_diffusers",
        ),
        log=log,
        unavailable_warning=(
            "[LoRAFormatAdapter] Kohya FLUX detected but diffusers is unavailable."
        ),
        no_converter_warning="[LoRAFormatAdapter] No Kohya FLUX converter found.",
        success_info="[LoRAFormatAdapter] Converted Kohya FLUX LoRA using {name}",
        all_failed_warning=(
            "[LoRAFormatAdapter] Kohya FLUX conversion failed; "
            "last error: {last_err}"
        ),
    )


# ---------------------------------------------------------------------------
# Conversion dispatcher
# ---------------------------------------------------------------------------


def convert_lora_state_dict_by_format(
    state_dict: Mapping[str, torch.Tensor],
    fmt: LoRAFormat,
    log: logging.Logger,
) -> Dict[str, torch.Tensor]:
    """Normalize a raw LoRA state_dict into A/B + .weight naming."""
    if fmt == LoRAFormat.QWEN_IMAGE_STANDARD:
        return _convert_qwen_image_standard(state_dict, log)

    if fmt == LoRAFormat.XLABS_FLUX:
        converted = _convert_xlabs_ai_via_diffusers(state_dict, log)
        return _convert_non_diffusers_sd_simple(converted, log)

    if fmt == LoRAFormat.KOHYA_FLUX:
        converted = _convert_kohya_flux_via_diffusers(state_dict, log)
        return _convert_non_diffusers_sd_simple(converted, log)

    if fmt == LoRAFormat.WAN:
        maybe = _convert_with_diffusers_utils_if_available(state_dict, log)
        if maybe is None:
            maybe = dict(state_dict)
        return _convert_non_diffusers_sd_simple(maybe, log)

    if fmt == LoRAFormat.STANDARD:
        maybe = _convert_with_diffusers_utils_if_available(state_dict, log)
        if maybe is None:
            maybe = dict(state_dict)

        if _looks_like_qwen_image(maybe):
            return _convert_qwen_image_standard(maybe, log)

        return maybe

    if fmt == LoRAFormat.NON_DIFFUSERS_SD:
        maybe = _convert_with_diffusers_utils_if_available(state_dict, log)
        if maybe is None:
            maybe = dict(state_dict)
        return _convert_non_diffusers_sd_simple(maybe, log)

    log.info(
        "[LoRAFormatAdapter] format %s not handled specially, returning as-is",
        fmt,
    )
    return dict(state_dict)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def normalize_lora_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, torch.Tensor]:
    """Normalize any supported LoRA format into a single canonical layout."""
    log = logger or globals()["logger"]

    keys = list(state_dict.keys())
    log.info(
        "[LoRAFormatAdapter] normalize_lora_state_dict called, #keys=%d",
        len(keys),
    )
    if keys:
        log.info(
            "[LoRAFormatAdapter] before convert, sample keys (<=20): %s",
            ", ".join(_sample_keys(keys, 20)),
        )

    fmt = detect_lora_format_from_state_dict(state_dict)
    log.info("[LoRAFormatAdapter] detected format: %s", fmt)

    normalized = convert_lora_state_dict_by_format(state_dict, fmt, log)
    apply_lora_alpha_scaling_inplace(normalized, log, remove_alpha=True)

    norm_keys = list(normalized.keys())
    if norm_keys:
        log.info(
            "[LoRAFormatAdapter] after convert, sample keys (<=20): %s",
            ", ".join(_sample_keys(norm_keys, 20)),
        )

    return normalized
