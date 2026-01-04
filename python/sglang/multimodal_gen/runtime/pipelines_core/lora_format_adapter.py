from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional

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


def set_lora_weights_dynamic_rank(
    layer: Any,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    *,
    lora_path: str | None,
    strength: float,
    rank: int = 0,
    layer_name: str = "",
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Set LoRA weights on a layer.

    Prefer layer.set_lora_weights(). If it fails due to rank/shape mismatch,
    fall back to replacing layer.lora_A/lora_B parameters (inference-oriented).
    """
    log = logger or globals()["logger"]

    try:
        layer.set_lora_weights(
            lora_A,
            lora_B,
            lora_path=lora_path,
            strength=strength,
        )
        return
    except Exception as exc:
        msg = str(exc).lower()
        looks_like_shape_issue = (
            isinstance(exc, AssertionError)
            or ("shape" in msg)
            or ("size" in msg)
            or ("rank" in msg)
            or ("mismatch" in msg)
            or ("dim" in msg)
        )
        if not looks_like_shape_issue:
            raise

        if rank == 0:
            log.warning(
                "Layer '%s': set_lora_weights failed (%s). "
                "Falling back to dynamic parameter replacement (inference-safe).",
                layer_name,
                exc,
            )

        try:
            if (
                not hasattr(layer, "lora_A")
                or getattr(layer, "lora_A") is None
                or layer.lora_A.shape != lora_A.shape
            ):
                layer.lora_A = torch.nn.Parameter(lora_A, requires_grad=False)
            else:
                layer.lora_A.data.copy_(lora_A)

            if (
                not hasattr(layer, "lora_B")
                or getattr(layer, "lora_B") is None
                or layer.lora_B.shape != lora_B.shape
            ):
                layer.lora_B = torch.nn.Parameter(lora_B, requires_grad=False)
            else:
                layer.lora_B.data.copy_(lora_B)

            if hasattr(layer, "disable_lora"):
                layer.disable_lora = False
            if hasattr(layer, "merged"):
                layer.merged = False
            if hasattr(layer, "lora_strength"):
                layer.lora_strength = strength
            if hasattr(layer, "lora_path"):
                layer.lora_path = lora_path
        except Exception as exc2:
            raise RuntimeError(
                f"Layer '{layer_name}': dynamic-rank fallback failed ({exc2})"
            ) from exc2


def _target_has_fused_qkv(target_layer_names: Iterable[str]) -> tuple[bool, bool]:
    """Return (has_to_qkv, has_to_added_qkv) based on target module names."""
    names = list(target_layer_names)
    has_to_qkv = any(n.endswith(("attn.to_qkv", "attention.to_qkv")) for n in names)
    has_to_added_qkv = any(
        n.endswith(("attn.to_added_qkv", "attention.to_added_qkv")) for n in names
    )
    return has_to_qkv, has_to_added_qkv


def _target_has_fused_w13(target_layer_names: Iterable[str]) -> bool:
    """Return True if target uses fused FFN projection w13 (SwiGLU gate+up)."""
    names = list(target_layer_names)
    return any(n.endswith(("feed_forward.w13", "ffn.w13", "mlp.w13")) for n in names)


def _sd_has_split_qkv(sd: Mapping[str, torch.Tensor]) -> bool:
    keys = sd.keys()
    return any(
        (".attn.to_q.lora_A.weight" in k) or (".attention.to_q.lora_A.weight" in k)
        for k in keys
    )


def _sd_has_split_added_qkv(sd: Mapping[str, torch.Tensor]) -> bool:
    keys = sd.keys()
    return any(
        (".attn.add_q_proj.lora_A.weight" in k)
        or (".attention.add_q_proj.lora_A.weight" in k)
        for k in keys
    )


def _sd_has_split_w1(sd: Mapping[str, torch.Tensor]) -> bool:
    keys = sd.keys()
    return any(
        (".feed_forward.w1.lora_A.weight" in k)
        or (".ffn.w1.lora_A.weight" in k)
        or (".mlp.w1.lora_A.weight" in k)
        for k in keys
    )


def _sd_has_split_w3(sd: Mapping[str, torch.Tensor]) -> bool:
    keys = sd.keys()
    return any(
        (".feed_forward.w3.lora_A.weight" in k)
        or (".ffn.w3.lora_A.weight" in k)
        or (".mlp.w3.lora_A.weight" in k)
        for k in keys
    )


def prepare_lora_state_dict_for_target(
    raw_state_dict: Mapping[str, torch.Tensor],
    target_lora_layer_names: Optional[Iterable[str]] = None,
    logger: Optional[logging.Logger] = None,
    *,
    rank: int | None = None,
    fuse_qkv: bool = True,
    fuse_w13: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Normalize an external LoRA state_dict and (optionally) fuse keys to match
    target fused projections (to_qkv/to_added_qkv, w13).
    """
    log = logger or globals()["logger"]
    is_rank0 = (rank is None) or (rank == 0)

    sd = normalize_lora_state_dict(raw_state_dict, logger=log)

    if not target_lora_layer_names:
        return sd

    target_names = list(target_lora_layer_names)
    if not target_names:
        return sd

    target_has_to_qkv, target_has_to_added_qkv = _target_has_fused_qkv(target_names)
    target_has_w13 = _target_has_fused_w13(target_names)

    sd_has_split_qkv = _sd_has_split_qkv(sd)
    sd_has_split_added = _sd_has_split_added_qkv(sd)
    sd_has_w1 = _sd_has_split_w1(sd)
    sd_has_w3 = _sd_has_split_w3(sd)

    if is_rank0:
        log.info(
            "[LoRAFormatAdapter] target-aware fuse check: "
            "target_has_to_qkv=%s sd_has_split_qkv=%s "
            "target_has_to_added_qkv=%s sd_has_split_added=%s "
            "target_has_w13=%s sd_has_w1=%s sd_has_w3=%s",
            target_has_to_qkv,
            sd_has_split_qkv,
            target_has_to_added_qkv,
            sd_has_split_added,
            target_has_w13,
            sd_has_w1,
            sd_has_w3,
        )

    if fuse_qkv and (
        (target_has_to_qkv and sd_has_split_qkv)
        or (target_has_to_added_qkv and sd_has_split_added)
    ):
        sd = maybe_fuse_qwen_image_qkv_lora_state_dict(sd, logger=log)

        if is_rank0:
            keys2 = list(sd.keys())
            log.info(
                "[LoRAFormatAdapter] after qkv fuse: has_to_qkv=%s has_split_to_q=%s",
                any(".to_qkv.lora_A.weight" in k for k in keys2),
                any(".to_q.lora_A.weight" in k for k in keys2),
            )

    if fuse_w13 and target_has_w13 and sd_has_w1 and sd_has_w3:
        sd = maybe_fuse_w13_lora_state_dict(sd, logger=log)

        if is_rank0:
            keys3 = list(sd.keys())
            log.info(
                "[LoRAFormatAdapter] after w13 fuse: has_w13=%s has_w1=%s has_w3=%s",
                any(".w13.lora_A.weight" in k for k in keys3),
                any(".w1.lora_A.weight" in k for k in keys3),
                any(".w3.lora_A.weight" in k for k in keys3),
            )

    return sd


def apply_lora_alpha_scaling_inplace(
    sd: Dict[str, torch.Tensor],
    log: logging.Logger,
    *,
    remove_alpha: bool = True,
    atol: float = 1e-8,
) -> int:
    """Bake (alpha / rank) scaling into lora_B.weight. Returns #modules scaled."""
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
            r = int(A.shape[0])
        else:
            r = int(B.shape[0])

        if r <= 0:
            continue

        scale = alpha_val / float(r)
        if abs(scale) <= atol:
            continue

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

    Returns:
      - "BA" if ΔW = B @ A (B: out×r, A: r×in)
      - "AB" if ΔW = A @ B (A: in×r, B: r×out)
    """
    if A.ndim != 2 or B.ndim != 2:
        return None

    ba_ok = B.shape[1] == A.shape[0]
    ab_ok = A.shape[1] == B.shape[0]

    if ba_ok and not ab_ok:
        return "BA"
    if ab_ok and not ba_ok:
        return "AB"
    if ba_ok and ab_ok:
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
    """Fuse (q,k,v) into a single fused projection. Returns (ok, used_blockdiag)."""
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

    if layout == "BA":
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
            A_fused = Aq
            B_fused = torch.cat([Bq, Bk, Bv], dim=0)
            used_blockdiag = False
            log.info(
                "[LoRAFormatAdapter] QKV A shared at %s.%s, using shared-A fusion (layout=%s).",
                prefix,
                fused,
                layout,
            )
        else:
            A_fused = torch.cat([Aq, Ak, Av], dim=0)
            B_fused = _block_diag([Bq, Bk, Bv])
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
            A_fused = Aq
            B_fused = torch.cat([Bq, Bk, Bv], dim=1)
            used_blockdiag = False
            log.info(
                "[LoRAFormatAdapter] QKV A shared at %s.%s, using shared-A fusion (layout=%s).",
                prefix,
                fused,
                layout,
            )
        else:
            A_fused = torch.cat([Aq, Ak, Av], dim=1)
            B_fused = _block_diag([Bq, Bk, Bv])
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

    alpha_q = f"{prefix}.{q}.alpha"
    if alpha_q in sd:
        sd[f"{prefix}.{fused}.alpha"] = sd[alpha_q]

    for p in (q, k, v):
        sd.pop(f"{prefix}.{p}.lora_A.weight", None)
        sd.pop(f"{prefix}.{p}.lora_B.weight", None)
        sd.pop(f"{prefix}.{p}.alpha", None)

    return True, used_blockdiag


def fuse_qkv_like_keys_inplace(
    sd: Dict[str, torch.Tensor], log: logging.Logger
) -> dict[str, int]:
    """Fuse split q/k/v (and add_q/k/v) LoRA keys into fused projections, in-place."""
    patterns = [
        ("to_q", "to_k", "to_v", "to_qkv"),
        ("add_q_proj", "add_k_proj", "add_v_proj", "to_added_qkv"),
    ]

    fused = 0
    skipped = 0
    a_mismatch = 0
    for q, k, v, fused_name in patterns:
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
    """Fuse split Q/K/V LoRA weights into to_qkv/to_added_qkv when applicable."""
    log = logger or globals()["logger"]
    if not state_dict:
        return dict(state_dict)

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
    """Fuse (w1,w3) into w13. Returns (ok, used_blockdiag)."""
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

    if layout == "BA":
        in_dim = int(A1.shape[1])
        if int(A3.shape[1]) != in_dim:
            return False, False

        shared_a = A1.shape == A3.shape and torch.allclose(A1, A3, atol=atol, rtol=rtol)

        if shared_a:
            A_fused = A1
            B_fused = torch.cat([B1, B3], dim=0)
            used_blockdiag = False
            log.info(
                "[LoRAFormatAdapter] FFN w1/w3 A shared at %s.%s, using shared-A fusion (layout=%s).",
                prefix,
                fused,
                layout,
            )
        else:
            A_fused = torch.cat([A1, A3], dim=0)
            B_fused = _block_diag([B1, B3])
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
        in_dim = int(A1.shape[0])
        if int(A3.shape[0]) != in_dim:
            return False, False

        shared_a = A1.shape == A3.shape and torch.allclose(A1, A3, atol=atol, rtol=rtol)

        if shared_a:
            A_fused = A1
            B_fused = torch.cat([B1, B3], dim=1)
            used_blockdiag = False
            log.info(
                "[LoRAFormatAdapter] FFN w1/w3 A shared at %s.%s, using shared-A fusion (layout=%s).",
                prefix,
                fused,
                layout,
            )
        else:
            A_fused = torch.cat([A1, A3], dim=1)
            B_fused = _block_diag([B1, B3])
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

    alpha_1 = f"{prefix}.{w1}.alpha"
    if alpha_1 in sd:
        sd[f"{prefix}.{fused}.alpha"] = sd[alpha_1]

    for p in (w1, w3):
        sd.pop(f"{prefix}.{p}.lora_A.weight", None)
        sd.pop(f"{prefix}.{p}.lora_B.weight", None)
        sd.pop(f"{prefix}.{p}.alpha", None)

    return True, used_blockdiag


def fuse_w13_like_keys_inplace(
    sd: Dict[str, torch.Tensor], log: logging.Logger
) -> dict[str, int]:
    """Fuse split FFN w1/w3 LoRA into fused w13, in-place."""
    fused = 0
    skipped = 0
    a_mismatch = 0

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
    """Fuse split FFN w1/w3 LoRA into fused w13 when needed."""
    log = logger or globals()["logger"]
    if not state_dict:
        return dict(state_dict)

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
    """Kohya FLUX LoRA (flux_lora.py) prefixes."""
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


def _looks_like_non_diffusers_qwen_unet(state_dict: Mapping[str, torch.Tensor]) -> bool:
    """A1111/kohya SD-LoRA keys that target Qwen-Image transformer blocks."""
    if not state_dict:
        return False
    return any(k.startswith("lora_unet_transformer_blocks_") for k in state_dict.keys())


_QWEN_UNET_REWRITES: list[tuple[str, str]] = [
    (
        r"^lora_unet_transformer_blocks_(\d+)_attn_to_q\.",
        r"transformer_blocks.\1.attn.to_q.",
    ),
    (
        r"^lora_unet_transformer_blocks_(\d+)_attn_to_k\.",
        r"transformer_blocks.\1.attn.to_k.",
    ),
    (
        r"^lora_unet_transformer_blocks_(\d+)_attn_to_v\.",
        r"transformer_blocks.\1.attn.to_v.",
    ),
    (
        r"^lora_unet_transformer_blocks_(\d+)_attn_to_out_0\.",
        r"transformer_blocks.\1.attn.to_out.0.",
    ),
    (
        r"^lora_unet_transformer_blocks_(\d+)_attn_add_q_proj\.",
        r"transformer_blocks.\1.attn.add_q_proj.",
    ),
    (
        r"^lora_unet_transformer_blocks_(\d+)_attn_add_k_proj\.",
        r"transformer_blocks.\1.attn.add_k_proj.",
    ),
    (
        r"^lora_unet_transformer_blocks_(\d+)_attn_add_v_proj\.",
        r"transformer_blocks.\1.attn.add_v_proj.",
    ),
    (
        r"^lora_unet_transformer_blocks_(\d+)_attn_to_add_out\.",
        r"transformer_blocks.\1.attn.to_add_out.",
    ),
    (
        r"^lora_unet_transformer_blocks_(\d+)_img_mlp_net_0_proj\.",
        r"transformer_blocks.\1.img_mlp.net.0.proj.",
    ),
    (
        r"^lora_unet_transformer_blocks_(\d+)_img_mlp_net_2\.",
        r"transformer_blocks.\1.img_mlp.net.2.",
    ),
    (
        r"^lora_unet_transformer_blocks_(\d+)_txt_mlp_net_0_proj\.",
        r"transformer_blocks.\1.txt_mlp.net.0.proj.",
    ),
    (
        r"^lora_unet_transformer_blocks_(\d+)_txt_mlp_net_2\.",
        r"transformer_blocks.\1.txt_mlp.net.2.",
    ),
    (
        r"^lora_unet_transformer_blocks_(\d+)_img_mod_1\.",
        r"transformer_blocks.\1.img_mod.1.",
    ),
    (
        r"^lora_unet_transformer_blocks_(\d+)_txt_mod_1\.",
        r"transformer_blocks.\1.txt_mod.1.",
    ),
]


def _convert_non_diffusers_qwen_lora_to_diffusers(
    state_dict: Mapping[str, torch.Tensor], log: logging.Logger
) -> Dict[str, torch.Tensor]:
    """Rewrite NON_DIFFUSERS_SD keys (Qwen-Image transformer blocks) into dot-notation."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        nk = k
        for pat, rep in _QWEN_UNET_REWRITES:
            nk2 = re.sub(pat, rep, nk)
            if nk2 != nk:
                nk = nk2
                break
        out[nk] = v

    sample = _sample_keys(out.keys(), 20)
    log.info(
        "[LoRAFormatAdapter] after NON_DIFFUSERS_QWEN rewrite, sample keys (<=20): %s",
        ", ".join(sample),
    )
    return out


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


def detect_lora_format_from_state_dict(
    state_dict: Mapping[str, torch.Tensor]
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


def _convert_qwen_image_standard(
    state_dict: Mapping[str, torch.Tensor], log: logging.Logger
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
    state_dict: Mapping[str, torch.Tensor], log: logging.Logger
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
        "[LoRAFormatAdapter] after NON_DIFFUSERS_SD simple conversion, sample keys (<=20): %s",
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
            converted = lcu.maybe_convert_state_dict(state_dict)  # type: ignore[attr-defined]
        else:
            converted = dict(state_dict)

        if not isinstance(converted, dict):
            converted = dict(converted)

        sample = _sample_keys(converted.keys(), 20)
        log.info(
            "[LoRAFormatAdapter] diffusers.lora_conversion_utils converted keys, sample keys (<=20): %s",
            ", ".join(sample),
        )
        return converted
    except Exception as exc:  # pragma: no cover
        log.warning(
            "[LoRAFormatAdapter] diffusers lora_conversion_utils failed, falling back. Error: %s",
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
    state_dict: Mapping[str, torch.Tensor], log: logging.Logger
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
        unavailable_warning="[LoRAFormatAdapter] XLabs FLUX detected but diffusers is unavailable.",
        no_converter_warning="[LoRAFormatAdapter] No XLabs FLUX converter found in diffusers.",
        success_info="[LoRAFormatAdapter] Converted XLabs FLUX LoRA using {name}",
        all_failed_warning="[LoRAFormatAdapter] All XLabs FLUX converters failed; last error: {last_err}",
    )


def _convert_kohya_flux_via_diffusers(
    state_dict: Mapping[str, torch.Tensor], log: logging.Logger
) -> Dict[str, torch.Tensor]:
    """Convert Kohya FLUX LoRA via diffusers helpers."""
    return _convert_via_diffusers_candidates(
        state_dict,
        (
            "_convert_kohya_flux_lora_to_diffusers",
            "convert_kohya_flux_lora_to_diffusers",
        ),
        log=log,
        unavailable_warning="[LoRAFormatAdapter] Kohya FLUX detected but diffusers is unavailable.",
        no_converter_warning="[LoRAFormatAdapter] No Kohya FLUX converter found.",
        success_info="[LoRAFormatAdapter] Converted Kohya FLUX LoRA using {name}",
        all_failed_warning="[LoRAFormatAdapter] Kohya FLUX conversion failed; last error: {last_err}",
    )


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

        if _looks_like_non_diffusers_qwen_unet(maybe):
            maybe = _convert_non_diffusers_qwen_lora_to_diffusers(maybe, log)

        return _convert_non_diffusers_sd_simple(maybe, log)

    log.info(
        "[LoRAFormatAdapter] format %s not handled specially, returning as-is", fmt
    )
    return dict(state_dict)


def normalize_lora_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, torch.Tensor]:
    """Normalize any supported LoRA format into a canonical layout."""
    log = logger or globals()["logger"]

    keys = list(state_dict.keys())
    log.info(
        "[LoRAFormatAdapter] normalize_lora_state_dict called, #keys=%d", len(keys)
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
