from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

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
        layer.set_lora_weights(lora_A, lora_B, lora_path=lora_path, strength=strength)
        return
    except Exception as exc:
        msg = str(exc).lower()
        looks_like_shape_issue = isinstance(exc, AssertionError) or any(
            s in msg for s in ("shape", "size", "rank", "mismatch", "dim")
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

            for attr, val in (
                ("disable_lora", False),
                ("merged", False),
                ("lora_strength", strength),
                ("lora_path", lora_path),
            ):
                if hasattr(layer, attr):
                    setattr(layer, attr, val)
        except Exception as exc2:
            raise RuntimeError(
                f"Layer '{layer_name}': dynamic-rank fallback failed ({exc2})"
            ) from exc2


# ----------------------------
# Common tensor helpers
# ----------------------------
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


def _block_diag(mats: Sequence[torch.Tensor]) -> torch.Tensor:
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
    r0 = c0 = 0
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


def _sample_keys(keys: Iterable[str], k: int = 20) -> list[str]:
    out: list[str] = []
    for i, key in enumerate(keys):
        if i >= k:
            break
        out.append(key)
    return out


def _strip_diffusion_model_prefix(
    state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Return a dict with optional 'diffusion_model.' prefix stripped from keys."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("diffusion_model."):
            k = k[len("diffusion_model.") :]
        out[k] = v
    return out


# ----------------------------
# Fuse split projections (data-driven)
# ----------------------------
@dataclass(frozen=True)
class _FuseSpec:
    block_pat: str  # regex alternatives, e.g. "attn|attention"
    parts: tuple[str, ...]  # e.g. ("to_q","to_k","to_v")
    fused: str  # e.g. "to_qkv"


_QKV_SPECS: tuple[_FuseSpec, ...] = (
    _FuseSpec("attn|attention", ("to_q", "to_k", "to_v"), "to_qkv"),
    _FuseSpec(
        "attn|attention", ("add_q_proj", "add_k_proj", "add_v_proj"), "to_added_qkv"
    ),
)

_W13_SPECS: tuple[_FuseSpec, ...] = (
    _FuseSpec("feed_forward|ffn|mlp", ("w1", "w3"), "w13"),
)


def _fuse_parts(
    sd: Dict[str, torch.Tensor],
    prefix: str,
    parts: Sequence[str],
    fused_name: str,
    log: logging.Logger,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-3,
) -> tuple[bool, bool]:
    """
    Fuse N split projections into a single fused projection.

    Returns:
      (ok, used_blockdiag)
    """
    A_keys = [f"{prefix}.{p}.lora_A.weight" for p in parts]
    B_keys = [f"{prefix}.{p}.lora_B.weight" for p in parts]
    if not all(k in sd for k in (*A_keys, *B_keys)):
        return False, False

    As = [sd[k] for k in A_keys]
    Bs = [sd[k] for k in B_keys]
    if any(t.ndim != 2 for t in (*As, *Bs)):
        return False, False

    ref = As[0]
    dev, dtype = ref.device, ref.dtype
    As = [
        t.to(device=dev, dtype=dtype) if (t.device != dev or t.dtype != dtype) else t
        for t in As
    ]
    Bs = [
        t.to(device=dev, dtype=dtype) if (t.device != dev or t.dtype != dtype) else t
        for t in Bs
    ]

    layout = _infer_lora_layout(As[0], Bs[0])
    if layout is None:
        return False, False
    if any(_infer_lora_layout(a, b) != layout for a, b in zip(As, Bs)):
        return False, False

    # Require consistent "in_dim" only; allow ranks/out_dims to differ.
    if layout == "BA":
        in_dim = int(As[0].shape[1])
        if any(int(a.shape[1]) != in_dim for a in As):
            return False, False
    else:
        in_dim = int(As[0].shape[0])
        if any(int(a.shape[0]) != in_dim for a in As):
            return False, False

    shared_a = all(
        As[0].shape == a.shape and torch.allclose(As[0], a, atol=atol, rtol=rtol)
        for a in As[1:]
    )

    used_blockdiag = not shared_a
    if shared_a:
        A_fused = As[0]
        B_fused = torch.cat(Bs, dim=0 if layout == "BA" else 1)
        log.info(
            "[LoRAFormatAdapter] fuse %s: shared-A fusion at %s.%s (layout=%s, parts=%s)",
            fused_name,
            prefix,
            fused_name,
            layout,
            ",".join(parts),
        )
    else:
        A_fused = torch.cat(As, dim=0 if layout == "BA" else 1)
        B_fused = _block_diag(Bs)
        diffs = [_max_abs_diff(As[0], a) for a in As[1:]]
        log.warning(
            "[LoRAFormatAdapter] fuse %s: A mismatch at %s.%s (max|A0-Ai|=%s). "
            "Using block-diagonal exact fusion (layout=%s, rank expands).",
            fused_name,
            prefix,
            fused_name,
            max(diffs) if diffs else 0.0,
            layout,
        )

    sd[f"{prefix}.{fused_name}.lora_A.weight"] = A_fused
    sd[f"{prefix}.{fused_name}.lora_B.weight"] = B_fused

    alpha0 = f"{prefix}.{parts[0]}.alpha"
    if alpha0 in sd:
        sd[f"{prefix}.{fused_name}.alpha"] = sd[alpha0]

    for p in parts:
        sd.pop(f"{prefix}.{p}.lora_A.weight", None)
        sd.pop(f"{prefix}.{p}.lora_B.weight", None)
        sd.pop(f"{prefix}.{p}.alpha", None)

    return True, used_blockdiag


def _fuse_specs_inplace(
    sd: Dict[str, torch.Tensor], specs: Sequence[_FuseSpec], log: logging.Logger
) -> dict[str, int]:
    """Apply fuse specs in-place. Returns stats dict."""
    fused = skipped = a_mismatch = 0

    for spec in specs:
        p0 = re.escape(spec.parts[0])
        pat = re.compile(
            rf"^(?P<prefix>.+\.(?:{spec.block_pat}))\.{p0}\.lora_A\.weight$"
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
                f"{prefix}.{spec.fused}.lora_A.weight" in sd
                or f"{prefix}.{spec.fused}.lora_B.weight" in sd
            ):
                continue

            ok, used_blockdiag = _fuse_parts(sd, prefix, spec.parts, spec.fused, log)
            if ok:
                fused += 1
                if used_blockdiag:
                    a_mismatch += 1
            else:
                skipped += 1

    return {"fused": fused, "skipped": skipped, "A_mismatch": a_mismatch}


# Back-compat wrappers — keep these names in case other code imports them.
def fuse_qkv_like_keys_inplace(
    sd: Dict[str, torch.Tensor], log: logging.Logger
) -> dict[str, int]:
    return _fuse_specs_inplace(sd, _QKV_SPECS, log)


def fuse_w13_like_keys_inplace(
    sd: Dict[str, torch.Tensor], log: logging.Logger
) -> dict[str, int]:
    return _fuse_specs_inplace(sd, _W13_SPECS, log)


def _maybe_fuse_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    specs: Sequence[_FuseSpec],
    sample_pred: Callable[[str], bool],
    log_prefix: str,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, torch.Tensor]:
    log = logger or globals()["logger"]
    if not state_dict:
        return dict(state_dict)

    sd = _strip_diffusion_model_prefix(state_dict)
    stats = _fuse_specs_inplace(sd, specs, log)

    if stats["fused"] > 0:
        sample = [k for k in sd.keys() if sample_pred(k)][:20]
        log.info(
            "[LoRAFormatAdapter] fused %s LoRA: fused=%d skipped=%d A_mismatch=%d, sample keys (<=20): %s",
            log_prefix,
            stats["fused"],
            stats["skipped"],
            stats["A_mismatch"],
            ", ".join(sample) if sample else "(none)",
        )
    return sd


def maybe_fuse_qwen_image_qkv_lora_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, torch.Tensor]:
    return _maybe_fuse_state_dict(
        state_dict,
        specs=_QKV_SPECS,
        sample_pred=lambda k: (".to_qkv." in k or ".to_added_qkv." in k),
        log_prefix="qkv/add_qkv",
        logger=logger,
    )


def maybe_fuse_w13_lora_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, torch.Tensor]:
    return _maybe_fuse_state_dict(
        state_dict,
        specs=_W13_SPECS,
        sample_pred=lambda k: ".w13." in k,
        log_prefix="w1/w3->w13",
        logger=logger,
    )


# ----------------------------
# Alpha bake
# ----------------------------
def apply_lora_alpha_scaling_inplace(
    sd: Dict[str, torch.Tensor],
    log: logging.Logger,
    *,
    remove_alpha: bool = True,
    atol: float = 1e-8,
) -> int:
    """Bake (alpha / rank) scaling into lora_B.weight. Returns #modules scaled."""
    scaled = 0
    for kA in list(sd.keys()):
        if not kA.endswith(".lora_A.weight"):
            continue
        prefix = kA[: -len(".lora_A.weight")]
        kB = prefix + ".lora_B.weight"
        kAlpha = prefix + ".alpha"
        if kB not in sd or kAlpha not in sd:
            continue

        A, B, alpha_t = sd[kA], sd[kB], sd[kAlpha]
        try:
            alpha_val = float(alpha_t.reshape(-1)[0].item())
        except Exception:
            log.warning("[LoRAFormatAdapter] cannot parse alpha for %s", prefix)
            continue

        layout = _infer_lora_layout(A, B)
        if layout is None:
            log.warning("[LoRAFormatAdapter] cannot infer layout for %s", prefix)
            continue

        r = int(A.shape[0]) if layout == "BA" else int(B.shape[0])
        if r <= 0:
            continue

        scale = alpha_val / float(r)
        if abs(scale) <= atol:
            continue

        sd[kB] = B * torch.tensor(scale, device=B.device, dtype=B.dtype)
        if remove_alpha:
            sd.pop(kAlpha, None)
        scaled += 1

    if scaled:
        log.info(
            "[LoRAFormatAdapter] baked alpha/rank into lora_B for %d modules", scaled
        )
    return scaled


# ----------------------------
# Target-aware fuse decisions
# ----------------------------
def _inspect_target_fused(target_layer_names: Iterable[str]) -> dict[str, bool]:
    """Inspect target module names for fused projections we may need to adapt to."""
    names = list(target_layer_names)

    def has_suffix(*suffixes: str) -> bool:
        return any(n.endswith(suffixes) for n in names)

    return {
        "to_qkv": has_suffix("attn.to_qkv", "attention.to_qkv"),
        "to_added_qkv": has_suffix("attn.to_added_qkv", "attention.to_added_qkv"),
        "w13": has_suffix("feed_forward.w13", "ffn.w13", "mlp.w13"),
    }


def _inspect_sd_split(sd: Mapping[str, torch.Tensor]) -> dict[str, bool]:
    keys = sd.keys()

    def any_contains(*subs: str) -> bool:
        return any(any(s in k for s in subs) for k in keys)

    return {
        "split_qkv": any_contains(
            ".attn.to_q.lora_A.weight", ".attention.to_q.lora_A.weight"
        ),
        "split_added_qkv": any_contains(
            ".attn.add_q_proj.lora_A.weight", ".attention.add_q_proj.lora_A.weight"
        ),
        "split_w1": any_contains(
            ".feed_forward.w1.lora_A.weight",
            ".ffn.w1.lora_A.weight",
            ".mlp.w1.lora_A.weight",
        ),
        "split_w3": any_contains(
            ".feed_forward.w3.lora_A.weight",
            ".ffn.w3.lora_A.weight",
            ".mlp.w3.lora_A.weight",
        ),
    }


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

    tgt = _inspect_target_fused(target_names)
    src = _inspect_sd_split(sd)

    if is_rank0:
        log.info(
            "[LoRAFormatAdapter] target-aware fuse check: "
            "target_to_qkv=%s sd_split_qkv=%s "
            "target_to_added_qkv=%s sd_split_added=%s "
            "target_w13=%s sd_split_w1=%s sd_split_w3=%s",
            tgt["to_qkv"],
            src["split_qkv"],
            tgt["to_added_qkv"],
            src["split_added_qkv"],
            tgt["w13"],
            src["split_w1"],
            src["split_w3"],
        )

    if fuse_qkv and (
        (tgt["to_qkv"] and src["split_qkv"])
        or (tgt["to_added_qkv"] and src["split_added_qkv"])
    ):
        sd = maybe_fuse_qwen_image_qkv_lora_state_dict(sd, logger=log)

    if fuse_w13 and tgt["w13"] and src["split_w1"] and src["split_w3"]:
        sd = maybe_fuse_w13_lora_state_dict(sd, logger=log)

    return sd


# ----------------------------
# Format detection & conversion
# ----------------------------
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
    return (".processor." in k) or (".proj_lora" in k) or (".qkv_lora" in k)


def _looks_like_kohya_flux(state_dict: Mapping[str, torch.Tensor]) -> bool:
    """Kohya FLUX LoRA (flux_lora.py) prefixes."""
    return any(
        k.startswith("lora_unet_double_blocks_")
        or k.startswith("lora_unet_single_blocks_")
        for k in state_dict.keys()
    )


def _looks_like_non_diffusers_sd(state_dict: Mapping[str, torch.Tensor]) -> bool:
    """Classic non-diffusers SD LoRA (Kohya/A1111/sd-scripts)."""
    keys = list(state_dict.keys())
    return bool(keys) and all(
        k.startswith(("lora_unet_", "lora_te_", "lora_te1_", "lora_te2_")) for k in keys
    )


def _looks_like_non_diffusers_qwen_unet(state_dict: Mapping[str, torch.Tensor]) -> bool:
    """A1111/kohya SD-LoRA keys that target Qwen-Image transformer blocks."""
    return any(
        k.startswith("lora_unet_transformer_blocks_")
        or k.startswith("diffusion_model.lora_unet_transformer_blocks_")
        for k in state_dict.keys()
    )


# --- Clean Qwen rewrite: single regex + token map ---
_QWEN_UNET_PREFIX_RE = re.compile(r"^lora_unet_transformer_blocks_(\d+)_([^.]+)\.")

_QWEN_UNET_TOKEN_MAP: dict[str, str] = {
    f"attn_to_{x}": f"attn.to_{x}" for x in ("q", "k", "v")
}
_QWEN_UNET_TOKEN_MAP.update(
    {
        "attn_to_out_0": "attn.to_out.0",
        "attn_add_q_proj": "attn.add_q_proj",
        "attn_add_k_proj": "attn.add_k_proj",
        "attn_add_v_proj": "attn.add_v_proj",
        "attn_to_add_out": "attn.to_add_out",
        "img_mlp_net_0_proj": "img_mlp.net.0.proj",
        "img_mlp_net_2": "img_mlp.net.2",
        "txt_mlp_net_0_proj": "txt_mlp.net.0.proj",
        "txt_mlp_net_2": "txt_mlp.net.2",
        "img_mod_1": "img_mod.1",
        "txt_mod_1": "txt_mod.1",
    }
)


def _rewrite_non_diffusers_qwen_key(key: str) -> str:
    """
    Rewrite:
      lora_unet_transformer_blocks_{i}_{token}.<rest>
    into:
      transformer_blocks.{i}.{mapped_token}.<rest>

    Keeps an optional leading 'diffusion_model.' prefix (if present).
    """
    dm_prefix = "diffusion_model."
    has_dm = key.startswith(dm_prefix)
    k = key[len(dm_prefix) :] if has_dm else key

    m = _QWEN_UNET_PREFIX_RE.match(k)
    if m is None:
        return key

    idx, token = m.group(1), m.group(2)
    mapped = _QWEN_UNET_TOKEN_MAP.get(token)
    if not mapped:
        return key

    rewritten = f"transformer_blocks.{idx}.{mapped}." + k[m.end() :]
    return (dm_prefix + rewritten) if has_dm else rewritten


def _convert_non_diffusers_qwen_lora_to_diffusers(
    state_dict: Mapping[str, torch.Tensor], log: logging.Logger
) -> Dict[str, torch.Tensor]:
    """Rewrite NON_DIFFUSERS_SD keys (Qwen-Image transformer blocks) into dot-notation."""
    out: Dict[str, torch.Tensor] = {
        _rewrite_non_diffusers_qwen_key(k): v for k, v in state_dict.items()
    }
    log.info(
        "[LoRAFormatAdapter] after NON_DIFFUSERS_QWEN rewrite, sample keys (<=20): %s",
        ", ".join(_sample_keys(out.keys(), 20)),
    )
    return out


def _looks_like_wan_lora(state_dict: Mapping[str, torch.Tensor]) -> bool:
    """Wan2.2 distill LoRAs (Wan-AI / Wan2.2-Distill-Loras style)."""
    for k in state_dict.keys():
        if not k.startswith("diffusion_model.blocks."):
            continue
        if ".lora_down" not in k and ".lora_up" not in k:
            continue
        if any(s in k for s in (".cross_attn.", ".self_attn.", ".ffn.", ".norm3.")):
            return True
    return False


def _looks_like_qwen_image(state_dict: Mapping[str, torch.Tensor]) -> bool:
    keys = list(state_dict.keys())
    return (
        bool(keys)
        and _has_prefix_key(keys, "transformer.transformer_blocks.")
        and (
            _has_substring_key(keys, ".lora.down.weight")
            or _has_substring_key(keys, ".lora.up.weight")
        )
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

    # Qwen-Image "standard" still goes through STANDARD branch (then rewrite down/up -> A/B)
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
        new_name = (
            name[len("transformer.") :] if name.startswith("transformer.") else name
        )
        if new_name.endswith(".lora.down.weight"):
            new_name = new_name.replace(".lora.down.weight", ".lora_A.weight")
        elif new_name.endswith(".lora.up.weight"):
            new_name = new_name.replace(".lora.up.weight", ".lora_B.weight")
        out[new_name] = tensor
    return out


def _convert_non_diffusers_sd_simple(
    state_dict: Mapping[str, torch.Tensor], log: logging.Logger
) -> Dict[str, torch.Tensor]:
    """Generic down/up -> A/B conversion for non-diffusers SD-like formats."""
    out: Dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        new_name = name.replace("lora_down.weight", "lora_A.weight").replace(
            "lora_up.weight", "lora_B.weight"
        )
        if new_name.endswith(".lora_down"):
            new_name = new_name[: -len(".lora_down")] + ".lora_A"
        elif new_name.endswith(".lora_up"):
            new_name = new_name[: -len(".lora_up")] + ".lora_B"
        out[new_name] = tensor

    log.info(
        "[LoRAFormatAdapter] after NON_DIFFUSERS_SD simple conversion, sample keys (<=20): %s",
        ", ".join(_sample_keys(out.keys(), 20)),
    )
    return out


def _convert_with_diffusers_utils_if_available(
    state_dict: Mapping[str, torch.Tensor],
    log: logging.Logger,
) -> Optional[Dict[str, torch.Tensor]]:
    """Use diffusers.lora_conversion_utils if available."""
    try:
        maybe_convert = getattr(lcu, "maybe_convert_state_dict", None)
        converted = (
            maybe_convert(state_dict) if callable(maybe_convert) else dict(state_dict)
        )
        if not isinstance(converted, dict):
            converted = dict(converted)

        log.info(
            "[LoRAFormatAdapter] diffusers.lora_conversion_utils converted keys, sample keys (<=20): %s",
            ", ".join(_sample_keys(converted.keys(), 20)),
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
    *,
    kind: str,
) -> Dict[str, torch.Tensor]:
    """Try multiple named converters in lora_conversion_utils, use the first that works."""
    converters = [
        (n, getattr(lcu, n)) for n in candidate_names if callable(getattr(lcu, n, None))
    ]
    if not converters:
        log.warning("[LoRAFormatAdapter] No %s converter found in diffusers.", kind)
        return dict(state_dict)

    last_err: Optional[Exception] = None
    for name, fn in converters:
        try:
            sd_copy = dict(state_dict)
            out = fn(sd_copy)
            if isinstance(out, tuple) and out and isinstance(out[0], dict):
                out = out[0]
            if not isinstance(out, dict):
                raise TypeError(f"Converter {name} returned {type(out)}")
            log.info("[LoRAFormatAdapter] Converted %s LoRA using %s", kind, name)
            return out
        except Exception as exc:
            last_err = exc

    log.warning(
        "[LoRAFormatAdapter] All %s converters failed; last error: %s", kind, last_err
    )
    return dict(state_dict)


def convert_lora_state_dict_by_format(
    state_dict: Mapping[str, torch.Tensor],
    fmt: LoRAFormat,
    log: logging.Logger,
) -> Dict[str, torch.Tensor]:
    """Normalize a raw LoRA state_dict into A/B + .weight naming."""
    if fmt == LoRAFormat.QWEN_IMAGE_STANDARD:
        return _convert_qwen_image_standard(state_dict, log)

    if fmt == LoRAFormat.XLABS_FLUX:
        converted = _convert_via_diffusers_candidates(
            state_dict,
            (
                "_convert_xlabs_flux_lora_to_diffusers",
                "convert_xlabs_lora_state_dict_to_diffusers",
                "convert_xlabs_lora_to_diffusers",
                "convert_xlabs_flux_lora_to_diffusers",
            ),
            log,
            kind="XLabs FLUX",
        )
        return _convert_non_diffusers_sd_simple(converted, log)

    if fmt == LoRAFormat.KOHYA_FLUX:
        converted = _convert_via_diffusers_candidates(
            state_dict,
            (
                "_convert_kohya_flux_lora_to_diffusers",
                "convert_kohya_flux_lora_to_diffusers",
            ),
            log,
            kind="Kohya FLUX",
        )
        return _convert_non_diffusers_sd_simple(converted, log)

    if fmt == LoRAFormat.WAN:
        maybe = _convert_with_diffusers_utils_if_available(state_dict, log) or dict(
            state_dict
        )
        return _convert_non_diffusers_sd_simple(maybe, log)

    if fmt == LoRAFormat.STANDARD:
        maybe = _convert_with_diffusers_utils_if_available(state_dict, log) or dict(
            state_dict
        )
        return (
            _convert_qwen_image_standard(maybe, log)
            if _looks_like_qwen_image(maybe)
            else maybe
        )

    if fmt == LoRAFormat.NON_DIFFUSERS_SD:
        maybe = _convert_with_diffusers_utils_if_available(state_dict, log) or dict(
            state_dict
        )
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
