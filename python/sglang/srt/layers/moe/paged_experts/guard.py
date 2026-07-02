"""Compatibility guard for Paged Experts.

Hard-fail at model init if the server is configured with a parallelism / placement mode the paging path
cannot honor yet, instead of silently paging the WRONG experts. Mirrors the style of sglang's own
``ServerArgs`` checks (assert/raise with a what / why / how-to-fix message) and fires before any weight
touches the GPU.

States (see the contribution plan, "TP/EP vs paging"):
  * not-supported-yet: ``tp_size`` / ``ep_size`` / ``pp_size`` / ``dp_size`` (single-GPU first cut; the
    rank-aware per-rank store is future work)
  * gate-now-subsume-later: ``enable_eplb`` (overlaps keep-warm; no-op at ``ep_size == 1`` anyway)
  * validate-before-allow: ``moe_a2a_backend`` (the dispatch/combine kernels must survive the K-slot remap)
  * hard: ``load_format == "dummy"`` (the host store reads REAL expert weights)
"""

from __future__ import annotations

from typing import Any


def check_paged_experts_compat(server_args: Any) -> None:
    """Raise ``RuntimeError`` if ``server_args`` is incompatible with Paged Experts.

    Call once, before wrapping any MoE layer. Paged Experts is single-GPU for now: any multi-device
    parallelism (tp/ep/pp/dp) is rejected.
    """
    tp = getattr(server_args, "tp_size", 1) or 1
    ep = getattr(server_args, "ep_size", 1) or 1
    pp = getattr(server_args, "pp_size", 1) or 1
    dp = getattr(server_args, "dp_size", 1) or 1
    a2a = getattr(server_args, "moe_a2a_backend", None)
    load_format = str(getattr(server_args, "load_format", "") or "")

    problems = []
    if tp > 1:
        problems.append(
            f"tensor parallelism (tp_size={tp}) is not supported yet: the host expert store is not "
            "rank-aware (single-GPU only for now). Use --tp-size 1."
        )
    if ep > 1:
        problems.append(
            f"expert parallelism (ep_size={ep}) is not supported yet: the store is built for all E "
            "experts, not this rank's E/ep_size local experts. Use --ep-size 1."
        )
    if pp > 1:
        problems.append(
            f"pipeline parallelism (pp_size={pp}) is not supported: the per-layer pool + pinned store "
            "assume all layers on one device. Use --pp-size 1."
        )
    if dp > 1:
        problems.append(
            f"data parallelism (dp_size={dp}) is untested: each replica needs its own pool + pinned "
            "store. Use --dp-size 1."
        )
    if getattr(server_args, "enable_eplb", False):
        problems.append(
            "EPLB (--enable-eplb) is gated: it relocates experts across ranks at runtime, but the "
            "resident map is built once (static, 1:1). It overlaps keep-warm and is a no-op at "
            "ep_size==1. Drop --enable-eplb."
        )
    if a2a not in (None, "none", ""):
        problems.append(
            f"MoE all-to-all backend (moe_a2a_backend={a2a!r}) is unvalidated: its dispatch/combine "
            "kernels may assume all local experts are GPU-resident & contiguously indexed, which the "
            "K-slot indirection breaks. Use --moe-a2a-backend none."
        )
    if load_format == "dummy":
        problems.append(
            "--load-format dummy is incompatible: the host expert store reads REAL weights. Use a real "
            "checkpoint."
        )
    if problems:
        raise RuntimeError(
            "Paged Experts is incompatible with the current parallelism / placement config:\n  - "
            + "\n  - ".join(problems)
        )


def check_paged_experts_quant(hf_text_config: Any) -> None:
    """Raise ``RuntimeError`` if the checkpoint's quantization is not one the paging path supports.

    The host store's fill + gather understand unquantized (bf16/fp16), gptq-marlin int4, and fp8
    BLOCK-quant tensor layouts. Anything else (AWQ, per-tensor fp8, compressed-tensors, ...) would be
    routed through the wrong fill and load WRONG weights — reject it up front instead.
    """
    qc = getattr(hf_text_config, "quantization_config", None)
    if qc is None:
        return  # unquantized (bf16/fp16) — supported
    quant_method = (
        (qc.get("quant_method") or "").lower() if isinstance(qc, dict) else ""
    )
    if quant_method == "gptq":
        return
    if quant_method == "fp8":
        # Only BLOCK quantization: its per-expert weight + block-scale rows copy straight from the
        # checkpoint. Per-tensor fp8 has [E]/[E,2] scalar scales the pinned transfer can't page
        # (sub-8-byte rows) and a different post-load path.
        if isinstance(qc, dict) and qc.get("weight_block_size"):
            return
        raise RuntimeError(
            "Paged Experts supports fp8 only with block quantization (weight_block_size, e.g. "
            "[128, 128]); this checkpoint uses per-tensor fp8 scales. Use a block-quant fp8, GPTQ "
            "int4, or unquantized checkpoint, or run without --enable-paged-experts."
        )
    raise RuntimeError(
        f"Paged Experts does not support quant_method={quant_method or 'unknown'!r}: the host "
        "store handles unquantized (bf16/fp16), gptq-marlin int4, and fp8 block-quant checkpoints "
        "only. Other packings (e.g. AWQ) would be routed through the wrong fill and load wrong "
        "weights. Use a supported checkpoint, or run without --enable-paged-experts."
    )
