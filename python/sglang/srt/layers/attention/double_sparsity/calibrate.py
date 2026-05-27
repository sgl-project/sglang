"""Offline calibration script for the Double Sparsity channel mask file.

The calibrator runs a forward pass over a calibration corpus on the target
model, collects per-channel L2 importance statistics on the K projections,
selects the top-``label_dim`` channels per (layer, head), and writes a
``safetensors`` file that the runtime selector consumes.

Default corpus: a NIAH-shaped synthetic dataset that puts a "needle"
token at a known position in a 4K-token "haystack" — small, deterministic,
and self-contained for CI. ``--dataset`` overrides with a path to an
external corpus (newline-delimited prompts).

Production recipe (DeepSeek-V3.2 FP8 on 2-node H200):

    python -m sglang.srt.layers.attention.double_sparsity.calibrate \\
        --model deepseek-ai/DeepSeek-V3.2 \\
        --dtype fp8_e4m3 \\
        --tp 8 \\
        --output dsv32-fp8-channel-mask.safetensors \\
        --label-dim 16 \\
        --page-size 64 \\
        --batch-size 4 \\
        --num-samples 1024

CI runs against a tiny NSA-shaped fixture under one minute with ``--tp 1
--allow-synthetic`` (the synthetic fallback is opt-in; HF repo IDs without
that flag must succeed through ``AutoModelForCausalLM.from_pretrained``).
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
from typing import List, Optional

import torch

from sglang.srt.layers.attention.double_sparsity.channel_mask import (
    save_channel_mask,
)

logger = logging.getLogger(__name__)


_SUPPORTED_DTYPES = ("fp8_e4m3", "bfloat16")


def _niah_synthetic_prompts(num_samples: int, ctx_len: int, *, seed: int = 0) -> List[str]:
    """Generate NIAH-shaped synthetic prompts.

    Each prompt is a deterministic "haystack" of repeated filler tokens with
    one "needle" sentence at a random position. The exact tokenization
    depends on the model's tokenizer; the prompt strings are designed to be
    short enough that even a 32K-token tokenizer expands to roughly
    ``ctx_len`` tokens per prompt.
    """

    rng = torch.Generator().manual_seed(seed)
    needle = "The hidden password is 47-RAVEN-92."
    filler = (
        "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do "
        "eiusmod tempor incididunt ut labore et dolore magna aliqua. "
    )
    prompts: List[str] = []
    for i in range(num_samples):
        haystack_size = max(1, int(ctx_len / max(1, len(filler.split()))))
        haystack = (filler * haystack_size).split()
        pos = int(torch.randint(0, len(haystack), (1,), generator=rng).item())
        haystack[pos] = needle
        prompts.append(" ".join(haystack))
    return prompts


def _read_corpus_file(path: str, num_samples: int) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise ValueError(
            f"calibration corpus {path!r} contains no non-empty lines. "
            "Provide a corpus with at least one prompt, or omit --dataset "
            "to fall back to the NIAH-shaped synthetic default."
        )
    if len(lines) < num_samples:
        logger.warning(
            "calibration corpus %s has %d non-empty lines; requested %d samples will be reused with wrap-around.",
            path,
            len(lines),
            num_samples,
        )
        # Wrap-around so callers always get exactly num_samples prompts.
        wrap = []
        i = 0
        while len(wrap) < num_samples:
            wrap.append(lines[i % len(lines)])
            i += 1
        return wrap
    return lines[:num_samples]


def _collect_channel_importance(
    *,
    model_path: str,
    dtype: str,
    tp: int,
    num_layers_hint: Optional[int],
    num_heads_hint: Optional[int],
    head_dim_hint: Optional[int],
    prompts: List[str],
    allow_synthetic: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the calibration forward pass and return ``(selection, weights)``.

    The real path implements **Method 1** from the original DoubleSparse paper:
    ``mean(abs(Q_nope * K_nope))`` per channel, accumulated over tokens and
    forward passes.  Both a Q-projection hook and a K-projection hook are
    registered per layer.  When both fire for the same pass the element-wise
    Q*K product is computed and added to the running importance tensor.  If a
    layer's Q projection cannot be found the accumulator falls back to
    K-only L2 (``sum(K^2)``) with a logged warning, preserving coverage for
    architectures that do not expose a hookable Q projection.

    For MLA layers (DeepSeek V3.2):
    * K side — ``kv_b_proj`` output ``[T, H*(nope+v)]`` → slice noPE prefix
      ``[..., :H*nope]`` → reshape ``[T, H, nope_dim]``.
    * Q side — ``q_b_proj`` output ``[T, H*(nope+rope)]`` → slice noPE prefix
      ``[..., :H*nope]`` → reshape ``[T, H, nope_dim]``.

    For standard attention (Llama / GLM):
    * K side — ``k_proj`` or ``wk`` output → reshape ``[T, H, head_dim]``.
    * Q side — ``q_proj`` output → reshape ``[T, H, head_dim]``.

    The synthetic-statistics path is gated behind ``allow_synthetic=True``
    and is reserved for CI fixtures + developer smoke tests.
    """

    if allow_synthetic and (
        not os.path.isdir(model_path) and not os.path.exists(model_path)
    ):
        logger.warning(
            "Synthetic calibration explicitly requested (--allow-synthetic) and "
            "model_path %s is not on disk. The resulting mask is for CI / dev "
            "smoke tests only and must NOT be used for production serving.",
            model_path,
        )
        L = num_layers_hint or 60
        H = num_heads_hint or 128
        D = head_dim_hint or 128
        rng = torch.Generator().manual_seed(7)
        importance = torch.rand((L, H, D), generator=rng, dtype=torch.float32)
        return importance, importance / importance.sum(dim=-1, keepdim=True)

    # Real forward-pass calibration: load the model + tokenizer, register
    # Method 1 Q+K hooks per attention layer, accumulate importance, then reduce.
    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Calibration requires the `transformers` library. Install via "
            "'pip install transformers' or run on the deployed image."
        ) from exc

    logger.info(
        "Loading %s for calibration (dtype=%s tp=%d num_prompts=%d method=Method1_QK).",
        model_path,
        dtype,
        tp,
        len(prompts),
    )
    if tp > 1:
        logger.warning(
            "calibrate.py defaults to a single-process forward pass; --tp=%d is "
            "recorded but does not initialize a distributed group. Production "
            "calibration typically uses --tp=1 with a model that fits one rank.",
            tp,
        )

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map={"": "cuda" if torch.cuda.is_available() else "cpu"},
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()

    num_layers = int(getattr(config, "num_hidden_layers", num_layers_hint or 0))
    num_heads = int(getattr(config, "num_attention_heads", num_heads_hint or 0))
    # For DeepSeek MLA the per-head K-noPE width is `qk_nope_head_dim`; for
    # plain Llama/GLM-style attention there is no such split and the per-head
    # K width equals `head_dim`.  The channel mask indices into K-noPE space.
    qk_nope_head_dim = int(getattr(config, "qk_nope_head_dim", 0))
    v_head_dim = int(getattr(config, "v_head_dim", 0))
    head_dim = int(getattr(config, "head_dim", head_dim_hint or 0)) or (
        getattr(config, "hidden_size", 0) // max(num_heads, 1)
    )
    k_head_dim = qk_nope_head_dim if qk_nope_head_dim > 0 else head_dim
    if num_layers <= 0 or num_heads <= 0 or k_head_dim <= 0:
        raise RuntimeError(
            f"Could not derive calibration shape from {model_path!r}: "
            f"L={num_layers} H={num_heads} D={k_head_dim}."
        )

    importance = torch.zeros((num_layers, num_heads, k_head_dim), dtype=torch.float32)
    seen = [0] * num_layers

    # Per-layer buffers for coordinating Q and K from the same forward pass.
    # Both hooks write here; _accumulate fires once both are populated.
    _q_buf: List[Optional[torch.Tensor]] = [None] * num_layers
    _k_buf: List[Optional[torch.Tensor]] = [None] * num_layers

    def _accumulate_method1(idx: int) -> None:
        """Method 1: mean(abs(Q_nope * K_nope)) over tokens."""
        q = _q_buf[idx]
        k = _k_buf[idx]
        if q is None or k is None:
            return
        _q_buf[idx] = None
        _k_buf[idx] = None
        # Both tensors are [T, H, k_head_dim] float32 at this point.
        contrib = (q * k).abs().mean(dim=0).cpu()  # [H, k_head_dim]
        importance[idx] += contrib
        seen[idx] += 1

    def _accumulate_k_only(idx: int, k: torch.Tensor) -> None:
        """K-only L2 fallback when Q projection is not accessible."""
        contrib = k.pow(2).mean(dim=0).cpu()  # [H, k_head_dim]
        importance[idx] += contrib
        seen[idx] += 1

    # Best-effort hook registration.  For MLA layers (kv_b_proj / q_b_proj)
    # the K-noPE and Q-noPE prefixes are sliced from the concatenated output
    # before accumulation.  The Q projection prefix width equals the K prefix
    # width (both == num_heads * k_head_dim) because Q_nope and K_nope share
    # the same noPE dimension.
    handles = []
    # noPE prefix width shared by K and Q in MLA, and equals full width in
    # standard attention (qk_nope_head_dim==0 case).
    nope_prefix_width = num_heads * k_head_dim
    full_mla_k_width = num_heads * (k_head_dim + v_head_dim) if v_head_dim > 0 else None

    for layer_idx, layer in enumerate(getattr(model, "model", model).layers):
        attn = getattr(layer, "self_attn", layer)

        # --- K projection ---
        k_proj = getattr(attn, "k_proj", None)
        kv_b_proj = getattr(attn, "kv_b_proj", None)
        wk = getattr(attn, "wk", None)
        if k_proj is not None:
            kproj, k_source = k_proj, "k_proj"
        elif kv_b_proj is not None:
            kproj, k_source = kv_b_proj, "kv_b_proj"
        elif wk is not None:
            kproj, k_source = wk, "wk"
        else:
            continue

        is_mla_k = k_source == "kv_b_proj"

        # --- Q projection (Method 1) ---
        q_b_proj = getattr(attn, "q_b_proj", None)
        q_proj_mod = getattr(attn, "q_proj", None)
        wq = getattr(attn, "wq", None)
        if q_b_proj is not None:
            qproj: Optional[object] = q_b_proj
        elif q_proj_mod is not None:
            qproj = q_proj_mod
        elif wq is not None:
            qproj = wq
        else:
            qproj = None

        has_q = qproj is not None
        if not has_q:
            logger.warning(
                "Layer %d: no Q projection found (tried q_b_proj, q_proj, wq). "
                "Falling back to K-only L2 importance for this layer.",
                layer_idx,
            )

        def _make_k_hook(
            idx,
            is_mla=is_mla_k,
            prefix=nope_prefix_width,
            full_w=full_mla_k_width,
            layer_has_q=has_q,
        ):
            def _hook(_module, _inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                if tensor.dim() < 2:
                    return
                if is_mla:
                    if full_w is not None and tensor.shape[-1] == full_w:
                        tensor = tensor[..., :prefix]
                    elif tensor.shape[-1] != prefix:
                        logger.warning(
                            "kv_b_proj output last-dim=%d does not match expected "
                            "K-noPE prefix=%d or [K|V] total=%s; skipping layer %d K hook.",
                            tensor.shape[-1], prefix, full_w, idx,
                        )
                        return
                k_nope = tensor.detach().to(torch.float32).reshape(-1, num_heads, k_head_dim)
                if layer_has_q:
                    _k_buf[idx] = k_nope
                    _accumulate_method1(idx)
                else:
                    _accumulate_k_only(idx, k_nope)

            return _hook

        def _make_q_hook(idx, prefix=nope_prefix_width):
            def _hook(_module, _inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                if tensor.dim() < 2:
                    return
                # Slice noPE prefix: works for both MLA (q_b_proj outputs
                # [Q_noPE | Q_RoPE]) and standard attention (full == prefix).
                if tensor.shape[-1] > prefix:
                    tensor = tensor[..., :prefix]
                elif tensor.shape[-1] < prefix:
                    logger.warning(
                        "q projection output last-dim=%d is narrower than expected "
                        "Q-noPE prefix=%d; skipping layer %d Q hook.",
                        tensor.shape[-1], prefix, idx,
                    )
                    return
                q_nope = tensor.detach().to(torch.float32).reshape(-1, num_heads, k_head_dim)
                _q_buf[idx] = q_nope
                _accumulate_method1(idx)

            return _hook

        handles.append(kproj.register_forward_hook(_make_k_hook(layer_idx)))
        if has_q:
            handles.append(qproj.register_forward_hook(_make_q_hook(layer_idx)))

    try:
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                model(**inputs)
    finally:
        for h in handles:
            h.remove()

    missing_layers = [i for i, s in enumerate(seen) if s == 0]
    if missing_layers:
        raise RuntimeError(
            f"Calibration hooks did not fire on {len(missing_layers)}/"
            f"{num_layers} layers (indices {missing_layers[:8]}"
            f"{'...' if len(missing_layers) > 8 else ''}). The resulting "
            "mask would have zero-importance rows that produce arbitrary "
            "top-K channels with zero weights — content-hash-valid but "
            "behaviorally invalid. The K-projection attribute search "
            "(k_proj / kv_b_proj / wk) failed on these layers, or "
            "kv_b_proj's output width did not match the expected K|V "
            "concatenation. Use --allow-synthetic if a dev/CI artifact is "
            "intentional; production calibration requires every layer to "
            "be covered."
        )

    weights = importance / importance.clamp_min(1e-6).sum(dim=-1, keepdim=True)
    return importance, weights


def calibrate(args: argparse.Namespace) -> str:
    if args.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"--dtype must be one of {_SUPPORTED_DTYPES}, got {args.dtype!r}."
        )
    if args.label_dim <= 0:
        raise ValueError(f"--label-dim must be positive, got {args.label_dim}.")
    if args.page_size <= 0:
        raise ValueError(f"--page-size must be positive, got {args.page_size}.")
    if args.tp <= 0:
        raise ValueError(f"--tp must be positive, got {args.tp}.")

    if args.dataset:
        prompts = _read_corpus_file(args.dataset, args.num_samples)
    else:
        prompts = _niah_synthetic_prompts(args.num_samples, args.ctx_len)

    importance, weights = _collect_channel_importance(
        model_path=args.model,
        dtype=args.dtype,
        tp=args.tp,
        num_layers_hint=args.num_layers,
        num_heads_hint=args.num_heads,
        head_dim_hint=args.head_dim,
        prompts=prompts,
        allow_synthetic=args.allow_synthetic,
    )

    L, H, head_dim = importance.shape
    if args.label_dim > head_dim:
        raise ValueError(
            f"--label-dim={args.label_dim} cannot exceed head_dim={head_dim}."
        )

    topk = torch.topk(importance, k=args.label_dim, dim=-1, largest=True)
    channel_selection = topk.indices.to(torch.int32)
    selected_weights = torch.gather(weights, dim=-1, index=topk.indices.long()).to(
        torch.float32
    )

    head_dim_arg = args.head_dim or head_dim
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    used_synthetic = args.allow_synthetic and not os.path.exists(args.model)
    extra = {
        "calibration_source": "synthetic" if used_synthetic else "real",
        "num_samples": str(len(prompts)),
        "ctx_len": str(args.ctx_len),
    }
    content_hash = save_channel_mask(
        args.output,
        channel_selection,
        selected_weights,
        dtype=args.dtype,
        head_dim=head_dim_arg,
        page_size=args.page_size,
        label_dim=args.label_dim,
        created_at=created_at,
        extra_metadata=extra,
    )
    logger.info(
        "Wrote channel mask to %s (content_sha256=%s, L=%d H=%d label_dim=%d).",
        args.output,
        content_hash[:12],
        L,
        H,
        args.label_dim,
    )
    return args.output


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m sglang.srt.layers.attention.double_sparsity.calibrate",
        description="Calibrate the Double Sparsity channel mask file for DeepSeek-V3.2 (FP8).",
    )
    p.add_argument("--model", required=True, help="HuggingFace ID or local path.")
    p.add_argument(
        "--dtype",
        required=True,
        choices=_SUPPORTED_DTYPES,
        help="kv_cache_dtype the channel mask is calibrated for.",
    )
    p.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor-parallel size at calibration time (informational; calibrate.py runs single-process).",
    )
    p.add_argument("--output", required=True, help="Output safetensors path.")
    p.add_argument("--label-dim", type=int, default=16)
    p.add_argument("--page-size", type=int, default=64)
    p.add_argument(
        "--num-samples", type=int, default=64, help="Calibration prompts."
    )
    p.add_argument(
        "--ctx-len", type=int, default=4096, help="Approx token length per prompt."
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="External corpus path (newline-delimited prompts). Defaults to NIAH synthetic.",
    )
    p.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Layer count hint when the model is not on disk (synthetic fallback).",
    )
    p.add_argument(
        "--num-heads",
        type=int,
        default=None,
        help="Head count hint when the model is not on disk (synthetic fallback).",
    )
    p.add_argument(
        "--head-dim",
        type=int,
        default=None,
        help="Head dim hint when the model is not on disk (synthetic fallback).",
    )
    p.add_argument(
        "--allow-synthetic",
        action="store_true",
        help=(
            "Allow the deterministic synthetic-statistics fallback when "
            "--model is not a local path. Reserved for CI fixtures and dev "
            "smoke tests; the resulting mask is NOT valid for production "
            "serving. Without this flag a HuggingFace repo ID (e.g. "
            "deepseek-ai/DeepSeek-V3.2) is loaded via "
            "AutoModelForCausalLM.from_pretrained."
        ),
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        calibrate(args)
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.error("calibration failed: %s", exc, exc_info=True)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
