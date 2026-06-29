"""Offline calibration script for the Double Sparsity channel mask file.

Runs a forward pass over a calibration corpus, accumulates Method 1
``mean(abs(Q_nope * K_nope))`` per-channel importance (hooks on the
``kv_b_proj`` K-side and ``q_b_proj`` Q-side), selects the top-``label_dim``
channels per (layer, head), and writes a ``safetensors`` mask for the runtime
selector. Default dataset is Pile-val; ``--dataset`` overrides; ``--allow-synthetic``
enables a CI/dev-only NIAH fallback.

``--dtype`` and ``--tp`` are recorded metadata only: the checkpoint loads in its
native dtype (``torch_dtype="auto"``) sharded by ``device_map="auto"``, with no
distributed group. ``--kv-cache-dtype`` goes into the mask metadata and must
match the serving ``--kv-cache-dtype``.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import torch

from sglang.srt.layers.attention.double_sparsity.channel_mask import (
    save_channel_mask,
)

logger = logging.getLogger(__name__)


_SUPPORTED_DTYPES = ("fp8_e4m3", "bfloat16")


def _niah_synthetic_prompts(
    num_samples: int, ctx_len: int, *, seed: int = 0
) -> List[str]:
    """Generate NIAH-shaped synthetic prompts (deterministic haystack + one needle)."""

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


def _extract_mla_nope_prefix(
    tensor: torch.Tensor,
    num_heads: int,
    nope_dim: int,
    suffix_dim: int,
) -> torch.Tensor:
    """Extract the noPE prefix from an MLA projection output, reshaping before slicing.

    MLA projections interleave per-head blocks (``kv_b_proj`` → ``[K_nope|V]``
    per head; ``q_b_proj`` → ``[Q_nope|Q_rope]`` per head), so flat-slicing the
    first ``H*nope_dim`` columns before reshape wrongly picks V/RoPE columns from
    later heads. Flatten leading dims first (handles 2-D and 3-D inputs), reshape
    ``[-1, H, nope_dim+suffix_dim]``, then slice ``[..., :nope_dim]``.
    """
    flat = tensor.reshape(-1, tensor.shape[-1])
    return flat.reshape(-1, num_heads, nope_dim + suffix_dim)[
        ..., :nope_dim
    ].contiguous()


def _build_pile_val_token_blocks(
    tokenizer,
    num_blocks: int,
    block_size: int,
    seed: int,
) -> List[torch.Tensor]:
    """Tokenize shuffled Pile-val docs, concatenate, and return fixed-size token blocks.

    Produces exactly ``num_blocks`` tensors of shape ``[1, block_size]``.
    Short documents are concatenated across boundaries; raises ``RuntimeError``
    if the corpus does not contain enough tokens.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Pile-val calibration requires the `datasets` library. "
            "Install via 'pip install datasets'."
        ) from exc

    needed = num_blocks * block_size
    ds = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    ds = ds.shuffle(seed=seed)

    all_ids: List[int] = []
    for example in ds:
        text = example.get("text", "")
        if not text.strip():
            continue
        enc = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
        ids = enc["input_ids"]
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        all_ids.extend(ids)
        if len(all_ids) >= needed:
            break

    if len(all_ids) < needed:
        raise RuntimeError(
            f"mit-han-lab/pile-val-backup produced only {len(all_ids)} tokens "
            f"after shuffling; need {needed} ({num_blocks} blocks × {block_size} "
            "tokens). Reduce --num-samples or increase corpus coverage."
        )

    blocks: List[torch.Tensor] = []
    for i in range(num_blocks):
        start = i * block_size
        block_ids = all_ids[start : start + block_size]
        blocks.append(torch.tensor([block_ids], dtype=torch.long))
    return blocks


def _summarize_param_placement(model) -> Dict[str, Any]:
    """Collect a dtype + device histogram across the model's parameters."""
    from collections import Counter

    dtype_counts: Counter[str] = Counter()
    device_counts: Counter[str] = Counter()
    total = 0
    for _name, p in model.named_parameters():
        dtype_counts[str(p.dtype)] += 1
        device_counts[str(p.device)] += 1
        total += 1
    hf_device_map = getattr(model, "hf_device_map", None)
    map_devices = (
        sorted({str(v) for v in hf_device_map.values()})
        if hf_device_map is not None
        else []
    )
    return {
        "total": total,
        "dtype_counts": dict(dtype_counts),
        "device_counts": dict(device_counts),
        "has_float8": any("float8" in d for d in dtype_counts),
        "hf_device_map_devices": map_devices,
    }


def _log_param_dtype_device_report(model) -> Dict[str, Any]:
    """Log and return the parameter dtype + device-placement report.

    Surfaces whether the FP8 checkpoint stayed FP8 (vs a silent bf16/fp16 upcast)
    and how Accelerate dispatched modules across GPUs — the evidence the dry-run gates on.
    """
    report = _summarize_param_placement(model)
    logger.info(
        "Loaded model parameter report: %d tensors; dtype histogram=%s; "
        "device histogram=%s; float8_present=%s.",
        report["total"],
        report["dtype_counts"],
        report["device_counts"],
        report["has_float8"],
    )
    if report["hf_device_map_devices"]:
        logger.info(
            "hf_device_map spans %d device(s): %s.",
            len(report["hf_device_map_devices"]),
            report["hf_device_map_devices"],
        )
    return report


def _enforce_dry_run_placement(report: Dict[str, Any]) -> None:
    """Fail-closed validation of a dry-run FP8 load on CUDA.

    Raises ``RuntimeError`` on a silently-degraded load: off-GPU placement
    (cpu/disk/meta offload), a single-GPU placement that did not shard, or a
    bf16/fp16 upcast (no float8 parameters).
    """
    param_devices = set(report["device_counts"])
    off_gpu = sorted(
        d for d in param_devices if any(x in d for x in ("cpu", "meta", "disk"))
    )
    if off_gpu:
        raise RuntimeError(
            f"Dry-run rejected: parameters placed off-GPU on {off_gpu} (device "
            f"histogram {report['device_counts']}). A sharded FP8 load must keep "
            "every parameter on CUDA; cpu/disk/meta means Accelerate offloaded, "
            "which breaks the calibration forward."
        )
    cuda_devices = sorted(d for d in param_devices if "cuda" in d)
    if len(cuda_devices) < 2:
        raise RuntimeError(
            f"Dry-run rejected: parameters span only {cuda_devices}; the V3.2 "
            "load must dispatch across multiple GPUs (device histogram "
            f"{report['device_counts']})."
        )
    if not report["has_float8"]:
        raise RuntimeError(
            "Dry-run rejected: the FP8-quantized checkpoint loaded with NO float8 "
            f"parameters (dtype histogram {report['dtype_counts']}) — a silent "
            "bf16/fp16 upcast. Calibration must run on the native FP8 weights."
        )


def _config_is_fp8(config) -> bool:
    """True when the model config declares FP8 quantization."""
    qc = getattr(config, "quantization_config", None)
    if qc is None:
        return False
    method = getattr(qc, "quant_method", None)
    if method is not None:
        return "fp8" in str(method).lower()
    return "fp8" in str(qc).lower()


def _resolve_calibration_config(model_path: str):
    """Resolve the HF config for calibration, remapping the unregistered V3.2.

    transformers has no ``deepseek_v32`` modeling and the checkpoint ships no
    remote code, but V3.2 is V3 plus the DSA indexer (irrelevant to calibration —
    only the identical MLA ``kv_b_proj``/``q_b_proj`` projections matter), so remap
    ``deepseek_v32`` → ``deepseek_v3``. Falls back to ``AutoConfig.from_pretrained``
    when the raw config cannot be read.
    """
    from transformers import AutoConfig, PretrainedConfig

    try:
        config_dict, _ = PretrainedConfig.get_config_dict(model_path)
    except Exception:
        return AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    if config_dict.get("model_type") == "deepseek_v32":
        logger.warning(
            "Remapping unregistered model_type 'deepseek_v32' -> 'deepseek_v3' "
            "for the calibration load: the V3.2 DSA indexer is irrelevant to "
            "channel-importance calibration (only MLA kv_b_proj/q_b_proj matter)."
        )
        remapped = {k: v for k, v in config_dict.items() if k != "model_type"}
        remapped["architectures"] = ["DeepseekV3ForCausalLM"]
        return AutoConfig.for_model("deepseek_v3", **remapped)

    return AutoConfig.from_pretrained(model_path, trust_remote_code=True)


def _force_triton_fp8_for_calibration() -> None:
    """Force transformers' FP8 matmul onto the finegrained-fp8 Triton path.

    The DeepGEMM hub kernel's fetch pulls a large cutlass tree that is unreliable
    under HF Hub rate limiting and raises a non-``ImportError`` deep inside the
    fetch, escaping transformers' ``except ImportError`` and crashing the forward.
    Making ``_load_deepgemm_kernel`` raise ``ImportError`` immediately routes the
    forward to the numerically-equivalent Triton fallback. Idempotent.
    """
    try:
        from transformers.integrations import finegrained_fp8 as _fgfp8
    except Exception:
        return
    if getattr(_fgfp8, "_ds_calib_force_triton", False):
        return

    def _skip_deepgemm():
        raise ImportError(
            "DeepGEMM kernel skipped for calibration (its hub fetch pulls a large "
            "cutlass source tree that is unreliable under HF Hub rate limiting); "
            "using the finegrained-fp8 Triton fallback instead."
        )

    _fgfp8._load_deepgemm_kernel = _skip_deepgemm
    _fgfp8._ds_calib_force_triton = True


def _load_calibration_model(model_path: str, use_cuda: bool):
    """Load the calibration model + tokenizer + resolved config.

    Loads in native dtype (no bf16/fp16 upcast) and shards across visible GPUs via
    Accelerate: a large FP8 checkpoint loaded as bf16 would roughly double the
    footprint and pin to one device, neither of which fits one H200.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Calibration requires the `transformers` library. Install via "
            "'pip install transformers' or run on the deployed image."
        ) from exc

    config = _resolve_calibration_config(model_path)
    # device_map="auto" (full GPU placement): a per-GPU max_memory cap spills
    # modules to cpu/disk, which the finegrained-fp8 quantizer rejects. Mid-load
    # OOM is fragmentation — fix with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype="auto",
        device_map="auto" if use_cuda else {"": "cpu"},
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    _force_triton_fp8_for_calibration()
    return model, tokenizer, config


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
    block_size: Optional[int] = None,
    use_pile_val: bool = False,
    pile_val_seed: int = 42,
    dry_run_blocks: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the calibration forward pass and return ``(importance, weights)``.

    Implements Method 1 ``mean(abs(Q_nope * K_nope))`` per channel: per-layer Q
    and K forward hooks accumulate the element-wise product, falling back to
    K-only L2 when a layer's Q projection is absent. MLA outputs are reshaped
    per-head before slicing the noPE prefix (see ``_extract_mla_nope_prefix``).
    ``allow_synthetic`` gates a CI/dev-only synthetic path.
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

    use_cuda = torch.cuda.is_available()
    model, tokenizer, config = _load_calibration_model(model_path, use_cuda)
    report = _log_param_dtype_device_report(model)
    # Fail-closed gate before a full run: reject off-GPU/un-sharded/upcast loads.
    if dry_run_blocks > 0 and use_cuda and _config_is_fp8(config):
        _enforce_dry_run_placement(report)

    num_layers = int(getattr(config, "num_hidden_layers", num_layers_hint or 0))
    num_heads = int(getattr(config, "num_attention_heads", num_heads_hint or 0))
    # MLA per-head K-noPE width is `qk_nope_head_dim`; plain attention has no
    # split, so K width equals `head_dim`. The mask indices into K-noPE space.
    qk_nope_head_dim = int(getattr(config, "qk_nope_head_dim", 0))
    v_head_dim = int(getattr(config, "v_head_dim", 0))
    head_dim = int(getattr(config, "head_dim", head_dim_hint or 0)) or (
        getattr(config, "hidden_size", 0) // max(num_heads, 1)
    )
    k_head_dim = qk_nope_head_dim if qk_nope_head_dim > 0 else head_dim
    # For MLA (DeepSeek), read qk_rope_head_dim directly from config when present.
    # Deriving from head_dim - qk_nope_head_dim is wrong when head_dim is itself
    # derived from hidden_size // num_heads (e.g. V3.2: 7168//128=56, not 128+64=192).
    _cfg_rope = int(getattr(config, "qk_rope_head_dim", 0))
    if _cfg_rope > 0:
        qk_rope_head_dim = _cfg_rope
    elif qk_nope_head_dim > 0:
        qk_rope_head_dim = head_dim - qk_nope_head_dim
        if qk_rope_head_dim <= 0:
            raise RuntimeError(
                f"Cannot derive qk_rope_head_dim for MLA config at {model_path!r}: "
                f"head_dim={head_dim} - qk_nope_head_dim={qk_nope_head_dim} = "
                f"{qk_rope_head_dim}. Ensure the config has an explicit "
                "'qk_rope_head_dim' field or pass --head-dim with the correct "
                "total per-head dimension."
            )
    else:
        qk_rope_head_dim = 0
    if num_layers <= 0 or num_heads <= 0 or k_head_dim <= 0:
        raise RuntimeError(
            f"Could not derive calibration shape from {model_path!r}: "
            f"L={num_layers} H={num_heads} D={k_head_dim}."
        )

    importance = torch.zeros((num_layers, num_heads, k_head_dim), dtype=torch.float32)
    seen = [0] * num_layers

    # Per-layer Q/K buffers; _accumulate_method1 fires once both are populated.
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

    handles = []
    # Expected flat output widths for MLA projections.
    full_mla_k_width = num_heads * (k_head_dim + v_head_dim) if v_head_dim > 0 else None
    full_mla_q_width = (
        num_heads * (k_head_dim + qk_rope_head_dim) if qk_rope_head_dim > 0 else None
    )
    # Standard-attention path: no per-head suffix, full output = noPE width.
    std_k_width = num_heads * k_head_dim

    for layer_idx, layer in enumerate(getattr(model, "model", model).layers):
        attn = getattr(layer, "self_attn", layer)

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

        q_b_proj = getattr(attn, "q_b_proj", None)
        q_proj_mod = getattr(attn, "q_proj", None)
        wq = getattr(attn, "wq", None)
        if q_b_proj is not None:
            qproj: Optional[object] = q_b_proj
            is_mla_q = True
        elif q_proj_mod is not None:
            qproj = q_proj_mod
            is_mla_q = False
        elif wq is not None:
            qproj = wq
            is_mla_q = False
        else:
            qproj = None
            is_mla_q = False

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
            full_w=full_mla_k_width,
            std_w=std_k_width,
            layer_has_q=has_q,
        ):
            def _hook(_module, _inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                if tensor.dim() < 2:
                    return
                t = tensor.detach().to(torch.float32)
                if is_mla:
                    if full_w is not None and t.shape[-1] == full_w:
                        k_nope = _extract_mla_nope_prefix(
                            t, num_heads, k_head_dim, v_head_dim
                        )
                    elif t.shape[-1] == std_w:
                        # kv_b_proj already outputs noPE-only (e.g. model variant).
                        k_nope = t.reshape(-1, num_heads, k_head_dim)
                    else:
                        logger.warning(
                            "kv_b_proj output last-dim=%d does not match expected "
                            "[K|V] total=%s or K-only=%d; skipping layer %d K hook.",
                            t.shape[-1],
                            full_w,
                            std_w,
                            idx,
                        )
                        return
                else:
                    k_nope = t.reshape(-1, num_heads, k_head_dim)
                if layer_has_q:
                    _k_buf[idx] = k_nope
                    _accumulate_method1(idx)
                else:
                    _accumulate_k_only(idx, k_nope)

            return _hook

        def _make_q_hook(
            idx,
            is_mla=is_mla_q,
            full_w=full_mla_q_width,
            std_w=std_k_width,
        ):
            def _hook(_module, _inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                if tensor.dim() < 2:
                    return
                t = tensor.detach().to(torch.float32)
                if is_mla and full_w is not None and t.shape[-1] == full_w:
                    q_nope = _extract_mla_nope_prefix(
                        t, num_heads, k_head_dim, qk_rope_head_dim
                    )
                elif t.shape[-1] == std_w:
                    # Standard attention or already-noPE-only output.
                    q_nope = t.reshape(-1, num_heads, k_head_dim)
                else:
                    logger.warning(
                        "q projection output last-dim=%d does not match expected "
                        "[Q_noPE|Q_RoPE] total=%s or Q-only=%d; skipping layer %d Q hook.",
                        t.shape[-1],
                        full_w,
                        std_w,
                        idx,
                    )
                    return
                _q_buf[idx] = q_nope
                _accumulate_method1(idx)

            return _hook

        handles.append(kproj.register_forward_hook(_make_k_hook(layer_idx)))
        if has_q:
            handles.append(qproj.register_forward_hook(_make_q_hook(layer_idx)))

    if use_pile_val:
        _bsz = block_size or 512
        logger.info(
            "Building Pile-val token blocks: %d blocks × %d tokens (seed=%d).",
            len(prompts),
            _bsz,
            pile_val_seed,
        )
        token_blocks = _build_pile_val_token_blocks(
            tokenizer, len(prompts), _bsz, pile_val_seed
        )
    else:
        tok_kwargs: dict = {"return_tensors": "pt"}
        if block_size is not None:
            tok_kwargs["max_length"] = block_size
            tok_kwargs["truncation"] = True
        token_blocks = None

    # device_map="auto" leaves no single model.device; inputs must land on the
    # device hosting the input embedding (first module Accelerate runs). Resolve
    # defensively so dispatched, single-device, and fake models all work.
    def _resolve_input_device() -> torch.device:
        get_emb = getattr(model, "get_input_embeddings", None)
        if callable(get_emb):
            try:
                emb = get_emb()
                weight = getattr(emb, "weight", None)
                if weight is not None:
                    return weight.device
            except Exception:
                pass
        dev = getattr(model, "device", None)
        if isinstance(dev, torch.device):
            return dev
        try:
            return next(model.parameters()).device
        except Exception:
            return torch.device("cpu")

    input_device = _resolve_input_device()

    if dry_run_blocks > 0:
        logger.info(
            "DRY RUN: limiting calibration to the first %d block(s); the mask "
            "will NOT be written. Inputs route to %s.",
            dry_run_blocks,
            input_device,
        )
        if token_blocks is not None:
            token_blocks = token_blocks[:dry_run_blocks]
        else:
            prompts = prompts[:dry_run_blocks]

    try:
        if token_blocks is not None:
            for block in token_blocks:
                with torch.no_grad():
                    model(input_ids=block.to(input_device))
        else:
            for prompt in prompts:
                inputs = tokenizer(prompt, **tok_kwargs).to(input_device)
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
    # --kv-cache-dtype goes into mask metadata and must match the serving value;
    # defaults to --dtype when omitted, though the two are semantically distinct.
    mask_dtype: str = getattr(args, "kv_cache_dtype", None) or args.dtype
    if mask_dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"--kv-cache-dtype must be one of {_SUPPORTED_DTYPES}, got {mask_dtype!r}."
        )
    if args.label_dim <= 0:
        raise ValueError(f"--label-dim must be positive, got {args.label_dim}.")
    if args.page_size <= 0:
        raise ValueError(f"--page-size must be positive, got {args.page_size}.")
    if args.tp <= 0:
        raise ValueError(f"--tp must be positive, got {args.tp}.")

    block_size = args.block_size
    seed = args.seed

    use_pile_val = False
    if args.dataset:
        prompts = _read_corpus_file(args.dataset, args.num_samples)
        dataset_source = f"file:{args.dataset}"
    elif args.allow_synthetic:
        prompts = _niah_synthetic_prompts(args.num_samples, args.ctx_len)
        dataset_source = "niah_synthetic"
    else:
        # Production path: Pile-val. Block construction is deferred to
        # _collect_channel_importance so token IDs concatenate across doc boundaries.
        prompts = [
            ""
        ] * args.num_samples  # length-hint only; content ignored when use_pile_val=True
        dataset_source = "mit-han-lab/pile-val-backup"
        use_pile_val = True

    dry_run_blocks = int(getattr(args, "dry_run_blocks", 0) or 0)
    importance, weights = _collect_channel_importance(
        model_path=args.model,
        dtype=args.dtype,
        tp=args.tp,
        num_layers_hint=args.num_layers,
        num_heads_hint=args.num_heads,
        head_dim_hint=args.head_dim,
        prompts=prompts,
        allow_synthetic=args.allow_synthetic,
        block_size=block_size,
        use_pile_val=use_pile_val,
        pile_val_seed=seed,
        dry_run_blocks=dry_run_blocks,
    )

    if dry_run_blocks > 0:
        L, H, head_dim = importance.shape
        logger.info(
            "DRY RUN complete: calibration hooks fired on all %d layers "
            "(H=%d, head_dim=%d) over %d block(s); parameter dtypes/devices "
            "logged above. Mask NOT written — rerun without --dry-run-blocks "
            "for the full calibration.",
            L,
            H,
            head_dim,
            dry_run_blocks,
        )
        return ""

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
        "dataset_source": dataset_source,
        "seed": str(seed),
        "num_samples": str(len(prompts)),
        "block_size": str(block_size),
    }
    content_hash = save_channel_mask(
        args.output,
        channel_selection,
        selected_weights,
        dtype=mask_dtype,
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
        description="Calibrate the Double Sparsity channel mask file (e.g. GLM-5.1-FP8).",
    )
    p.add_argument("--model", required=True, help="HuggingFace ID or local path.")
    p.add_argument(
        "--dtype",
        required=True,
        choices=_SUPPORTED_DTYPES,
        help=(
            "Model loading dtype for the calibration forward pass. "
            "Use 'bfloat16' for stability even when serving in FP8."
        ),
    )
    p.add_argument(
        "--kv-cache-dtype",
        default=None,
        choices=_SUPPORTED_DTYPES,
        help=(
            "KV-cache dtype written into the mask metadata; must match "
            "--kv-cache-dtype at serving time. Defaults to --dtype when omitted. "
            "For FP8 serving, pass --dtype bfloat16 --kv-cache-dtype fp8_e4m3."
        ),
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
        "--num-samples",
        type=int,
        default=256,
        help="Number of calibration prompts/blocks (Pile-val default: 256).",
    )
    p.add_argument(
        "--block-size",
        type=int,
        default=512,
        help="Token count per block when using Pile-val (default: 512).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Pile-val shuffle (default: 42).",
    )
    p.add_argument(
        "--ctx-len",
        type=int,
        default=4096,
        help="Approx token length per prompt (NIAH synthetic only).",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="External corpus path (newline-delimited prompts). Overrides Pile-val default.",
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
            "zai-org/GLM-5.1-FP8) is loaded via "
            "AutoModelForCausalLM.from_pretrained."
        ),
    )
    p.add_argument(
        "--dry-run-blocks",
        type=int,
        default=0,
        help=(
            "If > 0, load the model, log parameter dtypes + device placement, "
            "run only this many calibration blocks to confirm the Q/K hooks "
            "fire on every layer, then exit WITHOUT writing a mask. Use to "
            "validate the native-FP8 sharded load before the full run."
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
