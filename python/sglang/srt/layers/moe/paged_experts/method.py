"""PagedExpertsMoEMethod: the K-slot resident expert table.

Wraps the real fused-MoE quant method (unquantized bf16 / gptq-marlin int4) with a K-of-E resident
table; routing stays E-wide (the model's gate is untouched), only the expert TABLE is K, and the
forward remaps logical expert ids -> resident slots per step, paging misses from the pinned host store.

Weight loading for the K residents reuses sglang's NATIVE expert-parallel remap: setting
``layer.num_local_experts = K`` makes the default loader fill slots ``0..K-1`` and skip the rest (no
custom loader). Forward (``apply``) and the host store live in ``forward.py`` / ``pager.py`` (imported
lazily so this module loads without them). K sizing is ``sizing.compute_num_resident_experts``.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Any, Optional

import torch

from sglang.srt.layers.moe.paged_experts.guard import (
    check_paged_experts_compat,
    check_paged_experts_quant,
)
from sglang.srt.layers.moe.paged_experts.placement import make_placement
from sglang.srt.layers.moe.paged_experts.sizing import (
    compute_num_resident_experts,
    compute_window_experts,
    compute_window_size,
    kv_reserve_bytes_mha,
)

logger = logging.getLogger(__name__)

# Hybrid-state reserve: slots of per-request mamba/linear-attention state auto-K sets aside so the
# state cache sglang sizes AFTER weights cannot end up empty. Sized generously (32 slots ~ 2 GB at
# Qwen3.6 state sizes) because sglang splits the leftover between KV and mamba AND the radix cache
# multiplies the per-request slot need (~5x); this yields a couple of radix-on requests on a 16 GB
# card. For higher hybrid concurrency raise --paged-experts-kv-reserve-gb (smaller K, more state room).
_HYBRID_STATE_SLOTS = 32

# Captured at import (BEFORE model weights load) so the resolver sizes K against sglang's PRE-load free
# memory P — the basis of sglang's own KV accounting (KV_pool = post_load_free - P*(1-mem_fraction)).
# Using total board memory over-counts by the CUDA-context overhead and over-sizes K into an OOM.
try:
    _PRE_LOAD_FREE_BYTES = torch.cuda.mem_get_info()[0]
except Exception:
    _PRE_LOAD_FREE_BYTES = 0


from sglang.srt.layers.moe.paged_experts.store import (  # noqa: E402
    _host_available_bytes,
)

# Captured at import (BEFORE weights load) so the auto window is sized against the SAME pre-load host
# memory for every layer — a uniform W across layers (the store's on-device gather assumes it), and a
# stable basis unaffected by how much of the store has already streamed in.
_PRE_LOAD_HOST_AVAIL = _host_available_bytes()


# On top of the exact non-expert weight bytes: the serving runtime's own device allocations that land
# before KV profiling (loader workspaces, quant repack buffers, allocator fragmentation). Weights are
# exact, so this covers only the runtime. Sized ~2x the runtime need actually observed: configs whose
# estimated reserve happened to sit ~0.2 GB above their true non-expert weights booted reliably.
_NONEXPERT_RUNTIME_RESERVE = 0.5e9

# Aggressive auto-K (SGLANG_PAGED_AGGRESSIVE_K=1). The two concurrency-scaled safety margins — the
# mem_fraction activation headroom (~10% of VRAM) and the non-expert runtime padding above — are
# over-provisioned by ~a GB, capping K well below the physical ceiling. Aggressive mode reclaims them: a
# MEASURED activation reserve (below) replaces the percentage, and the non-expert padding shrinks to a small
# real-workspace slack. The KV reserve still floors the pool to one prefill chunk (+ window).
#
# Batch: chunked prefill caps the PREFILL activation peak at chunked_prefill_size regardless of batch (chunks
# are scheduled, not run as one forward), so activations barely scale with bs — verified stable at bs=2 with
# the bs=1 reserve. What does scale with bs is decode-time logits/workspaces + the captured-graph pool, so the
# reserve carries a small per-running-request term. base + per_req*mrr; bs=1 == 0.5 GB (the tuned value).
# Higher mrr also grows the KV reserve, which squeezes K on its own — so aggressive at high concurrency
# self-limits. Validated for modest bs (≤~4 on this card); the linear term errs toward safety beyond that.
_AGGRESSIVE_ACT_BASE = 0.4e9  # bs-independent: prefill-chunk activation + fragmentation slack
_AGGRESSIVE_ACT_PER_REQ = 0.1e9  # per running request: decode logits/workspaces + captured-graph pool
_AGGRESSIVE_WORKSPACE_SLACK = 0.2e9  # real loader/quant workspace kept on top of the EXACT non-expert bytes


def _nonexpert_weight_bytes_from_checkpoint(model_path: str) -> Optional[int]:
    """EXACT non-expert weight bytes, summed from the checkpoint's safetensors headers (8-byte length
    prefix + JSON; ``data_offsets`` give exact per-tensor sizes — no tensor data is read). Routed
    experts are the ``.experts.`` tensors; ``.shared_experts.`` deliberately does not match — shared
    experts stay resident, so they count as non-expert. Returns ``None`` when the checkpoint isn't a
    locally available safetensors layout (caller falls back to the config estimate)."""
    import glob
    import json
    import struct

    folder = model_path
    if not os.path.isdir(folder):
        try:
            from huggingface_hub import snapshot_download

            folder = snapshot_download(
                model_path, local_files_only=True, allow_patterns=["*.safetensors*"]
            )
        except Exception:
            return None
    files = sorted(glob.glob(os.path.join(folder, "*.safetensors")))
    if not files:
        return None
    index = os.path.join(folder, "model.safetensors.index.json")
    if os.path.exists(index):
        # a partially cached checkpoint would silently undercount — require every indexed shard
        try:
            with open(index) as f:
                need = set(json.load(f)["weight_map"].values())
            if not need.issubset({os.path.basename(p) for p in files}):
                return None
        except Exception:
            return None
    total = 0
    try:
        for path in files:
            with open(path, "rb") as f:
                (hdr_len,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(hdr_len))
            for name, entry in header.items():
                if name == "__metadata__" or ".experts." in name:
                    continue
                begin, end = entry["data_offsets"]
                total += end - begin
    except Exception:
        return None
    return total


def _quant_source(mc, htc):
    """The config object carrying ``quantization_config``. VL checkpoints (e.g. Qwen3.5/3.6 MoE) put it
    on the TOP-level config while ``hf_text_config`` is the nested text config — reading only the text
    config would silently treat a quantized model as bf16 (wrong bits -> halved auto-K, wrong guard).
    """
    if getattr(htc, "quantization_config", None) is not None:
        return htc
    return mc.hf_config


@functools.lru_cache(maxsize=None)
def _moe_geometry():
    """Model geometry shared by the K and W resolvers: ``(ModelConfig, hf_text_config, moe_layers,
    per_expert_layer_bytes)``. Cached — the resolvers run once per FusedMoE layer and the config is
    identical for all of them (both resolvers read import-time memory snapshots, so they are
    deterministic per process)."""
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import get_global_server_args

    mc = ModelConfig.from_server_args(get_global_server_args())
    htc = mc.hf_text_config
    moe_layers = mc.num_hidden_layers - (getattr(htc, "first_k_dense_replace", 0) or 0)
    # per-expert-per-layer bytes, estimated from config (gate+up+down at the quant bit-width, +~3% scales/zeros)
    bits = 16
    qc = getattr(_quant_source(mc, htc), "quantization_config", None)
    if isinstance(qc, dict):
        qm = (qc.get("quant_method") or "").lower()
        fmt = (qc.get("format") or "").lower()
        if qm == "fp8":
            bits = (
                8  # fp8 configs carry no "bits" key; block scales ride in the 3% margin
            )
        elif qm == "compressed-tensors" and "nvfp4" in fmt:
            # 4-bit packed weights + one fp8 block scale per 16 weights (8/16 = 0.5 bit-equiv);
            # tiny per-expert global scalars are negligible. ~4.5 effective bits/weight.
            bits = 4.5
        else:
            bits = qc.get("bits") or qc.get("weights", {}).get("num_bits") or 16
    per_el = 3 * htc.moe_intermediate_size * htc.hidden_size * (bits / 8.0) * 1.03
    return mc, htc, moe_layers, per_el


@functools.lru_cache(maxsize=None)
def resolve_num_resident_experts(
    num_experts_E: int,
    *,
    nonexpert_reserve_gb: float = 2.5,
) -> int:
    """Resolve K when the method is built, from sglang's OWN already-derived config (no CLI re-parse):
    read ``mem_fraction_static`` / ``context_length`` / ``kv_cache_dtype`` off
    ``get_global_server_args()`` and the arch off ``ModelConfig``, then call the pure sizing formula.
    Reading the SAME mem_fraction the server runs at keeps K and the KV pool coherent by construction.
    Cached: every MoE layer resolves the same K.
    """
    from sglang.srt.server_args import get_global_server_args

    sa = get_global_server_args()
    mc, htc, moe_layers, per_el = _moe_geometry()
    layers = mc.num_hidden_layers

    # KV headroom to reserve when sizing K — the K-slot pool and the KV pool compete for the SAME VRAM,
    # so the reserve is the KV/K split knob. Reserve for the ACTUALLY-declared concurrency
    # (``max_running_requests`` x context): low concurrency (single-stream) reserves little -> K claims
    # the surplus (fewer expert page-ins, the single-stream win); high concurrency reserves more -> the
    # KV pool rightly stays large. ``compute_num_resident_experts`` clamps the reserve to what's left
    # after a top_k pool, so a pathologically high max_running_requests floors K to top_k instead of
    # going negative. --paged-experts-kv-reserve-gb overrides with an explicit total KV budget.
    kv_elt = 1 if "fp8" in (sa.kv_cache_dtype or "").lower() else 2
    ctx = sa.context_length or getattr(mc, "context_len", None) or 2048

    # --- Effective K/KV budget, computed ONCE up front so the window fits-check and the K sizing agree.
    # (A budget mismatch flips the fits decision at the boundary — measured: it windowed at 32K when full-KV
    # was the better layout.) free VRAM, non-expert weights, and the aggressive-vs-conservative reserve all
    # feed both. ``k_budget`` is what the expert pool + KV/window share; it mirrors what
    # ``compute_num_resident_experts`` derives internally from the same inputs.
    free = _PRE_LOAD_FREE_BYTES or torch.cuda.mem_get_info()[0]
    top_k = getattr(htc, "num_experts_per_tok", 8) or 8
    mem_frac = sa.mem_fraction_static or 0.85
    mrr = max(1, int(getattr(sa, "max_running_requests", 1) or 1))

    # Non-expert weights: exact from the checkpoint safetensors headers when available (+ runtime reserve),
    # else a config estimate floored at the default. (Moved ahead of the window decision so the budget is
    # exact there too — the fits boundary is budget-sensitive.)
    exact = _nonexpert_weight_bytes_from_checkpoint(sa.model_path)
    if exact is not None:
        nonexpert_bytes = exact + _NONEXPERT_RUNTIME_RESERVE
    else:
        vocab = getattr(htc, "vocab_size", 0) or 0
        tied = bool(getattr(htc, "tie_word_embeddings", False))
        embed_bytes = vocab * htc.hidden_size * 2 * (1 if tied else 2)
        _NONEXPERT_BASE = 2.0e9
        if getattr(mc.hf_config, "vision_config", None) is not None:
            embed_bytes += int(1.0e9)  # VL checkpoints load a vision tower + projector
        nonexpert_bytes = max(nonexpert_reserve_gb * 1e9, _NONEXPERT_BASE + embed_bytes)

    # Aggressive reclaim (opt-in): replace the mem_fraction activation headroom with a measured, batch-scaled
    # reserve and shrink the non-expert padding to a small workspace slack (see _AGGRESSIVE_* above). Works at
    # bs>1 — the reserve carries a per-running-request term.
    aggressive = bool(os.environ.get("SGLANG_PAGED_AGGRESSIVE_K"))
    act_reserve = None
    if aggressive:
        if exact is not None:
            nonexpert_bytes = exact + _AGGRESSIVE_WORKSPACE_SLACK
        _act_gb = os.environ.get("SGLANG_PAGED_ACT_RESERVE_GB")
        act_reserve = (
            (float(_act_gb) * 1e9)
            if _act_gb
            else _AGGRESSIVE_ACT_BASE + _AGGRESSIVE_ACT_PER_REQ * mrr
        )

    # The budget the expert pool + KV/window share (before any window ring is charged).
    k_budget = (
        (free - act_reserve - nonexpert_bytes)
        if act_reserve is not None
        else (free * mem_frac - nonexpert_bytes)
    )

    # Auto-window (SGLANG_KV_WINDOW=auto): fits/doesn't-fit policy (see compute_window_size). Whenever the
    # full-context KV fits VRAM alongside a minimal expert pool, full-KV beats windowing at EVERY decode
    # position (resident HBM KV vs re-streamed PCIe tail — measured), so W=0 and aggressive-K takes the VRAM.
    # Window only when full-KV is infeasible (context too large to hold resident): capacity fallback. No
    # fitted thresholds, no bandwidth constant — machine-independent. Resolves to a concrete int written back
    # into the env so every downstream consumer (attention, allocator, admission) reads the SAME W.
    if os.environ.get("SGLANG_KV_WINDOW") == "auto":
        _tp = getattr(sa, "tp_size", 1) or 1
        if getattr(mc, "kv_lora_rank", None):
            _pt = (mc.kv_lora_rank + mc.qk_rope_head_dim) * layers * kv_elt
        else:
            _pt = 2 * layers * mc.get_num_kv_heads(_tp) * ((mc.head_dim + mc.v_head_dim) // 2) * kv_elt
        _W = compute_window_size(
            context_length=ctx,
            per_token_bytes=_pt,
            per_expert_pool_bytes=moe_layers * per_el,
            top_k=top_k,
            budget_bytes=k_budget,  # SAME budget K sizing uses -> consistent fits decision
            page_size=int(getattr(sa, "page_size", 1) or 1),
        )
        if _W > 0:
            os.environ["SGLANG_KV_WINDOW"] = str(_W)
            logger.info(
                "[paged-experts] auto window: full-KV infeasible at context=%d -> W=%d (capacity fallback)",
                ctx,
                _W,
            )
        else:
            os.environ.pop("SGLANG_KV_WINDOW", None)  # full-KV fits: no windowing, aggressive-K takes VRAM
            logger.info(
                "[paged-experts] auto window: full-KV fits at context=%d -> W=0 (no windowing)", ctx
            )

    # KV-streaming (SGLANG_KV_WINDOW=W): the windowed device KV pool holds only ~W live tokens + one prefill
    # chunk, regardless of context_length (the cold tail lives in host RAM). So the K/KV split must reserve
    # for THAT device footprint, not the full context — else long context reserves the full-context KV (e.g.
    # ~3.2GB @32K) and starves K to top_k. Capping the reserve length at W + chunked_prefill_size lets auto-K
    # claim the VRAM the window frees, so K need NOT be pinned; sglang then sizes the real KV pool from the
    # matching leftover (~W + chunk), so max-total-tokens need not be pinned either. No effect when
    # W + chunk >= context (short context already reserves little — matches the measured no-gain regime).
    _kvwin = os.environ.get("SGLANG_KV_WINDOW")
    if _kvwin:
        _cps = int(getattr(sa, "chunked_prefill_size", 0) or 2048)
        ctx = min(ctx, int(_kvwin) + _cps)
    kv_gb = getattr(sa, "paged_experts_kv_reserve_gb", -1.0)
    if kv_gb is not None and kv_gb >= 0:
        kv_reserve = kv_gb * 1e9
    elif getattr(mc, "kv_lora_rank", None):  # MLA
        cell = (mc.kv_lora_rank + mc.qk_rope_head_dim) * layers * kv_elt
        kv_reserve = mrr * ctx * cell
    else:  # MHA / GQA — reuse the pure helper (get_num_kv_heads handles GQA + TP)
        tp = getattr(sa, "tp_size", 1) or 1
        kv_reserve = kv_reserve_bytes_mha(
            num_layers=layers,
            num_kv_heads=mc.get_num_kv_heads(tp),
            head_dim=(mc.head_dim + mc.v_head_dim) // 2,  # combined K+V per-head width
            kv_dtype_bytes=kv_elt,
            max_running_requests=mrr,  # size KV for the declared concurrency; single-stream -> small
            context_length=ctx,
        )

    # Pool floor: a tiny leftover KV pool can't hold a prefill chunk — the extend allocator grabs the whole
    # chunk before the windowed ring frees, so a pool below one chunk silently fails prompts longer than it.
    # Reserve at least one chunk (+ window) of KV. Derive per-token from the reserve just computed
    # (branch-agnostic: covers both the MHA per-token and the MLA cell). Skipped when the reserve is pinned.
    min_kv_pool = 0.0
    if not (kv_gb is not None and kv_gb >= 0) and mrr * ctx > 0:
        _per_token = kv_reserve / (mrr * ctx)
        _chunk = int(getattr(sa, "chunked_prefill_size", 0) or 2048)
        _wfloor = int(_kvwin) if _kvwin else 0
        min_kv_pool = (_chunk + _wfloor) * _per_token

    # Pool-floor guarantee: force the device KV pool to hold >= W + chunk regardless of how K is set (pinned
    # or auto). The windowed admission charges each request ~W + chunk; if the pool is smaller the request
    # can never be admitted, head-of-lines the FCFS queue, and the scheduler wedges (measured: a hand-set
    # max-total-tokens=3072 with W=8192 hangs). Raise max_total_tokens to the floor so it boots-loud on a
    # genuine VRAM shortfall instead of silently wedging. Only when windowing.
    if _kvwin:
        _need = (
            int(_kvwin)
            + int(getattr(sa, "chunked_prefill_size", 0) or 2048)
            + int(getattr(sa, "page_size", 1) or 1)
        )
        _cur = getattr(sa, "max_total_tokens", None)
        if _cur is None or int(_cur) < _need:
            sa.max_total_tokens = _need
            logger.info(
                "[paged-experts] windowed pool floor: max_total_tokens -> %d (W=%s + chunk)",
                _need,
                _kvwin,
            )

    # Hybrid (mamba / linear-attention) models keep a per-request STATE cache outside the token-KV
    # pool; sglang sizes it from what is left after weights — which auto-K would otherwise consume
    # down to zero (boot failure: "Not enough GPU memory for hybrid state cache"). Reserve room for a
    # modest slot count using sglang's own per-request figure.
    # The params live on the HF (text) config (e.g. Qwen3NextConfig.mamba2_cache_params); the property
    # asserts an initialized TP group internally, which holds here (model build follows distributed init).
    mamba_per_req = 0
    for _cfg in (htc, mc.hf_config):
        try:
            mamba_per_req = _cfg.mamba2_cache_params.mamba_cache_per_req
            break
        except Exception:
            continue
    if mamba_per_req and not (kv_gb is not None and kv_gb >= 0):
        # An explicit --paged-experts-kv-reserve-gb is the user's TOTAL budget for KV + hybrid state
        # (the documented knob for hybrid concurrency) — don't stack the automatic reserve on top.
        kv_reserve += mamba_per_req * _HYBRID_STATE_SLOTS

    # (non-expert weights, free/top_k/mem_frac, and the aggressive reserve were resolved up front, before
    # the window decision, so the budget is consistent across both.)

    # Window-ring VRAM: the backend store keeps a W-slot device ring (rk/rv) SEPARATE from the KV pool — a
    # fixed device allocation K sizing must subtract, or a big window overshoots K and OOMs at load.
    # Negligible at small W (0.05 GB at W=512); ~1 GB at W=10k. Charge it as non-expert device memory.
    if _kvwin and mrr * ctx > 0:
        nonexpert_bytes += int(_kvwin) * (kv_reserve / (mrr * ctx))

    k = compute_num_resident_experts(
        free_vram_bytes=free,
        mem_fraction=mem_frac,
        nonexpert_bytes=nonexpert_bytes,
        kv_reserve_bytes=kv_reserve,
        moe_layers=moe_layers,
        per_expert_layer_bytes=per_el,
        top_k=top_k,
        num_experts=num_experts_E,
        activation_reserve_bytes=act_reserve,
        min_kv_pool_bytes=min_kv_pool,
    )
    logger.info(
        "[paged-experts] resident K=%d/%d (%d%%): free=%.2fGB mem_fraction=%.3f "
        "nonexpert=%.2fGB(%s) KV_reserve=%.2fGB(floor=%.2fGB) per_expert=%.2fMB moe_layers=%d%s",
        k,
        num_experts_E,
        k * 100 // num_experts_E,
        free / 1e9,
        mem_frac,
        nonexpert_bytes / 1e9,
        "exact+reserve" if exact is not None else "estimated",
        kv_reserve / 1e9,
        min_kv_pool / 1e9,
        per_el / 1e6,
        moe_layers,
        (f" AGGRESSIVE act_reserve={act_reserve / 1e9:.2f}GB" if aggressive else ""),
    )
    return k


def reset_sizing_state() -> None:
    """Clear the cached sizing state for a NEW model build in this process: the geometry/K/W caches
    (keyed only on E — a different model could alias the same key) and the pre-load memory snapshots
    (free VRAM / MemAvailable move between builds)."""
    global _PRE_LOAD_FREE_BYTES, _PRE_LOAD_HOST_AVAIL
    _moe_geometry.cache_clear()
    resolve_num_resident_experts.cache_clear()
    resolve_window_experts.cache_clear()
    # the repack-cache digest folds in shard mtimes — recompute after a reload (updated checkpoint)
    from sglang.srt.layers.moe.paged_experts.pager import _store_cache_dir

    _store_cache_dir.cache_clear()
    try:
        _PRE_LOAD_FREE_BYTES = torch.cuda.mem_get_info()[0]
    except Exception:
        _PRE_LOAD_FREE_BYTES = 0
    _PRE_LOAD_HOST_AVAIL = _host_available_bytes()


def _make_method_class():
    """Import the base lazily so this module can be imported before the rest of sglang's MoE stack."""
    from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase

    class PagedExpertsMoEMethod(FusedMoEMethodBase):
        def __init__(
            self,
            base_method,
            num_experts_E: int,
            num_resident_K: int,
            pin_host: bool = True,
            use_ondevice: bool = False,
            eviction: str = "lru",
            window: Optional[int] = 0,
            cold_backing: str = "ram",
            cold_dir: Optional[str] = None,
            breakable_decode: bool = False,
        ):
            self.base_method = base_method
            self.E = num_experts_E
            self.num_resident = num_resident_K
            self.pin_host = pin_host
            self.eviction = eviction
            # Pinned-window fallback: 0 = full pin (every expert page-locked); 0 < window < E pins only the
            # W hot experts and keeps the E-W cold tail pageable, for stores past the page-lock ceiling.
            # ``None`` = deferred: resolved (with the page-lock ceiling probe) in
            # process_weights_after_loading, after the weight loader's own pinned use has settled.
            self.window = window
            # Windowed cold-tier backing: "ram" (pageable, must fit RAM) | "disk" (mmap'd file, P4 — lets
            # the store exceed RAM). cold_dir is the disk location for the "disk" tier.
            self.cold_backing = cold_backing
            self.cold_dir = cold_dir
            # Decode placement: captured (on-device decide + UVA gather, needs a pinned store) when CUDA
            # graphs are on, else eager host; the captured variant is windowed (replay-twice) when a window
            # is set. The bool + window resolve to a Placement strategy (placement.py) — deferred alongside
            # a deferred window.
            self.use_ondevice = use_ondevice and pin_host
            self._breakable_decode = breakable_decode
            self._placement = (
                make_placement(
                    self.use_ondevice,
                    windowed=window > 0,
                    breakable_decode=breakable_decode,
                )
                if window is not None
                else None
            )
            self._pager = None

        def create_weights(
            self,
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra,
        ):
            # K-slot table. Weight loading uses FusedMoE's NATIVE expert-parallel remap: the loader
            # fills slots 0..K-1 and skips the rest (they are re-read from the checkpoint into the host
            # store by the fill). Our forward does its OWN routing remap, so K-local only affects load.
            # The loader's global->local skip keys off _num_local_routed (see FusedMoE.weight_loader /
            # _map_global_expert_id_to_local_expert_id), NOT num_local_experts, so shrink BOTH — else
            # loaders that go through the physical/EP path (e.g. the DeepSeek/Mistral family) index the
            # full-E global id into the K-slot param and IndexError.
            layer.num_local_experts = self.num_resident
            if getattr(layer, "_num_local_routed", None) is not None:
                layer._num_local_routed = self.num_resident
                layer._num_global_routed = self.num_resident
            self.base_method.create_weights(
                layer=layer,
                num_experts=self.num_resident,
                hidden_size=hidden_size,
                intermediate_size_per_partition=intermediate_size_per_partition,
                params_dtype=params_dtype,
                **extra,
            )

        def create_moe_runner(self, layer, moe_runner_config):
            from dataclasses import replace

            # The runner must size its expert loop to K, not the model's E local experts, else the
            # fused-MoE kernel indexes past the K slots. routed_scaling_factor is applied externally by
            # the model (deepseek_v2) -> strip it here to avoid double-scaling.
            cfg = replace(moe_runner_config, num_local_experts=self.num_resident)
            cfg = replace(cfg, routed_scaling_factor=None)
            self.base_method.create_moe_runner(layer, cfg)
            self.moe_runner_config = getattr(self.base_method, "moe_runner_config", cfg)

        def process_weights_after_loading(self, layer):
            if hasattr(self.base_method, "process_weights_after_loading"):
                # Base PWALs that build expert-count-sized structures from layer.num_experts (e.g. the
                # nvfp4 cutlass path's CutlassMoEParams) must see the K-slot count — the weights they run
                # over are the K-slot pool, not the model's E. create_weights/create_moe_runner already
                # got K; num_experts is the one attribute still at E. Swap it for the base call only.
                saved_ne = getattr(layer, "num_experts", None)
                if saved_ne is not None:
                    layer.num_experts = self.num_resident
                try:
                    self.base_method.process_weights_after_loading(layer)
                finally:
                    if saved_ne is not None:
                        layer.num_experts = saved_ne
            if self.window is None:
                # Deferred window sizing: the ceiling probe runs here, after the loader — cached, so the
                # first layer resolves it and every layer shares the same W.
                self.window = resolve_window_experts(self.E)
                self._placement = make_placement(
                    self.use_ondevice,
                    windowed=self.window > 0,
                    breakable_decode=self._breakable_decode,
                )
                if self.use_ondevice:
                    _shape_capture_bs_to_keep_warm(
                        self.num_resident, windowed=self.window > 0
                    )
            from sglang.srt.layers.moe.paged_experts.pager import setup_pager

            self._pager = setup_pager(self, layer)

        def apply(self, layer, dispatch_output):
            from sglang.srt.layers.moe.paged_experts.forward import paged_apply

            return paged_apply(self, layer, dispatch_output)

    return PagedExpertsMoEMethod


def resolve_breakable_decode(server_args: Any) -> bool:
    """True when the decode-phase CUDA-graph backend is the breakable one: the convenience flag
    ``--cuda-graph-backend-decode breakable``, else the ``cuda_graph_config`` decode phase.
    """
    bd = getattr(server_args, "cuda_graph_backend_decode", None)
    if bd is None:
        cfg = getattr(server_args, "cuda_graph_config", None)
        bd = getattr(getattr(cfg, "decode", None), "backend", None) if cfg else None
    return bd == "breakable"


# id() of the ServerArgs the current model build is for. A new engine in the same process brings a new
# ServerArgs object; detecting the change lets the first wrapped layer drop the previous model's cached
# sizing state and module-global pager state (registered pagers over freed tensors, spent horizon).
_BUILT_FOR_SA: Optional[int] = None


def make_for_layer(
    layer,
    base_method,
    server_args: Any,
    *,
    num_resident: Any = "auto",
    nonexpert_reserve_gb: float = 2.5,
) -> Any:
    """Factory invoked from the FusedMoE init hook when paged experts is enabled: enforce the
    compatibility guards, resolve K, and wrap ``base_method``. ``num_resident`` is ``"auto"`` or an int.
    The store choice comes from ``--paged-experts-store`` (pinned -> True, paged -> False) and the
    eviction policy from ``--paged-experts-eviction`` (lru | lfu).
    """
    global _BUILT_FOR_SA
    if _BUILT_FOR_SA != id(server_args):
        _BUILT_FOR_SA = id(server_args)
        reset_sizing_state()
        from sglang.srt.layers.moe.paged_experts.pager import reset_paged_experts_state

        reset_paged_experts_state()
    check_paged_experts_compat(server_args)
    _geo_mc, _geo_htc = _moe_geometry()[:2]
    check_paged_experts_quant(_quant_source(_geo_mc, _geo_htc))
    E = int(getattr(layer, "num_local_experts", None) or layer.num_experts)
    if num_resident == "auto":
        K = resolve_num_resident_experts(E, nonexpert_reserve_gb=nonexpert_reserve_gb)
    else:
        K = int(num_resident)
    pin_host = getattr(server_args, "paged_experts_store", "pinned") != "paged"
    # Use the on-device (capturable) decode path unless CUDA graphs are disabled. With graphs off it's the
    # eager kernel-free path (host decide + transfer_kv); with graphs on the decode step is captured.
    use_ondevice = not bool(getattr(server_args, "disable_cuda_graph", False))
    if use_ondevice and not pin_host:
        # The pageable store has no capture-safe gather: it would select the eager placement, whose
        # host-side decision syncs inside graph capture and fails cryptically at startup. Reject up front.
        raise RuntimeError(
            "Paged Experts: --paged-experts-store paged requires --disable-cuda-graph (the pageable "
            "store pages via a host-driven copy that cannot run inside a captured decode graph)."
        )
    eviction = getattr(server_args, "paged_experts_eviction", "lru")
    # The pinned window is sized automatically (the largest window that fits — page-locking every expert
    # when the whole store fits); there is no user knob. A pageable store pins nothing, so it has no window.
    # ``None`` defers resolution to process_weights_after_loading: the page-lock ceiling probe must run
    # AFTER the weight loader, whose allocations count against the same OS pin budget (WSL2) — a pre-load
    # probe measures headroom the loaded server no longer has.
    window = None if bool(pin_host) else 0
    cold_backing = getattr(server_args, "paged_experts_cold_backing", "ram")
    cold_dir = getattr(server_args, "paged_experts_cold_dir", "") or None
    # Windowed decode under the breakable backend -> BCG break-and-page-in (no replay-twice).
    breakable_decode = resolve_breakable_decode(server_args)
    return _make_method_class()(
        base_method,
        E,
        K,
        pin_host=bool(pin_host),
        use_ondevice=use_ondevice,
        eviction=eviction,
        window=window,
        cold_backing=cold_backing,
        cold_dir=cold_dir,
        breakable_decode=breakable_decode,
    )


# Fraction of MemAvailable the auto window is allowed to page-lock. Deliberately conservative: the rest
# of the process needs pageable RAM (a RAM cold tier holds E-W experts; a disk cold tier needs page cache;
# plus activations and general overhead), and on WSL the page-lock pool is itself capped below RAM.
_AUTO_WINDOW_HOST_FRACTION = 0.5
# With a DISK cold tier the cold tail lives in the page cache (clean pages evict under pressure), not in
# pageable RAM, so the window may claim a larger share — the leftover only needs to cover page cache for
# the cold working set + process overhead. The pin-ceiling probe still bounds the result.
_AUTO_WINDOW_HOST_FRACTION_DISK = 0.75
# Probe-ladder floor on pin-capped platforms, as a fraction of total VRAM: the ladder never probes below
# this, so sizing is never more conservative than a known-safe bound even if every probe rung fails.
_AUTO_WINDOW_VRAM_FRACTION = 0.9
# Probe-ladder decay: each failed rung retries at this fraction of the previous attempt.
_PIN_PROBE_DECAY = 0.85
# Headroom left un-pinned on top of the window: post-build pinned consumers (per-layer staging buffers,
# the BCG doorbell, token-transfer buffers) draw from the same OS pin budget.
_PIN_PROBE_HEADROOM = 1.0e9
# Pageable RAM kept free when greedily full-pinning the whole store (no cold tail to house — this covers
# the serving process itself: tokenizer, host-side runtime, page cache for logs/checkpoint reads).
_FULL_PIN_RAM_RESERVE = 8.0e9


def _pin_is_capped() -> bool:
    """True where the OS caps total page-locked memory below host RAM — notably WSL2, whose dxgkrnl
    GPU-paravirt layer bounds pinned allocations well below ``MemAvailable`` (the bound varies by box and
    driver: measured ~1.7x board memory on an 8 GB-VRAM laptop, ~2x on a 16 GB desktop). On native Linux
    pinning is host-only, so this is False and the window is bounded by host RAM alone.
    """
    try:
        with open("/proc/version") as f:
            v = f.read().lower()
        return "microsoft" in v or "wsl" in v
    except Exception:
        return False


def _shape_capture_bs_to_keep_warm(K: int, windowed: bool) -> None:
    """Shape the decode capture batch list around the keep-warm bound ``bs*top_k <= K``, known the
    moment K and the window resolve (post-load, before graph capture).

    Windowed stores: HARD clamp to ``K//top_k`` regardless of user settings — the distinct>K fallback
    is a host-driven wave, uncapturable, and capture at such sizes used to fail at startup.

    Full-pin stores: captured waves past the bound work but cost ``ceil(E/K)`` GEMMs per layer — a
    measured 2.7x throughput cliff right above the bound (int4-30B: 373 tok/s captured at bs=8 vs 139
    at bs=16). When the user did NOT set an explicit capture list, shape the default: clamp to the
    bound and make sure the bound itself is a capture bucket (bs up to ``K//top_k`` then pads to a
    captured graph instead of falling off). An explicit ``--cuda-graph-max-bs-decode`` /
    ``--cuda-graph-bs-decode`` is respected (the one-time wave-cliff warning still fires).
    """
    from sglang.srt.server_args import get_global_server_args

    sa = get_global_server_args()
    _, htc, _, _ = _moe_geometry()
    top_k = getattr(htc, "num_experts_per_tok", 8) or 8
    cap = max(1, K // top_k)
    user_set = (
        getattr(sa, "cuda_graph_max_bs_decode", None) is not None
        or getattr(sa, "cuda_graph_bs_decode", None) is not None
    )
    if not windowed and user_set:
        return  # full-pin + explicit setting: the user chose their operating point
    try:
        decode_cfg = sa.cuda_graph_config.decode
        old_bs = list(decode_cfg.bs)
        new_bs = [b for b in old_bs if b <= cap]
        if not windowed and cap not in new_bs:
            new_bs.append(
                cap
            )  # capture the bound itself: bs up to K//top_k stays captured
        new_bs = sorted(set(new_bs)) or [cap]
        if new_bs != old_bs:
            decode_cfg.bs = new_bs
            logger.info(
                "[paged-experts] keep-warm bound (bs*top_k <= K=%d): decode capture batch sizes "
                "-> %s (cap K//top_k=%d); larger batches serve through the %s path",
                K,
                new_bs,
                cap,
                "uncaptured wave" if windowed else "uncaptured",
            )
    except Exception as e:
        logger.warning(
            "[paged-experts] could not shape decode capture batch sizes (%s); the keep-warm bound "
            "K=%d applies at runtime",
            e,
            K,
        )


def _cudart_handle():
    """ctypes handle to the CUDA runtime torch itself loaded (bundled libcudart), for raw
    ``cudaHostAlloc``/``cudaFreeHost`` — torch's pinned allocator caches freed blocks, which would leave a
    probe-sized block resident and double the pin footprint when the real store allocates.
    """
    import ctypes
    import glob

    for cand in glob.glob(
        os.path.join(os.path.dirname(torch.__file__), "lib", "libcudart*so*")
    ) + ["libcudart.so"]:
        try:
            return ctypes.CDLL(cand)
        except OSError:
            continue
    return None


def _pin_ceiling_cache_path() -> str:
    root = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return os.path.join(root, "paged_experts_cache", "pin_ceiling.json")


def _pin_ceiling_cache_key() -> str:
    """The page-lock ceiling is a property of the box + OS build (not of the model or current load):
    key on MemTotal and the kernel version string so a RAM change or WSL update re-measures.
    """
    import hashlib

    memtotal = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    memtotal = int(line.split()[1])
                    break
    except Exception:
        pass
    try:
        with open("/proc/version") as f:
            ver = f.read().strip()
    except Exception:
        ver = "?"
    return f"{memtotal}:{hashlib.sha256(ver.encode()).hexdigest()[:12]}"


def _pin_ceiling_cache_load() -> dict:
    import json

    try:
        with open(_pin_ceiling_cache_path()) as f:
            return json.load(f).get(_pin_ceiling_cache_key(), {})
    except Exception:
        return {}


def _pin_ceiling_cache_store(
    ok: Optional[int] = None, fail: Optional[int] = None
) -> None:
    import json

    path = _pin_ceiling_cache_path()
    key = _pin_ceiling_cache_key()
    try:
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            data = {}
        ent = data.get(key, {})
        if ok is not None:
            ent["ok"] = max(int(ent.get("ok", 0)), int(ok))
        if fail is not None:
            ent["fail"] = min(int(ent.get("fail", 1 << 62)), int(fail))
        data[key] = ent
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except Exception:
        pass  # cache is an optimization; never fatal


def _pin_ceiling_cache_reset() -> None:
    """Drop this box's cached ceiling (called when a real pinned allocation fails despite the cache —
    the box state changed; next boot re-measures)."""
    import json

    path = _pin_ceiling_cache_path()
    try:
        with open(path) as f:
            data = json.load(f)
        data.pop(_pin_ceiling_cache_key(), None)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except Exception:
        pass


def _probe_pin_ceiling(target_bytes: int, floor_bytes: int) -> int:
    """Largest page-lockable size <= ``target_bytes``, measured by real ``cudaHostAlloc`` attempts (freed
    immediately) on a descending ladder down to ``floor_bytes``. On pin-capped platforms (WSL2) the true
    ceiling varies by box and driver, and guessing it low forces windowed mode on boxes that could full-pin
    the store (~2x decode cost), so measure instead of guessing. The common case — target fits — costs one
    transient pin pass at startup; a failed rung costs the time the allocator spends before hitting the
    ceiling. Falls back to ``floor_bytes`` (the pre-probe conservative cap) on any failure, so sizing is
    never worse than the guess it replaces."""
    import ctypes
    import math

    rt = _cudart_handle()
    if rt is None:
        return floor_bytes
    # ceil, not truncate: the store size is fractional (per-expert estimate), and losing even one byte
    # here sizes the window to E-1 instead of a full pin
    size = int(math.ceil(target_bytes))
    floor = int(floor_bytes)
    # The ceiling is a box property — a cached measurement skips the probe entirely (a 30 GB pin pass
    # costs ~30-60 s per boot) when a previous boot already verified at least this much, and starts the
    # ladder below a size known to fail.
    cached = _pin_ceiling_cache_load()
    if size <= int(cached.get("ok", 0)):
        logger.info(
            "[paged-experts] pin probe: %.2fGB covered by cached ceiling (>=%.2fGB verified) — skipped",
            size / 1e9,
            cached["ok"] / 1e9,
        )
        return size
    known_fail = int(cached.get("fail", 0))
    if known_fail and size >= known_fail:
        size = min(size, int(known_fail * _PIN_PROBE_DECAY))
    while size > floor:
        ptr = ctypes.c_void_p()
        try:
            rc = rt.cudaHostAlloc(ctypes.byref(ptr), ctypes.c_size_t(size), 0)
        except Exception:
            return floor
        logger.info(
            "[paged-experts] pin probe: cudaHostAlloc(%.2fGB) -> rc=%d", size / 1e9, rc
        )
        if rc == 0:
            rt.cudaFreeHost(ptr)
            _pin_ceiling_cache_store(ok=size)
            return size
        _pin_ceiling_cache_store(fail=size)
        size = int(size * _PIN_PROBE_DECAY)
    logger.info("[paged-experts] pin probe: fell through to floor %.2fGB", floor / 1e9)
    return floor


@functools.lru_cache(maxsize=None)
def resolve_window_experts(num_experts_E: int) -> int:
    """Size the pinned window ``W`` for the pinned store: page-lock the largest hot window that fits the pin
    budget, and ``0`` (full pin) when the whole store fits. The budget is
    ``MemAvailable * _AUTO_WINDOW_HOST_FRACTION``; where the OS caps page-locking below host RAM (WSL2, see
    ``_pin_is_capped``) the budget is verified against the *measured* pin ceiling (``_probe_pin_ceiling``)
    rather than a guessed fraction of VRAM. Mirrors ``resolve_num_resident_experts`` — reads geometry off
    ``ModelConfig`` and calls the pure ``compute_window_experts``. Sizing is automatic; there is no
    user-facing window knob. Cached: every MoE layer resolves the same W.
    """
    from sglang.srt.server_args import get_global_server_args

    _, _, moe_layers, per_el = _moe_geometry()

    cold_backing = getattr(
        get_global_server_args(), "paged_experts_cold_backing", "ram"
    )
    frac = (
        _AUTO_WINDOW_HOST_FRACTION_DISK
        if cold_backing == "disk"
        else _AUTO_WINDOW_HOST_FRACTION
    )
    avail = _PRE_LOAD_HOST_AVAIL or _host_available_bytes()
    frac_budget = avail * frac
    budget = frac_budget
    store_bytes = per_el * moe_layers * num_experts_E
    # GREEDY FULL PIN: the host fraction exists to reserve pageable RAM for the cold tail — a fully
    # pinned store has none, so when the whole store fits MemAvailable minus a process reserve, grow
    # the budget to the store and let the ceiling probe verify it (floored at the fraction, so sizing
    # is never worse than before). Converts mid-size "windowed" stores (e.g. a 30 GB fp8 store on a
    # 50 GB box) to full pin: streaming prefill applies and the cold tier disappears from decode.
    if frac_budget < store_bytes <= avail - _FULL_PIN_RAM_RESERVE:
        budget = float(store_bytes)
    probed = False
    if _pin_is_capped() or budget > frac_budget:
        if _pin_is_capped():
            try:
                total_vram = torch.cuda.mem_get_info()[1]
            except Exception:
                total_vram = 0
            floor = min(frac_budget, total_vram * _AUTO_WINDOW_VRAM_FRACTION)
        else:
            floor = frac_budget  # native Linux: never end below today's fraction
        # never need to pin more than the store itself (+ headroom for post-build pinned consumers)
        target = min(budget, store_bytes)
        if target > floor:
            got = _probe_pin_ceiling(target + _PIN_PROBE_HEADROOM, floor)
            budget, probed = max(floor, got - _PIN_PROBE_HEADROOM), True
        else:
            budget = target
    w = compute_window_experts(
        pin_budget_bytes=budget,
        moe_layers=moe_layers,
        per_expert_layer_bytes=per_el,
        num_experts=num_experts_E,
    )
    logger.info(
        "[paged-experts] auto window W=%d/%d (%s): host_avail=%.1fGB budget=%.1fGB%s "
        "per_expert=%.2fMB moe_layers=%d",
        w,
        num_experts_E,
        "full pin — whole store fits" if w == 0 else "windowed",
        avail / 1e9,
        budget / 1e9,
        " (pin-probed)" if probed else "",
        per_el / 1e6,
        moe_layers,
    )
    return w
