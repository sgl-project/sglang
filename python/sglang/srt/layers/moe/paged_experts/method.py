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
        if (qc.get("quant_method") or "").lower() == "fp8":
            bits = (
                8  # fp8 configs carry no "bits" key; block scales ride in the 3% margin
            )
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

    # KV headroom to reserve when sizing K. The K-slot pool is FIXED (it does not grow with concurrency);
    # sglang sizes the real KV pool from the post-weights leftover and derives max_running_requests from
    # THAT. So reserving the worst case (max_running_requests x full context) here double-counts and
    # starves K — a footgun on the constrained cards Paged Experts targets (a high --max-running-requests
    # silently floored K to top_k). Reserve a SINGLE-STREAM context by default; sglang's actual KV pool
    # (the leftover) then supports real concurrency. --paged-experts-kv-reserve-gb overrides to reserve a
    # larger guaranteed KV pool (smaller K). sizing.compute_num_resident_experts clamps it to physical.
    kv_elt = 1 if "fp8" in (sa.kv_cache_dtype or "").lower() else 2
    ctx = sa.context_length or getattr(mc, "context_len", None) or 2048
    kv_gb = getattr(sa, "paged_experts_kv_reserve_gb", -1.0)
    if kv_gb is not None and kv_gb >= 0:
        kv_reserve = kv_gb * 1e9
    elif getattr(mc, "kv_lora_rank", None):  # MLA
        cell = (mc.kv_lora_rank + mc.qk_rope_head_dim) * layers * kv_elt
        kv_reserve = ctx * cell  # single-stream
    else:  # MHA / GQA — reuse the pure helper (get_num_kv_heads handles GQA + TP)
        tp = getattr(sa, "tp_size", 1) or 1
        kv_reserve = kv_reserve_bytes_mha(
            num_layers=layers,
            num_kv_heads=mc.get_num_kv_heads(tp),
            head_dim=(mc.head_dim + mc.v_head_dim) // 2,  # combined K+V per-head width
            kv_dtype_bytes=kv_elt,
            max_running_requests=1,  # single-stream headroom, NOT worst-case concurrency
            context_length=ctx,
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

    # Non-expert weights: the fixed default underestimates big-vocab / VL checkpoints (a 248k untied
    # vocab is ~2 GB of embeddings + lm_head alone). Estimate the dominant variable term from config,
    # floored at the passed default so smaller models keep their K.
    vocab = getattr(htc, "vocab_size", 0) or 0
    tied = bool(getattr(htc, "tie_word_embeddings", False))
    embed_bytes = vocab * htc.hidden_size * 2 * (1 if tied else 2)
    # 2.0 GB base: attention/dense weights + the serving runtime's own allocations (workspaces,
    # capture pools) — sized so auto-K boots at default --mem-fraction-static across models.
    _NONEXPERT_BASE = 2.0e9
    if getattr(mc.hf_config, "vision_config", None) is not None:
        embed_bytes += int(
            1.0e9
        )  # VL checkpoints load a vision tower + projector alongside the text model
    nonexpert_bytes = max(nonexpert_reserve_gb * 1e9, _NONEXPERT_BASE + embed_bytes)

    free = _PRE_LOAD_FREE_BYTES or torch.cuda.mem_get_info()[0]
    top_k = getattr(htc, "num_experts_per_tok", 8) or 8
    mem_frac = sa.mem_fraction_static or 0.85
    k = compute_num_resident_experts(
        free_vram_bytes=free,
        mem_fraction=mem_frac,
        nonexpert_bytes=nonexpert_bytes,
        kv_reserve_bytes=kv_reserve,
        moe_layers=moe_layers,
        per_expert_layer_bytes=per_el,
        top_k=top_k,
        num_experts=num_experts_E,
    )
    logger.info(
        "[paged-experts] resident K=%d/%d (%d%%): free=%.2fGB mem_fraction=%.3f "
        "KV_reserve=%.2fGB per_expert=%.2fMB moe_layers=%d",
        k,
        num_experts_E,
        k * 100 // num_experts_E,
        free / 1e9,
        mem_frac,
        kv_reserve / 1e9,
        per_el / 1e6,
        moe_layers,
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
            window: int = 0,
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
            self.window = window
            # Windowed cold-tier backing: "ram" (pageable, must fit RAM) | "disk" (mmap'd file, P4 — lets
            # the store exceed RAM). cold_dir is the disk location for the "disk" tier.
            self.cold_backing = cold_backing
            self.cold_dir = cold_dir
            # Decode placement: captured (on-device decide + UVA gather, needs a pinned store) when CUDA
            # graphs are on, else eager host; the captured variant is windowed (replay-twice) when a window
            # is set. The bool + window resolve to a Placement strategy (placement.py).
            self.use_ondevice = use_ondevice and pin_host
            self._placement = make_placement(
                self.use_ondevice,
                windowed=window > 0,
                breakable_decode=breakable_decode,
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
            # K-slot table. Weight loading uses FusedMoE's NATIVE expert-parallel remap: num_local_experts
            # = K -> the default loader fills slots 0..K-1 and skips the rest (no custom loader). Our
            # forward does its OWN routing remap, so K-local only affects load.
            layer.num_local_experts = self.num_resident
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
                self.base_method.process_weights_after_loading(layer)
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
    window = resolve_window_experts(E) if bool(pin_host) else 0
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
# On platforms where page-locking is VRAM-coupled (see ``_pin_is_vram_coupled``), the pinned window also
# consumes a GPU-accessible aperture ~ board memory, so cap the budget by this fraction of total VRAM.
_AUTO_WINDOW_VRAM_FRACTION = 0.9


def _pin_is_vram_coupled() -> bool:
    """True where ``cudaHostRegister`` maps page-locked host memory into a GPU-accessible aperture that is
    bounded by device memory rather than host RAM — notably WSL2 (its dxgkrnl GPU-paravirt layer), where
    a window larger than VRAM fails to page-lock even with host RAM to spare. On native Linux pinning is
    host-only, so this is False and the window is bounded by host RAM alone.
    """
    try:
        with open("/proc/version") as f:
            v = f.read().lower()
        return "microsoft" in v or "wsl" in v
    except Exception:
        return False


@functools.lru_cache(maxsize=None)
def resolve_window_experts(num_experts_E: int) -> int:
    """Size the pinned window ``W`` for the pinned store: page-lock the largest hot window that fits the pin
    budget, and ``0`` (full pin) when the whole store fits. The budget is
    ``MemAvailable * _AUTO_WINDOW_HOST_FRACTION``, additionally capped by ``total_vram *
    _AUTO_WINDOW_VRAM_FRACTION`` where pinning is VRAM-coupled (WSL). Mirrors ``resolve_num_resident_experts``
    — reads geometry off ``ModelConfig`` and calls the pure ``compute_window_experts``. Sizing is automatic;
    there is no user-facing window knob. Cached: every MoE layer resolves the same W.
    """
    _, _, moe_layers, per_el = _moe_geometry()

    avail = _PRE_LOAD_HOST_AVAIL or _host_available_bytes()
    budget = avail * _AUTO_WINDOW_HOST_FRACTION
    vram_capped = False
    if _pin_is_vram_coupled():
        try:
            total_vram = torch.cuda.mem_get_info()[1]
        except Exception:
            total_vram = 0
        if total_vram:
            vram_cap = total_vram * _AUTO_WINDOW_VRAM_FRACTION
            if vram_cap < budget:
                budget, vram_capped = vram_cap, True
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
        " (VRAM-capped)" if vram_capped else "",
        per_el / 1e6,
        moe_layers,
    )
    return w
