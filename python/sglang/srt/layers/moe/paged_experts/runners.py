"""What paged experts needs from a fused-MoE runner backend, one ``RunnerContract`` per backend.

The executor runs a step in waves (one wave when its experts fit the K slots). The runner
returns every routed (token, expert) output already multiplied by its router weight
(``no_combine``); the contract runs each wave over only the entries whose expert it holds,
writes them into one merged buffer, and performs the top-k sum the runner would have performed
itself. The contract follows from the runner the base method builds (``contract_for``);
supporting a new backend means adding a contract here.
"""

from __future__ import annotations

import contextlib
import weakref
from dataclasses import replace

import torch
import torch.nn.functional as F

#: The most routed entries one compacted runner call computes: it bounds the call's working
#: memory (``RunnerContract.prefill_bytes``) whatever a wave routes, e.g. a long prompt whose
#: tokens all go to the experts one wave holds.
COMPACT_ROWS = 4096


def run_on_slots(base_method, layer, dispatch_output, slots, outside: int = 0):
    """Run the base method over the K-slot table: routed entry i goes to GPU slot ``slots[i]``;
    entries at -1 get expert id ``outside`` and router weight 0, so they contribute nothing."""
    masked = slots < 0
    topk_output = dispatch_output.topk_output
    wave_output = dispatch_output._replace(
        topk_output=topk_output._replace(
            topk_ids=slots.masked_fill(masked, outside),
            topk_weights=topk_output.topk_weights.masked_fill(masked, 0),
        ),
    )
    return base_method.apply(layer=layer, dispatch_output=wave_output).hidden_states


def runner_config(moe_runner_config, num_slots: int):
    """The base runner's config: K local experts, router-weighted per-expert outputs."""
    return replace(
        moe_runner_config,
        num_local_experts=num_slots,
        inplace=False,
        no_combine=True,
        no_combine_keep_router_weight=True,
    )


class RunnerContract:
    #: Whether the runner reads a layout the base method's post-load step repacks the
    #: experts into; the step then runs over the host store K experts at a time.
    repacks: bool = False

    def run_wave(
        self, base_method, layer, dispatch_output, slots, merged, num_entries=None
    ):
        """Compute the routed entries of one wave (``slots`` >= 0) into ``merged``, the step's
        ``[tokens, top_k, hidden]`` entries (None before its first wave), and return it.
        ``num_entries``, if the host knows it, is how many routed entries the wave has."""
        raise NotImplementedError

    def finish(self, merged, hidden_states, routed_scaling_factor: float):
        """The layer output: the top-k sum of the merged entries, trimmed to the input's width
        (a runner may pad the hidden dimension, e.g. MXFP4 Marlin)."""
        out = hidden_states.new_empty(merged.shape[0], merged.shape[-1])
        self._sum_top_k(merged, out, routed_scaling_factor)
        return out[..., : hidden_states.shape[-1]]

    def _sum_top_k(self, entries, out, routed_scaling_factor: float) -> None:
        """Sum ``[tokens, top_k, hidden]`` into ``out`` exactly as the runner would."""
        raise NotImplementedError

    def compacts(self, base_method, layer, dispatch_output) -> bool:
        """Whether ``run_wave`` runs only the wave's entries (in calls of at most
        ``COMPACT_ROWS``), so a step of any size may go through it."""
        return False

    def prefill_bytes(
        self, hidden_size: int, intermediate_size: int, tokens: int, top_k: int, dtype
    ) -> int:
        """GPU memory a step of ``tokens`` tokens takes in a layer of this size beyond the
        runner's own use for the same step unpaged: the merged ``[tokens, top_k, hidden]``
        entries and the largest compacted call (its gathered inputs, outputs and
        intermediates)."""
        hidden = _padded_hidden(hidden_size)
        inter = intermediate_size
        size = torch.empty((), dtype=dtype).element_size()
        merged = tokens * top_k * hidden * size
        rows = COMPACT_ROWS + _pad_rows(COMPACT_ROWS)
        call = rows * ((4 * hidden + 3 * inter) * size + hidden + inter)
        return merged + call


def _padded_hidden(hidden: int) -> int:
    """The widest hidden dimension a runner may pad its outputs to (MXFP4 Marlin: 256)."""
    return -(-hidden // 256) * 256


def _pad_rows(num: int) -> int:
    """Padding rows that bring ``num`` to the next of a few sizes per power of two, so the
    runner sees few distinct batch shapes."""
    step = max(16, 1 << max(0, num.bit_length() - 4))
    return -num % step


def _compact_calls(slots, num_entries):
    """The wave's routed entries (``slots`` >= 0, flattened), in calls of at most
    ``COMPACT_ROWS``; found without a sync when the host knows how many there are."""
    routed = (slots >= 0).flatten()
    entries = (
        routed.nonzero()
        if num_entries is None
        else routed.nonzero_static(size=num_entries)
    ).squeeze(1)
    return [part for part in entries.split(COMPACT_ROWS) if part.numel()]


def _run_compacted(base_method, layer, dispatch_output, slots, entries, merged):
    """Run ``entries`` of the step (flat indices into ``slots``), one per row with top-k 1,
    and write their outputs into ``merged`` (allocated, zeroed, at the runner's width on the
    first call). Padding rows go to slot 0 with weight 0 and are dropped."""
    hidden_states = dispatch_output.hidden_states
    num = entries.numel()
    pad = _pad_rows(num)
    tokens = F.pad(entries // slots.shape[1], (0, pad))
    topk = dispatch_output.topk_output
    compact = dispatch_output._replace(
        hidden_states=hidden_states[tokens],
        topk_output=topk._replace(
            topk_ids=F.pad(slots.flatten()[entries], (0, pad)).unsqueeze(1),
            topk_weights=F.pad(
                topk.topk_weights.flatten()[entries], (0, pad)
            ).unsqueeze(1),
            router_logits=(
                None if topk.router_logits is None else topk.router_logits[tokens]
            ),
        ),
    )
    out = base_method.apply(layer=layer, dispatch_output=compact).hidden_states
    if merged is None:  # at the runner's width, which may pad the hidden dimension
        merged = out.new_zeros(*slots.shape, out.shape[-1])
    merged.view(-1, merged.shape[-1])[entries] = out.view(-1, merged.shape[-1])[:num]
    return merged


def _unquantized_inputs(dispatch_output) -> bool:
    return (
        dispatch_output.hidden_states_scale is None
        and getattr(dispatch_output, "hidden_states_pre_quant", None) is None
    )


def _config_lookup(num_tokens: int, top_k: int):
    """The triton runner's kernel configs, looked up as for the step's full call."""
    try:
        from sglang.srt.layers.moe.moe_runner.triton_utils import (
            override_config_lookup,
        )
    except ImportError:  # no triton: no triton configs to keep
        return contextlib.nullcontext()
    return override_config_lookup(num_tokens, top_k)


def _row_local_activations(info) -> bool:
    """Whether the runner quantizes each row of both GEMMs' inputs on its own (no
    quantization, static scales, per-token or per-group scales): only then does a row's
    output not depend on which other rows share its call. A dynamic per-tensor scale (an
    input or intermediate scale left None) is a max over all rows."""
    if info is None or getattr(info, "use_mxfp8", False):
        return False
    if not (info.use_fp8_w8a8 or info.use_int8_w8a8):
        return True
    return (
        info.block_shape is not None
        or info.per_channel_quant
        or (info.a13_scale is not None and info.a2_scale is not None)
    )


def _triton_quant_info(base_method, layer):
    """The triton quant info the base method (or its scheme, or the scheme's kernel) builds
    for ``layer``; None if none of them exposes it."""
    scheme = getattr(layer, "scheme", None)
    for owner in (base_method, scheme, getattr(scheme, "kernel", None)):
        get = getattr(owner, "get_triton_quant_info", None)
        if get is not None:
            return get(layer)
    return None


class TritonContract(RunnerContract):
    def __init__(self):
        # Per layer: whether a compacted call computes exactly what the full call does.
        self._exact = weakref.WeakKeyDictionary()

    def compacts(self, base_method, layer, dispatch_output) -> bool:
        if not _unquantized_inputs(dispatch_output):
            return False
        exact = self._exact.get(layer)
        if exact is None:
            exact = _row_local_activations(_triton_quant_info(base_method, layer))
            self._exact[layer] = exact
        return exact

    def run_wave(
        self, base_method, layer, dispatch_output, slots, merged, num_entries=None
    ):
        if self.compacts(base_method, layer, dispatch_output):
            # Each row's output is what the step's full call computes for it, with the
            # kernel configs that call would use; only the rows it holds are computed.
            with _config_lookup(*slots.shape):
                for entries in _compact_calls(slots, num_entries):
                    merged = _run_compacted(
                        base_method, layer, dispatch_output, slots, entries, merged
                    )
            return merged
        config = runner_of(base_method, layer).config
        skips = (
            config.num_experts is None or config.num_experts != config.num_local_experts
        )
        partial = run_on_slots(
            base_method, layer, dispatch_output, slots, outside=-1 if skips else 0
        )
        if merged is None:
            return partial
        outside = (slots < 0).unsqueeze(-1)
        return torch.where(outside, merged, partial, out=merged)

    def _sum_top_k(self, entries, out, routed_scaling_factor: float) -> None:
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            moe_combine_topk,
        )

        moe_combine_topk(
            intermediate_cache3=entries,
            out_hidden_states=out,
            routed_scaling_factor=routed_scaling_factor,
        )


class MarlinContract(RunnerContract):
    repacks = True

    def compacts(self, base_method, layer, dispatch_output) -> bool:
        return True

    def run_wave(
        self, base_method, layer, dispatch_output, slots, merged, num_entries=None
    ):
        # Only the wave's entries, compacted: the runner sees no masked rows.
        assert _unquantized_inputs(dispatch_output), (
            "Paged experts: compacted waves need unquantized MoE inputs"
        )
        for entries in _compact_calls(slots, num_entries):
            merged = _run_compacted(
                base_method, layer, dispatch_output, slots, entries, merged
            )
        return merged

    def _sum_top_k(self, entries, out, routed_scaling_factor: float) -> None:
        # fused_marlin_moe's own reduce, vectorized: 6.5 us per layer with the general
        # moe_sum_reduce vs ~1.5-2 us here, at bs=8 decode ([8, 10, 2560]).
        if (
            entries.dtype == torch.bfloat16
            and entries.is_contiguous()
            and out.is_contiguous()
            and entries.shape[-1] % 8 == 0
            # moe_topk_sum's grid is (ceil(K / 8 / threads), M) and gridDim.y is capped at
            # 65,535: a prefill chunk past that stays on the general reduce.
            and entries.shape[0] <= 65535
        ):
            from sglang.kernels.ops.moe.moe_topk_sum import moe_topk_sum

            moe_topk_sum(entries, out, routed_scaling_factor, threads=64)
            return
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            moe_sum_reduce,
        )

        moe_sum_reduce(entries, out, routed_scaling_factor)


TRITON = TritonContract()
MARLIN = MarlinContract()


def runner_of(base_method, layer):
    """The runner the base method built: its own, or its scheme's (or the scheme kernel's)."""
    scheme = getattr(layer, "scheme", None)
    for owner in (base_method, scheme, getattr(scheme, "kernel", None)):
        runner = getattr(owner, "runner", None)
        if runner is not None:
            return runner
    return None


def contract_for(runner) -> RunnerContract:
    """The contract of the runner the base method built for paged experts."""
    backend = getattr(runner, "runner_backend", None)
    if backend is not None and backend.is_triton():
        return TRITON
    if (
        backend is not None
        and backend.is_marlin()
        and not backend.is_experimental_sgl_marlin()
    ):
        return MARLIN
    raise RuntimeError(
        "Paged experts: requires the triton or Marlin MoE runner, the layer's method uses "
        f"{getattr(backend, 'value', None)}; select one with --moe-runner-backend if its "
        "quantization method supports it"
    )
