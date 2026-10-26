"""What paged experts needs from a fused-MoE runner backend, one ``RunnerContract`` per backend.

The executor runs a step in waves (one wave when its experts fit the K slots). The runner
returns every routed (token, expert) output already multiplied by its router weight
(``no_combine``); the contract runs each wave over only the entries whose expert it holds,
writes them into one merged buffer, and performs the top-k sum the runner would have performed
itself. The contract follows from the runner the base method builds (``contract_for``);
supporting a new backend means adding a contract here.
"""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F


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

    def run_wave(self, base_method, layer, dispatch_output, slots, merged):
        """Compute the routed entries of one wave (``slots`` >= 0) into ``merged``, the step's
        ``[tokens, top_k, hidden]`` entries (None before its first wave), and return it."""
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


class TritonContract(RunnerContract):
    def run_wave(self, base_method, layer, dispatch_output, slots, merged):
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

    def run_wave(self, base_method, layer, dispatch_output, slots, merged):
        # Only the wave's entries, compacted: the runner sees no masked rows.
        hidden_states = dispatch_output.hidden_states
        assert (
            dispatch_output.hidden_states_scale is None
            and dispatch_output.hidden_states_pre_quant is None
        ), "Paged experts: compacted waves need unquantized MoE inputs"
        entries = (slots >= 0).flatten().nonzero().squeeze(1)
        num = entries.numel()
        if num == 0:
            return merged
        # Padded to a few sizes per power of two, so the runner sees few distinct batch
        # shapes; padding rows go to slot 0 with weight 0 and are dropped.
        step = max(16, 1 << max(0, num.bit_length() - 4))
        pad = -num % step
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
        merged.view(-1, merged.shape[-1])[entries] = out.view(-1, merged.shape[-1])[
            :num
        ]
        return merged

    def _sum_top_k(self, entries, out, routed_scaling_factor: float) -> None:
        # The reduce fused_marlin_moe itself ends with.
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
