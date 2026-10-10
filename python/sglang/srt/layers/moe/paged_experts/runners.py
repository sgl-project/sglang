"""What paged experts needs from a fused-MoE runner backend, one ``RunnerContract`` per backend.

The executor runs a step in waves (one wave when its experts fit the K slots). The runner
returns every routed (token, expert) output already multiplied by its router weight
(``no_combine``); the contract runs each wave over only the entries whose expert it holds,
writes them into one merged buffer, and performs the top-k sum the runner would have performed
itself. A quantization format names the contract of the runner its base method uses; supporting
a new backend means adding a contract here.
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


class RunnerContract:
    def runner_config(self, moe_runner_config, num_slots: int):
        """The base runner's config: K local experts, router-weighted per-expert outputs."""
        return replace(
            moe_runner_config,
            num_local_experts=num_slots,
            inplace=False,
            no_combine=True,
            no_combine_keep_router_weight=True,
        )

    def check(self, base_method) -> None:
        """Raise if the base method's runner cannot honor ``runner_config``."""

    def run_wave(self, base_method, layer, dispatch_output, slots, merged):
        """Compute the routed entries of one wave (``slots`` >= 0) into ``merged``, the step's
        ``[tokens, top_k, hidden]`` entries (None before its first wave), and return it."""
        hidden_states = dispatch_output.hidden_states
        assert (
            dispatch_output.hidden_states_scale is None
            and dispatch_output.hidden_states_pre_quant is None
        ), "Paged experts: compacted waves need unquantized MoE inputs"
        if merged is None:
            merged = hidden_states.new_zeros(*slots.shape, hidden_states.shape[-1])
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
        merged.view(-1, merged.shape[-1])[entries] = out.view(-1, merged.shape[-1])[
            :num
        ]
        return merged

    def finish(self, merged, hidden_states, routed_scaling_factor: float):
        """The layer output: the top-k sum of the merged entries."""
        out = torch.empty_like(hidden_states)
        self._sum_top_k(merged, out, routed_scaling_factor)
        return out

    def _sum_top_k(self, entries, out, routed_scaling_factor: float) -> None:
        """Sum ``[tokens, top_k, hidden]`` into ``out`` exactly as the runner would."""
        raise NotImplementedError


class TritonContract(RunnerContract):
    def check(self, base_method) -> None:
        backend = base_method.runner.runner_backend
        if not backend.is_triton():
            raise RuntimeError(
                f"Paged experts: requires the triton MoE runner, got {backend.value}"
            )

    def run_wave(self, base_method, layer, dispatch_output, slots, merged):
        config = base_method.runner.config
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
    # The Marlin MoE methods build a Marlin runner unconditionally, so there is nothing to check.

    def _sum_top_k(self, entries, out, routed_scaling_factor: float) -> None:
        # The reduce fused_marlin_moe itself ends with.
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            moe_sum_reduce,
        )

        moe_sum_reduce(entries, out, routed_scaling_factor)


TRITON = TritonContract()
MARLIN = MarlinContract()
