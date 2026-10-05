"""What paged experts needs from a fused-MoE runner backend, one ``RunnerContract`` per backend.

The executor runs a step in waves (one wave when its experts fit the K slots), each through the
base method with the experts outside the wave masked. The runner returns every routed
(token, expert) output already multiplied by its router weight (``no_combine``); the contract
keeps each entry from the one wave that holds its expert and performs the top-k sum the runner
would have performed itself, so the paged layer is bit-identical to the unpaged one for any
number of waves. A quantization format names the contract of the runner its base method uses;
supporting a new backend means adding a contract here.
"""

from __future__ import annotations

from dataclasses import replace

import torch


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

    def merge(self, merged, partial, masked):
        """Fold a wave's ``[tokens, top_k, hidden]`` entries into the previous waves';
        ``masked`` marks the entries outside the wave."""
        return torch.where(masked.unsqueeze(-1), merged, partial)

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
