"""Paged experts on Ascend: the runner contract of the existing Ascend MoE runner.

The Ascend runner applies router weights and combines top-k outputs in finalize
routing. Each wave returns a partial ``[tokens, hidden]`` output. Summing the waves
matches the unpaged runner's output, and callers handle any separate routed scaling.
The runner uses ``num_experts = K`` so the routing tables match the slot ids.

Registered with paged experts when this module is imported (from the Ascend runner's
module), so the shared paged-experts code carries no Ascend imports.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.paged_experts.runners import (
    RunnerContract,
    register_runner_contract,
    run_on_slots,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.model_executor.cuda_graph_config import cuda_graph_fully_disabled


def _ascend_host_alloc(shape, dtype: torch.dtype) -> torch.Tensor:
    """Pinned host memory through torch's pinned allocator, the NPU selection of
    ``ALLOC_MEMORY_FUNCS`` (the memfd + cudaHostRegister default is CUDA-only)."""
    from sglang.srt.mem_cache.pool_host.common import alloc_with_pin_memory

    return alloc_with_pin_memory(
        dims=tuple(shape), dtype=dtype, device="cpu", pin_memory=True, allocator=None
    )


class AscendRunnerContract(RunnerContract):
    #: Pinned host rows through the NPU allocator, not the store's CUDA default.
    host_alloc = staticmethod(_ascend_host_alloc)

    def runner_config(self, moe_runner_config, num_slots: int):
        # Slot ids are the runner's expert ids: its routing tables and grouped GEMM
        # group lists must be K wide, and its own combine (the finalize op) stays on.
        return replace(
            moe_runner_config,
            num_experts=num_slots,
            num_local_experts=num_slots,
            no_combine=False,
            no_combine_keep_router_weight=False,
        )

    def validate(self) -> None:
        if not cuda_graph_fully_disabled():
            raise RuntimeError(
                "Paged experts on Ascend: CUDA graphs are not supported yet; run with "
                "--disable-cuda-graph (every step takes the host-planned eager path)"
            )
        if not envs.SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT.get():
            raise RuntimeError(
                "Paged experts on Ascend: set SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=1 so "
                "the expert weights stay ND: the host store must equal the GPU rows "
                "verbatim, which the NZ format cast breaks"
            )

    def run_wave(
        self, base_method, layer, dispatch_output, slots, merged, num_entries=None
    ):
        # Routes outside the wave run as slot 0 with weight 0: they compute, and the
        # finalize op zeroes them, so waves add cleanly.
        partial = run_on_slots(base_method, layer, dispatch_output, slots)
        if merged is None:
            return partial
        return merged + partial

    def finish(self, merged, hidden_states, routed_scaling_factor: float):
        # Match the unpaged runner's weighted sum. The model owns routed scaling.
        out = hidden_states.new_empty(merged.shape[0], merged.shape[-1])
        out.copy_(merged)
        return out


register_runner_contract(MoeRunnerBackend.ASCEND.value, AscendRunnerContract())
