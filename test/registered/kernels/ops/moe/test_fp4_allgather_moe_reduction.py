"""On the FlashInfer cutlass FP4 all-gather MoE route, the combine completes the
MoE output, so ``post_experts_all_reduce`` must not reduce it again over EP.

The route (``modelopt_fp4`` + ``flashinfer_cutlass``, DP attention, no a2a
backend, ``moe_ep_size == attn_dp_size``) all-gathers every DP rank's tokens over
_TP and reduce-scatters the expert outputs over _TP, which spans the EP group.
Each rank then holds the complete outputs of its own DP-local tokens, so an EP
all-reduce after it would sum different tokens across ranks.

Run with ``python test_fp4_allgather_moe_reduction.py``; it relaunches itself
under torchrun on 4 GPUs. It runs the real combine and all-reduces over NCCL,
but no FP4 kernel and no model weights.
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from sglang.srt.distributed import parallel_state as ps
from sglang.srt.layers.dp_attention import (
    init_dp_gathered_buffer,
    initialize_dp_attention,
    set_dp_buffer_len,
)
from sglang.srt.layers.moe import (
    MoeRunnerConfig,
    initialize_moe_config,
    post_experts_all_reduce,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardCombineInput,
    StandardDispatcher,
)
from sglang.srt.runtime_context import get_flags, get_parallel, get_server_args
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main
from sglang.test.test_utils import publish_build_topology

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HIDDEN_SIZE = 256
TOKENS_PER_RANK = 3
# Each rank's share of every token's expert output. Powers of two keep every
# partial and every sum exact in bf16.
EXPERT_SHARES = (0.125, 0.125, 0.25, 0.5)


@pytest.fixture(scope="module")
def world_size():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    assert world_size == len(EXPERT_SHARES)
    torch.cuda.set_device(local_rank)
    ps.set_custom_all_reduce(False)
    ps.init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
    )
    # Pure EP over the attention-DP ranks: moe_ep_size == attn_dp_size == tp_size.
    publish_build_topology(
        world_rank=rank,
        tp_size=world_size,
        dp_size=world_size,
        enable_dp_attention=True,
        ep_size=world_size,
        moe_runner_backend="flashinfer_cutlass",
        quantization="modelopt_fp4",
        device="cuda",
    )
    ps.initialize_model_parallel()
    initialize_dp_attention(get_server_args())
    init_dp_gathered_buffer(
        SimpleNamespace(
            hf_config=SimpleNamespace(),
            hidden_size=HIDDEN_SIZE,
            dtype=torch.bfloat16,
        )
    )
    initialize_moe_config()
    yield world_size
    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()


def _filled(value: float, num_tokens: int = TOKENS_PER_RANK) -> torch.Tensor:
    return torch.full(
        (num_tokens, HIDDEN_SIZE), value, dtype=torch.bfloat16, device="cuda"
    )


def _values_per_rank(output: torch.Tensor) -> list:
    """The distinct values each rank's output holds, gathered from every rank."""
    group = get_parallel().tp_group
    gathered = torch.empty(
        (group.world_size, *output.shape), dtype=output.dtype, device=output.device
    )
    dist.all_gather_into_tensor(gathered, output, group=group.device_group)
    return [rank_output.unique().tolist() for rank_output in gathered]


def test_combined_output_is_not_reduced_again(world_size):
    assert should_use_flashinfer_cutlass_moe_fp4_allgather()
    rank = get_parallel().tp_rank

    # Every rank computed its share of the expert outputs for all gathered
    # tokens. DP rank j's tokens sum to j + 1 over the ranks.
    sizes = [TOKENS_PER_RANK] * world_size
    set_dp_buffer_len(
        global_dp_buffer_len=sum(sizes),
        local_dp_buffer_len=TOKENS_PER_RANK,
        dp_max_padding=False,
        global_num_tokens=sizes,
    )
    partial = torch.cat(
        [_filled((j + 1) * EXPERT_SHARES[rank]) for j in range(world_size)]
    )
    dispatcher = StandardDispatcher(
        MoeRunnerConfig(
            num_experts=2 * world_size, num_local_experts=2, num_fused_shared_experts=0
        )
    )
    combined = dispatcher.combine(StandardCombineInput(partial))
    # The combine reduce-scattered over _TP: each rank holds its tokens in full.
    assert _values_per_rank(combined) == [[j + 1.0] for j in range(world_size)]

    # As DeepseekV2MoE does after the experts. The output is already complete.
    output = post_experts_all_reduce(combined.clone())
    assert _values_per_rank(output) == [[j + 1.0] for j in range(world_size)]


def test_without_the_route_the_ep_sum_still_runs(world_size):
    rank = get_parallel().tp_rank
    with get_flags().moe.override(disable_fp4_allgather=True):
        assert not should_use_flashinfer_cutlass_moe_fp4_allgather()
        output = post_experts_all_reduce(_filled(rank + 1.0))
    total = sum(range(1, world_size + 1))
    assert _values_per_rank(output) == [[float(total)]] * world_size


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,))
