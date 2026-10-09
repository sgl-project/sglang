"""Two-rank EP checkpoint slicing, SP-aligned collective, and dense-output parity.

The test also covers the fixed-capacity dispatcher's no-local-route case.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import unittest
from typing import TYPE_CHECKING

import torch

from sglang.test.test_utils import CustomTestCase, find_available_port

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.layers.moe import (
        LingBotVideoSparseMoeBlock,
        MoeExpertParallelInfo,
    )

_NUM_EXPERTS = 32
_HIDDEN = 64
_INTERMEDIATE = 32
_TOP_K = 8
_DENSE_PARITY_REL_FRO_LIMIT = 5e-3


def _build_block(
    num_experts: int,
    *,
    ep_info: MoeExpertParallelInfo | None = None,
) -> LingBotVideoSparseMoeBlock:
    from sglang.multimodal_gen.runtime.layers.moe import LingBotVideoSparseMoeBlock

    return (
        LingBotVideoSparseMoeBlock(
            hidden_size=_HIDDEN,
            intermediate_size=_INTERMEDIATE,
            num_experts=num_experts,
            top_k=_TOP_K,
            score_func="sigmoid",
            norm_topk_prob=True,
            n_group=4,
            topk_group=2,
            routed_scaling_factor=2.5,
            n_shared_experts=1,
            ep_info=ep_info,
        )
        .cuda()
        .to(torch.bfloat16)
    )


def _set_router(block, num_experts: int, seed: int) -> None:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        block.router.weight.copy_(
            torch.randn(num_experts, _HIDDEN, generator=gen).cuda()
        )
        block.router.e_score_correction_bias.copy_(torch.zeros(num_experts).cuda())


def _set_shared_experts(block, seed: int) -> None:
    assert block.shared_experts is not None
    block.shared_experts.to(dtype=torch.bfloat16)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for param in block.shared_experts.parameters():
            param.copy_(torch.randn(param.shape, generator=gen).to(param))


def _append_topology_failures(world: int, failures: list[str]) -> None:
    from sglang.multimodal_gen.runtime.distributed import parallel_state as ps

    decode_group = ps.get_decode_parallel_group_coordinator()
    if decode_group.world_size != world or decode_group.ranks != list(range(world)):
        failures.append(
            "VAE decode group does not include the shared SP/EP ranks: "
            f"world_size={decode_group.world_size} ranks={decode_group.ranks}"
        )
    if ps.get_dit_world_size() != world:
        failures.append(
            "DiT world size does not count the shared SP/EP ranks exactly once: "
            f"got {ps.get_dit_world_size()}, expected {world}"
        )


def _append_dispatch_dense_parity_failure(
    *,
    block: LingBotVideoSparseMoeBlock,
    x: torch.Tensor,
    full_w1: torch.Tensor,
    full_w2: torch.Tensor,
    full_w3: torch.Tensor,
    label: str,
    failures: list[str],
) -> None:
    with torch.no_grad():
        dispatched_out = block(x)

    dense = _build_block(_NUM_EXPERTS)
    dense.router = block.router
    assert dense.shared_experts is not None
    assert block.shared_experts is not None
    dense.shared_experts.load_state_dict(block.shared_experts.state_dict())
    with torch.no_grad():
        dense.experts.w13_weight.copy_(torch.cat((full_w1, full_w3), dim=1).cuda())
        dense.experts.w2.copy_(full_w2.cuda())
        dense_out = dense(x)
    if not torch.isfinite(dispatched_out).all().item():
        failures.append(f"{label} token-dispatch EP output contains NaN or Inf")
        return
    if not torch.isfinite(dense_out).all().item():
        failures.append(f"{label} dense output contains NaN or Inf")
        return

    # no_combine changes BF16 rounding order, so compare relative Frobenius error.
    diff = dispatched_out.float() - dense_out.float()
    rel_fro = (
        torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(dense_out.float())
    ).item()
    if not math.isfinite(rel_fro) or rel_fro > _DENSE_PARITY_REL_FRO_LIMIT:
        failures.append(
            f"{label} token-dispatch EP exceeds the dense relative-Frobenius tolerance: "
            f"relF={rel_fro:.3e} (limit {_DENSE_PARITY_REL_FRO_LIMIT:.1e})"
        )


def _append_dispatch_parity_failures(
    *,
    rank: int,
    world: int,
    failures: list[str],
) -> None:
    from sglang.multimodal_gen.runtime.layers.moe import resolve_moe_expert_parallel
    from sglang.multimodal_gen.runtime.models.dits.lingbot_video_moe import (
        pack_expert_weights,
    )

    _append_topology_failures(world, failures)

    info = resolve_moe_expert_parallel(_NUM_EXPERTS, ep_size=world)
    block = _build_block(
        _NUM_EXPERTS,
        ep_info=info,
    )
    _set_router(block, _NUM_EXPERTS, seed=17)
    _set_shared_experts(block, seed=19)

    gen = torch.Generator(device="cpu").manual_seed(23)
    full_w1 = torch.randn(_NUM_EXPERTS, _INTERMEDIATE, _HIDDEN, generator=gen)
    full_w2 = torch.randn(_NUM_EXPERTS, _HIDDEN, _INTERMEDIATE, generator=gen)
    full_w3 = torch.randn(_NUM_EXPERTS, _INTERMEDIATE, _HIDDEN, generator=gen)
    sharded = dict(
        pack_expert_weights(
            iter(
                [
                    ("blocks.0.ffn.experts.w1", full_w1),
                    ("blocks.0.ffn.experts.w2", full_w2),
                    ("blocks.0.ffn.experts.w3", full_w3),
                ]
            ),
            ep_info=info,
        )
    )
    with torch.no_grad():
        block.experts.w13_weight.copy_(
            sharded["blocks.0.ffn.experts.w13_weight"].cuda()
        )
        block.experts.w2.copy_(sharded["blocks.0.ffn.experts.w2"].cuda())

    # Equal-split collectives require identical row counts; only values vary by rank.
    local_gen = torch.Generator(device="cuda").manual_seed(29 + rank)
    x = torch.randn(
        1,
        128,
        _HIDDEN,
        generator=local_gen,
        device="cuda",
        dtype=torch.bfloat16,
    )
    _append_dispatch_dense_parity_failure(
        block=block,
        x=x,
        full_w1=full_w1,
        full_w2=full_w2,
        full_w3=full_w3,
        label="fixed-capacity",
        failures=failures,
    )

    # Route every token to rank 0; rank 1 must return zero from its -1 slots.
    with torch.no_grad():
        block.router.weight.zero_()
        block.router.e_score_correction_bias.fill_(-1)
        block.router.e_score_correction_bias[: info.num_local_experts].fill_(1)
    no_local_routes_x = torch.randn(
        1,
        4,
        _HIDDEN,
        generator=local_gen,
        device="cuda",
        dtype=torch.bfloat16,
    )
    _append_dispatch_dense_parity_failure(
        block=block,
        x=no_local_routes_x,
        full_w1=full_w1,
        full_w2=full_w2,
        full_w3=full_w3,
        label="no-valid-local-routes",
        failures=failures,
    )


def _worker_impl() -> int:
    import torch.distributed as dist

    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )
    from sglang.srt.runtime_context import get_context
    from sglang.srt.server_args import ServerArgs as SrtServerArgs

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    if get_context()._server_args is None:
        get_context().set_server_args(SrtServerArgs(model_path="dummy"))
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1,
        sp_size=world,
        ulysses_degree=world,
    )

    failures: list[str] = []
    _append_dispatch_parity_failures(
        rank=rank,
        world=world,
        failures=failures,
    )
    for failure in failures:
        print(f"FAILURE rank{rank}: {failure}", flush=True)
    dist.barrier()
    if rank == 0 and not failures:
        print("MOE_EP_DISPATCH_PARITY PASS", flush=True)
    return 1 if failures else 0


def _worker() -> int:
    from sglang.multimodal_gen.runtime.distributed import cleanup_dist_env_and_memory

    try:
        return _worker_impl()
    finally:
        cleanup_dist_env_and_memory()


def _run_world_test(test_case: CustomTestCase, world: int) -> None:
    if not torch.cuda.is_available():
        test_case.skipTest("requires a CUDA or ROCm GPU")
    if torch.cuda.device_count() < world:
        test_case.skipTest(f"needs {world} GPUs")
    master_port = str(find_available_port(29500))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc-per-node={world}",
            "--master-addr=127.0.0.1",
            f"--master-port={master_port}",
            __file__,
            "--worker",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    output = result.stdout + result.stderr
    if result.returncode:
        test_case.fail(f"torchrun worker failed ({result.returncode}):\n{output}")
    test_case.assertIn("MOE_EP_DISPATCH_PARITY PASS", output)


class TestMoeExpertParallelTwoGpu(CustomTestCase):
    def test_expert_parallel_two_ranks(self):
        _run_world_test(self, 2)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        sys.exit(_worker())
    unittest.main()
