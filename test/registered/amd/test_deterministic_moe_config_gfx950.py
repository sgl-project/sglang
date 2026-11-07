"""Deterministic mode on gfx950 runs unquantized MoE through one Triton config up to
_DETERMINISTIC_GFX950_SMALL_M tokens and a faster one above. That keeps a token's
result independent of its batch only if both configs give the same bits, so check
exactly that: the same tokens through both configs, and the rows of a large batch
(large-M config) against the same rows computed in small batches (small-M config).
"""

import sys

import pytest
import torch

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import publish_build_topology

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_gfx95_supported(), reason="needs gfx950"
)

# (experts, top-k, hidden, intermediate per expert): Qwen3-30B-A3B and DeepSeek-V2-Lite at TP1,
# and a narrower expert.
SHAPES = [(128, 8, 2048, 768), (64, 6, 2048, 1408), (32, 4, 1024, 512)]


@pytest.fixture(scope="module", autouse=True)
def _runtime_scaffolding():
    """`fused_experts` needs global server args and a TP group (single rank, gloo)."""
    import os

    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29642")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    if not torch.distributed.is_initialized():
        init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
    if not model_parallel_is_initialized():
        publish_build_topology(tp_size=1, ep_size=1, pp_size=1)
        initialize_model_parallel(backend="gloo")


def _moe(x, w1, w2, topk_weights, topk_ids, config):
    from sglang.srt.layers.moe.moe_runner.triton_utils import override_config
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    out = StandardTopKOutput(topk_weights, topk_ids, None)
    with override_config(config):
        # fused_moe may write its output into its input.
        return fused_moe(x.clone(), w1, w2, out)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("experts,topk,hidden,inter", SHAPES)
def test_small_and_large_m_configs_give_the_same_bits(
    dtype, experts, topk, hidden, inter
):
    from sglang.srt.layers.moe.moe_runner.triton_utils import (
        fused_moe_triton_config as cfg,
    )

    small_cfg = cfg._DETERMINISTIC_CONFIG
    large_cfg = cfg._DETERMINISTIC_CONFIG_GFX950_LARGE_M
    g = torch.Generator(device="cpu").manual_seed(experts * hidden + inter)
    w1 = (
        (torch.randn(experts, 2 * inter, hidden, generator=g) / hidden**0.5)
        .to(dtype)
        .cuda()
    )
    w2 = (
        (torch.randn(experts, hidden, inter, generator=g) / inter**0.5).to(dtype).cuda()
    )
    m = 1024
    x = torch.randn(m, hidden, generator=g).to(dtype).cuda()
    weights, ids = torch.topk(
        torch.softmax(torch.randn(m, experts, generator=g), -1), topk, dim=-1
    )
    weights = (weights / weights.sum(-1, keepdim=True)).float().cuda()
    ids = ids.to(torch.int32).cuda()

    full_large = _moe(x, w1, w2, weights, ids, large_cfg)
    assert torch.equal(full_large, _moe(x, w1, w2, weights, ids, small_cfg))
    # Rows of the large batch against the same rows on their own or in small batches, as a
    # server would run them below the switch.
    for start, n in ((0, 1), (5, 7), (100, 64), (999, 25)):
        rows = slice(start, start + n)
        part = _moe(x[rows], w1, w2, weights[rows], ids[rows], small_cfg)
        assert torch.equal(part, full_large[rows]), f"rows {start}:{start + n}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
