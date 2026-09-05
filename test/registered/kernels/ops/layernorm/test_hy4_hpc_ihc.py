import sys

import pytest
import torch

from sglang.kernels.ops.layernorm.hy4_ihc import (
    _hpc_ihc_op,
    fused_hy4_ihc_head,
    fused_hy4_ihc_post_pre,
    fused_hy4_ihc_pre,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.parametrize("hidden_size", [4096, 6144])
def test_hpc_ihc_eager_graph_parity(hidden_size):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if _hpc_ihc_op("fuse_ihc_post_pre", 4, hidden_size) is None:
        pytest.skip("requires a compatible HPC-Ops iHC build")

    torch.manual_seed(13)
    num_tokens = 7
    hc_mult = 4
    norm_eps, hc_eps, magnitude = 1e-5, 1e-6, 2.0
    x = torch.rand(
        (num_tokens, hc_mult, hidden_size), dtype=torch.bfloat16, device="cuda"
    )
    output = torch.rand((num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda")
    pre_weight = (
        torch.rand(
            (2 * hc_mult, hc_mult * hidden_size),
            dtype=torch.float32,
            device="cuda",
        )
        * 6e-3
    )
    next_weight = torch.rand_like(pre_weight) * 6e-3
    head_weight = (
        torch.rand(
            (hc_mult, hc_mult * hidden_size),
            dtype=torch.float32,
            device="cuda",
        )
        * 6e-3
    )
    pre_scale = torch.rand((2,), dtype=torch.float32, device="cuda")
    next_scale = torch.rand((2,), dtype=torch.float32, device="cuda")
    head_scale = torch.rand((1,), dtype=torch.float32, device="cuda")
    pre_base = torch.rand((2 * hc_mult,), dtype=torch.float32, device="cuda")
    next_base = torch.rand((2 * hc_mult,), dtype=torch.float32, device="cuda")
    head_base = torch.rand((hc_mult,), dtype=torch.float32, device="cuda")
    rms_weight = torch.rand((hidden_size,), dtype=torch.bfloat16, device="cuda")

    def run():
        _, post = fused_hy4_ihc_pre(
            x,
            pre_weight,
            pre_scale,
            pre_base,
            magnitude,
            norm_eps,
            hc_eps,
            rms_weight,
            norm_eps,
        )
        residual, reduced, next_post = fused_hy4_ihc_post_pre(
            output,
            x,
            post,
            next_weight,
            next_scale,
            next_base,
            magnitude,
            norm_eps,
            hc_eps,
            rms_weight,
            norm_eps,
        )
        head = fused_hy4_ihc_head(
            residual,
            head_weight,
            head_scale,
            head_base,
            norm_eps,
            hc_eps,
            rms_weight,
            norm_eps,
        )
        return residual, reduced, next_post, head

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        run()
    torch.cuda.current_stream().wait_stream(warmup_stream)

    eager = tuple(value.clone() for value in run())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_outputs = run()
    graph.replay()

    for eager_output, graph_output in zip(eager, graph_outputs):
        assert torch.equal(eager_output, graph_output)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
