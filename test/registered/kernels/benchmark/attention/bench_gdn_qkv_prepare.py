"""Cold-cache GDN preparation; pass the backing tensor to preserve strides."""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.fla.l2norm import (
    gdn_prefill_qkv_prepare_fwd,
    l2norm_fwd,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=6, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def run(packed, provider):
    tokens = packed.shape[0]
    q = packed[:, :2048].view(tokens, 16, 128)
    k = packed[:, 2048:4096].view(tokens, 16, 128)
    v = packed[:, 4096:12288].view(tokens, 64, 128)
    if provider == "fused":
        return gdn_prefill_qkv_prepare_fwd(q, k, v)
    return l2norm_fwd(q.contiguous()), l2norm_fwd(k.contiguous()), v.contiguous()


@marker.parametrize("tokens", [1, 32, 128, 512, 2048], [32, 2048])
@marker.benchmark("provider", ["fused", "unfused_torch_layout"])
def benchmark(tokens, provider):
    packed = torch.randn(tokens, 12288, device="cuda", dtype=torch.bfloat16)
    return marker.do_bench(run, input_args=(packed, provider))


if __name__ == "__main__":
    benchmark.run()
