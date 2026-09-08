"""Compare Hopper BF16 QKNorm/RoPE at LTX video, audio, and cross-attention shapes."""

import torch
import torch.nn.functional as F

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.diffusion import ltx2_qknorm_split_rope_cuda
from sglang.kernels.ops.diffusion.rope.ltx2_rotary_triton import (
    apply_ltx2_split_rotary_emb,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=20, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def reference(q, qc, qs, qw, k, kc, ks, kw, heads, dim):
    return (
        apply_ltx2_split_rotary_emb(F.rms_norm(q, (heads * dim,), qw, 1e-6), qc, qs),
        apply_ltx2_split_rotary_emb(F.rms_norm(k, (heads * dim,), kw, 1e-6), kc, ks),
    )


def fused(q, qc, qs, qw, k, kc, ks, kw, heads, dim):
    return ltx2_qknorm_split_rope_cuda(
        q,
        qc,
        qs,
        qw,
        k,
        kc,
        ks,
        kw,
        eps=1e-6,
        num_heads=heads,
        head_dim=dim,
        allow_sm90=True,
    )


@marker.parametrize(
    "q_seq,k_seq,heads,dim",
    [
        (1536, 1536, 32, 128),
        (1536, 1024, 32, 128),
        (126, 126, 32, 64),
        (126, 1024, 32, 64),
        (6144, 6144, 32, 128),
    ],
    [(17, 9, 32, 64)],
)
@marker.benchmark("impl", ["eager", "fused"], unit="us")
def benchmark(q_seq, k_seq, heads, dim, impl):
    if torch.cuda.get_device_capability() != (9, 0):
        return marker.skip("Hopper BF16 intermediate-rounding comparison")

    def side(seq):
        x = torch.randn(1, seq, heads * dim, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(heads * dim, dtype=torch.bfloat16, device="cuda")
        cos = torch.randn(
            1, seq, heads, dim // 2, dtype=torch.bfloat16, device="cuda"
        ).transpose(1, 2)
        sin = torch.randn_like(cos)
        return x, cos, sin, w

    args = (*side(q_seq), *side(k_seq), heads, dim)
    return marker.do_bench(
        reference if impl == "eager" else fused,
        input_args=args,
    )


if __name__ == "__main__":
    benchmark.run()
