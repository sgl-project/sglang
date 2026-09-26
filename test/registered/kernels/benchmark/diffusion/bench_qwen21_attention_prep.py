"""Qwen-Image 2.1 attention prep on the 1024x1024 production shape.

4096 image tokens, 32 heads x 128, an 18-row text prefix. The `cuda` column of
`kv_pack` is the shipped path: the projection already wrote K/V behind the
prefix, so the kernel normalizes in place and never copies V, which is why it
moves 2/3 of the bytes of the other two columns.
"""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.diffusion import (
    qknorm_complex_rope_cuda,
    qknorm_complex_rope_pack_,
)
from sglang.kernels.ops.diffusion.rope.qknorm_complex_rope_kv_triton import (
    qknorm_complex_rope_kv,
)
from sglang.kernels.ops.diffusion.rope.qknorm_complex_rope_triton import (
    qknorm_complex_rope,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=20,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
)

HEADS, HEAD_DIM, PREFIX, EPS = 32, 128, 18, 1e-6


def _rope_eager(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    z = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(z * rope[None, :, None]).flatten(-2).to(x.dtype)


def _norm(weight: torch.Tensor) -> RMSNorm:
    norm = RMSNorm(HEAD_DIM, EPS, cast_x_before_out_mul=True, force_native=True)
    norm = norm.to(device=weight.device, dtype=weight.dtype)
    norm.weight.data.copy_(weight)
    return norm


def _make_inputs(tokens: int):
    generator = torch.Generator(device="cuda").manual_seed(20260925)
    kwargs = dict(device="cuda", dtype=torch.bfloat16, generator=generator)
    q = torch.randn(1, tokens, HEADS, HEAD_DIM, **kwargs)
    k = torch.randn(1, tokens, HEADS, HEAD_DIM, **kwargs)
    v = torch.randn(1, tokens, HEADS, HEAD_DIM, **kwargs)
    kp = torch.randn(1, PREFIX, HEADS, HEAD_DIM, **kwargs)
    vp = torch.randn(1, PREFIX, HEADS, HEAD_DIM, **kwargs)
    wq = torch.randn(HEAD_DIM, **kwargs)
    wk = torch.randn(HEAD_DIM, **kwargs)
    angles = torch.randn(
        tokens, HEAD_DIM // 2, device="cuda", dtype=torch.float32, generator=generator
    )
    rope = torch.polar(torch.ones_like(angles), angles)
    return q, k, v, kp, vp, wq, wk, rope


@marker.parametrize("case", ["qk_rope", "kv_pack"])
@marker.parametrize("tokens", [4096], [1024])
@marker.benchmark("impl", ["eager", "triton", "cuda"])
def benchmark_attention_prep(case: str, tokens: int, impl: str):
    q, k, v, kp, vp, wq, wk, rope = _make_inputs(tokens)
    norm_q, norm_k = _norm(wq), _norm(wk)
    if case == "qk_rope":
        fns = {
            "eager": lambda: _rope_eager(norm_q(q), rope),
            "triton": lambda: qknorm_complex_rope(q, wq, rope, EPS),
            "cuda": lambda: qknorm_complex_rope_cuda(q, wq, rope, EPS),
        }
        memory_args = (q, wq, rope)
    else:
        k_out = torch.empty(
            1, PREFIX + tokens, HEADS, HEAD_DIM, device="cuda", dtype=q.dtype
        )
        v_out = torch.empty_like(k_out)
        k_out[:, PREFIX:].copy_(k)
        v_out[:, PREFIX:].copy_(v)

        def eager():
            _rope_eager(norm_q(q), rope)
            torch.cat([kp, _rope_eager(norm_k(k), rope)], 1)
            torch.cat([vp, v], 1)

        def triton():
            qknorm_complex_rope(q, wq, rope, EPS)
            qknorm_complex_rope_kv(k, wk, rope, v, kp, vp, EPS)

        def cuda():
            qknorm_complex_rope_pack_(
                q, k_out, v_out, wq, wk, rope, kp, vp, None, None, EPS
            )

        fns = {"eager": eager, "triton": triton, "cuda": cuda}
        memory_args = (q, k, v, kp, vp, rope)
    return marker.do_bench(
        fns[impl],
        memory_args=memory_args,
        memory_output=None,
        use_cuda_graph=False,
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    benchmark_attention_prep.run()
