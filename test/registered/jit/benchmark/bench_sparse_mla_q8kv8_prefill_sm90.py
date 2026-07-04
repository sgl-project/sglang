from __future__ import annotations

import math

import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.sparse_mla_q8kv8_prefill_sm90 import sparse_mla_q8kv8_prefill_fwd
from sglang.srt.utils import is_sm90_supported
from sglang.test.ci.ci_register import register_cuda_ci

try:
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd

    HAS_Q16_FLASHMLA = True
except ImportError:
    flash_mla_sparse_fwd = None
    HAS_Q16_FLASHMLA = False

register_cuda_ci(
    est_time=120, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

DTYPE_FP8 = torch.float8_e4m3fn
D_V = 512
H_KV = 1

CASES = [
    (4096, 8192, 128, 576, 2048),
    (4096, 32768, 128, 576, 2048),
    (4096, 65536, 128, 576, 2048),
    (4096, 8192, 64, 512, 512),
    (4096, 32768, 64, 512, 512),
]
CASES_CI = [
    (2, 1024, 64, 512, 128),
    (2, 1024, 64, 576, 128),
]

# This official benchmark intentionally measures the no-sink path. Current
# DeepSeek NSA E2E does not pass a per-head attention sink into sparse MLA, so
# sink-enabled timings are kernel feature coverage rather than E2E proxy data.

LINE_VALS = ["q8_fp8_jit"]
if HAS_Q16_FLASHMLA:
    LINE_VALS.insert(0, "q16_bf16_flashmla")


def _sm90_available() -> bool:
    return is_sm90_supported()


def _make_indices(s_q: int, s_kv: int, topk: int, d_qk: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(1000 + d_qk + topk)
    return torch.randint(
        0,
        s_kv,
        (s_q, H_KV, topk),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )


def _make_q16_inputs(s_q: int, s_kv: int, h_q: int, d_qk: int, topk: int):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(2000 + d_qk + s_kv)
    q = torch.randn(
        (s_q, h_q, d_qk), dtype=torch.bfloat16, device="cuda", generator=generator
    )
    kv = torch.randn(
        (s_kv + 1, H_KV, d_qk), dtype=torch.bfloat16, device="cuda", generator=generator
    )
    indices = _make_indices(s_q, s_kv, topk, d_qk)
    sm_scale = 1.0 / math.sqrt(d_qk)
    return q, kv, indices, sm_scale


def _make_q8_inputs(s_q: int, s_kv: int, h_q: int, d_qk: int, topk: int):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(3000 + d_qk + s_kv)
    q = (torch.randn((s_q, h_q, d_qk), device="cuda", generator=generator) * 0.05).to(
        DTYPE_FP8
    )
    kv = torch.zeros((s_kv + 1, H_KV, d_qk), dtype=DTYPE_FP8, device="cuda")
    kv[:s_kv] = (
        torch.randn((s_kv, H_KV, d_qk), device="cuda", generator=generator) * 0.05
    ).to(DTYPE_FP8)
    indices = _make_indices(s_q, s_kv, topk, d_qk)
    q_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    kv_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    sm_scale = 1.0 / math.sqrt(d_qk)
    return q, kv, indices, sm_scale, q_scale, kv_scale


@marker.parametrize("s_q,s_kv,h_q,d_qk,topk", CASES, CASES_CI)
@marker.benchmark("provider", LINE_VALS)
def bench_sparse_mla_q8kv8_prefill_sm90(
    s_q: int, s_kv: int, h_q: int, d_qk: int, topk: int, provider: str
) -> marker.BenchResult:
    if provider == "q16_bf16_flashmla":
        if not HAS_Q16_FLASHMLA:
            marker.skip("sgl_kernel.flash_mla.flash_mla_sparse_fwd is unavailable")
        q, kv, indices, sm_scale = _make_q16_inputs(s_q, s_kv, h_q, d_qk, topk)

        def fn():
            return flash_mla_sparse_fwd(q, kv, indices, sm_scale, D_V)

    elif provider == "q8_fp8_jit":
        if not _sm90_available():
            marker.skip("Q8KV8 sparse prefill benchmark requires SM90 CUDA")
        q, kv, indices, sm_scale, q_scale, kv_scale = _make_q8_inputs(
            s_q, s_kv, h_q, d_qk, topk
        )

        def fn():
            return sparse_mla_q8kv8_prefill_fwd(
                q, kv, indices, sm_scale, q_scale, kv_scale, D_V
            )

    else:
        raise ValueError(f"Unknown provider: {provider}")

    return marker.do_bench(fn, use_cuda_graph=False, disable_log_bandwidth=True)


if __name__ == "__main__":
    bench_sparse_mla_q8kv8_prefill_sm90.run()
