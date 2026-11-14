"""Kimi-K3 latent MoE tail on ROCm, the part that differs between the two paths
(the up GEMM in between is the same shape either way):

  unfused: AITER rms_norm(latent) ... add3(up, shared, prefix)
  fused:   ... rstd_scale_add3(up, latent, shared, prefix)  (norm weight in w_up)

Rows cover the verify batches of the AgentX points: c4 (2 x 6 tokens), c16
(16 x 5), c56 (56..60 x 5). do_bench rotates past the 4 MB L2 but not the
256 MB MALL; in serving the GEMM and the all-reduce write up, latent and shared
just before this tail, so MALL-warm inputs are the serving pattern.
"""

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, stage="jit-kernel-benchmark", runner_config="amd")

LATENT, HIDDEN, EPS = 3584, 7168, 1e-5


def _unfused(up, latent, shared, prefix, gain):
    import aiter

    from sglang.kernels.ops.elementwise.add3 import add3

    aiter.rms_norm(latent, gain, EPS)
    return add3(up, shared, prefix, prefetch_bc=True)


def _fused(up, latent, shared, prefix, gain):
    from sglang.kernels.ops.layernorm.rstd_scale_add3 import rstd_scale_add3

    return rstd_scale_add3(up, latent, shared, prefix, eps=EPS)


@marker.parametrize("num_tokens", [1, 12, 80, 280, 300], [12, 300])
@marker.benchmark("impl", ["unfused", "fused"])
def benchmark(num_tokens: int, impl: str):
    up = create_random(num_tokens, HIDDEN)
    latent = create_random(num_tokens, LATENT)
    shared = create_random(num_tokens, HIDDEN)
    prefix = create_random(num_tokens, HIDDEN)
    gain = create_random(LATENT)
    fn = {"unfused": _unfused, "fused": _fused}[impl]
    return marker.do_bench(fn, input_args=(up, latent, shared, prefix, gain))


if __name__ == "__main__":
    benchmark.run()
