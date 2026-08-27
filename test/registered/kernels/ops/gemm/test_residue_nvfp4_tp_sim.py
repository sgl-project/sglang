"""Simulated TP=2 coverage for the residue NVFP4 linear op.

Real TP runs need a process group; the sharding MATH does not. Row-parallel
TP splits the contraction dim and all-reduces partial outputs, so this test
builds each rank's weight shard exactly the way the loader would (via the
shard plan's two-range gather for k_ext, contiguous split for mext_r1), runs
the opaque op per rank, sums the partials, and compares against the
unsharded run and the full-precision reference.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if torch.cuda.get_device_capability()[0] != 10:
    pytest.skip("sm100/sm103 (datacenter Blackwell) required", allow_module_level=True)
pytest.importorskip("cutlass", reason="nvidia-cutlass-dsl required")
pytest.importorskip("flashinfer", reason="flashinfer required")

from sglang.kernels.ops.gemm.residue_nvfp4_linear import nvfp4_linear
from sglang.kernels.ops.quantization.residue_nvfp4_quant import (
    indices_to_channel_masks,
)
from sglang.srt.layers.quantization.residue_nvfp4.tp import ResidueShardPlan
from sglang.test.kernels.residue_nvfp4 import quantize_nvfp4_weight

TP = 2
N, K, PER_BLOCK = 512, 4096, 2
NUM_SALIENT = K * PER_BLOCK // 8
K_EXT = K + NUM_SALIENT


def salient_indices():
    return torch.tensor(
        sorted(i for b in range(0, K, 8) for i in range(b, b + PER_BLOCK)),
        device="cuda",
    )


def run_kext(x, w_bf16_ext, indices, k_base, num_salient):
    """Quantize an extended weight and run the op's k_ext chain."""
    w_packed, w_sf, w_global = quantize_nvfp4_weight(w_bf16_ext)
    w_sf = w_sf.view(torch.float8_e4m3fn)
    x_global = ((448.0 * 6.0) / x.float().abs().max()).cuda()
    alpha = (1.0 / (x_global * w_global)).reshape(1).float().cuda()
    masks = indices_to_channel_masks(indices, k_base)
    return nvfp4_linear(
        x,
        w_packed,
        x_global.reshape(1),
        w_sf,
        w_sf,
        masks,
        alpha,
        k_base,
        num_salient,
        0,
        w_packed.shape[0],
        False,
        False,
    )


def test_row_parallel_kext_shards_sum_to_unsharded():
    torch.manual_seed(0)
    indices = salient_indices()
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / (K**0.5)
    w_ext = torch.cat([w, w[:, indices]], dim=1)
    m = 16
    x = torch.randn(m, K, dtype=torch.bfloat16, device="cuda")

    # Unsharded (TP=1) run.
    out_full = run_kext(x, w_ext, indices, K, NUM_SALIENT)

    # Per-rank runs: weight shard via the two-range gather, activation via
    # the contiguous base split, salient indices rebased per rank.
    out_sum = torch.zeros_like(out_full)
    for r in range(TP):
        plan = ResidueShardPlan(K, NUM_SALIENT, TP, r)
        plan.validate()
        w_shard = plan.gather(w_ext, scale=1)  # logical channels
        x_shard = x[:, r * plan.base_shard : (r + 1) * plan.base_shard]
        local_idx = plan.local_salient_indices(indices.cpu()).cuda()
        out_sum += run_kext(
            x_shard, w_shard, local_idx, plan.base_shard, plan.salient_shard
        )

    ref = x.float() @ w.float().T
    err_full = (out_full.float() - ref).norm() / ref.norm()
    err_sum = (out_sum.float() - ref).norm() / ref.norm()

    # Per-rank quantization is independent, so bitwise equality is not
    # expected -- but the sharded run must stay in the same error band as
    # the unsharded one (the silent-corruption failure mode is off by ~100%).
    assert err_sum < 2.0 * err_full + 5e-3, (
        f"TP={TP} row-parallel k_ext error band violated: "
        f"sharded={err_sum:.5f} unsharded={err_full:.5f}"
    )


def test_row_parallel_mext_r1_shards_sum_to_unsharded():
    from sglang.kernels.ops.quantization.residue_nvfp4_quant import (  # noqa: F401  (op path exercises it)
        scaled_fp4_quant_mext_r1,
    )

    torch.manual_seed(1)
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / (K**0.5)
    m = 8
    x = torch.randn(m, K, dtype=torch.bfloat16, device="cuda")

    def run_mext(x_part, w_part):
        w_packed, w_sf, w_global = quantize_nvfp4_weight(w_part)
        w_sf = w_sf.view(torch.float8_e4m3fn)
        x_global = ((448.0 * 6.0) / x_part.float().abs().max()).cuda()
        alpha = (1.0 / (x_global * w_global)).reshape(1).float().cuda()
        return nvfp4_linear(
            x_part,
            w_packed,
            x_global.reshape(1),
            w_sf,
            w_sf,
            x_part.new_zeros(1, dtype=torch.uint8),
            alpha,
            w_part.shape[1],
            0,
            0,
            w_packed.shape[0],
            True,
            True,
        )

    out_full = run_mext(x, w)
    out_sum = torch.zeros_like(out_full)
    for r in range(TP):
        # mext_r1 weights shard like stock NVFP4: contiguous K split.
        w_shard = w[:, r * (K // TP) : (r + 1) * (K // TP)]
        x_shard = x[:, r * (K // TP) : (r + 1) * (K // TP)]
        out_sum += run_mext(x_shard.contiguous(), w_shard.contiguous())

    ref = x.float() @ w.float().T
    err_full = (out_full.float() - ref).norm() / ref.norm()
    err_sum = (out_sum.float() - ref).norm() / ref.norm()
    assert err_sum < 2.0 * err_full + 5e-3, (
        f"TP={TP} row-parallel mext_r1 error band violated: "
        f"sharded={err_sum:.5f} unsharded={err_full:.5f}"
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
