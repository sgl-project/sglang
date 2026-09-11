"""Correctness of the uniform-FP8 SWA store (FusedKNormRopeFlashMLAKernel,
kUniformStore).

The kernel fuses RMSNorm + RoPE + bf16 round-trip + plain e4m3 cast
(per-tensor scale 1.0) + scatter into the 512-byte-per-token uniform SWA
pool at explicit out_loc rows (negative locs skipped). Reference is the
unfused pipeline it replaces: fused_norm_rope_inplace_triton followed by an
e4m3 cast + index_put.

The CUDA block reduction sums the RMSNorm squares in a different order than
the Triton reference, so a ~1e-6 fraction of elements can land one e4m3 ulp
apart; the assertions allow exactly that and nothing more.
"""

import pytest
import torch

from sglang.kernels.ops.attention.deepseek_v4_rope import fused_norm_rope_inplace_triton
from sglang.kernels.ops.attention.dsv4.elementwise import fused_k_norm_rope_flashmla
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

HEAD_DIM = 512
ROPE_DIM = 64
PAGE_SIZE = 128


def _make_freqs(max_pos: int) -> torch.Tensor:
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, ROPE_DIM, 2, device="cuda").float() / ROPE_DIM)
    )
    t = torch.arange(max_pos, device="cuda").float()
    angles = torch.outer(t, inv_freq)
    return torch.polar(torch.ones_like(angles), angles)  # complex64 [max_pos, 32]


def _reference_rows(kv, weight, eps, freqs, positions):
    x = kv.clone().contiguous()
    fused_norm_rope_inplace_triton(x, weight, eps, freqs, positions=positions)
    return x.to(torch.float8_e4m3fn)


def _assert_rows_match(got_fp8, ref_fp8):
    got, ref = got_fp8.float(), ref_fp8.float()
    mismatch = (got_fp8.view(torch.uint8) != ref_fp8.view(torch.uint8)).float().mean()
    assert mismatch.item() <= 1e-4, f"{mismatch.item()=}"
    denom = ref.abs().clamp(min=2**-9)
    assert ((got - ref).abs() / denom).max().item() <= 0.13


@pytest.mark.parametrize("num_rows", [1, 7, 128, 2048])
@pytest.mark.parametrize("noncontig_kv", [False, True])
def test_uniform_fp8_swa_store(num_rows, noncontig_kv):
    torch.manual_seed(num_rows + int(noncontig_kv))
    dev = torch.device("cuda")
    num_pages = (num_rows * 3) // PAGE_SIZE + 2
    num_slots = num_pages * PAGE_SIZE

    if noncontig_kv:
        # Mirror the real caller: kv is a trailing slice of the fused qkv_a
        # projection output, so its row stride exceeds HEAD_DIM.
        q_lora = 1536
        qkv_a = torch.randn(
            num_rows, q_lora + HEAD_DIM, device=dev, dtype=torch.bfloat16
        )
        kv = qkv_a[:, q_lora:]
    else:
        kv = torch.randn(num_rows, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    weight = torch.rand(HEAD_DIM, device=dev, dtype=torch.bfloat16) + 0.5
    freqs = _make_freqs(65536)
    positions = torch.randint(0, 65536, (num_rows,), device=dev, dtype=torch.int64)
    eps = 1e-6

    # Unique destinations; sprinkle in negative sentinels (skipped rows).
    out_loc = torch.randperm(num_slots, device=dev, dtype=torch.long)[:num_rows].to(
        torch.int32
    )
    skip = torch.rand(num_rows, device=dev) < 0.25
    skip[0] = False  # keep at least one valid row
    out_loc = torch.where(skip, torch.full_like(out_loc, -1), out_loc)

    pool = torch.zeros(
        num_pages, PAGE_SIZE * HEAD_DIM, dtype=torch.float8_e4m3fn, device=dev
    )
    ref_pool = pool.clone()

    fused_k_norm_rope_flashmla(
        kv=kv,
        kv_weight=weight,
        eps=eps,
        freqs_cis=freqs,
        positions=positions,
        out_loc=out_loc,
        kvcache=pool.view(torch.uint8),
        page_size=PAGE_SIZE,
        uniform_fp8_store=True,
    )

    ref_rows = _reference_rows(kv, weight, eps, freqs, positions)
    valid = out_loc >= 0
    ref_flat = ref_pool.view(torch.uint8).view(-1, HEAD_DIM)
    ref_flat[out_loc[valid].long()] = ref_rows[valid].view(torch.uint8)

    got_flat = pool.view(-1, HEAD_DIM)
    ref_flat_fp8 = ref_pool.view(-1, HEAD_DIM)
    _assert_rows_match(
        got_flat[out_loc[valid].long()], ref_flat_fp8[out_loc[valid].long()]
    )
    # Skipped and untouched rows must remain zero.
    touched = torch.zeros(num_slots, dtype=torch.bool, device=dev)
    touched[out_loc[valid].long()] = True
    assert (got_flat.view(torch.uint8)[~touched] == 0).all()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
