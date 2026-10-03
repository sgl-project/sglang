"""UltraQuant 4-bit KV cache: store and gather-dequant kernel parity."""

import sys

import pytest
import torch

from sglang.kernels.ops.kvcache.ultraquant import (
    ultraquant_gather_dequant,
    ultraquant_rotate,
    ultraquant_store,
)
from sglang.srt.layers.quantization.ultraquant_tensor import (
    GROUP_SIZE,
    UltraQuantKVQuantizeUtil,
    code_bytes,
    n_groups,
)
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=15, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_hip()),
    reason="UltraQuant KV cache currently targets ROCm",
)

HEAD_DIMS = (64, 128, 256)


def _build_cache(num_slots, kv_head_num, head_dim, seed):
    torch.manual_seed(seed)
    key = torch.randn(
        num_slots, kv_head_num, head_dim, device="cuda", dtype=torch.bfloat16
    )
    value = torch.randn(
        num_slots, kv_head_num, head_dim, device="cuda", dtype=torch.bfloat16
    )

    k_codes = torch.zeros(
        (num_slots, kv_head_num, code_bytes(head_dim)), dtype=torch.uint8, device="cuda"
    )
    v_codes = torch.zeros_like(k_codes)
    k_scales = torch.zeros(
        (num_slots, kv_head_num, n_groups(head_dim)), dtype=torch.uint8, device="cuda"
    )
    v_scales = torch.zeros_like(k_scales)

    slots = torch.arange(num_slots, device="cuda", dtype=torch.int64)
    ultraquant_store(key, value, k_codes, k_scales, v_codes, v_scales, slots)
    return k_codes, k_scales, v_codes, v_scales


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("head_num", (1, 6))
def test_store_kernel_matches_reference(head_dim, head_num):
    torch.manual_seed(head_dim * 31 + head_num)
    num_tokens, num_slots = 37, 512

    key = torch.randn(
        num_tokens, head_num, head_dim, device="cuda", dtype=torch.bfloat16
    )
    value = torch.randn(
        num_tokens, head_num, head_dim, device="cuda", dtype=torch.bfloat16
    )
    # Exercise the zero-group sentinel, a non-finite group and a channel outlier.
    key[:, :, :GROUP_SIZE] = 0.0
    value[3, :, 7] = 64.0
    value[5, :, GROUP_SIZE + 1] = float("inf")

    # Shuffled destinations catch address arithmetic that only works in order.
    loc = torch.randperm(num_slots, device="cuda")[:num_tokens].to(torch.int64)

    poison = 0xAA
    k_codes = torch.full(
        (num_slots, head_num, code_bytes(head_dim)),
        poison,
        dtype=torch.uint8,
        device="cuda",
    )
    v_codes = torch.full_like(k_codes, poison)
    k_scales = torch.full(
        (num_slots, head_num, n_groups(head_dim)),
        poison,
        dtype=torch.uint8,
        device="cuda",
    )
    v_scales = torch.full_like(k_scales, poison)

    ultraquant_store(key, value, k_codes, k_scales, v_codes, v_scales, loc)

    # Rotate with the kernel's own butterfly so the comparison is bit-exact.
    rotated = ultraquant_rotate(key.float())
    ref_k_codes, ref_k_scales = UltraQuantKVQuantizeUtil.batched_quantize(
        rotated, rotate=False
    )
    ref_v_codes, ref_v_scales = UltraQuantKVQuantizeUtil.batched_quantize(
        value, rotate=False
    )
    torch.testing.assert_close(k_scales[loc], ref_k_scales, rtol=0, atol=0)
    torch.testing.assert_close(k_codes[loc], ref_k_codes, rtol=0, atol=0)
    torch.testing.assert_close(v_scales[loc], ref_v_scales, rtol=0, atol=0)
    torch.testing.assert_close(v_codes[loc], ref_v_codes, rtol=0, atol=0)

    # Slots outside `loc` must be untouched.
    untouched = torch.ones(num_slots, dtype=torch.bool, device="cuda")
    untouched[loc] = False
    for buffer in (k_codes, v_codes, k_scales, v_scales):
        assert bool((buffer[untouched] == poison).all())


def test_store_kernel_rejects_unsupported_shapes():
    head_num, head_dim, num_slots = 2, 96, 8
    key = torch.randn(4, head_num, head_dim, device="cuda", dtype=torch.bfloat16)
    codes = torch.zeros(
        (num_slots, head_num, head_dim // 2), dtype=torch.uint8, device="cuda"
    )
    scales = torch.zeros((num_slots, head_num, 3), dtype=torch.uint8, device="cuda")
    loc = torch.arange(4, device="cuda", dtype=torch.int64)

    with pytest.raises(ValueError, match="power-of-two"):
        ultraquant_store(key, key, codes, scales, codes, scales, loc)

    key32 = torch.randn(4, head_num, 128, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="expects one of"):
        ultraquant_store(key32, key32, codes, scales, codes, scales, loc)


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("num_tokens", (1, 17, 256))
def test_gather_dequant_matches_reference(head_dim, num_tokens):
    """Gather-dequant must agree exactly with the reference dequantize.

    Both decode in fp32 from the same codes and scales, so anything short of
    equality means the kernel's addressing or unpacking has drifted from the
    format the attention kernels assume.
    """
    kv_head_num, num_slots = 4, 512
    k_codes, k_scales, v_codes, v_scales = _build_cache(
        num_slots, kv_head_num, head_dim, seed=5
    )

    # Scattered slots in an arbitrary order, as a ragged kv_indices run gives.
    torch.manual_seed(5)
    kv_indices = torch.randperm(num_slots, device="cuda")[:num_tokens].to(torch.int64)

    k_out = torch.empty(
        num_tokens, kv_head_num, head_dim, device="cuda", dtype=torch.bfloat16
    )
    v_out = torch.empty_like(k_out)
    ultraquant_gather_dequant(
        k_codes, k_scales, v_codes, v_scales, kv_indices, k_out, v_out
    )

    k_ref = UltraQuantKVQuantizeUtil.batched_dequantize(
        k_codes[kv_indices], k_scales[kv_indices], dtype=torch.bfloat16
    )
    v_ref = UltraQuantKVQuantizeUtil.batched_dequantize(
        v_codes[kv_indices], v_scales[kv_indices], dtype=torch.bfloat16
    )
    assert torch.equal(k_out, k_ref)
    assert torch.equal(v_out, v_ref)


def test_gather_dequant_empty_run_is_noop():
    """A zero-length run happens on padded batches and must not launch."""
    k_codes, k_scales, v_codes, v_scales = _build_cache(8, 2, 256, seed=6)
    empty = torch.empty(0, 2, 256, device="cuda", dtype=torch.bfloat16)
    ultraquant_gather_dequant(
        k_codes,
        k_scales,
        v_codes,
        v_scales,
        torch.empty(0, dtype=torch.int64, device="cuda"),
        empty,
        torch.empty_like(empty),
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
