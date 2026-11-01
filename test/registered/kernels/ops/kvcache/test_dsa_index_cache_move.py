"""DSA relocation must move token keys and scales, preserving neighboring slots."""

import sys

import pytest
import torch

from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.parametrize(
    "source_slots",
    [[], [1, 2], [64, 127, 128, 255, 256, 319]],
    ids=["empty", "small_ids", "cross_page_swap"],
)
def test_dsa_cache_move(source_slots):
    with get_parallel().override(attn_dcp_size=1):
        pool = DSATokenToKVPool(
            size=256,
            page_size=64,
            kv_lora_rank=512,
            dtype=torch.bfloat16,
            qk_rope_head_dim=64,
            layer_num=2,
            device="cuda",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=576,
            skip_topk_layers=[False, True],
        )
    torch.manual_seed(0)
    slots = torch.arange(320, device="cuda")
    keys = torch.randn(320, 128, device="cuda").to(torch.float8_e4m3fn)
    scales = (slots.float() + 1).unsqueeze(1)
    pool.set_index_k_scale_buffer(0, slots, keys, scales)
    for kv in pool.kv_buffer:
        kv.copy_(torch.randn_like(kv))
    original_kv = [kv.clone() for kv in pool.kv_buffer]
    source = torch.tensor(source_slots, device="cuda", dtype=torch.int64)
    target = source.flip(0)
    expected_keys = keys.view(torch.uint8).clone()
    expected_scales = scales.clone()
    expected_keys[target] = expected_keys[source]
    expected_scales[target] = expected_scales[source]
    pool.move_kv_cache(target, source)
    actual_keys, actual_scales = pool.get_index_k_scale_buffer(
        0,
        torch.tensor([320], device="cuda", dtype=torch.int32),
        torch.arange(5, device="cuda", dtype=torch.int32).unsqueeze(0),
        320,
        320,
    )
    torch.testing.assert_close(actual_keys, expected_keys, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_scales.view(torch.float32), expected_scales, rtol=0, atol=0
    )
    for actual, expected in zip(pool.kv_buffer, original_kv, strict=True):
        expected[target] = expected[source]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert pool.index_key_cache.buffer[1].shape[0] == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
