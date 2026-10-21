"""DSA cache moves and gathers must preserve token keys and scales."""

import sys
from types import SimpleNamespace

import pytest
import torch
from sglang.kernels.ops.kvcache.dsa_index_k_cache import GetKAndS, SetKAndS
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


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


def test_dsa_index_gather_int32_pages_above_2gib():
    """Int32 page tables must address keys and scales beyond byte offset 2**31.

    A physical page crosses the signed-int32 boundary within its key rows.
    Its scales and the next page are entirely above the boundary. Two unequal
    requests also check the flattened output offsets without a large output.
    """
    page_size, head_dim = 64, 128
    page_bytes = page_size * (head_dim + 4)
    boundary_page = (2**31) // page_bytes
    pool = SimpleNamespace(
        index_page_size=page_size, index_head_dim=head_dim, quant_block_size=head_dim
    )
    buf = torch.empty((boundary_page + 2, page_bytes), dtype=torch.uint8, device="cuda")
    # Keep the page table and writer input in the int32 metadata format used by
    # the runtime. Only four pages need initialization in this >2 GiB buffer.
    pages = torch.tensor(
        [[boundary_page + 1, 0], [boundary_page - 1, boundary_page]],
        dtype=torch.int32,
        device="cuda",
    )
    loc = (
        pages.flatten()[:, None] * page_size
        + torch.arange(page_size, dtype=torch.int32, device="cuda")
    ).flatten()
    num_rows = loc.numel()
    keys_cpu = (
        (torch.arange(num_rows * head_dim).reshape(num_rows, head_dim) % 61 - 30)
        .float()
        .div(4)
        .to(torch.float8_e4m3fn)
    )
    scales_cpu = (torch.arange(num_rows, dtype=torch.float32) + 0.5).unsqueeze(1)
    keys, scales = keys_cpu.cuda(), scales_cpu.cuda()
    SetKAndS.execute(pool, buf=buf, loc=loc, index_k=keys, index_k_scale=scales)

    # Check the writer against independently constructed raw pages so a common
    # gather/store address error cannot pass as a successful round trip.
    expected_pages = torch.cat(
        (
            keys_cpu.view(torch.uint8).reshape(4, page_size * head_dim),
            scales_cpu.view(torch.uint8).reshape(4, page_size * 4),
        ),
        dim=1,
    )
    torch.testing.assert_close(
        buf[pages.flatten().long()].cpu(), expected_pages, rtol=0, atol=0
    )
    seq_lens = torch.tensor([93, 128], dtype=torch.int32, device="cuda")
    actual_keys, actual_scales = GetKAndS.execute(
        pool, buf, pages, seq_lens, 93 + 128, 128
    )
    expected_keys = torch.cat((keys_cpu[:93], keys_cpu[128:])).view(torch.uint8)
    expected_scales = torch.cat((scales_cpu[:93], scales_cpu[128:])).view(torch.uint8)
    torch.testing.assert_close(actual_keys.cpu(), expected_keys, rtol=0, atol=0)
    torch.testing.assert_close(actual_scales.cpu(), expected_scales, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
