"""Token relocation must preserve keys in both physical index-cache layouts."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache import index_key_cache
from sglang.srt.mem_cache.index_key_cache import IndexKeyCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("preshuffle", [False, True])
@pytest.mark.parametrize("allocation_page_size", [64, 256])
@pytest.mark.parametrize("source_slots", [[], [1, 2], [15, 16, 63, 64, 127, 255]])
def test_move_preserves_physical_layout(
    monkeypatch, preshuffle, allocation_page_size, source_slots
):
    page_size, head_dim, num_pages = 64, 128, 4
    num_tokens = page_size * num_pages
    monkeypatch.setattr(index_key_cache, "_use_aiter_preshuffle", preshuffle)
    cache = IndexKeyCache.__new__(IndexKeyCache)
    cache.pool = SimpleNamespace(
        page_size=allocation_page_size,
        index_page_size=page_size,
        index_head_dim=head_dim,
        index_kpool=1,
    )
    torch.manual_seed(0)
    keys = torch.randint(0, 256, (num_tokens, head_dim), dtype=torch.uint8)
    scales = (
        torch.arange(1, num_tokens + 1, dtype=torch.float32)
        .view(torch.uint8)
        .reshape(num_tokens, 4)
    )
    slots = torch.arange(num_tokens)
    offsets = slots % page_size
    columns = torch.arange(head_dim)
    if preshuffle:
        # Match the production writer's byte offsets, independently of the move views.
        key_offsets = (
            (offsets[:, None] // 16) * (16 * head_dim)
            + (columns[None, :] // 16) * 256
            + (offsets[:, None] % 16) * 16
            + columns[None, :] % 16
        )
    else:
        key_offsets = offsets[:, None] * head_dim + columns[None, :]
    physical = torch.empty(num_pages, page_size * (head_dim + 4), dtype=torch.uint8)
    physical[slots[:, None] // page_size, key_offsets] = keys
    physical[:, page_size * head_dim :] = scales.reshape(num_pages, -1)
    cache.buffer = [
        physical.view(num_tokens // allocation_page_size, -1),
        torch.empty(0, 1, dtype=torch.uint8),
    ]
    source = torch.tensor(source_slots, dtype=torch.int64)
    target = source.flip(0)
    expected_keys, expected_scales = keys.clone(), scales.clone()
    expected_keys[target] = keys[source]
    expected_scales[target] = scales[source]

    cache.move(target, source)

    torch.testing.assert_close(
        physical[slots[:, None] // page_size, key_offsets], expected_keys
    )
    torch.testing.assert_close(
        physical[:, page_size * head_dim :].reshape(num_tokens, 4), expected_scales
    )
    assert cache.buffer[1].shape[0] == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
