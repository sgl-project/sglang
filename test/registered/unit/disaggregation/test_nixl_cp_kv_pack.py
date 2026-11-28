"""Byte-exact CP head packing, scratch reuse, and completion failure checks."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang.srt.disaggregation.common.staging_buffer import StagingBuffer
from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b", runner_config="1-gpu-small")


def production_method():
    return NixlKVManager.send_kvcache_slice_packed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("page_size", [1, 64])
@pytest.mark.parametrize("groups,head_group", [(2, 0), (2, 1), (4, 1), (4, 3)])
def test_fragmented_pages_bit_exact(page_size, groups, head_group):
    regions, slots, width = 4, 19, 32
    # Keep every byte observable, including the last token and head shard.
    src = [
        torch.randint(
            0,
            256,
            (slots * page_size, groups * width),
            dtype=torch.uint8,
            device="cuda",
        )
        for _ in range(regions)
    ]
    dst = [
        torch.full((slots * page_size, width), 173, dtype=torch.uint8, device="cuda")
        for _ in range(regions)
    ]
    pages = np.array([8, 1, 12, 4, 18, 3, 9], dtype=np.int64)
    target = np.array([2, 3, 7, 8, 0, 14, 15], dtype=np.int64)
    # Only two pages fit: force reuse four times, including a partial last pack.
    buf = StagingBuffer(
        regions * page_size * width * 2, "cuda", torch.cuda.current_device()
    )
    pending = []
    events = []
    manager = SimpleNamespace(
        kv_args=SimpleNamespace(
            page_size=page_size,
            kv_item_lens=[page_size * groups * width] * regions,
            kv_data_ptrs=[t.data_ptr() for t in src],
        ),
        decode_kv_args_table={
            "peer": SimpleNamespace(
                dst_kv_item_len=page_size * width,
                dst_kv_ptrs=[t.data_ptr() for t in dst],
                gpu_id=0,
            )
        },
        prep_handle_slice_src=(None, groups, regions, slots),
        prep_handles_slice_dst={"peer": (None, slots, head_group)},
        get_mha_kv_ptrs_with_pp=lambda s, d: (s[:2], s[2:], d[:2], d[2:], 2),
    )

    def send(**kw):
        assert not pending, "scratch overwritten before previous write completed"
        assert kw["force_flat"] and kw["bypass_prepped"]
        pending.append(kw)
        events.append("post")
        return len(events)

    def wait(handles, failure_seen):
        assert len(handles) == 1 and not failure_seen
        kw = pending.pop()
        for i, ptr in enumerate(kw["src_data_ptrs"]):
            offset = ptr - buf.get_ptr()
            count = len(kw["prefill_data_indices"])
            packed = buf.buffer[offset : offset + count * page_size * width].view(
                count, page_size, width
            )
            for j, page in enumerate(kw["dst_data_indices"]):
                dst[i].view(slots, page_size, width)[int(page)].copy_(packed[j])
        torch.cuda.synchronize()
        events.append("done")
        return True, False

    manager._send_kvcache_generic = send
    manager._await_handles = wait
    handles = []
    production_method()(manager, "peer", pages, target, buf, handles)
    assert events == ["post", "done"] * 4
    assert len(handles) == 4 and not buf.nixl_poisoned
    for original, actual in zip(src, dst):
        expected = torch.full_like(actual, 173).view(slots, page_size, width)
        original = original.view(slots, page_size, groups * width)
        for s, d in zip(pages, target):
            expected[int(d)] = original[
                int(s), :, head_group * width : (head_group + 1) * width
            ]
        assert torch.equal(actual.view_as(expected), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("outcome", [(False, True), (True, True)])
def test_failed_write_quarantines_scratch(outcome):
    src = torch.arange(64, dtype=torch.uint8, device="cuda").reshape(4, 16)
    buf = StagingBuffer(64, "cuda", torch.cuda.current_device())
    info = SimpleNamespace(dst_kv_item_len=8, dst_kv_ptrs=[123, 456], gpu_id=0)
    manager = SimpleNamespace(
        kv_args=SimpleNamespace(
            page_size=1, kv_item_lens=[16, 16], kv_data_ptrs=[src.data_ptr()] * 2
        ),
        decode_kv_args_table={"p": info},
        prep_handle_slice_src=(None, 2, 2, 4),
        prep_handles_slice_dst={"p": (None, 4, 1)},
        get_mha_kv_ptrs_with_pp=lambda s, d: (s[:1], s[1:], d[:1], d[1:], 1),
        _send_kvcache_generic=lambda **kw: "posted",
        _await_handles=lambda *a, **kw: outcome,
    )
    handles = []
    with pytest.raises(RuntimeError, match="did not complete"):
        production_method()(manager, "p", np.array([2]), np.array([1]), buf, handles)
    assert buf.nixl_poisoned and handles == ["posted"]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
