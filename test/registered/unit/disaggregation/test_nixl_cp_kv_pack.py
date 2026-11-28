"""Byte-exact CP head packing, scratch reuse, and completion failure checks."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang.srt.disaggregation.common.staging_buffer import StagingBuffer
from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b", runner_config="1-gpu-small")


def _posted():
    return SimpleNamespace(xfer=object(), result=None, error=None)


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
            "peer": SimpleNamespace(dst_kv_item_len=page_size * width)
        },
        _slice_src=SimpleNamespace(num_groups=groups, num_ptr_pairs=regions),
        _slice_dst={"peer": SimpleNamespace(head_group_idx=head_group)},
        _peers={"peer": SimpleNamespace(keys=SimpleNamespace(kv=list(range(regions))))},
        _staging_keys={buf.get_ptr(): "pack"},
        get_mha_kv_ptrs_with_pp=lambda s, d: (s[:2], s[2:], d[:2], d[2:], 2),
    )
    manager._dcp_pack_sources = lambda *a: NixlKVManager._dcp_pack_sources(manager, *a)

    def send(peer, copies):
        assert not pending, "scratch overwritten before previous write completed"
        pending.append(copies)
        events.append("post")
        return [_posted()]

    def wait(handles, failure_seen):
        assert len(handles) == 1 and not failure_seen
        for copy in pending.pop():
            assert copy.src == "pack"
            for src_off, dst_off, length in copy.ranges.tolist():
                dst[copy.dst].view(-1)[dst_off : dst_off + length].copy_(
                    buf.buffer[src_off : src_off + length]
                )
        torch.cuda.synchronize()
        events.append("done")
        return True, False

    manager._transfer_copies = send
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
    info = SimpleNamespace(dst_kv_item_len=8)
    posted = _posted()
    manager = SimpleNamespace(
        kv_args=SimpleNamespace(
            page_size=1, kv_item_lens=[16, 16], kv_data_ptrs=[src.data_ptr()] * 2
        ),
        decode_kv_args_table={"p": info},
        _slice_src=SimpleNamespace(num_groups=2, num_ptr_pairs=2),
        _slice_dst={"p": SimpleNamespace(head_group_idx=1)},
        _peers={"p": SimpleNamespace(keys=SimpleNamespace(kv=[0, 1]))},
        _dcp_pack_sources=lambda pb, ptrs: [("pack", p - pb.get_ptr()) for p in ptrs],
        get_mha_kv_ptrs_with_pp=lambda s, d: (s[:1], s[1:], d[:1], d[1:], 1),
        _transfer_copies=lambda peer, copies: [posted],
        _await_handles=lambda *a, **kw: outcome,
    )
    handles = []
    with pytest.raises(RuntimeError, match="did not complete"):
        production_method()(manager, "p", np.array([2]), np.array([1]), buf, handles)
    assert buf.nixl_poisoned and handles == [posted]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
