"""Exercise bounded Mooncake MLA packing without a CUDA or RDMA device."""

import ast
import concurrent.futures
import dataclasses
import importlib.util
import sys
from collections import deque
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace

import numpy as np
import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]

spec = importlib.util.spec_from_file_location(
    "transfer_profile",
    ROOT / "python/sglang/srt/disaggregation/common/transfer_profile.py",
)
transfer_profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer_profile)
DCPTransferProfile = transfer_profile.DCPTransferProfile


def _load(path, names, namespace, owner=None):
    tree = ast.parse((ROOT / path).read_text())
    if owner:
        tree = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner
        )
    body = [n for n in tree.body if getattr(n, "name", None) in names]
    exec(
        "from __future__ import annotations\n"
        + ast.unparse(ast.Module(body=body, type_ignores=[])),
        namespace,
    )


@pytest.mark.parametrize("profiled", [False, True])
@pytest.mark.parametrize("custom_pool", [False, True])
@pytest.mark.parametrize("fail_after", [None, 2])
def test_million_token_transfer_reuses_bounded_pack(
    monkeypatch, custom_pool, fail_after, profiled
):
    _exercise(monkeypatch, custom_pool, fail_after, million=True, profiled=profiled)


@pytest.mark.parametrize("rank", range(3))
@pytest.mark.parametrize("capacity", [3, 7, 100])
def test_pack_chunks_preserve_fragmented_target_and_draft_addresses(
    monkeypatch, rank, capacity
):
    _exercise(monkeypatch, False, None, rank=rank, capacity=capacity)


def _exercise(
    monkeypatch,
    custom_pool,
    fail_after,
    *,
    million=False,
    rank=0,
    capacity=3,
    profiled=False,
):
    namespace = dict(np=np, dataclasses=dataclasses, concurrent=concurrent, deque=deque)
    _load(
        "python/sglang/srt/disaggregation/common/utils.py",
        {
            "DCPTokenTransferPlan",
            "build_dcp_token_transfer_plan",
            "group_concurrent_contiguous",
        },
        namespace,
    )
    _load(
        "python/sglang/srt/disaggregation/utils.py",
        {"build_transfer_entry_pairs", "resolve_dcp_dst_entry_indices"},
        namespace,
    )
    _load(
        "python/sglang/srt/disaggregation/mooncake/conn.py",
        {"send_kvcache_dcp", "_await_transfer_futures"},
        namespace,
        "MooncakeKVManager",
    )
    page, dcp = (64, 8) if million else (2, 3)
    n = 1024000 if million else 37
    capacity = 8192 if million else capacity
    offset, prefix = (0, 0) if million else (1, page * dcp)
    draft = 0 if million else 1
    item_lens = [8, 12] + ([4] if draft else [])
    src_ptrs = [10**8 * (i + 1) for i in range(len(item_lens))]
    # Exercise non-contiguous layer IDs and reordered decode registration.
    layer_ids = [2, 9] + ([20] if draft else [])
    dst_ids = list(reversed(layer_ids))
    raw_dst_ptrs = [10**9 * (i + 1) for i in range(len(item_lens))]
    dst_ptrs = list(reversed(raw_dst_ptrs))
    src = np.arange((n + page - 1) // page, dtype=np.int32) * 3 + 5
    dst = (
        np.arange((offset * page + n + page * dcp - 1) // (page * dcp), dtype=np.int32)
        * 2
        + 7
    )
    pack_base = 10**12
    pending = {}
    packs, calls, copied = [], [], []
    packed_rows = None
    packed_ptrs = None

    def pack(**kwargs):
        nonlocal packed_rows, packed_ptrs
        # A pack may be overwritten only after all layer writes complete.
        assert not pending
        rows = kwargs["src_token_indices"]
        assert len(rows) <= capacity, "Whole long prefix exceeds the fixed pack buffer"
        packs.append(rows.copy())
        packed_rows = rows.copy()
        sizes = [len(rows) * size for size in item_lens[:2]]
        packed_ptrs = [pack_base, pack_base + sizes[0]]
        pending.update({i: sizes[i] for i in range(2)})
        return packed_ptrs, np.arange(len(rows), dtype=np.int64)

    module = ModuleType("sglang.srt.disaggregation.common.dcp_pack")
    module.try_pack_dcp_src = pack
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def transfer(session, blocks):
        assert session == "peer"
        assert len(blocks) <= capacity * len(item_lens) * dcp
        calls.append(len(blocks))
        for source, dest, size in blocks:
            entry = next(
                i for i, ptr in enumerate(dst_ptrs) if ptr <= dest < ptr + 10**8
            )
            width = item_lens[entry]
            if entry < 2:
                row = (source - packed_ptrs[entry]) // width
                pending[entry] -= size
                if not pending[entry]:
                    del pending[entry]
                if not million:
                    copied.extend(
                        (src_ptrs[entry] + int(token) * width, dest + j * width, width)
                        for j, token in enumerate(
                            packed_rows[row : row + size // width]
                        )
                    )
            elif not million:
                copied.extend(
                    (source + j, dest + j, width) for j in range(0, size, width)
                )
        return -1 if fail_after == len(calls) else 0

    manager = SimpleNamespace(
        kv_args=SimpleNamespace(
            page_size=page,
            kv_layer_ids=layer_ids,
            kv_data_ptrs=src_ptrs,
            num_draft_entries=draft,
        ),
        enable_custom_mem_pool=custom_pool,
        enable_deferred_decode_kv_release=True,
        _transfer_data=transfer,
    )
    for name in ("send_kvcache_dcp", "_await_transfer_futures"):
        setattr(manager, name, MethodType(namespace[name], manager))
    # One worker keeps the fake engine deterministic while exercising futures.
    profile = (
        DCPTransferProfile(
            room=1,
            worker=0,
            source_rank=0,
            chunk=SimpleNamespace(
                index_slice=slice(0, 1), num_kv_tokens=n, is_last_chunk=True
            ),
            enqueued_at=0,
        )
        if profiled
        else None
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        result = manager.send_kvcache_dcp(
            "peer",
            src,
            raw_dst_ptrs,
            dst,
            dcp_token_item_lens=item_lens,
            dst_dcp_size=dcp,
            dst_dcp_rank=rank,
            src_page_offset=offset,
            decode_prefix_len=prefix,
            num_kv_tokens=n,
            executor=executor,
            dst_layer_ids=dst_ids,
            transfer_profile=profile,
            pack_buffer=SimpleNamespace(get_size=lambda: capacity * sum(item_lens[:2])),
        )
    if fail_after is not None:
        assert result == -1
        assert len(packs) == (1 if custom_pool else fail_after)
        return
    assert result == 0
    if profile is not None:
        assert profile.values["mla_calls"] == len(calls)
        assert profile.values["mla_bytes"] == sum(len(rows) for rows in packs) * sum(
            item_lens
        )
        assert profile.values["mla_pack_s"] >= 0
        assert profile.values["mla_submit_s"] >= 0
    assert not pending
    expected = namespace["build_dcp_token_transfer_plan"](
        src,
        dst,
        physical_page_size=page,
        dcp_size=dcp,
        dcp_rank=rank,
        src_page_offset=offset,
        decode_prefix_len=prefix,
        num_kv_tokens=n,
    )
    np.testing.assert_array_equal(
        np.concatenate(packs), expected.target_src_token_indices
    )
    if not million:
        reference = []
        for i, width in enumerate(item_lens):
            source = (
                expected.target_src_token_indices
                if i < 2
                else expected.draft_src_token_indices
            )
            dest = (
                expected.target_dst_token_indices
                if i < 2
                else expected.draft_dst_token_indices
            )
            reference.extend(
                (src_ptrs[i] + int(s) * width, dst_ptrs[i] + int(d) * width, width)
                for s, d in zip(source, dest)
            )
        assert sorted(copied) == sorted(reference)
