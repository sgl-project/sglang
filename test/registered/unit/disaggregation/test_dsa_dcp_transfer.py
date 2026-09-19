"""Verify the actual PD byte-copy plan, including the separate scale region."""

import ast
import dataclasses
import importlib.util
import logging
import sys
from collections import deque
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location(
    "dsa_dcp_transfer", ROOT / "python/sglang/srt/disaggregation/common/dsa_dcp.py"
)
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)

profile_spec = importlib.util.spec_from_file_location(
    "transfer_profile",
    ROOT / "python/sglang/srt/disaggregation/common/transfer_profile.py",
)
profile_module = importlib.util.module_from_spec(profile_spec)
profile_spec.loader.exec_module(profile_module)


def _load_function(path, name, namespace, owner=None):
    tree = ast.parse((ROOT / path).read_text())
    if owner:
        tree = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner
        )
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    exec("from __future__ import annotations\n" + ast.unparse(function), namespace)
    return namespace[name]


def _bootstrap_manager(cp_size=8, layer_split=True, rank=0, dcp_size=8):
    data = dict(
        attn_tp_size=8 // cp_size,
        attn_cp_size=cp_size,
        dp_size=1,
        pp_size=1,
        page_size=64,
        kv_cache_dtype="fp8_e4m3",
        follow_bootstrap_room=False,
    )
    if layer_split is not None:
        data["enable_dsa_cache_layer_split"] = layer_split
    path = "python/sglang/srt/disaggregation/common/conn.py"
    namespace = dict(dataclasses=dataclasses, logger=logging.getLogger(__name__))
    tree = ast.parse((ROOT / path).read_text())
    info_class = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "PrefillServerInfo"
    )
    exec("from __future__ import annotations\n" + ast.unparse(info_class), namespace)
    calls = []

    def get(url, timeout):
        calls.append(url)
        return SimpleNamespace(status_code=200, json=lambda: data)

    namespace["requests"] = SimpleNamespace(get=get)
    resolve = _load_function(
        path, "_resolve_rank_mapping", namespace, "CommonKVManager"
    )
    ensure = _load_function(
        path, "try_ensure_parallel_info", namespace, "CommonKVManager"
    )
    manager = SimpleNamespace(
        prefill_info_table={},
        kv_args=SimpleNamespace(page_size=64, engine_rank=rank),
        kv_cache_dtype_str="fp8_e4m3",
        dcp_size=dcp_size,
        is_mla_backend=True,
        is_hybrid_mla_backend=False,
        attn_tp_size=8,
        attn_cp_size=1,
        attn_cp_rank=0,
        pp_size=1,
        pp_rank=0,
        enable_all_cp_ranks_for_transfer=False,
    )
    manager._resolve_rank_mapping = lambda info: resolve(manager, info)
    manager.try_ensure_parallel_info = lambda addr: ensure(manager, addr)
    return manager, calls


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("dcp_size", [1, 8])
@pytest.mark.parametrize("cp_size,layer_split", [(1, None), (8, True)])
def test_dcp_bootstrap_rank_mapping(rank, dcp_size, cp_size, layer_split):
    manager, calls = _bootstrap_manager(
        cp_size=cp_size,
        layer_split=layer_split,
        rank=rank,
        dcp_size=dcp_size,
    )
    assert manager.try_ensure_parallel_info("prefill:8998")
    info = manager.prefill_info_table["prefill:8998"]
    assert info.target_tp_ranks == [rank // cp_size]
    assert info.target_cp_ranks == list(range(cp_size))
    assert info.target_pp_ranks == [0]
    assert info.required_prefill_response_num == cp_size
    assert info.required_dst_info_num == cp_size
    assert manager.try_ensure_parallel_info("prefill:8998")
    assert len(calls) == 1


@pytest.mark.parametrize("layer_split", [False, None])
def test_dcp_bootstrap_rejects_cp_without_layer_split(layer_split):
    manager, _ = _bootstrap_manager(layer_split=layer_split)
    with pytest.raises(RuntimeError, match="--enable-dsa-cache-layer-split"):
        manager.try_ensure_parallel_info("prefill:8998")
    assert not manager.prefill_info_table


@pytest.mark.parametrize("backend_name", ["mooncake", "nixl"])
def test_transport_dispatch_pairs_layer_split_entries(backend_name):
    # Source CP rank owns layers 2 and 3; layer 3 shares top-k and has no
    # index buffer. Decode registers all four layers.
    namespace = dict(
        deque=deque,
        StateType=SimpleNamespace(DSA="dsa", MAMBA="mamba", DSA_TAIL="tail"),
        build_dsa_dcp_transfer_blocks=transfer.build_dsa_dcp_transfer_blocks,
        iter_dsa_dcp_transfer_batches=transfer.iter_dsa_dcp_transfer_batches,
    )
    utils = "python/sglang/srt/disaggregation/utils.py"
    for name in ["build_transfer_entry_pairs", "resolve_dcp_dst_entry_indices"]:
        _load_function(utils, name, namespace)
    send = _load_function(
        f"python/sglang/srt/disaggregation/{backend_name}/conn.py",
        "maybe_send_extra",
        namespace,
        "MooncakeKVManager" if backend_name == "mooncake" else "NixlKVManager",
    )
    kv_args = SimpleNamespace(
        state_types=["dsa"],
        state_data_ptrs=[[100000, 0]],
        state_item_lens=[[8448, 0]],
        state_dim_per_tensor=[[]],
        state_layer_ids=[[2, 3]],
        page_size=64,
        gpu_id=0,
    )
    actual_blocks = []
    manager = SimpleNamespace(kv_args=kv_args)
    destination = SimpleNamespace(
        dst_state_data_ptrs=[[200000, 300000, 400000, 0]],
        dst_state_item_lens=[[8448, 8448, 8448, 0]],
        dst_state_dim_per_tensor=[[]],
        dst_state_layer_ids=[[0, 1, 2, 3]],
        requires_dcp_relayout=True,
        dst_dcp_size=4,
        dst_dcp_rank=1,
    )
    if backend_name == "mooncake":
        manager._transfer_data = lambda session, blocks: (
            actual_blocks.extend(blocks) or 0
        )
        assert (
            send(
                manager,
                SimpleNamespace(dst_state_indices=[[7]], mooncake_session_id="peer"),
                [[3, 8]],
                None,
                destination,
            )
            == 0
        )
    else:

        def initialize(operation, src, dst, peer, notification):
            actual_blocks.extend((s[0], d[0], s[1]) for s, d in zip(src, dst))
            return 123

        manager.agent = SimpleNamespace(
            get_xfer_descs=lambda addrs, kind: addrs,
            initialize_xfer=initialize,
            transfer=lambda handle: "DONE",
        )
        assert send(
            manager,
            "peer",
            [[3, 8]],
            destination.dst_state_data_ptrs,
            [[7]],
            4,
            "test",
            4,
            dst_state_item_lens=destination.dst_state_item_lens,
            dst_state_layer_ids=destination.dst_state_layer_ids,
            dst_dcp_size=4,
            dst_dcp_rank=1,
            requires_dcp_relayout=True,
        ) == [123]
    expected = transfer.build_dsa_dcp_transfer_blocks(
        [100000],
        [400000],
        [8448],
        [8448],
        [3, 8],
        [7],
        page_size=64,
        dcp_size=4,
        dcp_rank=1,
    )
    assert actual_blocks == expected


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("profiled", [False, True])
@pytest.mark.parametrize("fail_after", [None, 2])
def test_mooncake_million_token_dsa_transfer_is_bounded(
    monkeypatch, fail_after, profiled, packed
):
    namespace = dict(
        vars(transfer),
        deque=deque,
        StateType=SimpleNamespace(DSA="dsa", MAMBA="mamba", DSA_TAIL="tail"),
    )
    utils = "python/sglang/srt/disaggregation/utils.py"
    for name in ["build_transfer_entry_pairs", "resolve_dcp_dst_entry_indices"]:
        _load_function(utils, name, namespace)
    send = _load_function(
        "python/sglang/srt/disaggregation/mooncake/conn.py",
        "maybe_send_extra",
        namespace,
        "MooncakeKVManager",
    )
    manager = SimpleNamespace(
        kv_args=SimpleNamespace(
            state_types=["dsa"],
            state_data_ptrs=[[100000]],
            state_item_lens=[[8448]],
            state_dim_per_tensor=[[]],
            state_layer_ids=[[2]],
            page_size=64,
        )
    )
    batch_sizes = []
    transferred_bytes = 0
    pack_calls = []
    if packed:

        def pack(plan, buffer, dcp_size, dcp_rank):
            # No buffer overwrite until the previous synchronous send returns.
            assert len(pack_calls) == len(batch_sizes)
            assert plan.packed_bytes <= buffer.get_size()
            pack_calls.append(plan)

        pack_namespace = dict(
            iter_dsa_dcp_pack_plans=transfer.iter_dsa_dcp_pack_plans, pack_dsa_plan=pack
        )
        packed_iterator = _load_function(
            "python/sglang/srt/disaggregation/common/dsa_pack.py",
            "iter_packed_dsa_transfer_batches",
            pack_namespace,
        )
        packed_module = ModuleType("sglang.srt.disaggregation.common.dsa_pack")
        packed_module.iter_packed_dsa_transfer_batches = packed_iterator
        monkeypatch.setitem(sys.modules, packed_module.__name__, packed_module)
    pack_buffer = (
        SimpleNamespace(get_size=lambda: 128 * 8448, get_ptr=lambda: 3000000)
        if packed
        else None
    )

    def submit(session, blocks):
        nonlocal transferred_bytes
        # Exercise the real state-transfer dispatch without requiring an RDMA
        # device. The limit guards against a single unbounded engine call.
        assert 0 < len(blocks) <= 16384
        batch_sizes.append(len(blocks))
        transferred_bytes += sum(block[2] for block in blocks)
        return -1 if len(batch_sizes) == fail_after else 0

    manager._transfer_data = submit
    destination = SimpleNamespace(
        dst_state_data_ptrs=[[200000]],
        dst_state_item_lens=[[8448]],
        dst_state_dim_per_tensor=[[]],
        dst_state_layer_ids=[[2]],
        requires_dcp_relayout=True,
        dst_dcp_size=8,
        dst_dcp_rank=7,
    )
    profile = (
        profile_module.DCPTransferProfile(
            room=1,
            worker=0,
            source_rank=0,
            chunk=SimpleNamespace(
                index_slice=slice(0, 16000), num_kv_tokens=1024000, is_last_chunk=True
            ),
            enqueued_at=0,
        )
        if profiled
        else None
    )
    result = send(
        manager,
        SimpleNamespace(
            dst_state_indices=[np.arange(1, 2001)], mooncake_session_id="peer"
        ),
        [np.arange(1, 16001)],
        None,
        destination,
        transfer_profile=profile,
        pack_buffer=pack_buffer,
    )
    if packed:
        assert len(pack_calls) == len(batch_sizes)
    if profile is not None:
        assert profile.values["dsa_bytes"] == transferred_bytes
        assert profile.values["dsa_blocks"] == sum(batch_sizes)
        assert profile.values["dsa_calls"] == len(batch_sizes)
        assert profile.values["dsa_failures"] == bool(fail_after)
        assert profile.values["dsa_build_s"] >= 0
    if fail_after:
        assert result == -1
        assert len(batch_sizes) == fail_after
    else:
        assert result == 0
        assert sum(batch_sizes) == (16 if packed else 256000)
        assert transferred_bytes == 128000 * 132


@pytest.mark.parametrize("size", [2, 3, 4, 8])
def test_batched_dsa_transfer_preserves_addresses_and_scales(size):
    rng = np.random.default_rng(42)
    # Multiple batches, fragmented pages, a partial final destination page,
    # two owned layers, and a layer without an independent index cache.
    src_pages = rng.permutation(50)[: 4 * size + 1] + 1
    dst_pages = np.array([6, 3, 8, 1, 4])
    args = (
        [100000, 0, 900000],
        [1700000, 0, 2400000],
        [8448, 0, 8448],
        [8448, 0, 8448],
        src_pages,
        dst_pages,
    )
    for rank in range(size):
        kwargs = dict(page_size=64, dcp_size=size, dcp_rank=rank)
        expected = transfer.build_dsa_dcp_transfer_blocks(*args, **kwargs)
        batches = list(
            transfer.iter_dsa_dcp_transfer_batches(
                *args, **kwargs, max_dst_pages_per_batch=2
            )
        )
        assert all(0 < len(batch) <= 256 for batch in batches)
        assert sorted(block for batch in batches for block in batch) == sorted(expected)


@pytest.mark.parametrize("size", [2, 3, 4, 8])
@pytest.mark.parametrize("length", [1, 63, 64, 65, 257, 513])
def test_keys_and_scales_relayout(size, length):
    page_size, page_bytes = 64, 64 * 132
    src_pages = np.array([9, 3, 8, 1, 7, 2, 6, 4, 5])[: (length + 63) // 64]
    dst_pages = np.array([4, 1, 3, 2, 5])[: (length + 64 * size - 1) // (64 * size)]
    src = np.arange(10 * page_bytes, dtype=np.int64).astype(np.uint8)
    # Each scale also varies by token, so copying keys correctly is insufficient.
    for pos in range(length):
        page, slot = src_pages[pos // 64], pos % 64
        src[
            page * page_bytes + 8192 + slot * 4 : page * page_bytes
            + 8192
            + slot * 4
            + 4
        ] = np.array([pos, pos // 256, 17, 253]) % 256
    for rank in range(size):
        dst = np.full(6 * page_bytes, 0xAC, dtype=np.uint8)
        blocks = transfer.build_dsa_dcp_transfer_blocks(
            [0],
            [0],
            [page_bytes],
            [page_bytes],
            src_pages,
            dst_pages,
            page_size=page_size,
            dcp_size=size,
            dcp_rank=rank,
        )
        for source, target, count in blocks:
            dst[target : target + count] = src[source : source + count]
        for pos in range(rank, length, size):
            sp, ss = src_pages[pos // 64], pos % 64
            local = pos // size
            dp, ds = dst_pages[local // 64], local % 64
            for width, base in [(128, 0), (4, 8192)]:
                source = sp * page_bytes + base + ss * width
                target = dp * page_bytes + base + ds * width
                np.testing.assert_array_equal(
                    dst[target : target + width], src[source : source + width]
                )
        # Reserved sink page is never used by PD writes.
        assert (dst[:page_bytes] == 0xAC).all()


def test_empty_index_layers_and_invalid_geometry():
    kwargs = dict(page_size=64, dcp_size=4, dcp_rank=2)
    assert (
        transfer.build_dsa_dcp_transfer_blocks([0], [0], [0], [0], [1], [2], **kwargs)
        == []
    )
    with pytest.raises(ValueError, match="too few"):
        transfer.build_dsa_dcp_transfer_blocks(
            [0], [0], [8448], [8448], [1, 2, 3, 4, 5], [2], **kwargs
        )
    with pytest.raises(ValueError, match="128-byte"):
        transfer.build_dsa_dcp_transfer_blocks(
            [0], [0], [8448], [8449], [1], [2], **kwargs
        )


@pytest.mark.parametrize("size", [2, 3, 4, 8])
@pytest.mark.parametrize("capacity_pages", [1, 3, 2048])
def test_packed_dsa_bytes_match_legacy_with_partial_pages(size, capacity_pages):
    rng = np.random.default_rng(12)
    page_bytes = 8448
    src_pages = rng.permutation(40)[: size * 3 + 1]
    dst_pages = np.array([9, 4, 5, 11])
    src_ptrs, dst_ptrs = [100000, 700000], [1300000, 1900000]
    source = {
        p: rng.integers(0, 256, size=40 * page_bytes, dtype=np.uint8) for p in src_ptrs
    }
    pack_ptr = 3000000
    for rank in range(size):
        expected = {p: np.full(12 * page_bytes, 173, dtype=np.uint8) for p in dst_ptrs}
        actual = {p: a.copy() for p, a in expected.items()}
        old = transfer.build_dsa_dcp_transfer_blocks(
            src_ptrs,
            dst_ptrs,
            [page_bytes] * 2,
            [page_bytes] * 2,
            src_pages,
            dst_pages,
            page_size=64,
            dcp_size=size,
            dcp_rank=rank,
        )
        for src, dst, n in old:
            layer = 0 if dst < dst_ptrs[1] else 1
            expected[dst_ptrs[layer]][
                dst - dst_ptrs[layer] : dst - dst_ptrs[layer] + n
            ] = source[src_ptrs[layer]][
                src - src_ptrs[layer] : src - src_ptrs[layer] + n
            ]
        plans = transfer.iter_dsa_dcp_pack_plans(
            src_ptrs,
            dst_ptrs,
            [page_bytes] * 2,
            [page_bytes] * 2,
            src_pages,
            dst_pages,
            page_size=64,
            dcp_size=size,
            dcp_rank=rank,
            pack_ptr=pack_ptr,
            pack_bytes=capacity_pages * page_bytes,
        )
        for plan in plans:
            assert plan.packed_bytes <= capacity_pages * page_bytes
            packed = np.full(plan.packed_bytes, 231, dtype=np.uint8)
            # CPU reference of the gather, independent of RDMA descriptor grouping.
            for local, offset in enumerate(range(rank, len(plan.src_pages) * 64, size)):
                sp, slot = divmod(offset, 64)
                dp, ds = divmod(local, 64)
                for width, base in ((128, 0), (4, 8192)):
                    s = int(plan.src_pages[sp]) * page_bytes + base + slot * width
                    d = dp * page_bytes + base + ds * width
                    packed[d : d + width] = source[plan.src_ptr][s : s + width]
            for src, dst, n in plan.blocks:
                layer = 0 if dst < dst_ptrs[1] else 1
                assert pack_ptr <= src < src + n <= pack_ptr + plan.packed_bytes
                actual[dst_ptrs[layer]][
                    dst - dst_ptrs[layer] : dst - dst_ptrs[layer] + n
                ] = packed[src - pack_ptr : src - pack_ptr + n]
        for ptr in dst_ptrs:
            np.testing.assert_array_equal(actual[ptr], expected[ptr])


def test_million_token_packed_dsa_descriptor_and_memory_bounds():
    # Match the slow CP7 trace: nine layers, eight destinations, 1,024,000 tokens.
    count = total_bytes = batches = 0
    for rank in range(8):
        plans = transfer.iter_dsa_dcp_pack_plans(
            list(range(9)),
            list(range(9)),
            [8448] * 9,
            [8448] * 9,
            np.arange(16000),
            np.arange(2000),
            page_size=64,
            dcp_size=8,
            dcp_rank=rank,
            pack_ptr=1000000,
            pack_bytes=48365568,
        )
        for plan in plans:
            assert plan.packed_bytes <= 2048 * 8448
            assert plan.src_pages.nbytes <= 128 * 1024
            batches += 1
            count += len(plan.blocks)
            total_bytes += sum(b[2] for b in plan.blocks)
    assert total_bytes == 1216512000
    assert batches == 72
    assert (
        count == 72
    )  # Contiguous destination pages coalesce into one write/layer/peer.
