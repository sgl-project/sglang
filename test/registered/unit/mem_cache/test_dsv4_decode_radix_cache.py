"""Unit tests for the DeepSeek-V4 additions to the decode-side radix cache.

The decode radix cache lifecycle (match / transfer / insert / evict) is the
shared one from PR #27770. DSV4 compressed KV follows the FULL slots and C4
state follows the SWA slots, so neither needs its own bookkeeping. With
--enable-hierarchical-cache the same prefix may additionally come back from
host (L2) or storage (L3).

Usage:
    python -m pytest test/registered/unit/mem_cache/test_dsv4_decode_radix_cache.py -v
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.disaggregation.decode import alloc_for_decode_prealloc
from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCachePreallocMixin,
    DecodeHiCacheTransferMixin,
    DecodePrefixMatch,
    HiCacheRestoreResult,
)
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _prefix_match(l1: int, l2: int = 0, l3: int = 0) -> DecodePrefixMatch:
    return DecodePrefixMatch(
        prefix_indices=torch.arange(l1, dtype=torch.int64),
        l2_host_hit_length=l2,
        l3_storage_hit_length=l3,
        last_device_node=None,
        last_host_node=object(),
    )


def test_cap_restore_below_l1_drops_the_restore_and_trims_the_device_slice():
    # A cap below the device hit (the SWA window start can be there when the L1
    # match reaches into the live window): the P-visible slice shrinks and the
    # L2/L3 restore tiers drop entirely. The raw L3 candidate remains anchored
    # for the D-local prefetch.
    prefix_match = _prefix_match(l1=512, l2=256, l3=256)
    assert prefix_match.decode_prefix_len == 1024

    prefix_match.cap_restore(384)

    assert prefix_match.l1_prefix_len == 384
    assert prefix_match.l2_host_hit_length == 0
    assert prefix_match.l3_storage_hit_length == 0
    assert prefix_match.decode_prefix_len == 384
    assert prefix_match.restore_token_count == 0
    assert not prefix_match.needs_local_restore
    assert prefix_match.raw_l3_storage_hit_length == 256
    assert prefix_match.raw_l3_match_start == 768
    assert prefix_match.last_host_node is not None


def test_cap_restore_trims_storage_before_host():
    # The window cap keeps the part of the restore below the window start, and
    # trims L3 first so the surviving tiers stay contiguous from the L1 prefix.
    prefix_match = _prefix_match(l1=512, l2=256, l3=256)

    prefix_match.cap_restore(768)
    assert prefix_match.l2_host_hit_length == 256
    assert prefix_match.l3_storage_hit_length == 0
    assert prefix_match.raw_l3_storage_hit_length == 256
    assert prefix_match.decode_prefix_len == 768
    assert prefix_match.restore_token_count == 256

    prefix_match.cap_restore(640)
    assert prefix_match.l2_host_hit_length == 128
    assert prefix_match.decode_prefix_len == 640

    # A cap at or below the device prefix drops the restore entirely.
    prefix_match.cap_restore(512)
    assert prefix_match.decode_prefix_len == 512
    assert not prefix_match.needs_local_restore
    assert prefix_match.last_host_node is not None


def test_cap_restore_is_a_noop_above_the_hit():
    prefix_match = _prefix_match(l1=512, l2=256, l3=256)
    prefix_match.cap_restore(2048)
    assert prefix_match.decode_prefix_len == 1024
    # No trimming happened, so the host anchor survives for the prefetch.
    assert prefix_match.last_host_node is not None


def test_swa_tail_prealloc_extends_from_the_committed_prefix():
    # The [prefix_len, total_prefix_len) gap is restored from host, so the
    # full-attention allocation must start at total_prefix_len -- otherwise the
    # page count derived from prefix_lens does not match extend_num_tokens.
    captured = {}
    page_size = 256

    def alloc_extend_swa_tail(**kwargs):
        captured.update(kwargs)
        return torch.arange(kwargs["extend_num_tokens"], dtype=torch.int64)

    allocator = SimpleNamespace(
        page_size=page_size,
        device="cpu",
        alloc_extend_swa_tail=alloc_extend_swa_tail,
    )
    req = SimpleNamespace(kv=ReqKvInfo())
    fill_len, prefix_len, total_prefix_len = 2048, 512, 1024

    alloc_for_decode_prealloc(
        allocator,
        req=req,
        fill_len=fill_len,
        delta_len=fill_len - total_prefix_len,
        prefix_len=prefix_len,
        total_prefix_len=total_prefix_len,
        prefix_indices=torch.arange(prefix_len, dtype=torch.int64),
        uses_swa_tail=True,
        swa_tail_len=page_size,
        req_to_token_pool=None,
    )

    assert captured["prefix_lens_cpu"].tolist() == [total_prefix_len]
    assert captured["extend_num_tokens"] == fill_len - total_prefix_len


def test_l3_prefetch_failure_degrades_to_l2_only():
    """Verify a prefetch setup failure clears the L3 tier of the contract."""
    mixin = DecodeHiCachePreallocMixin()
    prefix_match = _prefix_match(l1=2, l2=2, l3=2)
    req = SimpleNamespace(
        rid="req-1",
        origin_input_ids=list(range(8)),
        extra_key=None,
        cache_salt=None,
    )
    node = SimpleNamespace(get_last_hash_value=lambda: None)
    tree_cache = MagicMock()
    tree_cache.hicache_storage_pass_prefix_keys = False
    tree_cache.resolve_node_handle.return_value = node
    tree_cache.ongoing_prefetch = {}

    def partially_register_then_fail(*args, **kwargs):
        tree_cache.ongoing_prefetch[req.rid] = object()
        raise RuntimeError("prefetch setup failed")

    tree_cache.prefetch_from_storage.side_effect = partially_register_then_fail
    mixin.tree_cache = tree_cache

    mixin._start_hicache_prefetch(req, prefix_match)

    assert prefix_match.l3_storage_hit_length == 0
    assert prefix_match.decode_prefix_len == 4
    assert not prefix_match.prefetch_registered


def test_l3_prefetch_uses_raw_candidate_after_visible_cap():
    """The D-local L3 request must not inherit the P-visible CAP."""
    mixin = DecodeHiCachePreallocMixin()
    prefix_match = _prefix_match(l1=2, l2=2, l3=2)
    prefix_match.cap_restore(5)
    prefix_match.last_host_node = "host-anchor"
    req = SimpleNamespace(
        rid="req-raw-l3",
        origin_input_ids=list(range(8)),
        extra_key=None,
        cache_salt=None,
    )
    node = SimpleNamespace(
        get_last_hash_value=lambda: "last-hash",
        parent=None,
    )
    tree_cache = MagicMock()
    tree_cache.hicache_storage_pass_prefix_keys = False
    tree_cache.resolve_node_handle.return_value = node
    tree_cache.ongoing_prefetch = {req.rid: object()}
    mixin.tree_cache = tree_cache

    mixin._start_hicache_prefetch(req, prefix_match)

    args = tree_cache.prefetch_from_storage.call_args.args
    assert args[0] == req.rid
    assert args[1] == prefix_match.last_host_node
    assert args[2] == [4, 5]
    assert prefix_match.prefetch_registered


def test_prefetch_only_match_is_drained_without_load_back():
    """A fully capped P contract still waits for and consumes D-local L3 IO."""
    mixin = DecodeHiCacheTransferMixin()
    prefix_match = _prefix_match(l1=4, l3=0)
    prefix_match.raw_l3_storage_hit_length = 2
    prefix_match.prefetch_registered = True
    decode_req = SimpleNamespace(
        req=SimpleNamespace(rid="req-prefetch-only"),
        prefix_match=prefix_match,
        hicache_restore_status=HiCacheRestoreResult.PENDING,
    )
    tree_cache = MagicMock()
    tree_cache.check_prefetch_progress.return_value = True
    tree_cache.pop_prefetch_loaded_tokens.return_value = 2
    mixin.tree_cache = tree_cache

    assert not mixin._try_hicache_queue_load_back(decode_req)

    assert decode_req.hicache_restore_status == HiCacheRestoreResult.READY
    tree_cache.check_prefetch_progress.assert_called_once_with("req-prefetch-only")
    tree_cache.pop_prefetch_loaded_tokens.assert_called_once_with("req-prefetch-only")
    tree_cache.init_load_back.assert_not_called()


def test_hicache_rematch_restores_admission_lock_owner():
    """Verify HiCache rematch preserves the admission lock until commit."""
    mixin = DecodeHiCacheTransferMixin()
    admission_node = object()
    rematch_node = object()
    restored_node = object()
    req = SimpleNamespace(
        origin_input_ids=[1, 2, 3, 4],
        output_ids=[],
        extra_key=None,
        cache_salt=None,
        prefix_indices=torch.tensor([10, 11], dtype=torch.int64),
        last_node=admission_node,
        last_host_node="admission-host",
        best_match_node="admission-best",
        host_hit_length=2,
        swa_host_hit_length=3,
        mamba_host_hit_length=4,
        num_matched_prefix_tokens=2,
        swa_branching_seqlen=None,
        mamba_branching_seqlen=5,
        kv=ReqKvInfo(cache_protected_len=2),
        swa_prefix_lock_released=False,
        lock_receipt=DecLockRefParams(),
        rid="req-rematch",
        _compute_max_prefix_len=lambda length: length,
    )
    match_result = SimpleNamespace(
        device_indices=torch.tensor([20, 21, 22], dtype=torch.int64),
        last_device_node=rematch_node,
        last_host_node="rematch-host",
        best_match_node="rematch-best",
        host_hit_length=6,
        swa_host_hit_length=7,
        swa_branching_seqlen=None,
        mamba_host_hit_length=8,
        mamba_branching_seqlen=9,
        cache_protected_len=3,
    )
    tree_cache = MagicMock()
    tree_cache.swa_reprefill_tail_tokens.return_value = 0
    tree_cache.match_prefix.return_value = match_result
    init_load_back_last_node = {}

    def init_load_back(params):
        init_load_back_last_node["value"] = params.req.last_node
        return torch.empty(0, dtype=torch.int64), restored_node

    tree_cache.init_load_back.side_effect = init_load_back
    restored_receipt = DecLockRefParams()
    tree_cache.inc_lock_ref.return_value = SimpleNamespace(
        to_dec_params=lambda: restored_receipt,
    )
    mixin.tree_cache = tree_cache
    prefix_match = _prefix_match(l1=2, l2=1)
    prefix_match.last_device_node = admission_node
    decode_req = SimpleNamespace(
        req=req,
        prefix_match=prefix_match,
        hicache_restored_kv_indices=None,
        hicache_restored_node=None,
        hicache_restore_lock_receipt=None,
        hicache_restore_status=None,
    )

    mixin._try_hicache_queue_load_back(decode_req)

    assert init_load_back_last_node["value"] is rematch_node
    assert req.last_node is admission_node
    assert decode_req.hicache_restored_node is restored_node


def test_restore_commit_transfers_owner_and_clears_temporary_node():
    """Verify restore commit transfers ownership and clears temporary state."""
    mixin = DecodeHiCacheTransferMixin()
    admission_node = object()
    restored_node = object()
    prefix_match = DecodePrefixMatch(
        prefix_indices=torch.tensor([10, 11], dtype=torch.int64),
        l2_host_hit_length=2,
        l3_storage_hit_length=0,
        last_device_node=admission_node,
    )
    admission_receipt = DecLockRefParams()
    restored_receipt = DecLockRefParams()
    req = SimpleNamespace(
        kv=ReqKvInfo(req_pool_idx=0),
        prefix_indices=prefix_match.prefix_indices,
        last_node=admission_node,
        lock_receipt=admission_receipt,
        swa_prefix_lock_released=True,
    )
    decode_req = SimpleNamespace(
        req=req,
        prefix_match=prefix_match,
        hicache_restored_kv_indices=torch.tensor([20, 21], dtype=torch.int64),
        hicache_restored_node=restored_node,
        hicache_restore_lock_receipt=restored_receipt,
    )
    tree_cache = MagicMock()
    mixin.tree_cache = tree_cache

    mixin._commit_hicache_local_restore_to_req(decode_req)

    tree_cache.req_to_token_pool.write.assert_called_once()
    tree_cache.dec_lock_ref.assert_called_once_with(
        admission_node,
        admission_receipt,
        skip_swa=True,
    )
    assert req.prefix_indices.tolist() == [10, 11, 20, 21]
    assert req.last_node is restored_node
    assert req.lock_receipt is restored_receipt
    assert decode_req.hicache_restored_node is None
    assert decode_req.hicache_restore_lock_receipt is None


def test_restore_abort_releases_temporary_node_lock():
    """Verify restore abort releases and clears the temporary node lock."""
    mixin = DecodeHiCacheTransferMixin()
    restored_node = object()
    restored_receipt = DecLockRefParams()
    decode_req = SimpleNamespace(
        req=SimpleNamespace(rid="req-2", swa_prefix_lock_released=True),
        prefix_match=None,
        hicache_restored_kv_indices=torch.tensor([30, 31], dtype=torch.int64),
        hicache_restored_node=restored_node,
        hicache_restore_lock_receipt=restored_receipt,
    )
    tree_cache = MagicMock()
    mixin.tree_cache = tree_cache

    mixin._clean_hicache_prefetch_resources(decode_req)

    tree_cache.dec_lock_ref.assert_called_once_with(
        restored_node,
        restored_receipt,
        skip_swa=True,
    )
    assert decode_req.hicache_restored_node is None
    assert decode_req.hicache_restore_lock_receipt is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
