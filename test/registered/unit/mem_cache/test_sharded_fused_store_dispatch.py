# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the real sharded pool and Indexer dispatch methods.

Only capability probes, CUDA launchers, and the final fallback stores are
mocked. Some cases report CPU tensors as CUDA to exercise dispatch; these are
not CUDA numerical tests (those live in the registered kernel suites).
"""

import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import pytest
import torch

from sglang.kernels.ops.attention import fused_store_index_cache as index_store
from sglang.kernels.ops.kvcache import set_mla_kv_buffer as latent_store
from sglang.srt.layers.attention.dsa import dsa_indexer
from sglang.srt.mem_cache import page_interleave_pool as pools
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.mem_cache.page_interleave import PageInterleavePlacement, PageShardSpec
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _pool(start_layer=0):
    pool = pools.PageInterleaveDSATokenToKVPool.__new__(
        pools.PageInterleaveDSATokenToKVPool
    )
    pool.start_layer = start_layer
    pool.page_size = 64
    pool.shard_size = 4
    pool.shard_rank = 0
    pool.index_head_dim = pool.quant_block_size = 128
    pool.dtype = torch.float8_e4m3fn
    pool.dsa_kv_cache_store_fp8 = False
    pool._shard_extend_active = True
    pool._epoch = 1
    pool._write_plan_key = None
    pool._translate_cache = {}
    pool._page_pos = torch.arange(127, -1, -1, dtype=torch.int32)
    pool.shard_spec = PageShardSpec(0, 4, 64, 8192, 1024)
    pool.placement = PageInterleavePlacement(pool.shard_spec)
    pool.kv_buffer = [torch.empty(512, 1, 576, dtype=torch.uint8) for _ in range(2)]
    pool._slots = [
        SimpleNamespace(
            tensors={
                "kv": torch.empty(8192, 1, 576, dtype=torch.uint8),
                "index_k": torch.empty(128, 8448, dtype=torch.uint8),
            }
        )
        for _ in range(2)
    ]
    pool.index_key_cache = pools._PageInterleaveIndexKeyCache.__new__(
        pools._PageInterleaveIndexKeyCache
    )
    pool.index_key_cache.pool = pool
    pool.index_key_cache.buffer = [
        torch.empty(8, 8448, dtype=torch.uint8) for _ in range(2)
    ]
    return pool


@contextmanager
def _index_runtime(pool, *, cuda_tensor=True, jit=True):
    with (
        patch.object(torch.Tensor, "is_cuda", new_callable=PropertyMock) as is_cuda,
        patch.object(dsa_indexer, "get_token_to_kv_pool", return_value=pool),
        patch.object(dsa_indexer, "_is_cuda", True),
        patch.object(dsa_indexer, "_is_fp8_fnuz", False),
        patch.object(dsa_indexer, "_use_aiter", False),
        patch.object(dsa_indexer, "fused_store_index_k_cache") as ordinary,
        patch.object(index_store, "can_use_dsa_sharded_store", return_value=jit),
        patch.object(index_store, "fused_store_sharded_index_k_cache") as fused,
        patch.object(pools.index_buf_accessor.SetKAndS, "execute") as fallback,
    ):
        is_cuda.return_value = cuda_tensor
        yield fused, fallback, ordinary


def _store_index(pool, key, loc, quant, *, layer_id=None, **config):
    indexer = SimpleNamespace(block_size=128, scale_fmt=None)
    vars(indexer).update(config)
    dsa_indexer.Indexer._store_index_k_cache(
        indexer,
        SimpleNamespace(out_cache_loc=loc),
        pool.start_layer + 1 if layer_id is None else layer_id,
        key,
        act_quant=quant,
    )


@pytest.mark.parametrize("start_layer", [0, 9])
def test_indexer_fused_routes_target_and_draft_without_quant_or_owner_select(
    start_layer,
):
    pool = _pool(start_layer)
    loc = torch.tensor([512, 577, 1026], dtype=torch.int64)
    key = torch.ones(3, 128, dtype=torch.bfloat16)
    quant = Mock(side_effect=AssertionError("fallback quantization executed"))
    with (
        _index_runtime(pool) as (fused, fallback, ordinary),
        patch.object(
            torch.Tensor, "index_select", side_effect=AssertionError("owner copy")
        ),
    ):
        _store_index(pool, key, loc, quant)
    quant.assert_not_called()
    fallback.assert_not_called()
    ordinary.assert_not_called()
    args = fused.call_args.args
    assert args[0] is key
    assert args[1] is pool._slots[(start_layer + 1) % 2].tensors["index_k"]
    assert torch.equal(args[2], pool.translate_loc_to_scratch(loc))
    assert args[3] is pool.index_key_cache.buffer[1]
    assert args[4] is loc
    assert args[5:] == (4, 0, 64)


@pytest.mark.parametrize(
    "case",
    ["scale_format", "block_size", "int32", "non_bf16", "non_cuda", "jit", "inactive"],
)
def test_indexer_decline_preserves_quantization_and_owner_filtered_stores(case):
    pool = _pool()
    loc = torch.tensor(
        [512, 577, 1026], dtype=torch.int32 if case == "int32" else torch.int64
    )
    key = torch.ones(
        3, 128, dtype=torch.float16 if case == "non_bf16" else torch.bfloat16
    )
    quantized, scales = key.to(torch.float8_e4m3fn), torch.ones(3, 1)
    quant = Mock(return_value=(quantized, scales))
    config = {"scale_fmt": "ue8m0"} if case == "scale_format" else {}
    if case == "block_size":
        config["block_size"] = 64
    if case == "inactive":
        pool._shard_extend_active = False
    with _index_runtime(pool, cuda_tensor=case != "non_cuda", jit=case != "jit") as (
        fused,
        fallback,
        ordinary,
    ):
        _store_index(pool, key, loc, quant, **config)
    fused.assert_not_called()
    ordinary.assert_not_called()
    quant.assert_called_once_with(
        key, config.get("block_size", 128), config.get("scale_fmt")
    )
    assert fallback.call_count == (1 if case == "inactive" else 2)
    if case != "inactive":
        scratch = fallback.call_args_list[0].kwargs
        assert scratch["buf"] is pool._slots[1].tensors["index_k"]
        assert torch.equal(scratch["loc"], pool.translate_loc_to_scratch(loc))
        assert scratch["index_k"] is quantized
    local = fallback.call_args.kwargs
    assert local["buf"] is pool.index_key_cache.buffer[1]
    assert torch.equal(local["loc"], torch.tensor([128, 258]))
    assert torch.equal(
        local["index_k"].view(torch.uint8), quantized[[0, 2]].view(torch.uint8)
    )
    assert torch.equal(local["index_k_scale"], scales[[0, 2]])


def test_skip_topk_placeholder_declines_without_launch_and_retains_write_error():
    pool = _pool()
    pool.index_key_cache.buffer[1] = torch.empty(0, 8448, dtype=torch.uint8)
    loc, key = torch.tensor([512]), torch.ones(1, 128, dtype=torch.bfloat16)
    quant = Mock(return_value=(key.to(torch.float8_e4m3fn), torch.ones(1, 1)))
    with _index_runtime(pool) as (fused, fallback, _):
        assert not pool.try_store_sharded_index_k_cache(1, key, loc)
        with pytest.raises(AssertionError, match="skip-topk"):
            _store_index(pool, key, loc, quant)
    fused.assert_not_called()
    fallback.assert_not_called()


@pytest.mark.parametrize("destination", ["scratch", "local"])
@pytest.mark.parametrize("layout", ["noncontiguous", "width", "dtype", "ndim"])
def test_indexer_output_layout_declines_before_fused_capability_probe(
    destination, layout
):
    pool = _pool()
    if layout == "noncontiguous":
        buffer = torch.empty(8, 8449, dtype=torch.uint8)[:, :8448]
    elif layout == "width":
        buffer = torch.empty(8, 8447, dtype=torch.uint8)
    elif layout == "dtype":
        buffer = torch.empty(8, 8448, dtype=torch.float32)
    else:
        buffer = torch.empty(8, 1, 8448, dtype=torch.uint8)
    if destination == "scratch":
        pool._slots[1].tensors["index_k"] = buffer
    else:
        pool.index_key_cache.buffer[1] = buffer

    loc = torch.tensor([512, 577, 1026], dtype=torch.int64)
    key = torch.ones(3, 128, dtype=torch.bfloat16)
    quant = Mock(return_value=(key.to(torch.float8_e4m3fn), torch.ones(3, 1)))
    with (
        _index_runtime(pool) as (fused, fallback, ordinary),
        patch.object(index_store, "can_use_dsa_sharded_store") as capability,
    ):
        _store_index(pool, key, loc, quant)
    capability.assert_not_called()
    fused.assert_not_called()
    ordinary.assert_not_called()
    quant.assert_called_once_with(key, 128, None)
    # The existing stores retain their own format handling and error behavior;
    # dispatch must neither reinterpret the buffer nor enter the fused JIT.
    assert fallback.call_count == 2
    index = 0 if destination == "scratch" else 1
    assert fallback.call_args_list[index].kwargs["buf"] is buffer


def test_empty_owner_selection_still_stages_full_indexer_chunk():
    pool = _pool()
    loc, key = torch.tensor([576, 577]), torch.ones(2, 128, dtype=torch.bfloat16)
    quant = Mock(return_value=(key.to(torch.float8_e4m3fn), torch.ones(2, 1)))
    with _index_runtime(pool, jit=False) as (fused, fallback, _):
        _store_index(pool, key, loc, quant)
    fused.assert_not_called()
    fallback.assert_called_once()
    assert fallback.call_args.kwargs["buf"] is pool._slots[1].tensors["index_k"]


def test_indexer_fused_retranslates_in_place_mutated_location_tensor():
    pool = _pool()
    loc, key = torch.tensor([512, 577]), torch.ones(2, 128, dtype=torch.bfloat16)
    first = pool.translate_loc_to_scratch(loc).clone()
    with _index_runtime(pool) as (fused, _, _):
        _store_index(pool, key, loc, Mock())
        loc.add_(256)
        _store_index(pool, key, loc, Mock())
    assert torch.equal(fused.call_args_list[0].args[2], first)
    assert torch.equal(
        fused.call_args_list[1].args[2], pool.translate_loc_to_scratch(loc)
    )
    assert not torch.equal(first, fused.call_args_list[1].args[2])


def test_indexer_kernel_runtime_error_is_not_retried_as_fallback():
    pool = _pool()
    quant = Mock()
    with _index_runtime(pool) as (fused, fallback, _):
        fused.side_effect = RuntimeError("CUDA launch failed")
        with pytest.raises(RuntimeError, match="CUDA launch failed"):
            _store_index(
                pool,
                torch.ones(1, 128, dtype=torch.bfloat16),
                torch.tensor([512]),
                quant,
            )
    assert fused.call_count == 1
    quant.assert_not_called()
    fallback.assert_not_called()


@pytest.mark.parametrize("fused_supported", [False, True])
def test_nonsharded_indexer_keeps_existing_dispatch(fused_supported):
    buffer = torch.empty(8, 8448, dtype=torch.uint8)
    pool = SimpleNamespace(
        start_layer=0,
        page_size=64,
        get_index_k_with_scale_buffer=Mock(return_value=buffer),
        set_index_k_scale_buffer=Mock(),
    )
    loc, key = torch.tensor([512]), torch.ones(1, 128, dtype=torch.bfloat16)
    quant = Mock(return_value=(key.to(torch.float8_e4m3fn), torch.ones(1, 1)))
    with (
        _index_runtime(pool) as (sharded, _, ordinary),
        patch.object(
            dsa_indexer, "can_use_dsa_fused_store", return_value=fused_supported
        ),
    ):
        _store_index(pool, key, loc, quant)
    sharded.assert_not_called()
    if fused_supported:
        ordinary.assert_called_once_with(key, buffer, loc, 64)
        quant.assert_not_called()
        pool.set_index_k_scale_buffer.assert_not_called()
    else:
        ordinary.assert_not_called()
        quant.assert_called_once_with(key, 128, None)
        pool.set_index_k_scale_buffer.assert_called_once()


def test_unowned_indexer_layer_is_invalidated_without_writing():
    pool = _pool()
    pool._is_layer_owned = Mock(return_value=False)
    pool.invalidate_index_buffer_for_layer = Mock()
    quant = Mock()
    with _index_runtime(pool) as (fused, fallback, ordinary):
        _store_index(
            pool, torch.ones(1, 128, dtype=torch.bfloat16), torch.tensor([512]), quant
        )
    pool.invalidate_index_buffer_for_layer.assert_called_once_with(1)
    quant.assert_not_called()
    fused.assert_not_called()
    fallback.assert_not_called()
    ordinary.assert_not_called()


def _latent_inputs(n=768):
    loc = torch.arange(512, 512 + n, dtype=torch.int64)
    nope = torch.ones(n, 1, 512, dtype=torch.bfloat16)
    rope = torch.ones(n, 1, 64, dtype=torch.bfloat16)
    return loc, nope, rope


@contextmanager
def _latent_runtime(pool, *, cuda_tensor=True, jit=True, supported=True, dcp=False):
    with (
        patch.object(torch.Tensor, "is_cuda", new_callable=PropertyMock) as is_cuda,
        patch.object(
            pools, "get_parallel", return_value=SimpleNamespace(dcp_enabled=dcp)
        ),
        patch.object(
            latent_store, "can_use_set_sharded_mla_kv_buffer", return_value=jit
        ),
        patch.object(
            latent_store,
            "sharded_mla_kv_buffer_inputs_supported",
            return_value=supported,
        ),
        patch.object(latent_store, "set_sharded_mla_kv_buffer") as fused,
        patch.object(pool, "_write_mla_kv_buffer") as scratch,
        patch.object(MLATokenToKVPool, "set_mla_kv_buffer") as local,
    ):
        is_cuda.return_value = cuda_tensor
        yield fused, scratch, local


@pytest.mark.parametrize("start_layer", [0, 9])
def test_latent_fused_converts_each_source_once_and_honors_layer_override(start_layer):
    pool = _pool(start_layer)
    loc, nope, rope = _latent_inputs()
    expected = [x.to(pool.dtype).view(torch.uint8) for x in (nope, rope)]
    to_calls = []
    original_to = torch.Tensor.to

    def record_to(tensor, *args, **kwargs):
        if tensor is nope or tensor is rope:
            to_calls.append(tensor)
        return original_to(tensor, *args, **kwargs)

    with (
        _latent_runtime(pool) as (fused, scratch, local),
        patch.object(torch.Tensor, "to", record_to),
        patch.object(
            torch.Tensor, "index_select", side_effect=AssertionError("owner copy")
        ),
    ):
        pool.set_mla_kv_buffer(
            SimpleNamespace(layer_id=999),
            loc,
            nope,
            rope,
            layer_id_override=start_layer + 1,
        )
    assert len(to_calls) == 2 and to_calls[0] is nope and to_calls[1] is rope
    scratch.assert_not_called()
    local.assert_not_called()
    args, kwargs = fused.call_args
    assert (
        args[0].data_ptr()
        == pool._slots[(start_layer + 1) % 2].tensors["kv"].data_ptr()
    )
    assert args[2].data_ptr() == pool.kv_buffer[1].data_ptr()
    assert args[3] is loc
    assert all(x.device == nope.device for x in args[:6])
    assert torch.equal(args[4], expected[0]) and torch.equal(args[5], expected[1])
    assert kwargs == {"scratch_reserved_skip_index": 0, "local_reserved_skip_index": 0}


@pytest.mark.parametrize(
    "case",
    [
        "small",
        "packed656",
        "bf16_cache",
        "non_cuda",
        "geometry",
        "jit",
        "metadata",
        "inactive",
        "dcp",
    ],
)
def test_latent_decline_preserves_existing_scratch_and_owner_local_paths(case):
    pool = _pool()
    loc, nope, rope = _latent_inputs(767 if case == "small" else 768)
    if case == "packed656":
        pool.dsa_kv_cache_store_fp8 = True
        pool.kv_buffer = [torch.empty(512, 1, 656, dtype=torch.uint8) for _ in range(2)]
    elif case == "bf16_cache":
        pool.dtype = torch.bfloat16
    elif case == "geometry":
        nope = nope[..., :256]
    elif case == "inactive":
        pool._shard_extend_active = False
    with _latent_runtime(
        pool,
        cuda_tensor=case != "non_cuda",
        jit=case != "jit",
        supported=case != "metadata",
        dcp=case == "dcp",
    ) as (fused, scratch, local):
        pool.set_mla_kv_buffer(SimpleNamespace(layer_id=1), loc, nope, rope)
    fused.assert_not_called()
    assert scratch.call_count == (0 if case == "inactive" else 1)
    if case != "inactive":
        args = scratch.call_args.args
        assert args[0] is pool._slots[1].tensors["kv"]
        assert args[2] is nope and args[3] is rope
    owned = torch.nonzero(pool.placement.local_mask(loc, pool.shard_rank)).squeeze(1)
    local.assert_called_once()
    args = local.call_args.args
    assert torch.equal(args[1], pool.placement.local_index(loc[owned]))
    assert torch.equal(args[2], nope.index_select(0, owned))
    assert torch.equal(args[3], rope.index_select(0, owned))


def test_latent_fused_retranslates_in_place_mutated_locations():
    pool = _pool()
    loc, nope, rope = _latent_inputs()
    first = pool.translate_loc_to_scratch(loc).clone()
    with _latent_runtime(pool) as (fused, _, _):
        pool.set_mla_kv_buffer(SimpleNamespace(layer_id=1), loc, nope, rope)
        loc.add_(256)
        pool.set_mla_kv_buffer(SimpleNamespace(layer_id=1), loc, nope, rope)
    assert torch.equal(fused.call_args_list[0].args[1], first)
    assert torch.equal(
        fused.call_args_list[1].args[1], pool.translate_loc_to_scratch(loc)
    )
    assert not torch.equal(first, fused.call_args_list[1].args[1])


def test_latent_kernel_runtime_error_is_not_retried_as_fallback():
    pool = _pool()
    with _latent_runtime(pool) as (fused, scratch, local):
        fused.side_effect = RuntimeError("CUDA launch failed")
        with pytest.raises(RuntimeError, match="CUDA launch failed"):
            pool.set_mla_kv_buffer(SimpleNamespace(layer_id=1), *_latent_inputs())
    assert fused.call_count == 1
    scratch.assert_not_called()
    local.assert_not_called()


if __name__ == "__main__":
    args = ["-x" if arg == "-f" else arg for arg in sys.argv[1:]]
    raise SystemExit(pytest.main([__file__, *args]))
