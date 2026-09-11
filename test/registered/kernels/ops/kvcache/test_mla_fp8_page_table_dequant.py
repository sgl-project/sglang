"""GPU tests for FA3's page-id-preserving MLA FP8 shadow.

The production path must copy only live page-table rows, preserve physical page
ids, deduplicate pages shared by requests, reuse one workspace across layers,
and remain correct when page-table contents change between CUDA Graph replays.
Unsupported sources and backend routes must keep the legacy full-pool cast.
"""

from unittest.mock import Mock

import pytest
import torch

from sglang.kernels.ops.kvcache.mla_buffer import (
    FA3MLAFP8KVShadow,
    dequantize_mla_fp8_page_table,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)

DEVICE = "cuda"
NUM_PAGES = 64
ROW_WIDTH = 576


def _make_state(fp8_dtype, output_dtype):
    source = torch.randn(NUM_PAGES, 1, ROW_WIDTH, device=DEVICE).to(fp8_dtype)
    shadow = torch.full(
        (NUM_PAGES, 1, ROW_WIDTH),
        float("nan"),
        dtype=output_dtype,
        device=DEVICE,
    )
    page_table = torch.tensor(
        [
            [1, 2, 3, 4, 5, 0, 0, 0],
            [1, 2, 8, 9, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.int32,
        device=DEVICE,
    )
    cache_seqlens = torch.tensor([5, 4, 0], dtype=torch.int32, device=DEVICE)
    page_epochs = torch.zeros(NUM_PAGES, dtype=torch.int32, device=DEVICE)
    epoch = torch.zeros((), dtype=torch.int32, device=DEVICE)
    return source, shadow, page_table, cache_seqlens, page_epochs, epoch


@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float16])
def test_dequantize_only_referenced_pages(fp8_dtype, output_dtype):
    source, shadow, page_table, cache_seqlens, page_epochs, epoch = _make_state(
        fp8_dtype, output_dtype
    )

    dequantize_mla_fp8_page_table(
        source, shadow, page_table, cache_seqlens, page_epochs, epoch
    )

    referenced = torch.tensor([1, 2, 3, 4, 5, 8, 9], device=DEVICE)
    torch.testing.assert_close(
        shadow[referenced], source[referenced].to(output_dtype), rtol=0, atol=0
    )
    assert torch.isnan(shadow[6]).all()
    assert epoch.item() == 1
    assert torch.count_nonzero(page_epochs == epoch).item() == referenced.numel()

    source[10:12] = torch.randn(2, 1, ROW_WIDTH, device=DEVICE).to(fp8_dtype)
    page_table.copy_(
        torch.tensor(
            [
                [10, 11, 0, 0, 0, 0, 0, 0],
                [10, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=torch.int32,
            device=DEVICE,
        )
    )
    cache_seqlens.copy_(torch.tensor([2, 1, 0], dtype=torch.int32, device=DEVICE))
    dequantize_mla_fp8_page_table(
        source, shadow, page_table, cache_seqlens, page_epochs, epoch
    )

    torch.testing.assert_close(
        shadow[10:12], source[10:12].to(output_dtype), rtol=0, atol=0
    )
    assert epoch.item() == 2
    assert torch.count_nonzero(page_epochs == epoch).item() == 2


def test_dequantize_is_cuda_graph_replay_safe():
    source, shadow, page_table, cache_seqlens, page_epochs, epoch = _make_state(
        torch.float8_e4m3fn, torch.bfloat16
    )
    dequantize_mla_fp8_page_table(
        source, shadow, page_table, cache_seqlens, page_epochs, epoch
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dequantize_mla_fp8_page_table(
            source, shadow, page_table, cache_seqlens, page_epochs, epoch
        )

    new_values = torch.randn(3, 1, ROW_WIDTH, device=DEVICE).to(torch.float8_e4m3fn)
    source[20:23].copy_(new_values)
    page_table.copy_(
        torch.tensor(
            [
                [20, 21, 0, 0, 0, 0, 0, 0],
                [20, 22, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=torch.int32,
            device=DEVICE,
        )
    )
    cache_seqlens.copy_(torch.tensor([2, 2, 0], dtype=torch.int32, device=DEVICE))
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        shadow[20:23], source[20:23].to(torch.bfloat16), rtol=0, atol=0
    )
    assert epoch.item() == 2

    newer_values = torch.randn(2, 1, ROW_WIDTH, device=DEVICE).to(torch.float8_e4m3fn)
    source[30:32].copy_(newer_values)
    page_table.zero_()
    page_table[0, :2] = torch.tensor([30, 31], dtype=torch.int32, device=DEVICE)
    cache_seqlens.copy_(torch.tensor([2, 0, 0], dtype=torch.int32, device=DEVICE))
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        shadow[30:32], source[30:32].to(torch.bfloat16), rtol=0, atol=0
    )
    assert epoch.item() == 3


def test_empty_page_table_is_a_noop():
    source = torch.randn(NUM_PAGES, 1, ROW_WIDTH, device=DEVICE).to(torch.float8_e4m3fn)
    shadow = torch.full_like(source, float("nan"), dtype=torch.bfloat16)
    page_epochs = torch.zeros(NUM_PAGES, dtype=torch.int32, device=DEVICE)
    epoch = torch.zeros((), dtype=torch.int32, device=DEVICE)

    dequantize_mla_fp8_page_table(
        source,
        shadow,
        torch.empty((0, 0), dtype=torch.int32, device=DEVICE),
        torch.empty(0, dtype=torch.int32, device=DEVICE),
        page_epochs,
        epoch,
    )

    assert torch.isnan(shadow).all()
    assert epoch.item() == 0
    assert torch.count_nonzero(page_epochs).item() == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda state: state.__setitem__(1, state[1][:-1]), "matching row-major"),
        (lambda state: state.__setitem__(2, state[2].flatten()), "2D page table"),
        (
            lambda state: state.__setitem__(3, state[3][:-1]),
            "one entry for every page-table row",
        ),
        (
            lambda state: state.__setitem__(4, state[4][:-1]),
            "Epoch state does not match",
        ),
        (
            lambda state: state.__setitem__(5, torch.zeros(2, device=DEVICE)),
            "Epoch state does not match",
        ),
    ],
)
def test_input_validation_fails_before_launch(mutation, message):
    state = list(_make_state(torch.float8_e4m3fn, torch.bfloat16))
    mutation(state)
    with pytest.raises(ValueError, match=message):
        dequantize_mla_fp8_page_table(*state)


@pytest.mark.parametrize(
    ("source_factory", "output_dtype"),
    [
        (
            lambda: torch.zeros(
                NUM_PAGES, 1, ROW_WIDTH, dtype=torch.bfloat16, device=DEVICE
            ),
            torch.bfloat16,
        ),
        (
            lambda: torch.zeros(
                NUM_PAGES, 1, ROW_WIDTH, dtype=torch.float8_e4m3fn, device=DEVICE
            ),
            torch.float32,
        ),
        (
            lambda: torch.zeros(
                NUM_PAGES,
                1,
                ROW_WIDTH * 2,
                dtype=torch.float8_e4m3fn,
                device=DEVICE,
            )[..., ::2],
            torch.bfloat16,
        ),
    ],
)
def test_shadow_factory_rejects_unsupported_sources(source_factory, output_dtype):
    assert FA3MLAFP8KVShadow.maybe_create(source_factory(), output_dtype) is None


def test_shadow_workspace_is_shared_across_layers():
    first_layer = torch.randn(NUM_PAGES, 1, ROW_WIDTH, device=DEVICE).to(
        torch.float8_e4m3fn
    )
    second_layer = torch.randn(NUM_PAGES, 1, ROW_WIDTH, device=DEVICE).to(
        torch.float8_e4m3fn
    )
    page_table = torch.tensor([[1, 2, 8]], dtype=torch.int32, device=DEVICE)
    cache_seqlens = torch.tensor([3], dtype=torch.int32, device=DEVICE)

    shadow = FA3MLAFP8KVShadow.maybe_create(first_layer, torch.bfloat16)
    assert shadow is not None
    assert shadow.buffer.shape == first_layer.shape
    assert shadow.buffer.dtype == torch.bfloat16
    assert shadow.page_epochs.shape == (NUM_PAGES,)
    first_result = shadow.materialize(first_layer, page_table, cache_seqlens)
    first_pointer = first_result.data_ptr()
    torch.testing.assert_close(
        first_result[page_table[0]], first_layer[page_table[0]].to(torch.bfloat16)
    )

    second_result = shadow.materialize(second_layer, page_table, cache_seqlens)
    assert second_result.data_ptr() == first_pointer
    torch.testing.assert_close(
        second_result[page_table[0]], second_layer[page_table[0]].to(torch.bfloat16)
    )
    torch.testing.assert_close(
        page_table, torch.tensor([[1, 2, 8]], dtype=torch.int32, device=DEVICE)
    )

    assert not shadow.can_materialize(second_layer, torch.float16)
    assert not shadow.can_materialize(second_layer[:-1], torch.bfloat16)
    noncontiguous_layer = torch.zeros(
        NUM_PAGES,
        1,
        ROW_WIDTH * 2,
        dtype=torch.float8_e4m3fn,
        device=DEVICE,
    )[..., ::2]
    assert noncontiguous_layer.shape == second_layer.shape
    assert not noncontiguous_layer.is_contiguous()
    assert not shadow.can_materialize(noncontiguous_layer, torch.bfloat16)


def test_backend_selects_shadow_and_preserves_page_table():
    backend_module = pytest.importorskip(
        "sglang.srt.layers.attention.flashattention_backend",
        reason="requires a loadable sgl_kernel build",
        exc_type=ImportError,
    )
    FlashAttentionBackend = backend_module.FlashAttentionBackend

    source = torch.randn(NUM_PAGES, 1, ROW_WIDTH, device=DEVICE).to(torch.float8_e4m3fn)
    materialized = torch.empty_like(source, dtype=torch.bfloat16)
    page_table = torch.tensor([[1, 2]], dtype=torch.int32, device=DEVICE)
    cache_seqlens = torch.tensor([2], dtype=torch.int32, device=DEVICE)

    pool = Mock()
    pool.get_key_buffer.return_value = source
    shadow = Mock()
    shadow.can_materialize.return_value = True
    shadow.materialize.return_value = materialized

    backend = object.__new__(FlashAttentionBackend)
    backend.token_to_kv_pool = pool
    backend._fa3_mla_fp8_shadow = shadow

    result = backend._get_mla_kv_cache_for_fa3(
        layer_id=3,
        output_dtype=torch.bfloat16,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        use_fp8_shadow=True,
    )
    assert result is materialized
    pool.get_key_buffer.assert_called_once_with(3)
    shadow.materialize.assert_called_once_with(source, page_table, cache_seqlens)
    torch.testing.assert_close(
        page_table, torch.tensor([[1, 2]], dtype=torch.int32, device=DEVICE)
    )

    fallback = backend._get_mla_kv_cache_for_fa3(
        3, torch.bfloat16, page_table, cache_seqlens, use_fp8_shadow=False
    )
    torch.testing.assert_close(fallback, source.to(torch.bfloat16), rtol=0, atol=0)
    shadow.materialize.assert_called_once()

    shadow.can_materialize.return_value = False
    incompatible = backend._get_mla_kv_cache_for_fa3(
        3, torch.bfloat16, page_table, cache_seqlens, use_fp8_shadow=True
    )
    torch.testing.assert_close(incompatible, source.to(torch.bfloat16), rtol=0, atol=0)
    shadow.materialize.assert_called_once()

    backend._fa3_mla_fp8_shadow = None
    missing_shadow = backend._get_mla_kv_cache_for_fa3(
        3, torch.bfloat16, page_table, cache_seqlens, use_fp8_shadow=True
    )
    torch.testing.assert_close(
        missing_shadow, source.to(torch.bfloat16), rtol=0, atol=0
    )
