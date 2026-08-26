import importlib.util
from pathlib import Path

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_ROOT = Path(__file__).resolve().parents[5]
_DEQUANT_PATH = _ROOT / "python/sglang/kernels/ops/attention/dsa/dequant_k_cache.py"
_DEDUP_PATH = _ROOT / "python/sglang/kernels/ops/attention/dsa/selective_kv_dequant.py"
_RUNTIME_PATH = _ROOT / "python/sglang/srt/layers/attention/dsa/selective_kv_dequant.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sm89_or_newer_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)


pytestmark = pytest.mark.skipif(
    not _sm89_or_newer_available(),
    reason="the DSA FP8 Triton kernel requires compute capability 8.9 or newer",
)


def _build_quantized_kv_pool(pool_rows: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260826)
    nope = (torch.randn(pool_rows, 512, generator=generator, device="cuda") * 0.25).to(
        torch.float8_e4m3fn
    )
    scales = (
        torch.rand(pool_rows, 4, generator=generator, device="cuda") * 0.05 + 1e-3
    ).to(torch.float32)
    rope = torch.randn(pool_rows, 64, generator=generator, device="cuda").to(
        torch.bfloat16
    )

    # Reproduce the exact 656-byte DSA cache row:
    # [512 fp8 nope | 4 fp32 scales | 64 bf16 rope].
    raw = torch.empty(pool_rows, 656, dtype=torch.uint8, device="cuda")
    raw[:, :512] = nope.view(torch.uint8)
    raw[:, 512:528] = scales.view(torch.uint8)
    raw[:, 528:] = rope.view(torch.uint8)
    return raw.view(torch.float8_e4m3fn).view(pool_rows, 1, 656)


def _load_implementations():
    dequant = _load_module("dsa_dequant_fp8_test", _DEQUANT_PATH)
    dedup = _load_module("dsa_selective_dedup_fp8_test", _DEDUP_PATH)
    runtime = _load_module("dsa_selective_runtime_fp8_test", _RUNTIME_PATH)
    return dequant, dedup, runtime


def _expected_topk_rows(full_kv: torch.Tensor, topk: torch.Tensor) -> torch.Tensor:
    valid = topk >= 0
    return full_kv.view(full_kv.shape[0], -1)[topk[valid].long()]


def test_device_extent_dequant_grid_is_bounded_by_sm_count():
    dequant, _, _ = _load_implementations()

    # The CUDA Graph-safe path cannot size its launch from the device scalar
    # ``num_valid_rows``.  It should nevertheless cap the resident work grid
    # instead of launching one masked program for every capacity row.
    assert dequant._device_extent_grid_rows(32_768, sm_count=78) == 1_248
    assert dequant._device_extent_grid_rows(512, sm_count=78) == 512


def test_dense_selective_fp8_dequant_matches_full_prefix_with_physical_aliases():
    dequant, dedup, runtime = _load_implementations()
    pool = _build_quantized_kv_pool(32)
    # Logical prefix rows 0 and 2 intentionally alias physical KV slot 7.
    page_table = torch.tensor(
        [7, 3, 7, 9, 5, 11, 13, 15], dtype=torch.int32, device="cuda"
    )
    topk = torch.tensor([[0, 1, -1, 2], [3, 0, 4, 1]], dtype=torch.int32, device="cuda")
    workspace = runtime.SelectiveKVWorkspace(torch.device("cuda"))

    full = dequant.dequantize_k_cache_paged(pool, page_table)
    selection = runtime.prepare_dense_epoch_selection(
        page_table,
        topk,
        num_pool_rows=pool.shape[0],
        workspace=workspace,
        deduplicate_fn=dedup.deduplicate_kv_slots_dense_epoch,
    )
    selected = dequant.dequantize_k_cache_paged(
        pool,
        selection.selected_physical_slots,
        out=workspace.get_bf16(selection.capacity),
        num_valid_rows=selection.num_unique,
    )
    torch.cuda.synchronize()

    valid = selection.remapped_topk >= 0
    actual = selected.view(selection.capacity, -1)[
        selection.remapped_topk[valid].long()
    ]
    torch.testing.assert_close(
        actual,
        _expected_topk_rows(full, topk),
        rtol=0,
        atol=0,
    )
    assert int(selection.num_unique.item()) == 4
    torch.testing.assert_close(
        selection.selected_physical_slots[:4].cpu(),
        torch.tensor([3, 5, 7, 9], dtype=torch.int32),
    )


def test_dense_selective_fp8_dequant_combined_cuda_graph_replay():
    dequant, dedup, runtime = _load_implementations()
    pool = _build_quantized_kv_pool(32)
    page_table = torch.tensor(
        [2, 4, 6, 8, 10, 12, 14, 16], dtype=torch.int32, device="cuda"
    )
    topk = torch.tensor([[0, 1, 0, 1]], dtype=torch.int32, device="cuda")
    workspace = runtime.SelectiveKVWorkspace(torch.device("cuda"))

    # Allocate and JIT every persistent buffer before capture.
    initial = runtime.prepare_dense_epoch_selection(
        page_table,
        topk,
        num_pool_rows=pool.shape[0],
        workspace=workspace,
        deduplicate_fn=dedup.deduplicate_kv_slots_dense_epoch,
    )
    out = workspace.get_bf16(initial.capacity)

    def launch():
        selection = runtime.prepare_dense_epoch_selection(
            page_table,
            topk,
            num_pool_rows=pool.shape[0],
            workspace=workspace,
            deduplicate_fn=dedup.deduplicate_kv_slots_dense_epoch,
        )
        result = dequant.dequantize_k_cache_paged(
            pool,
            selection.selected_physical_slots,
            out=out,
            num_valid_rows=selection.num_unique,
        )
        return selection, result

    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_selection, captured_out = launch()

    topk.copy_(torch.tensor([[2, 3, 4, 2]], dtype=torch.int32, device="cuda"))
    graph.replay()
    torch.cuda.synchronize()

    full = dequant.dequantize_k_cache_paged(pool, page_table)
    valid = captured_selection.remapped_topk >= 0
    actual = captured_out.view(captured_selection.capacity, -1)[
        captured_selection.remapped_topk[valid].long()
    ]
    torch.testing.assert_close(
        actual,
        _expected_topk_rows(full, topk),
        rtol=0,
        atol=0,
    )
    assert int(captured_selection.num_unique.item()) == 3
