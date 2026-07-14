"""CPU unit tests for NPU MiniMax sparse-prefill union configuration."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _identity_decorator(fn=None, **_kwargs):
    return (lambda wrapped: wrapped) if fn is None else fn


def _load_prefill_union_module():
    triton = types.ModuleType("triton")
    triton.jit = _identity_decorator
    triton.heuristics = lambda _values: _identity_decorator
    triton.next_power_of_2 = lambda value: 1 << (int(value) - 1).bit_length()
    triton_language = types.ModuleType("triton.language")
    triton_language.constexpr = int
    triton.language = triton_language

    topk = types.ModuleType(
        "sglang.srt.layers.attention.minimax_sparse_ops.npu_triton.topk_sparse_decode"
    )
    topk._choose_num_topk_chunks = lambda *_args, **_kwargs: 1
    topk._floor_power_of_2 = lambda value: 1 << (int(value) - 1).bit_length()
    topk._get_vectorcore_num_safe = lambda: 1
    topk._merge_topk_attn_out_bnsd_kernel = object()
    topk._MERGE_NS = 1
    topk._MERGE_NW = 1
    topk._normalize_topk_idx_for_gqa = lambda indices, *_args: indices
    topk._SPARSE_DECODE_NS = 1
    topk._SPARSE_DECODE_NW = 1

    module_name = "_prefill_union_main_under_test"
    module_path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/layers/attention/minimax_sparse_ops/npu_triton/prefill_union_main.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with patch.dict(
        sys.modules,
        {
            "triton": triton,
            "triton.language": triton_language,
            topk.__name__: topk,
        },
    ):
        spec.loader.exec_module(module)
    return module


class TestMiniMaxSparsePrefillUnion(CustomTestCase):
    def test_prefill_head_tile_divides_gqa_and_fits_ub(self):
        module = _load_prefill_union_module()

        for gqa_group_size, bsq_kernel, expected_h_tile in (
            (16, 4, 8),
            (4, 8, 4),
            (8, 8, 4),
            (3, 4, 1),
        ):
            with self.subTest(gqa_group_size=gqa_group_size, bsq_kernel=bsq_kernel):
                h_tile = module._choose_prefill_h_tile(gqa_group_size, bsq_kernel)
                self.assertEqual(h_tile, expected_h_tile)
                self.assertEqual(gqa_group_size % h_tile, 0)
                self.assertLessEqual(bsq_kernel * h_tile, 32)


if __name__ == "__main__":
    unittest.main()
