"""
Unit tests for sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend.
"""

import math
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=4, suite="stage-a-unit-test-npu")

for mod in (
        "torch_npu",
        "torch_npu.contrib",
        "sgl_kernel_npu",
        "sgl_kernel_npu.attention",
        "sgl_kernel_npu.attention.sinks_attention",
        "sgl_kernel_npu.norm",
        "sgl_kernel_npu.norm.add_rmsnorm_bias",
        "sglang.srt.speculative",
        "sglang.srt.speculative.decoupled_spec_io",
        "sglang.srt.speculative.spec_info",
        "sglang.srt.speculative.eagle_info",
):
    sys.modules.setdefault(mod, MagicMock())

# Stub deepseek_v2._is_hip to avoid importing the heavy model.
_ds2_stub = ModuleType("sglang.srt.models.deepseek_v2")
_ds2_stub._is_hip = False
sys.modules.setdefault("sglang.srt.models.deepseek_v2", _ds2_stub)

# Stub eagle_utils with a faithful per_step_draft_out_cache_loc.
_eagle_stub = ModuleType("sglang.srt.speculative.eagle_utils")


def _per_step_draft_out_cache_loc(out_cache_loc, batch_size, topk, num_steps):
    expected = batch_size * topk * num_steps
    assert out_cache_loc.shape[0] == expected, (
        f"out_cache_loc.shape[0]={out_cache_loc.shape[0]} != "
        f"batch_size * topk * num_steps = {batch_size}*{topk}*{num_steps}={expected}"
    )
    return (
        out_cache_loc.view(batch_size, topk, num_steps)
        .permute(2, 0, 1)
        .reshape(num_steps, -1)
    )


_eagle_stub.per_step_draft_out_cache_loc = _per_step_draft_out_cache_loc
sys.modules.setdefault("sglang.srt.speculative", ModuleType("sglang.srt.speculative"))
sys.modules.setdefault("sglang.srt.speculative.eagle_utils", _eagle_stub)

from sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend import (
    DeepseekV4AscendMultiStepDraftBackend,
    _apply_hadamard,
    _get_kv_indices,
    _overlap_transform,
    _walsh_hadamard_matrix,
)


class TestWalshHadamardMatrix(unittest.TestCase):
    def test_shape_n1(self):
        had = _walsh_hadamard_matrix(1, torch.float32, "cpu")
        self.assertEqual(had.shape, (1, 1))

    def test_shape_n2(self):
        had = _walsh_hadamard_matrix(2, torch.float32, "cpu")
        self.assertEqual(had.shape, (2, 2))

    def test_shape_n4(self):
        had = _walsh_hadamard_matrix(4, torch.float32, "cpu")
        self.assertEqual(had.shape, (4, 4))

    def test_value_error_n3_not_power_of_two(self):
        with self.assertRaises(ValueError):
            _walsh_hadamard_matrix(3, torch.float32, "cpu")

    def test_value_error_n0(self):
        with self.assertRaises(ValueError):
            _walsh_hadamard_matrix(0, torch.float32, "cpu")

    def test_value_error_negative(self):
        with self.assertRaises(ValueError):
            _walsh_hadamard_matrix(-2, torch.float32, "cpu")

    def test_orthonormality_n2(self):
        had = _walsh_hadamard_matrix(2, torch.float32, "cpu").float()
        # bfloat16 truncates 1/sqrt(2), so use a looser tolerance
        self.assertTrue(torch.allclose(had @ had.T, torch.eye(2), atol=1e-2))

    def test_orthonormality_n4(self):
        had = _walsh_hadamard_matrix(4, torch.float32, "cpu").float()
        self.assertTrue(torch.allclose(had @ had.T, torch.eye(4), atol=1e-2))

    def test_orthonormality_n1(self):
        had = _walsh_hadamard_matrix(1, torch.float32, "cpu").float()
        self.assertTrue(torch.allclose(had @ had.T, torch.eye(1)))

    def test_caching_returns_same_object(self):
        h1 = _walsh_hadamard_matrix(4, torch.float32, "cpu")
        h2 = _walsh_hadamard_matrix(4, torch.float32, "cpu")
        self.assertIs(h1, h2)

    def test_caching_different_n_returns_different_object(self):
        h1 = _walsh_hadamard_matrix(2, torch.float32, "cpu")
        h2 = _walsh_hadamard_matrix(4, torch.float32, "cpu")
        self.assertIsNot(h1, h2)

    def test_dtype_always_bfloat16(self):
        had = _walsh_hadamard_matrix(4, torch.float32, "cpu")
        self.assertEqual(had.dtype, torch.bfloat16)

    def test_dtype_argument_ignored_for_cache_key(self):
        h1 = _walsh_hadamard_matrix(4, torch.float32, "cpu")
        h2 = _walsh_hadamard_matrix(4, torch.bfloat16, "cpu")
        self.assertIs(h1, h2)

    def test_entries_are_plus_minus_norm(self):
        n = 4
        had = _walsh_hadamard_matrix(n, torch.float32, "cpu").float()
        expected_abs = 1.0 / math.sqrt(n)
        self.assertTrue(torch.allclose(had.abs(), torch.full_like(had, expected_abs), atol=1e-2))


class TestApplyHadamard(unittest.TestCase):
    def test_shape_preserved_2d(self):
        n = 4
        H = _walsh_hadamard_matrix(n, torch.float32, "cpu")
        inp = torch.randn(3, n, dtype=H.dtype)
        out = _apply_hadamard(inp, H)
        self.assertEqual(out.shape, inp.shape)

    def test_shape_preserved_3d(self):
        n = 4
        H = _walsh_hadamard_matrix(n, torch.float32, "cpu")
        inp = torch.randn(2, 5, n, dtype=H.dtype)
        out = _apply_hadamard(inp, H)
        self.assertEqual(out.shape, inp.shape)

    def test_identity_times_hadamard_equals_hadamard(self):
        n = 4
        H = _walsh_hadamard_matrix(n, torch.float32, "cpu")
        eye = torch.eye(n, dtype=H.dtype)
        out = _apply_hadamard(eye, H)
        self.assertTrue(torch.equal(out, H))

    def test_identity_times_hadamard_n2(self):
        n = 2
        H = _walsh_hadamard_matrix(n, torch.float32, "cpu")
        eye = torch.eye(n, dtype=H.dtype)
        out = _apply_hadamard(eye, H)
        self.assertTrue(torch.equal(out, H))

    def test_output_dtype_is_bfloat16(self):
        n = 4
        H = _walsh_hadamard_matrix(n, torch.float32, "cpu")
        inp = torch.randn(3, n, dtype=H.dtype)
        out = _apply_hadamard(inp, H)
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_output_dtype_bfloat16_from_float32_input(self):
        n = 4
        H = _walsh_hadamard_matrix(n, torch.bfloat16, "cpu")
        inp = torch.randn(3, n, dtype=torch.bfloat16)
        out = _apply_hadamard(inp, H)
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_3d_values(self):
        n = 2
        H = _walsh_hadamard_matrix(n, torch.float32, "cpu").float()
        inp = torch.randn(2, 3, n, dtype=torch.float32)
        expected = inp.matmul(H).to(torch.bfloat16)
        out = _apply_hadamard(inp, H)
        self.assertTrue(torch.equal(out, expected))


class TestOverlapTransform(unittest.TestCase):
    def test_shape(self):
        # (n_chunks, ratio, 2*d) -> (n_chunks, 2*ratio, d)
        n_chunks, r, d = 3, 2, 4
        tensor = torch.randn(n_chunks, r, 2 * d)
        out = _overlap_transform(tensor, value=0.0, head_dim=d)
        self.assertEqual(out.shape, (n_chunks, 2 * r, d))

    def test_first_chunk_left_half_filled_with_value(self):
        n_chunks, r, d = 3, 2, 4
        tensor = torch.randn(n_chunks, r, 2 * d)
        fill = float("-inf")
        out = _overlap_transform(tensor, value=fill, head_dim=d)
        self.assertTrue(torch.equal(out[0, :r], torch.full((r, d), fill)))

    def test_first_chunk_left_half_filled_with_zero(self):
        n_chunks, r, d = 2, 2, 4
        tensor = torch.randn(n_chunks, r, 2 * d)
        out = _overlap_transform(tensor, value=0.0, head_dim=d)
        self.assertTrue(torch.equal(out[0, :r], torch.zeros(r, d)))

    def test_right_half_mirrors_tensor_second_half(self):
        n_chunks, r, d = 3, 2, 4
        tensor = torch.randn(n_chunks, r, 2 * d)
        out = _overlap_transform(tensor, value=0.0, head_dim=d)
        self.assertTrue(torch.equal(out[:, r:], tensor[..., d:]))

    def test_previous_chunk_left_half(self):
        n_chunks, r, d = 3, 2, 4
        tensor = torch.randn(n_chunks, r, 2 * d)
        out = _overlap_transform(tensor, value=0.0, head_dim=d)
        self.assertTrue(torch.equal(out[1:, :r], tensor[:-1, :, :d]))

    def test_single_chunk(self):
        n_chunks, r, d = 1, 2, 4
        tensor = torch.randn(n_chunks, r, 2 * d)
        fill = 7.0
        out = _overlap_transform(tensor, value=fill, head_dim=d)
        self.assertEqual(out.shape, (1, 2 * r, d))
        self.assertTrue(torch.equal(out[0, :r], torch.full((r, d), fill)))
        self.assertTrue(torch.equal(out[0, r:], tensor[0, :, d:]))

    def test_full_element_mapping(self):
        n_chunks, r, d = 2, 2, 3
        tensor = torch.arange(n_chunks * r * 2 * d, dtype=torch.float32).reshape(
            n_chunks, r, 2 * d
        )
        fill = -1.0
        out = _overlap_transform(tensor, value=fill, head_dim=d)

        for c in range(n_chunks):
            for row in range(2 * r):
                for col in range(d):
                    if c == 0 and row < r:
                        expected = fill
                    elif row >= r:
                        expected = tensor[c, row - r, d + col].item()
                    else:
                        expected = tensor[c - 1, row, col].item()
                    self.assertEqual(
                        out[c, row, col].item(),
                        expected,
                        f"mismatch at (c={c}, row={row}, col={col})",
                    )

    def test_preserves_input_dtype(self):
        n_chunks, r, d = 2, 2, 4
        tensor = torch.randn(n_chunks, r, 2 * d, dtype=torch.bfloat16)
        out = _overlap_transform(tensor, value=0.0, head_dim=d)
        self.assertEqual(out.dtype, torch.bfloat16)


class TestGetKvIndices(unittest.TestCase):
    _PATCH_TARGET = "sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend.get_attn_backend"

    @patch(_PATCH_TARGET)
    def test_page_size_one_plain_slice(self, mock_get_attn_backend):
        mock_get_attn_backend.return_value = SimpleNamespace(page_size=1)
        page_table = torch.arange(16, dtype=torch.int32).reshape(2, 8)
        # req_idx=0, seqlen=5, kv_len=5 -> logic_start=0, logic_end=5
        result = _get_kv_indices(MagicMock(), 5, page_table, 0, 5)
        expected = page_table[0, 0:5]
        self.assertEqual(result.tolist(), expected.tolist())

    @patch(_PATCH_TARGET)
    def test_page_size_one_partial_window(self, mock_get_attn_backend):
        mock_get_attn_backend.return_value = SimpleNamespace(page_size=1)
        page_table = torch.arange(16, dtype=torch.int32).reshape(2, 8)
        # req_idx=0, seqlen=10, kv_len=4 -> logic_start=6, logic_end=10
        result = _get_kv_indices(MagicMock(), 4, page_table, 0, 10)
        expected = page_table[0, 6:10]
        self.assertEqual(result.tolist(), expected.tolist())

    @patch(_PATCH_TARGET)
    def test_page_size_gt_one_paged(self, mock_get_attn_backend):
        page_size = 4
        mock_get_attn_backend.return_value = SimpleNamespace(page_size=page_size)
        page_table = torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=torch.int32)
        # req_idx=0, seqlen=6, kv_len=6 -> logic_pos=[0..5]; block_id=[0,0,0,0,1,1]
        # page_table[0, block_id]=[10,10,10,10,20,20]; physical=[40,41,42,43,80,81]
        result = _get_kv_indices(MagicMock(), 6, page_table, 0, 6)
        expected = [40, 41, 42, 43, 80, 81]
        self.assertEqual(result.tolist(), expected)

    @patch(_PATCH_TARGET)
    def test_page_size_gt_one_partial_window(self, mock_get_attn_backend):
        page_size = 4
        mock_get_attn_backend.return_value = SimpleNamespace(page_size=page_size)
        page_table = torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=torch.int32)
        # req_idx=0, seqlen=10, kv_len=4 -> logic_pos=[6,7,8,9]; block_id=[1,1,2,2]
        # physical=[82,83,120,121]
        result = _get_kv_indices(MagicMock(), 4, page_table, 0, 10)
        expected = [82, 83, 120, 121]
        self.assertEqual(result.tolist(), expected)

    @patch(_PATCH_TARGET)
    def test_page_size_gt_one_second_request(self, mock_get_attn_backend):
        page_size = 4
        mock_get_attn_backend.return_value = SimpleNamespace(page_size=page_size)
        page_table = torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=torch.int32)
        # req_idx=1, seqlen=6 -> page_table[1, block_id]=[50,50,50,50,60,60]
        # physical=[200,201,202,203,240,241]
        result = _get_kv_indices(MagicMock(), 6, page_table, 1, 6)
        expected = [200, 201, 202, 203, 240, 241]
        self.assertEqual(result.tolist(), expected)

    @patch(_PATCH_TARGET)
    def test_kv_len_clamped_to_zero(self, mock_get_attn_backend):
        # kv_len > seqlen -> logic_start = max(0, seqlen - kv_len) = 0
        mock_get_attn_backend.return_value = SimpleNamespace(page_size=1)
        page_table = torch.arange(16, dtype=torch.int32).reshape(2, 8)
        result = _get_kv_indices(MagicMock(), 100, page_table, 0, 3)
        expected = page_table[0, 0:3]
        self.assertEqual(result.tolist(), expected.tolist())


class TestStepOutCacheLoc(unittest.TestCase):
    def _make_backend(self, topk, speculative_num_steps):
        backend = object.__new__(DeepseekV4AscendMultiStepDraftBackend)
        backend.topk = topk
        backend.speculative_num_steps = speculative_num_steps
        return backend

    def test_none_out_cache_loc_returns_none(self):
        backend = self._make_backend(topk=2, speculative_num_steps=3)
        forward_batch = SimpleNamespace(out_cache_loc=None, batch_size=4)
        self.assertIsNone(backend._step_out_cache_loc(forward_batch, 0))

    def test_short_out_cache_loc_returns_as_is(self):
        # numel <= single_step_width (batch_size * topk) -> returned unchanged
        backend = self._make_backend(topk=2, speculative_num_steps=3)
        loc = torch.tensor([10, 20, 30], dtype=torch.int32)
        forward_batch = SimpleNamespace(out_cache_loc=loc, batch_size=4)
        result = backend._step_out_cache_loc(forward_batch, 0)
        self.assertIs(result, loc)

    def test_short_out_cache_loc_boundary_equal(self):
        backend = self._make_backend(topk=2, speculative_num_steps=3)
        loc = torch.arange(8, dtype=torch.int32)
        forward_batch = SimpleNamespace(out_cache_loc=loc, batch_size=4)
        # single_step_width = 4*2 = 8; numel=8 <= 8
        result = backend._step_out_cache_loc(forward_batch, 0)
        self.assertIs(result, loc)

    def test_indivisible_returns_as_is(self):
        backend = self._make_backend(topk=2, speculative_num_steps=3)
        # step_layout_width = 2*3 = 6; numel=10, 10 % 6 != 0
        loc = torch.arange(10, dtype=torch.int32)
        forward_batch = SimpleNamespace(out_cache_loc=loc, batch_size=2)
        result = backend._step_out_cache_loc(forward_batch, 0)
        self.assertIs(result, loc)

    def test_step_layout_width_zero_returns_as_is(self):
        backend = self._make_backend(topk=0, speculative_num_steps=3)
        loc = torch.arange(5, dtype=torch.int32)
        forward_batch = SimpleNamespace(out_cache_loc=loc, batch_size=2)
        result = backend._step_out_cache_loc(forward_batch, 0)
        self.assertIs(result, loc)

    def test_normal_case_step0(self):
        backend = self._make_backend(topk=2, speculative_num_steps=3)
        loc = torch.arange(12, dtype=torch.int32)
        forward_batch = SimpleNamespace(out_cache_loc=loc, batch_size=2)
        # view(2,2,3).permute(2,0,1).reshape(3,-1); step 0: [0,3,6,9]
        result = backend._step_out_cache_loc(forward_batch, 0)
        self.assertEqual(result.tolist(), [0, 3, 6, 9])

    def test_normal_case_step1(self):
        backend = self._make_backend(topk=2, speculative_num_steps=3)
        loc = torch.arange(12, dtype=torch.int32)
        forward_batch = SimpleNamespace(out_cache_loc=loc, batch_size=2)
        result = backend._step_out_cache_loc(forward_batch, 1)
        self.assertEqual(result.tolist(), [1, 4, 7, 10])

    def test_normal_case_step2(self):
        backend = self._make_backend(topk=2, speculative_num_steps=3)
        loc = torch.arange(12, dtype=torch.int32)
        forward_batch = SimpleNamespace(out_cache_loc=loc, batch_size=2)
        result = backend._step_out_cache_loc(forward_batch, 2)
        self.assertEqual(result.tolist(), [2, 5, 8, 11])

    def test_normal_case_different_dimensions(self):
        backend = self._make_backend(topk=3, speculative_num_steps=2)
        loc = torch.arange(24, dtype=torch.int32)
        forward_batch = SimpleNamespace(out_cache_loc=loc, batch_size=4)
        # view(4,3,2).permute(2,0,1).reshape(2,-1)
        result = backend._step_out_cache_loc(forward_batch, 0)
        self.assertEqual(result.tolist(), [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])

    def test_normal_case_returns_tensor(self):
        backend = self._make_backend(topk=2, speculative_num_steps=3)
        loc = torch.arange(12, dtype=torch.int32)
        forward_batch = SimpleNamespace(out_cache_loc=loc, batch_size=2)
        result = backend._step_out_cache_loc(forward_batch, 0)
        self.assertIsInstance(result, torch.Tensor)


class TestCommonTemplate(unittest.TestCase):
    def _make_backend(self, speculative_num_steps):
        backend = object.__new__(DeepseekV4AscendMultiStepDraftBackend)
        backend.speculative_num_steps = speculative_num_steps
        return backend

    def test_calls_call_fn_for_each_step(self):
        backend = self._make_backend(speculative_num_steps=4)
        forward_batch = SimpleNamespace(spec_info=object())
        call_fn = MagicMock()
        backend.common_template(forward_batch, call_fn)
        # range(speculative_num_steps - 1) = range(3) -> i=0,1,2
        self.assertEqual(call_fn.call_count, 3)
        for i, call in enumerate(call_fn.call_args_list):
            self.assertEqual(call.args[0], i)
            self.assertIs(call.args[1], forward_batch)

    def test_single_step_no_calls(self):
        backend = self._make_backend(speculative_num_steps=1)
        forward_batch = SimpleNamespace(spec_info=object())
        call_fn = MagicMock()
        backend.common_template(forward_batch, call_fn)
        self.assertEqual(call_fn.call_count, 0)

    def test_two_steps_one_call(self):
        backend = self._make_backend(speculative_num_steps=2)
        forward_batch = SimpleNamespace(spec_info=object())
        call_fn = MagicMock()
        backend.common_template(forward_batch, call_fn)
        self.assertEqual(call_fn.call_count, 1)
        self.assertEqual(call_fn.call_args_list[0].args[0], 0)

    def test_asserts_spec_info_not_none(self):
        backend = self._make_backend(speculative_num_steps=3)
        forward_batch = SimpleNamespace(spec_info=None)
        call_fn = MagicMock()
        with self.assertRaises(AssertionError):
            backend.common_template(forward_batch, call_fn)
        self.assertEqual(call_fn.call_count, 0)

    def test_call_fn_exception_propagates(self):
        backend = self._make_backend(speculative_num_steps=3)
        forward_batch = SimpleNamespace(spec_info=object())
        call_fn = MagicMock(side_effect=RuntimeError("boom"))
        with self.assertRaises(RuntimeError):
            backend.common_template(forward_batch, call_fn)
        self.assertEqual(call_fn.call_count, 1)


if __name__ == "__main__":
    unittest.main()
