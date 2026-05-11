# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Unit tests for sglang/srt/utils/weight_checker.py."""

import unittest
from typing import Iterable, List, Tuple
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_dequant,
    quant_weight_ue8m0,
    transform_scale_ue8m0,
)
from sglang.srt.utils.weight_checker import (
    ChecksumInfo,
    ParallelismInfo,
    WeightChecker,
    _check_tensors,
    _hash_tensor,
    _is_non_persistent_buffer_name,
    _postprocess_tensors,
    _random_like,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=9, suite="stage-b-test-1-gpu-small")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


Triple = Tuple[str, bool, torch.Tensor]


def _assert_triples_close(actual: Iterable[Triple], expected: Iterable[Triple]) -> None:
    """Compare two streams of (name, should_compare, tensor); element-wise tensor close."""
    actual_list: List[Triple] = list(actual)
    expected_list: List[Triple] = list(expected)
    assert len(actual_list) == len(
        expected_list
    ), f"length mismatch: actual={len(actual_list)} expected={len(expected_list)}"
    for i, ((a_name, a_flag, a_t), (e_name, e_flag, e_t)) in enumerate(
        zip(actual_list, expected_list)
    ):
        assert a_name == e_name, f"[{i}] name: {a_name!r} != {e_name!r}"
        assert a_flag == e_flag, f"[{i}] should_compare: {a_flag} != {e_flag}"
        torch.testing.assert_close(
            a_t, e_t, msg=f"[{i}] tensor mismatch for {a_name!r}"
        )


def _build_fp8_quant_pair(device: str = "cuda"):
    """Construct a real fp8-quantized weight + matching fp32 + ue8m0-packed scales.

    Returns (qweight, sf_fp32, sf_packed_int32) so callers can pick which scale dtype
    drives the _postprocess_tensors branch under test.
    """
    weight_bf16 = torch.randn((256, 128), dtype=torch.bfloat16, device=device)
    block_size = [128, 128]
    qweight, sf_fp32 = quant_weight_ue8m0(
        weight_dequant=weight_bf16, weight_block_size=block_size
    )
    sf_packed_int32 = transform_scale_ue8m0(sf_fp32, mn=qweight.shape[-2])
    return qweight, sf_fp32, sf_packed_int32


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """Mimics the buffer naming patterns _reset_tensors / _postprocess_tensors care about."""

    def __init__(self):
        super().__init__()
        # requires_grad=False matches sglang's inference-time params, so _reset_tensors
        # can do in-place copy_ on them (autograd would otherwise reject it).
        self.w = nn.Parameter(torch.randn(4, 4), requires_grad=False)
        self.b = nn.Parameter(torch.zeros(4), requires_grad=False)
        self.register_buffer("running_mean", torch.zeros(4))
        # Buffer names that match weight_checker's hard-coded skip patterns.
        self.register_buffer("rotary_emb_cos_sin_cache", torch.full((8,), 3.14))
        self.register_buffer("rotary_emb_freqs_cis", torch.full((8,), 2.71))
        self.register_buffer("gate_proj_weight_fp32_cache", torch.full((8,), 1.41))


class _FakeModelRunner:
    """Minimal stand-in: WeightChecker touches `.model.named_parameters()`,
    `.model.named_buffers()`, plus parallelism attributes for the checksum action."""

    def __init__(
        self,
        model: nn.Module,
        tp_rank: int = 0,
        tp_size: int = 1,
        dp_rank: int = 0,
        dp_size: int = 1,
        pp_rank: int = 0,
        pp_size: int = 1,
    ):
        self.model = model
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.pp_rank = pp_rank
        self.pp_size = pp_size


# ---------------------------------------------------------------------------
# _random_like
# ---------------------------------------------------------------------------


class TestRandomLike(CustomTestCase):

    def test_floating_point_preserves_dtype_shape_device(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            t = torch.zeros(8, 4, dtype=dtype)
            out = _random_like(t)
            self.assertEqual(out.dtype, dtype)
            self.assertEqual(out.shape, t.shape)
            self.assertEqual(out.device, t.device)
            self.assertGreater(out.float().abs().sum().item(), 0)

    def test_bool_returns_bool_with_both_values(self):
        t = torch.zeros(1024, dtype=torch.bool)
        out = _random_like(t)
        self.assertEqual(out.dtype, torch.bool)
        self.assertEqual(out.shape, t.shape)
        self.assertEqual(out.device, t.device)
        self.assertTrue(out.any().item())
        self.assertFalse(out.all().item())

    def test_int_returns_correct_dtype_in_range(self):
        for dtype in (torch.int8, torch.int32, torch.int64):
            t = torch.zeros(256, dtype=dtype)
            out = _random_like(t)
            self.assertEqual(out.dtype, dtype)
            self.assertEqual(out.shape, t.shape)
            info = torch.iinfo(dtype)
            self.assertGreaterEqual(out.min().item(), info.min)
            self.assertLessEqual(out.max().item(), info.max)
            self.assertGreater(out.unique().numel(), 1)

    def test_floating_point_values_in_unit_range(self):
        t = torch.zeros(1024, dtype=torch.float32)
        out = _random_like(t)
        self.assertGreaterEqual(out.min().item(), 0.0)
        self.assertLess(out.max().item(), 1.0)

    def test_does_not_mutate_input(self):
        t = torch.full((16,), 5.0)
        before = t.clone()
        _random_like(t)
        torch.testing.assert_close(t, before)


# ---------------------------------------------------------------------------
# _postprocess_tensors
# ---------------------------------------------------------------------------


class TestPostprocessTensors(CustomTestCase):

    # --- non-quant / non-skip ---

    def test_no_quant_yields_raw_with_should_compare_true(self):
        a = torch.randn(4)
        b = torch.randn(4)
        raw = {"a.weight": a, "b.bias": b}
        _assert_triples_close(
            _postprocess_tensors(raw, set()),
            [("a.weight", True, a), ("b.bias", True, b)],
        )

    def test_weight_alone_without_scale_inv_does_not_trigger_dequant(self):
        w = torch.randn(4)
        raw = {"x.weight": w}
        _assert_triples_close(_postprocess_tensors(raw, set()), [("x.weight", True, w)])

    # --- non-persistent buffer skip ---

    def test_skips_cos_sin_cache_substring(self):
        cache = torch.randn(8)
        plain = torch.randn(4)
        raw = {
            "model.rotary_emb.cos_sin_cache": cache,
            "model.layers.0.weight": plain,
        }
        _assert_triples_close(
            _postprocess_tensors(raw, set()),
            [
                ("model.rotary_emb.cos_sin_cache", False, cache),
                ("model.layers.0.weight", True, plain),
            ],
        )

    def test_skips_inv_freq_substring(self):
        t = torch.randn(4)
        _assert_triples_close(
            _postprocess_tensors({"model.rotary_emb.inv_freq": t}, set()),
            [("model.rotary_emb.inv_freq", False, t)],
        )

    def test_skips_weight_fp32_substring(self):
        t = torch.randn(4)
        _assert_triples_close(
            _postprocess_tensors({"model.layers.0.mlp.gate._weight_fp32": t}, set()),
            [("model.layers.0.mlp.gate._weight_fp32", False, t)],
        )

    def test_substring_match_not_endswith(self):
        # Pattern can appear anywhere in the name, not just at the end.
        t = torch.randn(4)
        _assert_triples_close(
            _postprocess_tensors({"weird.cos_sin_cache.foo.bar": t}, set()),
            [("weird.cos_sin_cache.foo.bar", False, t)],
        )

    # --- fp8 quant pair (real dequant on real fp8 tensors) ---

    def test_fp8_quant_pair_with_int32_scale_dequants_via_ue8m0(self):
        qweight, sf_fp32, sf_packed_int32 = _build_fp8_quant_pair()
        raw = {"x.weight": qweight, "x.weight_scale_inv": sf_packed_int32}

        # Reference: ue8m0 path inside _postprocess_tensors should eventually
        # call block_quant_dequant with the unpacked fp32 scale.
        expected_dequant = block_quant_dequant(
            qweight, sf_fp32, block_size=[128, 128], dtype=torch.bfloat16
        )
        _assert_triples_close(
            _postprocess_tensors(raw, set()),
            [
                ("x.weight", True, expected_dequant),
                ("x.weight", False, qweight),
                ("x.weight_scale_inv", False, sf_packed_int32),
            ],
        )

    def test_fp8_quant_pair_with_fp32_scale_dequants_directly(self):
        qweight, sf_fp32, _ = _build_fp8_quant_pair()
        raw = {"x.weight": qweight, "x.weight_scale_inv": sf_fp32}

        expected_dequant = block_quant_dequant(
            qweight, sf_fp32, block_size=[128, 128], dtype=torch.bfloat16
        )
        _assert_triples_close(
            _postprocess_tensors(raw, set()),
            [
                ("x.weight", True, expected_dequant),
                ("x.weight", False, qweight),
                ("x.weight_scale_inv", False, sf_fp32),
            ],
        )

    def test_fp8_quant_pair_yield_order_alongside_other_entries(self):
        qweight, sf_fp32, _ = _build_fp8_quant_pair()
        bias = torch.ones(4, device="cuda")
        raw = {
            "x.weight": qweight,
            "x.weight_scale_inv": sf_fp32,
            "y.bias": bias,
        }
        expected_dequant = block_quant_dequant(
            qweight, sf_fp32, block_size=[128, 128], dtype=torch.bfloat16
        )
        # All dequant entries come first, then a raw pass over every key.
        _assert_triples_close(
            _postprocess_tensors(raw, set()),
            [
                ("x.weight", True, expected_dequant),
                ("x.weight", False, qweight),
                ("x.weight_scale_inv", False, sf_fp32),
                ("y.bias", True, bias),
            ],
        )

    def test_only_scale_without_weight_does_not_trigger_dequant(self):
        # Without the matching `.weight`, no quant pair forms; the scale_inv flows
        # through as a normal entry with should_compare=True.
        s = torch.zeros(1, 1, dtype=torch.int32)
        _assert_triples_close(
            _postprocess_tensors({"x.weight_scale_inv": s}, set()),
            [("x.weight_scale_inv", True, s)],
        )


# ---------------------------------------------------------------------------
# _check_tensors  (implementation moves both sides via .cuda())
# ---------------------------------------------------------------------------


class TestCheckTensors(CustomTestCase):

    def test_passes_when_all_equal(self):
        t = torch.ones(2, 2)
        expect = [("a", True, t.clone()), ("b", True, t.clone())]
        actual = [("a", True, t.clone()), ("b", True, t.clone())]
        _check_tensors(expect_tensors=expect, actual_tensors=actual)

    def test_raises_when_should_compare_true_and_diff(self):
        expect = [("a", True, torch.ones(2, 2))]
        actual = [("a", True, torch.zeros(2, 2))]
        with self.assertRaises(Exception) as ctx:
            _check_tensors(expect_tensors=expect, actual_tensors=actual)
        msg = str(ctx.exception)
        self.assertIn("name=a", msg)
        self.assertIn("max_abs_err", msg)

    def test_passes_when_should_compare_false_even_if_diff(self):
        # should_compare=False -> diff is logged, not raised.
        expect = [("a", False, torch.ones(2, 2))]
        actual = [("a", False, torch.zeros(2, 2))]
        _check_tensors(expect_tensors=expect, actual_tensors=actual)

    def test_asserts_on_name_mismatch(self):
        expect = [("a", True, torch.ones(2, 2))]
        actual = [("b", True, torch.ones(2, 2))]
        with self.assertRaises(AssertionError):
            _check_tensors(expect_tensors=expect, actual_tensors=actual)

    def test_asserts_on_should_compare_mismatch(self):
        expect = [("a", True, torch.ones(2, 2))]
        actual = [("a", False, torch.ones(2, 2))]
        with self.assertRaises(AssertionError):
            _check_tensors(expect_tensors=expect, actual_tensors=actual)

    def test_zip_strict_raises_on_length_mismatch(self):
        t = torch.ones(2, 2)
        expect = [("a", True, t.clone()), ("b", True, t.clone())]
        actual = [("a", True, t.clone())]
        with self.assertRaises(ValueError):
            _check_tensors(expect_tensors=expect, actual_tensors=actual)


# ---------------------------------------------------------------------------
# WeightChecker class
# ---------------------------------------------------------------------------


class _WeightCheckerTestBase(CustomTestCase):
    """Shared fixture: fresh _TinyModel + WeightChecker per test, on CUDA.

    The model lives on CUDA so that _snapshot's `.detach().cpu()` produces
    an independent CPU copy. On a CPU model `.cpu()` is a no-op and the
    snapshot would alias the live storage, which masks reset-then-compare
    divergence.
    """

    def setUp(self):
        torch.manual_seed(0)
        self.model = _TinyModel().cuda()
        self.checker = WeightChecker(model_runner=_FakeModelRunner(self.model))


class TestSnapshot(_WeightCheckerTestBase):

    def test_captures_params_and_buffers(self):
        self.checker._snapshot()
        keys = set(self.checker._snapshot_tensors.keys())
        expected = {
            "w",
            "b",
            "running_mean",
            "rotary_emb_cos_sin_cache",
            "rotary_emb_freqs_cis",
            "gate_proj_weight_fp32_cache",
        }
        self.assertEqual(keys, expected)

    def test_detaches_and_moves_to_cpu(self):
        self.checker._snapshot()
        for tensor in self.checker._snapshot_tensors.values():
            self.assertEqual(tensor.device.type, "cpu")
        # Mutating the live model must not affect the snapshot copy.
        original_w = self.checker._snapshot_tensors["w"].clone()
        with torch.no_grad():
            self.model.w.data.fill_(99.0)
        torch.testing.assert_close(self.checker._snapshot_tensors["w"], original_w)


class TestResetTensors(_WeightCheckerTestBase):

    def test_changes_normal_params_in_place(self):
        before_w = self.model.w.clone()
        before_w_ptr = self.model.w.data_ptr()
        self.checker._reset_tensors()
        # In-place: storage pointer unchanged.
        self.assertEqual(self.model.w.data_ptr(), before_w_ptr)
        self.assertFalse(torch.equal(self.model.w, before_w))

    def test_skips_cos_sin_cache(self):
        before = self.model.rotary_emb_cos_sin_cache.clone()
        self.checker._reset_tensors()
        torch.testing.assert_close(self.model.rotary_emb_cos_sin_cache, before)

    def test_skips_freqs_cis(self):
        before = self.model.rotary_emb_freqs_cis.clone()
        self.checker._reset_tensors()
        torch.testing.assert_close(self.model.rotary_emb_freqs_cis, before)

    def test_skips_weight_fp32(self):
        before = self.model.gate_proj_weight_fp32_cache.clone()
        self.checker._reset_tensors()
        torch.testing.assert_close(self.model.gate_proj_weight_fp32_cache, before)


class TestCompare(_WeightCheckerTestBase):

    def test_without_snapshot_raises(self):
        with self.assertRaises(AssertionError):
            self.checker._compare()

    def test_passes_when_unchanged(self):
        self.checker._snapshot()
        self.checker._compare()  # no exception

    def test_fails_after_reset_on_normal_param(self):
        self.checker._snapshot()
        self.checker._reset_tensors()
        with self.assertRaises(Exception) as ctx:
            self.checker._compare()
        msg = str(ctx.exception)
        self.assertTrue(("name=w" in msg) or ("name=b" in msg))

    def test_passes_when_only_skipped_buffer_diverges(self):
        self.checker._snapshot()
        # Mutate a non-persistent skip-pattern buffer; compare must still pass.
        with torch.no_grad():
            self.model.rotary_emb_cos_sin_cache.fill_(99.0)
        self.checker._compare()

    def test_passes_after_reset_then_restoring_normal_params(self):
        # Full lifecycle: reset (skips cos_sin_cache et al.), then restore non-skip
        # params by hand. Compare must pass — proving reset+postprocess skip lists agree.
        self.checker._snapshot()
        snapshot = {k: v.clone() for k, v in self.checker._snapshot_tensors.items()}
        self.checker._reset_tensors()
        with torch.no_grad():
            for name, tensor in self.model.named_parameters():
                tensor.data.copy_(snapshot[name].to(tensor.device))
            for name, tensor in self.model.named_buffers():
                tensor.data.copy_(snapshot[name].to(tensor.device))
        self.checker._compare()


class TestHandle(_WeightCheckerTestBase):

    def test_routes_to_actions(self):
        with patch.object(self.checker, "_snapshot") as m_snap, patch.object(
            self.checker, "_reset_tensors"
        ) as m_reset, patch.object(self.checker, "_compare") as m_compare, patch.object(
            self.checker, "_compute_checksum", return_value={"checksums": {}}
        ) as m_checksum:
            self.checker.handle("snapshot")
            self.checker.handle("reset_tensors")
            self.checker.handle("compare")
            self.checker.handle("checksum")
            m_snap.assert_called_once()
            m_reset.assert_called_once()
            m_compare.assert_called_once()
            m_checksum.assert_called_once()

    def test_returns_none_for_non_checksum_actions(self):
        self.assertIsNone(self.checker.handle("snapshot"))
        self.assertIsNone(self.checker.handle("compare"))

    def test_returns_dict_for_checksum_action(self):
        out = self.checker.handle("checksum")
        self.assertIsInstance(out, dict)
        self.assertIn("checksums", out)
        self.assertIn("parallelism_info", out)

    def test_unknown_action_raises(self):
        with self.assertRaises(Exception) as ctx:
            self.checker.handle("nonsense_action")
        self.assertIn("Unsupported", str(ctx.exception))


# ---------------------------------------------------------------------------
# _is_non_persistent_buffer_name
# ---------------------------------------------------------------------------


class TestIsNonPersistentBufferName(CustomTestCase):

    def test_matches_cos_sin_cache_substring(self):
        self.assertTrue(
            _is_non_persistent_buffer_name("model.rotary_emb.cos_sin_cache")
        )

    def test_matches_inv_freq_substring(self):
        self.assertTrue(_is_non_persistent_buffer_name("model.rotary_emb.inv_freq"))

    def test_matches_freqs_cis_substring(self):
        self.assertTrue(_is_non_persistent_buffer_name("model.rotary_emb.freqs_cis"))

    def test_matches_weight_fp32_substring(self):
        self.assertTrue(
            _is_non_persistent_buffer_name("model.layers.0.mlp.gate._weight_fp32")
        )

    def test_does_not_match_normal_param_names(self):
        self.assertFalse(_is_non_persistent_buffer_name("model.layers.0.mlp.weight"))
        self.assertFalse(_is_non_persistent_buffer_name("model.embed_tokens.weight"))


# ---------------------------------------------------------------------------
# _hash_tensor
# ---------------------------------------------------------------------------


class TestHashTensor(CustomTestCase):

    def test_stable_for_same_input(self):
        t = torch.arange(64, dtype=torch.float32).cuda()
        self.assertEqual(_hash_tensor(t), _hash_tensor(t.clone()))

    def test_changes_with_data(self):
        a = torch.zeros(64, dtype=torch.float32).cuda()
        b = torch.ones(64, dtype=torch.float32).cuda()
        self.assertNotEqual(_hash_tensor(a), _hash_tensor(b))

    def test_returns_16_char_hex(self):
        t = torch.zeros(64, dtype=torch.float32).cuda()
        h = _hash_tensor(t)
        self.assertEqual(len(h), 16)
        int(h, 16)  # raises if not hex

    def test_does_not_mutate_input(self):
        t = torch.arange(64, dtype=torch.float32).cuda()
        before = t.clone()
        _hash_tensor(t)
        torch.testing.assert_close(t, before)


# ---------------------------------------------------------------------------
# _compute_checksum
# ---------------------------------------------------------------------------


class _ChecksumTestBase(CustomTestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.model = _TinyModel().cuda()
        self.runner = _FakeModelRunner(
            self.model,
            tp_rank=2,
            tp_size=4,
            dp_rank=1,
            dp_size=2,
            pp_rank=0,
            pp_size=1,
        )
        self.checker = WeightChecker(model_runner=self.runner)


class TestComputeChecksum(_ChecksumTestBase):

    def test_returns_dict_with_expected_top_level_keys(self):
        out = self.checker._compute_checksum()
        self.assertEqual(set(out.keys()), {"checksums", "parallelism_info"})

    def test_skips_non_persistent_buffers(self):
        out = self.checker._compute_checksum()
        names = set(out["checksums"].keys())
        # Normal params and buffers are present.
        self.assertIn("w", names)
        self.assertIn("b", names)
        self.assertIn("running_mean", names)
        # Non-persistent buffer patterns are filtered out.
        self.assertNotIn("rotary_emb_cos_sin_cache", names)
        self.assertNotIn("rotary_emb_freqs_cis", names)
        self.assertNotIn("gate_proj_weight_fp32_cache", names)

    def test_hashes_are_hex_strings(self):
        out = self.checker._compute_checksum()
        for name, h in out["checksums"].items():
            self.assertEqual(len(h), 16, f"unexpected hash length for {name!r}")
            int(h, 16)

    def test_parallelism_info_reflects_runner_state(self):
        info = self.checker._compute_checksum()["parallelism_info"]
        self.assertEqual(info["tp_rank"], 2)
        self.assertEqual(info["tp_size"], 4)
        self.assertEqual(info["dp_rank"], 1)
        self.assertEqual(info["dp_size"], 2)
        self.assertEqual(info["pp_rank"], 0)
        self.assertEqual(info["pp_size"], 1)
        # rank/size come from torch.distributed; default to 0/1 when uninitialized.
        self.assertIn("rank", info)
        self.assertIn("size", info)

    def test_checksum_is_stable_for_unchanged_weights(self):
        first = self.checker._compute_checksum()
        second = self.checker._compute_checksum()
        self.assertEqual(first, second)

    def test_checksum_changes_after_param_mutation(self):
        first = self.checker._compute_checksum()["checksums"]["w"]
        with torch.no_grad():
            self.model.w.data.fill_(99.0)
        second = self.checker._compute_checksum()["checksums"]["w"]
        self.assertNotEqual(first, second)

    def test_validates_against_pydantic_schema(self):
        out = self.checker._compute_checksum()
        info = ChecksumInfo.model_validate(out)
        self.assertIsInstance(info.parallelism_info, ParallelismInfo)


if __name__ == "__main__":
    unittest.main()
