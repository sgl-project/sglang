"""--lm-head-dtype mxfp8: argument, launch validation, target-only dispatch.

The MXFP8 copy itself needs a Blackwell GPU, so the copy is faked where the
dispatch is under test, and the chunked (re)build runs with a CPU stand-in for
the Triton quantizer that writes the same swizzled scale layout.
"""

import argparse
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from sglang.srt.arg_groups import model_hook
from sglang.srt.layers import logits_processor
from sglang.srt.layers.quantization import mxfp8_lm_head
from sglang.srt.layers.quantization.mxfp8_lm_head import (
    Mxfp8LmHead,
    mxfp8_lm_head_unsupported_reason,
)
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.srt.runtime_context import reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_VOCAB, _HIDDEN = 64, 32


class TestLmHeadDtypeArg(CustomTestCase):
    def _parse(self, *argv):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        return parser.parse_args(["--model-path", "dummy", *argv])

    def test_default_is_off(self):
        self.assertIsNone(ServerArgs(model_path="dummy").lm_head_dtype)
        self.assertIsNone(self._parse().lm_head_dtype)

    def test_mxfp8_parses(self):
        self.assertEqual(self._parse("--lm-head-dtype", "mxfp8").lm_head_dtype, "mxfp8")

    def test_other_dtypes_are_rejected(self):
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            self._parse("--lm-head-dtype", "fp8")


class TestValidateLmHeadDtype(CustomTestCase):
    def _validate(
        self,
        dtype="mxfp8",
        arch="DeepseekV4ForCausalLM",
        is_blackwell=True,
        enable_fp32_lm_head=False,
    ):
        cfg = SimpleNamespace(
            lm_head_dtype=dtype, enable_fp32_lm_head=enable_fp32_lm_head
        )
        facts = SimpleNamespace(is_blackwell=is_blackwell)
        with mock.patch.object(model_hook, "get_platform", return_value=facts):
            model_hook._validate_lm_head_dtype(cfg, arch)

    def test_supported_model_on_blackwell_passes(self):
        self._validate()

    def test_unset_is_a_no_op(self):
        self._validate(dtype=None, arch="LlamaForCausalLM", is_blackwell=False)

    def test_unsupported_model_raises(self):
        with self.assertRaisesRegex(ValueError, "LlamaForCausalLM"):
            self._validate(arch="LlamaForCausalLM")

    def test_non_blackwell_raises(self):
        with self.assertRaisesRegex(ValueError, "Blackwell"):
            self._validate(is_blackwell=False)

    def test_fp32_lm_head_raises(self):
        with self.assertRaisesRegex(ValueError, "--enable-fp32-lm-head"):
            self._validate(enable_fp32_lm_head=True)


class _FakeMxfp8LmHead:
    """Stands in for the GPU copy: tags its logits so the test sees who ran."""

    built = []

    def __init__(self, weight):
        self.weight = weight
        self.rebuilds = 0
        self.nbytes = weight.numel()
        _FakeMxfp8LmHead.built.append(self)

    def rebuild(self):
        self.rebuilds += 1

    def applies(self, hidden_states, weight):
        return weight is self.weight and hidden_states.dtype == torch.bfloat16

    def __call__(self, hidden_states):
        return torch.full(
            (hidden_states.shape[0], self.weight.shape[0]), 7.0, dtype=torch.bfloat16
        )


class _Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(_VOCAB, _HIDDEN, dtype=torch.bfloat16), requires_grad=False
        )


class TestTargetOnlyDispatch(CustomTestCase):
    def setUp(self):
        super().setUp()
        _FakeMxfp8LmHead.built = []
        for patcher in (
            mock.patch.object(mxfp8_lm_head, "Mxfp8LmHead", _FakeMxfp8LmHead),
            # The CPU weights here are otherwise refused for not being on CUDA.
            mock.patch.object(
                mxfp8_lm_head, "mxfp8_lm_head_unsupported_reason", return_value=None
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self._publish(lm_head_dtype="mxfp8")

    def _publish(self, **server_args):
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(model_path="dummy", device="cpu", **server_args), role="test"
        )

    @staticmethod
    def _processor(**config):
        cfg = SimpleNamespace(vocab_size=_VOCAB, final_logit_softcapping=None, **config)
        return logits_processor.LogitsProcessor(cfg, skip_all_gather=True)

    def test_target_runs_mxfp8_and_a_draft_on_the_same_head_stays_bf16(self):
        head = _Head()
        target, draft = self._processor(), self._processor()
        target.prepare_lm_head_dtype(head)
        hidden = torch.randn(3, _HIDDEN, dtype=torch.bfloat16)

        self.assertTrue(torch.all(target._compute_lm_head(hidden, head) == 7.0))
        self.assertIsNone(draft._mxfp8_lm_head)
        torch.testing.assert_close(
            draft._compute_lm_head(hidden, head), hidden @ head.weight.T
        )

    def test_dspark_draft_projection_is_unchanged(self):
        from sglang.srt.models.dspark import project_through_lm_head

        head = _Head()
        head.quant_method = None
        self._processor().prepare_lm_head_dtype(head)
        hidden = torch.randn(3, _HIDDEN, dtype=torch.bfloat16)
        torch.testing.assert_close(
            project_through_lm_head(hidden, head), hidden @ head.weight.T
        )

    def test_unset_dtype_builds_nothing(self):
        self._publish()
        head, processor = _Head(), self._processor()
        processor.prepare_lm_head_dtype(head)
        self.assertIsNone(processor._mxfp8_lm_head)
        self.assertEqual(_FakeMxfp8LmHead.built, [])

    def test_reload_rebuilds_in_place_and_a_new_weight_gets_a_new_copy(self):
        head, processor = _Head(), self._processor()
        processor.prepare_lm_head_dtype(head)
        copy = processor._mxfp8_lm_head
        processor.prepare_lm_head_dtype(head)
        self.assertIs(processor._mxfp8_lm_head, copy)
        self.assertEqual(copy.rebuilds, 1)

        # A swapped weight (set_embed_and_head) is not the one the copy holds.
        head.weight = nn.Parameter(
            torch.randn(_VOCAB, _HIDDEN, dtype=torch.bfloat16), requires_grad=False
        )
        hidden = torch.randn(2, _HIDDEN, dtype=torch.bfloat16)
        torch.testing.assert_close(
            processor._compute_lm_head(hidden, head), hidden @ head.weight.T
        )
        processor.prepare_lm_head_dtype(head)
        self.assertIsNot(processor._mxfp8_lm_head, copy)
        self.assertIs(processor._mxfp8_lm_head.weight, head.weight)

    def test_fp32_logits_keep_the_lm_head_dtype(self):
        processor = self._processor(enable_lm_head_fp32=True)
        processor.prepare_lm_head_dtype(_Head())
        self.assertIsNone(processor._mxfp8_lm_head)

    def test_quantized_lm_head_keeps_its_kernel(self):
        head, processor = _Head(), self._processor()
        with mock.patch.object(
            logits_processor, "should_apply_lm_head_quant_method", return_value=True
        ):
            processor.prepare_lm_head_dtype(head)
        self.assertIsNone(processor._mxfp8_lm_head)


class TestDeepseekV4BuildsOnTheTargetOnly(CustomTestCase):
    def _post_load(self, *, is_nextn, is_last_rank=True):
        model = SimpleNamespace(
            wo_a_fp8=False,
            modules=lambda: [],
            model=SimpleNamespace(start_layer=0, end_layer=0, layers=[]),
            pp_group=SimpleNamespace(is_last_rank=is_last_rank),
            lm_head=object(),
            logits_processor=mock.Mock(),
        )
        DeepseekV4ForCausalLM.post_load_weights(model, is_nextn=is_nextn)
        return model

    def test_target_prepares_its_own_lm_head(self):
        model = self._post_load(is_nextn=False)
        model.logits_processor.prepare_lm_head_dtype.assert_called_once_with(
            model.lm_head
        )

    def test_nextn_draft_does_not(self):
        model = self._post_load(is_nextn=True)
        model.logits_processor.prepare_lm_head_dtype.assert_not_called()

    def test_non_last_pipeline_rank_does_not(self):
        model = self._post_load(is_nextn=False, is_last_rank=False)
        model.logits_processor.prepare_lm_head_dtype.assert_not_called()


def _swizzled_reference_quantize(x):
    """CPU stand-in for mxfp8_quantize_swizzled: same scale layout, with each
    scale byte taken from the first value of its 32-element group."""
    rows, k = x.shape
    groups = k // 32
    col_blocks = -(-groups // 4)
    scale = torch.zeros(mxfp8_lm_head._swizzled_scale_bytes(rows, k), dtype=torch.uint8)
    first = x.float()[:, ::32].to(torch.int64) % 251
    for r in range(rows):
        for c in range(groups):
            tile = (r // 128) * col_blocks + c // 4
            offset = tile * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4
            scale[offset] = int(first[r, c])
    return x.to(torch.float8_e4m3fn), scale


def _reference_quantizer():
    # A stand-in module, so the test needs neither a GPU nor Triton.
    return mock.patch.dict(
        sys.modules,
        {
            "sglang.kernels.ops.quantization.mxfp8_swizzled_triton": SimpleNamespace(
                mxfp8_quantize_swizzled=_swizzled_reference_quantize
            )
        },
    )


class TestChunkedRebuild(CustomTestCase):
    def _build(self, weight, chunk_rows):
        with (
            _reference_quantizer(),
            mock.patch.object(
                mxfp8_lm_head, "mxfp8_lm_head_unsupported_reason", return_value=None
            ),
            mock.patch.object(mxfp8_lm_head, "_REBUILD_ROWS", chunk_rows),
        ):
            return Mxfp8LmHead(weight)

    def test_chunks_match_one_pass_and_rebuild_keeps_buffers(self):
        # 300 rows: two full 128-row tiles and a padded tail tile.
        rows = torch.arange(300, dtype=torch.float32)[:, None] % 97
        weight = (rows + torch.arange(64) / 64).to(torch.bfloat16)
        whole = self._build(weight, chunk_rows=1024)
        chunked = self._build(weight, chunk_rows=128)
        self.assertTrue(torch.equal(chunked.weight_scale, whole.weight_scale))
        self.assertTrue(
            torch.equal(
                chunked.qweight.view(torch.uint8), whole.qweight.view(torch.uint8)
            )
        )

        pointers = (chunked.qweight.data_ptr(), chunked.weight_scale.data_ptr())
        weight.mul_(-1)
        expected = self._build(weight.clone(), chunk_rows=1024)
        with (
            _reference_quantizer(),
            mock.patch.object(mxfp8_lm_head, "_REBUILD_ROWS", 128),
        ):
            chunked.rebuild()
        self.assertEqual(
            (chunked.qweight.data_ptr(), chunked.weight_scale.data_ptr()), pointers
        )
        self.assertTrue(torch.equal(chunked.weight_scale, expected.weight_scale))

    def test_scale_bytes(self):
        # DeepSeek-V4 head: 129280 rows, hidden 5120 -> 160 scale columns.
        self.assertEqual(
            mxfp8_lm_head._swizzled_scale_bytes(129280, 5120), 129280 * 160
        )
        self.assertEqual(mxfp8_lm_head._swizzled_scale_bytes(1, 64), 128 * 4)


class TestUnsupportedReason(CustomTestCase):
    def test_reasons(self):
        bf16 = torch.bfloat16
        cases = {
            "no weight": (None, "no weight"),
            "fp32": (torch.zeros(64, 64), "not torch.bfloat16"),
            "3d": (torch.zeros(2, 64, 64, dtype=bf16), "3 dims"),
            "k": (torch.zeros(64, 48, dtype=bf16), "hidden size 48"),
            "rows": (torch.zeros(40, 64, dtype=bf16), "40 rows"),
            "cpu": (torch.zeros(64, 64, dtype=bf16), "CUDA"),
        }
        for name, (weight, expected) in cases.items():
            with self.subTest(name):
                self.assertIn(expected, mxfp8_lm_head_unsupported_reason(weight))


if __name__ == "__main__":
    unittest.main()
