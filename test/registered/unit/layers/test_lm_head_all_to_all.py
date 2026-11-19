"""SGLANG_OPT_TP_LM_HEAD_ALL_TO_ALL_FP32: the TP LM-head all-to-all writes its bf16
blocks straight into fp32 logits, with the stock path's values."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers import lm_head_all_to_all
from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_parallel, override_platform
from sglang.srt.utils import log_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_TP, _ROWS, _SHARD, _VOCAB = 4, 3, 5, 18


def _enabled(*, softcap=None, use_all_to_all=True):
    return lm_head_all_to_all.fp32_output_enabled(
        use_tp_lm_head_all_to_all=use_all_to_all, final_logit_softcapping=softcap
    )


def _gathered_cast(gathered, final_logit_softcapping, output):
    """CPU stand-in for the Triton op's contract: rank-major [world, tokens, shard]
    to vocab-major fp32 output[tokens, :vocab]."""
    assert final_logit_softcapping is None
    shard = gathered.shape[2]
    cols = torch.arange(output.shape[1])
    output.copy_(gathered[cols // shard, :, cols % shard].T.float())
    return output


class _CopyingTpGroup:
    """Stands in for the exchange; both arms of a comparison see the same blocks."""

    cpu_group = None
    world_size = _TP

    def all_to_all_single(self, output, input_):
        output.copy_(input_)


def _a2a_blocks(dtype=torch.bfloat16):
    torch.manual_seed(0)
    blocks = (torch.randn(_TP * _ROWS, _SHARD) * 8).to(dtype)
    blocks[0, :4] = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0])
    return blocks


class TestLmHeadAllToAllFp32Gather(CustomTestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(override_platform(is_blackwell=True))
        self.op = stack.enter_context(
            patch.object(
                lm_head_all_to_all,
                "gathered_softcap_logits",
                side_effect=_gathered_cast,
            )
        )
        log_utils.print_info_once.cache_clear()
        self.addCleanup(log_utils.print_info_once.cache_clear)

    def _processor(self, *, child, config=None, logit_scale=None):
        parallel = SimpleNamespace(
            enable_dp_lm_head=False,
            enable_tp_lm_head_all_to_all=True,
            tp_size=_TP,
            attn_dp_size=_TP,
            tp_group=_CopyingTpGroup(),
        )
        with ExitStack() as stack:
            stack.enter_context(published_topology())
            # MultimemAllGatherer resolves TP outside the patched module lookup.
            stack.enter_context(
                get_parallel().override(
                    tp_group=SimpleNamespace(world_size=1, cpu_group=None)
                )
            )
            stack.enter_context(
                envs.SGLANG_OPT_TP_LM_HEAD_ALL_TO_ALL_FP32.override(child)
            )
            stack.enter_context(
                patch(
                    "sglang.srt.layers.logits_processor.get_parallel",
                    return_value=parallel,
                )
            )
            processor = LogitsProcessor(
                config or SimpleNamespace(vocab_size=_VOCAB), logit_scale=logit_scale
            )
        return processor, parallel

    def _logits(self, processor, parallel, blocks, *, use_logits_buffer=True):
        """Run _get_logits from the local LM-head output to the fp32 logits; the
        hidden-state gather and the GEMM are faked."""
        buffer = torch.full((_ROWS, _VOCAB), 7.0)
        metadata = LogitsMetadata(
            forward_mode=ForwardMode.DECODE,
            next_token_logits_buffer=buffer,
            global_num_tokens_for_logprob_cpu=[_ROWS] * _TP,
        )
        hidden = torch.zeros((_ROWS, 8))
        with (
            published_topology(),
            patch(
                "sglang.srt.layers.logits_processor.get_parallel",
                return_value=parallel,
            ),
            patch.object(
                processor,
                "_gather_dp_attn_hidden_states",
                return_value=(torch.zeros((_TP * _ROWS, 8)), hidden),
            ),
            patch.object(processor, "_compute_lm_head", return_value=blocks.clone()),
        ):
            logits = processor._get_logits(
                hidden,
                SimpleNamespace(tp_size=_TP),
                metadata,
                use_logits_buffer=use_logits_buffer,
            )
        return logits, buffer

    def test_each_structural_check_declines(self):
        """The child needs the all-to-all head and SM100, and a softcap declines it,
        also where a softcapped model's processor checks its own config."""
        with envs.SGLANG_OPT_TP_LM_HEAD_ALL_TO_ALL_FP32.override(True):
            self.assertTrue(_enabled())
            self.assertFalse(_enabled(use_all_to_all=False))
            self.assertFalse(_enabled(softcap=30.0))
            # The stock path applies no softcap of 0 either.
            self.assertTrue(_enabled(softcap=0.0))
            with override_platform(is_blackwell=False):
                self.assertFalse(_enabled())
        softcapped = SimpleNamespace(vocab_size=_VOCAB, final_logit_softcapping=30.0)
        processors = [
            self._processor(child=True)[0],
            self._processor(child=True, config=softcapped)[0],
        ]
        self.assertEqual(
            [p.tp_lm_head_all_to_all_fp32 for p in processors], [True, False]
        )

    def test_all_to_all_head_writes_stock_fp32_logits_into_the_buffer(self):
        """The a2a head's bf16 blocks land in the graph's fp32 logits buffer in one op
        call, bit for bit the stock reassembly and cast."""
        blocks = _a2a_blocks()
        stock, stock_parallel = self._processor(child=False)
        want, _ = self._logits(stock, stock_parallel, blocks)
        self.op.assert_not_called()

        fused, parallel = self._processor(child=True)
        with patch(
            "sglang.srt.layers.logits_processor."
            "_reassemble_tp_lm_head_all_to_all_output",
            side_effect=AssertionError("stock reassembly ran"),
        ):
            got, buffer = self._logits(fused, parallel, blocks)

        self.assertIs(got, buffer)
        self.assertEqual(got.dtype, torch.float32)
        self.assertTrue(torch.equal(got.view(torch.int32), want.view(torch.int32)))
        call = self.op.call_args_list[0].kwargs
        self.assertEqual(tuple(call["gathered"].shape), (_TP, _ROWS, _SHARD))
        self.assertIsNone(call["final_logit_softcapping"])
        self.assertIs(call["output"], buffer)

    def test_other_calls_give_the_stock_logits(self):
        """A logit scale, given at construction or set later as a model may do, and
        fp32 blocks keep the stock path; without a logits buffer the op writes a new
        tensor. Each call gives the stock logits bit for bit."""
        for case, dtype, logit_scale, set_later, use_buffer, op_calls in (
            ("logit scale at init", torch.bfloat16, 0.5, False, True, 0),
            ("logit scale set later", torch.bfloat16, 0.5, True, True, 0),
            ("fp32 blocks", torch.float32, None, False, True, 0),
            ("no logits buffer", torch.bfloat16, None, False, False, 1),
        ):
            with self.subTest(case=case):
                self.op.reset_mock()
                blocks = _a2a_blocks(dtype)
                stock, stock_parallel = self._processor(
                    child=False, logit_scale=logit_scale
                )
                want, _ = self._logits(
                    stock, stock_parallel, blocks, use_logits_buffer=use_buffer
                )
                fused, parallel = self._processor(
                    child=True, logit_scale=None if set_later else logit_scale
                )
                if set_later:
                    fused.logit_scale = logit_scale
                got, buffer = self._logits(
                    fused, parallel, blocks, use_logits_buffer=use_buffer
                )
                self.assertEqual(self.op.call_count, op_calls)
                if not use_buffer:
                    self.assertIsNot(got, buffer)
                self.assertTrue(
                    torch.equal(got.view(torch.int32), want.view(torch.int32))
                )


if __name__ == "__main__":
    unittest.main()
