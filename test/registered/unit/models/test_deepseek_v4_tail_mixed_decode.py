"""A decode DP rank keeps its live rows while joining a prefill tail graph."""

import copy
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import deepseek_v4_replay_tail as tail
from sglang.srt.models.deepseek_v4_mhc import HcState
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestTailMixedDecode(unittest.TestCase):
    def test_only_idle_ranks_drop_rows(self):
        state = HcState(torch.arange(48).reshape(3, 2, 8).float())
        positions, ids = torch.arange(3), torch.arange(3) + 10
        for mode in (ForwardMode.DECODE, ForwardMode.EXTEND, ForwardMode.IDLE):
            with self.subTest(mode=mode):
                obj = SimpleNamespace(run=Mock(return_value=state))
                batch = SimpleNamespace(forward_mode=mode)
                tail.DecoderReplayTail.run_without_tail(
                    obj,
                    state=state,
                    positions=positions,
                    input_ids=ids,
                    input_ids_global=ids,
                    forward_batch=batch,
                    capture_dspark=False,
                    dp_rows=[512, 3, 0, 0],
                )
                call = obj.run.call_args.kwargs
                rows = 0 if mode.is_idle() else 3
                torch.testing.assert_close(call["state"].streams, state.streams[:rows])
                torch.testing.assert_close(call["positions"], positions[:rows])
                torch.testing.assert_close(call["input_ids"], ids[:rows])
                self.assertIs(call["forward_batch"], batch)
                self.assertEqual(call["out_rows"], 3)

    def test_capture_uses_extend_but_breaks_keep_live_decode_metadata(self):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            num_token_non_padded=None,
            req_pool_indices=torch.arange(3),
        )
        context = SimpleNamespace(forward_batch=batch)
        capture = copy.copy(batch)
        rows = torch.zeros(2048, 2, 8)
        obj = tail.DecoderReplayTail.__new__(tail.DecoderReplayTail)
        obj._model = SimpleNamespace(late_layer_vl_routing=False)
        obj._break_context = lambda _: nullcontext()

        def runner(**kw):
            scope = kw["batch_scope"]
            with scope(capture, [rows]):
                self.assertEqual(capture.forward_mode, ForwardMode.EXTEND)
                self.assertEqual(context.forward_batch.forward_mode, ForwardMode.DECODE)
                self.assertEqual(len(context.forward_batch.req_pool_indices), 3)
            self.assertEqual(capture.forward_mode, ForwardMode.DECODE)
            with scope(batch, [rows]):
                self.assertEqual(batch.forward_mode, ForwardMode.DECODE)
            return [rows[:3], None]

        obj.runner = SimpleNamespace(run=runner)
        with (
            patch.object(
                tail, "get_tc_piecewise_forward_context", return_value=context
            ),
            patch.object(
                tail,
                "get_moe_a2a_backend",
                return_value=SimpleNamespace(is_none=lambda: True),
            ),
            patch.object(tail, "enter_late_layer_dp"),
            patch.object(tail, "exit_late_layer_dp"),
        ):
            output, _, _ = obj.run(
                state=HcState(rows[:3]),
                positions=torch.arange(3),
                input_ids=torch.arange(3),
                input_ids_global=None,
                forward_batch=batch,
                capture_dspark=False,
                dp_rows=[512, 3, 0, 0],
            )
        self.assertEqual(output.shape[0], 3)

    def test_capture_mode_restores_on_failure(self):
        live = SimpleNamespace(forward_mode=ForwardMode.DECODE)
        capture = copy.copy(live)
        with patch.object(
            tail,
            "get_tc_piecewise_forward_context",
            return_value=SimpleNamespace(forward_batch=live),
        ):
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                with tail._tail_capture_mode(capture):
                    raise RuntimeError("capture failed")
        self.assertEqual(capture.forward_mode, ForwardMode.DECODE)


if __name__ == "__main__":
    unittest.main()
