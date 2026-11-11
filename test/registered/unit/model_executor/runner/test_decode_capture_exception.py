"""Decode capture exceptions must restore the active attention variant."""

import contextlib
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from sglang.srt.model_executor.runner import decode_cuda_graph_runner as decode
from sglang.srt.model_executor.runner_utils.capture_mode import (
    _set_capture_attention_variant,
    get_capture_attention_variant,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestCaptureFailureContract(unittest.TestCase):
    def test_capture_loop_exception_restores_attention_variant(self):
        previous = get_capture_attention_variant()
        self.addCleanup(_set_capture_attention_variant, previous)
        _set_capture_attention_variant(None)
        runner = NS(
            model_runner=NS(device="cpu", gpu_id=0, model=object()),
            capture_bs=[1],
            captured_req_width=1,
            record_nolora_graph=False,
            attention_graph_variants=NS(capture_labels=("candidate_all",)),
            _attention_varies_at=lambda bs: True,
            logits_graph_variants=(None,),
            compile_bs=[],
            _set_profile_trace_tag=Mock(),
            backend=NS(
                prepare_capture_forward=lambda *a, **k: contextlib.nullcontext(Mock())
            ),
            capture_one_shape=Mock(side_effect=RuntimeError("capture failed")),
        )
        with (
            patch.object(decode, "get_available_gpu_memory", return_value=10.0),
            patch.object(
                decode, "get_parallel", return_value=NS(tp_rank=1, tp_group=None)
            ),
            self.assertRaisesRegex(RuntimeError, "capture failed"),
        ):
            decode.DecodeCudaGraphRunner._capture_one_stream(runner)
        self.assertIsNone(get_capture_attention_variant(), "S2_VARIANT_NOT_RESTORED")


if __name__ == "__main__":
    unittest.main()
