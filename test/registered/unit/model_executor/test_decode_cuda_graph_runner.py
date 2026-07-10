import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner import (
    MultiLayerEagleDraftExtendCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _NoSpecAlgorithm:
    def is_eagle(self):
        return False

    def is_standalone(self):
        return False

    def is_dflash(self):
        return False

    def is_ngram(self):
        return False


class _RecordingBackend:
    def __init__(self, supported_keys):
        self.supported_keys = set(supported_keys)
        self.seen_keys = []

    def can_run(self, forward_batch, shape_key):
        self.seen_keys.append(shape_key)
        return shape_key in self.supported_keys


class TestDecodeCudaGraphRunner(CustomTestCase):
    def _build_runner(self, supported_keys, *, enable_pdmux=False):
        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.backend = _RecordingBackend(supported_keys)
        runner.capture_hidden_mode = CaptureHiddenMode.NULL
        runner.disable_padding = True
        runner.enable_pdmux = enable_pdmux
        runner.enable_two_batch_overlap = False
        runner.is_encoder_decoder = False
        runner.num_tokens_per_bs = 1
        runner.record_nolora_graph = False
        runner.require_mlp_sync = False
        runner.require_mlp_tp_gather = False
        runner.model_runner = SimpleNamespace(spec_algorithm=_NoSpecAlgorithm())
        return runner

    def _build_forward_batch(self, batch_size, *, lora_ids=None):
        return SimpleNamespace(
            batch_size=batch_size,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            lora_ids=lora_ids,
            replace_embeds=None,
            spec_info=None,
        )

    def test_disable_padding_uses_exact_shape_key(self):
        expected_key = ShapeKey(size=16)
        runner = self._build_runner([expected_key])

        self.assertTrue(runner.can_run_graph(self._build_forward_batch(16)))
        self.assertFalse(runner.can_run_graph(self._build_forward_batch(15)))
        self.assertEqual(
            runner.backend.seen_keys,
            [expected_key, ShapeKey(size=15)],
        )

    @patch(
        "sglang.srt.model_executor.runner.decode_cuda_graph_runner."
        "get_current_stream_idx",
        return_value=3,
    )
    def test_disable_padding_uses_pdmux_stream_in_shape_key(self, _):
        expected_key = ShapeKey(size=16, stream_idx=3)
        runner = self._build_runner([expected_key], enable_pdmux=True)

        self.assertTrue(runner.can_run_graph(self._build_forward_batch(16)))
        self.assertEqual(runner.backend.seen_keys, [expected_key])

    def test_speculative_runners_use_exact_shape_key(self):
        runner_classes = (
            EAGLEDraftCudaGraphRunner,
            EAGLEDraftExtendCudaGraphRunner,
            MultiLayerEagleDraftExtendCudaGraphRunner,
        )

        for runner_class in runner_classes:
            with self.subTest(runner_class=runner_class.__name__):
                expected_key = ShapeKey(size=16)
                runner = runner_class.__new__(runner_class)
                runner.backend = _RecordingBackend([expected_key])
                runner.disable_padding = True
                runner.require_mlp_sync = False
                runner.require_mlp_tp_gather = False
                runner.model_runner = SimpleNamespace(spec_algorithm=_NoSpecAlgorithm())

                exact_batch = SimpleNamespace(
                    batch_size=16,
                    seq_lens=SimpleNamespace(numel=lambda: 16),
                )
                non_exact_batch = SimpleNamespace(
                    batch_size=15,
                    seq_lens=SimpleNamespace(numel=lambda: 15),
                )

                self.assertTrue(runner.can_run_graph(exact_batch))
                self.assertFalse(runner.can_run_graph(non_exact_batch))
                self.assertEqual(
                    runner.backend.seen_keys,
                    [expected_key, ShapeKey(size=15)],
                )

    def test_disable_padding_uses_lora_variant_in_shape_key(self):
        expected_key = ShapeKey(size=16, variant_label="lora")
        runner = self._build_runner([expected_key])
        runner.record_nolora_graph = True

        self.assertTrue(
            runner.can_run_graph(
                self._build_forward_batch(16, lora_ids=["adapter", None])
            )
        )
        self.assertEqual(runner.backend.seen_keys, [expected_key])


if __name__ == "__main__":
    unittest.main()
