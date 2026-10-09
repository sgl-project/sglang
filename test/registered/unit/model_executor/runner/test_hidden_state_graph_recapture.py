import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
    get_server_return_hidden_states_mode,
)
from sglang.srt.model_executor.runner.base_cuda_graph_runner import BaseCudaGraphRunner
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHiddenStateGraphRecapture(CustomTestCase):
    def test_prefill_capture_uses_resolved_speculative_hidden_requirement(self):
        for backend in (Backend.FULL, Backend.BREAKABLE, Backend.TC_PIECEWISE):
            for algorithm, aux_hidden, draft, server_mode, expected in (
                (
                    SpeculativeAlgorithm.DSPARK,
                    False,
                    False,
                    CaptureHiddenMode.NULL,
                    CaptureHiddenMode.NULL,
                ),
                (
                    SpeculativeAlgorithm.DSPARK,
                    False,
                    False,
                    CaptureHiddenMode.LAST,
                    CaptureHiddenMode.LAST,
                ),
                (
                    SpeculativeAlgorithm.DSPARK,
                    False,
                    False,
                    CaptureHiddenMode.FULL,
                    CaptureHiddenMode.FULL,
                ),
                (
                    SpeculativeAlgorithm.DSPARK,
                    True,
                    False,
                    CaptureHiddenMode.NULL,
                    CaptureHiddenMode.FULL,
                ),
                (
                    SpeculativeAlgorithm.DFLASH,
                    True,
                    False,
                    CaptureHiddenMode.NULL,
                    CaptureHiddenMode.FULL,
                ),
                (
                    SpeculativeAlgorithm.DSPARK,
                    False,
                    True,
                    CaptureHiddenMode.NULL,
                    CaptureHiddenMode.NULL,
                ),
                (
                    SpeculativeAlgorithm.NONE,
                    False,
                    False,
                    CaptureHiddenMode.LAST,
                    CaptureHiddenMode.LAST,
                ),
                (
                    SpeculativeAlgorithm.EAGLE,
                    False,
                    False,
                    CaptureHiddenMode.NULL,
                    (
                        CaptureHiddenMode.FULL
                        if backend == Backend.BREAKABLE
                        else CaptureHiddenMode.NULL
                    ),
                ),
                (
                    SpeculativeAlgorithm.EAGLE,
                    False,
                    True,
                    CaptureHiddenMode.NULL,
                    (
                        CaptureHiddenMode.LAST
                        if backend == Backend.BREAKABLE
                        else CaptureHiddenMode.NULL
                    ),
                ),
            ):
                model_runner = SimpleNamespace(
                    model=SimpleNamespace(),
                    model_config=SimpleNamespace(is_multimodal=False),
                    server_args=SimpleNamespace(
                        enable_lora=False,
                        cuda_graph_config=SimpleNamespace(
                            prefill=SimpleNamespace(backend=backend, bs=[16])
                        ),
                    ),
                    is_generation=True,
                    is_draft_worker=draft,
                    spec_algorithm=algorithm,
                    spec_aux_config=SimpleNamespace(
                        dflash_use_aux_hidden_state=aux_hidden
                    ),
                    req_to_token_pool=SimpleNamespace(size=4),
                )
                runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
                with (
                    self.subTest(
                        backend=backend,
                        algorithm=algorithm,
                        aux_hidden=aux_hidden,
                        draft=draft,
                        server_mode=server_mode,
                    ),
                    patch.object(
                        BaseCudaGraphRunner,
                        "__init__",
                        autospec=True,
                        side_effect=lambda value, _, mode=server_mode: setattr(
                            value, "return_hidden_states_mode", mode
                        ),
                    ),
                    patch.object(
                        PrefillCudaGraphRunner,
                        "_is_mamba_track_enabled",
                        side_effect=RuntimeError("capture mode selected"),
                    ),
                ):
                    # Stop before CUDA allocation, after the real constructor
                    # selects the mode used to build every captured shape.
                    with self.assertRaisesRegex(RuntimeError, "capture mode selected"):
                        runner.__init__(model_runner)
                    self.assertEqual(runner.capture_hidden_mode, expected)

    def test_server_mode_sets_graph_capture_ceiling(self):
        disabled = SimpleNamespace(
            enable_return_hidden_states=False,
            return_hidden_states_mode=None,
        )
        last = SimpleNamespace(
            enable_return_hidden_states=True,
            return_hidden_states_mode="last",
        )
        full = SimpleNamespace(
            enable_return_hidden_states=True,
            return_hidden_states_mode="full",
        )

        self.assertEqual(
            get_server_return_hidden_states_mode(disabled),
            CaptureHiddenMode.NULL,
        )
        self.assertEqual(
            get_server_return_hidden_states_mode(last),
            CaptureHiddenMode.LAST,
        )
        self.assertEqual(
            get_server_return_hidden_states_mode(full),
            CaptureHiddenMode.FULL,
        )

    @staticmethod
    def _make_runner(runner_cls, capture_hidden_mode):
        runner = runner_cls.__new__(runner_cls)
        runner.capture_hidden_mode = capture_hidden_mode
        runner.backend = Mock()
        runner.capture = Mock()
        return runner

    @staticmethod
    def _make_forward_batch(capture_hidden_mode):
        return SimpleNamespace(
            capture_hidden_mode=capture_hidden_mode,
            spec_info=None,
        )

    @staticmethod
    def _make_prefill_runner_for_can_run(capture_hidden_mode):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._is_full_backend = False
        runner.prefill_backend_name = Backend.BREAKABLE
        runner.has_mha_companion_layers = False
        runner.enable_lora = False
        runner._capture_chunked_prefix = False
        runner.capture_hidden_mode = capture_hidden_mode
        runner.capture_num_tokens = [4]
        runner.max_num_tokens = 4
        return runner

    @staticmethod
    def _make_prefill_forward_batch(capture_hidden_mode, spec_capture_hidden_mode):
        return SimpleNamespace(
            batch_size=1,
            input_embeds=None,
            replace_embeds=None,
            forward_mode=ForwardMode.EXTEND,
            capture_hidden_mode=capture_hidden_mode,
            spec_info=SimpleNamespace(capture_hidden_mode=spec_capture_hidden_mode),
            global_num_tokens_cpu=None,
            return_logprob=False,
            extend_prefix_lens_cpu=None,
            input_ids=list(range(4)),
        )

    def test_stronger_graph_is_reused_for_weaker_modes(self):
        runner = self._make_runner(DecodeCudaGraphRunner, CaptureHiddenMode.FULL)

        for required_mode in (
            CaptureHiddenMode.FULL,
            CaptureHiddenMode.NULL,
            CaptureHiddenMode.LAST,
            CaptureHiddenMode.FULL,
            CaptureHiddenMode.NULL,
        ):
            with self.subTest(required_mode=required_mode):
                runner._validate_capture_hidden_mode(
                    self._make_forward_batch(required_mode)
                )

        self.assertEqual(runner.capture_hidden_mode, CaptureHiddenMode.FULL)
        runner.backend.cleanup.assert_not_called()
        runner.capture.assert_not_called()

    def test_graph_does_not_recapture_above_fixed_server_mode(self):
        for runner_cls in (
            DecodeCudaGraphRunner,
            PrefillCudaGraphRunner,
            CPUGraphRunner,
        ):
            runner = self._make_runner(runner_cls, CaptureHiddenMode.NULL)

            with self.subTest(runner_cls=runner_cls), self.assertRaisesRegex(
                RuntimeError,
                "exceeds the fixed (CUDA|CPU) graph capture mode",
            ):
                runner._validate_capture_hidden_mode(
                    self._make_forward_batch(CaptureHiddenMode.LAST)
                )

            self.assertEqual(runner.capture_hidden_mode, CaptureHiddenMode.NULL)
            runner.backend.cleanup.assert_not_called()
            runner.capture.assert_not_called()

    def test_spec_worker_override_is_the_effective_runtime_mode(self):
        runner = self._make_prefill_runner_for_can_run(CaptureHiddenMode.LAST)
        forward_batch = self._make_prefill_forward_batch(
            CaptureHiddenMode.LAST,
            CaptureHiddenMode.FULL,
        )

        self.assertTrue(runner.can_run_graph(forward_batch))
        for runner_cls in (
            DecodeCudaGraphRunner,
            PrefillCudaGraphRunner,
            CPUGraphRunner,
        ):
            graph_runner = self._make_runner(runner_cls, CaptureHiddenMode.LAST)
            with self.subTest(runner_cls=runner_cls):
                graph_runner._validate_capture_hidden_mode(forward_batch)

    def test_prefill_graph_falls_back_for_stronger_effective_mode(self):
        runner = self._make_prefill_runner_for_can_run(CaptureHiddenMode.LAST)
        forward_batch = self._make_prefill_forward_batch(
            CaptureHiddenMode.FULL,
            CaptureHiddenMode.LAST,
        )

        self.assertFalse(runner.can_run_graph(forward_batch))

    def test_prefill_graph_accepts_weaker_spec_mode(self):
        runner = self._make_prefill_runner_for_can_run(CaptureHiddenMode.FULL)
        forward_batch = self._make_prefill_forward_batch(
            CaptureHiddenMode.NULL,
            CaptureHiddenMode.LAST,
        )

        self.assertTrue(runner.can_run_graph(forward_batch))


if __name__ == "__main__":
    unittest.main()
