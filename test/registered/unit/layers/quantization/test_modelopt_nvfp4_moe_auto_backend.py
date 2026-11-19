"""CPU unit tests for how ModelOptNvFp4FusedMoEMethod resolves ``auto``.

With ``--moe-runner-backend auto`` the method used to compute its TRT-LLM
flags from the live global (which still said ``auto``) while create_moe_runner
resolved ``auto`` to FlashInfer TRT-LLM on Blackwell. create_weights and
process_weights_after_loading then prepared a non-TRT-LLM layout (no
g1_scale_c) and apply() dispatched to the TRT-LLM kernel, which read
``layer.g1_scale_c`` and raised AttributeError on the first forward. These
tests pin that the flags and the cached runner backend come from one
resolution, for every entry point of the method.

The platform and device checks are stubbed so the tests stay on CPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization import modelopt_quant
from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod
from sglang.test.test_utils import CustomTestCase


@contextmanager
def _device(capability: tuple, live: MoeRunnerBackend):
    """A CUDA device of ``capability`` with ``live`` as the process-wide backend."""
    is_blackwell = capability >= (10, 0)
    with (
        mock.patch.object(modelopt_quant, "get_moe_runner_backend", return_value=live),
        mock.patch.object(
            modelopt_quant,
            "get_platform",
            return_value=SimpleNamespace(is_blackwell=is_blackwell),
        ),
        mock.patch.object(modelopt_quant, "is_cuda", return_value=True),
        mock.patch.object(
            modelopt_quant, "get_device_capability", return_value=capability
        ),
    ):
        yield


def _method() -> ModelOptNvFp4FusedMoEMethod:
    return ModelOptNvFp4FusedMoEMethod(SimpleNamespace(use_per_token_activation=False))


def _create_runner(method: ModelOptNvFp4FusedMoEMethod) -> None:
    with mock.patch.object(modelopt_quant, "MoeRunner", return_value="runner"):
        method.create_moe_runner(
            SimpleNamespace(),
            MoeRunnerConfig(
                num_experts=8,
                num_local_experts=8,
                hidden_size=64,
                intermediate_size_per_partition=32,
                activation="silu",
                is_gated=True,
            ),
        )


class TestNvFp4MoeAutoBackend(CustomTestCase):
    def test_auto_on_blackwell_prepares_trtllm_weights_for_the_trtllm_runner(self):
        for capability in ((10, 0), (10, 3)):
            with self.subTest(capability=capability):
                with _device(capability, MoeRunnerBackend.AUTO):
                    method = _method()
                    _create_runner(method)
                # The weight prep flags and the cached runner must agree.
                self.assertTrue(method.enable_flashinfer_trtllm_moe)
                self.assertTrue(method.use_flashinfer_trtllm_weight_layout)
                self.assertTrue(method._moe_runner_backend.is_flashinfer_trtllm())

    def test_auto_on_hopper_falls_back_to_marlin(self):
        with _device((9, 0), MoeRunnerBackend.AUTO):
            method = _method()
            _create_runner(method)
        self.assertFalse(method.enable_flashinfer_trtllm_moe)
        self.assertFalse(method.use_flashinfer_trtllm_weight_layout)
        self.assertTrue(method._moe_runner_backend.is_marlin())

    def test_explicit_backend_is_kept(self):
        for live, is_trtllm in (
            (MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED, True),
            (MoeRunnerBackend.FLASHINFER_TRTLLM, True),
            (MoeRunnerBackend.FLASHINFER_CUTLASS, False),
        ):
            with self.subTest(live=live.value):
                with _device((10, 0), live):
                    method = _method()
                    _create_runner(method)
                self.assertIs(method._moe_runner_backend, live)
                self.assertEqual(method.enable_flashinfer_trtllm_moe, is_trtllm)


if __name__ == "__main__":
    unittest.main()
