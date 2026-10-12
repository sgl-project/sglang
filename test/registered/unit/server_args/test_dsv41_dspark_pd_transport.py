"""DeepSeek-V4.1 DSpark PD admits the NIXL transfer backend alongside Mooncake."""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.arg_groups.deepseek_v4_hook import validate_deepseek_v41_features
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.ragged_verify import RaggedVerifyMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_RAGGED_VERIFY_MODE = "sglang.srt.speculative.ragged_verify.read_ragged_verify_mode"


def _pd_args(**overrides):
    args = ServerArgs(
        **{
            "model_path": "local-deepseek-v41",
            "attention_backend": "dsv4",
            "speculative_algorithm": "DSPARK",
            "disaggregation_mode": "prefill",
            "disaggregation_transfer_backend": "mooncake",
            "cuda_graph_config": CudaGraphConfig(
                prefill=PhaseConfig(backend=Backend.DISABLED)
            ),
            **overrides,
        }
    )
    args._model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=["DeepseekV4ForCausalLM"], model_type="deepseek_v41"
        ),
        hf_text_config=SimpleNamespace(model_type="deepseek_v41"),
        is_multimodal=False,
    )
    return args


class TestDSV41DSparkPDTransport(CustomTestCase):
    def test_nixl_and_mooncake_are_admitted(self):
        for backend in ("mooncake", "nixl"):
            with (
                self.subTest(backend=backend),
                mock.patch(_RAGGED_VERIFY_MODE, return_value=RaggedVerifyMode.STATIC),
            ):
                validate_deepseek_v41_features(
                    _pd_args(disaggregation_transfer_backend=backend)
                )

    def test_other_transports_are_rejected(self):
        with mock.patch(_RAGGED_VERIFY_MODE, return_value=RaggedVerifyMode.STATIC):
            with self.assertRaisesRegex(ValueError, "Mooncake or NIXL"):
                validate_deepseek_v41_features(
                    _pd_args(disaggregation_transfer_backend="fake")
                )


if __name__ == "__main__":
    unittest.main()
