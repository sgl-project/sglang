"""Decoder SWA bounded replay inside the breakable prefill CUDA graph is refused
up front, not at graph capture, off CUDA and with prefill context parallelism.

    python -m pytest test/registered/unit/server_args/test_dsv41_bounded_replay_graph_guards.py -v
"""

import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.deepseek_v4_hook import validate_deepseek_v41_features
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _v41_args(**overrides):
    args = ServerArgs(
        **{
            "model_path": "local-deepseek-v41",
            "attention_backend": "dsv4",
            "enable_decoder_swa_bounded_replay": True,
            "cuda_graph_config": CudaGraphConfig(
                prefill=PhaseConfig(backend=Backend.BREAKABLE)
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


class TestBoundedReplayGraphGuards(CustomTestCase):
    @override_platform(is_cuda=False, is_hip=True, is_npu=False, is_musa=False)
    def test_rejects_platforms_other_than_cuda(self):
        with self.assertRaisesRegex(ValueError, "platform other than CUDA"):
            validate_deepseek_v41_features(_v41_args())

    @override_platform(is_cuda=True, is_hip=False, is_npu=False, is_musa=False)
    def test_rejects_prefill_context_parallelism(self):
        with self.assertRaisesRegex(ValueError, "prefill context parallelism"):
            validate_deepseek_v41_features(_v41_args(attn_cp_size=2))
        with self.assertRaisesRegex(ValueError, "prefill context parallelism"):
            validate_deepseek_v41_features(
                _v41_args(enable_prefill_cp=True, cp_strategy="interleave")
            )

    @override_platform(is_cuda=True, is_hip=False, is_npu=False, is_musa=False)
    def test_eager_prefill_keeps_cp_and_hip(self):
        # The rejections are specific to the graph tail: eager prefill with
        # bounded replay still admits CP, and the breakable graph without
        # bounded replay is untouched.
        validate_deepseek_v41_features(
            _v41_args(
                attn_cp_size=2,
                cuda_graph_config=CudaGraphConfig(
                    prefill=PhaseConfig(backend=Backend.DISABLED)
                ),
            )
        )
        validate_deepseek_v41_features(
            _v41_args(enable_decoder_swa_bounded_replay=False)
        )


if __name__ == "__main__":
    unittest.main()
