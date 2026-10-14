"""Supported graph and bounded-replay combinations for DSv4.1 TRT-LLM."""

import sys
from types import SimpleNamespace

import pytest

from sglang.srt.arg_groups.deepseek_v4_hook import validate_deepseek_v41_features
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _args(backend, **overrides):
    args = ServerArgs(
        model_path="dummy",
        dsv4_attn_backend="trtllm",
        cuda_graph_config=CudaGraphConfig(
            prefill=PhaseConfig(backend=backend, max_seq_len=16384)
        ),
        max_running_requests=8,
        chunked_prefill_size=1024,
        **overrides,
    )
    args._model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="deepseek_v41")
    )
    return args


@pytest.mark.parametrize("backend", [Backend.DISABLED, Backend.BREAKABLE])
@pytest.mark.parametrize("bounded", [False, True])
def test_trtllm_supports_eager_and_breakable_prefill(backend, bounded):
    with override_platform(is_cuda=True, is_hip=False):
        validate_deepseek_v41_features(
            _args(backend, enable_decoder_swa_bounded_replay=bounded)
        )


def test_trtllm_still_rejects_full_prefill_graph():
    with pytest.raises(ValueError, match="disabled or breakable"):
        validate_deepseek_v41_features(_args(Backend.FULL))


def test_trtllm_still_rejects_encoder_replay():
    with (
        override_platform(is_cuda=True, is_hip=False),
        pytest.raises(ValueError, match="TRT-LLM with encoder SWA bounded replay"),
    ):
        validate_deepseek_v41_features(
            _args(Backend.DISABLED, enable_encoder_swa_bounded_replay=True)
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
