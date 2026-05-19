"""install_oracle_sampler registration into sglang's sampler-backend registry."""

from __future__ import annotations

import pytest
import torch

from sglang.srt.kv_cache_canary.mock_model import sampler as oracle_sampler_module
from sglang.srt.kv_cache_canary.mock_model.oracle import HashOracle, ScriptedOracle
from sglang.srt.kv_cache_canary.mock_model.sampler import (
    _OracleSampler,
    install_oracle_sampler,
)
from sglang.srt.layers.sampler import _CUSTOM_SAMPLER_FACTORIES, Sampler
from sglang.srt.server_args import (
    SAMPLING_BACKEND_CHOICES,
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="extra-a-1-gpu-large")


@pytest.fixture(autouse=True)
def _global_server_args() -> None:
    """Sampler.__init__ reads get_global_server_args().enable_nan_detection. Provide a dummy one."""
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))


def test_install_oracle_sampler_registers_oracle_backend() -> None:
    oracle = HashOracle(seed=1, vocab_size=100)

    install_oracle_sampler(oracle=oracle)

    assert "oracle" in _CUSTOM_SAMPLER_FACTORIES
    assert "oracle" in SAMPLING_BACKEND_CHOICES


def test_install_oracle_sampler_factory_produces_oracle_sampler() -> None:
    oracle = HashOracle(seed=2, vocab_size=100)
    install_oracle_sampler(oracle=oracle)

    factory = _CUSTOM_SAMPLER_FACTORIES["oracle"]
    instance = factory()

    assert isinstance(instance, _OracleSampler)
    assert isinstance(instance, Sampler)


def test_install_oracle_sampler_twice_replaces_oracle() -> None:
    oracle_a = HashOracle(seed=10, vocab_size=100)
    oracle_b = ScriptedOracle(table={(0, 0): 77})

    install_oracle_sampler(oracle=oracle_a)
    assert oracle_sampler_module._REGISTERED_ORACLE is oracle_a

    install_oracle_sampler(oracle=oracle_b)
    assert oracle_sampler_module._REGISTERED_ORACLE is oracle_b


def test_install_oracle_sampler_is_idempotent_for_backend_registration() -> None:
    install_oracle_sampler(oracle=HashOracle(seed=3, vocab_size=100))
    first_factory = _CUSTOM_SAMPLER_FACTORIES["oracle"]

    install_oracle_sampler(oracle=HashOracle(seed=4, vocab_size=100))
    second_factory = _CUSTOM_SAMPLER_FACTORIES["oracle"]

    assert first_factory is second_factory


def test_oracle_sampler_forward_raises_without_stashed_req_pool_indices() -> None:
    install_oracle_sampler(oracle=HashOracle(seed=5, vocab_size=100))
    oracle_sampler_module._LAST_REQ_POOL_INDICES = None
    sampler = _CUSTOM_SAMPLER_FACTORIES["oracle"]()

    with pytest.raises(RuntimeError, match="req_pool_indices not stashed"):
        sampler.forward(
            logits_output=_DummyLogitsOutput(),
            sampling_info=None,
            return_logprob=False,
            top_logprobs_nums=[],
            token_ids_logprobs=[],
            positions=None,
        )


class _DummyLogitsOutput:
    """Stand-in for LogitsProcessorOutput; only next_token_logits is accessed in the error path."""

    def __init__(self) -> None:
        self.next_token_logits = torch.zeros((1, 2), dtype=torch.float32)
