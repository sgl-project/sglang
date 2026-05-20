"""install_oracle_sampler registration into sglang's sampler-backend registry.

Instantiating the registered _OracleSampler factory requires a live distributed (TP) group
plus a populated global ServerArgs, so the forward-path behavior of _OracleSampler is covered
by the e2e harness rather than this unit file. Here we only assert the registration-side
contract: the backend name shows up in the registry / choice set, and second install replaces
the factory with one bound to the new oracle.
"""

from __future__ import annotations

from sglang.srt.kv_canary.token_oracle.oracle import HashOracle
from sglang.srt.kv_canary.token_oracle.sampler import install_oracle_sampler
from sglang.srt.layers.sampler import _CUSTOM_SAMPLER_FACTORIES
from sglang.srt.server_args import SAMPLING_BACKEND_CHOICES
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="extra-a-1-gpu-large")


def test_install_oracle_sampler_registers_oracle_backend() -> None:
    oracle = HashOracle(seed=1, vocab_size=100)

    hook = install_oracle_sampler(oracle=oracle)

    assert "oracle" in _CUSTOM_SAMPLER_FACTORIES
    assert "oracle" in SAMPLING_BACKEND_CHOICES
    assert hook.oracle is oracle


def test_install_oracle_sampler_returns_hook_owning_the_oracle() -> None:
    oracle = HashOracle(seed=2, vocab_size=100)

    hook = install_oracle_sampler(oracle=oracle)

    assert hook.oracle is oracle


def test_install_oracle_sampler_twice_returns_distinct_hooks_with_replaced_oracle() -> (
    None
):
    oracle_a = HashOracle(seed=10, vocab_size=100)
    oracle_b = HashOracle(seed=99, vocab_size=100)

    hook_a = install_oracle_sampler(oracle=oracle_a)
    factory_a = _CUSTOM_SAMPLER_FACTORIES["oracle"]
    assert hook_a.oracle is oracle_a

    hook_b = install_oracle_sampler(oracle=oracle_b)
    factory_b = _CUSTOM_SAMPLER_FACTORIES["oracle"]
    assert hook_b.oracle is oracle_b
    assert hook_a is not hook_b
    assert factory_a is not factory_b
