from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.srt.environ import envs
from sglang.srt.kv_canary.mock_model.oracle import HashOracle
from sglang.srt.kv_canary.mock_model.sampler import (
    OracleSamplerHook,
    install_oracle_sampler,
)

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


def install_mock_model_sampler(
    *, server_args: "ServerArgs", vocab_size: int
) -> Optional[OracleSamplerHook]:
    """Register the oracle sampler backend when mock_model_enabled is set.

    Must be called before create_sampler() so the factory is present when the
    Sampler is first constructed.  Returns the OracleSamplerHook so the caller
    can attach it to a CanaryRunner after install_canary completes; returns
    None when mock_model_enabled is False.
    """
    if not server_args.mock_model_enabled:
        return None

    seed = envs.SGLANG_MOCK_MODEL_ORACLE_SEED.get()
    oracle = HashOracle(seed=seed, vocab_size=vocab_size)
    return install_oracle_sampler(oracle=oracle)
