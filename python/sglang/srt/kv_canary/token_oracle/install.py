from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.srt.environ import envs
from sglang.srt.kv_canary.token_oracle.oracle import HashOracle
from sglang.srt.kv_canary.token_oracle.oracle_manager import TokenOracleManager
from sglang.srt.kv_canary.token_oracle.sampler import install_oracle_sampler

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


def install_token_oracle_from_env(
    *, server_args: "ServerArgs", vocab_size: int
) -> Optional[TokenOracleManager]:
    """Register the oracle sampler backend when sampling_backend == "oracle".

    Must be called before create_sampler() so the factory is present when the
    Sampler is first constructed.  Returns the TokenOracleManager so the caller
    can attach it to a CanaryRunner after install_canary completes; returns
    None when sampling_backend is anything other than "oracle".
    """
    if server_args.sampling_backend != "oracle":
        return None

    seed = envs.SGLANG_KV_CANARY_ORACLE_SEED.get()
    oracle = HashOracle(seed=seed, vocab_size=vocab_size)
    return install_oracle_sampler(oracle=oracle)
