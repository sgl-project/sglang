from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.srt.kv_canary.token_oracle.oracle import HashOracle
from sglang.srt.kv_canary.token_oracle.oracle_manager import TokenOracleManager
from sglang.srt.kv_canary.token_oracle.sampler import install_oracle_sampler

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


def install_token_oracle_from_env(
    *, server_args: ServerArgs, vocab_size: int
) -> Optional[TokenOracleManager]:
    # Must be called before create_sampler() so the factory is present when the
    # Sampler is first constructed.
    if server_args.sampling_backend != "token_oracle":
        return None

    oracle = HashOracle(vocab_size=vocab_size)
    return install_oracle_sampler(oracle=oracle)
