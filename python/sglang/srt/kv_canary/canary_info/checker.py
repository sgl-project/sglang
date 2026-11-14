from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Final, Optional, Sequence

from sglang.kernels.ops.kv_canary.consts import (
    U64_MASK,
    canary_chain_extend_hash,
    canary_chain_initial_hash,
    canary_info,
)
from sglang.srt.kv_canary.canary_info.digest import sampling_digest

if TYPE_CHECKING:
    from sglang.srt.sampling.sampling_params import SamplingParams

CANARY_INFO_MISMATCH_CODE: Final[str] = "kv_canary_info_mismatch"

_CANARY_INFO_MISMATCH_LOG_PREFIX: Final[str] = CANARY_INFO_MISMATCH_CODE + "="
_CANARY_INFO_MISMATCH_LINE_RE = re.compile(
    re.escape(_CANARY_INFO_MISMATCH_LOG_PREFIX) + r"(\S+)"
)


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryInfoMismatch:
    rid: str
    position: int
    token_id: Optional[int]
    expected: Optional[int]
    actual: Optional[int]

    def format(self) -> str:
        return _CANARY_INFO_MISMATCH_LOG_PREFIX + json.dumps(
            asdict(self), separators=(",", ":")
        )

    @classmethod
    def find_all(cls, text: str) -> list[CanaryInfoMismatch]:
        return [
            cls(**json.loads(match.group(1)))
            for match in _CANARY_INFO_MISMATCH_LINE_RE.finditer(text)
        ]


class CanaryInfoChecker:
    def __init__(
        self,
        *,
        rid: str,
        sampling_params: SamplingParams,
        prompt_ids: Sequence[int],
    ) -> None:
        if len(prompt_ids) == 0:
            raise ValueError("kv-canary: CanaryInfoChecker requires a non-empty prompt")

        self._rid = rid
        self._sampling_digest = sampling_digest(sampling_params)
        self._canary_prefix_hash = canary_chain_initial_hash()
        self._position = 0
        for token_id in prompt_ids:
            self._extend(int(token_id))

    @property
    def rid(self) -> str:
        return self._rid

    @property
    def position(self) -> int:
        return self._position

    def expected(self, token_id: int) -> int:
        return canary_info(self._canary_prefix_hash, token_id, self._sampling_digest)

    def advance(
        self, token_id: int, value: Optional[int]
    ) -> Optional[CanaryInfoMismatch]:
        token_id = int(token_id)
        expected = self.expected(token_id)
        actual = None if value is None else int(value) & U64_MASK
        if actual != expected:
            return CanaryInfoMismatch(
                rid=self._rid,
                position=self._position,
                token_id=token_id,
                expected=expected,
                actual=actual,
            )
        self._extend(token_id)
        return None

    def advance_many(
        self,
        token_ids: Sequence[int],
        values: Sequence[Optional[int]],
    ) -> Optional[CanaryInfoMismatch]:
        for token_id, value in zip(token_ids, values, strict=True):
            if (mismatch := self.advance(token_id, value)) is not None:
                return mismatch
        return None

    def _extend(self, token_id: int) -> None:
        self._canary_prefix_hash = canary_chain_extend_hash(
            self._canary_prefix_hash, token_id, self._position
        )
        self._position += 1
