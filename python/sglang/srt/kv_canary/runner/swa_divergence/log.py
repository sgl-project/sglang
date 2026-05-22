from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Optional

_SWA_DIVERGENCE_LOG_PREFIX: str = "kv_canary_swa_divergence="

_SWA_DIVERGENCE_LINE_RE = re.compile(re.escape(_SWA_DIVERGENCE_LOG_PREFIX) + r"(\S+)")


@dataclass(frozen=True, slots=True, kw_only=True)
class SwaDivergenceLog:
    forward_ct: int
    verify_full: int
    verify_swa: int
    swa_full_idx_divergence: int

    def format(self) -> str:
        return _SWA_DIVERGENCE_LOG_PREFIX + json.dumps(
            asdict(self), separators=(",", ":")
        )

    @classmethod
    def parse(cls, line: str) -> Optional["SwaDivergenceLog"]:
        match = _SWA_DIVERGENCE_LINE_RE.search(line)
        if match is None:
            return None
        return cls(**json.loads(match.group(1)))

    @classmethod
    def find_last(cls, text: str) -> Optional[tuple["SwaDivergenceLog", str]]:
        last_match: Optional[re.Match] = None
        for match in _SWA_DIVERGENCE_LINE_RE.finditer(text):
            last_match = match
        if last_match is None:
            return None
        return cls(**json.loads(last_match.group(1))), last_match.group(0)
