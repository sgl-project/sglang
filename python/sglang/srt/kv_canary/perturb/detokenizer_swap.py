from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Final, Optional

from sglang.srt.environ import envs
from sglang.srt.kv_canary.canary_info.aux_output import CANARY_INFOS_KEY
from sglang.srt.kv_canary.perturb.utils import (
    advance_delivered_counts,
    is_health_check_rid,
)
from sglang.srt.managers.io_struct import unwrap_from_pickle, wrap_as_pickle

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import BatchTokenIDOutput

logger = logging.getLogger(__name__)

_DETOKENIZER_SWAP_LOG_PREFIX: Final[str] = "kv_canary_detokenizer_swap="
_DETOKENIZER_SWAP_LINE_RE = re.compile(
    re.escape(_DETOKENIZER_SWAP_LOG_PREFIX) + r"(\S+)"
)


@dataclass(frozen=True, slots=True, kw_only=True)
class DetokenizerSwapSide:
    rid: str
    position: int
    token: int


@dataclass(frozen=True, slots=True, kw_only=True)
class DetokenizerSwapRecord:
    a: DetokenizerSwapSide
    b: DetokenizerSwapSide

    def format(self) -> str:
        return _DETOKENIZER_SWAP_LOG_PREFIX + json.dumps(
            asdict(self), separators=(",", ":")
        )

    @classmethod
    def find_all(cls, text: str) -> list[DetokenizerSwapRecord]:
        return [
            cls._from_dict(json.loads(match.group(1)))
            for match in _DETOKENIZER_SWAP_LINE_RE.finditer(text)
        ]

    @classmethod
    def _from_dict(cls, fields: dict) -> DetokenizerSwapRecord:
        return cls(
            a=DetokenizerSwapSide(**fields["a"]),
            b=DetokenizerSwapSide(**fields["b"]),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class DetokenizerSwapConfig:
    swap_at_output: int

    @classmethod
    def from_env(cls) -> DetokenizerSwapConfig:
        swap_at_output = envs.SGLANG_KV_CANARY_PERTURB_DETOKENIZER_SWAP_AT_OUTPUT.get()
        return cls(swap_at_output=swap_at_output)


_config: Optional[DetokenizerSwapConfig] = None
_delivered_counts: dict[str, int] = {}
_swapped_rids: set[str] = set()


def _get_config() -> DetokenizerSwapConfig:
    global _config
    if _config is None:
        _config = DetokenizerSwapConfig.from_env()
    return _config


def maybe_perturb_detokenizer_swap(recv_obj: BatchTokenIDOutput) -> None:
    config = _get_config()
    if config.swap_at_output < 0:
        return

    if recv_obj.output_ids is not None:
        _swap_due_pairs(recv_obj=recv_obj, swap_at_output=config.swap_at_output)
    advance_delivered_counts(_delivered_counts, recv_obj)


def _swap_due_pairs(*, recv_obj: BatchTokenIDOutput, swap_at_output: int) -> None:
    due = [
        index
        for index, rid in enumerate(recv_obj.rids)
        if not is_health_check_rid(rid)
        and rid not in _swapped_rids
        and len(recv_obj.output_ids[index]) > 0
        and _delivered_counts.get(rid, 0) >= swap_at_output
    ]
    if len(due) < 2:
        return

    customized_info = unwrap_from_pickle(recv_obj.customized_info)
    canary_infos = (
        None
        if customized_info is None or CANARY_INFOS_KEY not in customized_info
        else customized_info[CANARY_INFOS_KEY]
    )
    for i, j in zip(due[0::2], due[1::2]):
        record = DetokenizerSwapRecord(
            a=_swap_side(recv_obj, i), b=_swap_side(recv_obj, j)
        )
        output_ids = recv_obj.output_ids
        output_ids[i][0], output_ids[j][0] = output_ids[j][0], output_ids[i][0]
        if canary_infos is not None:
            canary_infos[i][0], canary_infos[j][0] = (
                canary_infos[j][0],
                canary_infos[i][0],
            )
        _swapped_rids.update((record.a.rid, record.b.rid))
        logger.info(record.format())
    if canary_infos is not None:
        recv_obj.customized_info = wrap_as_pickle(customized_info)


def _swap_side(recv_obj: BatchTokenIDOutput, index: int) -> DetokenizerSwapSide:
    rid = recv_obj.rids[index]
    return DetokenizerSwapSide(
        rid=rid,
        position=_delivered_counts.get(rid, 0),
        token=int(recv_obj.output_ids[index][0]),
    )
