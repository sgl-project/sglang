from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from sglang.srt.environ import envs
from sglang.srt.kv_canary.canary_info.aux_output import CANARY_INFOS_KEY
from sglang.srt.kv_canary.canary_info.checker import (
    CanaryInfoChecker,
    CanaryInfoMismatch,
)
from sglang.srt.kv_canary.config import CanaryMode, kv_canary_mode
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReqStateCanaryInfo:
    checker: CanaryInfoChecker
    mismatch: Optional[CanaryInfoMismatch] = None


class CanaryInfoTokenizerManagerHook:
    def __init__(self, *, raise_on_mismatch: bool) -> None:
        self._raise_on_mismatch = raise_on_mismatch

    @classmethod
    def maybe_create(cls) -> Optional[CanaryInfoTokenizerManagerHook]:
        if not envs.SGLANG_KV_CANARY_ENABLE_INFO.get():
            return None
        mode = kv_canary_mode()
        if mode is CanaryMode.NONE:
            return None
        return cls(raise_on_mismatch=mode is CanaryMode.RAISE)

    def on_request_dispatched(self, *, rid_to_state: dict, tokenized_obj: Any) -> None:
        if not isinstance(tokenized_obj, TokenizedGenerateReqInput):
            return
        if (state := rid_to_state.get(tokenized_obj.rid)) is None:
            return
        if (tokenized_obj.sampling_params.beam_width or 1) > 1:
            raise ValueError("KvCanaryInfo does not support beam search")
        if (
            tokenized_obj.mm_inputs is not None
            or tokenized_obj.need_wait_for_mm_inputs
            or tokenized_obj.session_params is not None
            or tokenized_obj.input_embeds is not None
            or not tokenized_obj.input_ids
        ):
            raise ValueError(
                "KvCanaryInfo requires a non-empty token prompt and does not support "
                "multimodal inputs, session_params, or input_embeds"
            )
        checker = _build_canary_info_checker(tokenized_obj)
        state.canary_info = ReqStateCanaryInfo(checker=checker)

    def on_batch_output(
        self,
        *,
        state: Any,
        recv_obj: Any,
        customized_info: Optional[dict],
        index: int,
    ) -> None:
        canary_info = state.canary_info
        assert canary_info is not None
        mismatch = _check_canary_infos(
            checker=canary_info.checker,
            token_ids=recv_obj.output_ids[index],
            canary_infos=_canary_infos_at(customized_info, index),
        )
        if mismatch is None:
            return
        if self._raise_on_mismatch:
            raise RuntimeError(mismatch.format())
        logger.error(mismatch.format())
        state.canary_info = ReqStateCanaryInfo(
            checker=canary_info.checker, mismatch=mismatch
        )


def _build_canary_info_checker(
    tokenized_obj: TokenizedGenerateReqInput,
) -> CanaryInfoChecker:
    return CanaryInfoChecker(
        rid=tokenized_obj.rid,
        sampling_params=tokenized_obj.sampling_params,
        prompt_ids=tokenized_obj.input_ids,
    )


def _check_canary_infos(
    *,
    checker: CanaryInfoChecker,
    token_ids: Sequence[int],
    canary_infos: Optional[Sequence[Optional[int]]],
) -> Optional[CanaryInfoMismatch]:
    if len(token_ids) == 0 and (canary_infos is None or len(canary_infos) == 0):
        return None
    if canary_infos is None or len(canary_infos) != len(token_ids):
        first_token = int(token_ids[0]) if token_ids else None
        return CanaryInfoMismatch(
            rid=checker.rid,
            position=checker.position,
            token_id=first_token,
            expected=checker.expected(first_token) if first_token is not None else None,
            actual=None,
        )
    return checker.advance_many(token_ids, canary_infos)


def _canary_infos_at(
    customized_info: Optional[dict], index: int
) -> Optional[Sequence[Optional[int]]]:
    if customized_info is None or CANARY_INFOS_KEY not in customized_info:
        return None
    return customized_info[CANARY_INFOS_KEY][index]
