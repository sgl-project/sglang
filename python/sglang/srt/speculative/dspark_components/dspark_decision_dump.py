from __future__ import annotations

import json
import logging
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class DsparkDecisionDumper:

    def __init__(
        self,
        *,
        gamma: int,
        verify_num_draft_tokens: int,
        tp_rank: int,
    ) -> None:
        self.gamma = int(gamma)
        self.verify_num_draft_tokens = int(verify_num_draft_tokens)
        self.tp_rank = int(tp_rank)

    def maybe_dump(
        self,
        *,
        forward_ct: Optional[int],
        bs: int,
        mode: str,
        budget: Optional[int],
        lag_steps: Optional[int],
        verify_lens: Optional[torch.Tensor],
        confidence: Optional[torch.Tensor],
        req_pool_indices: torch.Tensor,
        rids: Optional[list[str]],
        prefix_lens: torch.Tensor,
        anchor_tokens: torch.Tensor,
        draft_tokens: torch.Tensor,
        bonus_tokens: torch.Tensor,
        correct_len: torch.Tensor,
        cap_trim_lens: torch.Tensor,
        commit_lens: torch.Tensor,
        prefill_tail_start_positions: Optional[torch.Tensor] = None,
        prefill_tail_valid_lens: Optional[torch.Tensor] = None,
    ) -> None:
        if not envs.SGLANG_DSPARK_DEBUG_MAIN_OUTPUT.get():
            return
        if bs <= 0:
            return

        record = self._build_record(
            forward_ct=forward_ct,
            bs=bs,
            mode=mode,
            budget=budget,
            lag_steps=lag_steps,
            verify_lens=verify_lens,
            confidence=confidence,
            req_pool_indices=req_pool_indices,
            rids=rids,
            prefix_lens=prefix_lens,
            anchor_tokens=anchor_tokens,
            draft_tokens=draft_tokens,
            bonus_tokens=bonus_tokens,
            correct_len=correct_len,
            cap_trim_lens=cap_trim_lens,
            commit_lens=commit_lens,
            prefill_tail_start_positions=prefill_tail_start_positions,
            prefill_tail_valid_lens=prefill_tail_valid_lens,
        )
        logger.info(
            "DSPARK_DEBUG_MAIN_OUTPUT=%s", json.dumps(record, separators=(",", ":"))
        )

    def _build_record(
        self,
        *,
        forward_ct: Optional[int],
        bs: int,
        mode: str,
        budget: Optional[int],
        lag_steps: Optional[int],
        verify_lens: Optional[torch.Tensor],
        confidence: Optional[torch.Tensor],
        req_pool_indices: torch.Tensor,
        rids: Optional[list[str]],
        prefix_lens: torch.Tensor,
        anchor_tokens: torch.Tensor,
        draft_tokens: torch.Tensor,
        bonus_tokens: torch.Tensor,
        correct_len: torch.Tensor,
        cap_trim_lens: torch.Tensor,
        commit_lens: torch.Tensor,
        prefill_tail_start_positions: Optional[torch.Tensor],
        prefill_tail_valid_lens: Optional[torch.Tensor],
    ) -> dict:
        if verify_lens is None:
            verify_len_per_req = [self.verify_num_draft_tokens] * bs
        else:
            verify_len_per_req = [
                int(v) for v in verify_lens.detach().to("cpu").tolist()
            ]

        req_ids = req_pool_indices.detach().to("cpu").tolist()
        prefixes = prefix_lens.detach().to("cpu").tolist()
        anchors = anchor_tokens.detach().to("cpu").tolist()
        draft_rows = draft_tokens.detach().to("cpu").tolist()
        bonus = bonus_tokens.detach().to("cpu").tolist()
        correct = correct_len.detach().to("cpu").tolist()
        cap_trim = cap_trim_lens.detach().to("cpu").tolist()
        commit = commit_lens.detach().to("cpu").tolist()
        tail_starts = (
            [-1] * bs
            if prefill_tail_start_positions is None
            else prefill_tail_start_positions.detach().to("cpu").tolist()
        )
        tail_lens = (
            [0] * bs
            if prefill_tail_valid_lens is None
            else prefill_tail_valid_lens.detach().to("cpu").tolist()
        )

        if confidence is not None:
            conf_rows = confidence.detach().float().to("cpu")
            survival_rows = torch.cumprod(conf_rows, dim=1).tolist()
            conf_rows = conf_rows.tolist()
        else:
            conf_rows = None
            survival_rows = None

        reqs: list[dict] = []
        for row in range(bs):
            entry = {
                "rid": None if rids is None else rids[row],
                "req": int(req_ids[row]),
                "prefix": int(prefixes[row]),
                "anchor_token": int(anchors[row]),
                "verify_len": int(verify_len_per_req[row]),
                "acc_len": int(commit[row]),
                "correct_drafts": int(correct[row]),
                "cap_trim": int(cap_trim[row]),
                "prefill_tail_start": int(tail_starts[row]),
                "prefill_tail_len": int(tail_lens[row]),
                "draft_tokens": [int(t) for t in draft_rows[row]],
                "bonus_token": int(bonus[row]),
            }
            if conf_rows is not None:
                entry["confidence"] = [round(float(p), 4) for p in conf_rows[row]]
                entry["survival"] = [round(float(p), 4) for p in survival_rows[row]]
            reqs.append(entry)

        num_verify_tokens = sum(verify_len_per_req)
        return {
            "forward_ct": None if forward_ct is None else int(forward_ct),
            "tp_rank": self.tp_rank,
            "bs": int(bs),
            "gamma": self.gamma,
            "mode": mode,
            "budget": None if budget is None else int(budget),
            "lag_steps": None if lag_steps is None else int(lag_steps),
            "num_verify_tokens": int(num_verify_tokens),
            "avg_verify_len": round(num_verify_tokens / bs, 4),
            "reqs": reqs,
        }
