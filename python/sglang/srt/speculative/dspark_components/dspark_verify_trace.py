from __future__ import annotations

import json
import logging
import os
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.server_args import get_global_server_args

logger = logging.getLogger(__name__)


class DsparkVerifyTracer:
    """Debug-only DSpark verifier invariant checker and JSONL tracer.

    The greedy verifier rule is local to one target-verify step:
    candidates[:, 1:] must match target argmax predictions consecutively.
    This check does not compare against an independent target-only rollout, so it
    remains meaningful when target generation is not deterministic across runs.
    """

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
        self.path = envs.SGLANG_DSPARK_VERIFY_TRACE_PATH.get()
        self.limit = int(envs.SGLANG_DSPARK_VERIFY_TRACE_LIMIT.get())
        self.assert_enabled = envs.SGLANG_DSPARK_VERIFY_TRACE_ASSERT.get()
        self._records = 0
        self._warned_write_error = False

    def maybe_trace(
        self,
        *,
        forward_ct: Optional[int],
        bs: int,
        mode: str,
        budget: Optional[int],
        layout_graph_num_tokens: Optional[int],
        folded_accept: bool,
        run_compact: bool,
        can_run_cuda_graph: bool,
        sampling_info,
        verify_lens: Optional[torch.Tensor],
        req_pool_indices: torch.Tensor,
        rids: Optional[list[str]],
        prefix_lens: torch.Tensor,
        candidates: torch.Tensor,
        draft_tokens: torch.Tensor,
        target_logits: torch.Tensor,
        correct_len: torch.Tensor,
        bonus: torch.Tensor,
        cap_trim_lens: torch.Tensor,
        commit_lens: torch.Tensor,
        new_seq_lens: torch.Tensor,
        out_tokens: torch.Tensor,
        simulated_accept: bool,
    ) -> None:
        if self.tp_rank != 0 or bs <= 0:
            return

        all_greedy = sampling_info is None or sampling_info.is_all_greedy
        enabled = bool(self.path) or self.assert_enabled
        if not enabled:
            return

        record = self._build_record(
            forward_ct=forward_ct,
            bs=bs,
            mode=mode,
            budget=budget,
            layout_graph_num_tokens=layout_graph_num_tokens,
            folded_accept=folded_accept,
            run_compact=run_compact,
            can_run_cuda_graph=can_run_cuda_graph,
            all_greedy=all_greedy,
            sampling_trace_info=_sampling_trace_info(sampling_info),
            verify_lens=verify_lens,
            req_pool_indices=req_pool_indices,
            rids=rids,
            prefix_lens=prefix_lens,
            candidates=candidates,
            draft_tokens=draft_tokens,
            target_logits=target_logits,
            correct_len=correct_len,
            bonus=bonus,
            cap_trim_lens=cap_trim_lens,
            commit_lens=commit_lens,
            new_seq_lens=new_seq_lens,
            out_tokens=out_tokens,
            simulated_accept=simulated_accept,
        )
        if self.path and self._should_write():
            self._write_record(record)
        if self.assert_enabled and not record["verdict"]["passed"]:
            raise AssertionError(
                "DSpark verifier invariant failed: "
                + json.dumps(record["verdict"], separators=(",", ":"))
            )

    def _should_write(self) -> bool:
        return self.limit <= 0 or self._records < self.limit

    def _write_record(self, record: dict) -> None:
        try:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            self._records += 1
        except Exception:
            if not self._warned_write_error:
                logger.exception("Failed to write DSpark verify trace to %s", self.path)
                self._warned_write_error = True

    def _build_record(
        self,
        *,
        forward_ct: Optional[int],
        bs: int,
        mode: str,
        budget: Optional[int],
        layout_graph_num_tokens: Optional[int],
        folded_accept: bool,
        run_compact: bool,
        can_run_cuda_graph: bool,
        all_greedy: bool,
        sampling_trace_info: dict,
        verify_lens: Optional[torch.Tensor],
        req_pool_indices: torch.Tensor,
        rids: Optional[list[str]],
        prefix_lens: torch.Tensor,
        candidates: torch.Tensor,
        draft_tokens: torch.Tensor,
        target_logits: torch.Tensor,
        correct_len: torch.Tensor,
        bonus: torch.Tensor,
        cap_trim_lens: torch.Tensor,
        commit_lens: torch.Tensor,
        new_seq_lens: torch.Tensor,
        out_tokens: torch.Tensor,
        simulated_accept: bool,
    ) -> dict:
        target_predict = torch.argmax(target_logits, dim=-1).view(
            bs, self.verify_num_draft_tokens
        )

        expected = self._expected_greedy(
            candidates=candidates,
            target_predict=target_predict,
            prefix_lens=prefix_lens,
            verify_lens=verify_lens,
        )

        actual_cpu = {
            "correct_len": _cpu_list(correct_len),
            "bonus": _cpu_list(bonus),
            "cap_trim_lens": _cpu_list(cap_trim_lens),
            "commit_lens": _cpu_list(commit_lens),
            "new_seq_lens": _cpu_list(new_seq_lens),
            "out_tokens": _cpu_list(out_tokens),
        }
        expected_cpu = {key: _cpu_list(value) for key, value in expected.items()}

        failures = []
        skipped = []
        if not all_greedy:
            skipped.append("non-greedy sampling is not covered by this invariant")
        if simulated_accept:
            skipped.append("simulated accept length overrides verifier decisions")
        if all_greedy and not simulated_accept:
            _compare_vector(
                failures,
                "correct_len",
                actual_cpu["correct_len"],
                expected_cpu["correct_len"],
            )
            _compare_vector(failures, "bonus", actual_cpu["bonus"], expected_cpu["bonus"])
            _compare_vector(
                failures,
                "cap_trim_lens",
                actual_cpu["cap_trim_lens"],
                expected_cpu["cap_trim_lens"],
            )
            _compare_vector(
                failures,
                "commit_lens",
                actual_cpu["commit_lens"],
                expected_cpu["commit_lens"],
            )
            _compare_vector(
                failures,
                "new_seq_lens",
                actual_cpu["new_seq_lens"],
                expected_cpu["new_seq_lens"],
            )
            _compare_committed_out_tokens(
                failures=failures,
                actual_out=actual_cpu["out_tokens"],
                expected_out=expected_cpu["out_tokens"],
                commit_lens=actual_cpu["commit_lens"],
            )

        candidates_cpu = _cpu_list(candidates)
        target_predict_cpu = _cpu_list(target_predict)
        req_ids = _cpu_list(req_pool_indices)
        prefixes = _cpu_list(prefix_lens)
        verify_lens_cpu = (
            [self.verify_num_draft_tokens] * bs
            if verify_lens is None
            else _cpu_list(verify_lens)
        )

        reqs = []
        for row in range(bs):
            reqs.append(
                {
                    "rid": None if rids is None else rids[row],
                    "req": int(req_ids[row]),
                    "prefix_len": int(prefixes[row]),
                    "verify_len": int(verify_lens_cpu[row]),
                    "candidates": [int(x) for x in candidates_cpu[row]],
                    "target_predict": [int(x) for x in target_predict_cpu[row]],
                    "expected_correct_len": int(expected_cpu["correct_len"][row]),
                    "actual_correct_len": int(actual_cpu["correct_len"][row]),
                    "expected_bonus": int(expected_cpu["bonus"][row]),
                    "actual_bonus": int(actual_cpu["bonus"][row]),
                    "expected_commit_len": int(expected_cpu["commit_lens"][row]),
                    "actual_commit_len": int(actual_cpu["commit_lens"][row]),
                    "expected_new_seq_len": int(expected_cpu["new_seq_lens"][row]),
                    "actual_new_seq_len": int(actual_cpu["new_seq_lens"][row]),
                    "committed_out_tokens": [
                        int(x)
                        for x in actual_cpu["out_tokens"][row][
                            : int(actual_cpu["commit_lens"][row])
                        ]
                    ],
                }
            )

        return {
            "forward_ct": None if forward_ct is None else int(forward_ct),
            "bs": int(bs),
            "gamma": self.gamma,
            "verify_num_draft_tokens": self.verify_num_draft_tokens,
            "mode": mode,
            "budget": None if budget is None else int(budget),
            "layout_graph_num_tokens": (
                None if layout_graph_num_tokens is None else int(layout_graph_num_tokens)
            ),
            "verify_lens": verify_lens_cpu,
            "verify_lens_sum": int(sum(int(x) for x in verify_lens_cpu)),
            "non_uniform_verify_lens": len(set(int(x) for x in verify_lens_cpu)) > 1,
            "tp_rank": self.tp_rank,
            "folded_accept": bool(folded_accept),
            "run_compact": bool(run_compact),
            "can_run_cuda_graph": bool(can_run_cuda_graph),
            "all_greedy": bool(all_greedy),
            "sampling": sampling_trace_info,
            "simulated_accept": bool(simulated_accept),
            "reqs": reqs,
            "verdict": {
                "passed": not failures,
                "failures": failures,
                "skipped": skipped,
            },
        }

    def _expected_greedy(
        self,
        *,
        candidates: torch.Tensor,
        target_predict: torch.Tensor,
        prefix_lens: torch.Tensor,
        verify_lens: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        matches = candidates[:, 1:] == target_predict[:, :-1]
        raw_correct_len = matches.to(torch.int32).cumprod(dim=1).sum(dim=1)
        correct_len = raw_correct_len
        cap_trim_lens = torch.zeros_like(correct_len)
        if verify_lens is not None:
            max_correct = (verify_lens.to(device=correct_len.device) - 1).to(
                correct_len.dtype
            )
            correct_len = torch.minimum(correct_len, max_correct)
            cap_trim_lens = raw_correct_len - correct_len

        row_ids = torch.arange(candidates.shape[0], device=candidates.device)
        bonus = target_predict[row_ids, correct_len.to(torch.long)].to(torch.int64)
        commit_lens = correct_len.to(torch.int32) + 1
        new_seq_lens = prefix_lens.to(device=commit_lens.device) + commit_lens.to(
            prefix_lens.dtype
        )
        out_tokens = torch.empty(
            (candidates.shape[0], self.verify_num_draft_tokens),
            dtype=torch.int64,
            device=candidates.device,
        )
        out_tokens[:, : self.gamma].copy_(candidates[:, 1:])
        out_tokens[:, self.gamma].fill_(0)
        out_tokens.scatter_(1, correct_len.to(torch.int64)[:, None], bonus[:, None])
        return {
            "correct_len": correct_len,
            "bonus": bonus,
            "cap_trim_lens": cap_trim_lens,
            "commit_lens": commit_lens,
            "new_seq_lens": new_seq_lens,
            "out_tokens": out_tokens,
        }


def _cpu_list(tensor: torch.Tensor) -> list:
    return tensor.detach().to("cpu").tolist()


def _sampling_trace_info(sampling_info) -> dict:
    try:
        sampling_backend = get_global_server_args().sampling_backend
    except Exception:
        sampling_backend = None

    if sampling_info is None:
        return {
            "backend": sampling_backend,
            "seed_present": False,
            "seed_rows": 0,
            "any_greedy": True,
            "need_top_k": False,
            "need_top_p": False,
            "need_min_p": False,
        }

    sampling_seed = getattr(sampling_info, "sampling_seed", None)
    return {
        "backend": sampling_backend,
        "seed_present": sampling_seed is not None,
        "seed_rows": 0 if sampling_seed is None else int(sampling_seed.numel()),
        "any_greedy": bool(getattr(sampling_info, "is_any_greedy", False)),
        "need_top_k": bool(getattr(sampling_info, "need_top_k_sampling", False)),
        "need_top_p": bool(getattr(sampling_info, "need_top_p_sampling", False)),
        "need_min_p": bool(getattr(sampling_info, "need_min_p_sampling", False)),
    }


def _compare_vector(
    failures: list[str],
    name: str,
    actual: list,
    expected: list,
) -> None:
    if actual != expected:
        failures.append(f"{name} mismatch: actual={actual} expected={expected}")


def _compare_committed_out_tokens(
    *,
    failures: list[str],
    actual_out: list[list[int]],
    expected_out: list[list[int]],
    commit_lens: list[int],
) -> None:
    for row, commit_len in enumerate(commit_lens):
        if actual_out[row][:commit_len] != expected_out[row][:commit_len]:
            failures.append(
                "committed_out_tokens mismatch at row "
                f"{row}: actual={actual_out[row][:commit_len]} "
                f"expected={expected_out[row][:commit_len]}"
            )
