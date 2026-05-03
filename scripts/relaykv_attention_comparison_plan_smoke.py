from __future__ import annotations

import copy
import json
import math
from typing import Any

from sglang.srt.relaykv.metrics import (
    build_relaykv_attention_comparison_plans_for_smoke,
    summarize_relaykv_attention_comparison_plans_for_smoke,
)


class _PoisonTensorLike:
    def __init__(self) -> None:
        self.cpu_called = False
        self.item_called = False
        self.tolist_called = False
        self.iter_called = False
        self.len_called = False
        self.getitem_called = False
        self.shape = (1,)
        self.device = "cuda:0"
        self.dtype = "torch.int64"

    def __deepcopy__(self, memo: dict[int, Any]) -> "_PoisonTensorLike":
        return self

    def cpu(self) -> None:
        self.cpu_called = True
        raise AssertionError("cpu() must not be called")

    def item(self) -> None:
        self.item_called = True
        raise AssertionError("item() must not be called")

    def tolist(self) -> None:
        self.tolist_called = True
        raise AssertionError("tolist() must not be called")

    def __iter__(self):
        self.iter_called = True
        raise AssertionError("__iter__() must not be called")

    def __len__(self) -> int:
        self.len_called = True
        raise AssertionError("__len__() must not be called")

    def __getitem__(self, index: int) -> None:
        self.getitem_called = True
        raise AssertionError("__getitem__() must not be called")

    @property
    def forbidden_access_called(self) -> bool:
        return (
            self.cpu_called
            or self.item_called
            or self.tolist_called
            or self.iter_called
            or self.len_called
            or self.getitem_called
        )


def _noop_result(
    request_id: str,
    req_pool_idx: int,
    seq_len: int,
    layer_id: int,
    working_kv_block_ids: list[int],
    *,
    event_type: str = "relaykv_attention_override_noop_result",
    attention_connection_state: str = "override_noop",
    attention_connection_mode: str = "noop_guarded",
    attention_connection_attempted: bool = True,
    attention_override: bool = False,
    attention_override_noop: bool = True,
    recent_block_ids: list[int] | None = None,
    anchor_block_ids: list[int] | None = None,
    candidate_block_ids: list[int] | None = None,
    retrieved_block_ids: list[int] | None = None,
    materialized_block_ids: list[int] | None = None,
    poison: _PoisonTensorLike | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "event_type": event_type,
        "attention_connection_state": attention_connection_state,
        "attention_connection_mode": attention_connection_mode,
        "request_id": request_id,
        "req_pool_idx": req_pool_idx,
        "seq_len": seq_len,
        "layer_id": layer_id,
        "working_kv_block_ids": list(working_kv_block_ids),
        "working_kv_block_count": len(working_kv_block_ids),
        "working_kv_token_count": 0,
        "attention_target_layer_id": layer_id,
        "attention_target_backend": "unconnected",
        "attention_connection_attempted": attention_connection_attempted,
        "attention_override": attention_override,
        "attention_override_noop": attention_override_noop,
        "recent_block_ids": list(recent_block_ids or []),
        "anchor_block_ids": list(anchor_block_ids or []),
        "candidate_block_ids": list(candidate_block_ids or []),
        "retrieved_block_ids": list(retrieved_block_ids or []),
        "materialized_block_ids": list(materialized_block_ids or []),
        "kv_pool_read": False,
        "kv_snapshot": False,
        "runtime_writeback": False,
        "scheduler_policy_noop": True,
        "kv_cache_mutation": False,
        "source_mutated": False,
    }
    if poison is not None:
        result["unrelated_tensor_like"] = poison
    return result


def _noop_results(
    poison: _PoisonTensorLike | None = None,
) -> list[dict[str, Any]]:
    return [
        _noop_result(
            "rid-a",
            10,
            512,
            14,
            [1, 3],
            recent_block_ids=[5],
            anchor_block_ids=[0],
            candidate_block_ids=[1, 2, 3, 4],
            retrieved_block_ids=[1, 3],
            materialized_block_ids=[1, 3],
            poison=poison,
        ),
        _noop_result(
            "rid-b",
            11,
            1024,
            14,
            [2, 4],
            recent_block_ids=[5],
            anchor_block_ids=[0],
            candidate_block_ids=[1, 2, 3, 4],
            retrieved_block_ids=[2, 4],
            materialized_block_ids=[2, 4],
        ),
    ]


def _assert_pass_flow() -> dict[str, Any]:
    poison = _PoisonTensorLike()
    noop_results = _noop_results(poison)
    before = copy.deepcopy(noop_results)
    full_kv_by_request_layer = {
        ("rid-a", 14): [0, 1, 2, 3, 4, 5],
        "rid-b:14": [0, 1, 2, 3, 4, 5],
    }
    plans = build_relaykv_attention_comparison_plans_for_smoke(
        noop_results,
        full_kv_block_ids_by_request_layer=full_kv_by_request_layer,
    )
    if noop_results != before:
        raise AssertionError("attention override noop results were mutated")
    if poison.forbidden_access_called:
        raise AssertionError("poison tensor-like object was accessed")
    if len(plans) != 2:
        raise AssertionError(plans)
    for plan in plans:
        if plan["comparison_state"] != "plan_ready":
            raise AssertionError(plan)
        if plan["comparison_mode"] != "metadata_only":
            raise AssertionError(plan)
        if plan["attention_comparison_executed"] is not False:
            raise AssertionError(plan)
        if plan["attention_connection_attempted"] is not True:
            raise AssertionError(plan)
        if plan["attention_override"] is not False:
            raise AssertionError(plan)
        if plan["attention_override_noop"] is not True:
            raise AssertionError(plan)
        if "metadata_only_attention_comparison_plan" not in plan["warning_reasons"]:
            raise AssertionError(plan)
    summary = summarize_relaykv_attention_comparison_plans_for_smoke(plans)
    if summary["comparison_plan_ready_count"] != 2:
        raise AssertionError(summary)
    if summary["full_kv_block_count"] != 12:
        raise AssertionError(summary)
    if summary["relaykv_working_kv_block_count"] != 4:
        raise AssertionError(summary)
    if summary["reduced_block_count"] != 8:
        raise AssertionError(summary)
    if not math.isclose(
        summary["mean_working_to_full_block_ratio"], 2.0 / 6.0, rel_tol=1e-9
    ):
        raise AssertionError(summary)
    if not math.isclose(summary["mean_coverage_ratio"], 2.0 / 6.0, rel_tol=1e-9):
        raise AssertionError(summary)
    expected_zero = (
        "attention_comparison_executed_count",
        "attention_override_true_count",
        "kv_pool_read_count",
        "kv_snapshot_count",
        "runtime_writeback_true_count",
        "scheduler_policy_noop_false_count",
        "kv_cache_mutation_true_count",
        "source_mutated_true_count",
    )
    for key in expected_zero:
        if summary[key] != 0:
            raise AssertionError(summary)
    if summary["attention_connection_attempted_count"] != 2:
        raise AssertionError(summary)
    if summary["attention_override_noop_count"] != 2:
        raise AssertionError(summary)
    return {"plans": plans, "summary": summary}


def _assert_fallback_synthesis() -> dict[str, Any]:
    results = _noop_results()
    plans = build_relaykv_attention_comparison_plans_for_smoke(results)
    if any(plan["comparison_state"] != "plan_ready" for plan in plans):
        raise AssertionError(plans)
    summary = summarize_relaykv_attention_comparison_plans_for_smoke(plans)
    if summary["comparison_plan_ready_count"] != 2:
        raise AssertionError(summary)
    if summary["full_kv_block_count"] <= 0:
        raise AssertionError(summary)
    return summary


def _assert_blocked_cases() -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []

    wrong_event = _noop_results()
    wrong_event[0]["event_type"] = "wrong"
    plans = build_relaykv_attention_comparison_plans_for_smoke(wrong_event)
    if "not_attention_override_noop_result" not in plans[0]["blocking_reasons"]:
        raise AssertionError(plans[0])
    outputs.append(summarize_relaykv_attention_comparison_plans_for_smoke(plans))

    wrong_state = _noop_results()
    wrong_state[0]["attention_connection_state"] = "blocked"
    plans = build_relaykv_attention_comparison_plans_for_smoke(wrong_state)
    if "attention_connection_not_override_noop" not in plans[0]["blocking_reasons"]:
        raise AssertionError(plans[0])
    outputs.append(summarize_relaykv_attention_comparison_plans_for_smoke(plans))

    wrong_mode = _noop_results()
    wrong_mode[0]["attention_connection_mode"] = "other"
    plans = build_relaykv_attention_comparison_plans_for_smoke(wrong_mode)
    if "attention_connection_not_noop_guarded" not in plans[0]["blocking_reasons"]:
        raise AssertionError(plans[0])
    outputs.append(summarize_relaykv_attention_comparison_plans_for_smoke(plans))

    noop_false = _noop_results()
    noop_false[0]["attention_override_noop"] = False
    plans = build_relaykv_attention_comparison_plans_for_smoke(noop_false)
    if "attention_override_noop_not_true" not in plans[0]["blocking_reasons"]:
        raise AssertionError(plans[0])
    outputs.append(summarize_relaykv_attention_comparison_plans_for_smoke(plans))

    override_true = _noop_results()
    override_true[0]["attention_override"] = True
    plans = build_relaykv_attention_comparison_plans_for_smoke(override_true)
    if "attention_override_true_not_allowed" not in plans[0]["blocking_reasons"]:
        raise AssertionError(plans[0])
    outputs.append(summarize_relaykv_attention_comparison_plans_for_smoke(plans))

    empty_working = _noop_results()
    empty_working[0]["working_kv_block_ids"] = []
    empty_working[0]["working_kv_block_count"] = 0
    plans = build_relaykv_attention_comparison_plans_for_smoke(empty_working)
    if "no_working_kv_blocks" not in plans[0]["blocking_reasons"]:
        raise AssertionError(plans[0])
    outputs.append(summarize_relaykv_attention_comparison_plans_for_smoke(plans))

    no_full = _noop_results()
    no_full[0]["recent_block_ids"] = []
    no_full[0]["anchor_block_ids"] = []
    no_full[0]["candidate_block_ids"] = []
    no_full[0]["retrieved_block_ids"] = []
    no_full[0]["materialized_block_ids"] = []
    no_full[0]["working_kv_block_ids"] = [1, 3]
    no_full[0]["working_kv_block_count"] = 2
    plans = build_relaykv_attention_comparison_plans_for_smoke(
        no_full,
        full_kv_block_ids_by_request_layer={("rid-a", 14): []},
    )
    if "no_full_kv_blocks" not in plans[0]["blocking_reasons"]:
        raise AssertionError(plans[0])
    outputs.append(summarize_relaykv_attention_comparison_plans_for_smoke(plans))

    for summary in outputs:
        if summary["attention_override_true_count"] != 0:
            raise AssertionError(summary)
        if summary["kv_pool_read_count"] != 0:
            raise AssertionError(summary)
        if summary["kv_snapshot_count"] != 0:
            raise AssertionError(summary)
        if summary["runtime_writeback_true_count"] != 0:
            raise AssertionError(summary)
        if summary["scheduler_policy_noop_false_count"] != 0:
            raise AssertionError(summary)
        if summary["kv_cache_mutation_true_count"] != 0:
            raise AssertionError(summary)
        if summary["source_mutated_true_count"] != 0:
            raise AssertionError(summary)

    return outputs


def main() -> None:
    pass_flow = _assert_pass_flow()
    fallback = _assert_fallback_synthesis()
    blocked = _assert_blocked_cases()
    print(
        json.dumps(
            {
                "pass_flow": {
                    "comparison_plan_ready_count": pass_flow["summary"][
                        "comparison_plan_ready_count"
                    ],
                    "full_kv_block_count": pass_flow["summary"][
                        "full_kv_block_count"
                    ],
                    "relaykv_working_kv_block_count": pass_flow["summary"][
                        "relaykv_working_kv_block_count"
                    ],
                    "reduced_block_count": pass_flow["summary"][
                        "reduced_block_count"
                    ],
                },
                "fallback_synthesis_full_kv_block_count": fallback[
                    "full_kv_block_count"
                ],
                "blocked_case_count": len(blocked),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
