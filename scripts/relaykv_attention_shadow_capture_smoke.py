from __future__ import annotations

import copy
import json
import math
from typing import Any

from sglang.srt.relaykv.metrics import (
    build_relaykv_attention_shadow_capture_results_for_smoke,
    summarize_relaykv_attention_shadow_capture_results_for_smoke,
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


def _comparison_plan(
    request_id: str,
    req_pool_idx: int,
    seq_len: int,
    layer_id: int,
    full_kv_block_ids: list[int],
    relaykv_working_kv_block_ids: list[int],
    *,
    event_type: str = "relaykv_attention_comparison_plan",
    comparison_state: str = "plan_ready",
    comparison_mode: str = "metadata_only",
    attention_comparison_executed: bool = False,
    attention_override: bool = False,
    attention_override_noop: bool = True,
    poison: _PoisonTensorLike | None = None,
) -> dict[str, Any]:
    full_count = len(full_kv_block_ids)
    working_count = len(relaykv_working_kv_block_ids)
    ratio = (working_count / full_count) if full_count > 0 else None
    plan: dict[str, Any] = {
        "event_type": event_type,
        "comparison_state": comparison_state,
        "comparison_mode": comparison_mode,
        "source": "attention_override_noop_result_to_comparison_plan",
        "request_id": request_id,
        "req_pool_idx": req_pool_idx,
        "seq_len": seq_len,
        "layer_id": layer_id,
        "full_kv_block_ids": list(full_kv_block_ids),
        "relaykv_working_kv_block_ids": list(relaykv_working_kv_block_ids),
        "full_kv_block_count": full_count,
        "relaykv_working_kv_block_count": working_count,
        "reduced_block_count": max(full_count - working_count, 0),
        "working_to_full_block_ratio": ratio,
        "coverage_ratio": ratio,
        "attention_comparison_executed": attention_comparison_executed,
        "attention_connection_attempted": True,
        "attention_override": attention_override,
        "attention_override_noop": attention_override_noop,
        "kv_pool_read": False,
        "kv_snapshot": False,
        "runtime_writeback": False,
        "scheduler_policy_noop": True,
        "kv_cache_mutation": False,
        "source_mutated": False,
    }
    if poison is not None:
        plan["unrelated_tensor_like"] = poison
    return plan


def _comparison_plans(
    poison: _PoisonTensorLike | None = None,
) -> list[dict[str, Any]]:
    return [
        _comparison_plan("rid-a", 10, 512, 14, [0, 1, 2, 3, 4, 5], [1, 3], poison=poison),
        _comparison_plan("rid-b", 11, 1024, 14, [6, 7, 8, 9, 10, 11], [8, 10]),
    ]


def _assert_pass_flow() -> dict[str, Any]:
    poison = _PoisonTensorLike()
    plans = _comparison_plans(poison)
    before = copy.deepcopy(plans)
    results = build_relaykv_attention_shadow_capture_results_for_smoke(plans)
    if plans != before:
        raise AssertionError("attention comparison plans were mutated")
    if poison.forbidden_access_called:
        raise AssertionError("poison tensor-like object was accessed")
    if len(results) != 2:
        raise AssertionError(results)

    for result in results:
        if result["event_type"] != "relaykv_attention_shadow_capture_result":
            raise AssertionError(result)
        if result["shadow_capture_state"] != "metadata_shadow_captured":
            raise AssertionError(result)
        if result["shadow_capture_mode"] != "metadata_only":
            raise AssertionError(result)
        if result["source"] != "attention_comparison_plan_to_shadow_capture_result":
            raise AssertionError(result)
        if result["shadow_capture_attempted"] is not True:
            raise AssertionError(result)
        if result["attention_shadow_capture_count"] != 1:
            raise AssertionError(result)
        if result["attention_output_captured"] is not False:
            raise AssertionError(result)
        if result["attention_comparison_executed"] is not False:
            raise AssertionError(result)
        if result["attention_connection_attempted"] is not True:
            raise AssertionError(result)
        if result["attention_override"] is not False:
            raise AssertionError(result)
        if result["attention_override_noop"] is not True:
            raise AssertionError(result)
        if result["kv_pool_read"] is not False:
            raise AssertionError(result)
        if result["kv_snapshot"] is not False:
            raise AssertionError(result)
        if result["tensor_read"] is not False:
            raise AssertionError(result)
        if result["runtime_writeback"] is not False:
            raise AssertionError(result)
        if result["scheduler_policy_noop"] is not True:
            raise AssertionError(result)
        if result["kv_cache_mutation"] is not False:
            raise AssertionError(result)
        if result["source_mutated"] is not False:
            raise AssertionError(result)
        if result["blocking_reasons"] != []:
            raise AssertionError(result)
        if "metadata_only_attention_shadow_capture" not in result["warning_reasons"]:
            raise AssertionError(result)
        if "no_attention_output_tensor_captured" not in result["warning_reasons"]:
            raise AssertionError(result)

    summary = summarize_relaykv_attention_shadow_capture_results_for_smoke(results)
    if summary["shadow_capture_count"] != 2:
        raise AssertionError(summary)
    if summary["shadow_capture_attempted_count"] != 2:
        raise AssertionError(summary)
    if summary["attention_shadow_capture_count"] != 2:
        raise AssertionError(summary)
    if summary["attention_output_captured_count"] != 0:
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
        "tensor_read_count",
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
    return {"results": results, "summary": summary}


def _assert_blocked_cases() -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []

    capture_output = _comparison_plans()
    results = build_relaykv_attention_shadow_capture_results_for_smoke(
        capture_output, capture_attention_output=True
    )
    if (
        "capture_attention_output_not_allowed_in_metadata_smoke"
        not in results[0]["blocking_reasons"]
    ):
        raise AssertionError(results[0])
    outputs.append(summarize_relaykv_attention_shadow_capture_results_for_smoke(results))

    wrong_event = _comparison_plans()
    wrong_event[0]["event_type"] = "wrong"
    results = build_relaykv_attention_shadow_capture_results_for_smoke(wrong_event)
    if "not_attention_comparison_plan" not in results[0]["blocking_reasons"]:
        raise AssertionError(results[0])
    outputs.append(summarize_relaykv_attention_shadow_capture_results_for_smoke(results))

    wrong_state = _comparison_plans()
    wrong_state[0]["comparison_state"] = "blocked"
    results = build_relaykv_attention_shadow_capture_results_for_smoke(wrong_state)
    if "comparison_plan_not_ready" not in results[0]["blocking_reasons"]:
        raise AssertionError(results[0])
    outputs.append(summarize_relaykv_attention_shadow_capture_results_for_smoke(results))

    wrong_mode = _comparison_plans()
    wrong_mode[0]["comparison_mode"] = "other"
    results = build_relaykv_attention_shadow_capture_results_for_smoke(wrong_mode)
    if "comparison_plan_not_metadata_only" not in results[0]["blocking_reasons"]:
        raise AssertionError(results[0])
    outputs.append(summarize_relaykv_attention_shadow_capture_results_for_smoke(results))

    already_executed = _comparison_plans()
    already_executed[0]["attention_comparison_executed"] = True
    results = build_relaykv_attention_shadow_capture_results_for_smoke(already_executed)
    if "attention_comparison_already_executed" not in results[0]["blocking_reasons"]:
        raise AssertionError(results[0])
    outputs.append(summarize_relaykv_attention_shadow_capture_results_for_smoke(results))

    override_true = _comparison_plans()
    override_true[0]["attention_override"] = True
    results = build_relaykv_attention_shadow_capture_results_for_smoke(override_true)
    if "attention_override_true_not_allowed" not in results[0]["blocking_reasons"]:
        raise AssertionError(results[0])
    outputs.append(summarize_relaykv_attention_shadow_capture_results_for_smoke(results))

    noop_false = _comparison_plans()
    noop_false[0]["attention_override_noop"] = False
    results = build_relaykv_attention_shadow_capture_results_for_smoke(noop_false)
    if "attention_override_noop_not_true" not in results[0]["blocking_reasons"]:
        raise AssertionError(results[0])
    outputs.append(summarize_relaykv_attention_shadow_capture_results_for_smoke(results))

    empty_working = _comparison_plans()
    empty_working[0]["relaykv_working_kv_block_ids"] = []
    empty_working[0]["relaykv_working_kv_block_count"] = 0
    results = build_relaykv_attention_shadow_capture_results_for_smoke(empty_working)
    if "no_relaykv_working_kv_blocks" not in results[0]["blocking_reasons"]:
        raise AssertionError(results[0])
    outputs.append(summarize_relaykv_attention_shadow_capture_results_for_smoke(results))

    empty_full = _comparison_plans()
    empty_full[0]["full_kv_block_ids"] = []
    empty_full[0]["full_kv_block_count"] = 0
    results = build_relaykv_attention_shadow_capture_results_for_smoke(empty_full)
    if "no_full_kv_blocks" not in results[0]["blocking_reasons"]:
        raise AssertionError(results[0])
    outputs.append(summarize_relaykv_attention_shadow_capture_results_for_smoke(results))

    for summary in outputs:
        if summary["attention_override_true_count"] != 0:
            raise AssertionError(summary)
        if summary["kv_pool_read_count"] != 0:
            raise AssertionError(summary)
        if summary["kv_snapshot_count"] != 0:
            raise AssertionError(summary)
        if summary["tensor_read_count"] != 0:
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
    blocked = _assert_blocked_cases()
    print(
        json.dumps(
            {
                "pass_flow": {
                    "shadow_capture_count": pass_flow["summary"]["shadow_capture_count"],
                    "shadow_capture_attempted_count": pass_flow["summary"][
                        "shadow_capture_attempted_count"
                    ],
                    "attention_shadow_capture_count": pass_flow["summary"][
                        "attention_shadow_capture_count"
                    ],
                    "attention_output_captured_count": pass_flow["summary"][
                        "attention_output_captured_count"
                    ],
                    "full_kv_block_count": pass_flow["summary"]["full_kv_block_count"],
                    "relaykv_working_kv_block_count": pass_flow["summary"][
                        "relaykv_working_kv_block_count"
                    ],
                    "reduced_block_count": pass_flow["summary"]["reduced_block_count"],
                    "attention_comparison_executed_count": pass_flow["summary"][
                        "attention_comparison_executed_count"
                    ],
                    "attention_connection_attempted_count": pass_flow["summary"][
                        "attention_connection_attempted_count"
                    ],
                    "attention_override_true_count": pass_flow["summary"][
                        "attention_override_true_count"
                    ],
                    "attention_override_noop_count": pass_flow["summary"][
                        "attention_override_noop_count"
                    ],
                    "kv_pool_read_count": pass_flow["summary"]["kv_pool_read_count"],
                    "kv_snapshot_count": pass_flow["summary"]["kv_snapshot_count"],
                    "tensor_read_count": pass_flow["summary"]["tensor_read_count"],
                    "runtime_writeback_true_count": pass_flow["summary"][
                        "runtime_writeback_true_count"
                    ],
                    "scheduler_policy_noop_false_count": pass_flow["summary"][
                        "scheduler_policy_noop_false_count"
                    ],
                    "kv_cache_mutation_true_count": pass_flow["summary"][
                        "kv_cache_mutation_true_count"
                    ],
                    "source_mutated_true_count": pass_flow["summary"][
                        "source_mutated_true_count"
                    ],
                },
                "blocked_case_count": len(blocked),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
