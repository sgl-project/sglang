from __future__ import annotations

import copy
import json
from typing import Any

from sglang.srt.relaykv.metrics import (
    build_relaykv_attention_override_noop_results_for_smoke,
    summarize_relaykv_attention_override_noop_results_for_smoke,
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


def _dry_run_result(
    request_id: str,
    req_pool_idx: int,
    seq_len: int,
    layer_id: int,
    working_kv_block_ids: list[int],
    *,
    event_type: str = "relaykv_attention_connection_dry_run_result",
    attention_connection_state: str = "dry_run",
    attention_connection_mode: str = "metadata_only",
    attention_connection_attempted: bool = True,
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
        "attention_override": False,
        "attention_override_noop": False,
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


def _dry_run_results(
    poison: _PoisonTensorLike | None = None,
) -> list[dict[str, Any]]:
    return [
        _dry_run_result("rid-a", 10, 512, 0, [1, 2], poison=poison),
        _dry_run_result("rid-b", 11, 1024, 14, [9, 10]),
    ]


def _assert_pass_flow() -> dict[str, Any]:
    poison = _PoisonTensorLike()
    dry_run_results = _dry_run_results(poison)
    before = copy.deepcopy(dry_run_results)
    results = build_relaykv_attention_override_noop_results_for_smoke(
        dry_run_results,
        allow_override=False,
    )
    if dry_run_results != before:
        raise AssertionError("attention connection dry-run results were mutated")
    if poison.forbidden_access_called:
        raise AssertionError("poison tensor-like object was accessed")
    if len(results) != 2:
        raise AssertionError(results)
    for result in results:
        if result["attention_connection_state"] != "override_noop":
            raise AssertionError(result)
        if result["attention_connection_mode"] != "noop_guarded":
            raise AssertionError(result)
        if result["attention_connection_attempted"] is not True:
            raise AssertionError(result)
        if result["attention_override"] is not False:
            raise AssertionError(result)
        if result["attention_override_noop"] is not True:
            raise AssertionError(result)
        if "attention_override_noop_guarded" not in result["warning_reasons"]:
            raise AssertionError(result)
        if "no_runtime_attention_backend_connection" not in result["warning_reasons"]:
            raise AssertionError(result)
    summary = summarize_relaykv_attention_override_noop_results_for_smoke(results)
    if summary["attention_override_noop_count"] != 2:
        raise AssertionError(summary)
    if summary["attention_connection_attempted_count"] != 2:
        raise AssertionError(summary)
    if summary["working_kv_block_count"] != 4:
        raise AssertionError(summary)
    expected_zero = (
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
    return {"results": results, "summary": summary}


def _assert_blocked_cases() -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []

    results = build_relaykv_attention_override_noop_results_for_smoke(
        _dry_run_results(),
        allow_override=True,
    )
    if any(r["attention_connection_state"] != "blocked" for r in results):
        raise AssertionError(results)
    if any(
        "attention_override_not_allowed_in_phase4_noop"
        not in r["blocking_reasons"]
        for r in results
    ):
        raise AssertionError(results)
    outputs.append(summarize_relaykv_attention_override_noop_results_for_smoke(results))

    wrong_event = _dry_run_results()
    wrong_event[0]["event_type"] = "wrong"
    results = build_relaykv_attention_override_noop_results_for_smoke(wrong_event)
    if "not_attention_connection_dry_run_result" not in results[0]["blocking_reasons"]:
        raise AssertionError(results[0])
    outputs.append(summarize_relaykv_attention_override_noop_results_for_smoke(results))

    wrong_state = _dry_run_results()
    wrong_state[0]["attention_connection_state"] = "blocked"
    results = build_relaykv_attention_override_noop_results_for_smoke(wrong_state)
    if "attention_connection_not_dry_run" not in results[0]["blocking_reasons"]:
        raise AssertionError(results[0])
    outputs.append(summarize_relaykv_attention_override_noop_results_for_smoke(results))

    wrong_mode = _dry_run_results()
    wrong_mode[0]["attention_connection_mode"] = "other"
    results = build_relaykv_attention_override_noop_results_for_smoke(wrong_mode)
    if "attention_connection_not_metadata_only" not in results[0]["blocking_reasons"]:
        raise AssertionError(results[0])
    outputs.append(summarize_relaykv_attention_override_noop_results_for_smoke(results))

    not_attempted = _dry_run_results()
    not_attempted[0]["attention_connection_attempted"] = False
    results = build_relaykv_attention_override_noop_results_for_smoke(not_attempted)
    if "attention_connection_not_attempted" not in results[0]["blocking_reasons"]:
        raise AssertionError(results[0])
    outputs.append(summarize_relaykv_attention_override_noop_results_for_smoke(results))

    empty_blocks = _dry_run_results()
    empty_blocks[0]["working_kv_block_ids"] = []
    empty_blocks[0]["working_kv_block_count"] = 0
    results = build_relaykv_attention_override_noop_results_for_smoke(empty_blocks)
    if "no_working_kv_blocks" not in results[0]["blocking_reasons"]:
        raise AssertionError(results[0])
    outputs.append(summarize_relaykv_attention_override_noop_results_for_smoke(results))

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
    blocked = _assert_blocked_cases()
    print(
        json.dumps(
            {
                "pass_flow": {
                    "attention_override_noop_count": pass_flow["summary"][
                        "attention_override_noop_count"
                    ],
                    "attention_connection_attempted_count": pass_flow["summary"][
                        "attention_connection_attempted_count"
                    ],
                    "working_kv_block_count": pass_flow["summary"][
                        "working_kv_block_count"
                    ],
                },
                "blocked_case_count": len(blocked),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
