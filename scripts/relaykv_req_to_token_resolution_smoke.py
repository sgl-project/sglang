from __future__ import annotations

import copy
import json
from typing import Any

from sglang.srt.relaykv.metrics import (
    build_relaykv_req_to_token_resolution_results_for_smoke,
    summarize_relaykv_req_to_token_resolution_results_for_smoke,
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


def _block_span(
    request_id: str,
    layer_id: int,
    block_id: int,
    token_start: int,
    token_end: int,
) -> dict[str, Any]:
    return {
        "block_id": block_id,
        "token_start": token_start,
        "token_end": token_end,
        "token_count": token_end - token_start,
        "request_id": request_id,
        "layer_id": layer_id,
        "span_source": "synthetic_block_span",
    }


def _kv_index_resolution_plan(
    request_id: str,
    req_pool_idx: Any,
    *,
    event_type: str = "relaykv_kv_index_resolution_plan",
    resolution_state: str = "block_span_resolved",
    resolution_mode: str = "metadata_only",
    seq_len: Any = 768,
    layer_id: int = 14,
    token_to_kv_pool_read: bool = False,
    kv_pool_read: bool = False,
    tensor_read: bool = False,
    attention_comparison_executed: bool = False,
    attention_override: bool = False,
    poison: _PoisonTensorLike | None = None,
) -> dict[str, Any]:
    full_spans = [
        _block_span(request_id, layer_id, block_id, block_id * 128, (block_id + 1) * 128)
        for block_id in range(6)
    ]
    working_block_ids = [1, 3] if request_id == "req-a" else [2, 4]
    working_spans = [
        _block_span(request_id, layer_id, block_id, block_id * 128, (block_id + 1) * 128)
        for block_id in working_block_ids
    ]

    plan: dict[str, Any] = {
        "event_type": event_type,
        "resolution_state": resolution_state,
        "resolution_mode": resolution_mode,
        "source": "attention_shadow_capture_result_to_kv_index_resolution_plan",
        "request_id": request_id,
        "req_pool_idx": req_pool_idx,
        "seq_len": seq_len,
        "layer_id": layer_id,
        "relaykv_working_block_spans": working_spans,
        "full_kv_block_spans": full_spans,
        "req_to_token_read": False,
        "token_to_kv_pool_read": token_to_kv_pool_read,
        "kv_pool_read": kv_pool_read,
        "kv_snapshot": False,
        "tensor_read": tensor_read,
        "attention_comparison_executed": attention_comparison_executed,
        "attention_override": attention_override,
        "runtime_writeback": False,
        "scheduler_policy_noop": True,
        "kv_cache_mutation": False,
        "source_mutated": False,
    }
    if poison is not None:
        plan["unrelated_tensor_like"] = poison
    return plan


def _kv_index_resolution_plans(
    poison: _PoisonTensorLike | None = None,
) -> list[dict[str, Any]]:
    return [
        _kv_index_resolution_plan("req-a", 7, poison=poison),
        _kv_index_resolution_plan("req-b", "8"),
    ]


def _req_to_token_tables() -> dict[Any, Any]:
    return {
        7: list(range(1000, 1768)),
        "8": list(range(2000, 2768)),
    }


def _assert_span(
    span: dict[str, Any],
    *,
    request_id: str,
    req_pool_idx: Any,
    layer_id: int,
) -> None:
    if not isinstance(span["block_id"], int):
        raise AssertionError(span)
    if not isinstance(span["token_start"], int):
        raise AssertionError(span)
    if not isinstance(span["token_end"], int):
        raise AssertionError(span)
    if not isinstance(span["token_count"], int):
        raise AssertionError(span)
    if span["token_start"] < 0:
        raise AssertionError(span)
    if span["token_end"] <= span["token_start"]:
        raise AssertionError(span)
    if span["token_count"] != span["token_end"] - span["token_start"]:
        raise AssertionError(span)
    if span["request_id"] != request_id:
        raise AssertionError(span)
    if span["req_pool_idx"] != int(req_pool_idx):
        raise AssertionError(span)
    if span["layer_id"] != layer_id:
        raise AssertionError(span)
    if span["entry_count"] != span["token_count"]:
        raise AssertionError(span)
    if len(span["req_to_token_entries"]) != span["token_count"]:
        raise AssertionError(span)
    if span["resolution_source"] != "synthetic_req_to_token_table":
        raise AssertionError(span)


def _assert_pass_flow() -> dict[str, Any]:
    poison = _PoisonTensorLike()
    plans = _kv_index_resolution_plans(poison)
    tables = _req_to_token_tables()
    before_plans = copy.deepcopy(plans)
    before_tables = copy.deepcopy(tables)

    results = build_relaykv_req_to_token_resolution_results_for_smoke(
        plans,
        req_to_token_table_by_req_pool_idx=tables,
        read_req_to_token=True,
    )
    if plans != before_plans:
        raise AssertionError("kv index resolution plans were mutated")
    if tables != before_tables:
        raise AssertionError("req_to_token tables were mutated")
    if poison.forbidden_access_called:
        raise AssertionError("poison tensor-like object was accessed")
    if len(results) != 2:
        raise AssertionError(results)

    for result in results:
        if result["event_type"] != "relaykv_req_to_token_resolution_result":
            raise AssertionError(result)
        if result["resolution_state"] != "req_to_token_resolved":
            raise AssertionError(result)
        if result["resolution_mode"] != "readonly_synthetic_table":
            raise AssertionError(result)
        if (
            result["source"]
            != "kv_index_resolution_plan_to_req_to_token_resolution_result"
        ):
            raise AssertionError(result)
        if result["resolved_block_count"] != 6:
            raise AssertionError(result)
        if result["resolved_token_count"] != 768:
            raise AssertionError(result)
        if result["req_to_token_entry_count"] != 768:
            raise AssertionError(result)
        if result["req_to_token_read"] is not True:
            raise AssertionError(result)
        if result["req_to_token_read_count"] != 768:
            raise AssertionError(result)
        if result["token_to_kv_pool_read"] is not False:
            raise AssertionError(result)
        if result["token_to_kv_pool_read_count"] != 0:
            raise AssertionError(result)
        if result["kv_pool_read"] is not False:
            raise AssertionError(result)
        if result["kv_snapshot"] is not False:
            raise AssertionError(result)
        if result["tensor_read"] is not False:
            raise AssertionError(result)
        if result["attention_comparison_executed"] is not False:
            raise AssertionError(result)
        if result["attention_override"] is not False:
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
        if "readonly_req_to_token_resolution" not in result["warning_reasons"]:
            raise AssertionError(result)
        if "no_token_to_kv_pool_read" not in result["warning_reasons"]:
            raise AssertionError(result)
        if len(result["full_kv_req_to_token_spans"]) != 6:
            raise AssertionError(result)
        if len(result["relaykv_working_req_to_token_spans"]) != 2:
            raise AssertionError(result)

        for span in result["full_kv_req_to_token_spans"]:
            _assert_span(
                span,
                request_id=result["request_id"],
                req_pool_idx=result["req_pool_idx"],
                layer_id=result["layer_id"],
            )

    summary = summarize_relaykv_req_to_token_resolution_results_for_smoke(results)
    if summary["req_to_token_resolved_count"] != 2:
        raise AssertionError(summary)
    if summary["resolved_block_count"] != 12:
        raise AssertionError(summary)
    if summary["resolved_token_count"] != 1536:
        raise AssertionError(summary)
    if summary["req_to_token_entry_count"] != 1536:
        raise AssertionError(summary)
    if summary["req_to_token_read_count"] != 1536:
        raise AssertionError(summary)

    expected_zero = (
        "token_to_kv_pool_read_count",
        "kv_pool_read_count",
        "kv_snapshot_count",
        "tensor_read_count",
        "attention_comparison_executed_count",
        "attention_override_true_count",
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

    wrong_event = [_kv_index_resolution_plan("req-a", 7)]
    wrong_event[0]["event_type"] = "wrong"
    results = build_relaykv_req_to_token_resolution_results_for_smoke(
        wrong_event,
        req_to_token_table_by_req_pool_idx=_req_to_token_tables(),
        read_req_to_token=True,
    )
    if "not_kv_index_resolution_plan" not in results[0]["blocking_reasons"]:
        raise AssertionError(results[0])
    outputs.append(summarize_relaykv_req_to_token_resolution_results_for_smoke(results))

    wrong_state = [_kv_index_resolution_plan("req-a", 7)]
    wrong_state[0]["resolution_state"] = "blocked"
    results = build_relaykv_req_to_token_resolution_results_for_smoke(
        wrong_state,
        req_to_token_table_by_req_pool_idx=_req_to_token_tables(),
        read_req_to_token=True,
    )
    if (
        "kv_index_resolution_not_block_span_resolved"
        not in results[0]["blocking_reasons"]
    ):
        raise AssertionError(results[0])
    outputs.append(summarize_relaykv_req_to_token_resolution_results_for_smoke(results))

    wrong_mode = [_kv_index_resolution_plan("req-a", 7)]
    wrong_mode[0]["resolution_mode"] = "other"
    results = build_relaykv_req_to_token_resolution_results_for_smoke(
        wrong_mode,
        req_to_token_table_by_req_pool_idx=_req_to_token_tables(),
        read_req_to_token=True,
    )
    if (
        "kv_index_resolution_not_metadata_only"
        not in results[0]["blocking_reasons"]
    ):
        raise AssertionError(results[0])
    outputs.append(summarize_relaykv_req_to_token_resolution_results_for_smoke(results))

    read_disabled = [_kv_index_resolution_plan("req-a", 7)]
    results = build_relaykv_req_to_token_resolution_results_for_smoke(
        read_disabled,
        req_to_token_table_by_req_pool_idx=_req_to_token_tables(),
        read_req_to_token=False,
    )
    if "read_req_to_token_not_enabled" not in results[0]["blocking_reasons"]:
        raise AssertionError(results[0])
    outputs.append(summarize_relaykv_req_to_token_resolution_results_for_smoke(results))

    bad_req_pool_idx = [_kv_index_resolution_plan("req-a", 7)]
    bad_req_pool_idx[0]["req_pool_idx"] = None
    results = build_relaykv_req_to_token_resolution_results_for_smoke(
        bad_req_pool_idx,
        req_to_token_table_by_req_pool_idx=_req_to_token_tables(),
        read_req_to_token=True,
    )
    if "req_pool_idx_missing_or_invalid" not in results[0]["blocking_reasons"]:
        raise AssertionError(results[0])
    outputs.append(summarize_relaykv_req_to_token_resolution_results_for_smoke(results))

    bad_seq_len = [_kv_index_resolution_plan("req-a", 7)]
    bad_seq_len[0]["seq_len"] = None
    results = build_relaykv_req_to_token_resolution_results_for_smoke(
        bad_seq_len,
        req_to_token_table_by_req_pool_idx=_req_to_token_tables(),
        read_req_to_token=True,
    )
    if "seq_len_missing_or_invalid" not in results[0]["blocking_reasons"]:
        raise AssertionError(results[0])
    outputs.append(summarize_relaykv_req_to_token_resolution_results_for_smoke(results))

    missing_table = [_kv_index_resolution_plan("req-a", 7)]
    results = build_relaykv_req_to_token_resolution_results_for_smoke(
        missing_table,
        req_to_token_table_by_req_pool_idx=None,
        read_req_to_token=True,
    )
    if "req_to_token_table_missing" not in results[0]["blocking_reasons"]:
        raise AssertionError(results[0])
    outputs.append(summarize_relaykv_req_to_token_resolution_results_for_smoke(results))

    missing_req_pool_table = [_kv_index_resolution_plan("req-a", 7)]
    tables = _req_to_token_tables()
    del tables[7]
    results = build_relaykv_req_to_token_resolution_results_for_smoke(
        missing_req_pool_table,
        req_to_token_table_by_req_pool_idx=tables,
        read_req_to_token=True,
    )
    if "req_to_token_table_for_req_pool_missing" not in results[0]["blocking_reasons"]:
        raise AssertionError(results[0])
    outputs.append(summarize_relaykv_req_to_token_resolution_results_for_smoke(results))

    non_indexable_table = [_kv_index_resolution_plan("req-a", 7)]
    tables = _req_to_token_tables()
    tables[7] = {"bad": "table"}
    results = build_relaykv_req_to_token_resolution_results_for_smoke(
        non_indexable_table,
        req_to_token_table_by_req_pool_idx=tables,
        read_req_to_token=True,
    )
    if "req_to_token_table_not_indexable" not in results[0]["blocking_reasons"]:
        raise AssertionError(results[0])
    outputs.append(summarize_relaykv_req_to_token_resolution_results_for_smoke(results))

    out_of_table = [_kv_index_resolution_plan("req-a", 7)]
    tables = _req_to_token_tables()
    tables[7] = list(range(1000, 1500))
    results = build_relaykv_req_to_token_resolution_results_for_smoke(
        out_of_table,
        req_to_token_table_by_req_pool_idx=tables,
        read_req_to_token=True,
    )
    if (
        "token_position_out_of_req_to_token_table"
        not in results[0]["blocking_reasons"]
    ):
        raise AssertionError(results[0])
    outputs.append(summarize_relaykv_req_to_token_resolution_results_for_smoke(results))

    bad_entry = [_kv_index_resolution_plan("req-a", 7)]
    tables = _req_to_token_tables()
    tables[7][0] = "bad"
    results = build_relaykv_req_to_token_resolution_results_for_smoke(
        bad_entry,
        req_to_token_table_by_req_pool_idx=tables,
        read_req_to_token=True,
    )
    if "req_to_token_entry_not_int" not in results[0]["blocking_reasons"]:
        raise AssertionError(results[0])
    outputs.append(summarize_relaykv_req_to_token_resolution_results_for_smoke(results))

    invalid_span = [_kv_index_resolution_plan("req-a", 7)]
    invalid_span[0]["full_kv_block_spans"][0]["token_count"] = 0
    results = build_relaykv_req_to_token_resolution_results_for_smoke(
        invalid_span,
        req_to_token_table_by_req_pool_idx=_req_to_token_tables(),
        read_req_to_token=True,
    )
    if "invalid_block_span" not in results[0]["blocking_reasons"]:
        raise AssertionError(results[0])
    outputs.append(summarize_relaykv_req_to_token_resolution_results_for_smoke(results))

    out_of_seq = [_kv_index_resolution_plan("req-a", 7)]
    out_of_seq[0]["full_kv_block_spans"][0]["token_end"] = 900
    out_of_seq[0]["full_kv_block_spans"][0]["token_count"] = 900
    results = build_relaykv_req_to_token_resolution_results_for_smoke(
        out_of_seq,
        req_to_token_table_by_req_pool_idx=_req_to_token_tables(),
        read_req_to_token=True,
    )
    if "token_span_out_of_seq_len" not in results[0]["blocking_reasons"]:
        raise AssertionError(results[0])
    outputs.append(summarize_relaykv_req_to_token_resolution_results_for_smoke(results))

    token_to_kv_pool_true = [_kv_index_resolution_plan("req-a", 7)]
    token_to_kv_pool_true[0]["token_to_kv_pool_read"] = True
    results = build_relaykv_req_to_token_resolution_results_for_smoke(
        token_to_kv_pool_true,
        req_to_token_table_by_req_pool_idx=_req_to_token_tables(),
        read_req_to_token=True,
    )
    if "token_to_kv_pool_read_not_allowed" not in results[0]["blocking_reasons"]:
        raise AssertionError(results[0])
    outputs.append(summarize_relaykv_req_to_token_resolution_results_for_smoke(results))

    kv_pool_true = [_kv_index_resolution_plan("req-a", 7)]
    kv_pool_true[0]["kv_pool_read"] = True
    results = build_relaykv_req_to_token_resolution_results_for_smoke(
        kv_pool_true,
        req_to_token_table_by_req_pool_idx=_req_to_token_tables(),
        read_req_to_token=True,
    )
    if "kv_pool_read_not_allowed" not in results[0]["blocking_reasons"]:
        raise AssertionError(results[0])
    outputs.append(summarize_relaykv_req_to_token_resolution_results_for_smoke(results))

    tensor_true = [_kv_index_resolution_plan("req-a", 7)]
    tensor_true[0]["tensor_read"] = True
    results = build_relaykv_req_to_token_resolution_results_for_smoke(
        tensor_true,
        req_to_token_table_by_req_pool_idx=_req_to_token_tables(),
        read_req_to_token=True,
    )
    if "tensor_read_not_allowed" not in results[0]["blocking_reasons"]:
        raise AssertionError(results[0])
    outputs.append(summarize_relaykv_req_to_token_resolution_results_for_smoke(results))

    comparison_true = [_kv_index_resolution_plan("req-a", 7)]
    comparison_true[0]["attention_comparison_executed"] = True
    results = build_relaykv_req_to_token_resolution_results_for_smoke(
        comparison_true,
        req_to_token_table_by_req_pool_idx=_req_to_token_tables(),
        read_req_to_token=True,
    )
    if (
        "attention_comparison_executed_not_allowed"
        not in results[0]["blocking_reasons"]
    ):
        raise AssertionError(results[0])
    outputs.append(summarize_relaykv_req_to_token_resolution_results_for_smoke(results))

    override_true = [_kv_index_resolution_plan("req-a", 7)]
    override_true[0]["attention_override"] = True
    results = build_relaykv_req_to_token_resolution_results_for_smoke(
        override_true,
        req_to_token_table_by_req_pool_idx=_req_to_token_tables(),
        read_req_to_token=True,
    )
    if "attention_override_true_not_allowed" not in results[0]["blocking_reasons"]:
        raise AssertionError(results[0])
    outputs.append(summarize_relaykv_req_to_token_resolution_results_for_smoke(results))

    for summary in outputs:
        if summary["req_to_token_read_count"] != 0:
            raise AssertionError(summary)
        if summary["token_to_kv_pool_read_count"] != 0:
            raise AssertionError(summary)
        if summary["kv_pool_read_count"] != 0:
            raise AssertionError(summary)
        if summary["kv_snapshot_count"] != 0:
            raise AssertionError(summary)
        if summary["tensor_read_count"] != 0:
            raise AssertionError(summary)
        if summary["attention_comparison_executed_count"] != 0:
            raise AssertionError(summary)
        if summary["attention_override_true_count"] != 0:
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
                    "req_to_token_resolved_count": pass_flow["summary"][
                        "req_to_token_resolved_count"
                    ],
                    "resolved_block_count": pass_flow["summary"][
                        "resolved_block_count"
                    ],
                    "resolved_token_count": pass_flow["summary"][
                        "resolved_token_count"
                    ],
                    "req_to_token_entry_count": pass_flow["summary"][
                        "req_to_token_entry_count"
                    ],
                    "req_to_token_read_count": pass_flow["summary"][
                        "req_to_token_read_count"
                    ],
                    "token_to_kv_pool_read_count": pass_flow["summary"][
                        "token_to_kv_pool_read_count"
                    ],
                    "kv_pool_read_count": pass_flow["summary"]["kv_pool_read_count"],
                    "kv_snapshot_count": pass_flow["summary"]["kv_snapshot_count"],
                    "tensor_read_count": pass_flow["summary"]["tensor_read_count"],
                    "attention_comparison_executed_count": pass_flow["summary"][
                        "attention_comparison_executed_count"
                    ],
                    "attention_override_true_count": pass_flow["summary"][
                        "attention_override_true_count"
                    ],
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
