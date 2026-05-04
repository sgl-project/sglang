from __future__ import annotations

import copy
import json
from typing import Any

from sglang.srt.relaykv.metrics import (
    build_relaykv_kv_index_resolution_plans_for_smoke,
    summarize_relaykv_kv_index_resolution_plans_for_smoke,
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


def _shadow_capture_result(
    request_id: str,
    req_pool_idx: int,
    seq_len: int,
    layer_id: int,
    relaykv_working_kv_block_ids: list[int],
    full_kv_block_ids: list[int],
    *,
    event_type: str = "relaykv_attention_shadow_capture_result",
    shadow_capture_state: str = "metadata_shadow_captured",
    shadow_capture_mode: str = "metadata_only",
    attention_output_captured: bool = False,
    attention_comparison_executed: bool = False,
    attention_override: bool = False,
    poison: _PoisonTensorLike | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "event_type": event_type,
        "shadow_capture_state": shadow_capture_state,
        "shadow_capture_mode": shadow_capture_mode,
        "source": "attention_comparison_plan_to_shadow_capture_result",
        "request_id": request_id,
        "req_pool_idx": req_pool_idx,
        "seq_len": seq_len,
        "layer_id": layer_id,
        "relaykv_working_kv_block_ids": list(relaykv_working_kv_block_ids),
        "full_kv_block_ids": list(full_kv_block_ids),
        "attention_output_captured": attention_output_captured,
        "attention_comparison_executed": attention_comparison_executed,
        "attention_override": attention_override,
        "kv_pool_read": False,
        "kv_snapshot": False,
        "tensor_read": False,
        "runtime_writeback": False,
        "scheduler_policy_noop": True,
        "kv_cache_mutation": False,
        "source_mutated": False,
    }
    if poison is not None:
        result["unrelated_tensor_like"] = poison
    return result


def _shadow_capture_results(
    poison: _PoisonTensorLike | None = None,
) -> list[dict[str, Any]]:
    return [
        _shadow_capture_result("rid-a", 10, 768, 14, [1, 3], [0, 1, 2, 3, 4, 5], poison=poison),
        _shadow_capture_result("rid-b", 11, 768, 14, [2, 4], [0, 1, 2, 3, 4, 5]),
    ]


def _block_metadata() -> dict[Any, Any]:
    return {
        0: {"token_start": 0, "token_end": 128},
        "1": {"start_token": 128, "end_token": 256},
        ("rid-a", 14, 2): {"token_span": [256, 384]},
        ("rid-b", 14, 2): {"token_span": [256, 384]},
        "rid-a:14:3": (384, 512),
        "rid-b:14:3": [384, 512],
        4: [512, 640],
        "5": {"token_start": 640, "token_end": 768},
    }


def _assert_span_record(span: dict[str, Any], block_id: int, request_id: str, layer_id: int) -> None:
    if span["block_id"] != block_id:
        raise AssertionError(span)
    if not isinstance(span["token_start"], int):
        raise AssertionError(span)
    if not isinstance(span["token_end"], int):
        raise AssertionError(span)
    if span["token_start"] < 0:
        raise AssertionError(span)
    if span["token_end"] <= span["token_start"]:
        raise AssertionError(span)
    if span["token_count"] != span["token_end"] - span["token_start"]:
        raise AssertionError(span)
    if span["request_id"] != request_id:
        raise AssertionError(span)
    if span["layer_id"] != layer_id:
        raise AssertionError(span)
    if not isinstance(span["span_source"], str):
        raise AssertionError(span)


def _assert_pass_flow() -> dict[str, Any]:
    poison = _PoisonTensorLike()
    results = _shadow_capture_results(poison)
    metadata = _block_metadata()
    before_results = copy.deepcopy(results)
    before_metadata = copy.deepcopy(metadata)

    plans = build_relaykv_kv_index_resolution_plans_for_smoke(
        results, block_metadata_by_id=metadata
    )
    if results != before_results:
        raise AssertionError("attention shadow capture results were mutated")
    if metadata != before_metadata:
        raise AssertionError("block metadata was mutated")
    if poison.forbidden_access_called:
        raise AssertionError("poison tensor-like object was accessed")
    if len(plans) != 2:
        raise AssertionError(plans)

    for plan in plans:
        if plan["event_type"] != "relaykv_kv_index_resolution_plan":
            raise AssertionError(plan)
        if plan["resolution_state"] != "block_span_resolved":
            raise AssertionError(plan)
        if plan["resolution_mode"] != "metadata_only":
            raise AssertionError(plan)
        if plan["source"] != "attention_shadow_capture_result_to_kv_index_resolution_plan":
            raise AssertionError(plan)
        if plan["missing_block_ids"] != []:
            raise AssertionError(plan)
        if plan["invalid_block_ids"] != []:
            raise AssertionError(plan)
        if plan["resolved_block_count"] != 6:
            raise AssertionError(plan)
        if plan["token_span_count"] != 6:
            raise AssertionError(plan)
        if plan["total_token_count"] != 768:
            raise AssertionError(plan)
        if plan["req_to_token_read"] is not False:
            raise AssertionError(plan)
        if plan["token_to_kv_pool_read"] is not False:
            raise AssertionError(plan)
        if plan["kv_pool_read"] is not False:
            raise AssertionError(plan)
        if plan["kv_snapshot"] is not False:
            raise AssertionError(plan)
        if plan["tensor_read"] is not False:
            raise AssertionError(plan)
        if plan["attention_comparison_executed"] is not False:
            raise AssertionError(plan)
        if plan["attention_override"] is not False:
            raise AssertionError(plan)
        if plan["runtime_writeback"] is not False:
            raise AssertionError(plan)
        if plan["scheduler_policy_noop"] is not True:
            raise AssertionError(plan)
        if plan["kv_cache_mutation"] is not False:
            raise AssertionError(plan)
        if plan["source_mutated"] is not False:
            raise AssertionError(plan)
        if plan["blocking_reasons"] != []:
            raise AssertionError(plan)
        if "metadata_only_kv_index_resolution_plan" not in plan["warning_reasons"]:
            raise AssertionError(plan)
        if "no_req_to_token_pool_read" not in plan["warning_reasons"]:
            raise AssertionError(plan)

        request_id = plan["request_id"]
        layer_id = plan["layer_id"]
        if len(plan["full_kv_block_spans"]) != 6:
            raise AssertionError(plan)
        if len(plan["relaykv_working_block_spans"]) != 2:
            raise AssertionError(plan)

        for idx, span in enumerate(plan["full_kv_block_spans"]):
            _assert_span_record(span, idx, request_id, layer_id)
        expected_working = [1, 3] if request_id == "rid-a" else [2, 4]
        for idx, span in enumerate(plan["relaykv_working_block_spans"]):
            _assert_span_record(span, expected_working[idx], request_id, layer_id)

    summary = summarize_relaykv_kv_index_resolution_plans_for_smoke(plans)
    if summary["block_span_resolved_count"] != 2:
        raise AssertionError(summary)
    if summary["resolved_block_count"] != 12:
        raise AssertionError(summary)
    if summary["token_span_count"] != 12:
        raise AssertionError(summary)
    if summary["total_token_count"] != 1536:
        raise AssertionError(summary)
    if summary["missing_block_count"] != 0:
        raise AssertionError(summary)
    if summary["invalid_block_count"] != 0:
        raise AssertionError(summary)

    expected_zero = (
        "req_to_token_read_count",
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
    return {"plans": plans, "summary": summary}


def _assert_blocked_cases() -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []

    wrong_event = _shadow_capture_results()
    wrong_event[0]["event_type"] = "wrong"
    plans = build_relaykv_kv_index_resolution_plans_for_smoke(
        wrong_event, block_metadata_by_id=_block_metadata()
    )
    if "not_attention_shadow_capture_result" not in plans[0]["blocking_reasons"]:
        raise AssertionError(plans[0])
    outputs.append(summarize_relaykv_kv_index_resolution_plans_for_smoke(plans))

    wrong_state = _shadow_capture_results()
    wrong_state[0]["shadow_capture_state"] = "blocked"
    plans = build_relaykv_kv_index_resolution_plans_for_smoke(
        wrong_state, block_metadata_by_id=_block_metadata()
    )
    if "shadow_capture_not_metadata_captured" not in plans[0]["blocking_reasons"]:
        raise AssertionError(plans[0])
    outputs.append(summarize_relaykv_kv_index_resolution_plans_for_smoke(plans))

    wrong_mode = _shadow_capture_results()
    wrong_mode[0]["shadow_capture_mode"] = "other"
    plans = build_relaykv_kv_index_resolution_plans_for_smoke(
        wrong_mode, block_metadata_by_id=_block_metadata()
    )
    if "shadow_capture_not_metadata_only" not in plans[0]["blocking_reasons"]:
        raise AssertionError(plans[0])
    outputs.append(summarize_relaykv_kv_index_resolution_plans_for_smoke(plans))

    output_captured = _shadow_capture_results()
    output_captured[0]["attention_output_captured"] = True
    plans = build_relaykv_kv_index_resolution_plans_for_smoke(
        output_captured, block_metadata_by_id=_block_metadata()
    )
    if "attention_output_captured_not_allowed" not in plans[0]["blocking_reasons"]:
        raise AssertionError(plans[0])
    outputs.append(summarize_relaykv_kv_index_resolution_plans_for_smoke(plans))

    comparison_executed = _shadow_capture_results()
    comparison_executed[0]["attention_comparison_executed"] = True
    plans = build_relaykv_kv_index_resolution_plans_for_smoke(
        comparison_executed, block_metadata_by_id=_block_metadata()
    )
    if "attention_comparison_executed_not_allowed" not in plans[0]["blocking_reasons"]:
        raise AssertionError(plans[0])
    outputs.append(summarize_relaykv_kv_index_resolution_plans_for_smoke(plans))

    override_true = _shadow_capture_results()
    override_true[0]["attention_override"] = True
    plans = build_relaykv_kv_index_resolution_plans_for_smoke(
        override_true, block_metadata_by_id=_block_metadata()
    )
    if "attention_override_true_not_allowed" not in plans[0]["blocking_reasons"]:
        raise AssertionError(plans[0])
    outputs.append(summarize_relaykv_kv_index_resolution_plans_for_smoke(plans))

    empty_working = _shadow_capture_results()
    empty_working[0]["relaykv_working_kv_block_ids"] = []
    plans = build_relaykv_kv_index_resolution_plans_for_smoke(
        empty_working, block_metadata_by_id=_block_metadata()
    )
    if "no_relaykv_working_kv_blocks" not in plans[0]["blocking_reasons"]:
        raise AssertionError(plans[0])
    outputs.append(summarize_relaykv_kv_index_resolution_plans_for_smoke(plans))

    empty_full = _shadow_capture_results()
    empty_full[0]["full_kv_block_ids"] = []
    plans = build_relaykv_kv_index_resolution_plans_for_smoke(
        empty_full, block_metadata_by_id=_block_metadata()
    )
    if "no_full_kv_blocks" not in plans[0]["blocking_reasons"]:
        raise AssertionError(plans[0])
    outputs.append(summarize_relaykv_kv_index_resolution_plans_for_smoke(plans))

    missing_seq_len = _shadow_capture_results()
    missing_seq_len[0]["seq_len"] = None
    plans = build_relaykv_kv_index_resolution_plans_for_smoke(
        missing_seq_len, block_metadata_by_id=_block_metadata()
    )
    if "seq_len_missing_or_invalid" not in plans[0]["blocking_reasons"]:
        raise AssertionError(plans[0])
    outputs.append(summarize_relaykv_kv_index_resolution_plans_for_smoke(plans))

    missing_metadata = _shadow_capture_results()
    broken_metadata = _block_metadata()
    del broken_metadata[4]
    plans = build_relaykv_kv_index_resolution_plans_for_smoke(
        missing_metadata, block_metadata_by_id=broken_metadata
    )
    if "missing_block_metadata" not in plans[0]["blocking_reasons"]:
        raise AssertionError(plans[0])
    outputs.append(summarize_relaykv_kv_index_resolution_plans_for_smoke(plans))

    invalid_span = _shadow_capture_results()
    invalid_metadata = _block_metadata()
    invalid_metadata[4] = [640, 512]
    plans = build_relaykv_kv_index_resolution_plans_for_smoke(
        invalid_span, block_metadata_by_id=invalid_metadata
    )
    if "invalid_block_span" not in plans[0]["blocking_reasons"]:
        raise AssertionError(plans[0])
    outputs.append(summarize_relaykv_kv_index_resolution_plans_for_smoke(plans))

    out_of_seq = _shadow_capture_results()
    out_of_seq_metadata = _block_metadata()
    out_of_seq_metadata["5"] = {"token_start": 640, "token_end": 900}
    plans = build_relaykv_kv_index_resolution_plans_for_smoke(
        out_of_seq, block_metadata_by_id=out_of_seq_metadata
    )
    if "block_span_out_of_seq_len" not in plans[0]["blocking_reasons"]:
        raise AssertionError(plans[0])
    outputs.append(summarize_relaykv_kv_index_resolution_plans_for_smoke(plans))

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
                    "block_span_resolved_count": pass_flow["summary"][
                        "block_span_resolved_count"
                    ],
                    "resolved_block_count": pass_flow["summary"][
                        "resolved_block_count"
                    ],
                    "token_span_count": pass_flow["summary"]["token_span_count"],
                    "total_token_count": pass_flow["summary"]["total_token_count"],
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
