from __future__ import annotations

import copy
import json
from typing import Any

from sglang.srt.relaykv.metrics import (
    build_relaykv_req_to_token_readonly_adapter_payloads_for_smoke,
    summarize_relaykv_req_to_token_readonly_adapter_payloads_for_smoke,
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
    block_id: int,
    token_start: int,
    token_end: int,
) -> dict[str, int]:
    return {
        "block_id": block_id,
        "token_start": token_start,
        "token_end": token_end,
        "token_count": token_end - token_start,
    }


def _kv_index_resolution_plan(
    request_id: str,
    req_pool_idx: Any,
    *,
    event_type: str = "relaykv_kv_index_resolution_plan",
    resolution_state: str = "block_span_resolved",
    resolution_mode: str = "metadata_only",
    seq_len: Any = 256,
    layer_id: int = 14,
    full_kv_block_spans: list[dict[str, int]] | None = None,
    relaykv_working_block_spans: list[dict[str, int]] | None = None,
    token_to_kv_pool_read: bool = False,
    kv_pool_read: bool = False,
    tensor_read: bool = False,
    attention_comparison_executed: bool = False,
    attention_override: bool = False,
    poison: _PoisonTensorLike | None = None,
) -> dict[str, Any]:
    plan: dict[str, Any] = {
        "event_type": event_type,
        "resolution_state": resolution_state,
        "resolution_mode": resolution_mode,
        "source": "attention_shadow_capture_result_to_kv_index_resolution_plan",
        "request_id": request_id,
        "req_pool_idx": req_pool_idx,
        "seq_len": seq_len,
        "layer_id": layer_id,
        "relaykv_working_block_spans": relaykv_working_block_spans
        if relaykv_working_block_spans is not None
        else [_block_span(1, 64, 128)],
        "full_kv_block_spans": full_kv_block_spans
        if full_kv_block_spans is not None
        else [_block_span(0, 0, 64), _block_span(1, 64, 128)],
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


def _pass_flow_plans(poison: _PoisonTensorLike | None = None) -> list[dict[str, Any]]:
    return [
        _kv_index_resolution_plan(
            "req-a",
            7,
            full_kv_block_spans=[_block_span(0, 0, 64), _block_span(1, 64, 128)],
            relaykv_working_block_spans=[_block_span(1, 64, 128)],
            poison=poison,
        ),
        _kv_index_resolution_plan(
            "req-b",
            8,
            full_kv_block_spans=[_block_span(0, 0, 64), _block_span(1, 64, 128)],
            relaykv_working_block_spans=[_block_span(0, 0, 64)],
        ),
    ]


def _req_to_token_backing() -> dict[Any, Any]:
    return {
        7: list(range(1000, 1256)),
        "8": list(range(2000, 2256)),
    }


def _checksum(entries: list[int]) -> int:
    return sum((index + 1) * entry for index, entry in enumerate(entries)) % 1000000007


def _assert_pass_flow() -> dict[str, Any]:
    poison = _PoisonTensorLike()
    plans = _pass_flow_plans(poison)
    backing = _req_to_token_backing()
    before_plans = copy.deepcopy(plans)
    before_backing = copy.deepcopy(backing)

    payloads = build_relaykv_req_to_token_readonly_adapter_payloads_for_smoke(
        plans,
        req_to_token_backing_by_req_pool_idx=backing,
        read_req_to_token=True,
        max_tokens_per_request=128,
        max_blocks_per_request=2,
        max_total_tokens=256,
        max_preview_entries=8,
    )
    if plans != before_plans:
        raise AssertionError("kv index resolution plans were mutated")
    if backing != before_backing:
        raise AssertionError("req_to_token backing was mutated")
    if poison.forbidden_access_called:
        raise AssertionError("poison tensor-like object was accessed")
    if len(payloads) != 2:
        raise AssertionError(payloads)

    req_a_entries = list(range(1000, 1128))
    req_b_entries = list(range(2000, 2128))
    expected_by_request = {
        "req-a": {
            "req_pool_idx": 7,
            "preview_entries": req_a_entries[:8],
            "entry_min": min(req_a_entries),
            "entry_max": max(req_a_entries),
            "entry_checksum": _checksum(req_a_entries),
        },
        "req-b": {
            "req_pool_idx": 8,
            "preview_entries": req_b_entries[:8],
            "entry_min": min(req_b_entries),
            "entry_max": max(req_b_entries),
            "entry_checksum": _checksum(req_b_entries),
        },
    }

    for payload in payloads:
        if payload["event_type"] != "relaykv_req_to_token_readonly_adapter_payload":
            raise AssertionError(payload)
        if payload["adapter_state"] != "adapter_payload_ready":
            raise AssertionError(payload)
        if payload["adapter_mode"] != "readonly_bounded_preview":
            raise AssertionError(payload)
        if (
            payload["source"]
            != "kv_index_resolution_plan_to_req_to_token_readonly_adapter_payload"
        ):
            raise AssertionError(payload)
        if payload["requested_block_count"] != 2:
            raise AssertionError(payload)
        if payload["requested_token_count"] != 128:
            raise AssertionError(payload)
        if payload["read_token_count"] != 128:
            raise AssertionError(payload)
        if payload["preview_entry_count"] != 8:
            raise AssertionError(payload)
        if payload["entry_count"] != 128:
            raise AssertionError(payload)
        if payload["truncated_preview"] is not True:
            raise AssertionError(payload)
        if payload["req_to_token_read"] is not True:
            raise AssertionError(payload)
        if payload["req_to_token_read_count"] != 128:
            raise AssertionError(payload)
        if payload["token_to_kv_pool_read"] is not False:
            raise AssertionError(payload)
        if payload["token_to_kv_pool_read_count"] != 0:
            raise AssertionError(payload)
        if payload["kv_pool_read"] is not False:
            raise AssertionError(payload)
        if payload["kv_snapshot"] is not False:
            raise AssertionError(payload)
        if payload["tensor_read"] is not False:
            raise AssertionError(payload)
        if payload["attention_comparison_executed"] is not False:
            raise AssertionError(payload)
        if payload["attention_override"] is not False:
            raise AssertionError(payload)
        if payload["runtime_writeback"] is not False:
            raise AssertionError(payload)
        if payload["scheduler_policy_noop"] is not True:
            raise AssertionError(payload)
        if payload["kv_cache_mutation"] is not False:
            raise AssertionError(payload)
        if payload["source_mutated"] is not False:
            raise AssertionError(payload)
        if payload["blocking_reasons"] != []:
            raise AssertionError(payload)
        if "readonly_bounded_req_to_token_adapter" not in payload["warning_reasons"]:
            raise AssertionError(payload)
        if "no_token_to_kv_pool_read" not in payload["warning_reasons"]:
            raise AssertionError(payload)
        if "preview_only_no_full_entries_logged" not in payload["warning_reasons"]:
            raise AssertionError(payload)
        if "req_to_token_entries" in payload:
            raise AssertionError(payload)
        expected = expected_by_request[payload["request_id"]]
        if payload["req_pool_idx"] != expected["req_pool_idx"]:
            raise AssertionError(payload)
        if payload["preview_entries"] != expected["preview_entries"]:
            raise AssertionError(payload)
        if payload["entry_min"] != expected["entry_min"]:
            raise AssertionError(payload)
        if payload["entry_max"] != expected["entry_max"]:
            raise AssertionError(payload)
        if payload["entry_checksum"] != expected["entry_checksum"]:
            raise AssertionError(payload)

    summary = summarize_relaykv_req_to_token_readonly_adapter_payloads_for_smoke(
        payloads
    )
    if summary["adapter_payload_ready_count"] != 2:
        raise AssertionError(summary)
    if summary["requested_block_count"] != 4:
        raise AssertionError(summary)
    if summary["requested_token_count"] != 256:
        raise AssertionError(summary)
    if summary["read_token_count"] != 256:
        raise AssertionError(summary)
    if summary["preview_entry_count"] != 16:
        raise AssertionError(summary)
    if summary["truncated_preview_count"] != 2:
        raise AssertionError(summary)
    if summary["req_to_token_read_count"] != 256:
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

    return {"payloads": payloads, "summary": summary}


def _assert_blocked_case(
    plans: list[dict[str, Any]],
    *,
    expected_reason: str,
    backing: dict[Any, Any] | None = None,
    read_req_to_token: bool = True,
    max_tokens_per_request: int = 128,
    max_blocks_per_request: int = 2,
    max_total_tokens: int = 256,
    max_preview_entries: int = 8,
) -> dict[str, Any]:
    payloads = build_relaykv_req_to_token_readonly_adapter_payloads_for_smoke(
        plans,
        req_to_token_backing_by_req_pool_idx=backing,
        read_req_to_token=read_req_to_token,
        max_tokens_per_request=max_tokens_per_request,
        max_blocks_per_request=max_blocks_per_request,
        max_total_tokens=max_total_tokens,
        max_preview_entries=max_preview_entries,
    )
    payload = payloads[0]
    if expected_reason not in payload["blocking_reasons"]:
        raise AssertionError(payload)
    if payload["adapter_state"] != "blocked":
        raise AssertionError(payload)
    if payload["requested_block_count"] != 0:
        raise AssertionError(payload)
    if payload["requested_token_count"] != 0:
        raise AssertionError(payload)
    if payload["read_token_count"] != 0:
        raise AssertionError(payload)
    if payload["preview_entry_count"] != 0:
        raise AssertionError(payload)
    if payload["preview_entries"] != []:
        raise AssertionError(payload)
    if payload["entry_count"] != 0:
        raise AssertionError(payload)
    if payload["entry_min"] is not None:
        raise AssertionError(payload)
    if payload["entry_max"] is not None:
        raise AssertionError(payload)
    if payload["entry_checksum"] is not None:
        raise AssertionError(payload)
    if payload["truncated_preview"] is not False:
        raise AssertionError(payload)
    if payload["req_to_token_read"] is not False:
        raise AssertionError(payload)
    if payload["req_to_token_read_count"] != 0:
        raise AssertionError(payload)
    if payload["token_to_kv_pool_read_count"] != 0:
        raise AssertionError(payload)
    if payload["kv_pool_read"] is not False:
        raise AssertionError(payload)
    if payload["kv_snapshot"] is not False:
        raise AssertionError(payload)
    if payload["tensor_read"] is not False:
        raise AssertionError(payload)
    if payload["attention_comparison_executed"] is not False:
        raise AssertionError(payload)
    if payload["attention_override"] is not False:
        raise AssertionError(payload)
    if payload["runtime_writeback"] is not False:
        raise AssertionError(payload)
    if payload["scheduler_policy_noop"] is not True:
        raise AssertionError(payload)
    if payload["kv_cache_mutation"] is not False:
        raise AssertionError(payload)
    if payload["source_mutated"] is not False:
        raise AssertionError(payload)

    summary = summarize_relaykv_req_to_token_readonly_adapter_payloads_for_smoke(
        payloads
    )
    if summary["blocked_count"] != 1:
        raise AssertionError(summary)
    for key in (
        "requested_block_count",
        "requested_token_count",
        "read_token_count",
        "preview_entry_count",
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
    ):
        if summary[key] != 0:
            raise AssertionError(summary)
    return summary


def _assert_blocked_cases() -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []

    wrong_event = [_kv_index_resolution_plan("req-a", 7)]
    wrong_event[0]["event_type"] = "wrong"
    outputs.append(
        _assert_blocked_case(
            wrong_event,
            expected_reason="not_kv_index_resolution_plan",
            backing=_req_to_token_backing(),
        )
    )

    wrong_state = [_kv_index_resolution_plan("req-a", 7)]
    wrong_state[0]["resolution_state"] = "blocked"
    outputs.append(
        _assert_blocked_case(
            wrong_state,
            expected_reason="kv_index_resolution_not_block_span_resolved",
            backing=_req_to_token_backing(),
        )
    )

    wrong_mode = [_kv_index_resolution_plan("req-a", 7)]
    wrong_mode[0]["resolution_mode"] = "other"
    outputs.append(
        _assert_blocked_case(
            wrong_mode,
            expected_reason="kv_index_resolution_not_metadata_only",
            backing=_req_to_token_backing(),
        )
    )

    read_disabled = [_kv_index_resolution_plan("req-a", 7)]
    outputs.append(
        _assert_blocked_case(
            read_disabled,
            expected_reason="read_req_to_token_not_enabled",
            backing=_req_to_token_backing(),
            read_req_to_token=False,
        )
    )

    bad_req_pool_idx = [_kv_index_resolution_plan("req-a", 7)]
    bad_req_pool_idx[0]["req_pool_idx"] = None
    outputs.append(
        _assert_blocked_case(
            bad_req_pool_idx,
            expected_reason="req_pool_idx_missing_or_invalid",
            backing=_req_to_token_backing(),
        )
    )

    bad_seq_len = [_kv_index_resolution_plan("req-a", 7)]
    bad_seq_len[0]["seq_len"] = None
    outputs.append(
        _assert_blocked_case(
            bad_seq_len,
            expected_reason="seq_len_missing_or_invalid",
            backing=_req_to_token_backing(),
        )
    )

    missing_backing = [_kv_index_resolution_plan("req-a", 7)]
    outputs.append(
        _assert_blocked_case(
            missing_backing,
            expected_reason="req_to_token_backing_missing",
            backing=None,
        )
    )

    missing_req_pool_backing = [_kv_index_resolution_plan("req-a", 7)]
    backing = _req_to_token_backing()
    del backing[7]
    outputs.append(
        _assert_blocked_case(
            missing_req_pool_backing,
            expected_reason="req_to_token_backing_for_req_pool_missing",
            backing=backing,
        )
    )

    bad_backing_type = [_kv_index_resolution_plan("req-a", 7)]
    backing = _req_to_token_backing()
    backing[7] = {"bad": "table"}
    outputs.append(
        _assert_blocked_case(
            bad_backing_type,
            expected_reason="req_to_token_backing_not_list_or_tuple",
            backing=backing,
        )
    )

    too_many_blocks = [_kv_index_resolution_plan("req-a", 7)]
    outputs.append(
        _assert_blocked_case(
            too_many_blocks,
            expected_reason="requested_block_count_exceeds_limit",
            backing=_req_to_token_backing(),
            max_blocks_per_request=1,
        )
    )

    too_many_tokens = [_kv_index_resolution_plan("req-a", 7)]
    outputs.append(
        _assert_blocked_case(
            too_many_tokens,
            expected_reason="requested_token_count_exceeds_limit",
            backing=_req_to_token_backing(),
            max_tokens_per_request=64,
        )
    )

    too_many_total_tokens = _pass_flow_plans()
    payloads = build_relaykv_req_to_token_readonly_adapter_payloads_for_smoke(
        too_many_total_tokens,
        req_to_token_backing_by_req_pool_idx=_req_to_token_backing(),
        read_req_to_token=True,
        max_tokens_per_request=128,
        max_blocks_per_request=2,
        max_total_tokens=200,
        max_preview_entries=8,
    )
    if payloads[1]["adapter_state"] != "blocked":
        raise AssertionError(payloads)
    if (
        "total_requested_token_count_exceeds_limit"
        not in payloads[1]["blocking_reasons"]
    ):
        raise AssertionError(payloads[1])
    outputs.append(
        summarize_relaykv_req_to_token_readonly_adapter_payloads_for_smoke(payloads)
    )

    out_of_table = [_kv_index_resolution_plan("req-a", 7)]
    backing = _req_to_token_backing()
    backing[7] = list(range(1000, 1100))
    outputs.append(
        _assert_blocked_case(
            out_of_table,
            expected_reason="token_position_out_of_req_to_token_table",
            backing=backing,
        )
    )

    bad_entry = [_kv_index_resolution_plan("req-a", 7)]
    backing = _req_to_token_backing()
    backing[7][0] = "bad"
    outputs.append(
        _assert_blocked_case(
            bad_entry,
            expected_reason="req_to_token_entry_not_int",
            backing=backing,
        )
    )

    invalid_span = [_kv_index_resolution_plan("req-a", 7)]
    invalid_span[0]["full_kv_block_spans"][0]["token_count"] = 0
    outputs.append(
        _assert_blocked_case(
            invalid_span,
            expected_reason="invalid_block_span",
            backing=_req_to_token_backing(),
        )
    )

    out_of_seq = [_kv_index_resolution_plan("req-a", 7)]
    out_of_seq[0]["full_kv_block_spans"][0]["token_end"] = 300
    out_of_seq[0]["full_kv_block_spans"][0]["token_count"] = 300
    outputs.append(
        _assert_blocked_case(
            out_of_seq,
            expected_reason="token_span_out_of_seq_len",
            backing=_req_to_token_backing(),
        )
    )

    token_to_kv_pool_true = [_kv_index_resolution_plan("req-a", 7)]
    token_to_kv_pool_true[0]["token_to_kv_pool_read"] = True
    outputs.append(
        _assert_blocked_case(
            token_to_kv_pool_true,
            expected_reason="token_to_kv_pool_read_not_allowed",
            backing=_req_to_token_backing(),
        )
    )

    kv_pool_true = [_kv_index_resolution_plan("req-a", 7)]
    kv_pool_true[0]["kv_pool_read"] = True
    outputs.append(
        _assert_blocked_case(
            kv_pool_true,
            expected_reason="kv_pool_read_not_allowed",
            backing=_req_to_token_backing(),
        )
    )

    tensor_true = [_kv_index_resolution_plan("req-a", 7)]
    tensor_true[0]["tensor_read"] = True
    outputs.append(
        _assert_blocked_case(
            tensor_true,
            expected_reason="tensor_read_not_allowed",
            backing=_req_to_token_backing(),
        )
    )

    comparison_true = [_kv_index_resolution_plan("req-a", 7)]
    comparison_true[0]["attention_comparison_executed"] = True
    outputs.append(
        _assert_blocked_case(
            comparison_true,
            expected_reason="attention_comparison_executed_not_allowed",
            backing=_req_to_token_backing(),
        )
    )

    override_true = [_kv_index_resolution_plan("req-a", 7)]
    override_true[0]["attention_override"] = True
    outputs.append(
        _assert_blocked_case(
            override_true,
            expected_reason="attention_override_true_not_allowed",
            backing=_req_to_token_backing(),
        )
    )

    return outputs


def main() -> None:
    pass_flow = _assert_pass_flow()
    blocked = _assert_blocked_cases()
    print(
        json.dumps(
            {
                "date_basis_jst": "2026-05-04",
                "pass_flow": {
                    "adapter_payload_ready_count": pass_flow["summary"][
                        "adapter_payload_ready_count"
                    ],
                    "requested_block_count": pass_flow["summary"][
                        "requested_block_count"
                    ],
                    "requested_token_count": pass_flow["summary"][
                        "requested_token_count"
                    ],
                    "read_token_count": pass_flow["summary"]["read_token_count"],
                    "preview_entry_count": pass_flow["summary"][
                        "preview_entry_count"
                    ],
                    "truncated_preview_count": pass_flow["summary"][
                        "truncated_preview_count"
                    ],
                    "req_to_token_read_count": pass_flow["summary"][
                        "req_to_token_read_count"
                    ],
                },
                "blocked_case_count": len(blocked),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
