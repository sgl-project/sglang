from __future__ import annotations

import copy
import json
from typing import Any

from sglang.srt.relaykv.metrics import (
    build_relaykv_req_to_token_runtime_inspection_payloads_for_smoke,
    summarize_relaykv_req_to_token_runtime_inspection_payloads_for_smoke,
)


class FakeReqToTokenTable:
    def __init__(self, shape: tuple[int, int], device: str, dtype: str) -> None:
        self.shape = shape
        self.device = device
        self.dtype = dtype
        self.len_called = False
        self.iter_called = False
        self.getitem_called = False
        self.cpu_called = False
        self.tolist_called = False
        self.item_called = False
        self.numpy_called = False

    def __deepcopy__(self, memo: dict[int, Any]) -> "FakeReqToTokenTable":
        return self

    def __len__(self) -> int:
        self.len_called = True
        raise AssertionError("__len__() must not be called")

    def __iter__(self):
        self.iter_called = True
        raise AssertionError("__iter__() must not be called")

    def __getitem__(self, index: int) -> None:
        self.getitem_called = True
        raise AssertionError("__getitem__() must not be called")

    def cpu(self) -> None:
        self.cpu_called = True
        raise AssertionError("cpu() must not be called")

    def tolist(self) -> None:
        self.tolist_called = True
        raise AssertionError("tolist() must not be called")

    def item(self) -> None:
        self.item_called = True
        raise AssertionError("item() must not be called")

    def numpy(self) -> None:
        self.numpy_called = True
        raise AssertionError("numpy() must not be called")

    @property
    def forbidden_read_called(self) -> bool:
        return (
            self.len_called
            or self.iter_called
            or self.getitem_called
            or self.cpu_called
            or self.tolist_called
            or self.item_called
            or self.numpy_called
        )


class FakeReqToTokenPool:
    def __init__(self, req_to_token: Any) -> None:
        self._req_to_token = req_to_token
        self.req_to_token_access_count = 0

    @property
    def req_to_token(self) -> Any:
        self.req_to_token_access_count += 1
        return self._req_to_token


class _MissingReqToTokenPool:
    def __init__(self) -> None:
        self.other = "missing"


class _AttrFailureReqToTokenPool:
    @property
    def req_to_token(self) -> Any:
        raise RuntimeError("attr access must be handled")


def _forward_batch_like(**overrides: Any) -> dict[str, Any]:
    payload = {
        "request_id": "req-a",
        "layer_id": 14,
        "batch_id": "runtime-inspection-batch-a",
        "token_to_kv_pool_read": False,
        "kv_pool_read": False,
        "tensor_read": False,
        "attention_comparison_executed": False,
        "attention_override": False,
    }
    payload.update(overrides)
    return payload


def _assert_pass_flow() -> dict[str, Any]:
    table = FakeReqToTokenTable(
        shape=(16, 1024),
        device="cuda:0",
        dtype="torch.int32",
    )
    pool = FakeReqToTokenPool(table)
    forward_batch = _forward_batch_like()
    before_forward_batch = copy.deepcopy(forward_batch)

    payloads = build_relaykv_req_to_token_runtime_inspection_payloads_for_smoke(
        forward_batch_like=forward_batch,
        req_to_token_pool=pool,
        inspect_req_to_token=True,
    )
    if forward_batch != before_forward_batch:
        raise AssertionError("forward_batch_like was mutated")
    if pool.req_to_token_access_count != 1:
        raise AssertionError(pool.req_to_token_access_count)
    if table.forbidden_read_called:
        raise AssertionError("req_to_token values or tensor reads were touched")
    if len(payloads) != 1:
        raise AssertionError(payloads)

    payload = payloads[0]
    if payload["event_type"] != "relaykv_req_to_token_runtime_inspection_payload":
        raise AssertionError(payload)
    if payload["inspection_state"] != "metadata_observed":
        raise AssertionError(payload)
    if payload["inspection_mode"] != "metadata_only":
        raise AssertionError(payload)
    if payload["source"] != "req_to_token_pool_to_runtime_inspection_payload":
        raise AssertionError(payload)
    if payload["metadata_observed"] is not True:
        raise AssertionError(payload)
    if payload["req_to_token_attr_present"] is not True:
        raise AssertionError(payload)
    if payload["req_to_token_attr_observed"] is not True:
        raise AssertionError(payload)
    if payload["actual_req_to_token_pool_inspection"] is not True:
        raise AssertionError(payload)
    if payload["req_to_token_type"] != "FakeReqToTokenTable":
        raise AssertionError(payload)
    if payload["req_to_token_module"] != "__main__":
        raise AssertionError(payload)
    if payload["req_to_token_qualname"] != "FakeReqToTokenTable":
        raise AssertionError(payload)
    if payload["req_to_token_shape"] != (16, 1024):
        raise AssertionError(payload)
    if payload["req_to_token_device"] != "cuda:0":
        raise AssertionError(payload)
    if payload["req_to_token_dtype"] != "torch.int32":
        raise AssertionError(payload)
    if payload["req_to_token_read"] is not False:
        raise AssertionError(payload)
    if payload["req_to_token_read_count"] != 0:
        raise AssertionError(payload)
    if payload["actual_req_to_token_pool_read"] is not False:
        raise AssertionError(payload)
    if payload["actual_req_to_token_pool_read_count"] != 0:
        raise AssertionError(payload)
    if payload["token_to_kv_pool_read_count"] != 0:
        raise AssertionError(payload)
    if payload["kv_pool_read"] is not False:
        raise AssertionError(payload)
    if payload["tensor_read"] is not False:
        raise AssertionError(payload)
    if payload["attention_override"] is not False:
        raise AssertionError(payload)
    if payload["blocking_reasons"] != []:
        raise AssertionError(payload)

    summary = summarize_relaykv_req_to_token_runtime_inspection_payloads_for_smoke(
        payloads
    )
    if summary["metadata_observed_count"] != 1:
        raise AssertionError(summary)
    if summary["req_to_token_attr_present_count"] != 1:
        raise AssertionError(summary)
    if summary["actual_req_to_token_pool_inspection_count"] != 1:
        raise AssertionError(summary)
    if summary["req_to_token_attr_observed_count"] != 1:
        raise AssertionError(summary)
    for key in (
        "req_to_token_read_count",
        "actual_req_to_token_pool_read_count",
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
    return {"payloads": payloads, "summary": summary}


def _assert_blocked_case(
    *,
    expected_reason: str,
    forward_batch_like: dict[str, Any] | None = None,
    req_to_token_pool: Any = None,
    inspect_req_to_token: bool = True,
) -> dict[str, Any]:
    payloads = build_relaykv_req_to_token_runtime_inspection_payloads_for_smoke(
        forward_batch_like=forward_batch_like,
        req_to_token_pool=req_to_token_pool,
        inspect_req_to_token=inspect_req_to_token,
    )
    payload = payloads[0]
    if expected_reason not in payload["blocking_reasons"]:
        raise AssertionError(payload)
    if payload["inspection_state"] != "blocked":
        raise AssertionError(payload)
    if payload["metadata_observed"] is not False:
        raise AssertionError(payload)
    if payload["req_to_token_read_count"] != 0:
        raise AssertionError(payload)
    if payload["actual_req_to_token_pool_read_count"] != 0:
        raise AssertionError(payload)

    summary = summarize_relaykv_req_to_token_runtime_inspection_payloads_for_smoke(
        payloads
    )
    if summary["blocked_count"] != 1:
        raise AssertionError(summary)
    if summary["metadata_observed_count"] != 0:
        raise AssertionError(summary)
    for key in (
        "req_to_token_read_count",
        "actual_req_to_token_pool_read_count",
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

    outputs.append(
        _assert_blocked_case(
            expected_reason="inspect_req_to_token_not_enabled",
            forward_batch_like=_forward_batch_like(),
            req_to_token_pool=FakeReqToTokenPool(
                FakeReqToTokenTable((16, 1024), "cuda:0", "torch.int32")
            ),
            inspect_req_to_token=False,
        )
    )
    outputs.append(
        _assert_blocked_case(
            expected_reason="req_to_token_pool_missing",
            forward_batch_like=_forward_batch_like(),
            req_to_token_pool=None,
        )
    )
    outputs.append(
        _assert_blocked_case(
            expected_reason="req_to_token_attr_missing",
            forward_batch_like=_forward_batch_like(),
            req_to_token_pool=_MissingReqToTokenPool(),
        )
    )
    outputs.append(
        _assert_blocked_case(
            expected_reason="req_to_token_attr_access_failed",
            forward_batch_like=_forward_batch_like(),
            req_to_token_pool=_AttrFailureReqToTokenPool(),
        )
    )
    outputs.append(
        _assert_blocked_case(
            expected_reason="token_to_kv_pool_read_not_allowed",
            forward_batch_like=_forward_batch_like(token_to_kv_pool_read=True),
            req_to_token_pool=FakeReqToTokenPool(
                FakeReqToTokenTable((16, 1024), "cuda:0", "torch.int32")
            ),
        )
    )
    outputs.append(
        _assert_blocked_case(
            expected_reason="kv_pool_read_not_allowed",
            forward_batch_like=_forward_batch_like(kv_pool_read=True),
            req_to_token_pool=FakeReqToTokenPool(
                FakeReqToTokenTable((16, 1024), "cuda:0", "torch.int32")
            ),
        )
    )
    outputs.append(
        _assert_blocked_case(
            expected_reason="tensor_read_not_allowed",
            forward_batch_like=_forward_batch_like(tensor_read=True),
            req_to_token_pool=FakeReqToTokenPool(
                FakeReqToTokenTable((16, 1024), "cuda:0", "torch.int32")
            ),
        )
    )
    outputs.append(
        _assert_blocked_case(
            expected_reason="attention_comparison_executed_not_allowed",
            forward_batch_like=_forward_batch_like(
                attention_comparison_executed=True
            ),
            req_to_token_pool=FakeReqToTokenPool(
                FakeReqToTokenTable((16, 1024), "cuda:0", "torch.int32")
            ),
        )
    )
    outputs.append(
        _assert_blocked_case(
            expected_reason="attention_override_true_not_allowed",
            forward_batch_like=_forward_batch_like(attention_override=True),
            req_to_token_pool=FakeReqToTokenPool(
                FakeReqToTokenTable((16, 1024), "cuda:0", "torch.int32")
            ),
        )
    )

    return outputs


def main() -> None:
    pass_flow = _assert_pass_flow()
    blocked = _assert_blocked_cases()
    print(
        json.dumps(
            {
                "pass_flow": {
                    "metadata_observed_count": pass_flow["summary"][
                        "metadata_observed_count"
                    ],
                    "req_to_token_attr_present_count": pass_flow["summary"][
                        "req_to_token_attr_present_count"
                    ],
                    "actual_req_to_token_pool_inspection_count": pass_flow[
                        "summary"
                    ]["actual_req_to_token_pool_inspection_count"],
                    "req_to_token_attr_observed_count": pass_flow["summary"][
                        "req_to_token_attr_observed_count"
                    ],
                },
                "blocked_case_count": len(blocked),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
