from __future__ import annotations

import json
import signal

import torch

from sglang.srt.relaykv import RelayKVConfig, make_shadow_plan
from sglang.srt.relaykv.memory import copy_host_backup_candidate_for_smoke


def _make_source_tensor() -> torch.Tensor:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    return torch.arange(64, dtype=dtype, device=device).reshape(2, 4, 8)


def _assert_applied_copy(log: dict[str, object], source: torch.Tensor) -> None:
    if log["runtime_policy_state"] != "applied_candidate":
        raise AssertionError(log)
    if log["host_backup_copy_candidate"] is not True:
        raise AssertionError(log)
    if log["host_backup_copy_executed"] is not True:
        raise AssertionError(log)
    if log["copy_equal"] is not True:
        raise AssertionError(log)
    if log["backup_device"] != "cpu":
        raise AssertionError(log)
    if log["source_shape"] != [2, 4, 8]:
        raise AssertionError(log)
    if log["backup_shape"] != [2, 4, 8]:
        raise AssertionError(log)
    if log["source_dtype"] != str(source.dtype):
        raise AssertionError(log)
    if log["backup_dtype"] != str(source.dtype):
        raise AssertionError(log)
    if log["copy_numel"] != source.numel():
        raise AssertionError(log)
    if log["copy_nbytes"] != source.numel() * source.element_size():
        raise AssertionError(log)
    if log["attention_override"] is not False:
        raise AssertionError(log)
    if log["kv_cache_mutation"] is not False:
        raise AssertionError(log)
    if log["scheduler_policy_noop"] is not True:
        raise AssertionError(log)


def _assert_fallback_noop(log: dict[str, object]) -> None:
    if log["runtime_policy_state"] != "fallback_candidate":
        raise AssertionError(log)
    if log["host_backup_copy_candidate"] is not False:
        raise AssertionError(log)
    if log["host_backup_copy_executed"] is not False:
        raise AssertionError(log)
    if log["fallback_candidate_noop_guard"] is not True:
        raise AssertionError(log)
    if log["host_backup_copy_skipped_reason"] != "fallback_candidate_noop_guard":
        raise AssertionError(log)
    if log["attention_override"] is not False:
        raise AssertionError(log)
    if log["kv_cache_mutation"] is not False:
        raise AssertionError(log)
    if log["scheduler_policy_noop"] is not True:
        raise AssertionError(log)
    if log["copy_numel"] != 0:
        raise AssertionError(log)
    if log["copy_nbytes"] != 0:
        raise AssertionError(log)
    if log["copy_equal"] is not False:
        raise AssertionError(log)


def main() -> None:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    source = _make_source_tensor()

    applied_plan = make_shadow_plan(
        2535,
        RelayKVConfig(
            enabled=True,
            mode="shadow",
            kv_working_budget_tokens=2048,
            recent_window=512,
            anchor_blocks=2,
            budget_block_size=128,
            retrieval_top_k=4,
        ),
        kv_bytes_per_token=28672,
        request_id="copy-smoke-applied",
    )
    if applied_plan.runtime_policy_state != "applied_candidate":
        raise AssertionError(applied_plan)

    fallback_plan = make_shadow_plan(
        8192,
        RelayKVConfig(
            enabled=True,
            mode="shadow",
            kv_working_budget_tokens=2048,
            recent_window=512,
            anchor_blocks=2,
            budget_block_size=128,
            retrieval_top_k=4,
        ),
        kv_bytes_per_token=28672,
        request_id="copy-smoke-fallback",
    )
    if fallback_plan.runtime_policy_state != "fallback_candidate":
        raise AssertionError(fallback_plan)

    applied_result = copy_host_backup_candidate_for_smoke(
        plan=applied_plan,
        source_tensor=source,
    )
    fallback_result = copy_host_backup_candidate_for_smoke(
        plan=fallback_plan,
        source_tensor=source,
    )

    applied_log = applied_result.to_log_dict()
    fallback_log = fallback_result.to_log_dict()
    _assert_applied_copy(applied_log, source)
    _assert_fallback_noop(fallback_log)
    if applied_result.backup_tensor is None:
        raise AssertionError(applied_result)
    if not applied_result.backup_tensor.equal(source.detach().to("cpu")):
        raise AssertionError(applied_log)
    if fallback_result.backup_tensor is not None:
        raise AssertionError(fallback_log)

    print("relaykv_host_backup_copy_smoke: ok")
    print("relaykv_host_backup_copy_applied=" + json.dumps(applied_log, sort_keys=True))
    print("relaykv_host_backup_copy_fallback=" + json.dumps(fallback_log, sort_keys=True))


if __name__ == "__main__":
    main()
