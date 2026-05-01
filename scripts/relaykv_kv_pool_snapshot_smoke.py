from __future__ import annotations

import json
import signal

import torch

from sglang.srt.relaykv import RelayKVConfig, make_shadow_plan
from sglang.srt.relaykv.memory import (
    copy_host_backup_candidate_for_smoke,
    snapshot_kv_pool_for_host_backup_smoke,
)


class _FakeMHATokenToKVPool:
    def __init__(self, *, device: str) -> None:
        self.start_layer = 0
        self.dtype = torch.float16
        self.device = device
        self.k_buffer = [
            torch.arange(8 * 2 * 8, dtype=self.dtype, device=device).reshape(8, 2, 8),
        ]
        self.v_buffer = [
            (torch.arange(8 * 2 * 8, dtype=self.dtype, device=device) + 1024).reshape(
                8, 2, 8
            ),
        ]


class _FakeAllocator:
    page_size = 1

    def __init__(self, kvcache: _FakeMHATokenToKVPool) -> None:
        self._kvcache = kvcache

    def get_kvcache(self) -> _FakeMHATokenToKVPool:
        return self._kvcache


def _relaykv_config() -> RelayKVConfig:
    return RelayKVConfig(
        enabled=True,
        mode="shadow",
        kv_working_budget_tokens=2048,
        recent_window=512,
        anchor_blocks=2,
        budget_block_size=128,
        retrieval_top_k=4,
    )


def _assert_applied_snapshot(
    snapshot_log: dict[str, object],
    copy_log: dict[str, object],
) -> None:
    if snapshot_log["runtime_policy_state"] != "applied_candidate":
        raise AssertionError(snapshot_log)
    if snapshot_log["kv_pool_type"] != "_FakeMHATokenToKVPool":
        raise AssertionError(snapshot_log)
    if snapshot_log["snapshot_created"] is not True:
        raise AssertionError(snapshot_log)
    if snapshot_log["source_shape"] != [8, 2, 8]:
        raise AssertionError(snapshot_log)
    if snapshot_log["snapshot_shape"] != [4, 2, 2, 8]:
        raise AssertionError(snapshot_log)
    if snapshot_log["token_indices"] != [2, 3, 4, 5]:
        raise AssertionError(snapshot_log)
    if snapshot_log["source_mutated"] is not False:
        raise AssertionError(snapshot_log)
    if snapshot_log["attention_override"] is not False:
        raise AssertionError(snapshot_log)
    if snapshot_log["kv_cache_mutation"] is not False:
        raise AssertionError(snapshot_log)
    if snapshot_log["scheduler_policy_noop"] is not True:
        raise AssertionError(snapshot_log)

    if copy_log["host_backup_copy_executed"] is not True:
        raise AssertionError(copy_log)
    if copy_log["backup_shape"] != [4, 2, 2, 8]:
        raise AssertionError(copy_log)
    if copy_log["backup_device"] != "cpu":
        raise AssertionError(copy_log)
    if copy_log["copy_numel"] != 128:
        raise AssertionError(copy_log)
    if copy_log["copy_nbytes"] != 256:
        raise AssertionError(copy_log)
    if copy_log["copy_equal"] is not True:
        raise AssertionError(copy_log)
    if copy_log["attention_override"] is not False:
        raise AssertionError(copy_log)
    if copy_log["kv_cache_mutation"] is not False:
        raise AssertionError(copy_log)
    if copy_log["scheduler_policy_noop"] is not True:
        raise AssertionError(copy_log)


def _assert_fallback_noop(
    snapshot_log: dict[str, object],
    copy_log: dict[str, object],
) -> None:
    if snapshot_log["runtime_policy_state"] != "fallback_candidate":
        raise AssertionError(snapshot_log)
    if snapshot_log["snapshot_created"] is not False:
        raise AssertionError(snapshot_log)
    if snapshot_log["fallback_candidate_noop_guard"] is not True:
        raise AssertionError(snapshot_log)
    if snapshot_log["snapshot_skipped_reason"] != "fallback_candidate_noop_guard":
        raise AssertionError(snapshot_log)
    if snapshot_log["source_mutated"] is not False:
        raise AssertionError(snapshot_log)
    if snapshot_log["attention_override"] is not False:
        raise AssertionError(snapshot_log)
    if snapshot_log["kv_cache_mutation"] is not False:
        raise AssertionError(snapshot_log)

    if copy_log["host_backup_copy_executed"] is not False:
        raise AssertionError(copy_log)
    if copy_log["fallback_candidate_noop_guard"] is not True:
        raise AssertionError(copy_log)
    if copy_log["host_backup_copy_skipped_reason"] != "fallback_candidate_noop_guard":
        raise AssertionError(copy_log)
    if copy_log["copy_numel"] != 0:
        raise AssertionError(copy_log)
    if copy_log["copy_nbytes"] != 0:
        raise AssertionError(copy_log)
    if copy_log["copy_equal"] is not False:
        raise AssertionError(copy_log)


def _merge_snapshot_and_copy_logs(
    snapshot_log: dict[str, object],
    copy_log: dict[str, object],
) -> dict[str, object]:
    copy_fields = {
        "host_backup_copy_candidate",
        "host_backup_copy_executed",
        "host_backup_copy_skipped_reason",
        "backup_shape",
        "backup_dtype",
        "backup_device",
        "copy_numel",
        "copy_nbytes",
        "copy_equal",
    }
    payload = dict(snapshot_log)
    payload.update({key: copy_log[key] for key in copy_fields})
    return payload


def main() -> None:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    allocator = _FakeAllocator(_FakeMHATokenToKVPool(device=device))
    token_indices = [2, 3, 4, 5]

    applied_plan = make_shadow_plan(
        2535,
        _relaykv_config(),
        kv_bytes_per_token=28672,
        request_id="kv-pool-snapshot-applied",
    )
    if applied_plan.runtime_policy_state != "applied_candidate":
        raise AssertionError(applied_plan)

    fallback_plan = make_shadow_plan(
        8192,
        _relaykv_config(),
        kv_bytes_per_token=28672,
        request_id="kv-pool-snapshot-fallback",
    )
    if fallback_plan.runtime_policy_state != "fallback_candidate":
        raise AssertionError(fallback_plan)

    applied_snapshot = snapshot_kv_pool_for_host_backup_smoke(
        plan=applied_plan,
        token_to_kv_pool_allocator=allocator,
        token_indices=token_indices,
        layer_idx=0,
    )
    if applied_snapshot.snapshot_tensor is None:
        raise AssertionError(applied_snapshot)
    applied_copy = copy_host_backup_candidate_for_smoke(
        plan=applied_plan,
        source_tensor=applied_snapshot.snapshot_tensor,
    )

    fallback_snapshot = snapshot_kv_pool_for_host_backup_smoke(
        plan=fallback_plan,
        token_to_kv_pool_allocator=allocator,
        token_indices=token_indices,
        layer_idx=0,
    )
    fallback_source = (
        applied_snapshot.snapshot_tensor
        if applied_snapshot.snapshot_tensor is not None
        else allocator.get_kvcache().k_buffer[0][:1]
    )
    fallback_copy = copy_host_backup_candidate_for_smoke(
        plan=fallback_plan,
        source_tensor=fallback_source,
    )

    applied_snapshot_log = applied_snapshot.to_log_dict()
    applied_copy_log = applied_copy.to_log_dict()
    fallback_snapshot_log = fallback_snapshot.to_log_dict()
    fallback_copy_log = fallback_copy.to_log_dict()
    _assert_applied_snapshot(applied_snapshot_log, applied_copy_log)
    _assert_fallback_noop(fallback_snapshot_log, fallback_copy_log)

    print("relaykv_kv_pool_snapshot_smoke: ok")
    print(
        "relaykv_kv_pool_snapshot_applied="
        + json.dumps(
            _merge_snapshot_and_copy_logs(applied_snapshot_log, applied_copy_log),
            sort_keys=True,
        )
    )
    print(
        "relaykv_kv_pool_snapshot_fallback="
        + json.dumps(
            _merge_snapshot_and_copy_logs(fallback_snapshot_log, fallback_copy_log),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
