from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import signal
import traceback
from typing import Any

# flashinfer.jit.env reads FLASHINFER_WORKSPACE_BASE at import time. Set it
# before any SGLang imports so smoke-only JIT logs go to a writable /tmp path.
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp/relaykv_flashinfer_cache")

import torch

from sglang.srt.relaykv import RelayKVConfig, make_shadow_plan
from sglang.srt.relaykv.memory import (
    copy_host_backup_candidate_for_smoke,
    snapshot_mha_kv_pool_readonly_for_smoke,
)


class _ObservedMHATokenToKVPool:
    """Duck-typed split K/V pool matching MHATokenToKVPool's buffer layout."""

    def __init__(self, *, device: str) -> None:
        self.start_layer = 0
        self.dtype = torch.float16
        self.device = device
        # Actual MHATokenToKVPool allocates [size + page_size, heads, head_dim].
        self.k_buffer = [
            torch.arange(9 * 2 * 8, dtype=self.dtype, device=device).reshape(9, 2, 8),
        ]
        self.v_buffer = [
            (torch.arange(9 * 2 * 8, dtype=self.dtype, device=device) + 2048).reshape(
                9, 2, 8
            ),
        ]


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


def _error_reason(exc: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()


def _configure_flashinfer_cache() -> dict[str, object]:
    # flashinfer.jit.env uses FLASHINFER_WORKSPACE_BASE at import time and writes
    # flashinfer_jit.log below it. Keep smoke-only import artifacts in /tmp.
    workspace_base = os.environ.setdefault(
        "FLASHINFER_WORKSPACE_BASE", "/tmp/relaykv_flashinfer_cache"
    )
    cache_path = Path(workspace_base) / ".cache" / "flashinfer"
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
        probe_path = cache_path / ".relaykv_write_probe"
        probe_path.write_text("ok", encoding="utf-8")
        probe_path.unlink()
        writable = True
        error = ""
    except OSError as exc:
        writable = False
        error = _error_reason(exc)
    return {
        "flashinfer_cache_env": "FLASHINFER_WORKSPACE_BASE",
        "flashinfer_cache_path": str(cache_path),
        "flashinfer_cache_path_writable": writable,
        "flashinfer_cache_probe_error": error,
    }


def _try_actual_mha_pool(device: str) -> tuple[Any, dict[str, object]]:
    flashinfer_cache_status = _configure_flashinfer_cache()
    status: dict[str, object] = {
        "kv_pool_type": None,
        "kv_pool_import_ok": False,
        "kv_pool_import_error_type": "",
        "kv_pool_import_error": "",
        "kv_pool_instantiate_ok": False,
        "kv_pool_instantiate_error_type": "",
        "kv_pool_instantiate_error": "",
        "fallback_used": False,
        **flashinfer_cache_status,
    }
    try:
        module = importlib.import_module("sglang.srt.mem_cache.memory_pool")
        pool_cls = getattr(module, "MHATokenToKVPool")
        status["kv_pool_import_ok"] = True
    except BaseException as exc:
        status["kv_pool_import_error_type"] = type(exc).__name__
        status["kv_pool_import_error"] = _error_reason(exc)
        status["kv_pool_type"] = "_ObservedMHATokenToKVPool"
        status["fallback_used"] = True
        return _ObservedMHATokenToKVPool(device=device), status

    try:
        pool = pool_cls(
            size=8,
            page_size=1,
            dtype=torch.float16,
            head_num=2,
            head_dim=8,
            layer_num=1,
            device=device,
            enable_memory_saver=False,
            enable_alt_stream=False,
        )
        status["kv_pool_type"] = type(pool).__name__
        status["kv_pool_instantiate_ok"] = True
        return pool, status
    except BaseException as exc:
        status["kv_pool_type"] = "_ObservedMHATokenToKVPool"
        status["kv_pool_instantiate_error_type"] = type(exc).__name__
        status["kv_pool_instantiate_error"] = _error_reason(exc)
        status["fallback_used"] = True
        return _ObservedMHATokenToKVPool(device=device), status


def _merge_logs(
    base: dict[str, object],
    snapshot_log: dict[str, object],
    copy_log: dict[str, object],
) -> dict[str, object]:
    payload = dict(base)
    payload.update(
        {
            "runtime_policy_state": snapshot_log["runtime_policy_state"],
            "kv_pool_type": snapshot_log["kv_pool_type"],
            "observed_layout": snapshot_log["observed_layout"],
            "has_k_buffer": snapshot_log["has_k_buffer"],
            "has_v_buffer": snapshot_log["has_v_buffer"],
            "k_shape": snapshot_log["k_shape"],
            "v_shape": snapshot_log["v_shape"],
            "k_dtype": snapshot_log["k_dtype"],
            "v_dtype": snapshot_log["v_dtype"],
            "k_device": snapshot_log["k_device"],
            "v_device": snapshot_log["v_device"],
            "layer_idx": snapshot_log["layer_idx"],
            "token_indices": snapshot_log["token_indices"],
            "snapshot_created": snapshot_log["snapshot_created"],
            "snapshot_shape": snapshot_log["snapshot_shape"],
            "backup_shape": copy_log["backup_shape"],
            "host_backup_copy_executed": copy_log["host_backup_copy_executed"],
            "copy_equal": copy_log["copy_equal"],
            "copy_numel": copy_log["copy_numel"],
            "copy_nbytes": copy_log["copy_nbytes"],
            "source_mutated": snapshot_log["source_mutated"],
            "kv_cache_mutation": snapshot_log["kv_cache_mutation"],
            "attention_override": snapshot_log["attention_override"],
            "scheduler_policy_noop": snapshot_log["scheduler_policy_noop"],
            "fallback_candidate_noop_guard": snapshot_log[
                "fallback_candidate_noop_guard"
            ],
            "host_backup_copy_skipped_reason": copy_log[
                "host_backup_copy_skipped_reason"
            ],
        }
    )
    return payload


def _assert_applied(log: dict[str, object]) -> None:
    if log["runtime_policy_state"] != "applied_candidate":
        raise AssertionError(log)
    if log["observed_layout"] is None:
        raise AssertionError(log)
    if log["has_k_buffer"] is not True:
        raise AssertionError(log)
    if log["has_v_buffer"] is not True:
        raise AssertionError(log)
    if log["snapshot_created"] is not True:
        raise AssertionError(log)
    if log["snapshot_shape"] != [4, 2, 2, 8]:
        raise AssertionError(log)
    if log["backup_shape"] != [4, 2, 2, 8]:
        raise AssertionError(log)
    if log["copy_equal"] is not True:
        raise AssertionError(log)
    if log["source_mutated"] is not False:
        raise AssertionError(log)
    if log["kv_cache_mutation"] is not False:
        raise AssertionError(log)
    if log["attention_override"] is not False:
        raise AssertionError(log)
    if log["scheduler_policy_noop"] is not True:
        raise AssertionError(log)


def _assert_fallback(log: dict[str, object]) -> None:
    if log["runtime_policy_state"] != "fallback_candidate":
        raise AssertionError(log)
    if log["snapshot_created"] is not False:
        raise AssertionError(log)
    if log["host_backup_copy_executed"] is not False:
        raise AssertionError(log)
    if log["fallback_candidate_noop_guard"] is not True:
        raise AssertionError(log)
    if log["copy_equal"] is not False:
        raise AssertionError(log)
    if log["source_mutated"] is not False:
        raise AssertionError(log)


def main() -> None:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer_idx = 0
    token_indices = [2, 3, 4, 5]
    pool, pool_status = _try_actual_mha_pool(device)

    applied_plan = make_shadow_plan(
        2535,
        _relaykv_config(),
        kv_bytes_per_token=28672,
        request_id="actual-kv-pool-observation-applied",
    )
    fallback_plan = make_shadow_plan(
        8192,
        _relaykv_config(),
        kv_bytes_per_token=28672,
        request_id="actual-kv-pool-observation-fallback",
    )

    applied_snapshot = snapshot_mha_kv_pool_readonly_for_smoke(
        kv_pool=pool,
        token_indices=token_indices,
        layer_idx=layer_idx,
        runtime_policy_state=applied_plan.runtime_policy_state,
    )
    if applied_snapshot.snapshot_tensor is None:
        raise AssertionError(applied_snapshot)
    applied_copy = copy_host_backup_candidate_for_smoke(
        plan=applied_plan,
        source_tensor=applied_snapshot.snapshot_tensor,
    )

    fallback_snapshot = snapshot_mha_kv_pool_readonly_for_smoke(
        kv_pool=pool,
        token_indices=token_indices,
        layer_idx=layer_idx,
        runtime_policy_state=fallback_plan.runtime_policy_state,
    )
    fallback_copy = copy_host_backup_candidate_for_smoke(
        plan=fallback_plan,
        source_tensor=applied_snapshot.snapshot_tensor,
    )

    applied_log = _merge_logs(
        pool_status,
        applied_snapshot.to_log_dict(),
        applied_copy.to_log_dict(),
    )
    fallback_log = _merge_logs(
        pool_status,
        fallback_snapshot.to_log_dict(),
        fallback_copy.to_log_dict(),
    )
    _assert_applied(applied_log)
    _assert_fallback(fallback_log)

    print("relaykv_actual_kv_pool_observation_smoke: ok")
    print(
        "relaykv_actual_kv_pool_observation_applied="
        + json.dumps(applied_log, sort_keys=True)
    )
    print(
        "relaykv_actual_kv_pool_observation_fallback="
        + json.dumps(fallback_log, sort_keys=True)
    )


if __name__ == "__main__":
    main()
