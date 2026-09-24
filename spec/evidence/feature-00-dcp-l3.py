"""Reproduce the pre-implementation DCP/L3 guard and host-page baseline.

Run from the repository root in the SGLang development environment:
    python spec/evidence/feature-00-dcp-l3.py
This is a baseline probe, not an end-to-end L3 support test.
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups import hicache_hook
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost


def main():
    report = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
    }
    config = SimpleNamespace(
        dcp_size=2,
        enable_hierarchical_cache=True,
        hicache_storage_backend="file",
        speculative_algorithm=None,
        enable_lmcache=False,
        enable_hisparse=False,
    )
    with patch.object(hicache_hook, "resolving_view", return_value=config), patch.object(
        hicache_hook, "use_mla_backend", return_value=True
    ):
        try:
            hicache_hook.resolve_hicache_dcp_compatibility(config)
        except NotImplementedError as error:
            report["startup_guard"] = str(error)
        else:
            raise AssertionError("Baseline changed: the DCP/L3 guard did not reject.")

    device_pool = SimpleNamespace(
        size=256,
        host_capacity_tokens=None,
        store_dtype=torch.bfloat16,
        kv_lora_rank=8,
        qk_rope_head_dim=4,
        layer_num=2,
        start_layer=0,
        end_layer=1,
        device="cpu",
        layers_to_capture=None,
        layer_shard_enabled=False,
    )
    pool = MLATokenToKVPoolHost(
        device_pool,
        host_to_device_ratio=2.0,
        host_size=0,
        page_size=128,
        layout="page_first",
        pin_memory=False,
        device="cpu",
        dcp_size=2,
        dcp_rank=1,
    )
    dummy = pool.get_dummy_flat_data_page()
    report["host_pool"] = {
        "dcp_size": pool.dcp_size,
        "logical_page_tokens": pool.logical_page_size,
        "local_page_rows": pool.page_size,
        "layers": pool.layer_num,
        "kv_width": pool.kv_cache_dim,
        "local_page_elements": dummy.numel(),
        "local_page_bytes": dummy.numel() * dummy.element_size(),
    }
    try:
        pool.get_data_page(0)
    except AssertionError as error:
        report["page_accessor_guard"] = str(error)
    else:
        raise AssertionError("Baseline changed: L3 page access did not reject.")

    command = [
        sys.executable,
        "test/registered/unit/mem_cache/test_hicache_dcp_host_pool.py",
        "-v",
    ]
    tests = subprocess.run(command, text=True, capture_output=True, timeout=120)
    report["existing_host_pool_tests"] = {
        "command": command,
        "exit_code": tests.returncode,
        "output": tests.stdout + tests.stderr,
    }
    print(json.dumps(report, indent=2))
    return tests.returncode


if __name__ == "__main__":
    raise SystemExit(main())
