"""Tests for the host-pool transfer I/O log (``SGLANG_DEBUG_HOST_POOL_IO``).

The log exists because a CUDA fault raised by a host<->device KV transfer is
asynchronous: the copy returns, and the sticky error only surfaces at a later
synchronization, so the traceback names an unrelated frame. The tests here pin
down the properties that make the record usable for attribution:

* the per-page copy size is derived the same way the staged write-back kernel
  derives it, so ``batch_threshold`` matches ``kLargeCopyThresholdBytes``;
* a real transfer emits exactly one record, naming the allocator that owns the
  host pool (``alloc_with_host_register`` vs ``alloc_with_pin_memory``);
* nothing is logged unless the flag is on;
* a raising transfer is still recorded, at ERROR level.
"""

import logging
import types

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.pool_host.common import ALLOC_MEMORY_FUNCS
from sglang.srt.mem_cache.pool_host.io_log import (
    BATCH_COPY_THRESHOLD_BYTES,
    describe_transfer,
)
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or is_npu()
    or is_xpu()
    or not (is_cuda() or is_hip()),
    reason="HiCache host-pool tests require CUDA/ROCm.",
)

LOG_NAME = "sglang.srt.mem_cache.pool_host.io_log"
PAGE_SIZE = 32
# token_stride_size = head_num * head_dim * itemsize = 2 * 64 * 2 = 256 bytes
TOKEN_STRIDE_SIZE = 256


def _records(caplog, level=logging.INFO):
    return [r for r in caplog.records if r.name == LOG_NAME and r.levelno >= level]


def _fake_pool(layer_num):
    """Metadata-only stand-in: ``describe_transfer`` reads pool attributes only."""
    return types.SimpleNamespace(
        page_size=PAGE_SIZE,
        layer_num=layer_num,
        token_stride_size=TOKEN_STRIDE_SIZE,
        layout="page_first",
        can_use_jit=True,
        can_use_write_back_jit=True,
        device_pool=types.SimpleNamespace(device="cuda"),
    )


def _build_pools(layer_num):
    device_pool = MHATokenToKVPool(
        size=PAGE_SIZE * 4,
        page_size=PAGE_SIZE,
        head_num=2,
        head_dim=64,
        dtype=torch.bfloat16,
        layer_num=layer_num,
        device="cuda",
        enable_memory_saver=False,
    )
    host_pool = MHATokenToKVPoolHost(
        host_to_device_ratio=2.0,
        host_size=0,
        page_size=PAGE_SIZE,
        pin_memory=True,
        device="cpu",
        allocator_type="default",
        device_pool=device_pool,
        layout="page_first",
    )
    return device_pool, host_pool


def _indices(pages, device):
    return torch.cat(
        [
            torch.arange(
                p * PAGE_SIZE, (p + 1) * PAGE_SIZE, dtype=torch.int64, device=device
            )
            for p in pages
        ]
    )


@pytest.fixture(scope="module")
def pools():
    # One pool pair per module: re-registering the same host range for every test
    # trips cudaErrorHostMemoryAlreadyRegistered, whose error path reports the
    # wrong API and poisons the allocation that follows.
    return _build_pools(15)


def test_per_page_bytes_matches_kernel_threshold(caplog):
    """At the geometry we care about, the derived size equals the kernel constant."""
    at_threshold = PAGE_SIZE * 16 * TOKEN_STRIDE_SIZE
    assert at_threshold == BATCH_COPY_THRESHOLD_BYTES == 128 * 1024

    bound = types.SimpleNamespace(arguments={})
    high = describe_transfer(_fake_pool(16), "backup_from_device_all_layer", bound, 0.0)
    low = describe_transfer(_fake_pool(15), "backup_from_device_all_layer", bound, 0.0)
    assert "per_page_bytes=131072 batch_threshold=True" in high
    assert "per_page_bytes=122880 batch_threshold=False" in low


def test_real_transfer_is_logged_once(caplog, pools):
    caplog.set_level(logging.INFO, logger=LOG_NAME)
    device_pool, host_pool = pools
    assert ALLOC_MEMORY_FUNCS["cuda"].__name__ == "alloc_with_host_register"

    with envs.SGLANG_DEBUG_HOST_POOL_IO.override(True):
        host_pool.backup_from_device_all_layer(
            device_pool, _indices([0], "cpu"), _indices([1], "cuda"), "kernel"
        )
        torch.cuda.synchronize()

    records = _records(caplog)
    assert len(records) == 1
    message = records[0].getMessage()
    assert "[host_pool_io] backup_from_device_all_layer direction=D2H" in message
    assert "per_page_bytes=122880 batch_threshold=False" in message
    assert "host_alloc=alloc_with_host_register" in message
    assert (
        "indices[device_indices=(32,):int64@cuda,host_indices=(32,):int64@cpu]"
        in message
    )


def test_nothing_is_logged_when_disabled(caplog, pools):
    caplog.set_level(logging.INFO, logger=LOG_NAME)
    device_pool, host_pool = pools

    with envs.SGLANG_DEBUG_HOST_POOL_IO.override(False):
        host_pool.backup_from_device_all_layer(
            device_pool, _indices([0], "cpu"), _indices([1], "cuda"), "kernel"
        )
        torch.cuda.synchronize()

    assert _records(caplog) == []


def test_failing_transfer_is_logged_at_error_level(caplog, pools):
    caplog.set_level(logging.INFO, logger=LOG_NAME)
    device_pool, host_pool = pools

    with envs.SGLANG_DEBUG_HOST_POOL_IO.override(True):
        with pytest.raises(ValueError):
            host_pool.backup_from_device_all_layer(
                device_pool,
                _indices([0], "cpu"),
                _indices([1], "cuda"),
                "unsupported-backend",
            )

    errors = _records(caplog, level=logging.ERROR)
    assert len(errors) == 1
    assert errors[0].getMessage().endswith("raised")
