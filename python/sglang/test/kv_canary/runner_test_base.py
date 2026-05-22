from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.jit_kernel.kv_canary.consts import RealKvHashMode
from sglang.jit_kernel.kv_canary.verify import CanaryLaunchTag
from sglang.srt.kv_canary import endpoint as endpoint_module
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup
from sglang.srt.kv_canary.capacities import CanaryLaunchCapacities
from sglang.srt.kv_canary.config import CanaryConfig, CanaryMode
from sglang.srt.kv_canary.runner import kernel_launch as kernel_launch_module
from sglang.srt.kv_canary.runner.canary_runner import CanaryRunner
from sglang.test.kv_canary.fixtures import (
    DEFAULT_DEVICE,
    make_buffer_group,
    make_req_to_token_pool,
)
from sglang.test.test_utils import CustomTestCase


def make_config(
    *,
    mode: CanaryMode = CanaryMode.RAISE,
    ring_capacity: int = 1024,
    sweep_interval: int = 0,
    real_kv_hash_mode: RealKvHashMode = RealKvHashMode.OFF,
    input_check_mode: bool = False,
    stats_print_every_n_steps: int = 100,
) -> CanaryConfig:
    return CanaryConfig(
        mode=mode,
        ring_capacity=ring_capacity,
        sweep_interval=sweep_interval,
        real_kv_hash_mode=real_kv_hash_mode,
        input_check_mode=input_check_mode,
        stats_print_every_n_steps=stats_print_every_n_steps,
    )


class RecordingEndpoint:
    def __init__(self, *, kernel_kind: CanaryLaunchTag) -> None:
        self.kernel_kind = kernel_kind
        self.calls: list[dict[str, object]] = []

    def launch_per_forward(self, **kwargs: object) -> None:
        self.calls.append(kwargs)


def make_runner(
    *,
    device: torch.device,
    config: CanaryConfig | None = None,
    group: CanaryBufferGroup | None = None,
    req_pool: SimpleNamespace | None = None,
    per_forward_verify_capacity: int = 16,
) -> CanaryRunner:
    if config is None:
        config = make_config()
    if group is None:
        group = make_buffer_group(device=device)
    if req_pool is None:
        req_pool = make_req_to_token_pool(device=device, max_reqs=4, max_seq_len=8)
    return CanaryRunner(
        config=config,
        buffer_groups=(group,),
        device=device,
        req_to_token_pool=req_pool,
        launch_capacities=CanaryLaunchCapacities(
            per_forward_verify_capacity=per_forward_verify_capacity,
            per_forward_write_req_capacity=2,
            per_forward_write_entry_capacity=8,
        ),
    )


class CanaryRunnerTestCase(CustomTestCase):
    def setUp(self) -> None:
        self.device = DEFAULT_DEVICE
        # Stub plan/verify/write kernels so CPU runs don't need CUDA JIT.
        self._patchers = [
            patch.object(
                kernel_launch_module,
                "launch_canary_plan_kernels",
                lambda **kwargs: None,
            ),
            patch.object(
                endpoint_module, "launch_canary_verify_kernel", lambda **kwargs: None
            ),
            patch.object(
                endpoint_module, "launch_canary_write_kernel", lambda **kwargs: None
            ),
        ]
        for patcher in self._patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
