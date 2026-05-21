from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.jit_kernel.kv_canary.consts import RealKvHashMode
from sglang.jit_kernel.kv_canary.verify import CANARY_SLOT_BYTES, CanaryLaunchTag
from sglang.srt.kv_canary import endpoint as endpoint_module
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.capacities import CanaryLaunchCapacities
from sglang.srt.kv_canary.config import CanaryConfig, CanaryMode
from sglang.srt.kv_canary.runner import kernel_launch as kernel_launch_module
from sglang.srt.kv_canary.runner.canary_runner import CanaryRunner
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.fixtures import DEFAULT_DEVICE
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=1, stage="extra-a", runner_config="1-gpu-large")


def make_group(
    *,
    device: torch.device,
    has_v: bool = True,
    kind: PoolKind = PoolKind.FULL,
) -> CanaryBufferGroup:
    return CanaryBufferGroup(
        kind=kind,
        k_head=torch.zeros(4, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device),
        k_tail=torch.zeros(4, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device),
        v_head=(
            torch.zeros(4, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device)
            if has_v
            else None
        ),
        v_tail=(
            torch.zeros(4, CANARY_SLOT_BYTES, dtype=torch.uint8, device=device)
            if has_v
            else None
        ),
        real_kv_sources_k=(),
        real_kv_sources_v=(),
        swa_index_lut=None,
    )


def make_pool(
    device: torch.device,
    max_reqs: int = 4,
    max_seq: int = 8,
) -> SimpleNamespace:
    req_to_token = torch.zeros(max_reqs, max_seq, dtype=torch.int32, device=device)
    return SimpleNamespace(req_to_token=req_to_token, size=max_reqs)


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


class FakeDecodeForwardMode:
    def is_extend(self) -> bool:
        return False

    def is_mixed(self) -> bool:
        return False

    def is_decode_or_idle(self) -> bool:
        return True

    def is_target_verify(self) -> bool:
        return False

    def is_draft_extend_v2(self) -> bool:
        return False

    def is_extend_or_draft_extend_or_mixed(self) -> bool:
        return False


def make_forward_batch(
    device: torch.device,
    bs: int = 2,
    seq_lens_list: tuple[int, ...] = (3, 4),
) -> SimpleNamespace:
    seq_lens = list(seq_lens_list[:bs])
    return SimpleNamespace(
        forward_mode=FakeDecodeForwardMode(),
        spec_info=None,
        batch_size=bs,
        req_pool_indices=torch.tensor([1, 2][:bs], dtype=torch.int64, device=device),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        seq_lens_sum=int(sum(seq_lens)),
        extend_prefix_lens=None,
        extend_seq_lens=None,
        extend_prefix_lens_cpu=None,
        input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
        positions=torch.zeros(bs, dtype=torch.int32, device=device),
        out_cache_loc=torch.zeros(bs, dtype=torch.int32, device=device),
        num_token_non_padded_cpu=None,
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
        group = make_group(device=device)
    if req_pool is None:
        req_pool = make_pool(device)
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
