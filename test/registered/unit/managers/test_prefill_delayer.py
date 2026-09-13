import sys
import time
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.prefill_delayer import PrefillDelayer, _State
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _run_queue_timeout_rank_skew(rank, world_size, rendezvous_path):
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        delayer = PrefillDelayer(
            dp_size=world_size,
            attn_tp_size=1,
            cpu_group=torch.distributed.group.WORLD,
            server_args=SimpleNamespace(
                enable_dp_attention=True,
                disable_overlap_schedule=False,
                prefill_delayer_queue_min_ratio=0.5,
                prefill_delayer_max_delay_ms=50,
            ),
            max_delay_passes=100,
            token_usage_low_watermark=None,
        )
        negotiate_kwargs = dict(
            local_prefillable=True,
            token_usage=0.9,
            running_batch=100,
            max_prefill_bs=80,
            max_running_requests=1024,
            waiting_queue_len=10,
        )

        initial = delayer._negotiate_should_allow_prefill(**negotiate_kwargs)
        assert initial.output_allow
        assert initial.output_reason == "no_wait"

        state_age_seconds = 0.06 if rank == 0 else 0.01
        delayer._curr_state = _State(
            delayed_count=1,
            start_time=time.perf_counter() - state_age_seconds,
        )
        result = delayer._negotiate_should_allow_prefill(**negotiate_kwargs)

        assert result.output_allow
        assert result.output_reason == "wait_success"
        assert result.wait_forward_passes == 1
    finally:
        torch.distributed.destroy_process_group()


def test_queue_timeout_rank_skew_stays_lockstep(tmp_path):
    world_size = 4
    torch.multiprocessing.spawn(
        _run_queue_timeout_rank_skew,
        args=(world_size, str(tmp_path / "distributed_init")),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
