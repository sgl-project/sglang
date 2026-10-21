import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.prefill_delayer import PrefillDelayer, _State
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

TOPOLOGIES = {
    "tp": dict(tp_size=4),
    "attn_cp": dict(tp_size=4, attn_cp_size=2),
    "dp_attention": dict(tp_size=4, attn_dp_size=2),
    "dp_attention_attn_cp": dict(tp_size=8, attn_dp_size=2, attn_cp_size=2),
}


def _all_gather_over_tp(output, local, group=None):
    """All-gather over the TP group, as ``all_gather_into_tensor`` checks it."""
    world_size = get_parallel().tp_size
    if output.numel() != world_size * local.numel():
        raise RuntimeError(
            f"output holds {output.numel()} elements, not {world_size} x "
            f"{local.numel()}"
        )
    rows = output.view(world_size, -1)
    for rank in range(world_size):
        rows[rank] = local + 100 * rank


class TestPrefillDelayerGather(CustomTestCase):
    def test_gather_returns_the_first_rank_of_each_dp_group(self):
        for name, topology in TOPOLOGIES.items():
            with (
                self.subTest(name),
                published_topology(**topology),
                patch(
                    "sglang.srt.managers.prefill_delayer.all_gather_single",
                    side_effect=_all_gather_over_tp,
                ),
                get_parallel().override(
                    tp_group=SimpleNamespace(cpu_group=None, device_group=None)
                ),
            ):
                delayer = PrefillDelayer(
                    max_delay_passes=4, token_usage_low_watermark=None
                )
                info = delayer._gather_info(
                    local_prefillable=True,
                    local_token_watermark_force_allow=False,
                    running_batch=3,
                    max_prefill_bs=4,
                    waiting_queue_len=5,
                    queue_timeout_expired=True,
                )
                parallel = get_parallel()
                dp_groups = parallel.num_dp_ranks if parallel.attn_dp_enabled else 1
                ranks_per_dp_group = parallel.tp_size // dp_groups
                local = torch.tensor([1, 0, 3, 4, 5, 1])
                expected = torch.stack(
                    [local + 100 * g * ranks_per_dp_group for g in range(dp_groups)]
                )
                self.assertTrue(torch.equal(info, expected), f"{info} != {expected}")


def _negotiate_with_skewed_clocks(rank, world_size, init_file):
    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        for name in ("tp", "dp_attention"):
            topology = TOPOLOGIES[name]
            with (
                published_topology(
                    ranks=dict(world_rank=rank),
                    prefill_delayer_queue_min_ratio=0.5,
                    prefill_delayer_max_delay_ms=1000,
                    **topology,
                ),
                get_parallel().override(
                    tp_group=SimpleNamespace(
                        cpu_group=torch.distributed.group.WORLD, device_group=None
                    )
                ),
            ):
                delayer = PrefillDelayer(
                    max_delay_passes=100, token_usage_low_watermark=None
                )
                delayer.skip_first_delayer = False
                for tp0_expired in (True, False):
                    local_expired = tp0_expired == (rank % 2 == 0)
                    age = 60.0 if local_expired else 0.0
                    out = delayer._negotiate_should_allow_prefill_pure(
                        prev_state=_State(
                            delayed_count=1, start_time=time.perf_counter() - age
                        ),
                        local_prefillable=True,
                        token_usage=0.8,
                        running_batch=8,
                        max_prefill_bs=4,
                        max_running_requests=128,
                        waiting_queue_len=1,
                    )
                    assert out.output_allow == tp0_expired, (
                        f"{name} rank={rank} local_expired={local_expired} "
                        f"tp0_expired={tp0_expired} allow={out.output_allow}"
                    )
    finally:
        torch.distributed.destroy_process_group()


class TestPrefillDelayerQueueTimeoutMultiRank(CustomTestCase):
    def test_ranks_with_skewed_clocks_follow_tp0(self):
        world_size = 4
        with tempfile.TemporaryDirectory() as tmp:
            torch.multiprocessing.spawn(
                _negotiate_with_skewed_clocks,
                args=(world_size, os.path.join(tmp, "init")),
                nprocs=world_size,
            )


if __name__ == "__main__":
    unittest.main()
