import datetime
import json
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.distributed.utils import all_gather_single
from sglang.srt.managers.prefill_delayer import (
    PrefillDelayer,
    PrefillDelayerSinglePassExecutor,
    _State,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=39, suite="base-a-test-cpu")

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


# (enable_lora_overlap_loading, lora_drain_wait_threshold) per case; None: no LoRA.
_LORA_CASES = {"overlap_loading": (True, 0.0), "drain": (False, 5.0), "no_lora": None}


def _negotiate_with_diverging_readiness(rank, world_size, init_file, out_dir):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=20),
    )
    traces = {}
    try:
        for name, lora in _LORA_CASES.items():
            gathers = []

            def _counting_gather(*args, **kwargs):
                gathers.append(1)
                return all_gather_single(*args, **kwargs)

            with (
                published_topology(ranks=dict(world_rank=rank), tp_size=world_size),
                get_parallel().override(
                    tp_group=SimpleNamespace(
                        cpu_group=torch.distributed.group.WORLD, device_group=None
                    )
                ),
                patch(
                    "sglang.srt.managers.prefill_delayer.all_gather_single",
                    side_effect=_counting_gather,
                ),
                patch(
                    "sglang.srt.managers.prefill_delayer.get_lora",
                    return_value=SimpleNamespace(
                        enable_lora=lora is not None,
                        enable_lora_overlap_loading=lora is not None and lora[0],
                        lora_drain_wait_threshold=lora[1] if lora else 0.0,
                    ),
                ),
            ):
                delayer = PrefillDelayer(
                    max_delay_passes=4, token_usage_low_watermark=None
                )
                delayer.skip_first_delayer = False
                # With LoRA, rank 1 skips the request on the first pass (its adapter
                # copy is still in flight); without LoRA both ranks see the same.
                ready = [rank == 0 or lora is None, True, True]
                trace = []
                for local_prefillable in ready:
                    num_gathers = len(gathers)
                    out = delayer._negotiate_should_allow_prefill(
                        local_prefillable=local_prefillable,
                        token_usage=0.9,
                        running_batch=8,
                        max_prefill_bs=4,
                        max_running_requests=8,
                        waiting_queue_len=1,
                    )
                    trace.append(
                        [
                            len(gathers) > num_gathers,
                            out.output_allow,
                            delayer._curr_state is None,
                        ]
                    )
                traces[name] = trace
                traces[f"{name}:holds"] = PrefillDelayerSinglePassExecutor(
                    delayer, token_usage=0.9
                ).holds_requests_during_prefill
    finally:
        torch.distributed.destroy_process_group()
    with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as f:
        json.dump(traces, f)


class TestPrefillDelayerRankLocalAdmission(CustomTestCase):
    def test_ranks_with_diverging_lora_readiness_take_the_same_path(self):
        """Every rank must agree on whether the next pass gathers. A LoRA adapter
        ready on one TP rank only lets that rank alone reach the delayer; a verdict
        from its own row would start a wait there alone, and the next pass would
        all-gather on that rank only and hang."""
        world_size = 2
        with tempfile.TemporaryDirectory() as tmp:
            torch.multiprocessing.spawn(
                _negotiate_with_diverging_readiness,
                args=(world_size, os.path.join(tmp, "init"), tmp),
                nprocs=world_size,
            )
            traces = []
            for rank in range(world_size):
                with open(os.path.join(tmp, f"rank{rank}.json")) as f:
                    traces.append(json.load(f))
        for name in _LORA_CASES:
            with self.subTest(name):
                self.assertEqual(traces[0][name], traces[1][name])
        # Rank 0's row starts the wait on both ranks, which then gather until it ends.
        expected_lora = [
            [True, False, False],
            [True, False, False],
            [True, False, False],
        ]
        self.assertEqual(traces[0]["overlap_loading"], expected_lora)
        self.assertEqual(traces[0]["drain"], expected_lora)
        # Identical rows: the first verdict needs no gather.
        self.assertEqual(traces[0]["no_lora"][0], [False, False, False])
        # Rank-local admission also keeps the delay verdict binding once a chunk
        # prefills, so ranks cannot build different batches on a delay pass.
        self.assertTrue(traces[0]["overlap_loading:holds"])
        self.assertTrue(traces[0]["drain:holds"])
        self.assertFalse(traces[0]["no_lora:holds"])


if __name__ == "__main__":
    unittest.main()
