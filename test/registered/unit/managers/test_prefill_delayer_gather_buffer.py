"""PrefillDelayer's all-gather buffer must match its gather group.

With context parallelism (e.g. TP8 + --enable-prefill-cp) ``attn_tp_size`` is 1 while the
delayer's gather group spans every rank, so a buffer shaped
``(dp_size_dim, attn_tp_size, 5)`` is world_size times too small and
``all_gather_into_tensor`` raises on the first negotiation.
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

WORLD_SIZE = 4


def _schedule():
    return SimpleNamespace(
        prefill_delayer_queue_min_ratio=None,
        prefill_delayer_max_delay_ms=None,
        prefill_max_requests=None,
        disable_overlap_schedule=False,
    )


def _worker(rank, init_file, dp_size, enable_dp_attention, attn_tp_size):
    from sglang.srt.managers import prefill_delayer as pd

    torch.distributed.init_process_group(
        backend="gloo",
        init_method=Path(init_file).as_uri(),
        rank=rank,
        world_size=WORLD_SIZE,
    )
    try:
        parallel = SimpleNamespace(
            dp_size=dp_size,
            enable_dp_attention=enable_dp_attention,
            attn_tp_size=attn_tp_size,
        )
        with (
            mock.patch.object(pd, "get_parallel", return_value=parallel),
            mock.patch.object(pd, "get_schedule", return_value=_schedule()),
        ):
            delayer = pd.PrefillDelayer(
                cpu_group=torch.distributed.group.WORLD,
                max_delay_passes=30,
                token_usage_low_watermark=None,
            )
        # Each rank reports a distinct running_batch so the gathered view is checkable.
        info = delayer._gather_info(
            local_prefillable=True,
            local_token_watermark_force_allow=False,
            running_batch=rank + 1,
        )
        dp_size_dim = dp_size if enable_dp_attention else 1
        ranks_per_dp = WORLD_SIZE // dp_size_dim
        assert tuple(info.shape) == (dp_size_dim, 5), tuple(info.shape)
        expected_first_ranks = [g * ranks_per_dp + 1 for g in range(dp_size_dim)]
        assert info[:, 2].tolist() == expected_first_ranks, info[:, 2].tolist()
    finally:
        torch.distributed.destroy_process_group()


class TestPrefillDelayerGatherBuffer(CustomTestCase):
    def _run(self, dp_size, enable_dp_attention, attn_tp_size):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                _worker,
                args=(
                    str(Path(directory) / "gloo-init"),
                    dp_size,
                    enable_dp_attention,
                    attn_tp_size,
                ),
                nprocs=WORLD_SIZE,
                join=True,
            )

    def test_context_parallel_attn_tp_size_one(self):
        # CP: attention TP size collapses to 1, gather group spans all ranks.
        self._run(dp_size=1, enable_dp_attention=False, attn_tp_size=1)

    def test_plain_tensor_parallel_unchanged(self):
        self._run(dp_size=1, enable_dp_attention=False, attn_tp_size=WORLD_SIZE)

    def test_dp_attention_unchanged(self):
        self._run(dp_size=2, enable_dp_attention=True, attn_tp_size=WORLD_SIZE // 2)


if __name__ == "__main__":
    unittest.main()
