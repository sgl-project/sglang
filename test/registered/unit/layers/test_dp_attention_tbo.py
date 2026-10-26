"""CPU unit tests for the two-batch-overlap DP gather and its inverse return."""

import threading
import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers import dp_attention
from sglang.srt.layers.dp_attention import dp_shard_sizes
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _patch_streams(stack):
    stack.enter_context(patch.object(torch.cuda, "current_stream", return_value=Mock()))
    stack.enter_context(
        patch.object(torch.cuda, "stream", side_effect=lambda stream: nullcontext())
    )
    stack.enter_context(
        patch.object(dp_attention, "get_dp_tbo_comm_stream", return_value=Mock())
    )
    stack.enter_context(patch.object(dp_attention, "_tbo_event", return_value=Mock()))


class TestDpShardSizes(unittest.TestCase):
    def test_attention_tp_shards_preserve_dp_layout(self):
        dp_sizes, shard_sizes = dp_shard_sizes(
            [8, 8, 8, 8, 12, 12, 12, 12], attn_tp_size=4
        )

        self.assertEqual(dp_sizes, [8, 12])
        self.assertEqual(shard_sizes, [2, 2, 2, 2, 3, 3, 3, 3])
        self.assertEqual(sum(dp_sizes), sum(shard_sizes))

    def test_rejects_inconsistent_attention_tp_replicas(self):
        with self.assertRaisesRegex(ValueError, "differ within attention TP"):
            dp_shard_sizes([8, 8, 4, 8], attn_tp_size=4)

    def test_rejects_unshardable_token_count(self):
        with self.assertRaisesRegex(ValueError, "not divisible"):
            dp_shard_sizes([6, 6, 6, 6], attn_tp_size=4)


class TestDpGatherAsync(unittest.TestCase):
    def test_tp1_uses_dp_gather_and_tp4_gathers_one_shard(self):
        local = torch.arange(32).reshape(8, 4)
        global_tokens = torch.empty(16, 4)
        batch = SimpleNamespace()
        for attn_tp_size in (1, 4):
            with self.subTest(attn_tp_size=attn_tp_size), ExitStack() as stack:
                group = Mock(rank_in_group=0)
                parallel = SimpleNamespace(
                    attn_tp_size=attn_tp_size, attn_tp_rank=0, tp_group=group
                )
                sizes = [8 // attn_tp_size] * (2 * attn_tp_size)
                _patch_streams(stack)
                stack.enter_context(
                    patch.object(dp_attention, "get_parallel", return_value=parallel)
                )
                dp_gather = stack.enter_context(
                    patch.object(dp_attention, "dp_gather_replicate")
                )
                _, sent = dp_attention.dp_gather_async(
                    global_tokens, local, batch, sizes
                )
                if attn_tp_size == 1:
                    dp_gather.assert_called_once_with(global_tokens, local, batch)
                    group.all_gatherv.assert_not_called()
                    self.assertIs(sent, local)
                else:
                    dp_gather.assert_not_called()
                    group.all_gatherv.assert_called_once_with(
                        sent, sizes=[2] * 8, output=global_tokens
                    )
                    torch.testing.assert_close(sent, local[:2])


class _Collective:
    """Ranks deposit their inputs; once all have, each reads the whole set."""

    def __init__(self, world_size):
        self.slots = [None] * world_size
        self.barrier = threading.Barrier(world_size, timeout=30)

    def exchange(self, rank, value):
        self.slots[rank] = value.clone()
        self.barrier.wait()
        values = list(self.slots)
        self.barrier.wait()
        return values


class _TpGroup:
    def __init__(self, collective, rank):
        self.collective, self.rank_in_group = collective, rank

    def all_gatherv(self, input_, *, sizes, output):
        values = self.collective.exchange(self.rank_in_group, input_)
        assert [v.shape[0] for v in values] == sizes
        output.copy_(torch.cat(values))

    def reduce_scatterv(self, input_, *, output, sizes):
        total = torch.stack(self.collective.exchange(self.rank_in_group, input_))
        start = sum(sizes[: self.rank_in_group])
        output.copy_(total.sum(0)[start : start + sizes[self.rank_in_group]])


class TestDpGatherRoundTrip(unittest.TestCase):
    def test_gather_then_return_restores_each_ranks_rows(self):
        """TP 4 with attention DP 2: each DP rank's rows, replicated over its two
        attention-TP ranks, are gathered by shard, summed across the TP group and
        returned whole to both replicas."""
        tp_size, attn_tp_size, hidden = 4, 2, 3
        dp_rows = [4, 6]
        rank_sizes = [rows for rows in dp_rows for _ in range(attn_tp_size)]
        dp_sizes, shard_sizes = dp_shard_sizes(rank_sizes, attn_tp_size)
        dp_locals = [
            torch.arange(rows * hidden, dtype=torch.float32).view(rows, hidden)
            + 100 * dp
            for dp, rows in enumerate(dp_rows)
        ]
        tp_collective = _Collective(tp_size)
        attn_tp_collectives = [_Collective(attn_tp_size) for _ in dp_rows]
        local_state = threading.local()
        results, errors = [None] * tp_size, []

        def all_gather_single(output, input_, *, group):
            output.copy_(torch.cat(group.exchange(local_state.attn_tp_rank, input_)))

        def run(rank):
            dp_rank, attn_tp_rank = divmod(rank, attn_tp_size)
            local_state.attn_tp_rank = attn_tp_rank
            local_state.parallel = SimpleNamespace(
                attn_tp_size=attn_tp_size,
                attn_tp_rank=attn_tp_rank,
                tp_group=_TpGroup(tp_collective, rank),
                attn_tp_group=SimpleNamespace(
                    device_group=attn_tp_collectives[dp_rank]
                ),
            )
            try:
                local = dp_locals[dp_rank].clone()
                global_tokens = torch.empty(sum(dp_sizes), hidden)
                dp_attention.dp_gather_async(
                    global_tokens, local, SimpleNamespace(), shard_sizes
                )
                # A different partial on each TP rank: the return must sum all.
                partial = global_tokens * (rank + 1)
                output = torch.empty_like(local)
                dp_attention.dp_reduce_scatterv_async(output, partial, shard_sizes)
                results[rank] = output
            except Exception as error:  # surfaced on the main thread
                errors.append(error)
                tp_collective.barrier.abort()

        with ExitStack() as stack:
            _patch_streams(stack)
            stack.enter_context(
                patch.object(
                    dp_attention,
                    "get_parallel",
                    side_effect=lambda: local_state.parallel,
                )
            )
            stack.enter_context(
                patch.object(
                    dp_attention,
                    "get_tbo_persistent_buffer",
                    side_effect=lambda key, rows, h, dtype, device: torch.empty(
                        rows, h, dtype=dtype
                    ),
                )
            )
            stack.enter_context(
                patch.object(
                    dp_attention, "all_gather_single", side_effect=all_gather_single
                )
            )
            threads = [threading.Thread(target=run, args=(r,)) for r in range(tp_size)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        if errors:
            raise errors[0]
        owed_sum = sum(range(1, tp_size + 1))
        for rank, output in enumerate(results):
            torch.testing.assert_close(
                output, dp_locals[rank // attn_tp_size] * owed_sum
            )


if __name__ == "__main__":
    unittest.main()
