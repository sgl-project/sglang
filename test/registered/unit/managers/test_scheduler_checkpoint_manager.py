import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.managers.io_struct import CheckpointState
from sglang.srt.managers.scheduler_components.checkpoint_manager import (
    SchedulerCheckpointManager,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _server(*drained, tp_size=1, release_error=None):
    """A tokenizer over real scheduler checkpoint managers, `tp_size` per DP
    rank. `drained` says whether each DP rank goes idle, and the last TP rank
    of each fails its KV cache release with `release_error`. Pause, continue
    and the KV cache release are recorded in `calls`."""
    calls = []
    engine_paused = [False]

    async def pause_generation(obj):
        calls.append("pause")
        # Abort only holds requests in the tokenizer. Retract pauses the scheduler.
        engine_paused[0] = obj.mode == "retract"

    async def continue_generation(obj):
        calls.append("continue")
        engine_paused[0] = False

    def release(fails):
        def release_memory_occupation(req):
            if fails and release_error is not None:
                raise release_error
            calls.append("release")

        return release_memory_occupation

    # TP ranks of a DP rank handle each request in turn, the first one last,
    # and see every output gathered so far.
    gathered = []

    def all_gather_object(out):
        gathered.append(out)
        return list(gathered)

    ranks = [
        SchedulerCheckpointManager(
            enable_memory_saver=True,
            pp_size=1,
            offload_tags=set(),
            is_engine_paused=lambda: engine_paused[0],
            is_fully_idle=MagicMock(return_value=idle),
            release_memory_occupation=release(fails=tp_rank == tp_size - 1),
            resume_memory_occupation=lambda req: calls.append("resume_kv"),
            all_gather_object=all_gather_object,
        )
        for idle in drained
        for tp_rank in range(tp_size)
    ]

    async def checkpoint_communicator(obj):
        # Only the first TP rank of each DP rank replies.
        replies = []
        for dp_rank in range(len(drained)):
            gathered.clear()
            tp_ranks = ranks[dp_rank * tp_size : (dp_rank + 1) * tp_size]
            replies.append([rank.handle(obj) for rank in reversed(tp_ranks)][-1])
        return replies

    tokenizer = SimpleNamespace(
        auto_create_handle_loop=lambda: None,
        checkpoint_lock=asyncio.Lock(),
        is_pause=False,
        checkpoint_communicator=checkpoint_communicator,
        pause_generation=pause_generation,
        continue_generation=continue_generation,
    )
    tokenizer._call_checkpoint = lambda req_type: (
        TokenizerControlMixin._call_checkpoint(tokenizer, req_type)
    )
    return tokenizer, ranks, calls


def _prepare(tokenizer, mode="abort"):
    return asyncio.run(TokenizerControlMixin.checkpoint_prepare(tokenizer, mode))


def _restore(tokenizer):
    return asyncio.run(TokenizerControlMixin.checkpoint_restore(tokenizer))


class TestCheckpointManager(unittest.TestCase):
    def test_prepare_releases_kv_cache_only_after_pause_and_drain(self):
        """A request scheduled after the KV cache is released runs without one,
        so generation is paused first and the release waits for the drain.
        Orchestrators redeliver prepare, so a repeat is a no-op."""
        tokenizer, ranks, calls = _server(True, False)

        out = _prepare(tokenizer)
        self.assertTrue(out.success)
        self.assertEqual(out.state, CheckpointState.PREPARING)
        self.assertEqual(calls, ["pause", "release"])

        ranks[1]._is_fully_idle.return_value = True
        ranks[1].check_pending()
        out = _prepare(tokenizer)
        self.assertEqual(out.state, CheckpointState.PREPARED)
        self.assertEqual(calls, ["pause", "release", "release"])

    def test_undrainable_retract_is_refused_on_every_dp_rank(self):
        """A retract-paused scheduler stops advancing grammar, KV transfer and
        HiCache work, so a busy rank never drains. The refusal must come before
        any rank releases its KV cache, and the pause must be undone."""
        tokenizer, ranks, calls = _server(False, True)

        out = _prepare(tokenizer, mode="retract")
        self.assertFalse(out.success)
        self.assertIn("retract", out.message)
        self.assertEqual(calls, ["pause", "continue"])
        self.assertEqual([r.state for r in ranks], [CheckpointState.RUNNING] * 2)

    def test_restore_continues_only_the_pause_of_its_cycle(self):
        """Orchestrators redeliver restore. A repeat must not lift a pause a user
        started after the cycle ended."""
        tokenizer, ranks, calls = _server(True, True)
        _prepare(tokenizer)
        calls.clear()

        self.assertEqual(_restore(tokenizer).state, CheckpointState.RUNNING)
        self.assertEqual(calls, ["resume_kv", "resume_kv", "continue"])
        self.assertEqual(_restore(tokenizer).state, CheckpointState.RUNNING)
        self.assertEqual(calls, ["resume_kv", "resume_kv", "continue"])

    def test_failed_release_latches(self):
        """Restoring a half-released server would serve without a KV cache, so a
        failure stops it for good and generation stays paused."""
        tokenizer, ranks, calls = _server(True, release_error=RuntimeError("oom"))

        out = _prepare(tokenizer)
        self.assertEqual(out.state, CheckpointState.FAILED)
        self.assertIn("oom", out.message)
        out = _restore(tokenizer)
        self.assertFalse(out.success)
        self.assertEqual(calls, ["pause"])

    def test_failure_on_a_tp_rank_that_does_not_reply_is_reported(self):
        """Only the first TP rank of a DP rank replies to the tokenizer. If
        another one fails to release its KV cache, the server must not report
        prepared, or it would be captured with that KV cache."""
        tokenizer, ranks, calls = _server(
            True, tp_size=2, release_error=RuntimeError("oom")
        )

        out = _prepare(tokenizer)
        self.assertEqual(
            [r.state for r in ranks], [CheckpointState.PREPARED, CheckpointState.FAILED]
        )
        self.assertEqual(out.state, CheckpointState.FAILED)
        self.assertIn("oom", out.message)


if __name__ == "__main__":
    unittest.main()
