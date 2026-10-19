import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.managers.io_struct import CheckpointState
from sglang.srt.managers.scheduler_components.checkpoint_lifecycle import (
    SchedulerCheckpointLifecycle,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _server(*drained, release_error=None):
    """A tokenizer over one real scheduler lifecycle per DP rank. `drained`
    says whether each rank goes idle; pause, continue and the KV cache
    release are recorded in `calls`."""
    calls = []
    engine_paused = [False]

    async def pause_generation(obj):
        calls.append("pause")
        # Abort only holds requests in the tokenizer; retract pauses the scheduler.
        engine_paused[0] = obj.mode == "retract"

    async def continue_generation(obj):
        calls.append("continue")
        engine_paused[0] = False

    def release(req):
        if release_error is not None:
            raise release_error
        calls.append("release")

    ranks = [
        SchedulerCheckpointLifecycle(
            enable_memory_saver=True,
            offload_tags=set(),
            is_engine_paused=lambda: engine_paused[0],
            is_fully_idle=MagicMock(return_value=idle),
            release_memory_occupation=release,
            resume_memory_occupation=lambda req: calls.append("resume_kv"),
        )
        for idle in drained
    ]

    async def checkpoint_communicator(obj):
        return [rank.handle(obj) for rank in ranks]

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


def _resume(tokenizer):
    return asyncio.run(TokenizerControlMixin.checkpoint_resume(tokenizer))


class TestCheckpointLifecycle(unittest.TestCase):
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

    def test_resume_continues_only_the_pause_of_its_cycle(self):
        """Orchestrators redeliver resume; a repeat must not lift a pause a user
        started after the cycle ended."""
        tokenizer, ranks, calls = _server(True, True)
        _prepare(tokenizer)
        calls.clear()

        self.assertEqual(_resume(tokenizer).state, CheckpointState.RUNNING)
        self.assertEqual(calls, ["resume_kv", "resume_kv", "continue"])
        self.assertEqual(_resume(tokenizer).state, CheckpointState.RUNNING)
        self.assertEqual(calls, ["resume_kv", "resume_kv", "continue"])

    def test_failed_release_latches(self):
        """Resuming a half-released server would serve without a KV cache, so a
        failure stops it for good and generation stays paused."""
        tokenizer, ranks, calls = _server(True, release_error=RuntimeError("oom"))

        out = _prepare(tokenizer)
        self.assertEqual(out.state, CheckpointState.FAILED)
        self.assertIn("oom", out.message)
        out = _resume(tokenizer)
        self.assertFalse(out.success)
        self.assertEqual(calls, ["pause"])


if __name__ == "__main__":
    unittest.main()
