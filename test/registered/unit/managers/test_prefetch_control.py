import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sglang.srt.managers.communicator import FanOutCommunicator
from sglang.srt.managers.io_struct import PrefetchReqOutput
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize("tokens", [[], [True], [-1], [100], [1.5], [[1]], [1] * 65])
def test_prefetch_rejects_invalid_tokens_before_dispatch(tokens):
    manager = SimpleNamespace(
        model_config=SimpleNamespace(context_len=64, vocab_size=100)
    )
    with pytest.raises(ValueError):
        asyncio.run(TokenizerControlMixin.prefetch(manager, tokens))


def test_prefetch_cancellation_does_not_mix_replies():
    async def run():
        sent = []
        comm = FanOutCommunicator(sent.append, 1)
        manager = SimpleNamespace(
            model_config=SimpleNamespace(context_len=64, vocab_size=100),
            auto_create_handle_loop=lambda: None,
            prefetch_communicator=comm,
        )

        async def wait_for_sends(count):
            while len(sent) < count:
                await asyncio.sleep(0)

        first = asyncio.create_task(TokenizerControlMixin.prefetch(manager, [1, 2]))
        await asyncio.wait_for(wait_for_sends(1), 1)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(TokenizerControlMixin.prefetch(manager, [3, 4]))
        comm.handle_recv(PrefetchReqOutput(success=True, cached_tokens=1))
        await asyncio.wait_for(wait_for_sends(2), 1)
        comm.handle_recv(
            PrefetchReqOutput(success=True, prefetched_tokens=2, cached_tokens=2)
        )
        result = await asyncio.wait_for(second, 1)
        assert result.success and result.cached_tokens == 2
        assert sent[0].rid != sent[1].rid
        assert sent[1].token_ids == [3, 4]

    asyncio.run(run())


@pytest.mark.parametrize("success", [True, False])
def test_prefetch_reports_all_replica_success_and_minimum_counts(success):
    manager = SimpleNamespace(
        model_config=SimpleNamespace(context_len=64, vocab_size=100),
        auto_create_handle_loop=lambda: None,
        prefetch_communicator=AsyncMock(
            return_value=[
                PrefetchReqOutput(success=True, prefetched_tokens=4, cached_tokens=8),
                PrefetchReqOutput(
                    success=success,
                    message="second replica",
                    prefetched_tokens=2,
                    cached_tokens=6,
                ),
            ]
        ),
    )
    result = asyncio.run(TokenizerControlMixin.prefetch(manager, list(range(8))))
    assert result.success is success
    assert (result.prefetched_tokens, result.cached_tokens) == (2, 6)
    assert "second replica" in result.message
