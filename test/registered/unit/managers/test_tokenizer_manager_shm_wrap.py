"""Unit tests for SGLANG_SHM_WRAP_OFF_LOOP in TokenizerManager: wrap_shm_features
runs on a worker thread, and a cancelled request releases what it wrapped."""

import asyncio
import os
import threading
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

maybe_stub_sgl_kernel()

from sglang.srt.managers import tokenizer_manager as tm_module  # noqa: E402
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402


class TestWrapShmFeaturesOffLoop(CustomTestCase):
    def test_wraps_on_a_pool_thread(self):
        threads = []

        def wrap(obj):
            threads.append(threading.current_thread().name)
            return obj

        obj = object()
        with patch.object(tm_module, "wrap_shm_features", side_effect=wrap):
            result = asyncio.run(tm_module._wrap_shm_features_off_loop(obj))
        self.assertIs(result, obj)
        self.assertTrue(threads[0].startswith("shm-wrap"), threads)

    def test_cancelled_caller_discards_the_wrapped_features(self):
        started, release = threading.Event(), threading.Event()
        discarded = threading.Event()
        obj = object()

        def wrap(o):
            started.set()
            release.wait(5)
            return o

        async def drive():
            task = asyncio.create_task(tm_module._wrap_shm_features_off_loop(obj))
            while not started.is_set():
                await asyncio.sleep(0.01)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            release.set()
            for _ in range(500):
                if discarded.is_set():
                    break
                await asyncio.sleep(0.01)

        discard = MagicMock(side_effect=lambda o: discarded.set())
        with (
            patch.object(tm_module, "wrap_shm_features", side_effect=wrap),
            patch.object(tm_module, "discard_shm_features", discard),
        ):
            asyncio.run(drive())
        discard.assert_called_once_with(obj)


class TestSendOneRequestShmWrap(CustomTestCase):
    def _run(self, env_value):
        tm = TokenizerManager.__new__(TokenizerManager)
        tm.cuda_vmm_feature_transport = MagicMock()
        tm.cuda_vmm_feature_transport.prepare_for_dispatch_async = AsyncMock(
            return_value=[]
        )
        tm.positional_embeds_transport = MagicMock()
        tm.positional_embeds_transport.prepare_for_dispatch.return_value = []
        tm._dispatch_to_scheduler = MagicMock()
        tm._mark_state_dispatched = MagicMock()
        tm.encoder_dispatch_ready = {}
        tokenized = MagicMock()
        off_loop = AsyncMock(side_effect=lambda obj: obj)
        with (
            patch.dict(os.environ, {"SGLANG_SHM_WRAP_OFF_LOOP": env_value}),
            patch.object(tm_module, "_wrap_shm_features_off_loop", off_loop),
            patch.object(
                tm_module, "wrap_shm_features", side_effect=lambda obj: obj
            ) as wrap,
        ):
            asyncio.run(tm._send_one_request(tokenized))
        tm._dispatch_to_scheduler.assert_called_once_with(tokenized)
        return off_loop, wrap

    def test_default_wraps_inline(self):
        off_loop, wrap = self._run("0")
        off_loop.assert_not_called()
        wrap.assert_called_once()

    def test_flag_wraps_off_loop(self):
        off_loop, wrap = self._run("1")
        off_loop.assert_awaited_once()
        wrap.assert_not_called()


if __name__ == "__main__":
    unittest.main()
