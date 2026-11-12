"""A finished request's prompt token ids are freed after the next request is
dispatched, or after an idle delay, not between its final output and the end of
its response stream. A stream closed early holds nothing."""

import asyncio
import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.deferred_input_release import (  # noqa: E402
    DeferredInputRelease,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402
from sglang.srt.runtime_context import get_context  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Ids(list):
    """Token ids that a weak reference can watch."""


class _NullAsyncCM:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _obj(rid, input_ids):
    return SimpleNamespace(
        rid=rid,
        is_single=True,
        return_prompt_token_ids=False,
        input_ids=input_ids,
        normalize_batch_and_arguments=lambda: None,
    )


def _make_manager(case, observe):
    """A TokenizerManager reduced to the single-request path of generate_request.
    ``observe()`` runs when each request starts waiting for its outputs, after
    its dispatch. The stand-ins keep no reference to the request."""
    override = get_context().override_server_args()
    override.install()
    case.addCleanup(override.restore)
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.init_deferred_input_release()
    tm.rid_to_state = {}

    def init_req_state(obj, request=None):
        tm.rid_to_state[obj.rid] = SimpleNamespace(waiting_at_pause_gate=True)

    async def no_op(*args, **kwargs):
        return None

    async def tokenize(obj):
        return SimpleNamespace(input_ids=None)

    async def wait_one_response(obj, request=None):
        del obj
        observe()
        yield {"text": "a"}
        yield {"text": "b"}

    tm.auto_create_handle_loop = lambda: None
    tm._set_default_priority = lambda obj: None
    tm._init_req_state = init_req_state
    tm.tokenizer = None
    tm.request_logger = SimpleNamespace(log_received_request=lambda *args: None)
    tm.is_pause = False
    tm.is_pause_cond = asyncio.Condition()
    tm.model_update_lock = SimpleNamespace(reader_lock=_NullAsyncCM())
    tm._validate_and_resolve_lora = no_op
    tm._tokenize_one_request = tokenize
    tm._send_one_request = no_op
    tm._wait_one_response = wait_one_response
    tm._release_req_states_on_failure = Mock()
    return tm


class TestDeferredInputRelease(CustomTestCase):
    def test_idle_timer_frees_without_a_next_request(self):
        async def run():
            holder = DeferredInputRelease(idle_delay_s=0.01)
            ids = _Ids(range(8))
            ref = weakref.ref(ids)
            holder.hold(ids)
            del ids
            gc.collect()
            self.assertIsNotNone(ref())
            await asyncio.sleep(0.05)
            gc.collect()
            self.assertIsNone(ref())
            self.assertEqual(len(holder), 0)

        asyncio.run(run())

    def test_release_cancels_the_timer_and_a_later_hold_rearms_it(self):
        async def run():
            holder = DeferredInputRelease(idle_delay_s=0.01)
            holder.hold(_Ids(range(8)))
            holder.release()
            ids = _Ids(range(8))
            ref = weakref.ref(ids)
            holder.hold(ids)
            del ids
            await asyncio.sleep(0.05)
            gc.collect()
            self.assertIsNone(ref())

        asyncio.run(run())


class TestTokenizerManagerDefersInputFree(CustomTestCase):
    def test_finished_input_is_freed_after_the_next_dispatch(self):
        alive_at_next_dispatch = []
        refs = []

        def observe():
            if refs:
                gc.collect()
                alive_at_next_dispatch.append(refs[0]() is not None)

        tm = _make_manager(self, observe)

        async def run():
            ids = _Ids(range(8))
            ref = weakref.ref(ids)
            outputs = [r async for r in tm.generate_request(_obj("first", ids))]
            self.assertEqual(len(outputs), 2)
            del ids
            gc.collect()
            # The stream ended without freeing the ids.
            self.assertIsNotNone(ref())
            refs.append(ref)
            self.assertEqual(len(tm.deferred_input_release), 1)
            async for _ in tm.generate_request(_obj("second", _Ids(range(4)))):
                pass

        asyncio.run(run())
        self.assertEqual(alive_at_next_dispatch, [False])
        # Only the second request's ids are held now.
        self.assertEqual(len(tm.deferred_input_release), 1)

    def test_stream_closed_early_holds_nothing(self):
        """A client that disconnects mid-stream closes the generator; its input
        is freed with it, as before, and nothing is left held."""
        tm = _make_manager(self, lambda: None)

        async def run():
            ids = _Ids(range(8))
            ref = weakref.ref(ids)
            stream = tm.generate_request(_obj("aborted", ids))
            del ids
            await stream.__anext__()
            await stream.aclose()
            del stream
            await asyncio.sleep(0)
            gc.collect()
            self.assertIsNone(ref())
            self.assertEqual(len(tm.deferred_input_release), 0)

        asyncio.run(run())
        tm._release_req_states_on_failure.assert_called_once()


if __name__ == "__main__":
    unittest.main()
