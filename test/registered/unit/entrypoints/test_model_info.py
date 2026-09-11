"""Endpoint-level tests for `/model_info`.

`/model_info` reports the identity the server currently answers under, plus
the handful of serving settings a client needs before it can send a request.
Everything here is read from manager-owned state, which is what makes the
endpoint answerable while the scheduler is still warming.

Current coverage:

* `TestModelInfoEnableHttp2Contract` — `enable_http2` as a wire contract
  with sgl-router, which reads it to choose a forwarding client per worker.
"""

import asyncio
import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints import http_server
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _call_model_info_with(server_args: ServerArgs) -> dict:
    """Invoke `http_server.model_info()` against a stub global state.

    Bypasses the FastAPI HTTP layer: the handler is an `async def` reading
    module-level `_global_state`, so a `SimpleNamespace` stub plus a published
    context (for `get_serving()`) is enough to exercise it without booting a
    model server.
    """
    tokenizer_manager = SimpleNamespace(
        server_args=server_args,
        model_path=server_args.model_path,
        served_model_name=server_args.served_model_name,
        is_generation=True,
        config_value=lambda name: None,
        model_config=SimpleNamespace(
            is_image_understandable_model=False,
            is_audio_understandable_model=False,
            hf_config=SimpleNamespace(),
        ),
    )
    prior_state = http_server.get_global_state()
    http_server.set_global_state(SimpleNamespace(tokenizer_manager=tokenizer_manager))
    try:
        # Inside the try: a `publish` that raises must still restore the module
        # global, or the stub leaks into every later test in this process.
        publish(server_args, role="tokenizer")
        return asyncio.run(http_server.model_info())
    finally:
        http_server.set_global_state(prior_state)
        reset_context()


class TestModelInfoEnableHttp2Contract(CustomTestCase):
    """`enable_http2` is a wire contract with sgl-router, not just a flag.

    The router reads this key to decide whether to forward to a worker over
    cleartext h2c (`experimental/sgl-router/src/workers/manager.rs`,
    `resolve_protocol`). Its parser treats the key as optional, because engines
    predating the flag do not send it — so if the field is renamed, dropped, or
    stops being a bool, the router reads `None`, silently leaves every worker on
    HTTP/1.1, and nothing on either side fails.

    It is pinned on `/model_info` specifically. `/server_info` also carries it
    as part of the launch record, but that handler awaits a scheduler
    round-trip, so on a warming engine it can fail while this one answers —
    and a protocol the router reads late is a protocol it never applies.
    """

    def test_enable_http2_is_present_and_boolean_by_default(self):
        info = _call_model_info_with(ServerArgs(model_path="dummy"))

        self.assertIn(
            "enable_http2",
            info,
            "sgl-router reads `enable_http2` from /model_info to enable h2c "
            "forwarding; renaming or dropping it silently disables the feature",
        )
        self.assertIsInstance(
            info["enable_http2"],
            bool,
            "the router parses `enable_http2` as Option<bool>; a non-bool "
            "deserialises to None and reads as 'no h2c'",
        )
        self.assertFalse(info["enable_http2"], "default must be off")

    def test_enable_http2_reports_the_launched_value(self):
        """Presence is not enough — the value has to track the launch flag.

        A field pinned to its default would pass the case above while telling
        the router "no h2c" on every engine that enabled it.
        """
        info = _call_model_info_with(ServerArgs(model_path="dummy", enable_http2=True))

        self.assertTrue(
            info["enable_http2"],
            "an engine launched with --enable-http2 must advertise it, or the "
            "router will never upgrade that worker to h2c",
        )


if __name__ == "__main__":
    unittest.main()
