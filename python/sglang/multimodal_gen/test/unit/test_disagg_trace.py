"""Unit tests for OTel trace-context propagation across the diffusion disagg
JSON hop (encoder -> denoiser, denoiser -> decoder).

These exercise the serialization contract only (no GPUs, no server, no OTLP
collector required):

 - ``extract_transfer_fields`` emits a JSON-safe ``_trace_state`` (W3C carrier)
   when tracing is enabled, and omits it when tracing is disabled. It never
   serializes the live ``TraceReqContext`` object itself.
 - The ``_trace_state`` payload round-trips through ``codec.pack_tensors``
   (the same ``json.dumps`` path the RDMA metadata frame uses).
 - ``TraceReqContext.__setstate__`` reconstructs a live, ``is_copy=True``
   context whose ``root_span_context`` is an OTel ``Context`` object.
 - ``_build_disagg_req`` pops ``_trace_state`` and installs a rebuilt
   ``TraceReqContext`` on the receiver-side Req.
"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin import (
    SchedulerDisaggMixin,
    extract_transfer_fields,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.codec import pack_tensors
from sglang.multimodal_gen.runtime.pipelines_core import Req
from sglang.srt.observability import trace as srt_trace
from sglang.srt.observability.trace import TraceNullContext, TraceReqContext

try:
    from opentelemetry import propagate as otel_propagate
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider

    _OTEL_AVAILABLE = srt_trace.opentelemetry_imported
except ImportError:
    _OTEL_AVAILABLE = False


_MISSING = object()
_OTEL_BOOTSTRAPPED = False


def _enable_minimal_otel() -> None:
    """Bootstrap just enough OTel state for TraceReqContext to produce real
    spans. Idempotent — the TracerProvider can only be set once per process."""
    global _OTEL_BOOTSTRAPPED
    if not _OTEL_BOOTSTRAPPED:
        # Own the provider for this hermetic unit test, then restore the
        # process-global OTel state in tearDown.
        otel_trace._TRACER_PROVIDER_SET_ONCE._done = False
        otel_trace._TRACER_PROVIDER = None
        otel_trace.set_tracer_provider(TracerProvider())
        _OTEL_BOOTSTRAPPED = True
    srt_trace.opentelemetry_initialized = True
    srt_trace.tracer = otel_trace.get_tracer("test-diffusion-disagg")
    srt_trace.trace_set_thread_info("TestThread")


def _traceparent_from(ctx) -> str | None:
    """Re-inject a W3C carrier from an OTel Context and return the traceparent.

    Used to assert that a carrier round-trip preserves trace_id/span_id, which
    is the actual correctness property (OTel's Context is a dict subclass, so
    ``isinstance(ctx, dict)`` isn't useful).
    """
    carrier: dict = {}
    otel_propagate.inject(carrier, ctx)
    return carrier.get("traceparent")


def _roundtrip_scalar_fields(scalar_fields: dict) -> dict:
    """Run the actual RDMA metadata codec path: pack -> json bytes -> decode."""
    metadata_bytes, _ = pack_tensors({}, scalar_fields)
    decoded = json.loads(metadata_bytes.decode("utf-8"))
    return decoded["scalar_fields"]


class TestDisaggTracePropagation(unittest.TestCase):
    def setUp(self):
        global _OTEL_BOOTSTRAPPED

        self._orig_otel_initialized = srt_trace.opentelemetry_initialized
        self._orig_tracer = srt_trace.tracer
        self._orig_trace_level = srt_trace.global_trace_level
        self._orig_threads_info = srt_trace.threads_info.copy()
        self._orig_otel_bootstrapped = _OTEL_BOOTSTRAPPED
        if _OTEL_AVAILABLE:
            self._orig_otel_provider = getattr(otel_trace, "_TRACER_PROVIDER", _MISSING)
            self._orig_otel_provider_set_once_done = getattr(
                otel_trace._TRACER_PROVIDER_SET_ONCE, "_done", _MISSING
            )
        else:
            self._orig_otel_provider = _MISSING
            self._orig_otel_provider_set_once_done = _MISSING

        _OTEL_BOOTSTRAPPED = False
        srt_trace.set_global_trace_level(3)

        # Isolate the disagg JSON trace-carrier contract from ServerArgs
        # defaults; those defaults are covered by server-args/trace tests.
        self._server_args_patcher = patch.object(
            srt_trace,
            "get_global_server_args",
            return_value=SimpleNamespace(trace_modules="request"),
        )
        self._server_args_patcher.start()

    def tearDown(self):
        global _OTEL_BOOTSTRAPPED

        self._server_args_patcher.stop()
        srt_trace.opentelemetry_initialized = self._orig_otel_initialized
        srt_trace.tracer = self._orig_tracer
        srt_trace.global_trace_level = self._orig_trace_level
        srt_trace.threads_info.clear()
        srt_trace.threads_info.update(self._orig_threads_info)
        _OTEL_BOOTSTRAPPED = self._orig_otel_bootstrapped
        if self._orig_otel_provider is not _MISSING:
            otel_trace._TRACER_PROVIDER = self._orig_otel_provider
        if self._orig_otel_provider_set_once_done is not _MISSING:
            otel_trace._TRACER_PROVIDER_SET_ONCE._done = (
                self._orig_otel_provider_set_once_done
            )

    def test_transfer_keeps_seed_needed_to_rebuild_generator(self):
        req = Req(request_id="test-seed", prompt="x")
        req.generator = torch.Generator(device="cpu").manual_seed(req.seed)

        _, scalar_fields = extract_transfer_fields(req)

        self.assertEqual(scalar_fields["seed"], 42)

        rebuilt = SchedulerDisaggMixin._build_disagg_req(None, dict(scalar_fields), {})
        self.assertIsInstance(rebuilt.generator, torch.Generator)
        self.assertEqual(rebuilt.seed, 42)

        expected = torch.rand(
            (), generator=torch.Generator(device="cpu").manual_seed(42)
        )
        actual = torch.rand((), generator=rebuilt.generator)
        self.assertEqual(actual.item(), expected.item())

    def test_build_disagg_req_rebuilds_generator_list(self):
        scalar_fields = {
            "request_id": "test-seed-list",
            "prompt": "x",
            "num_outputs_per_prompt": 2,
            "seed": [11, 12],
        }

        rebuilt = SchedulerDisaggMixin._build_disagg_req(None, dict(scalar_fields), {})

        self.assertEqual(rebuilt.seed, [11, 12])
        self.assertEqual(len(rebuilt.generator), 2)
        for seed, generator in zip(rebuilt.seed, rebuilt.generator):
            expected = torch.rand(
                (), generator=torch.Generator(device="cpu").manual_seed(seed)
            )
            actual = torch.rand((), generator=generator)
            self.assertEqual(actual.item(), expected.item())

    def test_tracing_disabled_omits_trace_state(self):
        """With a default TraceNullContext Req, no _trace_state is emitted and
        the JSON codec does not encounter any live OTel objects."""
        req = Req(request_id="test-off", prompt="x")
        self.assertIsInstance(req.trace_ctx, TraceNullContext)

        _, scalar_fields = extract_transfer_fields(req)
        self.assertNotIn("_trace_state", scalar_fields)
        # trace_ctx must never ride the JSON scalar path.
        self.assertNotIn("trace_ctx", scalar_fields)
        # json.dumps must succeed (this is the path that pre-fix crashed).
        _roundtrip_scalar_fields(scalar_fields)

    @unittest.skipUnless(_OTEL_AVAILABLE, "opentelemetry SDK not installed")
    def test_tracing_enabled_state_roundtrip(self):
        """Sender emits a W3C-carrier _trace_state, it round-trips through
        json encode/decode, and __setstate__ reconstructs a live is_copy=True
        TraceReqContext with an OTel Context (not the raw dict)."""
        _enable_minimal_otel()

        ctx = TraceReqContext(rid="test-on", role="server", module_name="request")
        ctx.trace_req_start()
        self.assertTrue(ctx.tracing_enable)
        self.assertFalse(ctx.is_copy)

        req = Req(request_id="test-on", prompt="x")
        req.trace_ctx = ctx

        _, scalar_fields = extract_transfer_fields(req)
        self.assertNotIn("trace_ctx", scalar_fields)
        self.assertIn("_trace_state", scalar_fields)
        state = scalar_fields["_trace_state"]
        self.assertTrue(state.get("tracing_enable"))
        # W3C carrier must be present so downstream roles can nest spans.
        self.assertIn("traceparent", state.get("root_span_context", {}))

        decoded = _roundtrip_scalar_fields(scalar_fields)
        self.assertEqual(decoded["_trace_state"], state)

        rebuilt = object.__new__(TraceReqContext)
        rebuilt.__setstate__(decoded["_trace_state"])
        self.assertTrue(rebuilt.tracing_enable)
        self.assertTrue(rebuilt.is_copy)
        # The sender's traceparent must survive into the rebuilt Context so
        # downstream role spans nest under the original trace_id.
        self.assertEqual(
            _traceparent_from(rebuilt.root_span_context),
            state["root_span_context"]["traceparent"],
        )

    @unittest.skipUnless(_OTEL_AVAILABLE, "opentelemetry SDK not installed")
    def test_build_disagg_req_installs_rebuilt_ctx(self):
        """_build_disagg_req pops _trace_state from scalar_fields and installs
        a live TraceReqContext on the rebuilt Req; the key does not leak onto
        the Req as a stray attribute."""
        _enable_minimal_otel()

        ctx = TraceReqContext(rid="test-brq", role="server", module_name="request")
        ctx.trace_req_start()

        req = Req(request_id="test-brq", prompt="x")
        req.trace_ctx = ctx
        _, scalar_fields = extract_transfer_fields(req)
        self.assertIn("_trace_state", scalar_fields)

        # _build_disagg_req is an instance method but its body does not touch
        # ``self``; call via __func__ to avoid needing a real Scheduler.
        rebuilt = SchedulerDisaggMixin._build_disagg_req(None, dict(scalar_fields), {})

        self.assertIsInstance(rebuilt.trace_ctx, TraceReqContext)
        self.assertTrue(rebuilt.trace_ctx.tracing_enable)
        self.assertTrue(rebuilt.trace_ctx.is_copy)
        self.assertFalse(hasattr(rebuilt, "_trace_state"))

    def test_build_disagg_req_falls_back_when_tracing_off(self):
        """If the sender's context is a TraceNullContext, the receiver's Req
        keeps its default TraceNullContext (no _trace_state to apply)."""
        req = Req(request_id="test-brq-off", prompt="x")
        self.assertIsInstance(req.trace_ctx, TraceNullContext)

        _, scalar_fields = extract_transfer_fields(req)
        self.assertNotIn("_trace_state", scalar_fields)

        rebuilt = SchedulerDisaggMixin._build_disagg_req(None, dict(scalar_fields), {})
        self.assertIsInstance(rebuilt.trace_ctx, TraceNullContext)


if __name__ == "__main__":
    unittest.main()
