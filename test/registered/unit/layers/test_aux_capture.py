"""Unit tests for srt/layers/aux_capture.

The fused AuxCaptureSink must be numerically identical to the legacy
per-layer append + final torch.cat it replaces, across all three write paths
(append, append_add, static buffer) and consistent with the stash protocol
the prefill CUDA-graph runner drives.
"""

import unittest

import torch

from sglang.srt.layers.aux_capture import AuxCaptureMixin, AuxCaptureSink
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

T, H, K = 7, 5, 3


def _reference_cat(layers):
    """Legacy path: list of [T, H] tensors -> [T, K*H] via torch.cat."""
    return torch.cat(layers, dim=-1)


class TestAuxCaptureSink(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.layers = [torch.randn(T, H) for _ in range(K)]
        self.residuals = [torch.randn(T, H) for _ in range(K)]

    def test_append_matches_cat(self):
        sink = AuxCaptureSink(num_layers=K)
        for layer in self.layers:
            sink.append(layer)
        torch.testing.assert_close(sink.buf, _reference_cat(self.layers))

    def test_append_add_matches_cat(self):
        sink = AuxCaptureSink(num_layers=K)
        for layer, residual in zip(self.layers, self.residuals):
            sink.append_add(layer, residual)
        ref = _reference_cat(
            [layer + residual for layer, residual in zip(self.layers, self.residuals)]
        )
        torch.testing.assert_close(sink.buf, ref)

    def test_append_add_none_residual_falls_back_to_copy(self):
        sink = AuxCaptureSink(num_layers=K)
        for layer in self.layers:
            sink.append_add(layer, None)
        torch.testing.assert_close(sink.buf, _reference_cat(self.layers))

    def test_static_buffer_is_reused_and_sliced(self):
        # Static buffer over-allocates rows; the sink writes into a stable
        # address view sliced to the live token count.
        static = torch.zeros(T + 4, K * H)
        sink = AuxCaptureSink(num_layers=K, static_buf=static)
        for layer in self.layers:
            sink.append(layer)
        self.assertEqual(sink.buf.shape, (T, K * H))
        # buf aliases the head of the static buffer (graph replay refreshes it).
        self.assertEqual(sink.buf.data_ptr(), static.data_ptr())
        torch.testing.assert_close(sink.buf, _reference_cat(self.layers))
        # Untouched tail rows stay zero.
        torch.testing.assert_close(static[T:], torch.zeros(4, K * H))

    def test_static_buffer_replay_refreshes_in_place(self):
        # A second capture pass into the same static buffer overwrites the rows,
        # mirroring a CUDA-graph replay reusing the baked address.
        static = torch.zeros(T, K * H)
        first = [torch.randn(T, H) for _ in range(K)]
        sink1 = AuxCaptureSink(num_layers=K, static_buf=static)
        for layer in first:
            sink1.append(layer)
        second = [torch.randn(T, H) for _ in range(K)]
        sink2 = AuxCaptureSink(num_layers=K, static_buf=static)
        for layer in second:
            sink2.append(layer)
        torch.testing.assert_close(static, _reference_cat(second))

    def test_finalize_returns_fused_buffer(self):
        sink = AuxCaptureSink(num_layers=K)
        for layer in self.layers:
            sink.append(layer)
        torch.testing.assert_close(sink.finalize(), _reference_cat(self.layers))

    def test_finalize_underfilled_raises(self):
        # Partial capture (e.g. capture layers inside an un-threaded TBO
        # segment) must fail loudly instead of returning garbage columns.
        sink = AuxCaptureSink(num_layers=K)
        sink.append(self.layers[0])
        with self.assertRaisesRegex(RuntimeError, "underfilled"):
            sink.finalize()

    def test_finalize_unwritten_raises(self):
        # A sink that no layer ever wrote (e.g. the whole loop ran in a TBO
        # branch) must not silently stash None.
        sink = AuxCaptureSink(num_layers=K)
        with self.assertRaisesRegex(RuntimeError, "underfilled"):
            sink.finalize()

    def test_overfill_raises(self):
        sink = AuxCaptureSink(num_layers=K)
        for layer in self.layers:
            sink.append(layer)
        with self.assertRaisesRegex(RuntimeError, "overfilled"):
            sink.append(self.layers[0])

    def test_width_mismatch_raises(self):
        sink = AuxCaptureSink(num_layers=K)
        sink.append(self.layers[0])
        with self.assertRaisesRegex(RuntimeError, "width mismatch"):
            sink.append(torch.randn(T, H + 1))


class TestAuxCaptureSinkUnderDynamo(CustomTestCase):
    """The sink runs inside torch.compile'd model code on the tc_piecewise
    prefill path, so every write path must trace under Dynamo without graph
    breaks (fullgraph=True). Regression guard: ``torch.add(..., out=column)``
    into the strided column view is rejected by Dynamo ("Attempted to call op
    with non-contiguous `out=` tensor") and broke PCG + DFLASH server boot.
    ``backend="eager"`` exercises exactly the Dynamo tracing layer without
    requiring an inductor toolchain on CPU CI.
    """

    def _run_compiled(self, static_buf=None):
        def fn(layers, residuals):
            sink = AuxCaptureSink(num_layers=K, static_buf=static_buf)
            for layer, residual in zip(layers, residuals):
                sink.append_add(layer, residual)
            return sink.finalize()

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        torch.manual_seed(2)
        layers = [torch.randn(T, H) for _ in range(K)]
        residuals = [torch.randn(T, H) for _ in range(K)]
        out = compiled(layers, residuals)
        ref = torch.cat(
            [layer + residual for layer, residual in zip(layers, residuals)], dim=-1
        )
        torch.testing.assert_close(out, ref)
        return out

    def test_append_add_traces_fullgraph(self):
        self._run_compiled()

    def test_append_add_static_buffer_traces_fullgraph(self):
        static = torch.zeros(T, K * H)
        out = self._run_compiled(static_buf=static)
        # The compiled writes must land in the caller-owned buffer.
        torch.testing.assert_close(static, out)

    def test_stash_overwrites_under_compile(self):
        # The tc_piecewise compile trampoline runs the inner forward twice
        # back-to-back on first compile (compile-triggering call, then the
        # returned-value call) with no pop in between, so a compiled stash
        # over an unconsumed stash must overwrite instead of raising the
        # eager collision error (which aborted PCG + DFLASH server boot).
        class _M(AuxCaptureMixin):
            pass

        m = _M()

        def fn(module, buf):
            module.stash_aux_hidden_states(buf)
            return buf

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        first = torch.randn(T, K * H)
        second = torch.randn(T, K * H)
        compiled(m, first)
        compiled(m, second)  # collides in eager; must overwrite when compiled
        self.assertIs(m.pop_aux_hidden_states(), second)


class TestAuxCaptureMixin(CustomTestCase):
    def _make(self):
        class _M(AuxCaptureMixin):
            pass

        return _M()

    def test_disabled_by_default(self):
        m = self._make()
        self.assertFalse(m.aux_capture_enabled)
        self.assertIsNone(m.make_aux_sink())

    def test_enable_makes_sink(self):
        m = self._make()
        m.enable_aux_capture(K)
        self.assertTrue(m.aux_capture_enabled)
        self.assertEqual(m.num_aux_capture_layers, K)
        sink = m.make_aux_sink()
        self.assertIsInstance(sink, AuxCaptureSink)
        self.assertEqual(sink.num_layers, K)

    def test_static_buffer_routes_into_sink(self):
        m = self._make()
        m.enable_aux_capture(K)
        static = torch.zeros(T, K * H)
        m.set_aux_capture_static_buffer(static)
        sink = m.make_aux_sink()
        sink.append(torch.randn(T, H))
        self.assertEqual(sink.buf.data_ptr(), static.data_ptr())
        # Detaching restores per-forward allocation.
        m.set_aux_capture_static_buffer(None)
        sink2 = m.make_aux_sink()
        sink2.append(torch.randn(T, H))
        self.assertNotEqual(sink2.buf.data_ptr(), static.data_ptr())

    def test_stash_pop_is_one_shot(self):
        m = self._make()
        buf = torch.randn(T, K * H)
        m.stash_aux_hidden_states(buf)
        self.assertIs(m.pop_aux_hidden_states(), buf)
        # Second pop yields None (consumed).
        self.assertIsNone(m.pop_aux_hidden_states())

    def test_two_instances_do_not_share_stash(self):
        a, b = self._make(), self._make()
        buf = torch.randn(T, K * H)
        a.stash_aux_hidden_states(buf)
        self.assertIsNone(b.pop_aux_hidden_states())
        self.assertIs(a.pop_aux_hidden_states(), buf)

    def test_stash_collision_raises(self):
        # A stash over an unconsumed buffer means an interleaved forward or a
        # caller that never pops; it must not silently leak one request's aux
        # into another.
        m = self._make()
        m.stash_aux_hidden_states(torch.randn(T, K * H))
        with self.assertRaisesRegex(RuntimeError, "stash collision"):
            m.stash_aux_hidden_states(torch.randn(T, K * H))

    def test_stash_after_pop_is_allowed(self):
        m = self._make()
        m.stash_aux_hidden_states(torch.randn(T, K * H))
        m.pop_aux_hidden_states()
        second = torch.randn(T, K * H)
        m.stash_aux_hidden_states(second)
        self.assertIs(m.pop_aux_hidden_states(), second)


class TestAuxCaptureDraftSingleLayer(CustomTestCase):
    """EAGLE3-draft path: a single captured hidden (num_layers == 1).

    Migrated EAGLE3 draft inner models stash exactly one aux tensor (the
    pre/post-norm final hidden). The fused buffer must equal that tensor
    unchanged -- no concat wrapping, no extra column -- and the stash/pop
    round-trip must return the same rows the logits processor stores. A
    regression that made num_layers==1 allocate/return a 2-column buffer or
    re-wrap the tensor in a list would silently corrupt the draft input.
    """

    def setUp(self):
        torch.manual_seed(1)
        self.aux = torch.randn(T, H)

    def test_single_layer_buffer_equals_input(self):
        sink = AuxCaptureSink(num_layers=1)
        sink.append(self.aux)
        self.assertEqual(sink.buf.shape, (T, H))
        torch.testing.assert_close(sink.buf, self.aux)

    def test_single_layer_static_buffer_width(self):
        # Prefill body-capture arms a [max_tokens, 1 * H] buffer for the draft
        # worker; the sink writes the single hidden into the live-row view.
        static = torch.zeros(T + 3, 1 * H)
        sink = AuxCaptureSink(num_layers=1, static_buf=static)
        sink.append(self.aux)
        self.assertEqual(sink.buf.shape, (T, H))
        self.assertEqual(sink.buf.data_ptr(), static.data_ptr())
        torch.testing.assert_close(sink.buf, self.aux)
        torch.testing.assert_close(static[T:], torch.zeros(3, H))

    def test_draft_mixin_stash_pop_roundtrip(self):
        class _Draft(AuxCaptureMixin):
            pass

        draft = _Draft()
        draft.enable_aux_capture(1)
        self.assertEqual(draft.num_aux_capture_layers, 1)
        sink = draft.make_aux_sink()
        sink.append(self.aux)
        draft.stash_aux_hidden_states(sink.finalize())
        torch.testing.assert_close(draft.pop_aux_hidden_states(), self.aux)
        # One-shot: consumed after the wrapper pops (pairs with the draft's
        # idle-batch early return that also stashes then returns).
        self.assertIsNone(draft.pop_aux_hidden_states())


if __name__ == "__main__":
    unittest.main()
