"""Unit tests for forward_context — PoC M0.6 (per-forward flags). CPU-only.

Covers the frozen per-forward flags tree, the dataclasses.replace round-trip set
(riding the forward_context() scope, auto-reset on exit), and the default that
matches the legacy is_extend_in_batch global (False).
"""

import dataclasses
import unittest

from sglang.srt.model_executor.forward_context import (
    AttnForwardFlags,
    ForwardContext,
    ForwardFlags,
    forward_context,
    get_forward_context,
    get_forward_flags,
    has_forward_context,
    set_attn_forward_flag,
    set_forward_context,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_FAKE_BACKEND = object()  # ForwardContext only stores the ref; never called here


def _ctx() -> ForwardContext:
    return ForwardContext(attn_backend=_FAKE_BACKEND)


class TestForwardFlags(unittest.TestCase):
    def tearDown(self):
        # ensure no context leaks across tests
        set_forward_context(None)

    def test_default_is_false(self):
        # parity with the legacy dp_attention is_extend_in_batch default
        self.assertFalse(_ctx().flags.attn.is_extend_in_batch)

    def test_structs_are_frozen(self):
        ctx = _ctx()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ctx.flags.attn.is_extend_in_batch = True
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ctx.flags.attn = AttnForwardFlags()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ctx.flags = ForwardFlags()

    def test_set_round_trip_inside_scope(self):
        with forward_context(_ctx()):
            set_attn_forward_flag(is_extend_in_batch=True)
            self.assertTrue(get_forward_context().flags.attn.is_extend_in_batch)
            self.assertTrue(get_forward_flags().attn.is_extend_in_batch)
            set_attn_forward_flag(is_extend_in_batch=False)
            self.assertFalse(get_forward_context().flags.attn.is_extend_in_batch)

    def test_scope_exit_resets(self):
        with forward_context(_ctx()):
            set_attn_forward_flag(is_extend_in_batch=True)
        self.assertFalse(has_forward_context())  # reset is automatic

    def test_replace_does_not_mutate_parent_ctx(self):
        # set_forward_context returns the prev ctx; the helper's replace builds a
        # NEW frozen instance, so the original object is untouched.
        original = _ctx()
        with forward_context(original):
            set_attn_forward_flag(is_extend_in_batch=True)
            # the published ctx is a new object, the original stays default-False
            self.assertIsNot(get_forward_context(), original)
        self.assertFalse(original.flags.attn.is_extend_in_batch)

    def test_set_outside_scope_is_noop(self):
        set_forward_context(None)
        set_attn_forward_flag(is_extend_in_batch=True)  # must not raise
        self.assertFalse(has_forward_context())


if __name__ == "__main__":
    unittest.main()
