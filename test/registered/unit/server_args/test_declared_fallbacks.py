"""A field can declare what it means when nobody said anything.

`Arg(fallback=...)` is the value a field takes when the operator did not type
it and resolution did not decide it. It is not the dataclass default: that
stays `None`, because `None` is how the record spells "not typed" and the
record is what crosses a process boundary.

The point of the tests below is that the three surfaces keep disagreeing, on
purpose:

* the **record** still holds `None` -- so a model family asking "did anyone set
  this?" still gets an answer, and the wire format is unchanged;
* `resolving_view` -- the view a resolution pass decides on -- still answers
  `None`, so a family's `if cfg.x is None` fires and its declaration lands.
  This is the half that carries the design: `model_overrides/inkling.py` and
  `deepseek_v4.py` are the only `is None` readers of either declared field, and
  both read this view. Pinned by the family-shaped test below rather than by
  asserting the view directly, because the shape is what has to keep working.
  `resolved_view` is a separate class and is not pinned here -- nothing reads a
  declared field through it, and `with_fallback` is called from exactly one
  place (`resolution_result`), which is what makes both views answer `None`
  without either of them knowing about fallbacks;
* the **effective** surface -- `resolution_result`, the projection, and the
  config bags every runtime reader goes through -- answers with the fallback.

That last split is the whole design. Putting the fallback in the views instead
would make `if cfg.swa_full_tokens_ratio is None` in `model_overrides/inkling.py`
never fire, and the family's 0.1 would be silently replaced by the generic 0.8.
"""

import unittest

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    resolution_result,
    resolving_view,
)
from sglang.srt.runtime_context import get_schedule, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _resolved(**kwargs) -> ServerArgs:
    server_args = ServerArgs(model_path="dummy", **kwargs)
    server_args.resolve_once()
    return server_args


class TestAFallbackIsWhatNobodySaid(CustomTestCase):
    def test_the_effective_value_is_the_declared_fallback(self):
        server_args = _resolved()
        self.assertEqual(resolution_result(server_args, "swa_full_tokens_ratio"), 0.8)
        self.assertEqual(resolution_result(server_args, "mamba_full_memory_ratio"), 0.9)

    def test_the_record_still_says_the_operator_typed_nothing(self):
        # The fallback is not the dataclass default. A child process that
        # unpickles this record has to be able to tell "unset" from "set to
        # the value resolution would have picked anyway".
        server_args = _resolved()
        self.assertIsNone(server_args.swa_full_tokens_ratio)
        self.assertIsNone(server_args.mamba_full_memory_ratio)

    def test_what_the_operator_typed_wins(self):
        server_args = _resolved(swa_full_tokens_ratio=0.25)
        self.assertEqual(resolution_result(server_args, "swa_full_tokens_ratio"), 0.25)

    def test_a_decision_wins(self):
        server_args = _resolved()
        declare_resolution(server_args, "a_model_family", swa_full_tokens_ratio=0.1)
        self.assertEqual(resolution_result(server_args, "swa_full_tokens_ratio"), 0.1)

    def test_the_published_bag_answers_with_the_fallback(self):
        # Every runtime reader goes through the bags, so this is the surface
        # that decides how the pools are sized.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="test")
        self.assertEqual(get_schedule().swa_full_tokens_ratio, 0.8)
        self.assertEqual(get_schedule().mamba_full_memory_ratio, 0.9)

    def test_the_dummy_short_circuit_leaves_no_bag_holding_none(self):
        # The dummy path returns before most of the pipeline. It used to have
        # to call the ratio pass by hand on the way out; a declared fallback
        # needs no slot, so there is nothing left to forget.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="none"), role="test")
        self.assertIsNotNone(get_schedule().swa_full_tokens_ratio)
        self.assertIsNotNone(get_schedule().mamba_full_memory_ratio)


class TestAPassDecidingStillSeesUnset(CustomTestCase):
    """The views are the Decision-over-Input surface, not the Effect surface.

    `model_overrides/inkling.py` and `model_overrides/deepseek_v4.py` both ask
    `if cfg.swa_full_tokens_ratio is None` and declare 0.1 when it is. If a
    fallback answered here, that branch would be dead and the family's value
    would never be declared.
    """

    def test_a_family_that_tests_is_none_still_fires(self):
        server_args = _resolved()
        cfg = resolving_view(server_args)
        declared = {}
        if cfg.swa_full_tokens_ratio is None:  # the family's exact shape
            declared["swa_full_tokens_ratio"] = 0.1
        self.assertEqual(declared, {"swa_full_tokens_ratio": 0.1})


if __name__ == "__main__":
    unittest.main()
