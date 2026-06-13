"""Foundation tests for the shape-classified ForwardMode names + predicates.

These cover the additive step: the two clean enum aliases
(``VAR_LEN``/``SINGLE_TOKEN``), the three shape predicates
(``is_var_len``/``is_single_token``/``is_uniform_len``), and the
``AttentionBackend`` shape-named forward methods that default-delegate to the
legacy names. Pure CPU — no kernels.

Key invariants:
  * ``VAR_LEN is EXTEND`` / ``SINGLE_TOKEN is DECODE`` (value-aliases).
  * ``is_single_token() == is_decode()`` for every member (behavior-equivalent).
  * The shape predicates are disjoint and complete over all members.
  * ``forward_var_len/single_token/uniform_len`` delegate to
    ``forward_extend/decode/mixed``.
"""

import unittest

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-test-cpu")


class TestForwardModeShapeClasses(CustomTestCase):
    def test_clean_enum_aliases(self) -> None:
        # Value-aliases: the new names resolve to the existing members.
        self.assertIs(ForwardMode.VAR_LEN, ForwardMode.EXTEND)
        self.assertIs(ForwardMode.SINGLE_TOKEN, ForwardMode.DECODE)
        # MIXED is intentionally NOT aliased to a shape class in this step.

    def test_is_single_token_equivalent_to_is_decode(self) -> None:
        for mode in ForwardMode:
            self.assertEqual(mode.is_single_token(), mode.is_decode(), msg=mode.name)

    def test_shape_predicates_disjoint_and_complete(self) -> None:
        # Every member is in exactly one shape class (var-len / single-token /
        # uniform-len / idle).
        for mode in ForwardMode:
            classes = [
                mode.is_var_len(),
                mode.is_single_token(),
                mode.is_uniform_len(),
                mode.is_idle(),
            ]
            self.assertEqual(
                sum(classes), 1, msg=f"{mode.name} -> {classes} (not exactly one)"
            )

    def test_shape_predicate_membership(self) -> None:
        self.assertEqual(
            {m for m in ForwardMode if m.is_var_len()},
            {
                ForwardMode.EXTEND,
                ForwardMode.MIXED,
                ForwardMode.DRAFT_EXTEND,
                ForwardMode.SPLIT_PREFILL,
                ForwardMode.DLLM_EXTEND,
            },
        )
        self.assertEqual(
            {m for m in ForwardMode if m.is_uniform_len()},
            {
                ForwardMode.TARGET_VERIFY,
                ForwardMode.DRAFT_EXTEND_V2,
                ForwardMode.PREBUILT,
            },
        )
        self.assertEqual(
            {m for m in ForwardMode if m.is_single_token()}, {ForwardMode.DECODE}
        )

    def test_is_var_len_is_not_is_extend(self) -> None:
        # Guard the documented difference: is_extend() is a semantic grab-bag
        # that includes TARGET_VERIFY; is_var_len() is the clean shape class.
        self.assertTrue(ForwardMode.TARGET_VERIFY.is_extend())
        self.assertFalse(ForwardMode.TARGET_VERIFY.is_var_len())

    def test_abc_shape_methods_delegate_to_legacy(self) -> None:
        class _MockBackend(AttentionBackend):
            def __init__(self):  # bypass any base init
                pass

            def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True):
                return ("extend", save_kv_cache)

            def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True):
                return ("decode", save_kv_cache)

            def forward_mixed(self, q, k, v, layer, forward_batch, save_kv_cache=True):
                return ("mixed", save_kv_cache)

        b = _MockBackend()
        self.assertEqual(b.forward_var_len(0, 0, 0, None, None), ("extend", True))
        self.assertEqual(b.forward_single_token(0, 0, 0, None, None), ("decode", True))
        self.assertEqual(
            b.forward_uniform_len(0, 0, 0, None, None, save_kv_cache=False),
            ("mixed", False),
        )


if __name__ == "__main__":
    unittest.main()
