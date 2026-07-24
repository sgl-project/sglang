"""Unit tests for _handle_decoupled_spec in srt/arg_groups/speculative_hook.

_handle_decoupled_spec validates + normalizes the decoupled-spec role args. It
raises on unsupported combos (missing transport endpoints, negative rank,
dp_size != 1, dp attention, page_size > 1, missing speculative-algorithm /
num_steps, topk != 1) and mutates the rest (max_running_requests default,
overlap off, mixed-chunk off, topk pinned to 1, num_draft_tokens = num_steps + 1,
plus radix off and mamba no_buffer for the drafter).

It reads/writes only plain ServerArgs attributes and the is_decoupled_* helpers —
no torch, model config, or transport — so these tests drive it on CPU via a
lightweight stub with no GPU / numpy / torch dependency.
"""

import unittest

from sglang.srt.arg_groups.speculative_hook import (
    _handle_decoupled_spec as handle_decoupled_spec,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class _StubArgs:
    """Minimal stand-in for ServerArgs carrying only the fields the validator reads.

    Defaults mirror a valid decoupled config so each test can flip a single field
    to drive one ENFORCE/NORMALIZE/FORCE branch. The two role helpers below are
    copied from ServerArgs.is_decoupled_verifier/drafter so the stub stays faithful.
    """

    def __init__(self, role="verifier", **overrides):
        self.decoupled_spec_role = role
        self.decoupled_spec_bind_endpoint = "ipc:///tmp/bind"
        self.decoupled_spec_connect_endpoints = ["ipc:///tmp/peer0"]
        self.decoupled_spec_rank = 0
        self.dp_size = 1
        self.enable_dp_attention = False
        self.page_size = 1
        self.max_running_requests = 16
        self.speculative_algorithm = "STANDALONE"
        self.speculative_num_steps = 3
        self.speculative_num_draft_tokens = 4
        self.speculative_eagle_topk = 1
        self.disable_overlap_schedule = False
        self.enable_mixed_chunk = False
        self.disable_radix_cache = False
        self.mamba_radix_cache_strategy = "no_buffer"
        for key, value in overrides.items():
            setattr(self, key, value)

    def is_decoupled_verifier(self) -> bool:
        return self.decoupled_spec_role == "verifier"

    def is_decoupled_drafter(self) -> bool:
        return self.decoupled_spec_role == "drafter"


class TestDecoupledSpecHookEnforce(CustomTestCase):
    def test_missing_bind_endpoint_raises(self):
        for role in ("verifier", "drafter"):
            with self.assertRaises(ValueError):
                handle_decoupled_spec(
                    _StubArgs(role, decoupled_spec_bind_endpoint=None)
                )

    def test_missing_connect_endpoints_raises(self):
        with self.assertRaises(ValueError):
            handle_decoupled_spec(_StubArgs(decoupled_spec_connect_endpoints=None))

    def test_missing_rank_raises(self):
        with self.assertRaises(ValueError):
            handle_decoupled_spec(_StubArgs(decoupled_spec_rank=None))

    def test_negative_rank_raises(self):
        with self.assertRaises(ValueError):
            handle_decoupled_spec(_StubArgs(decoupled_spec_rank=-1))

    def test_dp_size_gt_1_raises(self):
        for role in ("verifier", "drafter"):
            with self.assertRaises(ValueError):
                handle_decoupled_spec(_StubArgs(role, dp_size=2))

    def test_dp_attention_raises(self):
        with self.assertRaises(ValueError):
            handle_decoupled_spec(_StubArgs(enable_dp_attention=True))

    def test_page_size_gt_1_raises(self):
        with self.assertRaises(ValueError):
            handle_decoupled_spec(_StubArgs(page_size=2))

    def test_page_size_none_is_allowed(self):
        args = _StubArgs(page_size=None)
        handle_decoupled_spec(args)  # must not raise
        self.assertIsNone(args.page_size)

    def test_missing_speculative_algorithm_raises(self):
        with self.assertRaises(ValueError):
            handle_decoupled_spec(_StubArgs(speculative_algorithm=None))

    def test_missing_num_steps_raises(self):
        with self.assertRaises(ValueError):
            handle_decoupled_spec(_StubArgs(speculative_num_steps=None))

    def test_topk_gt_1_raises(self):
        with self.assertRaises(ValueError):
            handle_decoupled_spec(_StubArgs(speculative_eagle_topk=2))


class TestDecoupledSpecHookNormalizeAndForce(CustomTestCase):
    def test_topk_none_normalized_to_one(self):
        args = _StubArgs(speculative_eagle_topk=None)
        handle_decoupled_spec(args)
        self.assertEqual(args.speculative_eagle_topk, 1)

    def test_num_draft_tokens_adjusted_to_num_steps_plus_one(self):
        args = _StubArgs(speculative_num_steps=3, speculative_num_draft_tokens=99)
        handle_decoupled_spec(args)
        self.assertEqual(args.speculative_num_draft_tokens, 4)

    def test_max_running_requests_defaulted(self):
        args = _StubArgs(max_running_requests=None)
        handle_decoupled_spec(args)
        self.assertEqual(args.max_running_requests, 64)

    def test_verifier_forces_overlap_and_mixed_chunk_off(self):
        args = _StubArgs(
            "verifier", disable_overlap_schedule=False, enable_mixed_chunk=True
        )
        handle_decoupled_spec(args)
        self.assertTrue(args.disable_overlap_schedule)
        self.assertFalse(args.enable_mixed_chunk)

    def test_verifier_does_not_disable_radix_cache(self):
        args = _StubArgs("verifier", disable_radix_cache=False)
        handle_decoupled_spec(args)
        # Radix-off is drafter-only; the verifier must leave it untouched.
        self.assertFalse(args.disable_radix_cache)

    def test_verifier_does_not_force_mamba_no_buffer(self):
        # Mamba no_buffer force is drafter-only; the verifier leaves it untouched.
        args = _StubArgs("verifier", mamba_radix_cache_strategy="extra_buffer")
        handle_decoupled_spec(args)
        self.assertEqual(args.mamba_radix_cache_strategy, "extra_buffer")

    def test_drafter_forces_overlap_mixed_chunk_and_radix_off(self):
        args = _StubArgs(
            "drafter",
            disable_overlap_schedule=False,
            enable_mixed_chunk=True,
            disable_radix_cache=False,
        )
        handle_decoupled_spec(args)
        self.assertTrue(args.disable_overlap_schedule)
        self.assertFalse(args.enable_mixed_chunk)
        self.assertTrue(args.disable_radix_cache)

    def test_drafter_forces_mamba_no_buffer(self):
        args = _StubArgs("drafter", mamba_radix_cache_strategy="extra_buffer")
        handle_decoupled_spec(args)
        self.assertEqual(args.mamba_radix_cache_strategy, "no_buffer")

    def test_valid_config_is_idempotent(self):
        # A config that already satisfies every constraint passes unchanged.
        args = _StubArgs(
            "drafter", disable_overlap_schedule=True, disable_radix_cache=True
        )
        handle_decoupled_spec(args)
        self.assertTrue(args.disable_overlap_schedule)
        self.assertTrue(args.disable_radix_cache)
        self.assertFalse(args.enable_mixed_chunk)
        self.assertEqual(args.speculative_eagle_topk, 1)
        self.assertEqual(args.speculative_num_draft_tokens, 4)


if __name__ == "__main__":
    unittest.main()
