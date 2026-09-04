"""Unit tests for the --enable-draft-prefetch server-arg validation.

Covers _check_draft_prefetch (sglang.srt.arg_groups.speculative_hook): the
feature only supports EAGLE/EAGLE3/NEXTN with topk=1 and num_steps > 1, is
incompatible with adaptive speculative decoding, rejection sampling and
multi-layer EAGLE, and must be a no-op for configs that don't enable it. The
E2E behavior is covered by test/registered/spec/eagle/test_spec_eagle_draft_prefetch.py.
"""

import unittest

from sglang.srt.arg_groups.speculative_hook import _check_draft_prefetch
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_HOOK_LOGGER = "sglang.srt.arg_groups.speculative_hook"


def _make_args(**overrides) -> ServerArgs:
    # model_path="dummy" short-circuits ServerArgs.__post_init__; invoke the
    # validation hook directly (same pattern as test_spec_cpu_overlap_constraint).
    args = ServerArgs(model_path="dummy")
    args.speculative_algorithm = "EAGLE3"
    args.speculative_num_steps = 5
    args.speculative_eagle_topk = 1
    args.speculative_num_draft_tokens = 6
    args.enable_draft_prefetch = True
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestDraftPrefetchArgValidation(CustomTestCase):
    def test_supported_algorithms_pass(self):
        for algo in ("EAGLE3", "EAGLE"):
            with self.subTest(algo=algo):
                _check_draft_prefetch(_make_args(speculative_algorithm=algo))

    def test_nextn_promoted_to_eagle_passes(self):
        # NEXTN never reaches the check as-is: the alias hook promotes it to
        # EAGLE in the resolution stash before _check_draft_prefetch runs, so
        # the check only ever sees EAGLE (never "NEXTN").
        args = _make_args(speculative_algorithm="NEXTN")
        args._resolved_overrides = [
            ("handle_speculative_decoding", {"speculative_algorithm": "EAGLE"}),
        ]
        _check_draft_prefetch(args)

    def test_disabled_skips_validation(self):
        # topk>1 / num_steps<=1 / other algorithms are only rejected when the
        # feature is enabled; other spec configs must stay unaffected.
        _check_draft_prefetch(
            _make_args(
                enable_draft_prefetch=False,
                speculative_eagle_topk=8,
                speculative_adaptive=True,
            )
        )
        _check_draft_prefetch(
            _make_args(
                enable_draft_prefetch=False,
                speculative_algorithm="DFLASH",
                speculative_num_steps=1,
                speculative_eagle_topk=1,
            )
        )
        _check_draft_prefetch(
            _make_args(
                enable_draft_prefetch=False,
                speculative_num_steps=None,
            )
        )

    def test_num_steps_must_exceed_one(self):
        for steps in (None, 1):
            with self.subTest(num_steps=steps):
                args = _make_args(speculative_num_steps=steps)
                with self.assertRaisesRegex(ValueError, "num-steps > 1"):
                    _check_draft_prefetch(args)
        # 2 steps (the minimum chain with a pre-run draft) is accepted.
        _check_draft_prefetch(
            _make_args(speculative_num_steps=2, speculative_num_draft_tokens=3)
        )

    def test_unsupported_algorithm_rejected(self):
        args = _make_args(speculative_algorithm="DFLASH")
        with self.assertRaisesRegex(ValueError, "only supports EAGLE/EAGLE3/NEXTN"):
            _check_draft_prefetch(args)

    def test_topk_not_one_rejected(self):
        args = _make_args(speculative_eagle_topk=2)
        with self.assertRaisesRegex(ValueError, "topk == 1"):
            _check_draft_prefetch(args)

    def test_adaptive_rejected(self):
        args = _make_args(speculative_adaptive=True)
        with self.assertRaisesRegex(ValueError, "speculative-adaptive"):
            _check_draft_prefetch(args)

    def test_multi_layer_eagle_rejected(self):
        # Explicit flag: raw field set directly on the record.
        args = _make_args(enable_multi_layer_eagle=True)
        with self.assertRaisesRegex(ValueError, "multi-layer EAGLE"):
            _check_draft_prefetch(args)

    def test_multi_layer_eagle_declared_override_rejected(self):
        # Auto-declared by a model override (e.g. MiMoV2): the field stays
        # False; the check must read it through resolved_view's overlay.
        args = _make_args()
        args._resolved_overrides = [
            ("_mimo_v2_overrides", {"enable_multi_layer_eagle": True}),
        ]
        with self.assertRaisesRegex(ValueError, "multi-layer EAGLE"):
            _check_draft_prefetch(args)

    def test_reads_resolution_overlay(self):
        # declare_resolution never writes the fields during __post_init__;
        # the decisions live only in the stash. The check must read through
        # the overlay: raw reads would reject auto-params configs
        # (num_steps/topk derived, fields still None), lowercase CLI
        # algorithms (upper() declared), and auto-disabled adaptive.
        args = _make_args()
        args.speculative_algorithm = "eagle3"
        args.speculative_num_steps = None
        args.speculative_eagle_topk = None
        args.speculative_adaptive = True
        args._resolved_overrides = [
            ("handle_speculative_decoding", {"speculative_algorithm": "EAGLE3"}),
            (
                "_handle_eagle_family.auto_params",
                {
                    "speculative_num_steps": 3,
                    "speculative_eagle_topk": 1,
                    "speculative_num_draft_tokens": 4,
                },
            ),
            ("_maybe_disable_adaptive", {"speculative_adaptive": False}),
        ]
        _check_draft_prefetch(args)

    def test_alias_promoted_algorithm_rejected(self):
        # Gemma4 assistant drafts promote NEXTN/EAGLE to FROZEN_KV_MTP in
        # the stash while the raw field keeps the user's input; a raw read
        # would let the promoted algorithm through the gate.
        args = _make_args(speculative_algorithm="NEXTN")
        args._resolved_overrides = [
            (
                "handle_speculative_decoding",
                {"speculative_algorithm": "FROZEN_KV_MTP"},
            ),
        ]
        with self.assertRaisesRegex(ValueError, "only supports EAGLE/EAGLE3/NEXTN"):
            _check_draft_prefetch(args)

    def test_rejection_sampling_rejected(self):
        args = _make_args(speculative_use_rejection_sampling=True)
        with self.assertRaisesRegex(ValueError, "rejection-sampling"):
            _check_draft_prefetch(args)

    def test_overlap_plan_stream_warns(self):
        args = _make_args()
        with envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.override(True):
            with self.assertLogs(_HOOK_LOGGER, "WARNING") as logs:
                _check_draft_prefetch(args)
        self.assertTrue(
            any("OVERLAP_PLAN_STREAM" in message for message in logs.output),
            f"expected an overlap-plan-stream warning, got: {logs.output}",
        )

    def test_no_warning_when_feature_simply_enabled(self):
        # A plain enabled config must validate silently (no warnings).
        with self.assertNoLogs(_HOOK_LOGGER, "WARNING"):
            _check_draft_prefetch(_make_args())


if __name__ == "__main__":
    unittest.main()
