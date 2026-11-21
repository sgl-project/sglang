"""Unit tests for DFLASH/DSpark rejecting an explicit multi-step draft."""

import unittest

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import (
    _handle_dflash,
    _handle_dspark,
    _resolve_single_step_drafting,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_ALGORITHMS = {
    "DFLASH": (_handle_dflash, "--speculative-dflash-block-size"),
    "DSPARK": (_handle_dspark, "--speculative-dspark-block-size"),
}


class TestSingleStepSpeculativeArgs(CustomTestCase):
    def _args(self, algorithm, **overrides):
        fields = dict(
            model_path="dummy",
            device="cuda",
            speculative_algorithm=algorithm,
            speculative_draft_model_path="draft",
        )
        if algorithm == "DFLASH":
            fields.update(
                speculative_num_draft_tokens=8,
                speculative_draft_attention_backend="flashinfer",
            )
        else:
            fields["speculative_dspark_block_size"] = 8
        return ServerArgs(**(fields | overrides))

    def _resolve(self, algorithm, **overrides):
        args = self._args(algorithm, **overrides)
        _resolve_single_step_drafting(
            args,
            "test",
            algorithm=algorithm,
            block_size_flag=_ALGORITHMS[algorithm][1],
        )
        return args

    def test_explicit_multi_step_is_rejected_at_startup(self):
        for algorithm, (handler, _flag) in _ALGORITHMS.items():
            for steps in (2, 4):
                with (
                    self.subTest(algorithm=algorithm, steps=steps),
                    self.assertRaisesRegex(
                        ValueError,
                        rf"(?i:{algorithm}) only supports --speculative-num-steps 1",
                    ),
                ):
                    handler(self._args(algorithm, speculative_num_steps=steps))

    def test_multi_step_error_points_at_block_size_knob(self):
        for algorithm, (handler, block_size_flag) in _ALGORITHMS.items():
            with (
                self.subTest(algorithm=algorithm),
                self.assertRaisesRegex(ValueError, block_size_flag),
            ):
                handler(self._args(algorithm, speculative_num_steps=2))

    def test_unset_num_steps_defaults_to_one(self):
        for algorithm in _ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                args = self._resolve(algorithm)
                self.assertIsNone(args.speculative_num_steps)
                self.assertEqual(resolution_result(args, "speculative_num_steps"), 1)

    def test_explicit_single_step_is_accepted(self):
        for algorithm in _ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                args = self._resolve(algorithm, speculative_num_steps=1)
                self.assertFalse(getattr(args, "_resolved_overrides", None))
                self.assertEqual(resolution_result(args, "speculative_num_steps"), 1)


if __name__ == "__main__":
    unittest.main()
