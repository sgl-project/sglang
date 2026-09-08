import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.deepseek_v4_hook import apply_deepseek_v4_defaults
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_MODEL_ARCH = "DeepseekV4ForCausalLM"


def _make_v4_args(algorithm, **overrides) -> ServerArgs:
    # model_path="dummy" short-circuits ServerArgs.__post_init__; invoke the model
    # hook directly (same pattern as test_spec_cpu_overlap_constraint.py). The
    # injected _model_config carries no _model_config_built_from key, so
    # model_config_of hands it back instead of loading a checkpoint.
    args = ServerArgs(model_path="dummy")
    args.speculative_algorithm = algorithm
    args.speculative_num_steps = 2
    args.speculative_eagle_topk = 1
    args.speculative_num_draft_tokens = 3
    args._model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=[_MODEL_ARCH],
            get_text_config=lambda: SimpleNamespace(),
        )
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestDeepseekV4SpeculativeAlgorithmAlias(CustomTestCase):
    """apply_deepseek_v4_defaults must resolve the algorithm name before matching it.

    It runs from handle_model_specific_adjustments, well before
    handle_speculative_decoding case-folds --speculative-algorithm and collapses the
    NEXTN alias, so the raw CLI string reaches it. Matching that string literally
    rejected three valid command lines (issue #38236).
    """

    def test_nextn_alias_is_accepted(self):
        # The reported bug: NEXTN is a documented EAGLE alias.
        apply_deepseek_v4_defaults(_make_v4_args("NEXTN"), _MODEL_ARCH)

    def test_lowercase_algorithm_is_accepted(self):
        # The same root cause: .upper() also happens in the later hook.
        apply_deepseek_v4_defaults(_make_v4_args("eagle"), _MODEL_ARCH)
        apply_deepseek_v4_defaults(_make_v4_args("nextn"), _MODEL_ARCH)

    def test_unset_eagle_topk_is_accepted(self):
        # None is the unset default; the later hook fills it with 1, so rejecting it
        # here rejects a command line that simply omits --speculative-eagle-topk.
        apply_deepseek_v4_defaults(
            _make_v4_args("EAGLE", speculative_eagle_topk=None), _MODEL_ARCH
        )

    def test_supported_algorithms_still_pass(self):
        apply_deepseek_v4_defaults(_make_v4_args("EAGLE"), _MODEL_ARCH)
        apply_deepseek_v4_defaults(
            _make_v4_args("DSPARK", speculative_eagle_topk=None), _MODEL_ARCH
        )

    def test_topk_guard_stays_live_through_the_alias(self):
        # Resolving the name must not silence the topk check: accepting "NEXTN"
        # without resolving it would let a bad topk through unnoticed.
        for algorithm in ("EAGLE", "NEXTN", "nextn"):
            with self.subTest(algorithm=algorithm):
                args = _make_v4_args(algorithm, speculative_eagle_topk=2)
                with self.assertRaisesRegex(AssertionError, "topk == 1"):
                    apply_deepseek_v4_defaults(args, _MODEL_ARCH)

    def test_unsupported_algorithms_are_still_rejected(self):
        for algorithm in ("EAGLE3", "DFLASH"):
            with self.subTest(algorithm=algorithm):
                args = _make_v4_args(algorithm)
                with self.assertRaisesRegex(AssertionError, "EAGLE and DSPARK"):
                    apply_deepseek_v4_defaults(args, _MODEL_ARCH)


if __name__ == "__main__":
    unittest.main()
