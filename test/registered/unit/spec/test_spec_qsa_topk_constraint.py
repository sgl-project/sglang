import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_spec_args(*, qsa=True, nested=True, algorithm="EAGLE", **overrides):
    # As in test_spec_cpu_overlap_constraint.py, invoke the real speculative
    # hook with raw arguments and an in-memory model config.
    args = ServerArgs(model_path="dummy")
    args.device = "cuda"
    args.attention_backend = "triton"
    args.page_size = 64
    args.speculative_algorithm = algorithm
    args.speculative_num_steps = 2
    args.speculative_eagle_topk = 1
    args.speculative_num_draft_tokens = 3
    text_config = SimpleNamespace()
    if qsa:
        text_config = SimpleNamespace(
            indexer_n_heads=4,
            indexer_kv_heads=1,
            indexer_head_dim=128,
            indexer_budget=2048,
            indexer_compress_ratio=4,
        )
    hf_config = SimpleNamespace(text_config=text_config) if nested else text_config
    hf_config.architectures = ["Qwen4ExpForConditionalGeneration"]
    hf_config.get_text_config = lambda: text_config
    args._model_config = SimpleNamespace(hf_config=hf_config)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestSpecQSATopKConstraint(CustomTestCase):
    def test_qsa_rejects_tree_speculation(self):
        for algorithm in ("EAGLE", "EAGLE3", "NEXTN", "STANDALONE"):
            with self.subTest(algorithm=algorithm):
                args = _make_spec_args(
                    algorithm=algorithm,
                    speculative_eagle_topk=2,
                    speculative_num_draft_tokens=4,
                )

                with self.assertRaisesRegex(
                    ValueError,
                    r"QSA.*--speculative-eagle-topk=1.*got 2",
                ):
                    handle_speculative_decoding(args)

    def test_qsa_accepts_chain_speculation(self):
        args = _make_spec_args()

        handle_speculative_decoding(args)

        self.assertEqual(resolution_result(args, "speculative_eagle_topk"), 1)

    def test_qsa_rejects_single_step_tree_speculation(self):
        args = _make_spec_args(speculative_num_steps=1, speculative_eagle_topk=2)

        with self.assertRaisesRegex(ValueError, r"QSA.*got 2"):
            handle_speculative_decoding(args)

    def test_qsa_rejects_tree_after_adaptive_fallback(self):
        args = _make_spec_args(speculative_adaptive=True, speculative_eagle_topk=2)

        with self.assertRaisesRegex(ValueError, r"QSA.*got 2"):
            handle_speculative_decoding(args)

        self.assertFalse(resolution_result(args, "speculative_adaptive"))

    def test_flat_qsa_config_rejects_tree_speculation(self):
        args = _make_spec_args(nested=False, speculative_eagle_topk=2)

        with self.assertRaisesRegex(ValueError, r"QSA.*got 2"):
            handle_speculative_decoding(args)

    def test_non_qsa_accepts_tree_speculation(self):
        # Sharing the Qwen4-Exp architecture alone must not impose QSA's limit.
        args = _make_spec_args(qsa=False, speculative_eagle_topk=2)

        handle_speculative_decoding(args)

        self.assertEqual(resolution_result(args, "speculative_eagle_topk"), 2)

    def test_qsa_defaults_are_validated_after_resolution(self):
        args = _make_spec_args(
            speculative_num_steps=None,
            speculative_eagle_topk=None,
            speculative_num_draft_tokens=None,
        )

        handle_speculative_decoding(args)

        # Resolution declares effective defaults without changing raw inputs.
        self.assertIsNone(args.speculative_eagle_topk)
        self.assertEqual(resolution_result(args, "speculative_eagle_topk"), 1)
        self.assertEqual(resolution_result(args, "speculative_num_steps"), 3)
        self.assertEqual(resolution_result(args, "speculative_num_draft_tokens"), 4)

    def test_non_qsa_tree_defaults_are_preserved(self):
        args = _make_spec_args(
            qsa=False,
            speculative_num_steps=None,
            speculative_eagle_topk=None,
            speculative_num_draft_tokens=None,
        )
        args._model_config.hf_config.architectures = ["LlamaForCausalLM"]

        handle_speculative_decoding(args)

        self.assertEqual(resolution_result(args, "speculative_eagle_topk"), 4)

    def test_qsa_without_speculation_is_unaffected(self):
        args = _make_spec_args(algorithm=None, speculative_eagle_topk=2)

        handle_speculative_decoding(args)

        self.assertEqual(resolution_result(args, "speculative_eagle_topk"), 2)

    def test_draft_token_normalization_is_preserved(self):
        args = _make_spec_args(speculative_num_draft_tokens=6)

        handle_speculative_decoding(args)

        self.assertEqual(resolution_result(args, "speculative_eagle_topk"), 1)
        self.assertEqual(resolution_result(args, "speculative_num_draft_tokens"), 3)


if __name__ == "__main__":
    unittest.main()
