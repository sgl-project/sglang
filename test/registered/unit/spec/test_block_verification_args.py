import json
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


_MINI_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "vocab_size": 128,
    "max_position_embeddings": 2048,
}


class TestBlockVerificationArgs(CustomTestCase):
    def _args(self, **overrides):
        args = ServerArgs(
            model_path="dummy",
            device="cuda",
            speculative_algorithm="EAGLE",
            speculative_eagle_topk=1,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            speculative_use_block_verification=True,
        )
        args._model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"])
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def _real_args(self, **overrides):
        checkpoint = tempfile.mkdtemp(prefix="pp_spec_verification_")
        self.addCleanup(shutil.rmtree, checkpoint, ignore_errors=True)
        with open(os.path.join(checkpoint, "config.json"), "w") as handle:
            json.dump(_MINI_CONFIG, handle)
        args = ServerArgs(
            model_path=checkpoint,
            device="cuda",
            pp_size=2,
            speculative_algorithm="EAGLE",
            speculative_eagle_topk=1,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_pp_rejects_proposal_distribution_verification_before_launch(self):
        cases = [
            {"speculative_use_rejection_sampling": True},
            {"speculative_use_block_verification": True},
            {
                "speculative_algorithm": "NEXTN",
                "speculative_use_rejection_sampling": True,
            },
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                args = self._real_args(**overrides)
                with (
                    mock.patch.dict(os.environ, {"SGLANG_ENABLE_PP_SPEC": "1"}),
                    self.assertRaisesRegex(
                        ValueError, "pipeline parallelism.*draft proposal distribution"
                    ),
                ):
                    args.resolve_once()

    def test_supported_pp_and_single_stage_configurations_unchanged(self):
        cases = [
            {},
            {"pp_size": 1, "speculative_use_rejection_sampling": True},
            {"pp_size": 1, "speculative_use_block_verification": True},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                args = self._real_args(**overrides)
                with mock.patch.dict(
                    os.environ, {"SGLANG_ENABLE_PP_SPEC": "1"}, clear=False
                ):
                    args.resolve_once()

    def test_hip_auto_enable_rejection_is_rejected_for_pp(self):
        args = self._real_args()
        with (
            mock.patch.dict(os.environ, {"SGLANG_ENABLE_PP_SPEC": "1"}),
            mock.patch(
                "sglang.srt.arg_groups.speculative_hook.get_platform",
                return_value=SimpleNamespace(is_hip=True),
            ),
            self.assertRaisesRegex(
                ValueError, "pipeline parallelism.*draft proposal distribution"
            ),
        ):
            args.resolve_once()

    def test_supported_algorithms(self):
        for algorithm in ("EAGLE", "EAGLE3", "NEXTN"):
            for use_rejection_sampling in (False, True):
                with self.subTest(
                    algorithm=algorithm, use_rejection_sampling=use_rejection_sampling
                ):
                    args = self._args(
                        speculative_algorithm=algorithm,
                        speculative_use_rejection_sampling=use_rejection_sampling,
                    )
                    handle_speculative_decoding(args)
                    self.assertTrue(
                        resolution_result(args, "speculative_use_block_verification")
                    )
                    self.assertTrue(
                        resolution_result(args, "speculative_use_rejection_sampling")
                    )
                    self.assertEqual(
                        args.speculative_use_rejection_sampling, use_rejection_sampling
                    )

    def test_dflash_family(self):
        """DFLASH/DSPARK sampled drafts publish their own proposal distribution,
        so block verification must not switch on EAGLE's rejection sampling."""
        for algorithm in ("DFLASH", "DSPARK"):
            with self.subTest(algorithm=algorithm):
                overrides = {"speculative_algorithm": algorithm}
                if algorithm == "DSPARK":
                    # PP is supported for DSpark PD prefill independently of the
                    # experimental EAGLE PP relay guarded by this change.
                    overrides.update(pp_size=2, disaggregation_mode="prefill")
                args = self._args(**overrides)
                # The per-algorithm handlers need a real draft checkpoint.
                with (
                    mock.patch("sglang.srt.arg_groups.speculative_hook._handle_dflash"),
                    mock.patch("sglang.srt.arg_groups.speculative_hook._handle_dspark"),
                ):
                    handle_speculative_decoding(args)
                self.assertTrue(
                    resolution_result(args, "speculative_use_block_verification")
                )
                self.assertFalse(
                    resolution_result(args, "speculative_use_rejection_sampling")
                )

    def test_other_verification_modes_unchanged(self):
        for use_rejection_sampling in (False, True):
            with self.subTest(use_rejection_sampling=use_rejection_sampling):
                args = self._args(
                    speculative_algorithm="EAGLE3",
                    speculative_use_block_verification=False,
                    speculative_use_rejection_sampling=use_rejection_sampling,
                )
                handle_speculative_decoding(args)
                self.assertFalse(
                    resolution_result(args, "speculative_use_block_verification")
                )
                self.assertEqual(
                    resolution_result(args, "speculative_use_rejection_sampling"),
                    use_rejection_sampling,
                )

    def test_unsupported_configurations(self):
        cases = [
            ({"speculative_algorithm": None}, "only supports EAGLE"),
            ({"speculative_algorithm": "STANDALONE"}, "only supports EAGLE"),
            ({"speculative_algorithm": "NGRAM"}, "only supports EAGLE"),
            ({"device": "cpu"}, "only supports CUDA or ROCm"),
            ({"device": "npu"}, "only supports CUDA or ROCm"),
            # DFLASH/DSPARK serve on NPU, whose chain sampler has no block mode.
            (
                {"speculative_algorithm": "DFLASH", "device": "npu"},
                "only supports CUDA or ROCm",
            ),
            (
                {"speculative_algorithm": "DSPARK", "device": "npu"},
                "only supports CUDA or ROCm",
            ),
            ({"speculative_eagle_topk": 2}, "requires --speculative-eagle-topk=1"),
            ({"speculative_accept_threshold_single": 0.5}, "incompatible"),
            ({"speculative_accept_threshold_acc": 0.5}, "incompatible"),
        ]
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    handle_speculative_decoding(self._args(**overrides))

    def test_deterministic_inference_supported_on_cuda(self):
        # Block verification reuses rejection sampling's seeded draft and verify draws,
        # so deterministic inference must stay allowed on CUDA.
        args = self._args(enable_deterministic_inference=True)
        handle_speculative_decoding(args)
        self.assertTrue(resolution_result(args, "speculative_use_block_verification"))
        self.assertTrue(resolution_result(args, "speculative_use_rejection_sampling"))


if __name__ == "__main__":
    unittest.main()
