import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestBlockVerificationArgs(unittest.TestCase):
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
                args = self._args(speculative_algorithm=algorithm)
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
