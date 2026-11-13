import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


class TestFlyArgs(unittest.TestCase):
    def _args(self, **overrides):
        args = ServerArgs(
            model_path="dummy",
            device="cuda",
            speculative_algorithm="DSPARK",
            speculative_eagle_topk=1,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            speculative_use_fly=True,
        )
        args._model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"])
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def _handle(self, args, *, is_hip=True):
        # The DSPARK handler needs a real draft checkpoint.
        with (
            override_platform(is_hip=is_hip),
            mock.patch("sglang.srt.arg_groups.speculative_hook._handle_dspark"),
        ):
            handle_speculative_decoding(args)

    def test_dspark_on_rocm(self):
        self._handle(self._args())

    def test_unsupported_configurations(self):
        cases = [
            (
                {"speculative_algorithm": "EAGLE"},
                True,
                "only supports --speculative-algorithm DSPARK",
            ),
            ({}, False, "only supported on ROCm"),
            (
                {"speculative_use_block_verification": True},
                True,
                "does not support --speculative-use-block-verification",
            ),
        ]
        for overrides, is_hip, message in cases:
            with self.subTest(overrides=overrides, is_hip=is_hip):
                with self.assertRaisesRegex(ValueError, message):
                    self._handle(self._args(**overrides), is_hip=is_hip)


if __name__ == "__main__":
    unittest.main()
