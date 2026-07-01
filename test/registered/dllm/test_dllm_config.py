"""Unit tests for baseline diffusion-LM configuration handling."""

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.dllm.config import DllmConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


class TestFromServerArgs(unittest.TestCase):
    def test_max_steps_defaults_to_yaml_block_size(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
            f.write("block_size: 8\n")
            f.flush()
            server_args = SimpleNamespace(
                dllm_algorithm="FastDiffuser",
                dllm_algorithm_config=f.name,
                max_running_requests=None,
                model_path="dummy",
                revision=None,
            )
            model_config = SimpleNamespace(
                hf_config=SimpleNamespace(architectures=["NemotronLabsDiffusionModel"])
            )
            with patch(
                "sglang.srt.dllm.config.ModelConfig.from_server_args",
                return_value=model_config,
            ):
                cfg = DllmConfig.from_server_args(server_args)

        self.assertEqual(cfg.block_size, 8)
        self.assertEqual(cfg.max_steps, 8)


class TestServerArgsDllmValidation(unittest.TestCase):
    def test_pipeline_parallelism_is_disabled_for_dllm(self):
        from sglang.srt.server_args import ServerArgs

        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
            f.write("block_size: 32\n")
            f.flush()
            server_args = ServerArgs(
                model_path="dummy",
                dllm_algorithm="LinearSpec",
                dllm_algorithm_config=f.name,
                attention_backend="flashinfer",
                disable_radix_cache=True,
                disable_overlap_schedule=True,
                max_running_requests=128,
                pp_size=2,
            )
            model_config = SimpleNamespace(
                hf_config=SimpleNamespace(architectures=["NemotronLabsDiffusionModel"])
            )
            with patch(
                "sglang.srt.dllm.config.ModelConfig.from_server_args",
                return_value=model_config,
            ):
                server_args._handle_dllm_inference()

        self.assertEqual(server_args.pp_size, 1)


if __name__ == "__main__":
    unittest.main()
