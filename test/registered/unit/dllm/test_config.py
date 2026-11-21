import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.dllm.algorithm import get_algorithm
from sglang.srt.dllm.config import DllmConfig

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDllmConfig(CustomTestCase):
    def _parse_config(self, algorithm, content):
        architecture = (
            "DiffusionGemmaForBlockDiffusion"
            if algorithm == "Gemma4Renoise"
            else "LLaDA2MoeModelLM"
        )
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=[architecture],
                delete_token_id=156930,
                split_token_id=156931,
            ),
            vocab_size=157184,
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as config_file:
            config_file.write(content)
            config_file.flush()
            server_args = SimpleNamespace(
                dllm_algorithm=algorithm,
                dllm_algorithm_config=config_file.name,
                model_path="local/diffusion-model",
                revision=None,
                max_running_requests=None,
                dllm_fdfo=False,
            )
            with patch(
                "sglang.srt.dllm.config.ModelConfig.from_server_args",
                return_value=model_config,
            ):
                return DllmConfig.from_server_args(server_args)

    def test_unknown_keys_list_the_supported_keys(self):
        key = "threshhold"  # codespell:ignore threshhold
        with self.assertRaises(ValueError) as error:
            self._parse_config("LowConfidence", yaml.safe_dump({key: 0.8}))
        self.assertEqual(
            str(error.exception),
            f"Unknown configuration keys for LowConfidence: {key}. "
            "Supported keys: block_size, threshold",
        )

    def test_keys_from_other_algorithms_are_rejected(self):
        cases = (
            ("LowConfidence", "edit_threshold"),
            ("JointThreshold", "sampler_config"),
            ("JointThresholdInDel", "penalty_lambda"),
            ("JointThresholdInDel", "vectorized_decoding"),
            ("Gemma4Renoise", "threshold"),
        )
        for algorithm, key in cases:
            with self.subTest(algorithm=algorithm, key=key):
                with self.assertRaises(ValueError) as error:
                    self._parse_config(algorithm, yaml.safe_dump({key: 1}))
                self.assertIn(
                    f"Unknown configuration keys for {algorithm}: {key}. "
                    "Supported keys: ",
                    str(error.exception),
                )

    def test_unknown_gemma_keys_include_the_section(self):
        cases = (
            (
                "sampler_config",
                "entrophy_bound",  # codespell:ignore entrophy
                "sampler_config.entropy_bound",
            ),
            (
                "temperature_schedule",
                "min_temperature",
                "temperature_schedule.t_max, temperature_schedule.t_min",
            ),
            (
                "stopping_config",
                "confidence_treshold",  # codespell:ignore treshold
                "stopping_config.confidence_threshold, "
                "stopping_config.stability_threshold",
            ),
        )
        for section, key, supported_keys in cases:
            with self.subTest(section=section):
                with self.assertRaises(ValueError) as error:
                    self._parse_config(
                        "Gemma4Renoise", yaml.safe_dump({section: {key: 1}})
                    )
                self.assertEqual(
                    str(error.exception),
                    f"Unknown configuration keys for Gemma4Renoise: {section}.{key}. "
                    f"Supported keys: {supported_keys}",
                )

    def test_mixed_unknown_key_types_report_a_value_error(self):
        with self.assertRaises(ValueError) as error:
            self._parse_config(
                "LowConfidence", yaml.safe_dump({7: 1, "extra_option": 2})
            )
        self.assertIn(
            "Unknown configuration keys for LowConfidence: 7, extra_option. "
            "Supported keys: ",
            str(error.exception),
        )

    def test_input_preparation_option_requires_algorithm_support(self):
        for algorithm in ("LowConfidence", "JointThreshold", "JointThresholdInDel"):
            with self.subTest(algorithm=algorithm):
                with self.assertRaises(ValueError) as error:
                    self._parse_config(
                        algorithm,
                        yaml.safe_dump({"capture_input_preparation": True}),
                    )
                self.assertIn(
                    f"Unknown configuration keys for {algorithm}: "
                    "capture_input_preparation. Supported keys: ",
                    str(error.exception),
                )

    def test_gemma_input_preparation_option(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                config = self._parse_config(
                    "Gemma4Renoise",
                    yaml.safe_dump({"capture_input_preparation": enabled}),
                )
                self.assertEqual(config.capture_input_preparation, enabled)
                self.assertEqual(
                    get_algorithm(config).use_graph_input_preparation, enabled
                )

    def test_valid_threshold_configs_reach_the_algorithm(self):
        cases = (
            ("LowConfidence", {"threshold": 0.8}),
            (
                "JointThreshold",
                {
                    "threshold": 0.7,
                    "edit_threshold": 0.2,
                    "max_post_edit_steps": 3,
                    "penalty_lambda": 0.1,
                    "vectorized_decoding": True,
                },
            ),
            (
                "JointThresholdInDel",
                {
                    "threshold": 0.6,
                    "edit_threshold": 0.3,
                    "max_post_edit_steps": 2,
                },
            ),
        )
        for name, values in cases:
            with self.subTest(algorithm=name):
                config = self._parse_config(
                    name,
                    yaml.safe_dump({**values, "block_size": 16}),
                )
                algorithm = get_algorithm(config)
                self.assertEqual(config.block_size, 16)
                self.assertEqual(algorithm.block_size, 16)
                self.assertEqual(config.mask_id, 156895)
                self.assertFalse(config.capture_input_preparation)
                for key, value in values.items():
                    self.assertEqual(getattr(algorithm, key), value)

    def test_gemma_nested_values_override_flat_aliases(self):
        config = self._parse_config(
            "Gemma4Renoise",
            yaml.safe_dump(
                {
                    "block_size": 16,
                    "capture_input_preparation": False,
                    "max_denoising_steps": 12,
                    "seed": 42,
                    "use_temperature_table": False,
                    "t_min": 0.1,
                    "t_max": 0.9,
                    "confidence_threshold": 0.03,
                    "stability_threshold": 3,
                    "sampler_config": {"entropy_bound": 0.2},
                    "temperature_schedule": {"t_min": 0.3, "t_max": 0.7},
                    "stopping_config": {
                        "confidence_threshold": 0.02,
                        "stability_threshold": 2,
                    },
                }
            ),
        )
        algorithm = get_algorithm(config)
        self.assertEqual(algorithm.block_size, 16)
        self.assertEqual(algorithm.max_denoising_steps, 12)
        self.assertEqual(algorithm.seed, 42)
        self.assertFalse(algorithm.use_temperature_table)
        self.assertFalse(algorithm.use_graph_input_preparation)
        self.assertEqual(algorithm.entropy_bound, 0.2)
        self.assertEqual((algorithm.t_min, algorithm.t_max), (0.3, 0.7))
        self.assertEqual(algorithm.confidence_threshold, 0.02)
        self.assertEqual(algorithm.stability_threshold, 2)

    def test_empty_yaml_preserves_defaults(self):
        for name, threshold in (
            ("LowConfidence", 0.95),
            ("Gemma4Renoise", None),
        ):
            with self.subTest(algorithm=name):
                config = self._parse_config(name, "")
                algorithm = get_algorithm(config)
                self.assertEqual(config.algorithm_config, {})
                self.assertEqual(config.max_running_requests, 1)
                if threshold is not None:
                    self.assertEqual(config.block_size, 32)
                    self.assertEqual(algorithm.threshold, threshold)
                else:
                    self.assertEqual(config.block_size, 256)
                    self.assertEqual(algorithm.max_denoising_steps, 48)
                    self.assertEqual(algorithm.entropy_bound, 0.1)
                    self.assertEqual((algorithm.t_min, algorithm.t_max), (0.4, 0.8))
                    self.assertTrue(algorithm.use_graph_input_preparation)

    def test_null_gemma_sections_preserve_defaults(self):
        config = self._parse_config(
            "Gemma4Renoise",
            "sampler_config: null\ntemperature_schedule: null\nstopping_config: null\n",
        )
        algorithm = get_algorithm(config)
        self.assertEqual(algorithm.entropy_bound, 0.1)
        self.assertEqual((algorithm.t_min, algorithm.t_max), (0.4, 0.8))
        self.assertEqual(algorithm.confidence_threshold, 0.005)
        self.assertEqual(algorithm.stability_threshold, 1)

    def test_non_mapping_yaml_keeps_the_existing_error(self):
        with self.assertRaisesRegex(
            ValueError, "The dLLM algorithm config must be a YAML mapping"
        ):
            self._parse_config("LowConfidence", "42\n")


if __name__ == "__main__":
    unittest.main()
