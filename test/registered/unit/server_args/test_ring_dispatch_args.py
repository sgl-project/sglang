import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_CUDA = dict(
    is_cuda=True, is_hip=False, is_sm90=False, is_sm100=True, is_blackwell=True
)


class TestRingDispatchArgs(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.configs.muse_spark import MuseSparkConfig

        cls._dir = tempfile.TemporaryDirectory()
        cls.muse = Path(cls._dir.name) / "muse"
        config = MuseSparkConfig(architectures=["MuseSparkForConditionalGeneration"])
        config.quantization_config = {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "group_size": 16,
        }
        config.save_pretrained(cls.muse)
        cls.num_experts = config.get_text_config().num_local_experts
        cls.qwen = Path(cls._dir.name) / "qwen3_moe"
        cls.qwen.mkdir()
        (cls.qwen / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["Qwen3MoeForCausalLM"],
                    "model_type": "qwen3_moe",
                    "hidden_size": 256,
                    "intermediate_size": 512,
                    "moe_intermediate_size": 128,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 8,
                    "head_dim": 32,
                    "num_experts": 64,
                    "num_experts_per_tok": 8,
                    "vocab_size": 1024,
                    "max_position_embeddings": 4096,
                }
            )
        )

    @classmethod
    def tearDownClass(cls):
        cls._dir.cleanup()

    def _ring(self, **overrides):
        return {
            "ep_num_redundant_experts": self.num_experts,
            "init_expert_location": "ring",
            "ep_dispatch_algorithm": "ring",
            **overrides,
        }

    def _resolve(self, model_path, platform=_CUDA, **kwargs):
        with (
            patch.dict(os.environ),
            override_platform(**platform),
            patch(
                "sglang.srt.arg_groups.memory_hook.get_device_memory_capacity",
                return_value=275000,
            ),
        ):
            args = ServerArgs(
                **{
                    "model_path": str(model_path),
                    "device": "cuda",
                    "tp_size": 8,
                    "ep_size": 8,
                    "skip_tokenizer_init": True,
                    **kwargs,
                }
            )
            args.resolve_once()
        return args

    def test_muse_ring_resolves(self):
        args = self._resolve(self.muse, **self._ring())
        self.assertEqual(resolution_result(args, "ep_dispatch_algorithm"), "ring")

    def test_ring_placement_defaults_the_algorithm_to_ring(self):
        args = self._resolve(self.muse, **self._ring(ep_dispatch_algorithm=None))
        self.assertEqual(resolution_result(args, "ep_dispatch_algorithm"), "ring")

    def test_defaults_unchanged_without_ring_args(self):
        for model in (self.muse, self.qwen):
            with self.subTest(model=model.name):
                args = self._resolve(model)
                self.assertIsNone(resolution_result(args, "ep_dispatch_algorithm"))
                self.assertEqual(resolution_result(args, "ep_num_redundant_experts"), 0)
                self.assertEqual(
                    resolution_result(args, "init_expert_location"), "trivial"
                )

    def test_other_models_keep_their_eplb_options(self):
        args = self._resolve(
            self.qwen, ep_num_redundant_experts=8, ep_dispatch_algorithm="dynamic"
        )
        self.assertEqual(resolution_result(args, "ep_dispatch_algorithm"), "dynamic")

    def test_other_models_reject_ring(self):
        for kwargs in (
            dict(ep_num_redundant_experts=64, init_expert_location="ring"),
            dict(ep_dispatch_algorithm="ring"),
            dict(init_expert_location="ring"),
            dict(
                ep_num_redundant_experts=64,
                init_expert_location="ring",
                ep_dispatch_algorithm="ring",
            ),
        ):
            with (
                self.subTest(**kwargs),
                self.assertRaisesRegex(ValueError, "support only"),
            ):
                self._resolve(self.qwen, **kwargs)

    def test_muse_rejects_other_placements_and_algorithms(self):
        cases = [
            dict(ep_dispatch_algorithm=algorithm)
            for algorithm in ("static", "dynamic", "lp", "fake")
        ]
        cases += [
            self._ring(ep_dispatch_algorithm=algorithm)
            for algorithm in ("static", "dynamic", "lp", "fake")
        ]
        cases += [
            dict(ep_num_redundant_experts=self.num_experts),
            dict(ep_num_redundant_experts=-1),
            self._ring(ep_num_redundant_experts=self.num_experts - 1),
            self._ring(init_expert_location="trivial"),
            dict(init_expert_location=json.dumps({"physical_to_logical_map": [[0]]})),
            dict(enable_eplb=True),
            dict(expert_distribution_recorder_mode="stat"),
            dict(expert_balancedness_report_mode="server_log"),
        ]
        for kwargs in cases:
            with (
                self.subTest(**kwargs),
                self.assertRaisesRegex(ValueError, "supports expert placement only"),
            ):
                self._resolve(self.muse, **kwargs)

    def test_muse_ring_accepts_the_flashinfer_a2a_backend_and_its_alias(self):
        # The deprecated spelling is renamed before the ring check reads it.
        for backend in ("flashinfer_nvlink_one_sided", "flashinfer"):
            with (
                self.subTest(backend=backend),
                patch.dict(os.environ, {"SGLANG_MOE_NVFP4_DISPATCH": "0"}),
            ):
                args = self._resolve(
                    self.muse,
                    **self._ring(
                        moe_a2a_backend=backend,
                        enable_layernorm_sp=True,
                        enable_megamoe_for_layernorm_sp=False,
                    ),
                )
                self.assertEqual(
                    resolution_result(args, "moe_a2a_backend"),
                    "flashinfer_nvlink_one_sided",
                )

    def test_muse_ring_rejects_unsupported_features(self):
        for kwargs, match in (
            (dict(enable_eplb=True), "--enable-eplb"),
            (dict(expert_distribution_recorder_mode="stat"), "recorder"),
            (dict(expert_balancedness_report_mode="server_log"), "balancedness"),
            (dict(weight_cache_mode="daemon"), "weight-cache"),
            (dict(load_format="sharded_state"), "load-format"),
            (dict(attn_dp_size=8), "attention DP"),
            (dict(moe_a2a_backend="deepep"), "moe-a2a-backend"),
            (dict(ep_size=4), "EP == TP"),
        ):
            with self.subTest(**kwargs), self.assertRaisesRegex(ValueError, match):
                self._resolve(self.muse, **self._ring(**kwargs))

    def test_ring_is_cuda_only(self):
        for platform in (
            dict(is_cuda=False, is_hip=True),
            dict(is_cuda=False, is_hip=False, is_xpu=True),
            dict(is_cuda=False, is_hip=False, is_npu=True),
            dict(is_cuda=False, is_hip=False),
        ):
            with (
                self.subTest(**platform),
                self.assertRaisesRegex(ValueError, "CUDA only"),
            ):
                self._resolve(self.muse, platform=platform, **self._ring())


if __name__ == "__main__":
    unittest.main()
