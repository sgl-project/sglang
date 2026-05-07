import os
import sys
import unittest
from unittest.mock import patch

from sglang.multimodal_gen.configs.models.fsdp import (
    is_module_list_entry,
    is_module_list_entry_in,
    is_zimage_layer,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImagePipelineConfig,
)
from sglang.multimodal_gen.registry import _get_config_info
from sglang.multimodal_gen.runtime.models.dits.qwen_image import (
    QwenImageTransformer2DModel,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.utils import FlexibleArgumentParser


class TestServerArgsPathExpansion(unittest.TestCase):
    def _from_dict_without_model_resolution(self, kwargs):
        with patch.object(
            PipelineConfig, "from_kwargs", return_value=QwenImagePipelineConfig()
        ):
            return ServerArgs.from_dict(kwargs)

    def test_tilde_model_path_is_expanded(self):
        args = self._from_dict_without_model_resolution(
            {"model_path": "~/fake/local/model"}
        )
        expected = os.path.expanduser("~/fake/local/model")
        self.assertEqual(args.model_path, expected)
        self.assertFalse(args.model_path.startswith("~"))

    def test_absolute_path_is_unchanged(self):
        args = self._from_dict_without_model_resolution(
            {"model_path": "/data/my-model"}
        )
        self.assertEqual(args.model_path, "/data/my-model")

    def test_component_paths_are_expanded_before_pipeline_resolution(self):
        args = self._from_dict_without_model_resolution(
            {
                "model_path": "/data/my-model",
                "component_paths": {"vae": "~/fake/local/vae"},
            }
        )

        self.assertEqual(
            args.component_paths["vae"], os.path.expanduser("~/fake/local/vae")
        )

    def test_component_attention_backends_are_normalized(self):
        args = self._from_dict_without_model_resolution(
            {
                "model_path": "/data/my-model",
                "component_attention_backends": "text-encoder=torch_sdpa,transformer=fa3",
            }
        )

        self.assertEqual(
            args.component_attention_backends,
            {"text_encoder": "torch_sdpa", "transformer": "fa"},
        )

    def test_component_attention_backend_lookup(self):
        args = self._from_dict_without_model_resolution(
            {
                "model_path": "/data/my-model",
                "component_attention_backends": {"text_encoder": "torch_sdpa"},
            }
        )

        backend, matched_key = args.resolve_component_attention_backend(
            "text_encoder", "transformer"
        )

        self.assertEqual(backend.name, "TORCH_SDPA")
        self.assertEqual(matched_key, "text_encoder")

    def test_invalid_component_attention_backend_raises(self):
        with self.assertRaises(ValueError):
            self._from_dict_without_model_resolution(
                {
                    "model_path": "/data/my-model",
                    "component_attention_backends": {"text_encoder": "bad_backend"},
                }
            )
        with self.assertRaises(ValueError):
            self._from_dict_without_model_resolution(
                {
                    "model_path": "/data/my-model",
                    "component_attention_backends": "text_encoder",
                }
            )

    def test_dynamic_component_attention_backend_cli_args(self):
        parser = FlexibleArgumentParser()
        ServerArgs.add_cli_args(parser)
        argv = [
            "--model-path",
            "/fake",
            "--component-attention-backends.text-encoder",
            "torch_sdpa",
        ]

        with patch.object(sys, "argv", ["sglang"] + argv):
            args, unknown_args = parser.parse_known_args(argv)
            with patch.object(
                PipelineConfig, "from_kwargs", return_value=QwenImagePipelineConfig()
            ):
                server_args = ServerArgs.from_cli_args(args, unknown_args)

        self.assertEqual(
            server_args.component_attention_backends, {"text_encoder": "torch_sdpa"}
        )


class TestOffloadDefaults(unittest.TestCase):
    def _from_dict_with_task_type(
        self,
        task_type,
        *,
        memory_gb=80,
        kwargs=None,
    ):
        pipeline_config = PipelineConfig()
        pipeline_config.task_type = task_type
        with (
            patch.object(PipelineConfig, "from_kwargs", return_value=pipeline_config),
            patch(
                "sglang.multimodal_gen.runtime.server_args.current_platform.is_cpu",
                return_value=False,
            ),
            patch(
                "sglang.multimodal_gen.runtime.server_args.current_platform.get_device_total_memory",
                return_value=memory_gb * 1024**3,
            ),
        ):
            return ServerArgs.from_dict({"model_path": "/fake", **(kwargs or {})})

    def test_vae_cpu_offload_defaults_false_for_video_generation(self):
        args = self._from_dict_with_task_type(ModelTaskType.T2V)

        self.assertFalse(args.vae_cpu_offload)

    def test_vae_cpu_offload_defaults_false_on_low_memory_gpu(self):
        args = self._from_dict_with_task_type(ModelTaskType.T2V, memory_gb=16)

        self.assertFalse(args.vae_cpu_offload)
        self.assertTrue(args.dit_cpu_offload)
        self.assertTrue(args.text_encoder_cpu_offload)
        self.assertTrue(args.image_encoder_cpu_offload)

    def test_explicit_vae_cpu_offload_true_is_preserved(self):
        args = self._from_dict_with_task_type(
            ModelTaskType.T2V,
            kwargs={"vae_cpu_offload": True},
        )

        self.assertTrue(args.vae_cpu_offload)


class TestFSDPShardConditions(unittest.TestCase):
    def test_helpers_match_only_direct_block_entries(self):
        self.assertTrue(
            is_module_list_entry("transformer_blocks.0", "transformer_blocks")
        )
        self.assertFalse(
            is_module_list_entry("transformer_blocks.0.ff.net.0", "transformer_blocks")
        )
        self.assertTrue(
            is_module_list_entry_in(
                "single_transformer_blocks.12",
                ("transformer_blocks", "single_transformer_blocks"),
            )
        )
        self.assertFalse(
            is_module_list_entry_in(
                "single_transformer_blocks.12.attn.to_out.0",
                ("transformer_blocks", "single_transformer_blocks"),
            )
        )

    def test_qwen_dit_has_fsdp_shard_condition(self):
        conditions = QwenImageTransformer2DModel._fsdp_shard_conditions

        self.assertTrue(conditions)
        self.assertTrue(conditions[0]("transformer_blocks.0", None))
        self.assertFalse(conditions[0]("transformer_blocks.0.attn", None))
        self.assertFalse(conditions[0]("transformer_blocks.0.ff.net.0", None))

    def test_zimage_condition_keeps_inner_numbered_modules(self):
        self.assertTrue(is_zimage_layer("layers.0.mlp.0", None))
        self.assertTrue(is_zimage_layer("noise_refiner.0.attention.to_out.0", None))
        self.assertFalse(is_zimage_layer("transformer_blocks.0", None))


class TestModelIdResolution(unittest.TestCase):
    def setUp(self):
        _get_config_info.cache_clear()

    def test_model_id_overrides_arbitrary_local_path(self):
        # a local path whose directory name does not match any HF repo name;
        # --model-id tells the engine which config to use
        info = _get_config_info("/data/my-custom-qwen", model_id="Qwen-Image")
        self.assertIsNotNone(info)

        self.assertIs(info.pipeline_config_cls, QwenImagePipelineConfig)

    def test_model_id_works_after_tilde_expansion(self):
        # simulate the full flow: user passes ~/..., engine expands and resolves
        expanded = os.path.expanduser("~/.cache/huggingface/hub/bbb/snapshots/ccc")
        _get_config_info.cache_clear()
        info = _get_config_info(expanded, model_id="Qwen-Image")
        self.assertIsNotNone(info)

    def test_hf_cache_snapshot_path_resolves_registered_nvfp4_model(self):
        path = (
            "/root/.cache/huggingface/hub/"
            "models--black-forest-labs--FLUX.2-dev-NVFP4/"
            "snapshots/142b87e70bc3006937b7093d89ff287b5f59f071"
        )
        info = _get_config_info(path)
        self.assertIsNotNone(info)

    def test_model_id_unknown_falls_back_without_crash(self):
        # unrecognized model_id: should warn and fall back to path-based detection
        # with an unresolvable path, expect RuntimeError from the detector step
        with self.assertRaises((RuntimeError, Exception)):
            _get_config_info("/data/no-such-model", model_id="NonExistentModelXYZ")


class TestPerRoleParallelism(unittest.TestCase):
    """Test per-role parallelism args and get_role_parallelism helper."""

    def _from_dict(self, kwargs):
        with patch.object(
            PipelineConfig, "from_kwargs", return_value=QwenImagePipelineConfig()
        ):
            return ServerArgs.from_dict(kwargs)

    def test_defaults_are_none(self):
        args = self._from_dict({"model_path": "/fake"})
        from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType

        for role in [RoleType.ENCODER, RoleType.DENOISER, RoleType.DECODER]:
            par = args.get_role_parallelism(role)
            self.assertIsNone(par["tp_size"])
            self.assertIsNone(par["sp_degree"])
            self.assertIsNone(par["ulysses_degree"])
            self.assertIsNone(par["ring_degree"])

    def test_encoder_overrides(self):
        args = self._from_dict({"model_path": "/fake", "encoder_tp": 2})
        from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType

        par = args.get_role_parallelism(RoleType.ENCODER)
        self.assertEqual(par["tp_size"], 2)
        self.assertIsNone(par["sp_degree"])
        self.assertIsNone(par["ulysses_degree"])
        self.assertIsNone(par["ring_degree"])

    def test_denoiser_overrides(self):
        args = self._from_dict(
            {
                "model_path": "/fake",
                "denoiser_tp": 1,
                "denoiser_sp": 8,
                "denoiser_ulysses": 4,
                "denoiser_ring": 2,
            }
        )
        from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType

        par = args.get_role_parallelism(RoleType.DENOISER)
        self.assertEqual(par["tp_size"], 1)
        self.assertEqual(par["sp_degree"], 8)
        self.assertEqual(par["ulysses_degree"], 4)
        self.assertEqual(par["ring_degree"], 2)

    def test_decoder_overrides(self):
        args = self._from_dict({"model_path": "/fake", "decoder_tp": 2})
        from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType

        par = args.get_role_parallelism(RoleType.DECODER)
        self.assertEqual(par["tp_size"], 2)
        self.assertIsNone(par["sp_degree"])
        self.assertIsNone(par["ulysses_degree"])
        self.assertIsNone(par["ring_degree"])

    def test_monolithic_returns_all_none(self):
        args = self._from_dict({"model_path": "/fake", "encoder_tp": 2})
        from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType

        par = args.get_role_parallelism(RoleType.MONOLITHIC)
        self.assertIsNone(par["tp_size"])
        self.assertIsNone(par["sp_degree"])

    def test_mixed_roles_independent(self):
        """Per-role args don't interfere with each other."""
        args = self._from_dict(
            {
                "model_path": "/fake",
                "encoder_tp": 1,
                "denoiser_tp": 2,
                "decoder_tp": 4,
            }
        )
        from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType

        self.assertEqual(args.get_role_parallelism(RoleType.ENCODER)["tp_size"], 1)
        self.assertEqual(args.get_role_parallelism(RoleType.DENOISER)["tp_size"], 2)
        self.assertEqual(args.get_role_parallelism(RoleType.DECODER)["tp_size"], 4)

    def test_cli_args_parsed(self):
        """Per-role parallelism args are parsed from CLI."""
        parser = FlexibleArgumentParser()
        ServerArgs.add_cli_args(parser)
        argv = [
            "--model-path",
            "/fake",
            "--denoiser-tp",
            "2",
            "--denoiser-sp",
            "4",
            "--denoiser-ulysses",
            "2",
            "--denoiser-ring",
            "2",
            "--encoder-tp",
            "1",
        ]
        args, unknown = parser.parse_known_args(argv)
        self.assertEqual(args.denoiser_tp, 2)
        self.assertEqual(args.denoiser_sp, 4)
        self.assertEqual(args.denoiser_ulysses, 2)
        self.assertEqual(args.denoiser_ring, 2)
        self.assertEqual(args.encoder_tp, 1)
        self.assertIsNone(args.decoder_tp)


class TestPipelineResolutionCliOverride(unittest.TestCase):
    def setUp(self):
        _get_config_info.cache_clear()

    def test_resolution_flag_overrides_qwen_image_layered_pipeline_config(self):
        parser = FlexibleArgumentParser()
        ServerArgs.add_cli_args(parser)
        argv = [
            "--model-path",
            "Qwen/Qwen-Image-Layered",
            "--resolution",
            "768",
        ]

        with patch.object(sys, "argv", ["sglang"] + argv):
            args, unknown_args = parser.parse_known_args(argv)
            server_args = ServerArgs.from_cli_args(args, unknown_args)

        self.assertEqual(server_args.pipeline_config.resolution, 768)

    def test_disable_autocast_is_preserved_after_pipeline_config_resolution(self):
        parser = FlexibleArgumentParser()
        ServerArgs.add_cli_args(parser)
        argv = [
            "--model-path",
            "Qwen/Qwen-Image-Layered",
            "--disable-autocast",
            "true",
        ]

        with patch.object(sys, "argv", ["sglang"] + argv):
            args, unknown_args = parser.parse_known_args(argv)
            server_args = ServerArgs.from_cli_args(args, unknown_args)

        self.assertTrue(server_args.pipeline_config.disable_autocast)
        self.assertTrue(server_args.disable_autocast)


if __name__ == "__main__":
    unittest.main()
