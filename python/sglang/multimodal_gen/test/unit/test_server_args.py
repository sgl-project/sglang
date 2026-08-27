import json
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from sglang.cli.utils import get_is_diffusion_model
from sglang.multimodal_gen.configs.models.fsdp import (
    is_module_list_entry,
    is_module_list_entry_in,
    is_zimage_layer,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.cosmos3 import Cosmos3Config
from sglang.multimodal_gen.configs.pipeline_configs.hunyuan import FastHunyuanConfig
from sglang.multimodal_gen.configs.pipeline_configs.lingbot_world import (
    LingBotWorldCausalDMDConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.longcat_image import (
    LongCatImagePipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.longlive2 import (
    LongLive2T2VConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import (
    LTX2PipelineConfig,
    LTX23PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.mova import MOVAPipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImageLayeredPipelineConfig,
    QwenImagePipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.sana_wm import (
    SanaWMPipelineConfig,
    SanaWMRealtimeConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.wan import (
    FastWan2_2_TI2V_5B_Config,
    TurboWanT2V480PConfig,
    Wan2_2_I2V_A14B_Config,
    Wan2_2_T2V_A14B_Config,
    WanI2V480PConfig,
    WanI2V720PConfig,
    WanT2V480PConfig,
    WanT2V720PConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.zimage import ZImagePipelineConfig
from sglang.multimodal_gen.registry import (
    _get_config_info,
    get_non_diffusers_pipeline_name,
    is_known_non_diffusers_multimodal_model,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    COMPONENT_OFFLOAD,
    LAYERWISE_OFFLOAD,
    RESIDENT,
    normalize_component_residency,
    resolve_component_residency_mode,
    resolve_diffusers_pipeline_offload,
)
from sglang.multimodal_gen.runtime.models.dits.qwen_image import (
    QwenImageTransformer2DModel,
)
from sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline import (
    MiniMaxH3Pipeline,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import (
    MAX_SCHEDULER_RPC_TIMEOUT_S,
    ServerArgs,
)
from sglang.multimodal_gen.utils import FlexibleArgumentParser


@contextmanager
def _mock_cuda_platform(
    *,
    memory_gb: int = 80,
    available_memory_gb: int | dict[int, int] | None = None,
):
    def get_available_gpu_memory(device_id=0, **_kwargs):
        if isinstance(available_memory_gb, dict):
            return available_memory_gb[device_id]
        if available_memory_gb is not None:
            return available_memory_gb
        return memory_gb

    with (
        patch(
            "sglang.multimodal_gen.runtime.platforms.current_platform.is_cpu",
            return_value=False,
        ),
        patch(
            "sglang.multimodal_gen.runtime.platforms.current_platform.is_mps",
            return_value=False,
        ),
        patch(
            "sglang.multimodal_gen.runtime.platforms.current_platform.is_cuda",
            return_value=True,
        ),
        patch(
            "sglang.multimodal_gen.runtime.platforms.current_platform.get_device_total_memory",
            return_value=memory_gb * 1024**3,
        ),
        patch(
            "sglang.multimodal_gen.runtime.platforms.current_platform.get_available_gpu_memory",
            side_effect=get_available_gpu_memory,
        ),
        patch(
            "sglang.multimodal_gen.runtime.platforms.current_platform.enable_dit_layerwise_offload_by_default",
            return_value=True,
        ),
    ):
        yield


def _from_dict_without_model_resolution(
    kwargs, pipeline_config: PipelineConfig | None = None
):
    pipeline_config = pipeline_config or QwenImagePipelineConfig()
    with (
        patch.object(PipelineConfig, "from_kwargs", return_value=pipeline_config),
        _mock_cuda_platform(),
    ):
        return ServerArgs.from_dict(kwargs)


class TestServerArgsPathExpansion(unittest.TestCase):
    def _from_dict_without_model_resolution(self, kwargs):
        return _from_dict_without_model_resolution(kwargs)

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

    def test_component_weight_file_keeps_base_component_config(self):
        args = self._from_dict_without_model_resolution(
            {
                "model_path": "/data/my-model",
                "component_paths": {
                    "text_encoder": "owner/repo/text_encoder/model.safetensors",
                    "audio_vae": "owner/repo/vae/audio.safetensors",
                    "vae": "owner/repo/vae",
                },
            }
        )

        self.assertEqual(args.component_paths, {"vae": "owner/repo/vae"})
        self.assertEqual(
            args.component_weights_paths,
            {
                "text_encoder": "owner/repo/text_encoder/model.safetensors",
                "audio_vae": "owner/repo/vae/audio.safetensors",
            },
        )

    def test_supplemental_weight_file_remains_a_component_path(self):
        args = self._from_dict_without_model_resolution(
            {
                "model_path": "/data/my-model",
                "component_paths": {
                    "conditioning_projection": "owner/repo/projection.safetensors"
                },
            }
        )

        self.assertEqual(
            args.component_paths,
            {"conditioning_projection": "owner/repo/projection.safetensors"},
        )
        self.assertEqual(args.component_weights_paths, {})

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

        with (
            patch.object(sys, "argv", ["sglang"] + argv),
            patch.object(
                PipelineConfig, "from_kwargs", return_value=QwenImagePipelineConfig()
            ),
            patch(
                "sglang.multimodal_gen.runtime.platforms.current_platform.is_cpu",
                return_value=False,
            ),
            patch(
                "sglang.multimodal_gen.runtime.platforms.current_platform.is_mps",
                return_value=False,
            ),
            patch(
                "sglang.multimodal_gen.runtime.platforms.current_platform.is_cuda",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.platforms.current_platform.get_device_total_memory",
                return_value=80 * 1024**3,
            ),
            patch(
                "sglang.multimodal_gen.runtime.platforms.current_platform.get_available_gpu_memory",
                return_value=80,
            ),
        ):
            args, unknown_args = parser.parse_known_args(argv)
            server_args = ServerArgs.from_cli_args(args, unknown_args)

        self.assertEqual(
            server_args.component_attention_backends, {"text_encoder": "torch_sdpa"}
        )

    def test_layerwise_offload_components_imply_layerwise(self):
        args = self._from_dict_without_model_resolution(
            {
                "model_path": "/data/my-model",
                "performance_mode": "manual",
            }
        )
        args.layerwise_offload_components = ["text_encoder", "transformer"]
        args._adjust_layerwise_offload_components()

        self.assertTrue(args.layerwise_offload_components)
        self.assertEqual(
            args.layerwise_offload_components, ["text_encoder", "transformer"]
        )

    def test_dit_layerwise_offload_selects_dit_group(self):
        args = self._from_dict_without_model_resolution(
            {
                "model_path": "/data/my-model",
                "performance_mode": "manual",
                "dit_layerwise_offload": True,
            }
        )

        self.assertTrue(args.layerwise_offload_components)
        self.assertEqual(args.layerwise_offload_components, ["dit"])

    def test_dit_layerwise_offload_from_kwargs(self):
        with patch.object(
            PipelineConfig, "from_kwargs", return_value=QwenImagePipelineConfig()
        ):
            args = ServerArgs.from_kwargs(
                model_path="/data/my-model",
                performance_mode="manual",
                dit_layerwise_offload=True,
            )

        self.assertTrue(args.layerwise_offload_components)
        self.assertEqual(args.layerwise_offload_components, ["dit"])

    def test_layerwise_offload_components_normalize_commas(self):
        args = self._from_dict_without_model_resolution(
            {
                "model_path": "/data/my-model",
                "performance_mode": "manual",
            }
        )
        args.layerwise_offload_components = ["text-encoder,transformer"]
        args._adjust_layerwise_offload_components()

        self.assertEqual(
            args.layerwise_offload_components, ["text_encoder", "transformer"]
        )

    def test_layerwise_offload_components_normalize_default_group(self):
        args = self._from_dict_without_model_resolution(
            {
                "model_path": "/data/my-model",
                "performance_mode": "manual",
            }
        )
        args.layerwise_offload_components = ["default", "text_encoder"]
        args._adjust_layerwise_offload_components()

        self.assertEqual(
            args.layerwise_offload_components,
            ["text_encoder", "image_encoder", "vae"],
        )

    def test_served_model_name_cli_arg(self):
        parser = FlexibleArgumentParser()
        ServerArgs.add_cli_args(parser)
        cases = [
            (
                [
                    "--model-path",
                    "/fake",
                    "--model-id",
                    "Qwen-Image",
                    "--served-model-name",
                    "my-served-name",
                ],
                "my-served-name",
            ),
            (
                ["--model-path", "/fake", "--model-id", "Qwen-Image"],
                "Qwen-Image",
            ),
            (["--model-path", "/fake"], "/fake"),
        ]

        for argv, expected in cases:
            with self.subTest(argv=argv):
                with patch.object(sys, "argv", ["sglang"] + argv):
                    args, unknown_args = parser.parse_known_args(argv)
                    with patch.object(
                        PipelineConfig,
                        "from_kwargs",
                        return_value=QwenImagePipelineConfig(),
                    ):
                        server_args = ServerArgs.from_cli_args(args, unknown_args)

                self.assertEqual(server_args.served_model_name, expected)

    def test_dit_layerwise_offload_cli_arg(self):
        parser = FlexibleArgumentParser()
        ServerArgs.add_cli_args(parser)
        argv = [
            "--model-path",
            "/fake",
            "--performance-mode",
            "manual",
            "--dit-layerwise-offload",
            "true",
        ]

        with patch.object(sys, "argv", ["sglang"] + argv):
            args, unknown_args = parser.parse_known_args(argv)
            with patch.object(
                PipelineConfig, "from_kwargs", return_value=QwenImagePipelineConfig()
            ):
                server_args = ServerArgs.from_cli_args(args, unknown_args)

        self.assertTrue(server_args.layerwise_offload_components)
        self.assertEqual(server_args.layerwise_offload_components, ["dit"])

    def test_layerwise_offload_components_cli_args(self):
        parser = FlexibleArgumentParser()
        ServerArgs.add_cli_args(parser)
        argv = [
            "--model-path",
            "/fake",
            "--performance-mode",
            "manual",
            "--layerwise-offload-components",
            "transformer",
            "text_encoder",
        ]

        with patch.object(sys, "argv", ["sglang"] + argv):
            args, unknown_args = parser.parse_known_args(argv)
            with patch.object(
                PipelineConfig, "from_kwargs", return_value=QwenImagePipelineConfig()
            ):
                server_args = ServerArgs.from_cli_args(args, unknown_args)

        self.assertEqual(
            server_args.layerwise_offload_components, ["transformer", "text_encoder"]
        )

    def test_cpu_offload_components_cli_args(self):
        parser = FlexibleArgumentParser()
        ServerArgs.add_cli_args(parser)
        argv = [
            "--model-path",
            "/fake",
            "--performance-mode",
            "manual",
            "--cpu-offload-components",
            "transformer",
            "vae",
        ]

        with patch.object(sys, "argv", ["sglang"] + argv):
            args, unknown_args = parser.parse_known_args(argv)
            with patch.object(
                PipelineConfig, "from_kwargs", return_value=QwenImagePipelineConfig()
            ):
                server_args = ServerArgs.from_cli_args(args, unknown_args)

        self.assertEqual(server_args.cpu_offload_components, ["transformer", "vae"])
        self.assertEqual(server_args.residency_mode("transformer"), COMPONENT_OFFLOAD)
        self.assertEqual(server_args.residency_mode("vae"), COMPONENT_OFFLOAD)

    def test_serve_cli_preserves_config_and_dynamic_unknown_args(self):
        from sglang.multimodal_gen.runtime.entrypoints.cli.serve import (
            add_multimodal_gen_serve_args,
        )

        with tempfile.NamedTemporaryFile("w", suffix=".json") as config_file:
            json.dump(
                {
                    "model_path": "/from/config",
                    "num_gpus": 2,
                    "component_weights_paths": {
                        "transformer": "owner/repo/transformer.safetensors"
                    },
                },
                config_file,
            )
            config_file.flush()
            parser = FlexibleArgumentParser()
            add_multimodal_gen_serve_args(parser)
            argv = [
                "--config",
                config_file.name,
                "--model-path",
                "/from/cli",
                "--vae-path",
                "/custom/vae",
                "--component-weights-paths.text_encoder",
                "owner/repo/text_encoder.safetensors",
                "--image-encoder-weights-path=/custom/image_encoder.safetensors",
                "--component-quantizations.text_encoder",
                "kitchen_int8",
                "--component-quantization-ignored-layers.text_encoder",
                "model.layers.0",
                "lm_head",
                "--transformer-quantization=fp8",
                "--component-attention-backends.transformer",
                "fa3",
            ]

            with patch.object(sys, "argv", ["sglang", "serve"] + argv):
                args, unknown_args = parser.parse_known_args(argv)
                with (
                    patch.object(
                        PipelineConfig,
                        "from_kwargs",
                        return_value=QwenImagePipelineConfig(),
                    ),
                    patch(
                        "sglang.multimodal_gen.registry.get_model_info",
                        return_value=None,
                    ),
                    patch(
                        "sglang.multimodal_gen.runtime.platforms.current_platform.get_device_total_memory",
                        return_value=80 * 1024**3,
                    ),
                    patch(
                        "sglang.multimodal_gen.runtime.platforms.current_platform.get_available_gpu_memory",
                        return_value=80,
                    ),
                ):
                    server_args = ServerArgs.from_cli_args(args, unknown_args)

        self.assertEqual("/from/cli", server_args.model_path)
        self.assertEqual(2, server_args.num_gpus)
        self.assertEqual("/custom/vae", server_args.component_paths["vae"])
        self.assertEqual(
            {
                "transformer": "owner/repo/transformer.safetensors",
                "text_encoder": "owner/repo/text_encoder.safetensors",
                "image_encoder": "/custom/image_encoder.safetensors",
            },
            server_args.component_weights_paths,
        )
        self.assertEqual(
            {"transformer": "fa"},
            server_args.component_attention_backends,
        )
        self.assertEqual(
            {"text_encoder": "kitchen_int8", "transformer": "fp8"},
            server_args.component_quantizations,
        )
        self.assertEqual(
            {"text_encoder": ["model.layers.0", "lm_head"]},
            server_args.component_quantization_ignored_layers,
        )

    def test_serve_cli_defaults_warmup_on(self):
        from sglang.multimodal_gen.runtime.entrypoints.cli.serve import (
            add_multimodal_gen_serve_args,
            execute_serve_cmd,
        )

        parser = FlexibleArgumentParser()
        add_multimodal_gen_serve_args(parser)
        argv = [
            "--model-path",
            "/fake",
        ]

        with (
            patch.object(sys, "argv", ["sglang", "serve"] + argv),
            patch.object(
                PipelineConfig, "from_kwargs", return_value=QwenImagePipelineConfig()
            ),
            patch(
                "sglang.multimodal_gen.runtime.entrypoints.cli.serve.dispatch_launch"
            ) as dispatch_launch,
        ):
            args, unknown_args = parser.parse_known_args(argv)
            execute_serve_cmd(args, unknown_args)

        server_args = dispatch_launch.call_args.args[0]
        self.assertEqual(server_args.warmup_mode, "server")
        self.assertFalse(server_args.is_arg_explicitly_set("warmup_mode"))

    def test_serve_cli_preserves_explicit_warmup_mode_off(self):
        from sglang.multimodal_gen.runtime.entrypoints.cli.serve import (
            add_multimodal_gen_serve_args,
            execute_serve_cmd,
        )

        parser = FlexibleArgumentParser()
        add_multimodal_gen_serve_args(parser)
        argv = [
            "--model-path",
            "/fake",
            "--warmup-mode",
            "off",
        ]

        with (
            patch.object(sys, "argv", ["sglang", "serve"] + argv),
            patch.object(
                PipelineConfig, "from_kwargs", return_value=QwenImagePipelineConfig()
            ),
            patch(
                "sglang.multimodal_gen.runtime.entrypoints.cli.serve.dispatch_launch"
            ) as dispatch_launch,
        ):
            args, unknown_args = parser.parse_known_args(argv)
            execute_serve_cmd(args, unknown_args)

        server_args = dispatch_launch.call_args.args[0]
        self.assertEqual(server_args.warmup_mode, "off")
        self.assertTrue(server_args.is_arg_explicitly_set("warmup_mode"))

    def test_serve_cli_preserves_config_warmup_mode_off(self):
        from sglang.multimodal_gen.runtime.entrypoints.cli.serve import (
            add_multimodal_gen_serve_args,
            execute_serve_cmd,
        )

        with tempfile.NamedTemporaryFile("w", suffix=".json") as config_file:
            json.dump({"model_path": "/fake", "warmup_mode": "off"}, config_file)
            config_file.flush()

            parser = FlexibleArgumentParser()
            add_multimodal_gen_serve_args(parser)
            argv = [
                "--config",
                config_file.name,
            ]

            with (
                patch.object(sys, "argv", ["sglang", "serve"] + argv),
                patch.object(
                    PipelineConfig,
                    "from_kwargs",
                    return_value=QwenImagePipelineConfig(),
                ),
                patch(
                    "sglang.multimodal_gen.runtime.entrypoints.cli.serve.dispatch_launch"
                ) as dispatch_launch,
            ):
                args, unknown_args = parser.parse_known_args(argv)
                execute_serve_cmd(args, unknown_args)

        server_args = dispatch_launch.call_args.args[0]
        self.assertEqual(server_args.warmup_mode, "off")
        self.assertTrue(server_args.is_arg_explicitly_set("warmup_mode"))

    def test_retired_warmup_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "warmup.*warmup_mode"):
            _from_dict_without_model_resolution(
                {"model_path": "/fake", "warmup": False}
            )

    def test_retired_warmup_kwargs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "warmup.*warmup_mode"):
            ServerArgs.from_kwargs(model_path="/fake", warmup=False)

    def test_disagg_role_disables_server_warmup(self):
        with patch.object(
            PipelineConfig, "from_kwargs", return_value=QwenImagePipelineConfig()
        ):
            server_args = ServerArgs.from_dict(
                {
                    "model_path": "/fake",
                    "warmup_mode": "server",
                    "disagg_role": "server",
                }
            )

        self.assertEqual(server_args.warmup_mode, "request")


class TestWarmupModeNormalization(unittest.TestCase):
    """`_adjust_warmup` resolves the canonical warmup mode."""

    def _resolve(
        self,
        *,
        warmup_mode=None,
        warmup_resolutions=None,
        warmup_num_frames=None,
        enable_torch_compile=False,
        enable_breakable_cuda_graph=False,
        disagg_role=None,
    ):
        from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType

        sa = ServerArgs.__new__(ServerArgs)
        sa.warmup_mode = warmup_mode
        sa.warmup_resolutions = warmup_resolutions
        sa.warmup_num_frames = warmup_num_frames
        sa.enable_torch_compile = enable_torch_compile
        sa.enable_breakable_cuda_graph = enable_breakable_cuda_graph
        sa.disagg_role = RoleType.MONOLITHIC if disagg_role is None else disagg_role
        sa._adjust_warmup()
        return sa

    def test_explicit_mode_off_disables_all(self):
        sa = self._resolve(warmup_mode="off")
        self.assertEqual(sa.warmup_mode, "off")

    def test_explicit_mode_request(self):
        sa = self._resolve(warmup_mode="request")
        self.assertEqual(sa.warmup_mode, "request")

    def test_explicit_mode_server(self):
        sa = self._resolve(warmup_mode="server")
        self.assertEqual(sa.warmup_mode, "server")

    def test_defaulted_mode_applies_without_legacy_flags(self):
        # Bare `sglang serve` defaults to server-based warmup.
        sa = self._resolve(warmup_mode="server")
        self.assertEqual(sa.warmup_mode, "server")

    def test_resolutions_force_warmup_on(self):
        sa = self._resolve(
            warmup_mode="off",
            warmup_resolutions=["512x512"],
        )
        self.assertEqual(sa.warmup_mode, "request")

    def test_num_frames_forces_warmup_on(self):
        sa = self._resolve(warmup_mode="off", warmup_num_frames=17)
        self.assertEqual(sa.warmup_mode, "request")

    def test_num_frames_must_be_positive(self):
        for num_frames in (0, -1):
            with self.subTest(num_frames=num_frames):
                with self.assertRaisesRegex(ValueError, "positive"):
                    self._resolve(warmup_num_frames=num_frames)

    def test_torch_compile_defaults_to_server_warmup(self):
        sa = self._resolve(enable_torch_compile=True)

        self.assertEqual(sa.warmup_mode, "server")

    def test_torch_compile_respects_explicit_warmup_off(self):
        sa = self._resolve(
            warmup_mode="off",
            enable_torch_compile=True,
        )
        self.assertEqual(sa.warmup_mode, "off")

    def test_torch_compile_uses_server_warmup_for_explicit_resolutions(self):
        sa = self._resolve(
            warmup_resolutions=["1024x1024"],
            enable_torch_compile=True,
        )
        self.assertEqual(sa.warmup_mode, "server")

    def test_breakable_cuda_graph_forces_server_warmup(self):
        sa = self._resolve(enable_breakable_cuda_graph=True)
        self.assertEqual(sa.warmup_mode, "server")

    def test_breakable_cuda_graph_allows_unset_resolutions(self):
        # BCG no longer hard-requires --warmup-resolutions; the model
        # default warmup resolution is captured at warmup instead.
        sa = ServerArgs.__new__(ServerArgs)
        sa.enable_breakable_cuda_graph = True
        sa.warmup_resolutions = None
        sa.bcg_text_buckets = None
        sa._validate_breakable_cuda_graph()  # must not raise

    def test_disagg_role_disables_server_warmup(self):
        from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType

        sa = self._resolve(
            warmup_mode="server",
            disagg_role=RoleType.DENOISER,
        )
        self.assertEqual(sa.warmup_mode, "request")

    def test_torch_compile_server_warmup_disabled_for_disagg_role(self):
        from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType

        sa = self._resolve(enable_torch_compile=True, disagg_role=RoleType.DENOISER)
        self.assertEqual(sa.warmup_mode, "request")

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            self._resolve(warmup_mode="bogus")


class TestWarmupImageIsModelValid(unittest.TestCase):
    """The server-warmup placeholder image must be large enough for real pipelines."""

    def test_minimum_warmup_image_is_at_least_64px(self):
        import base64
        import struct

        from sglang.multimodal_gen.runtime.server_warmup import (
            MINIMUM_PICTURE_BASE64_FOR_WARMUP,
        )

        payload = MINIMUM_PICTURE_BASE64_FOR_WARMUP.split(",", 1)[-1]
        raw = base64.b64decode(payload)
        self.assertEqual(raw[:8], b"\x89PNG\r\n\x1a\n")
        # IHDR width/height are the two big-endian uint32 after the chunk header.
        width, height = struct.unpack(">II", raw[16:24])
        self.assertGreaterEqual(width, 64)
        self.assertGreaterEqual(height, 64)


class TestDiffusionModelDetection(unittest.TestCase):
    def test_registered_local_model_path_is_detected_as_diffusion(self):
        with tempfile.TemporaryDirectory() as root:
            model_path = os.path.join(root, "Z-Image-Turbo")
            os.mkdir(model_path)
            self.assertTrue(get_is_diffusion_model(model_path))


class TestMiniMaxH3Routing(unittest.TestCase):

    def test_semantic_variants_map_to_checkpoint_partitions(self):
        self.assertEqual(
            MiniMaxH3Pipeline.model_subfolder_for_variant("fl2va"), "FL2VA"
        )
        self.assertEqual(
            MiniMaxH3Pipeline.model_subfolder_for_variant("ref2va"), "Ref2VA"
        )
        with self.assertRaisesRegex(ValueError, "unsupported MiniMax H3 model variant"):
            MiniMaxH3Pipeline.model_subfolder_for_variant("v2v")

    def test_modelscope_id_resolves_to_the_huggingface_config(self):
        expected = _get_config_info("MiniMaxAI/MiniMax-H3")
        actual = _get_config_info("MiniMax/MiniMax-H3")
        self.assertIsNotNone(expected)
        self.assertIs(actual, expected)
        self.assertTrue(is_known_non_diffusers_multimodal_model("MiniMax/MiniMax-H3"))
        self.assertEqual(
            get_non_diffusers_pipeline_name("MiniMax/MiniMax-H3"),
            "MiniMaxH3Pipeline",
        )
        self.assertEqual(
            get_non_diffusers_pipeline_name("/models/MiniMax-H3"),
            "MiniMaxH3Pipeline",
        )


class TestOffloadDefaults(unittest.TestCase):
    def test_wan_decode_precision_defaults(self):
        for pipeline_config in (
            WanT2V480PConfig(),
            WanI2V480PConfig(),
        ):
            with self.subTest(pipeline_config=pipeline_config.__class__.__name__):
                self.assertEqual(pipeline_config.vae_precision, "fp32")
                self.assertEqual(pipeline_config.vae_decode_precision, "bf16")

        for pipeline_config in (
            FastWan2_2_TI2V_5B_Config(),
            Wan2_2_T2V_A14B_Config(),
            Wan2_2_I2V_A14B_Config(),
        ):
            with self.subTest(pipeline_config=pipeline_config.__class__.__name__):
                self.assertEqual(pipeline_config.vae_precision, "fp32")
                self.assertEqual(pipeline_config.vae_decode_precision, "fp32")

        generic_config = PipelineConfig()
        self.assertEqual(generic_config.vae_precision, "fp32")
        self.assertIsNone(generic_config.vae_decode_precision)

    def _from_dict_with_pipeline_config(
        self,
        pipeline_config,
        *,
        memory_gb=80,
        available_memory_gb=None,
        kwargs=None,
    ):
        def get_available_gpu_memory(device_id=0, **_kwargs):
            if isinstance(available_memory_gb, dict):
                return available_memory_gb[device_id]
            if available_memory_gb is not None:
                return available_memory_gb
            return memory_gb

        with (
            patch.object(PipelineConfig, "from_kwargs", return_value=pipeline_config),
            patch(
                "sglang.multimodal_gen.runtime.platforms.current_platform.is_cpu",
                return_value=False,
            ),
            patch(
                "sglang.multimodal_gen.runtime.platforms.current_platform.is_mps",
                return_value=False,
            ),
            patch(
                "sglang.multimodal_gen.runtime.platforms.current_platform.is_cuda",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.platforms.current_platform.enable_dit_layerwise_offload_by_default",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.platforms.current_platform.get_device_total_memory",
                return_value=memory_gb * 1024**3,
            ),
            patch(
                "sglang.multimodal_gen.runtime.platforms.current_platform.get_available_gpu_memory",
                side_effect=get_available_gpu_memory,
            ),
        ):
            return ServerArgs.from_dict({"model_path": "/fake", **(kwargs or {})})

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
                "sglang.multimodal_gen.runtime.platforms.current_platform.is_cpu",
                return_value=False,
            ),
            patch(
                "sglang.multimodal_gen.runtime.platforms.current_platform.is_cuda",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.platforms.current_platform.get_device_total_memory",
                return_value=memory_gb * 1024**3,
            ),
            patch(
                "sglang.multimodal_gen.runtime.platforms.current_platform.get_available_gpu_memory",
                return_value=memory_gb,
            ),
        ):
            return ServerArgs.from_dict({"model_path": "/fake", **(kwargs or {})})

    def test_vae_cpu_offload_defaults_false_for_video_generation(self):
        args = self._from_dict_with_task_type(ModelTaskType.T2V)

        self.assertFalse(args.vae_cpu_offload)

    def test_cpu_offload_components_preserves_model_index_names(self):
        args = self._from_dict_with_task_type(
            ModelTaskType.T2V,
            kwargs={
                "performance_mode": "manual",
                "cpu_offload_components": [
                    "transformer_2",
                    "audio_vae",
                    "connectors",
                ],
            },
        )

        self.assertEqual(
            args.cpu_offload_components,
            ["transformer_2", "audio_vae", "connectors"],
        )
        self.assertTrue(args.should_cpu_offload_component("transformer_2"))
        self.assertTrue(args.should_cpu_offload_component("audio_vae"))
        self.assertTrue(args.should_cpu_offload_component("connectors"))
        self.assertEqual(args.residency_mode("transformer"), RESIDENT)
        self.assertEqual(args.residency_mode("vae"), RESIDENT)

    def test_cpu_offload_components_all_matches_dynamic_components(self):
        args = self._from_dict_with_task_type(
            ModelTaskType.T2V,
            kwargs={
                "performance_mode": "manual",
                "cpu_offload_components": ["all"],
            },
        )

        self.assertEqual(args.cpu_offload_components, ["all"])
        self.assertTrue(args.should_cpu_offload_component("transformer_2"))
        self.assertTrue(args.should_cpu_offload_component("connectors"))

    def test_cpu_offload_components_none_disables_all_legacy_flags(self):
        args = self._from_dict_with_task_type(
            ModelTaskType.T2V,
            kwargs={
                "performance_mode": "manual",
                "cpu_offload_components": ["none"],
            },
        )

        self.assertEqual(args.cpu_offload_components, [])
        self.assertEqual(args.residency_mode("transformer"), RESIDENT)
        self.assertEqual(args.residency_mode("text_encoder"), RESIDENT)
        self.assertEqual(args.residency_mode("image_encoder"), RESIDENT)
        self.assertEqual(args.residency_mode("vae"), RESIDENT)

        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            self._from_dict_with_task_type(
                ModelTaskType.T2V,
                kwargs={"cpu_offload_components": ["none", "vae"]},
            )

    def test_cpu_offload_components_can_mix_with_legacy_flags(self):
        args = self._from_dict_with_task_type(
            ModelTaskType.T2V,
            kwargs={
                "performance_mode": "manual",
                "cpu_offload_components": ["vae"],
                "dit_cpu_offload": True,
                "text_encoder_cpu_offload": False,
            },
        )

        self.assertEqual(args.residency_mode("vae"), COMPONENT_OFFLOAD)
        self.assertEqual(args.residency_mode("transformer"), COMPONENT_OFFLOAD)
        self.assertEqual(args.residency_mode("text_encoder"), RESIDENT)

    def test_legacy_component_cpu_offload_flags_remain_supported(self):
        cases = (
            ("dit_cpu_offload", "transformer_2"),
            ("text_encoder_cpu_offload", "text_encoder_3"),
            ("image_encoder_cpu_offload", "image_encoder"),
            ("vae_cpu_offload", "audio_vae"),
        )
        for flag_name, component_name in cases:
            with self.subTest(flag_name=flag_name, enabled=True):
                args = self._from_dict_with_task_type(
                    ModelTaskType.T2V,
                    kwargs={"performance_mode": "manual", flag_name: True},
                )
                self.assertEqual(args.residency_mode(component_name), COMPONENT_OFFLOAD)
            with self.subTest(flag_name=flag_name, enabled=False):
                args = self._from_dict_with_task_type(
                    ModelTaskType.T2V,
                    kwargs={"performance_mode": "manual", flag_name: False},
                )
                self.assertEqual(args.residency_mode(component_name), RESIDENT)

    def test_component_residency_normalizes_assignments(self):
        self.assertEqual(
            normalize_component_residency(
                [
                    "all=resident,dit=layerwise_offload",
                    "transformer_2=component-offload",
                ]
            ),
            {
                "all": RESIDENT,
                "dit": LAYERWISE_OFFLOAD,
                "transformer_2": COMPONENT_OFFLOAD,
            },
        )
        with self.assertRaisesRegex(ValueError, "COMPONENT=MODE"):
            normalize_component_residency(["dit"])
        with self.assertRaisesRegex(ValueError, "Invalid component residency mode"):
            normalize_component_residency(["dit=cpu"])

    def test_component_residency_resolves_exact_group_and_all_precedence(self):
        assignments = normalize_component_residency(
            [
                "all=component-offload",
                "dit=layerwise-offload",
                "transformer_2=resident",
                "connectors=resident",
            ]
        )

        self.assertEqual(
            resolve_component_residency_mode("transformer", assignments),
            LAYERWISE_OFFLOAD,
        )
        self.assertEqual(
            resolve_component_residency_mode("transformer_2", assignments),
            RESIDENT,
        )
        self.assertEqual(
            resolve_component_residency_mode("text_encoder_2", assignments),
            COMPONENT_OFFLOAD,
        )
        self.assertEqual(
            resolve_component_residency_mode("connectors", assignments), RESIDENT
        )

    def test_component_residency_groups_exclude_legacy_helpers(self):
        assignments = normalize_component_residency(
            ["dit=component-offload", "vae=component-offload"]
        )

        for component_name in (
            "connectors",
            "dual_tower_bridge",
            "vision_language_encoder",
            "condition_image_encoder",
            "sound_tokenizer",
            "spatial_upsampler",
            "vocoder",
        ):
            with self.subTest(component_name=component_name):
                self.assertIsNone(
                    resolve_component_residency_mode(component_name, assignments)
                )

    def test_component_residency_groups_include_native_texture_models(self):
        assignments = normalize_component_residency(
            ["dit=layerwise-offload", "vae=component-offload"]
        )

        for component_name in ("paint_transformer", "delight_transformer"):
            with self.subTest(component_name=component_name):
                self.assertEqual(
                    resolve_component_residency_mode(component_name, assignments),
                    LAYERWISE_OFFLOAD,
                )
        for component_name in ("paint_vae", "delight_vae"):
            with self.subTest(component_name=component_name):
                self.assertEqual(
                    resolve_component_residency_mode(component_name, assignments),
                    COMPONENT_OFFLOAD,
                )

    def test_component_residency_overrides_matching_legacy_flags_only(self):
        args = self._from_dict_with_task_type(
            ModelTaskType.T2V,
            kwargs={
                "performance_mode": "manual",
                "component_residency": ["dit=resident"],
                "dit_cpu_offload": True,
                "text_encoder_cpu_offload": True,
            },
        )

        self.assertEqual(args.residency_mode("transformer"), RESIDENT)
        self.assertEqual(args.residency_mode("text_encoder"), COMPONENT_OFFLOAD)
        self.assertTrue(args.dit_cpu_offload)
        self.assertTrue(args.text_encoder_cpu_offload)

    def test_explicit_false_layerwise_keeps_dit_resident(self):
        args = self._from_dict_with_pipeline_config(
            QwenImagePipelineConfig(),
            kwargs={
                "model_path": "Qwen/Qwen-Image",
                "dit_layerwise_offload": False,
            },
        )

        self.assertEqual(args.residency_mode("transformer"), RESIDENT)

    def test_explicit_false_layerwise_preserves_other_explicit_dit_mode(self):
        for kwargs, expected_mode in (
            (
                {
                    "dit_layerwise_offload": False,
                    "dit_cpu_offload": True,
                },
                COMPONENT_OFFLOAD,
            ),
            (
                {
                    "dit_layerwise_offload": False,
                    "layerwise_offload_components": ["dit"],
                },
                LAYERWISE_OFFLOAD,
            ),
        ):
            with self.subTest(expected_mode=expected_mode):
                args = self._from_dict_with_pipeline_config(
                    QwenImagePipelineConfig(),
                    kwargs={"model_path": "Qwen/Qwen-Image", **kwargs},
                )
                self.assertEqual(args.residency_mode("transformer"), expected_mode)

    def test_exact_canonical_residency_preserves_unmatched_legacy_dit_scope(self):
        args = self._from_dict_with_pipeline_config(
            QwenImagePipelineConfig(),
            kwargs={
                "model_path": "Qwen/Qwen-Image",
                "component_residency": ["transformer=resident"],
                "dit_layerwise_offload": False,
                "dit_cpu_offload": True,
            },
        )

        self.assertEqual(args.residency_mode("transformer"), RESIDENT)
        self.assertEqual(args.residency_mode("transformer_2"), COMPONENT_OFFLOAD)
        self.assertEqual(args.residency_mode("connectors"), COMPONENT_OFFLOAD)

    def test_explicit_layerwise_takes_precedence_over_legacy_cpu_offload(self):
        args = self._from_dict_with_pipeline_config(
            QwenImagePipelineConfig(),
            kwargs={
                "model_path": "Qwen/Qwen-Image",
                "performance_mode": "manual",
                "dit_cpu_offload": True,
                "dit_layerwise_offload": True,
            },
        )

        self.assertTrue(args.dit_cpu_offload)
        self.assertEqual(args.residency_mode("transformer"), LAYERWISE_OFFLOAD)
        self.assertEqual(args.residency_mode("connectors"), COMPONENT_OFFLOAD)

    def test_component_residency_overrides_only_matching_legacy_components(self):
        args = self._from_dict_with_task_type(
            ModelTaskType.T2V,
            kwargs={
                "performance_mode": "manual",
                "component_residency": ["text_encoder=resident"],
                "dit_cpu_offload": True,
                "text_encoder_cpu_offload": True,
                "image_encoder_cpu_offload": True,
                "vae_cpu_offload": True,
            },
        )

        self.assertEqual(args.residency_mode("text_encoder"), RESIDENT)
        self.assertEqual(args.residency_mode("transformer"), COMPONENT_OFFLOAD)
        self.assertEqual(args.residency_mode("image_encoder"), COMPONENT_OFFLOAD)
        self.assertEqual(args.residency_mode("vae"), COMPONENT_OFFLOAD)

    def test_component_residency_fsdp_decision_is_component_scoped(self):
        args = self._from_dict_with_task_type(
            ModelTaskType.T2V,
            kwargs={
                "performance_mode": "manual",
                "use_fsdp_inference": True,
                "component_residency": [
                    "transformer=layerwise-offload",
                    "text_encoder=resident",
                ],
            },
        )

        self.assertFalse(args.should_use_fsdp_for_component("transformer"))
        self.assertTrue(args.should_use_fsdp_for_component("text_encoder"))

        args.disable_fsdp_for_component("text_encoder")
        self.assertFalse(args.should_use_fsdp_for_component("text_encoder"))

    def test_diffusers_component_residency_is_pipeline_wide(self):
        self.assertFalse(resolve_diffusers_pipeline_offload({"all": RESIDENT}))
        self.assertTrue(resolve_diffusers_pipeline_offload({"all": COMPONENT_OFFLOAD}))
        with self.assertRaisesRegex(ValueError, "pipeline-wide"):
            resolve_diffusers_pipeline_offload({"dit": COMPONENT_OFFLOAD})
        with self.assertRaisesRegex(ValueError, "native SGLang backend"):
            resolve_diffusers_pipeline_offload({"all": LAYERWISE_OFFLOAD})

    def test_memory_mode_layerwise_offloads_vae_on_low_memory_gpu(self):
        args = self._from_dict_with_task_type(
            ModelTaskType.T2V,
            memory_gb=16,
            kwargs={"performance_mode": "memory"},
        )

        self.assertFalse(args.vae_cpu_offload)
        self.assertTrue(args.dit_cpu_offload)
        self.assertFalse(args.text_encoder_cpu_offload)
        self.assertFalse(args.image_encoder_cpu_offload)
        self.assertEqual(
            args.layerwise_offload_components,
            ["text_encoder", "image_encoder", "vae"],
        )
        self.assertEqual(args.residency_mode("vae"), LAYERWISE_OFFLOAD)

    def test_memory_mode_preserves_explicit_vae_residency(self):
        for kwargs, expected_mode in (
            ({"component_residency": ["vae=resident"]}, RESIDENT),
            ({"vae_cpu_offload": True}, COMPONENT_OFFLOAD),
        ):
            with self.subTest(expected_mode=expected_mode):
                args = self._from_dict_with_task_type(
                    ModelTaskType.T2V,
                    memory_gb=16,
                    kwargs={"performance_mode": "memory", **kwargs},
                )

                self.assertNotIn("vae", args.layerwise_offload_components or [])
                self.assertEqual(args.residency_mode("vae"), expected_mode)

    def test_explicit_vae_cpu_offload_true_is_preserved_by_default_layerwise(
        self,
    ):
        args = self._from_dict_with_task_type(
            ModelTaskType.T2V,
            kwargs={"vae_cpu_offload": True},
        )

        self.assertTrue(args.vae_cpu_offload)
        self.assertEqual(
            args.layerwise_offload_components, ["text_encoder", "image_encoder"]
        )

    def test_explicit_component_resident_is_preserved_by_default_layerwise(self):
        args = self._from_dict_with_task_type(
            ModelTaskType.T2V,
            kwargs={"text_encoder_cpu_offload": False},
        )

        self.assertFalse(args.text_encoder_cpu_offload)
        self.assertEqual(args.layerwise_offload_components, ["image_encoder", "vae"])

    def test_layerwise_components_override_matching_cpu_offload_modes(self):
        args = self._from_dict_with_task_type(
            ModelTaskType.T2V,
            memory_gb=16,
            kwargs={
                "performance_mode": "manual",
                "dit_cpu_offload": True,
                "text_encoder_cpu_offload": True,
                "image_encoder_cpu_offload": True,
                "vae_cpu_offload": True,
                "layerwise_offload_components": [
                    "text_encoder",
                    "image_encoder",
                    "video_dit",
                    "vae",
                ],
            },
        )

        self.assertTrue(args.layerwise_offload_components)
        self.assertTrue(args.dit_cpu_offload)
        self.assertFalse(args.text_encoder_cpu_offload)
        self.assertFalse(args.image_encoder_cpu_offload)
        self.assertFalse(args.vae_cpu_offload)
        for component_name in (
            "text_encoder",
            "image_encoder",
            "video_dit",
            "vae",
        ):
            self.assertEqual(args.residency_mode(component_name), LAYERWISE_OFFLOAD)

    def test_legacy_layerwise_wins_over_component_offload(self):
        """Each component resolves to one mode under mixed legacy flags."""
        args = self._from_dict_with_task_type(
            ModelTaskType.T2I,
            memory_gb=32,
            kwargs={
                "dit_cpu_offload": True,
                "dit_layerwise_offload": True,
            },
        )

        self.assertTrue(args.dit_cpu_offload)
        self.assertTrue(args.dit_layerwise_offload)
        self.assertEqual(args.layerwise_offload_components, ["dit"])
        self.assertEqual(args.residency_mode("transformer"), LAYERWISE_OFFLOAD)
        self.assertEqual(args.residency_mode("connectors"), COMPONENT_OFFLOAD)

    def test_explicit_layerwise_false_keeps_independent_auto_residency(self):
        args = self._from_dict_with_pipeline_config(
            QwenImagePipelineConfig(),
            kwargs={
                "performance_mode": "auto",
                "dit_layerwise_offload": False,
            },
        )

        self.assertFalse(args.dit_layerwise_offload)
        self.assertFalse(args.dit_cpu_offload)
        self.assertEqual(args.residency_mode("transformer"), RESIDENT)

    def test_explicit_dit_cpu_offload_is_preserved_by_auto_residency(self):
        args = self._from_dict_with_pipeline_config(
            QwenImagePipelineConfig(),
            kwargs={
                "performance_mode": "auto",
                "dit_layerwise_offload": False,
                "dit_cpu_offload": True,
            },
        )

        self.assertFalse(args.dit_layerwise_offload)
        self.assertTrue(args.dit_cpu_offload)
        self.assertEqual(args.residency_mode("transformer"), COMPONENT_OFFLOAD)

    def test_explicit_layerwise_true_wins_over_auto_component_offload(self):
        args = self._from_dict_with_pipeline_config(
            QwenImagePipelineConfig(),
            kwargs={
                "performance_mode": "auto",
                "dit_layerwise_offload": True,
            },
        )

        self.assertTrue(args.dit_layerwise_offload)
        self.assertTrue(args.dit_cpu_offload)
        self.assertEqual(args.residency_mode("transformer"), LAYERWISE_OFFLOAD)

    def test_explicit_vae_cpu_offload_is_preserved_by_auto_residency(self):
        args = self._from_dict_with_pipeline_config(
            QwenImagePipelineConfig(),
            kwargs={
                "performance_mode": "auto",
                "dit_layerwise_offload": False,
                "vae_cpu_offload": True,
            },
        )

        self.assertFalse(args.dit_cpu_offload)
        self.assertTrue(args.vae_cpu_offload)

    def test_explicit_cpu_offload_components_are_preserved_by_auto_residency(self):
        args = self._from_dict_with_pipeline_config(
            QwenImagePipelineConfig(),
            kwargs={
                "performance_mode": "auto",
                "cpu_offload_components": ["dit", "vae"],
            },
        )

        self.assertEqual(args.residency_mode("transformer"), COMPONENT_OFFLOAD)
        self.assertEqual(args.residency_mode("vae"), COMPONENT_OFFLOAD)

    def test_cpu_offload_component_selector_keeps_unmatched_auto_defaults(self):
        args = self._from_dict_with_pipeline_config(
            QwenImagePipelineConfig(),
            kwargs={
                "performance_mode": "auto",
                "cpu_offload_components": ["vae"],
            },
        )

        self.assertEqual(args.residency_mode("vae"), COMPONENT_OFFLOAD)
        self.assertEqual(args.residency_mode("transformer"), RESIDENT)
        self.assertEqual(args.residency_mode("text_encoder"), LAYERWISE_OFFLOAD)

    def test_explicit_dit_layerwise_component_wins_over_auto_residency(self):
        args = self._from_dict_with_pipeline_config(
            QwenImagePipelineConfig(),
            kwargs={
                "performance_mode": "auto",
                "layerwise_offload_components": ["dit"],
            },
        )

        self.assertTrue(args.dit_cpu_offload)
        self.assertEqual(args.layerwise_offload_components, ["dit"])
        self.assertEqual(args.residency_mode("transformer"), LAYERWISE_OFFLOAD)

    def test_pipeline_configs_declare_auto_tune_hints(self):
        qwen_deployment = QwenImagePipelineConfig().get_model_deployment_config()
        cosmos3_deployment = Cosmos3Config(
            model_path="nvidia/Cosmos3-Nano"
        ).get_model_deployment_config()
        wan_deployment = WanT2V480PConfig().get_model_deployment_config()
        mova_deployment = MOVAPipelineConfig().get_model_deployment_config()
        zimage_deployment = ZImagePipelineConfig().get_model_deployment_config()
        lingbot_deployment = LingBotWorldCausalDMDConfig().get_model_deployment_config()
        ltx_deployment = LTX2PipelineConfig().get_model_deployment_config()
        ltx23_config = LTX23PipelineConfig()
        longlive_deployment = LongLive2T2VConfig().get_model_deployment_config()
        sana_wm_deployment = SanaWMPipelineConfig().get_model_deployment_config()

        self.assertIsNone(qwen_deployment.fsdp_auto_min_available_memory_gb)
        self.assertEqual(qwen_deployment.dit_layerwise_offload_modes, ())

        self.assertEqual(cosmos3_deployment.keep_resident_min_available_gb, 120)
        self.assertEqual(cosmos3_deployment.keep_resident_components, ("dit", "vae"))

        self.assertIsNone(wan_deployment.fsdp_auto_min_available_memory_gb)
        self.assertEqual(wan_deployment.dit_layerwise_offload_modes, ("memory",))
        self.assertEqual(wan_deployment.keep_resident_min_available_gb, 60)
        self.assertEqual(wan_deployment.keep_resident_components, ("dit",))

        self.assertIsNone(mova_deployment.fsdp_auto_min_available_memory_gb)
        self.assertEqual(
            mova_deployment.dit_layerwise_offload_modes, ("auto", "memory")
        )
        self.assertEqual(mova_deployment.keep_resident_min_available_gb, 130)
        self.assertEqual(mova_deployment.keep_resident_components, ("dit", "vae"))

        self.assertEqual(zimage_deployment.fsdp_auto_min_available_memory_gb, 40)
        self.assertEqual(zimage_deployment.keep_resident_min_available_gb, 30)
        self.assertTrue(zimage_deployment.fsdp_auto_requires_cfg)
        self.assertEqual(zimage_deployment.dit_layerwise_offload_modes, ())

        self.assertEqual(lingbot_deployment.dit_layerwise_offload_modes, ("memory",))
        self.assertEqual(lingbot_deployment.keep_resident_min_available_gb, 70)
        self.assertEqual(lingbot_deployment.keep_resident_components, ("dit",))

        self.assertEqual(ltx_deployment.keep_resident_min_available_gb, 70)
        self.assertEqual(ltx_deployment.keep_resident_components, ("dit",))
        self.assertEqual(
            ltx_deployment.auto_cfg_parallel_degree_by_num_gpus, ((4, 1), (8, 1))
        )
        self.assertEqual(ltx_deployment.get_auto_cfg_parallel_degree(4), 1)
        self.assertEqual(ltx_deployment.get_auto_cfg_parallel_degree(8), 1)
        self.assertEqual(ltx_deployment.get_auto_cfg_parallel_degree(2), 2)
        self.assertEqual(longlive_deployment.keep_resident_min_available_gb, 60)
        self.assertEqual(
            longlive_deployment.keep_resident_components,
            ("dit", "text_encoder", "vae"),
        )
        self.assertFalse(
            LTX2PipelineConfig().dit_config.arch_config.enable_packed_qkv_input_a2a
        )
        self.assertFalse(
            ltx23_config.dit_config.arch_config.enable_packed_qkv_input_a2a
        )

        self.assertEqual(sana_wm_deployment.fsdp_auto_min_available_memory_gb, 60)
        self.assertEqual(sana_wm_deployment.dit_layerwise_offload_modes, ("memory",))

        fast_hunyuan_deployment = FastHunyuanConfig().get_model_deployment_config()
        self.assertEqual(fast_hunyuan_deployment.keep_resident_min_available_gb, 60)
        self.assertEqual(
            fast_hunyuan_deployment.keep_resident_components, ("dit", "vae")
        )

        fast_wan_deployment = FastWan2_2_TI2V_5B_Config().get_model_deployment_config()
        self.assertEqual(fast_wan_deployment.keep_resident_min_available_gb, 60)
        self.assertEqual(fast_wan_deployment.keep_resident_components, ("dit",))

        for dual_dit_config in (
            Wan2_2_T2V_A14B_Config(),
            Wan2_2_I2V_A14B_Config(),
        ):
            dual_dit_deployment = dual_dit_config.get_model_deployment_config()
            self.assertIsNone(dual_dit_deployment.keep_resident_min_available_gb)
            self.assertEqual(dual_dit_deployment.keep_resident_components, ("vae",))

        # default keeps only vae resident (encoders are large, dit owned by FSDP)
        self.assertEqual(qwen_deployment.keep_resident_components, ("vae",))
        self.assertIsNone(qwen_deployment.keep_resident_min_available_gb)

    def test_longlive_residency_scales_with_available_memory(self):
        high_memory_args = self._from_dict_with_pipeline_config(
            LongLive2T2VConfig(),
            memory_gb=80,
            kwargs={"performance_mode": "auto"},
        )
        high_memory_offload = high_memory_args.layerwise_offload_components or []
        self.assertNotIn("text_encoder", high_memory_offload)
        self.assertNotIn("vae", high_memory_offload)

        constrained_args = self._from_dict_with_pipeline_config(
            LongLive2T2VConfig(),
            memory_gb=50,
            kwargs={"performance_mode": "auto"},
        )
        constrained_offload = constrained_args.layerwise_offload_components or []
        self.assertIn("text_encoder", constrained_offload)
        self.assertIn("vae", constrained_offload)

    def test_qwen_ar_generation_residency_scales_with_available_memory(self):
        pipeline_configs = (
            QwenImageLayeredPipelineConfig(),
            LongCatImagePipelineConfig(),
        )

        for pipeline_config in pipeline_configs:
            high_memory_args = self._from_dict_with_pipeline_config(
                pipeline_config,
                memory_gb=80,
                kwargs={"performance_mode": "auto"},
            )
            self.assertNotIn(
                "text_encoder", high_memory_args.layerwise_offload_components or []
            )
            self.assertFalse(high_memory_args.text_encoder_cpu_offload)

            constrained_args = self._from_dict_with_pipeline_config(
                pipeline_config,
                memory_gb=60,
                kwargs={"performance_mode": "auto"},
            )
            self.assertIn(
                "text_encoder", constrained_args.layerwise_offload_components or []
            )

    def test_auto_multi_gpu_sana_wm_prefers_fsdp_and_cfg_parallel(self):
        args = self._from_dict_with_pipeline_config(
            SanaWMPipelineConfig(),
            kwargs={
                "model_path": "Efficient-Large-Model/SANA-WM_bidirectional",
                "num_gpus": 2,
                "performance_mode": "auto",
            },
        )

        self.assertTrue(args.use_fsdp_inference)
        self.assertTrue(args.enable_cfg_parallel)

    def test_cache_dit_rejects_explicit_fsdp(self):
        with patch.dict(os.environ, {"SGLANG_CACHE_DIT_ENABLED": "true"}):
            with self.assertRaisesRegex(ValueError, "FSDP inference"):
                self._from_dict_with_pipeline_config(
                    SanaWMPipelineConfig(),
                    kwargs={
                        "model_path": "Efficient-Large-Model/SANA-WM_bidirectional",
                        "num_gpus": 2,
                        "use_fsdp_inference": True,
                    },
                )

    def test_cache_dit_auto_disables_implicit_fsdp(self):
        with patch.dict(os.environ, {"SGLANG_CACHE_DIT_ENABLED": "true"}):
            args = self._from_dict_with_pipeline_config(
                SanaWMPipelineConfig(),
                kwargs={
                    "model_path": "Efficient-Large-Model/SANA-WM_bidirectional",
                    "num_gpus": 2,
                    "performance_mode": "auto",
                },
            )

        self.assertFalse(args.use_fsdp_inference)
        self.assertTrue(args.enable_cfg_parallel)

    def test_auto_multi_gpu_sana_wm_realtime_disables_cfg_parallel(self):
        args = self._from_dict_with_pipeline_config(
            SanaWMRealtimeConfig(),
            kwargs={
                "model_path": "Efficient-Large-Model/SANA-WM_streaming",
                "num_gpus": 2,
                "performance_mode": "auto",
            },
        )

        self.assertFalse(args.use_fsdp_inference)
        self.assertFalse(args.enable_cfg_parallel)

    def test_auto_ltx23_large_gpu_counts_prefer_sp_over_cfg_parallel(self):
        for num_gpus in (4, 8):
            with self.subTest(num_gpus=num_gpus):
                args = self._from_dict_with_pipeline_config(
                    LTX2PipelineConfig(),
                    kwargs={
                        "model_path": "Lightricks/LTX-2.3",
                        "num_gpus": num_gpus,
                        "performance_mode": "auto",
                    },
                )

                self.assertFalse(args.enable_cfg_parallel)
                self.assertEqual(args.cfg_parallel_degree, 1)
                self.assertEqual(args.sp_degree, num_gpus)
                self.assertEqual(args.ulysses_degree, num_gpus)
                self.assertEqual(args.ring_degree, 1)

    def test_manual_mode_preserves_unset_performance_args(self):
        args = self._from_dict_with_pipeline_config(
            QwenImagePipelineConfig(),
            kwargs={
                "model_path": "Qwen/Qwen-Image",
                "num_gpus": 2,
                "performance_mode": "manual",
            },
        )

        self.assertEqual(args.performance_mode, "manual")
        self.assertIsNone(args.use_fsdp_inference)
        self.assertIsNone(args.dit_cpu_offload)
        self.assertIsNone(args.dit_layerwise_offload)
        self.assertIsNone(args.layerwise_offload_components)
        self.assertIsNone(args.text_encoder_cpu_offload)
        self.assertIsNone(args.image_encoder_cpu_offload)
        self.assertFalse(args.enable_cfg_parallel)

    def test_default_auto_keeps_image_vae_resident_when_memory_allows(self):
        args = self._from_dict_with_pipeline_config(
            QwenImagePipelineConfig(),
            kwargs={"model_path": "Qwen/Qwen-Image"},
        )

        self.assertEqual(args.performance_mode, "auto")
        self.assertFalse(args.use_fsdp_inference)
        # 80gb > image threshold (45gb): vae and dit stay resident, while the
        # large encoders use layerwise offload.
        self.assertFalse(args.dit_cpu_offload)
        self.assertEqual(
            args.layerwise_offload_components,
            ["text_encoder", "image_encoder"],
        )
        self.assertFalse(args.vae_cpu_offload)

    def test_auto_image_offloads_aux_below_resident_threshold(self):
        # 40gb < image threshold (45gb): aux incl. vae still offloaded to save vram
        args = self._from_dict_with_pipeline_config(
            QwenImagePipelineConfig(),
            memory_gb=40,
            kwargs={"model_path": "Qwen/Qwen-Image"},
        )

        self.assertEqual(args.performance_mode, "auto")
        self.assertTrue(args.dit_cpu_offload)
        self.assertEqual(
            args.layerwise_offload_components,
            ["text_encoder", "image_encoder", "vae"],
        )

    def test_auto_zimage_keeps_dit_resident_on_5090(self):
        args = self._from_dict_with_pipeline_config(
            ZImagePipelineConfig(),
            memory_gb=32,
            available_memory_gb=31,
            kwargs={
                "model_path": "Tongyi-MAI/Z-Image-Turbo",
                "performance_mode": "auto",
            },
        )

        self.assertFalse(args.dit_cpu_offload)
        self.assertEqual(
            args.layerwise_offload_components,
            ["text_encoder", "image_encoder"],
        )

    def test_auto_lingbot_keeps_dit_resident_on_h100(self):
        args = self._from_dict_with_pipeline_config(
            LingBotWorldCausalDMDConfig(),
            memory_gb=80,
            available_memory_gb=72,
            kwargs={
                "model_path": "robbyant/lingbot-world-fast-diffusers",
                "performance_mode": "auto",
                "text_encoder_cpu_offload": True,
            },
        )

        self.assertFalse(args.dit_cpu_offload)
        self.assertTrue(args.text_encoder_cpu_offload)
        self.assertEqual(
            args.layerwise_offload_components,
            ["image_encoder", "vae"],
        )

    def test_auto_image_preserves_explicit_dit_cpu_offload(self):
        args = self._from_dict_with_pipeline_config(
            QwenImagePipelineConfig(),
            kwargs={
                "model_path": "Qwen/Qwen-Image",
                "dit_cpu_offload": True,
            },
        )

        self.assertTrue(args.dit_cpu_offload)

    def test_auto_ltx_original_replaces_component_cpu_offload(
        self,
    ):
        args = self._from_dict_with_pipeline_config(
            LTX2PipelineConfig(),
            available_memory_gb=76,
            kwargs={
                "model_path": "Lightricks/LTX-2.3",
                "pipeline_class_name": "LTX2TwoStageHQPipeline",
                "performance_mode": "auto",
            },
        )

        self.assertEqual(args.ltx2_two_stage_device_mode, "original")
        self.assertFalse(args.dit_cpu_offload)
        self.assertTrue(args.layerwise_offload_components)
        self.assertFalse(args.text_encoder_cpu_offload)
        self.assertFalse(args.image_encoder_cpu_offload)
        self.assertEqual(
            args.layerwise_offload_components,
            ["text_encoder", "image_encoder", "vae"],
        )

    def test_auto_wan_keeps_single_dit_resident_on_h100(self):
        args = self._from_dict_with_pipeline_config(
            WanT2V480PConfig(),
            kwargs={"performance_mode": "auto"},
        )

        self.assertTrue(args.layerwise_offload_components)
        self.assertFalse(args.use_fsdp_inference)
        self.assertFalse(args.dit_cpu_offload)
        self.assertFalse(args.text_encoder_cpu_offload)
        self.assertFalse(args.image_encoder_cpu_offload)
        self.assertEqual(
            args.layerwise_offload_components,
            ["text_encoder", "image_encoder", "vae"],
        )

    def test_auto_wan_offloads_single_dit_below_resident_threshold(self):
        args = self._from_dict_with_pipeline_config(
            WanT2V480PConfig(),
            memory_gb=48,
            kwargs={"performance_mode": "auto"},
        )

        self.assertTrue(args.dit_cpu_offload)
        self.assertEqual(
            args.layerwise_offload_components,
            ["text_encoder", "image_encoder", "vae"],
        )

    def test_auto_wan2_2_a14b_layerwise_offload_adds_dit(self):
        for pipeline_config, model_path in (
            (Wan2_2_T2V_A14B_Config(), "Wan-AI/Wan2.2-T2V-A14B-Diffusers"),
            (Wan2_2_I2V_A14B_Config(), "Wan-AI/Wan2.2-I2V-A14B-Diffusers"),
        ):
            with self.subTest(pipeline_config=pipeline_config.__class__.__name__):
                args = self._from_dict_with_pipeline_config(
                    pipeline_config,
                    kwargs={
                        "model_path": model_path,
                        "performance_mode": "auto",
                    },
                )

                self.assertTrue(args.layerwise_offload_components)
                self.assertFalse(args.use_fsdp_inference)
                self.assertTrue(args.dit_cpu_offload)
                self.assertEqual(args.residency_mode("transformer"), LAYERWISE_OFFLOAD)
                self.assertFalse(args.text_encoder_cpu_offload)
                self.assertFalse(args.image_encoder_cpu_offload)
                self.assertEqual(args.dit_offload_prefetch_size, 2)
                self.assertEqual(
                    args.layerwise_offload_components,
                    ["dit", "text_encoder", "image_encoder", "vae"],
                )

    def test_auto_wan2_1_14b_keeps_dit_resident_on_h100(self):
        for pipeline_config, model_path in (
            (WanT2V720PConfig(), "Wan-AI/Wan2.1-T2V-14B-Diffusers"),
            (WanI2V480PConfig(), "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"),
            (WanI2V720PConfig(), "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"),
        ):
            with self.subTest(pipeline_config=pipeline_config.__class__.__name__):
                args = self._from_dict_with_pipeline_config(
                    pipeline_config,
                    kwargs={
                        "model_path": model_path,
                        "performance_mode": "auto",
                    },
                )

                self.assertTrue(args.layerwise_offload_components)
                self.assertFalse(args.dit_cpu_offload)
                self.assertEqual(args.dit_offload_prefetch_size, 0.0)
                self.assertEqual(
                    args.layerwise_offload_components,
                    ["text_encoder", "image_encoder", "vae"],
                )

    def test_auto_wan2_1_14b_offloads_dit_below_resident_threshold(self):
        args = self._from_dict_with_pipeline_config(
            WanI2V720PConfig(),
            memory_gb=48,
            kwargs={
                "model_path": "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
                "performance_mode": "auto",
            },
        )

        self.assertTrue(args.dit_cpu_offload)

    def test_memory_wan_layerwise_offload_is_enabled_without_fsdp(self):
        args = self._from_dict_with_pipeline_config(
            WanT2V480PConfig(),
            kwargs={"performance_mode": "memory"},
        )

        self.assertTrue(args.layerwise_offload_components)
        self.assertFalse(args.use_fsdp_inference)
        self.assertTrue(args.dit_cpu_offload)
        self.assertFalse(args.text_encoder_cpu_offload)
        self.assertFalse(args.image_encoder_cpu_offload)
        self.assertEqual(
            args.layerwise_offload_components,
            ["dit", "text_encoder", "image_encoder", "vae"],
        )

    def test_auto_wan_layerwise_offload_does_not_disable_explicit_fsdp(self):
        args = self._from_dict_with_pipeline_config(
            WanT2V480PConfig(),
            kwargs={
                "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                "num_gpus": 2,
                "performance_mode": "auto",
                "use_fsdp_inference": True,
            },
        )

        self.assertEqual(
            args.layerwise_offload_components,
            ["text_encoder", "image_encoder", "vae"],
        )
        self.assertTrue(args.use_fsdp_inference)

    def test_auto_wan_layerwise_offload_preserves_explicit_dit_cpu_offload(self):
        args = self._from_dict_with_pipeline_config(
            WanT2V480PConfig(),
            kwargs={
                "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                "performance_mode": "auto",
                "dit_cpu_offload": True,
            },
        )

        self.assertTrue(args.dit_cpu_offload)
        self.assertEqual(
            args.layerwise_offload_components,
            ["text_encoder", "image_encoder", "vae"],
        )

    def test_auto_mova_layerwise_offload_adds_dit_below_memory_threshold(self):
        args = self._from_dict_with_pipeline_config(
            MOVAPipelineConfig(),
            kwargs={
                "model_path": "OpenMOSS-Team/MOVA-360p",
                "performance_mode": "auto",
            },
        )

        self.assertTrue(args.dit_cpu_offload)
        self.assertEqual(
            args.layerwise_offload_components,
            ["dit", "text_encoder", "image_encoder", "vae"],
        )

    def test_auto_mova_keeps_dit_resident_at_memory_threshold(self):
        args = self._from_dict_with_pipeline_config(
            MOVAPipelineConfig(),
            memory_gb=140,
            kwargs={
                "model_path": "OpenMOSS-Team/MOVA-360p",
                "performance_mode": "auto",
            },
        )

        self.assertFalse(args.dit_cpu_offload)
        self.assertEqual(
            args.layerwise_offload_components,
            ["text_encoder", "image_encoder"],
        )

    def test_auto_cosmos3_keeps_dit_resident_on_high_memory_gpu(self):
        args = self._from_dict_with_pipeline_config(
            Cosmos3Config(model_path="nvidia/Cosmos3-Nano"),
            available_memory_gb=139,
            kwargs={
                "model_path": "nvidia/Cosmos3-Nano",
                "performance_mode": "auto",
            },
        )

        self.assertFalse(args.dit_cpu_offload)
        self.assertFalse(args.vae_cpu_offload)
        self.assertEqual(
            args.layerwise_offload_components,
            ["text_encoder", "image_encoder"],
        )

    def test_auto_cosmos3_offloads_dit_below_resident_threshold(self):
        args = self._from_dict_with_pipeline_config(
            Cosmos3Config(model_path="nvidia/Cosmos3-Nano"),
            available_memory_gb=100,
            kwargs={
                "model_path": "nvidia/Cosmos3-Nano",
                "performance_mode": "auto",
            },
        )

        self.assertTrue(args.dit_cpu_offload)
        self.assertFalse(args.vae_cpu_offload)

    def test_auto_cosmos3_super_keeps_dit_resident_on_high_memory_gpu(self):
        # Super is a single-DiT pipeline like Nano, so above the threshold the
        # component-offload round trip is pure per-request copy cost.
        args = self._from_dict_with_pipeline_config(
            Cosmos3Config(model_path="nvidia/Cosmos3-Super"),
            available_memory_gb=139,
            kwargs={
                "model_path": "nvidia/Cosmos3-Super",
                "performance_mode": "auto",
            },
        )

        self.assertFalse(args.dit_cpu_offload)
        self.assertFalse(args.vae_cpu_offload)

    def test_auto_cosmos3_super_offloads_dit_below_resident_threshold(self):
        args = self._from_dict_with_pipeline_config(
            Cosmos3Config(model_path="nvidia/Cosmos3-Super"),
            available_memory_gb=100,
            kwargs={
                "model_path": "nvidia/Cosmos3-Super",
                "performance_mode": "auto",
            },
        )

        self.assertTrue(args.dit_cpu_offload)
        self.assertFalse(args.vae_cpu_offload)

    def test_memory_sana_wm_layerwise_offload_adds_dit(self):
        args = self._from_dict_with_pipeline_config(
            SanaWMPipelineConfig(),
            kwargs={
                "model_path": "Efficient-Large-Model/SANA-WM_bidirectional",
                "performance_mode": "memory",
            },
        )

        self.assertEqual(
            args.layerwise_offload_components,
            ["dit", "text_encoder", "image_encoder", "vae"],
        )

    def test_auto_fastwan_keeps_dit_resident_on_h100(self):
        args = self._from_dict_with_pipeline_config(
            FastWan2_2_TI2V_5B_Config(),
            available_memory_gb=72,
            kwargs={
                "model_path": "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
                "performance_mode": "auto",
            },
        )

        self.assertFalse(args.dit_cpu_offload)
        self.assertEqual(
            args.layerwise_offload_components,
            ["text_encoder", "image_encoder", "vae"],
        )

    def test_auto_fastwan_offloads_dit_below_resident_threshold(self):
        args = self._from_dict_with_pipeline_config(
            FastWan2_2_TI2V_5B_Config(),
            memory_gb=48,
            kwargs={
                "model_path": "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
                "performance_mode": "auto",
            },
        )

        self.assertTrue(args.dit_cpu_offload)

    def test_auto_fast_hunyuan_keeps_dit_resident_on_h100(self):
        args = self._from_dict_with_pipeline_config(
            FastHunyuanConfig(),
            available_memory_gb=72,
            kwargs={
                "model_path": "FastVideo/FastHunyuan-diffusers",
                "performance_mode": "auto",
            },
        )

        self.assertFalse(args.dit_cpu_offload)
        self.assertFalse(args.vae_cpu_offload)

    def test_auto_fast_hunyuan_offloads_dit_below_resident_threshold(self):
        args = self._from_dict_with_pipeline_config(
            FastHunyuanConfig(),
            memory_gb=48,
            kwargs={
                "model_path": "FastVideo/FastHunyuan-diffusers",
                "performance_mode": "auto",
            },
        )

        self.assertTrue(args.dit_cpu_offload)

    def test_auto_turbo_wan_keeps_dit_resident_on_h100(self):
        args = self._from_dict_with_pipeline_config(
            TurboWanT2V480PConfig(),
            kwargs={
                "model_path": "IPostYellow/TurboWan2.1-T2V-1.3B-Diffusers",
                "performance_mode": "auto",
            },
        )

        self.assertFalse(args.dit_cpu_offload)
        self.assertEqual(
            args.layerwise_offload_components,
            ["text_encoder", "image_encoder", "vae"],
        )

    def test_explicit_fastwan_dit_layerwise_still_selects_dit_group(self):
        args = self._from_dict_with_pipeline_config(
            FastWan2_2_TI2V_5B_Config(),
            kwargs={
                "model_path": "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
                "dit_layerwise_offload": True,
            },
        )

        self.assertTrue(args.dit_cpu_offload)
        self.assertEqual(args.layerwise_offload_components, ["dit"])

    def test_auto_multi_gpu_wan_uses_layerwise_offload_without_cfg(self):
        with patch.object(ServerArgs, "_model_default_uses_cfg", return_value=False):
            args = self._from_dict_with_pipeline_config(
                WanT2V480PConfig(),
                kwargs={
                    "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                    "num_gpus": 2,
                    "performance_mode": "auto",
                },
            )

        self.assertFalse(args.use_fsdp_inference)
        self.assertFalse(args.enable_cfg_parallel)
        self.assertFalse(args.dit_cpu_offload)
        self.assertTrue(args.layerwise_offload_components)
        self.assertFalse(args.text_encoder_cpu_offload)
        self.assertFalse(args.image_encoder_cpu_offload)
        self.assertEqual(
            args.layerwise_offload_components,
            ["text_encoder", "image_encoder", "vae"],
        )

    def test_explicit_multi_gpu_dit_layerwise_only_selects_dit_group(self):
        args = self._from_dict_with_pipeline_config(
            MOVAPipelineConfig(),
            kwargs={
                "model_path": "OpenMOSS-Team/MOVA-360p",
                "num_gpus": 2,
                "dit_layerwise_offload": True,
            },
        )

        self.assertFalse(args.use_fsdp_inference)
        self.assertTrue(args.dit_cpu_offload)
        self.assertTrue(args.layerwise_offload_components)
        self.assertTrue(args.text_encoder_cpu_offload)
        self.assertTrue(args.image_encoder_cpu_offload)
        self.assertEqual(args.layerwise_offload_components, ["dit"])

    def test_auto_multi_gpu_ltx_replaces_component_cpu_offload_with_resident_dit(self):
        args = self._from_dict_with_pipeline_config(
            LTX2PipelineConfig(),
            available_memory_gb=76,
            kwargs={
                "model_path": "Lightricks/LTX-2",
                "num_gpus": 2,
                "pipeline_class_name": "LTX2TwoStagePipeline",
            },
        )

        self.assertFalse(args.use_fsdp_inference)
        self.assertFalse(args.dit_cpu_offload)
        self.assertTrue(args.layerwise_offload_components)
        self.assertFalse(args.text_encoder_cpu_offload)
        self.assertFalse(args.image_encoder_cpu_offload)
        self.assertEqual(
            args.layerwise_offload_components,
            ["text_encoder", "image_encoder", "vae"],
        )

    def test_auto_high_memory_ltx23_resident_keeps_aux_components_resident(self):
        args = self._from_dict_with_pipeline_config(
            LTX2PipelineConfig(),
            memory_gb=140,
            available_memory_gb=134,
            kwargs={
                "model_path": "Lightricks/LTX-2.3",
                "num_gpus": 2,
                "pipeline_class_name": "LTX2TwoStagePipeline",
            },
        )

        self.assertEqual(args.ltx2_two_stage_device_mode, "resident")
        self.assertFalse(args.use_fsdp_inference)
        self.assertFalse(args.dit_cpu_offload)
        self.assertFalse(args.text_encoder_cpu_offload)
        self.assertFalse(args.image_encoder_cpu_offload)
        self.assertFalse(args.vae_cpu_offload)
        self.assertIsNone(args.layerwise_offload_components)

    def test_auto_high_memory_ltx23_original_keeps_default_layerwise_components(self):
        args = self._from_dict_with_pipeline_config(
            LTX2PipelineConfig(),
            memory_gb=140,
            available_memory_gb=134,
            kwargs={
                "model_path": "Lightricks/LTX-2.3",
                "num_gpus": 2,
                "pipeline_class_name": "LTX2TwoStagePipeline",
                "ltx2_two_stage_device_mode": "original",
            },
        )

        self.assertEqual(
            args.layerwise_offload_components,
            ["text_encoder", "image_encoder", "vae"],
        )

    def test_ltx23_snapshot_device_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Expected one of"):
            self._from_dict_with_pipeline_config(
                LTX2PipelineConfig(),
                memory_gb=140,
                available_memory_gb=134,
                kwargs={
                    "model_path": "Lightricks/LTX-2.3",
                    "num_gpus": 2,
                    "pipeline_class_name": "LTX2TwoStagePipeline",
                    "ltx2_two_stage_device_mode": "snapshot",
                },
            )

    def test_explicit_layerwise_components_preserved_in_ltx23_resident(self):
        args = self._from_dict_with_pipeline_config(
            LTX2PipelineConfig(),
            memory_gb=140,
            available_memory_gb=134,
            kwargs={
                "model_path": "Lightricks/LTX-2.3",
                "num_gpus": 2,
                "pipeline_class_name": "LTX2TwoStagePipeline",
                "layerwise_offload_components": ["text_encoder"],
            },
        )

        self.assertEqual(args.ltx2_two_stage_device_mode, "resident")
        self.assertEqual(args.layerwise_offload_components, ["text_encoder"])

    def test_auto_multi_gpu_qwen_keeps_vae_resident_with_cfg(self):
        args = self._from_dict_with_pipeline_config(
            QwenImagePipelineConfig(),
            kwargs={
                "model_path": "Qwen/Qwen-Image",
                "num_gpus": 2,
                "performance_mode": "auto",
            },
        )

        self.assertFalse(args.use_fsdp_inference)
        self.assertTrue(args.enable_cfg_parallel)
        # 80gb > image threshold (45gb): vae and dit stay resident, while the
        # large encoders use layerwise offload.
        self.assertFalse(args.dit_cpu_offload)
        self.assertEqual(
            args.layerwise_offload_components,
            ["text_encoder", "image_encoder"],
        )
        self.assertFalse(args.vae_cpu_offload)

    def test_auto_multi_gpu_zimage_base_prefers_fsdp(self):
        args = self._from_dict_with_pipeline_config(
            ZImagePipelineConfig(),
            kwargs={
                "model_path": "Tongyi-MAI/Z-Image",
                "num_gpus": 2,
                "performance_mode": "auto",
            },
        )

        self.assertTrue(args.use_fsdp_inference)
        self.assertTrue(args.enable_cfg_parallel)

    def test_auto_multi_gpu_zimage_turbo_skips_fsdp(self):
        args = self._from_dict_with_pipeline_config(
            ZImagePipelineConfig(),
            kwargs={
                "model_path": "Tongyi-MAI/Z-Image-Turbo",
                "num_gpus": 2,
                "performance_mode": "auto",
            },
        )

        self.assertFalse(args.use_fsdp_inference)
        self.assertFalse(args.enable_cfg_parallel)

    def test_auto_multi_gpu_qwen_preserves_explicit_fsdp_false(self):
        args = self._from_dict_with_pipeline_config(
            QwenImagePipelineConfig(),
            kwargs={
                "model_path": "Qwen/Qwen-Image",
                "num_gpus": 2,
                "performance_mode": "auto",
                "use_fsdp_inference": False,
            },
        )

        self.assertFalse(args.use_fsdp_inference)
        self.assertTrue(args.enable_cfg_parallel)
        # Explicit FSDP selection must not freeze unrelated, implicit DiT
        # residency decisions on a high-memory GPU.
        self.assertFalse(args.dit_cpu_offload)
        self.assertFalse(args.vae_cpu_offload)
        # The layerwise filter still drops VAE (kept resident); encoders stay
        # offloaded.
        self.assertEqual(
            args.layerwise_offload_components,
            ["text_encoder", "image_encoder"],
        )

    def test_auto_multi_gpu_qwen_skips_fsdp_when_available_memory_is_low(self):
        args = self._from_dict_with_pipeline_config(
            QwenImagePipelineConfig(),
            memory_gb=50,
            kwargs={
                "model_path": "Qwen/Qwen-Image",
                "num_gpus": 2,
                "performance_mode": "auto",
            },
        )

        self.assertFalse(args.use_fsdp_inference)
        self.assertTrue(args.enable_cfg_parallel)
        # 50gb still > image threshold (45gb): vae and dit stay resident, while
        # the encoders remain offloaded; qwen does not opt into auto fsdp.
        self.assertFalse(args.dit_cpu_offload)
        self.assertEqual(
            args.layerwise_offload_components,
            ["text_encoder", "image_encoder"],
        )
        self.assertFalse(args.vae_cpu_offload)

    def test_auto_multi_gpu_qwen_uses_selected_gpu_min_available_memory(self):
        args = self._from_dict_with_pipeline_config(
            QwenImagePipelineConfig(),
            available_memory_gb={1: 50, 2: 80},
            kwargs={
                "model_path": "Qwen/Qwen-Image",
                "base_gpu_id": 1,
                "num_gpus": 2,
                "performance_mode": "auto",
            },
        )

        self.assertFalse(args.use_fsdp_inference)
        self.assertTrue(args.enable_cfg_parallel)

    def test_auto_multi_gpu_qwen_keeps_vae_resident_with_headroom(self):
        args = self._from_dict_with_pipeline_config(
            QwenImagePipelineConfig(),
            available_memory_gb={1: 72, 2: 80},
            kwargs={
                "model_path": "Qwen/Qwen-Image",
                "base_gpu_id": 1,
                "num_gpus": 2,
                "performance_mode": "auto",
            },
        )

        self.assertFalse(args.use_fsdp_inference)
        self.assertTrue(args.enable_cfg_parallel)
        # min available across selected gpus is 72gb > image threshold (45gb):
        # vae and dit stay resident, while the encoders remain offloaded.
        self.assertFalse(args.dit_cpu_offload)
        self.assertEqual(
            args.layerwise_offload_components,
            ["text_encoder", "image_encoder"],
        )
        self.assertFalse(args.vae_cpu_offload)

    def test_auto_minimax_h3_keeps_large_components_resident_with_headroom(self):
        args = self._from_dict_with_pipeline_config(
            MiniMaxH3PipelineConfig(),
            memory_gb=141,
            available_memory_gb=130,
            kwargs={
                "model_path": "MiniMaxAI/MiniMax-H3",
                "num_gpus": 8,
                "ulysses_degree": 8,
                "performance_mode": "auto",
            },
        )

        self.assertFalse(args.dit_cpu_offload)
        self.assertFalse(args.text_encoder_cpu_offload)
        self.assertFalse(args.vae_cpu_offload)
        self.assertNotIn("text_encoder", args.layerwise_offload_components or [])
        self.assertNotIn("vae", args.layerwise_offload_components or [])

    def test_auto_minimax_h3_keeps_memory_policy_below_residency_threshold(self):
        args = self._from_dict_with_pipeline_config(
            MiniMaxH3PipelineConfig(),
            memory_gb=96,
            available_memory_gb=90,
            kwargs={
                "model_path": "MiniMaxAI/MiniMax-H3",
                "num_gpus": 8,
                "ulysses_degree": 8,
                "performance_mode": "auto",
            },
        )

        self.assertTrue(args.dit_cpu_offload)
        self.assertIn("text_encoder", args.layerwise_offload_components or [])
        self.assertIn("vae", args.layerwise_offload_components or [])

    def test_memory_minimax_h3_combines_fsdp_with_aux_layerwise_offload(self):
        args = self._from_dict_with_pipeline_config(
            MiniMaxH3PipelineConfig(),
            kwargs={
                "model_path": "MiniMaxAI/MiniMax-H3",
                "num_gpus": 8,
                "ulysses_degree": 8,
                "performance_mode": "memory",
                "use_fsdp_inference": True,
            },
        )

        self.assertTrue(args.use_fsdp_inference)
        self.assertFalse(args.dit_cpu_offload)
        self.assertFalse(args.dit_layerwise_offload)
        self.assertIn("text_encoder", args.layerwise_offload_components or [])
        self.assertIn("vae", args.layerwise_offload_components or [])
        self.assertFalse(args.vae_cpu_offload)
        self.assertEqual(args.residency_mode("vae"), LAYERWISE_OFFLOAD)

    def test_minimax_h3_rejects_explicit_cfg_parallel(self):
        with self.assertRaisesRegex(
            ValueError, "MiniMaxH3PipelineConfig does not support CFG parallelism"
        ):
            self._from_dict_with_pipeline_config(
                MiniMaxH3PipelineConfig(),
                kwargs={
                    "model_path": "MiniMaxAI/MiniMax-H3",
                    "num_gpus": 4,
                    "cfg_parallel_degree": 2,
                },
            )

    def test_speed_mode_single_gpu_disables_offload(self):
        args = self._from_dict_with_pipeline_config(
            QwenImagePipelineConfig(),
            kwargs={
                "model_path": "Qwen/Qwen-Image",
                "performance_mode": "speed",
            },
        )

        self.assertEqual(args.performance_mode, "speed")
        self.assertFalse(args.use_fsdp_inference)
        self.assertFalse(args.dit_cpu_offload)
        self.assertFalse(args.layerwise_offload_components)
        self.assertFalse(args.text_encoder_cpu_offload)
        self.assertFalse(args.image_encoder_cpu_offload)

    def test_speed_mode_preserves_explicit_offload(self):
        args = self._from_dict_with_pipeline_config(
            QwenImagePipelineConfig(),
            kwargs={
                "model_path": "Qwen/Qwen-Image",
                "performance_mode": "speed",
                "dit_cpu_offload": True,
            },
        )

        self.assertEqual(args.performance_mode, "speed")
        self.assertTrue(args.dit_cpu_offload)
        self.assertFalse(args.text_encoder_cpu_offload)
        self.assertFalse(args.image_encoder_cpu_offload)

    def test_speed_mode_keeps_torch_compile_off_by_default(self):
        args = self._from_dict_with_pipeline_config(
            QwenImagePipelineConfig(),
            kwargs={
                "model_path": "Qwen/Qwen-Image",
                "performance_mode": "speed",
            },
        )

        self.assertFalse(args.enable_torch_compile)

    def test_speed_mode_preserves_explicit_torch_compile_setting(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                args = self._from_dict_with_pipeline_config(
                    QwenImagePipelineConfig(),
                    kwargs={
                        "model_path": "Qwen/Qwen-Image",
                        "performance_mode": "speed",
                        "enable_torch_compile": enabled,
                    },
                )

                self.assertEqual(args.enable_torch_compile, enabled)

    def test_speed_mode_honors_model_torch_compile_opt_in(self):
        with patch.object(
            QwenImagePipelineConfig,
            "get_model_deployment_config",
            return_value=ModelDeploymentConfig(
                speed_mode_enable_torch_compile_by_default=True
            ),
        ):
            args = self._from_dict_with_pipeline_config(
                QwenImagePipelineConfig(),
                kwargs={
                    "model_path": "Qwen/Qwen-Image",
                    "performance_mode": "speed",
                },
            )

        self.assertTrue(args.enable_torch_compile)

    def test_speed_mode_uses_minimax_h3_compile_policy(self):
        for explicit, expected in ((None, False), (True, True)):
            kwargs = {
                "model_path": "MiniMaxAI/MiniMax-H3",
                "performance_mode": "speed",
            }
            if explicit is not None:
                kwargs["enable_torch_compile"] = explicit
            with self.subTest(explicit=explicit):
                args = self._from_dict_with_pipeline_config(
                    MiniMaxH3PipelineConfig(), kwargs=kwargs
                )
                self.assertEqual(args.enable_torch_compile, expected)

    def test_auto_mode_leaves_torch_compile_off(self):
        args = self._from_dict_with_pipeline_config(
            QwenImagePipelineConfig(),
            kwargs={
                "model_path": "Qwen/Qwen-Image",
                "performance_mode": "auto",
            },
        )

        self.assertFalse(args.enable_torch_compile)

    def test_memory_mode_wan_uses_layerwise_offload(self):
        args = self._from_dict_with_pipeline_config(
            WanT2V480PConfig(),
            kwargs={
                "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                "performance_mode": "memory",
            },
        )

        self.assertFalse(args.use_fsdp_inference)
        self.assertTrue(args.layerwise_offload_components)
        self.assertTrue(args.dit_cpu_offload)
        self.assertFalse(args.text_encoder_cpu_offload)
        self.assertFalse(args.image_encoder_cpu_offload)
        self.assertEqual(
            args.layerwise_offload_components,
            ["dit", "text_encoder", "image_encoder", "vae"],
        )

    def test_memory_mode_preserves_explicit_fsdp(self):
        args = self._from_dict_with_pipeline_config(
            WanT2V480PConfig(),
            kwargs={
                "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                "num_gpus": 2,
                "performance_mode": "memory",
                "use_fsdp_inference": True,
            },
        )

        self.assertTrue(args.use_fsdp_inference)
        self.assertEqual(
            args.layerwise_offload_components,
            ["text_encoder", "image_encoder", "vae"],
        )
        self.assertFalse(args.dit_cpu_offload)

    def test_invalid_performance_mode_raises(self):
        with self.assertRaises(ValueError):
            self._from_dict_with_pipeline_config(
                QwenImagePipelineConfig(),
                kwargs={"performance_mode": "turbo"},
            )

    def test_cfg_parallel_cli_can_be_disabled_explicitly(self):
        parser = FlexibleArgumentParser()
        ServerArgs.add_cli_args(parser)
        argv = [
            "--model-path",
            "Qwen/Qwen-Image",
            "--num-gpus",
            "2",
            "--performance-mode",
            "auto",
            "--enable-cfg-parallel",
            "false",
        ]

        with (
            patch.object(sys, "argv", ["sglang"] + argv),
            patch.object(
                PipelineConfig, "from_kwargs", return_value=QwenImagePipelineConfig()
            ),
            patch(
                "sglang.multimodal_gen.runtime.platforms.current_platform.is_cpu",
                return_value=False,
            ),
            patch(
                "sglang.multimodal_gen.runtime.platforms.current_platform.is_mps",
                return_value=False,
            ),
            patch(
                "sglang.multimodal_gen.runtime.platforms.current_platform.is_cuda",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.platforms.current_platform.get_device_total_memory",
                return_value=80 * 1024**3,
            ),
            patch(
                "sglang.multimodal_gen.runtime.platforms.current_platform.get_available_gpu_memory",
                return_value=80,
            ),
        ):
            args, unknown_args = parser.parse_known_args(argv)
            server_args = ServerArgs.from_cli_args(args, unknown_args)

        self.assertFalse(server_args.use_fsdp_inference)
        self.assertFalse(server_args.enable_cfg_parallel)

    def test_ltx23_snapshot_device_mode_cli_is_rejected(self):
        parser = FlexibleArgumentParser()
        ServerArgs.add_cli_args(parser)
        argv = [
            "--model-path",
            "Lightricks/LTX-2.3",
            "--pipeline-class-name",
            "LTX2TwoStagePipeline",
            "--ltx2-two-stage-device-mode",
            "snapshot",
        ]

        with self.assertRaises(SystemExit):
            parser.parse_known_args(argv)


class TestKVGatherDegree(unittest.TestCase):
    def test_sp2_defaults_to_kv_gather(self):
        args = _from_dict_without_model_resolution(
            {
                "model_path": "/fake",
                "num_gpus": 2,
                "performance_mode": "manual",
            }
        )

        self.assertEqual(args.kv_gather_degree, 2)
        self.assertTrue(args.sp_split_auto)
        # gather rows occupy the contiguous inner SP dimension
        self.assertEqual(args.ulysses_degree, 2)
        self.assertEqual(args.sp_degree, 2)

    def test_higher_sp_defaults_to_ulysses(self):
        args = _from_dict_without_model_resolution(
            {
                "model_path": "/fake",
                "num_gpus": 4,
                "performance_mode": "manual",
            }
        )

        self.assertEqual(args.kv_gather_degree, 1)
        self.assertFalse(args.sp_split_auto)
        self.assertEqual(args.ulysses_degree, 4)

    def test_explicit_ulysses_is_not_overridden(self):
        args = _from_dict_without_model_resolution(
            {
                "model_path": "/fake",
                "num_gpus": 2,
                "ulysses_degree": 2,
                "performance_mode": "manual",
            }
        )

        self.assertEqual(args.kv_gather_degree, 1)
        self.assertEqual(args.ulysses_degree, 2)

    def test_explicit_degree_is_not_auto(self):
        args = _from_dict_without_model_resolution(
            {
                "model_path": "/fake",
                "num_gpus": 2,
                "kv_gather_degree": 2,
                "performance_mode": "manual",
            }
        )

        self.assertEqual(args.kv_gather_degree, 2)
        self.assertFalse(args.sp_split_auto)

    def test_kv_gather_supports_tp(self):
        args = _from_dict_without_model_resolution(
            {
                "model_path": "/fake",
                "num_gpus": 4,
                "tp_size": 2,
                "sp_degree": 2,
                "kv_gather_degree": 2,
                "performance_mode": "manual",
            }
        )

        self.assertEqual(args.tp_size, 2)
        self.assertEqual(args.sp_degree, 2)
        self.assertEqual(args.kv_gather_degree, 2)

    def test_kv_gather_supports_fsdp(self):
        args = _from_dict_without_model_resolution(
            {
                "model_path": "/fake",
                "num_gpus": 2,
                "sp_degree": 2,
                "kv_gather_degree": 2,
                "use_fsdp_inference": True,
                "performance_mode": "manual",
            }
        )

        self.assertTrue(args.use_fsdp_inference)
        self.assertEqual(args.kv_gather_degree, 2)

    def test_kv_gather_does_not_compose_yet(self):
        for extra in ({"ulysses_degree": 2}, {"ring_degree": 2}):
            with self.assertRaisesRegex(ValueError, "does not compose"):
                _from_dict_without_model_resolution(
                    {
                        "model_path": "/fake",
                        "num_gpus": 4,
                        "sp_degree": 4,
                        "kv_gather_degree": 2,
                        "performance_mode": "manual",
                        **extra,
                    }
                )


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

    def test_sana_wm_model_path_resolves_registry(self):
        info = _get_config_info("Efficient-Large-Model/SANA-WM_bidirectional")
        self.assertIs(info.pipeline_config_cls, SanaWMPipelineConfig)

    def test_model_id_unknown_falls_back_without_crash(self):
        # unrecognized model_id: should warn and fall back to path-based detection
        # with an unresolvable path, expect RuntimeError from the detector step
        with self.assertRaises((RuntimeError, Exception)):
            _get_config_info("/data/no-such-model", model_id="NonExistentModelXYZ")


class TestPerRoleParallelism(unittest.TestCase):
    """Test per-role parallelism args and get_role_parallelism helper."""

    def _from_dict(self, kwargs):
        return _from_dict_without_model_resolution(kwargs)

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
        args = self._from_dict({"model_path": "/fake", "decoder_sp": 2})
        from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType

        par = args.get_role_parallelism(RoleType.DECODER)
        self.assertIsNone(par["tp_size"])
        self.assertEqual(par["sp_degree"], 2)
        self.assertIsNone(par["ulysses_degree"])
        self.assertIsNone(par["ring_degree"])

    def test_removed_decoder_tp_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "decoder_tp.*decoder_sp"):
            self._from_dict({"model_path": "/fake", "decoder_tp": 2})

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
                "decoder_sp": 4,
            }
        )
        from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType

        self.assertEqual(args.get_role_parallelism(RoleType.ENCODER)["tp_size"], 1)
        self.assertEqual(args.get_role_parallelism(RoleType.DENOISER)["tp_size"], 2)
        self.assertEqual(args.get_role_parallelism(RoleType.DECODER)["sp_degree"], 4)

    def test_disagg_args_import_path_matches_server_args_package(self):
        from sglang.multimodal_gen.runtime.disaggregation import disagg_args
        from sglang.multimodal_gen.runtime.server_args.disagg import (
            DisaggServerArgsMixin,
        )

        self.assertIs(disagg_args.DisaggArgsMixin, DisaggServerArgsMixin)
        self.assertIs(
            disagg_args.DISAGG_RESULT_PORT_OFFSETS,
            DisaggServerArgsMixin.DISAGG_RESULT_PORT_OFFSETS,
        )

    def test_gpu_ids_normalize_lists_and_commas(self):
        args = self._from_dict({"model_path": "/fake", "gpu_ids": ["0,1", "6", "7 8"]})

        self.assertEqual(args.gpu_ids, [0, 1, 6, 7, 8])

    def test_gpu_ids_reject_duplicates(self):
        with self.assertRaisesRegex(ValueError, "duplicate GPU ids"):
            self._from_dict({"model_path": "/fake", "gpu_ids": ["0,1", "1"]})

    def test_pool_endpoints_use_role_and_scheduler_ports(self):
        args = self._from_dict(
            {
                "model_path": "/fake",
                "disagg_role": "denoiser",
                "disagg_server_addr": "tcp://127.0.0.1:30000",
                "scheduler_port": 5600,
                "host": "0.0.0.0",
                "disagg_p2p_hostname": "10.0.0.7",
            }
        )

        self.assertEqual(args.derive_pool_result_endpoint(), "tcp://127.0.0.1:30002")
        self.assertEqual(
            args.derive_pool_work_endpoint(),
            f"tcp://0.0.0.0:{args.scheduler_port}",
        )
        self.assertEqual(
            args.derive_pool_control_endpoint(),
            f"tcp://0.0.0.0:{args.scheduler_port + 1}",
        )
        self.assertEqual(
            args.derive_pool_control_advertised_endpoint(),
            f"tcp://10.0.0.7:{args.scheduler_port + 1}",
        )

    def test_pool_result_endpoint_validates_addr_and_role(self):
        args = self._from_dict({"model_path": "/fake", "disagg_server_addr": "bad"})
        with self.assertRaisesRegex(ValueError, "disagg_server_addr must be"):
            args.derive_pool_result_endpoint()

        args = self._from_dict(
            {"model_path": "/fake", "disagg_server_addr": "127.0.0.1:30000"}
        )
        with self.assertRaisesRegex(ValueError, "only defined for encoder"):
            args.derive_pool_result_endpoint()

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
            "--decoder-sp",
            "8",
        ]
        args, unknown = parser.parse_known_args(argv)
        self.assertEqual(args.denoiser_tp, 2)
        self.assertEqual(args.denoiser_sp, 4)
        self.assertEqual(args.denoiser_ulysses, 2)
        self.assertEqual(args.denoiser_ring, 2)
        self.assertEqual(args.encoder_tp, 1)
        self.assertEqual(args.decoder_sp, 8)


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

        with (
            patch.object(sys, "argv", ["sglang"] + argv),
            _mock_cuda_platform(),
        ):
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

        with (
            patch.object(sys, "argv", ["sglang"] + argv),
            _mock_cuda_platform(),
        ):
            args, unknown_args = parser.parse_known_args(argv)
            server_args = ServerArgs.from_cli_args(args, unknown_args)

        self.assertTrue(server_args.pipeline_config.disable_autocast)
        self.assertTrue(server_args.disable_autocast)


class TestDisaggTimeoutArgs(unittest.TestCase):
    def test_disagg_defaults_match_reviewed_values(self):
        args = _from_dict_without_model_resolution({"model_path": "/fake"})
        self.assertEqual(args.disagg_max_slots_per_instance, 8)
        self.assertEqual(args.disagg_downstream_wait_timeout, 1800)
        self.assertEqual(args.disagg_timeout, 3600)

    def test_downstream_wait_timeout_cli_arg_is_parsed(self):
        parser = FlexibleArgumentParser()
        ServerArgs.add_cli_args(parser)
        argv = [
            "--model-path",
            "/fake",
            "--disagg-downstream-wait-timeout",
            "45",
        ]

        args, _unknown = parser.parse_known_args(argv)
        self.assertEqual(args.disagg_downstream_wait_timeout, 45)

    def test_disagg_timeout_help_uses_current_defaults(self):
        parser = FlexibleArgumentParser()
        ServerArgs.add_cli_args(parser)
        help_text = parser.format_help()

        self.assertIn("Default: 3600.", help_text)
        self.assertIn("Default: 1800.", help_text)

    def test_disagg_role_alias_cli_arg_is_accepted(self):
        parser = FlexibleArgumentParser()
        ServerArgs.add_cli_args(parser)
        args, _unknown = parser.parse_known_args(
            ["--model-path", "/fake", "--disagg-role", "denoising"]
        )

        self.assertEqual(args.disagg_role, "denoising")

    def test_disagg_role_alias_normalizes_to_denoiser(self):
        from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType

        args = _from_dict_without_model_resolution(
            {"model_path": "/fake", "disagg_role": "denoising"}
        )

        self.assertEqual(args.disagg_role, RoleType.DENOISER)


class TestSchedulerRpcTimeoutArgs(unittest.TestCase):
    def test_scheduler_rpc_timeout_defaults_to_unbounded(self):
        args = _from_dict_without_model_resolution({"model_path": "/fake"})
        self.assertIsNone(args.scheduler_rpc_timeout)

    def test_scheduler_rpc_timeout_cli_arg_is_parsed_in_seconds(self):
        parser = FlexibleArgumentParser()
        ServerArgs.add_cli_args(parser)
        argv = [
            "--model-path",
            "/fake",
            "--scheduler-rpc-timeout",
            "7200",
        ]

        args, _unknown = parser.parse_known_args(argv)
        self.assertEqual(args.scheduler_rpc_timeout, 7200)

    def test_scheduler_rpc_timeout_rejects_invalid_values(self):
        invalid_values = (0, -1, MAX_SCHEDULER_RPC_TIMEOUT_S + 1, True, 1.5, "1")

        for invalid_value in invalid_values:
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaisesRegex(
                    ValueError, "scheduler_rpc_timeout must be None"
                ):
                    _from_dict_without_model_resolution(
                        {
                            "model_path": "/fake",
                            "scheduler_rpc_timeout": invalid_value,
                        }
                    )


class TestDisaggTransferBackendArgs(unittest.TestCase):
    def test_transfer_backend_defaults_to_auto(self):
        args = _from_dict_without_model_resolution({"model_path": "/fake"})
        self.assertEqual(args.disagg_transfer_backend, "auto")

    def test_transfer_backend_cli_arg_is_parsed(self):
        parser = FlexibleArgumentParser()
        ServerArgs.add_cli_args(parser)
        argv = [
            "--model-path",
            "/fake",
            "--disagg-transfer-backend",
            "mock",
        ]

        args, _unknown = parser.parse_known_args(argv)
        self.assertEqual(args.disagg_transfer_backend, "mock")


class TestNcclNvlsArgs(unittest.TestCase):
    def test_enable_nccl_nvls_cli_arg(self):
        parser = FlexibleArgumentParser()
        ServerArgs.add_cli_args(parser)

        default_args, _ = parser.parse_known_args(["--model-path", "/fake"])
        enabled_args, _ = parser.parse_known_args(
            ["--model-path", "/fake", "--enable-nccl-nvls"]
        )
        disabled_args, _ = parser.parse_known_args(
            ["--model-path", "/fake", "--enable-nccl-nvls", "false"]
        )

        self.assertFalse(default_args.enable_nccl_nvls)
        self.assertTrue(enabled_args.enable_nccl_nvls)
        self.assertFalse(disabled_args.enable_nccl_nvls)


class TestDirectGpuWeightLoading(unittest.TestCase):
    def _args(self) -> ServerArgs:
        args = ServerArgs.__new__(ServerArgs)
        args.direct_gpu_weight_loading = True
        args.component_residency = None
        args.cpu_offload_components = None
        args.dit_cpu_offload = False
        args.dit_layerwise_offload = False
        args.layerwise_offload_components = []
        args.text_encoder_cpu_offload = False
        args.image_encoder_cpu_offload = False
        args.vae_cpu_offload = False
        args.use_fsdp_inference = False
        args.tp_size = 1
        args._explicit_arg_names = set()
        args._required_resident_components = set()
        args._component_layerwise_capabilities = {}
        return args

    def test_cli_defaults_off_and_parses_explicit_enable(self):
        parser = FlexibleArgumentParser()
        ServerArgs.add_cli_args(parser)

        default_args, _ = parser.parse_known_args(["--model-path", "/fake"])
        enabled_args, _ = parser.parse_known_args(
            ["--model-path", "/fake", "--direct-gpu-weight-loading"]
        )

        self.assertFalse(default_args.direct_gpu_weight_loading)
        self.assertTrue(enabled_args.direct_gpu_weight_loading)

    def test_rejects_cpu_offload_fsdp_and_tp(self):
        cpu_offload_args = self._args()
        cpu_offload_args.dit_cpu_offload = True
        fsdp_args = self._args()
        fsdp_args.use_fsdp_inference = True
        tp_args = self._args()
        tp_args.tp_size = 2

        with patch.object(current_platform, "is_cuda", return_value=True):
            with self.assertRaisesRegex(ValueError, "GPU-resident DiT"):
                cpu_offload_args._validate_direct_gpu_weight_loading()
            with self.assertRaisesRegex(ValueError, "FSDP"):
                fsdp_args._validate_direct_gpu_weight_loading()
            with self.assertRaisesRegex(ValueError, "tp-size 1"):
                tp_args._validate_direct_gpu_weight_loading()


if __name__ == "__main__":
    unittest.main()
