import json
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
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
from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import LTX2PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.mova import MOVAPipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
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
from sglang.multimodal_gen.registry import _get_config_info
from sglang.multimodal_gen.runtime.models.dits.qwen_image import (
    QwenImageTransformer2DModel,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
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
            "sglang.multimodal_gen.runtime.platforms.current_platform.enable_dit_layerwise_offload_for_wan_by_default",
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
                "sglang.multimodal_gen.runtime.server_args.current_platform.is_cpu",
                return_value=False,
            ),
            patch(
                "sglang.multimodal_gen.runtime.server_args.current_platform.is_mps",
                return_value=False,
            ),
            patch(
                "sglang.multimodal_gen.runtime.server_args.current_platform.is_cuda",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.server_args.current_platform.get_device_total_memory",
                return_value=80 * 1024**3,
            ),
            patch(
                "sglang.multimodal_gen.runtime.server_args.current_platform.get_available_gpu_memory",
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

    def test_serve_cli_preserves_config_and_dynamic_unknown_args(self):
        from sglang.multimodal_gen.runtime.entrypoints.cli.serve import (
            add_multimodal_gen_serve_args,
        )

        with tempfile.NamedTemporaryFile("w", suffix=".json") as config_file:
            json.dump({"model_path": "/from/config", "num_gpus": 2}, config_file)
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
                        "sglang.multimodal_gen.runtime.server_args.current_platform.get_device_total_memory",
                        return_value=80 * 1024**3,
                    ),
                    patch(
                        "sglang.multimodal_gen.runtime.server_args.current_platform.get_available_gpu_memory",
                        return_value=80,
                    ),
                ):
                    server_args = ServerArgs.from_cli_args(args, unknown_args)

        self.assertEqual("/from/cli", server_args.model_path)
        self.assertEqual(2, server_args.num_gpus)
        self.assertEqual("/custom/vae", server_args.component_paths["vae"])
        self.assertEqual(
            {"transformer": "fa"},
            server_args.component_attention_backends,
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
        self.assertTrue(server_args.warmup)
        self.assertTrue(server_args.server_warmup)
        self.assertFalse(server_args.is_arg_explicitly_set("warmup"))
        self.assertFalse(server_args.is_arg_explicitly_set("server_warmup"))

    def test_serve_cli_preserves_explicit_warmup_false(self):
        from sglang.multimodal_gen.runtime.entrypoints.cli.serve import (
            add_multimodal_gen_serve_args,
            execute_serve_cmd,
        )

        parser = FlexibleArgumentParser()
        add_multimodal_gen_serve_args(parser)
        argv = [
            "--model-path",
            "/fake",
            "--warmup",
            "false",
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
        self.assertFalse(server_args.warmup)
        self.assertFalse(server_args.server_warmup)
        self.assertTrue(server_args.is_arg_explicitly_set("warmup"))

    def test_serve_cli_preserves_config_warmup_false(self):
        from sglang.multimodal_gen.runtime.entrypoints.cli.serve import (
            add_multimodal_gen_serve_args,
            execute_serve_cmd,
        )

        with tempfile.NamedTemporaryFile("w", suffix=".json") as config_file:
            json.dump({"model_path": "/fake", "warmup": False}, config_file)
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
        self.assertFalse(server_args.warmup)
        self.assertFalse(server_args.server_warmup)
        self.assertTrue(server_args.is_arg_explicitly_set("warmup"))

    def test_disagg_role_disables_server_warmup(self):
        with patch.object(
            PipelineConfig, "from_kwargs", return_value=QwenImagePipelineConfig()
        ):
            server_args = ServerArgs.from_dict(
                {
                    "model_path": "/fake",
                    "warmup": True,
                    "server_warmup": True,
                    "disagg_role": "server",
                }
            )

        self.assertTrue(server_args.warmup)
        self.assertFalse(server_args.server_warmup)


class TestWarmupModeNormalization(unittest.TestCase):
    """`_adjust_warmup` resolves the canonical warmup_mode and its derived booleans."""

    def _resolve(
        self,
        *,
        warmup_mode=None,
        warmup=False,
        server_warmup=False,
        warmup_resolutions=None,
        disagg_role=None,
        explicit=(),
    ):
        from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType

        sa = ServerArgs.__new__(ServerArgs)
        sa.warmup_mode = warmup_mode
        sa.warmup = warmup
        sa.server_warmup = server_warmup
        sa.warmup_resolutions = warmup_resolutions
        sa.disagg_role = RoleType.MONOLITHIC if disagg_role is None else disagg_role
        sa._explicit_arg_names = set(explicit)
        sa._adjust_warmup()
        return sa

    def test_explicit_mode_off_disables_all(self):
        sa = self._resolve(warmup_mode="off", explicit=("warmup_mode",))
        self.assertEqual(sa.warmup_mode, "off")
        self.assertFalse(sa.warmup)
        self.assertFalse(sa.server_warmup)

    def test_explicit_mode_request(self):
        sa = self._resolve(warmup_mode="request", explicit=("warmup_mode",))
        self.assertEqual(sa.warmup_mode, "request")
        self.assertTrue(sa.warmup)
        self.assertFalse(sa.server_warmup)

    def test_explicit_mode_server(self):
        sa = self._resolve(warmup_mode="server", explicit=("warmup_mode",))
        self.assertEqual(sa.warmup_mode, "server")
        self.assertTrue(sa.warmup)
        self.assertTrue(sa.server_warmup)

    def test_explicit_mode_overrides_explicit_legacy(self):
        sa = self._resolve(
            warmup_mode="request",
            warmup=True,
            server_warmup=True,
            explicit=("warmup_mode", "warmup", "server_warmup"),
        )
        self.assertEqual(sa.warmup_mode, "request")
        self.assertTrue(sa.warmup)
        self.assertFalse(sa.server_warmup)

    def test_explicit_legacy_false_beats_defaulted_mode(self):
        # serve defaults warmup_mode="server" (not explicit); `--warmup false` wins.
        sa = self._resolve(
            warmup_mode="server",
            warmup=False,
            server_warmup=False,
            explicit=("warmup",),
        )
        self.assertEqual(sa.warmup_mode, "off")
        self.assertFalse(sa.warmup)
        self.assertFalse(sa.server_warmup)

    def test_defaulted_mode_applies_without_legacy_flags(self):
        # bare `sglang serve`: warmup_mode="server" defaulted, no legacy override.
        sa = self._resolve(warmup_mode="server")
        self.assertEqual(sa.warmup_mode, "server")
        self.assertTrue(sa.warmup)
        self.assertTrue(sa.server_warmup)

    def test_legacy_only_maps_to_request(self):
        sa = self._resolve(warmup_mode=None, warmup=True, explicit=("warmup",))
        self.assertEqual(sa.warmup_mode, "request")
        self.assertTrue(sa.warmup)
        self.assertFalse(sa.server_warmup)

    def test_resolutions_force_warmup_on(self):
        sa = self._resolve(
            warmup_mode="off",
            warmup_resolutions=["512x512"],
            explicit=("warmup_mode",),
        )
        self.assertTrue(sa.warmup)
        self.assertFalse(sa.server_warmup)
        self.assertEqual(sa.warmup_mode, "request")

    def test_disagg_role_disables_server_warmup(self):
        from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType

        sa = self._resolve(
            warmup_mode="server",
            disagg_role=RoleType.DENOISER,
            explicit=("warmup_mode",),
        )
        self.assertTrue(sa.warmup)
        self.assertFalse(sa.server_warmup)
        self.assertEqual(sa.warmup_mode, "request")

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            self._resolve(warmup_mode="bogus", explicit=("warmup_mode",))


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


class TestOffloadDefaults(unittest.TestCase):
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
                "sglang.multimodal_gen.runtime.server_args.current_platform.is_cpu",
                return_value=False,
            ),
            patch(
                "sglang.multimodal_gen.runtime.server_args.current_platform.is_mps",
                return_value=False,
            ),
            patch(
                "sglang.multimodal_gen.runtime.server_args.current_platform.is_cuda",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.server_args.current_platform.enable_dit_layerwise_offload_for_wan_by_default",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.server_args.current_platform.get_device_total_memory",
                return_value=memory_gb * 1024**3,
            ),
            patch(
                "sglang.multimodal_gen.runtime.server_args.current_platform.get_available_gpu_memory",
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
                "sglang.multimodal_gen.runtime.server_args.current_platform.is_cpu",
                return_value=False,
            ),
            patch(
                "sglang.multimodal_gen.runtime.server_args.current_platform.is_cuda",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.server_args.current_platform.get_device_total_memory",
                return_value=memory_gb * 1024**3,
            ),
            patch(
                "sglang.multimodal_gen.runtime.server_args.current_platform.get_available_gpu_memory",
                return_value=memory_gb,
            ),
        ):
            return ServerArgs.from_dict({"model_path": "/fake", **(kwargs or {})})

    def test_vae_cpu_offload_defaults_false_for_video_generation(self):
        args = self._from_dict_with_task_type(ModelTaskType.T2V)

        self.assertFalse(args.vae_cpu_offload)

    def test_vae_cpu_offload_defaults_false_on_low_memory_gpu(self):
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

    def test_layerwise_components_disable_matching_non_dit_cpu_offloads(self):
        args = self._from_dict_with_task_type(
            ModelTaskType.T2V,
            memory_gb=16,
            kwargs={
                "performance_mode": "manual",
                "dit_cpu_offload": True,
                "text_encoder_cpu_offload": True,
                "image_encoder_cpu_offload": True,
                "vae_cpu_offload": True,
            },
        )
        args.layerwise_offload_components = [
            "text_encoder",
            "image_encoder",
            "video_dit",
            "vae",
        ]
        args._adjust_layerwise_offload_components()

        self.assertTrue(args.layerwise_offload_components)
        # dit_cpu_offload is complementary to DiT layerwise offload (keeps
        # weights off-device during load), so it must be preserved here.
        self.assertTrue(args.dit_cpu_offload)
        self.assertFalse(args.text_encoder_cpu_offload)
        self.assertFalse(args.image_encoder_cpu_offload)
        self.assertFalse(args.vae_cpu_offload)

    def test_dit_layerwise_offload_preserves_dit_cpu_offload(self):
        """Combining --dit-cpu-offload with --dit-layerwise-offload must keep both on.

        dit_cpu_offload controls initial residency (host memory), while
        dit_layerwise_offload only swaps layers on/off device at inference.
        Force-disabling dit_cpu_offload here would push the full DiT to GPU at
        load time and OOM low-VRAM cards.
        """
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

    def test_pipeline_configs_declare_auto_tune_hints(self):
        qwen_deployment = QwenImagePipelineConfig().get_model_deployment_config()
        wan_deployment = WanT2V480PConfig().get_model_deployment_config()
        mova_deployment = MOVAPipelineConfig().get_model_deployment_config()
        zimage_deployment = ZImagePipelineConfig().get_model_deployment_config()
        ltx_deployment = LTX2PipelineConfig().get_model_deployment_config()
        sana_wm_deployment = SanaWMPipelineConfig().get_model_deployment_config()

        self.assertIsNone(qwen_deployment.fsdp_auto_min_available_memory_gb)
        self.assertFalse(qwen_deployment.auto_dit_layerwise_offload)

        self.assertIsNone(wan_deployment.fsdp_auto_min_available_memory_gb)
        self.assertTrue(wan_deployment.auto_dit_layerwise_offload)

        self.assertIsNone(mova_deployment.fsdp_auto_min_available_memory_gb)
        self.assertTrue(mova_deployment.auto_dit_layerwise_offload)

        self.assertEqual(zimage_deployment.fsdp_auto_min_available_memory_gb, 40)
        self.assertTrue(zimage_deployment.fsdp_auto_requires_cfg)
        self.assertFalse(zimage_deployment.auto_dit_layerwise_offload)

        self.assertEqual(
            ltx_deployment.auto_disable_component_offload_min_available_memory_gb, 70
        )
        self.assertEqual(
            ltx_deployment.auto_disable_component_offload_components, ("dit",)
        )

        self.assertEqual(sana_wm_deployment.fsdp_auto_min_available_memory_gb, 60)
        self.assertTrue(sana_wm_deployment.auto_dit_layerwise_offload)

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

    def test_default_auto_replaces_text_encoder_cpu_offload_with_layerwise(self):
        args = self._from_dict_with_pipeline_config(
            QwenImagePipelineConfig(),
            kwargs={"model_path": "Qwen/Qwen-Image"},
        )

        self.assertEqual(args.performance_mode, "auto")
        self.assertFalse(args.use_fsdp_inference)
        self.assertTrue(args.dit_cpu_offload)
        self.assertTrue(args.layerwise_offload_components)
        self.assertFalse(args.text_encoder_cpu_offload)
        self.assertFalse(args.image_encoder_cpu_offload)
        self.assertEqual(
            args.layerwise_offload_components,
            ["text_encoder", "image_encoder", "vae"],
        )

    def test_auto_ltx_snapshot_keeps_dit_offload_and_replaces_encoder_cpu_offload(
        self,
    ):
        args = self._from_dict_with_pipeline_config(
            LTX2PipelineConfig(),
            available_memory_gb=76,
            kwargs={
                "model_path": "Lightricks/LTX-2.3",
                "pipeline_class_name": "LTX2TwoStageHQPipeline",
                "ltx2_two_stage_device_mode": "snapshot",
                "performance_mode": "auto",
            },
        )

        self.assertEqual(args.ltx2_two_stage_device_mode, "snapshot")
        self.assertTrue(args.dit_cpu_offload)
        self.assertTrue(args.layerwise_offload_components)
        self.assertFalse(args.text_encoder_cpu_offload)
        self.assertFalse(args.image_encoder_cpu_offload)
        self.assertEqual(
            args.layerwise_offload_components,
            ["text_encoder", "image_encoder", "vae"],
        )

    def test_auto_wan_layerwise_offload_is_enabled_without_fsdp(self):
        args = self._from_dict_with_pipeline_config(
            WanT2V480PConfig(),
            kwargs={"performance_mode": "auto"},
        )

        self.assertTrue(args.layerwise_offload_components)
        self.assertFalse(args.use_fsdp_inference)
        self.assertTrue(args.dit_cpu_offload)
        self.assertFalse(args.text_encoder_cpu_offload)
        self.assertFalse(args.image_encoder_cpu_offload)
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
                # dit_cpu_offload is complementary to DiT layerwise offload:
                # layerwise only moves layers on/off device at runtime, while
                # dit_cpu_offload keeps the initial weights on host memory.
                self.assertTrue(args.dit_cpu_offload)
                self.assertFalse(args.text_encoder_cpu_offload)
                self.assertFalse(args.image_encoder_cpu_offload)
                self.assertEqual(args.dit_offload_prefetch_size, 2)
                self.assertEqual(
                    args.layerwise_offload_components,
                    ["dit", "text_encoder", "image_encoder", "vae"],
                )

    def test_auto_wan2_1_14b_layerwise_offload_uses_non_dit_default(self):
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
                self.assertTrue(args.dit_cpu_offload)
                self.assertEqual(args.dit_offload_prefetch_size, 0.0)
                self.assertEqual(
                    args.layerwise_offload_components,
                    ["text_encoder", "image_encoder", "vae"],
                )

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

    def test_auto_mova_layerwise_offload_does_not_implicitly_add_dit(self):
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
            ["text_encoder", "image_encoder", "vae"],
        )

    def test_auto_fastwan_layerwise_offload_does_not_implicitly_add_dit(self):
        args = self._from_dict_with_pipeline_config(
            FastWan2_2_TI2V_5B_Config(),
            kwargs={
                "model_path": "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
                "performance_mode": "auto",
            },
        )

        self.assertTrue(args.dit_cpu_offload)
        self.assertEqual(
            args.layerwise_offload_components,
            ["text_encoder", "image_encoder", "vae"],
        )

    def test_auto_turbo_wan_layerwise_offload_does_not_implicitly_add_dit(self):
        args = self._from_dict_with_pipeline_config(
            TurboWanT2V480PConfig(),
            kwargs={
                "model_path": "IPostYellow/TurboWan2.1-T2V-1.3B-Diffusers",
                "performance_mode": "auto",
            },
        )

        self.assertTrue(args.dit_cpu_offload)
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

        # dit_cpu_offload defaults to True from _adjust_offload and is now
        # preserved alongside DiT layerwise offload (the two are complementary).
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
        self.assertTrue(args.dit_cpu_offload)
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

    def test_auto_multi_gpu_qwen_replaces_text_encoder_offload_with_cfg(self):
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
        self.assertTrue(args.dit_cpu_offload)
        self.assertTrue(args.layerwise_offload_components)
        self.assertFalse(args.text_encoder_cpu_offload)
        self.assertFalse(args.image_encoder_cpu_offload)
        self.assertEqual(
            args.layerwise_offload_components,
            ["text_encoder", "image_encoder", "vae"],
        )

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
        self.assertTrue(args.dit_cpu_offload)
        self.assertFalse(args.text_encoder_cpu_offload)
        self.assertFalse(args.image_encoder_cpu_offload)
        self.assertEqual(
            args.layerwise_offload_components,
            ["text_encoder", "image_encoder", "vae"],
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
        self.assertTrue(args.dit_cpu_offload)
        self.assertFalse(args.text_encoder_cpu_offload)
        self.assertFalse(args.image_encoder_cpu_offload)
        self.assertEqual(
            args.layerwise_offload_components,
            ["text_encoder", "image_encoder", "vae"],
        )

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

    def test_auto_multi_gpu_qwen_replaces_text_encoder_offload_with_headroom(self):
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
        self.assertTrue(args.dit_cpu_offload)
        self.assertFalse(args.text_encoder_cpu_offload)
        self.assertFalse(args.image_encoder_cpu_offload)
        self.assertEqual(
            args.layerwise_offload_components,
            ["text_encoder", "image_encoder", "vae"],
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
                "sglang.multimodal_gen.runtime.server_args.current_platform.is_cpu",
                return_value=False,
            ),
            patch(
                "sglang.multimodal_gen.runtime.server_args.current_platform.is_mps",
                return_value=False,
            ),
            patch(
                "sglang.multimodal_gen.runtime.server_args.current_platform.is_cuda",
                return_value=True,
            ),
            patch(
                "sglang.multimodal_gen.runtime.server_args.current_platform.get_device_total_memory",
                return_value=80 * 1024**3,
            ),
            patch(
                "sglang.multimodal_gen.runtime.server_args.current_platform.get_available_gpu_memory",
                return_value=80,
            ),
        ):
            args, unknown_args = parser.parse_known_args(argv)
            server_args = ServerArgs.from_cli_args(args, unknown_args)

        self.assertFalse(server_args.use_fsdp_inference)
        self.assertFalse(server_args.enable_cfg_parallel)


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

    def test_decoder_tp_is_alias_of_decoder_sp(self):
        args = self._from_dict({"model_path": "/fake", "decoder_tp": 2})
        from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType

        self.assertEqual(args.decoder_sp, 2)
        par = args.get_role_parallelism(RoleType.DECODER)
        self.assertIsNone(par["tp_size"])
        self.assertEqual(par["sp_degree"], 2)

    def test_conflicting_decoder_tp_and_decoder_sp_raise(self):
        with self.assertRaisesRegex(ValueError, "decoder_tp is deprecated"):
            self._from_dict(
                {
                    "model_path": "/fake",
                    "decoder_tp": 2,
                    "decoder_sp": 4,
                }
            )

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

    def test_disagg_args_import_path_stays_compatible(self):
        from sglang.multimodal_gen.runtime.disaggregation import disagg_args
        from sglang.multimodal_gen.runtime.server_args_disagg import (
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


if __name__ == "__main__":
    unittest.main()
