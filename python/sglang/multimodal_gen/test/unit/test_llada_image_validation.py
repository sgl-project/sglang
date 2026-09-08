# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from fastapi import HTTPException

from sglang.multimodal_gen.configs.pipeline_configs.llada_image import (
    LLaDAImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.image_api import (
    _build_sampling_params_or_400,
    _early_validate_edit_bounds,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    ImageGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.models.dits.llada_image import (
    LLaDAImageRopeEmbedder,
    _LLaDAImageTransformer2DModel,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.llada_image.conditioning import (
    _resolve_llada_image_component_path,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class TestLLaDAImageValidation(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model_root = Path(self.directory.name)
        (self.model_root / "model_index.json").write_text(
            json.dumps(
                {"_class_name": "LLaDAImagePipeline", "_diffusers_version": "0.33.1"}
            )
        )
        for component in ("transformer", "vae"):
            (self.model_root / component).mkdir()
        self.config = LLaDAImagePipelineConfig()

    def server_args(self, **overrides):
        defaults = dict(
            model_path=str(self.model_root),
            pipeline_class_name="LLaDAImagePipeline",
            pipeline_config=self.config,
            performance_mode="manual",
            num_gpus=1,
            tp_size=1,
            sp_degree=1,
            ulysses_degree=1,
            ring_degree=1,
            enable_cfg_parallel=False,
            dit_cpu_offload=False,
            text_encoder_cpu_offload=False,
            image_encoder_cpu_offload=False,
            vae_cpu_offload=False,
            attention_backend="torch_sdpa",
            log_level="error",
        )
        return ServerArgs.from_kwargs(**(defaults | overrides))

    def sampling_params(self, args, width, height):
        request = ImageGenerationsRequest(
            prompt="a red cube", size=f"{width}x{height}", seed=42
        )
        with patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.utils.get_global_server_args",
            return_value=args,
        ):
            return _build_sampling_params_or_400(
                "llada-image-validation",
                prompt=request.prompt,
                size=request.size,
                seed=request.seed,
                num_outputs_per_prompt=1,
                generator_device="cpu",
                save_output=False,
            )

    def test_rejects_normalized_embedded_text_encoder_overrides(self):
        weight_path = str(self.model_root / "encoder.safetensors")
        overrides = [
            {"component_paths": {"text_encoder": weight_path}},
            {"component_weights_paths": {"text_encoder": weight_path}},
            {"component_precisions": {"text-encoder": "fp32"}},
            {"component_quantizations": {"text-encoder": "FP8"}},
            {"component_attention_backends": "text-encoder=torch_sdpa"},
            {"component_direct_gpu_weight_loading": {"text-encoder": True}},
        ]
        for override in overrides:
            cuda_capability = (
                patch(
                    "sglang.multimodal_gen.runtime.server_args.server_args.current_platform.is_cuda",
                    return_value=True,
                )
                if "component_direct_gpu_weight_loading" in override
                else nullcontext()
            )
            with (
                self.subTest(override=override),
                cuda_capability,
                self.assertRaisesRegex(
                    ValueError, "embedded text_encoder does not support component_"
                ),
            ):
                self.server_args(**override)

    def test_preserves_encoder_directory_and_auxiliary_overrides(self):
        encoder_path = self.model_root / "alternate_encoder"
        encoder_path.mkdir()
        (encoder_path / "config.json").write_text("{}")
        auxiliary_paths = {
            name: str(self.model_root / f"{name}.safetensors")
            for name in ("queryformer", "text_projection", "sigvq")
        }
        args = self.server_args(
            component_paths={"text_encoder": str(encoder_path)} | auxiliary_paths,
            component_precisions={"queryformer": "fp32"},
            component_quantizations={"sigvq": "fp8"},
            component_attention_backends={"queryformer": "torch_sdpa"},
        )

        self.assertEqual(args.component_weights_paths, auxiliary_paths)
        self.assertEqual(args.component_precisions, {"queryformer": "fp32"})
        self.assertEqual(args.component_quantizations, {"sigvq": "fp8"})
        self.assertEqual(
            args.component_attention_backends, {"queryformer": "torch_sdpa"}
        )
        self.assertEqual(
            _resolve_llada_image_component_path(
                str(self.model_root), args, "text_encoder"
            ),
            str(encoder_path),
        )

    def test_rejects_disaggregation_during_server_args_construction(self):
        args = self.server_args(disagg_role="monolithic")
        self.assertEqual(args.disagg_role.value, "monolithic")
        for role in ("encoder", "denoiser", "decoder"):
            with (
                self.subTest(role=role),
                self.assertRaisesRegex(
                    ValueError, "only supports monolithic deployment"
                ),
            ):
                self.server_args(disagg_role=role)

    def test_rejects_each_spatial_axis_before_scheduling_or_edit_loading(self):
        for sp_degree in (1, 2):
            args = self.server_args(
                num_gpus=sp_degree,
                sp_degree=sp_degree,
                ulysses_degree=sp_degree,
            )
            stride = self.config.latent_scale_factor
            max_height = self.config.dit_config.arch_config.axes_lens[1] * stride
            max_width = self.config.dit_config.arch_config.axes_lens[2] * stride
            for width, height in (
                (max_width + stride, stride * sp_degree),
                (stride, max_height + stride * sp_degree),
            ):
                with self.subTest(sp_degree=sp_degree, width=width, height=height):
                    with self.assertRaises(HTTPException) as generation:
                        self.sampling_params(args, width, height)
                    self.assertEqual(generation.exception.status_code, 400)
                    self.assertIn("RoPE", generation.exception.detail)
                    with self.assertRaises(HTTPException) as editing:
                        _early_validate_edit_bounds(args, f"{width}x{height}")
                    self.assertEqual(editing.exception.status_code, 400)
                    self.assertIn("RoPE", editing.exception.detail)

    def test_spatial_limits_follow_checkpoint_config_before_latent_allocation(self):
        self.config.dit_config.arch_config.axes_lens = (32768, 4, 8)
        args = self.server_args()
        for width, height in ((128, 64), (128, 16), (16, 64)):
            with self.subTest(width=width, height=height):
                self.sampling_params(args, width, height)
                shape = self.config.prepare_latent_shape(
                    SimpleNamespace(width=width, height=height), 1, 1
                )
                self.assertEqual(shape[-2:], (height // 16, width // 16))
        for width, height in ((144, 16), (16, 80)):
            with self.subTest(width=width, height=height):
                with self.assertRaisesRegex(ValueError, "RoPE"):
                    self.config.prepare_latent_shape(
                        SimpleNamespace(width=width, height=height), 1, 1
                    )
                with self.assertRaises(HTTPException) as context:
                    self.sampling_params(args, width, height)
                self.assertEqual(context.exception.status_code, 400)
                self.assertIn("RoPE", context.exception.detail)

    def test_exact_spatial_boundaries_fit_rope_with_global_sp_positions(self):
        model = object.__new__(_LLaDAImageTransformer2DModel)
        rope = LLaDAImageRopeEmbedder(256.0, (2, 2, 2), (32768, 1024, 1024))
        module = "sglang.multimodal_gen.runtime.models.dits.llada_image"
        for sp_degree in (1, 2):
            args = self.server_args(
                num_gpus=sp_degree,
                sp_degree=sp_degree,
                ulysses_degree=sp_degree,
            )
            for width, height in ((16384, 16 * sp_degree), (16, 16384)):
                with self.subTest(sp_degree=sp_degree, width=width, height=height):
                    sampling = self.sampling_params(args, width, height)
                    batch = prepare_request(args, sampling)
                    shape = self.config.prepare_latent_shape(batch, 1, 1)
                    latent = torch.zeros(shape[1], 1, shape[2] // sp_degree, shape[3])
                    with (
                        patch(f"{module}.get_sp_world_size", return_value=sp_degree),
                        patch(
                            f"{module}.get_sp_parallel_rank",
                            return_value=sp_degree - 1,
                        ),
                    ):
                        image_sequence, _, _, _ = model._prepare_t2i_sequences(
                            [latent], [torch.zeros(1, 1)], None, 1, 1
                        )
                    position_ids = image_sequence.position_ids[0]
                    spatial_axis = 2 if width == 16384 else 1
                    self.assertEqual(position_ids[:, spatial_axis].max().item(), 1023)
                    self.assertTrue(torch.isfinite(rope(position_ids)).all())


if __name__ == "__main__":
    unittest.main()
