# SPDX-License-Identifier: Apache-2.0
"""
Hunyuan3D image-to-mesh pipeline implementation.

This module provides the main pipeline class for Hunyuan3D 3D generation,
composing shape generation and optional texture painting stages.
"""

from __future__ import annotations

from typing import Any

from sglang.multimodal_gen.configs.pipeline_configs.hunyuan3d import (
    Hunyuan3D2PipelineConfig,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    Hunyuan3DInputStage,
    Hunyuan3DPaintDelightStage,
    Hunyuan3DPaintDiffusionStage,
    Hunyuan3DPaintPostprocessStage,
    Hunyuan3DPaintRenderStage,
    Hunyuan3DPaintUVUnwrapStage,
    Hunyuan3DShapeConditioningStage,
    Hunyuan3DShapeDenoisingStage,
    Hunyuan3DShapeExportStage,
    Hunyuan3DShapeLatentStage,
    Hunyuan3DShapeOnlyOutputStage,
    Hunyuan3DShapePreprocessStage,
    Hunyuan3DShapeSaveStage,
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class Hunyuan3D2Pipeline(ComposedPipelineBase):
    """Hunyuan3D 2.0 image-to-mesh pipeline.

    This pipeline generates 3D meshes from input images using:
    1. Shape generation stages: preprocess -> conditioning -> latent -> denoising -> export -> save
    2. Optional paint stages: preprocess -> render -> diffusion -> postprocess

    The pipeline supports two modes:
    - Shape only: Generates untextured 3D mesh
    - Shape + Paint: Generates textured 3D mesh with multi-view texture baking
    """

    pipeline_name = "Hunyuan3D2Pipeline"
    _required_config_modules = [
        "hy3dshape_model",
        "hy3dshape_vae",
        "hy3dshape_scheduler",
        "hy3dshape_conditioner",
        "hy3dshape_image_processor",
    ]

    def _load_config(self) -> dict[str, Any]:
        return {
            "_class_name": self.pipeline_name,
            "_diffusers_version": "0.0.0",
            "hy3dshape_model": ["diffusers", "Hunyuan3DShapeModel"],
            "hy3dshape_vae": ["diffusers", "Hunyuan3DShapeVAE"],
            "hy3dshape_scheduler": ["diffusers", "Hunyuan3DShapeScheduler"],
            "hy3dshape_conditioner": ["diffusers", "Hunyuan3DShapeConditioner"],
            "hy3dshape_image_processor": ["diffusers", "Hunyuan3DShapeImageProcessor"],
        }

    def initialize_pipeline(self, server_args: ServerArgs):
        config = server_args.pipeline_config
        if not isinstance(config, Hunyuan3D2PipelineConfig):
            raise TypeError(
                "Hunyuan3D2Pipeline requires Hunyuan3D2PipelineConfig, "
                f"got {type(config)}"
            )

        if config.paint_enable:
            self._initialize_paint_pipeline(config)

    def _initialize_paint_pipeline(self, config: Hunyuan3D2PipelineConfig):
        """Initialize the paint pipeline for texture generation.

        This sets up the new 5-stage texture generation pipeline:
        1. PaintUVUnwrapStage: UV unwrap mesh
        2. PaintDelightStage: Remove lighting from reference image
        3. PaintRenderStage: Multi-view normal/position rendering
        4. PaintDiffusionStage: Texture diffusion generation
        5. PaintPostprocessStage: Texture baking and export
        """
        logger.info(
            "Paint pipeline (texture generation) initialized with 5-stage architecture."
        )

    def create_pipeline_stages(self, server_args: ServerArgs):
        config = server_args.pipeline_config
        assert isinstance(config, Hunyuan3D2PipelineConfig)

        # Input validation
        self.add_stage(
            stage_name="input_validation_stage", stage=InputValidationStage()
        )
        self.add_stage(
            stage_name="input_stage", stage=Hunyuan3DInputStage(config=config)
        )

        # Shape generation stages
        self.add_stage(
            stage_name="shape_preprocess_stage",
            stage=Hunyuan3DShapePreprocessStage(
                image_processor=self.get_module("hy3dshape_image_processor"),
            ),
        )
        self.add_stage(
            stage_name="shape_conditioning_stage",
            stage=Hunyuan3DShapeConditioningStage(
                conditioner=self.get_module("hy3dshape_conditioner"),
                model=self.get_module("hy3dshape_model"),
            ),
        )
        self.add_stage(
            stage_name="shape_latent_stage",
            stage=Hunyuan3DShapeLatentStage(
                scheduler=self.get_module("hy3dshape_scheduler"),
                vae=self.get_module("hy3dshape_vae"),
                model=self.get_module("hy3dshape_model"),
            ),
        )
        self.add_stage(
            stage_name="shape_denoising_stage",
            stage=Hunyuan3DShapeDenoisingStage(
                transformer=self.get_module("hy3dshape_model"),
                scheduler=self.get_module("hy3dshape_scheduler"),
            ),
        )
        self.add_stage(
            stage_name="shape_export_stage",
            stage=Hunyuan3DShapeExportStage(
                vae=self.get_module("hy3dshape_vae"),
                config=config,
            ),
        )
        self.add_stage(
            stage_name="shape_save_stage", stage=Hunyuan3DShapeSaveStage(config)
        )

        # Paint stages (optional)
        if config.paint_enable:
            # New 5-stage texture generation pipeline
            self.add_stage(
                stage_name="paint_uv_unwrap_stage",
                stage=Hunyuan3DPaintUVUnwrapStage(config=config),
            )
            self.add_stage(
                stage_name="paint_delight_stage",
                stage=Hunyuan3DPaintDelightStage(config=config),
            )
            self.add_stage(
                stage_name="paint_render_stage",
                stage=Hunyuan3DPaintRenderStage(config=config),
            )
            self.add_stage(
                stage_name="paint_diffusion_stage",
                stage=Hunyuan3DPaintDiffusionStage(config=config),
            )
            self.add_stage(
                stage_name="paint_postprocess_stage",
                stage=Hunyuan3DPaintPostprocessStage(config=config),
            )
        else:
            self.add_stage(
                stage_name="paint_stage",
                stage=Hunyuan3DShapeOnlyOutputStage(config=config),
            )


EntryClass = Hunyuan3D2Pipeline
