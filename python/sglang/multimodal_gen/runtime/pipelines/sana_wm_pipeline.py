# SPDX-License-Identifier: Apache-2.0
#
# SANA-WM TI2V pipelines.
#
# Two variants share stages 1–4 and differ only in whether an LTX-2 latent
# refiner runs before VAE decode:
#
#   SanaWMPipeline (single-stage):
#     InputValidation → TextEncoding (Gemma-2) → SanaWMBeforeDenoising
#       → Denoising → standard decoding (LTX-2 VAE)
#
#   SanaWMTwoStagePipeline (matches NVlabs official inference):
#     ... → Denoising → SanaWMLTX2RefinerStage
#       → refiner decoding (LTX-2 VAE + drop clean sink anchor frame)
#
# The two-stage variant requires the upstream refiner subtree
# (refiner/{transformer,connectors,text_encoder}). When using the
# `Efficient-Large-Model/SANA-WM_bidirectional` overlay this is materialized
# at `<model_path>/refiner/`; component_paths overrides are also honored.

import os

from sglang.multimodal_gen.configs.pipeline_configs.sana_wm import SanaWMPipelineConfig
from sglang.multimodal_gen.configs.sample.sana_wm import SanaWMSamplingParams
from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    DenoisingStage,
    InputValidationStage,
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm import (
    SanaWMBeforeDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm_refiner import (
    SanaWMLTX2RefinerStage,
    SanaWMRefinerDecodingStage,
    default_sana_wm_refiner_dtype,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class SanaWMPipeline(LoRAPipeline, ComposedPipelineBase):
    """SANA-WM TI2V pipeline (single-stage)."""

    # Must match `_class_name` in model_index.json of the materialized checkpoint.
    pipeline_name = "SanaWMPipeline"
    pipeline_config_cls = SanaWMPipelineConfig
    sampling_params_cls = SanaWMSamplingParams

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(InputValidationStage())

        self.add_stage(
            TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
            "prompt_encoding_stage",
        )

        self.add_stage(
            SanaWMBeforeDenoisingStage(
                vae=self.get_module("vae"),
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                pipeline_config=server_args.pipeline_config,
            ),
            "sana_wm_before_denoising",
        )

        self.add_stage(
            DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        # Subclasses (e.g. SanaWMTwoStagePipeline) insert latent-domain stages
        # between denoising and VAE decoding.
        self._maybe_add_refiner_stage(server_args)

        self._add_decoding_stage()

    def _add_decoding_stage(self) -> None:
        self.add_standard_decoding_stage()

    def _maybe_add_refiner_stage(self, server_args: ServerArgs) -> None:
        """Hook for subclasses; single-stage pipeline is a no-op."""
        return None


class SanaWMTwoStagePipeline(SanaWMPipeline):
    """SANA-WM two-stage pipeline: SANA-WM DiT + LTX-2 latent refiner.

    Stage-1 (SanaWMPipeline) generates a coarse 720p latent; the LTX-2 video
    refiner then runs a few Euler steps on that latent before VAE decode,
    matching the NVlabs ``inference_sana_wm.py`` default. The refiner loads its
    own LTX-2 transformer/connectors/Gemma3 text encoder lazily on first use.
    """

    pipeline_name = "SanaWMTwoStagePipeline"

    def _resolve_refiner_paths(self, server_args: ServerArgs) -> tuple[str, str]:
        component_paths = getattr(server_args, "component_paths", {}) or {}
        refiner_root = component_paths.get(
            "refiner", os.path.join(self.model_path, "refiner")
        )
        refiner_gemma_root = component_paths.get(
            "refiner_text_encoder",
            os.path.join(refiner_root, "text_encoder"),
        )
        return refiner_root, refiner_gemma_root

    def _maybe_add_refiner_stage(self, server_args: ServerArgs) -> None:
        refiner_root, refiner_gemma_root = self._resolve_refiner_paths(server_args)
        self.add_stage(
            SanaWMLTX2RefinerStage(
                refiner_root=refiner_root,
                refiner_gemma_root=refiner_gemma_root,
                dtype=default_sana_wm_refiner_dtype(server_args),
            ),
            "sana_wm_refiner",
        )

    def _add_decoding_stage(self) -> None:
        self.add_stage(
            SanaWMRefinerDecodingStage(
                vae=self.get_module("vae"),
                pipeline=self,
                component_name="vae",
            ),
            "decoding_stage",
        )


EntryClass = [SanaWMPipeline, SanaWMTwoStagePipeline]
