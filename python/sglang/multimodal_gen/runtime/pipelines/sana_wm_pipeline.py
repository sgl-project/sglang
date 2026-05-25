# SPDX-License-Identifier: Apache-2.0
#
# SANA-WM TI2V pipelines.
#
#   SanaWMPipeline (single-stage):
#     InputValidation -> TextEncoding (Gemma-2) -> SanaWMBeforeDenoising
#       -> Denoising -> standard decoding (LTX-2 VAE)
#
#   SanaWMTwoStagePipeline (matches NVlabs official inference):
#     ... -> Denoising -> SanaWMLTX2RefinerStage
#       -> refiner decoding (LTX-2 VAE + drop clean sink anchor frame)
#
# The two-stage variant loads four extra refiner sub-modules through the
# framework's `PipelineComponentLoader` by declaring them in
# `_required_config_modules` and pointing each at the appropriate
# `refiner/<subdir>` via `_extra_config_module_map`. The component loader
# normalizes the trailing `_2` so `transformer_2` -> TransformerLoader,
# `text_encoder_2` -> TextEncoderLoader, `tokenizer_2` -> TokenizerLoader.

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

    Stage-1 generates a coarse 720p latent; the LTX-2 video refiner then runs
    3 Euler steps on that latent before VAE decode, matching the NVlabs
    ``inference_sana_wm.py`` default.
    """

    pipeline_name = "SanaWMTwoStagePipeline"

    # Stage-2 refiner sub-modules live under `refiner/{transformer,connectors,text_encoder}`
    # in the materialized checkpoint. We register them with the standard module
    # keys (`transformer_2`, `connectors`, `text_encoder_2`, `tokenizer_2`) so
    # the existing component loaders can drive them. The `_2` suffix is
    # stripped by `_normalize_component_type` for loader dispatch.
    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
        "transformer_2",
        "connectors",
        "text_encoder_2",
        "tokenizer_2",
    ]
    _extra_config_module_map = {
        "transformer_2": "refiner/transformer",
        "connectors": "refiner/connectors",
        "text_encoder_2": "refiner/text_encoder",
        # The refiner Gemma-3 ships its tokenizer files alongside the
        # text_encoder weights under refiner/text_encoder/.
        "tokenizer_2": "refiner/text_encoder",
    }

    def _maybe_add_refiner_stage(self, server_args: ServerArgs) -> None:
        self.add_stage(
            SanaWMLTX2RefinerStage(
                transformer=self.get_module("transformer_2"),
                connectors=self.get_module("connectors"),
                text_encoder=self.get_module("text_encoder_2"),
                tokenizer=self.get_module("tokenizer_2"),
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
