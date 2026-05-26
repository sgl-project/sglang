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
# Stage-2 refiner sub-modules live under `<model_path>/refiner/...` rather
# than at the model root. We load them manually in `initialize_pipeline`
# (mirroring how `LTX2TwoStagePipeline._initialize_premerged_stage2_transformer`
# loads its stage-2 DiT) instead of registering them in
# `_required_config_modules`, because the framework verifier resolves every
# required module key as a literal top-level subdir of the materialized model.

from sglang.multimodal_gen.configs.pipeline_configs.sana_wm import SanaWMPipelineConfig
from sglang.multimodal_gen.configs.sample.sana_wm import SanaWMSamplingParams
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    PipelineComponentLoader,
)
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

    # Stage-2 refiner sub-modules and their on-disk layout. Tuples are
    # (module_name, subpath_under_model_root, transformers_or_diffusers).
    # The `_2` suffix is stripped by `_normalize_component_type` so the
    # component loader dispatches `transformer_2` -> TransformerLoader,
    # `text_encoder_2` -> TextEncoderLoader, `tokenizer_2` -> TokenizerLoader.
    # `connectors` is special-cased to "diffusers" by
    # `ComponentLoader.resolve_transformers_or_diffusers`.
    _REFINER_SUB_MODULES: tuple[tuple[str, str, str], ...] = (
        ("transformer_2", "refiner/transformer", "diffusers"),
        ("connectors", "refiner/connectors", "diffusers"),
        ("text_encoder_2", "refiner/text_encoder", "transformers"),
        # The refiner Gemma-3 ships its tokenizer files alongside the encoder.
        ("tokenizer_2", "refiner/text_encoder", "transformers"),
    )

    def initialize_pipeline(self, server_args: ServerArgs) -> None:
        super().initialize_pipeline(server_args)
        self._load_refiner_modules(server_args)

    def _load_refiner_modules(self, server_args: ServerArgs) -> None:
        for module_name, subpath, library in self._REFINER_SUB_MODULES:
            component_path = self._resolve_component_path(
                server_args, module_name, subpath
            )
            module, memory_usage = PipelineComponentLoader.load_component(
                component_name=module_name,
                component_model_path=component_path,
                transformers_or_diffusers=library,
                server_args=server_args,
            )
            self.modules[module_name] = module
            self.memory_usages[module_name] = memory_usage

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
