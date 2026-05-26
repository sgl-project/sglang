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
import os
from types import SimpleNamespace

# Stage-2 refiner sub-modules live under `<model_path>/refiner/...` rather
# than at the model root. We load them manually in `initialize_pipeline`
# (mirroring how `LTX2TwoStagePipeline._initialize_premerged_stage2_transformer`
# loads its stage-2 DiT) instead of registering them in
# `_required_config_modules`, because the framework verifier resolves every
# required module key as a literal top-level subdir of the materialized model.

from sglang.multimodal_gen.configs.models.dits.sana_wm_refiner import (
    SanaWMRefinerConfig,
)
from sglang.multimodal_gen.configs.models.encoders.gemma_3 import Gemma3Config
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
    InputValidationStage,
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm import (
    SanaWMBeforeDenoisingStage,
    SanaWMDecodingStage,
    SanaWMDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm_refiner import (
    SanaWMLTX2RefinerStage,
    SanaWMRefinerDecodingStage,
    default_sana_wm_refiner_dtype,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class _SanaWMRefinerGemma3Config(Gemma3Config):
    """Gemma-3 config adapter for the SANA-WM refiner text encoder.

    TextEncoderLoader supports multiple encoders by suffix, but only the first
    encoder receives the Transformers AutoConfig object. The refiner Gemma-3
    config is available as JSON, so convert nested dicts to attribute objects
    here and keep the workaround local to SANA-WM.
    """

    def update_model_arch(self, source_model_dict):
        super().update_model_arch(source_model_dict)
        for key in ("text_config", "vision_config"):
            value = getattr(self.arch_config, key, None)
            if isinstance(value, dict):
                setattr(self.arch_config, key, SimpleNamespace(**value))


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
            SanaWMDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        # Subclasses (e.g. SanaWMTwoStagePipeline) insert latent-domain stages
        # between denoising and VAE decoding.
        self._maybe_add_refiner_stage(server_args)

        self._add_decoding_stage()

    def _add_decoding_stage(self) -> None:
        self.add_stage(
            SanaWMDecodingStage(
                vae=self.get_module("vae"),
                pipeline=self,
                component_name="vae",
            ),
            "decoding_stage",
        )

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

    def _resolve_refiner_paths(self, server_args: ServerArgs) -> tuple[str, str]:
        component_paths = getattr(server_args, "component_paths", {}) or {}
        refiner_root = component_paths.get(
            "refiner", os.path.join(self.model_path, "refiner")
        )
        refiner_gemma_root = component_paths.get(
            "refiner_text_encoder",
            component_paths.get(
                "text_encoder_2", os.path.join(refiner_root, "text_encoder")
            ),
        )
        return refiner_root, refiner_gemma_root

    def _resolve_refiner_component_path(
        self, server_args: ServerArgs, module_name: str, subpath: str
    ) -> str:
        component_paths = getattr(server_args, "component_paths", {}) or {}
        if module_name in component_paths:
            return self._resolve_component_path(server_args, module_name, subpath)

        if (
            "refiner" not in component_paths
            and "refiner_text_encoder" not in component_paths
        ):
            return self._resolve_component_path(server_args, module_name, subpath)

        refiner_root, refiner_gemma_root = self._resolve_refiner_paths(server_args)
        if module_name in ("text_encoder_2", "tokenizer_2"):
            return refiner_gemma_root

        rel_subpath = subpath.removeprefix("refiner/")
        return os.path.join(refiner_root, rel_subpath)

    @staticmethod
    def _ensure_refiner_text_encoder_config(server_args: ServerArgs):
        """Temporarily expose the refiner Gemma-3 config to TextEncoderLoader.

        The stage-1 SANA-WM pipeline has one Gemma-2 text encoder. The
        stage-2 LTX-2 refiner carries its own Gemma-3 encoder under
        refiner/text_encoder, loaded manually as text_encoder_2. The generic
        TextEncoderLoader indexes configs by component suffix, so provide the
        second config only for this manual load instead of making the stage-1
        TextEncodingStage believe it owns two encoders.
        """
        pipeline_config = server_args.pipeline_config
        saved = (
            pipeline_config.text_encoder_configs,
            pipeline_config.text_encoder_precisions,
        )

        configs = list(pipeline_config.text_encoder_configs)
        while len(configs) <= 1:
            configs.append(_SanaWMRefinerGemma3Config())
        pipeline_config.text_encoder_configs = tuple(configs)

        precisions = list(pipeline_config.text_encoder_precisions)
        while len(precisions) <= 1:
            precisions.append("bf16")
        pipeline_config.text_encoder_precisions = tuple(precisions)
        return saved

    def _load_refiner_modules(self, server_args: ServerArgs) -> None:
        for module_name, subpath, library in self._REFINER_SUB_MODULES:
            component_path = self._resolve_refiner_component_path(
                server_args, module_name, subpath
            )
            logger.info(
                "SANA-WM loading refiner component %s from %s",
                module_name,
                component_path,
            )
            # `transformer_2` is a different model class
            # (`SanaWMLTX2VideoRefiner`) than stage-1's
            # `SanaWMTransformer3DModel` and needs its own DiT config
            # (`SanaWMRefinerConfig`). `TransformerLoader` keys its
            # `dit_config` lookup on the *normalized* component name, so both
            # `transformer` and `transformer_2` resolve to
            # `pipeline_config.dit_config`. Temporarily swap in the refiner
            # config for this one load.
            #
            # TODO: replace with a framework-level "per-component dit_config"
            # mechanism (e.g. `TransformerLoader` consulting
            # `pipeline_config.dit_config_2` for `transformer_2`) once that
            # exists. See the SANA-WM PR's follow-up notes.
            saved_dit_config = None
            saved_text_encoder_config = None
            if module_name == "transformer_2":
                saved_dit_config = server_args.pipeline_config.dit_config
                server_args.pipeline_config.dit_config = SanaWMRefinerConfig()
            elif module_name == "text_encoder_2":
                saved_text_encoder_config = (
                    self._ensure_refiner_text_encoder_config(server_args)
                )
            try:
                module, memory_usage = PipelineComponentLoader.load_component(
                    component_name=module_name,
                    component_model_path=component_path,
                    transformers_or_diffusers=library,
                    server_args=server_args,
                )
            finally:
                if saved_dit_config is not None:
                    server_args.pipeline_config.dit_config = saved_dit_config
                if saved_text_encoder_config is not None:
                    (
                        server_args.pipeline_config.text_encoder_configs,
                        server_args.pipeline_config.text_encoder_precisions,
                    ) = saved_text_encoder_config

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
