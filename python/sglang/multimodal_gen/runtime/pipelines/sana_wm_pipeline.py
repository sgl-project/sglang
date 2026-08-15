# SPDX-License-Identifier: Apache-2.0

import os

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
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm import (
    SanaWMBeforeDenoisingStage,
    SanaWMDecodingStage,
    SanaWMDenoisingStage,
    SanaWMTextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.refiner import (
    SanaWMLTX2RefinerStage,
    SanaWMRefinerDecodingStage,
    default_sana_wm_refiner_dtype,
    sana_wm_skip_refiner_enabled,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.streaming import (
    SanaWMStreamingDecodingStage,
    SanaWMStreamingDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.streaming_refiner import (
    SanaWMStreamingRefinerStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

# Stage-2 refiner sub-modules live under `<model_path>/refiner/...`, not at the
# model root. They're loaded manually in `initialize_pipeline` rather than via
# `_required_config_modules`, because the framework verifier resolves every
# required module key as a literal top-level subdir of the materialized model.


logger = init_logger(__name__)


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

    @staticmethod
    def _validate_parallelism_args(server_args: ServerArgs) -> None:
        tp_size = getattr(server_args, "tp_size", 1) or 1
        if tp_size != 1:
            raise ValueError(
                "SANA-WM does not support tensor parallelism yet. "
                "Use --num-gpus with FSDP/CFG parallelism instead of "
                f"--tp-size {tp_size}."
            )

        sp_degree = getattr(server_args, "sp_degree", 1) or 1
        if sp_degree != 1:
            raise ValueError(
                "SANA-WM does not support temporal sequence parallelism yet. "
                "Stage-1 GDN/GLUMBConvTemp span frames and require halo/state "
                "exchange before latents can be sharded. Use --num-gpus with "
                "FSDP/CFG parallelism instead of "
                f"--sp-degree {sp_degree}."
            )

    def create_pipeline_stages(self, server_args: ServerArgs):
        self._validate_parallelism_args(server_args)
        self.add_stage(InputValidationStage())

        self.add_stage(
            SanaWMTextEncodingStage(
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

        if getattr(server_args.pipeline_config, "streaming", False):
            DenoiseStage = SanaWMStreamingDenoisingStage
        else:
            DenoiseStage = SanaWMDenoisingStage
        self.add_stage(
            DenoiseStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        # Subclasses (e.g. SanaWMTwoStagePipeline) insert latent-domain stages
        # between denoising and VAE decoding.
        self._maybe_add_refiner_stage(server_args)

        self._add_decoding_stage(server_args)

    def _add_decoding_stage(self, server_args: ServerArgs = None) -> None:
        if server_args is not None and getattr(
            server_args.pipeline_config, "streaming", False
        ):
            DecodeStage = SanaWMStreamingDecodingStage
        else:
            DecodeStage = SanaWMDecodingStage
        self.add_stage(
            DecodeStage(
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

    Stage-1 produces a coarse 720p latent; the LTX-2 refiner runs 3 Euler steps
    on it before VAE decode, matching the NVlabs ``inference_sana_wm.py`` default.
    """

    pipeline_name = "SanaWMTwoStagePipeline"

    _REFINER_SUB_MODULES: tuple[tuple[str, str, str], ...] = (
        ("transformer_2", "refiner/transformer", "diffusers"),
        ("connectors", "refiner/connectors", "diffusers"),
        ("text_encoder_2", "refiner/text_encoder", "transformers"),
        # The refiner Gemma-3 ships its tokenizer files alongside the encoder.
        ("tokenizer_2", "refiner/text_encoder", "transformers"),
    )

    def initialize_pipeline(self, server_args: ServerArgs) -> None:
        super().initialize_pipeline(server_args)
        if sana_wm_skip_refiner_enabled():
            logger.info(
                "SANA-WM refiner component loading skipped by "
                "SGLANG_SANA_WM_SKIP_REFINER."
            )
            return
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

    def _load_refiner_modules(self, server_args: ServerArgs) -> None:
        for module_name, subpath, library in self._REFINER_SUB_MODULES:
            component_path = self._resolve_refiner_component_path(
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
        if sana_wm_skip_refiner_enabled():
            return
        pc = server_args.pipeline_config
        common = dict(
            transformer=self.get_module("transformer_2"),
            connectors=self.get_module("connectors"),
            text_encoder=self.get_module("text_encoder_2"),
            tokenizer=self.get_module("tokenizer_2"),
            dtype=default_sana_wm_refiner_dtype(server_args),
        )
        if getattr(pc, "streaming", False) and getattr(pc, "refiner_chunked", True):
            stage = SanaWMStreamingRefinerStage(
                **common,
                block_size=int(getattr(pc, "refiner_block_size", 3)),
                kv_max_frames=int(getattr(pc, "refiner_kv_max_frames", 11)),
                sink_size=int(getattr(pc, "sink_size", 1)),
                seed=int(getattr(pc, "refiner_seed", 42)),
            )
        else:
            stage = SanaWMLTX2RefinerStage(**common)
        self.add_stage(stage, "sana_wm_refiner")

    def _add_decoding_stage(self, server_args: ServerArgs = None) -> None:
        # Streaming and skip-refiner both route to the base decode
        # (SanaWMStreamingDecodingStage / dense decode); otherwise dense refiner-decode.
        streaming = server_args is not None and getattr(
            server_args.pipeline_config, "streaming", False
        )
        if streaming or sana_wm_skip_refiner_enabled():
            return super()._add_decoding_stage(server_args)
        self.add_stage(
            SanaWMRefinerDecodingStage(
                vae=self.get_module("vae"),
                pipeline=self,
                component_name="vae",
            ),
            "decoding_stage",
        )


EntryClass = [SanaWMPipeline, SanaWMTwoStagePipeline]
