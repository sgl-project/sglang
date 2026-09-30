# SPDX-License-Identifier: Apache-2.0

import os

from sglang.multimodal_gen.configs.pipeline_configs.sana_wm import SanaWMPipelineConfig
from sglang.multimodal_gen.configs.sample.sana_wm import SanaWMSamplingParams
from sglang.multimodal_gen.runtime.loader.utils import get_memory_usage_of_component
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
    OfficialDiffusersLTX2RefinerModule,
    OfficialGemma3TextEncoderModule,
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

    # Stage-2 refiner sub-modules and their on-disk layout. Loaded through the
    # official Diffusers/Transformers classes because NVlabs' reference refiner
    # is a narrow video-only wrapper around those modules.
    _REFINER_SUB_MODULES: tuple[tuple[str, str], ...] = (
        ("transformer_2", "refiner/transformer"),
        ("connectors", "refiner/connectors"),
        ("text_encoder_2", "refiner/text_encoder"),
        # The refiner Gemma-3 ships its tokenizer files alongside the encoder.
        ("tokenizer_2", "refiner/text_encoder"),
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
        for module_name, subpath in self._REFINER_SUB_MODULES:
            component_path = self._resolve_refiner_component_path(
                server_args, module_name, subpath
            )
            logger.info(
                "SANA-WM loading refiner component %s from %s",
                module_name,
                component_path,
            )
            module, memory_usage = self._load_official_refiner_component(
                module_name,
                component_path,
                server_args,
            )
            self.modules[module_name] = module
            self.memory_usages[module_name] = memory_usage

    @staticmethod
    def _load_official_refiner_component(
        module_name: str,
        component_path: str,
        server_args: ServerArgs,
    ):
        """Load SANA-WM refiner modules through the same libraries as NVlabs.

        The upstream wrapper (``diffusion/refiner/diffusers_ltx2_refiner.py``)
        keeps the LTX-2 transformer/connectors as Diffusers modules and only
        customizes the video-only forward surface; use that path for the
        quality-critical stage-2 refiner instead of the experimental native port.
        """

        dtype = default_sana_wm_refiner_dtype(server_args)
        if module_name == "transformer_2":
            from diffusers.models.transformers.transformer_ltx2 import (
                LTX2VideoTransformer3DModel,
            )

            module = LTX2VideoTransformer3DModel.from_pretrained(
                component_path,
                torch_dtype=dtype,
            ).eval()
            module = OfficialDiffusersLTX2RefinerModule(module)
        elif module_name == "connectors":
            from diffusers.pipelines.ltx2 import LTX2TextConnectors

            module = LTX2TextConnectors.from_pretrained(
                component_path,
                torch_dtype=dtype,
            ).eval()
        elif module_name == "text_encoder_2":
            from transformers import Gemma3ForConditionalGeneration

            module = Gemma3ForConditionalGeneration.from_pretrained(
                component_path,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            ).eval()
            module = OfficialGemma3TextEncoderModule(module)
        elif module_name == "tokenizer_2":
            from transformers import AutoTokenizer

            module = AutoTokenizer.from_pretrained(component_path)
        else:
            raise ValueError(f"Unsupported SANA-WM refiner component: {module_name}")

        memory_usage = get_memory_usage_of_component(module)
        logger.info(
            "Loaded %s: %s (official native version). model size: %s GB",
            module_name,
            module.__class__.__name__,
            memory_usage if memory_usage is not None else "NA",
        )
        return module, memory_usage or 0.0

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
