# SPDX-License-Identifier: Apache-2.0

import os

import torch.nn as nn

from sglang.multimodal_gen.configs.pipeline_configs.sana_wm import SanaWMPipelineConfig
from sglang.multimodal_gen.configs.sample.sana_wm import SanaWMSamplingParams
from sglang.multimodal_gen.runtime.loader.utils import get_memory_usage_of_component
from sglang.multimodal_gen.runtime.managers.memory_managers.component_resident_strategies import (
    VanillaD2HStrategy,
    is_fsdp_managed_module,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    is_layerwise_offloaded_module,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload_components import (
    LAYERWISE_OFFLOAD_ALL_COMPONENTS,
    LAYERWISE_OFFLOAD_DIT_GROUP,
    normalize_layerwise_offload_components,
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
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm_refiner import (
    OfficialDiffusersLTX2RefinerModule,
    OfficialGemma3TextEncoderModule,
    SanaWMLTX2RefinerStage,
    SanaWMRefinerDecodingStage,
    default_sana_wm_refiner_dtype,
    sana_wm_skip_refiner_enabled,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

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
        if tp_size < 1:
            raise ValueError(f"Invalid SANA-WM tensor parallelism size: {tp_size}.")
        if tp_size > 1:
            pipeline_config = getattr(server_args, "pipeline_config", None)
            dit_config = getattr(pipeline_config, "dit_config", None)
            arch = getattr(dit_config, "arch_config", None)
            num_heads = getattr(arch, "num_attention_heads", None)
            if num_heads is not None and num_heads % tp_size != 0:
                raise ValueError(
                    "SANA-WM tensor parallelism requires num_attention_heads "
                    f"({num_heads}) to be divisible by tp_size ({tp_size})."
                )
            logger.info(
                "SANA-WM tensor parallelism enabled with tp_size=%d. "
                "Attention/GDN heads are sharded across TP ranks; sequence "
                "parallelism remains disabled.",
                tp_size,
            )

        sp_degree = getattr(server_args, "sp_degree", 1) or 1
        if sp_degree != 1:
            raise ValueError(
                "SANA-WM does not support true sequence parallelism yet. "
                "Its stage-1 DiT has layout-dependent non-attention operators "
                "(GDN scan, GLUMBConvTemp, camera UCPE, and Plucker conditioning) "
                "that need explicit state/reduction/halo exchange before tokens "
                f"can be sharded. Use TP/FSDP/CFG instead of --sp-degree {sp_degree}."
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

    # Stage-2 refiner sub-modules and their on-disk layout. These are loaded
    # through the official Diffusers/Transformers classes because NVlabs'
    # reference refiner is a narrow video-only wrapper around those modules.
    _REFINER_SUB_MODULES: tuple[tuple[str, str], ...] = (
        ("transformer_2", "refiner/transformer"),
        ("connectors", "refiner/connectors"),
        ("text_encoder_2", "refiner/text_encoder"),
        # The refiner Gemma-3 ships its tokenizer files alongside the encoder.
        ("tokenizer_2", "refiner/text_encoder"),
    )
    _SEQUENTIAL_RESIDENCY_COMPONENTS: tuple[str, ...] = (
        "text_encoder",
        "transformer",
        "text_encoder_2",
        "connectors",
        "transformer_2",
    )
    _TWO_STAGE_RESIDENCY_ENV = "SGLANG_SANA_WM_TWO_STAGE_RESIDENCY"
    _TWO_STAGE_RESIDENCY_MODES = frozenset(("auto", "resident", "sequential"))

    def initialize_pipeline(self, server_args: ServerArgs) -> None:
        super().initialize_pipeline(server_args)
        if sana_wm_skip_refiner_enabled():
            logger.info(
                "SANA-WM refiner component loading skipped by "
                "SGLANG_SANA_WM_SKIP_REFINER."
            )
            return
        self._load_refiner_modules(server_args)
        self._configure_two_stage_component_residency(server_args)

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

    @classmethod
    def _two_stage_residency_mode(cls) -> str:
        mode = os.getenv(cls._TWO_STAGE_RESIDENCY_ENV, "auto").strip().lower()
        if not mode:
            return "auto"
        if mode in cls._TWO_STAGE_RESIDENCY_MODES:
            return mode
        logger.warning(
            "Ignoring invalid %s=%r. Expected one of %s; using 'auto'.",
            cls._TWO_STAGE_RESIDENCY_ENV,
            mode,
            sorted(cls._TWO_STAGE_RESIDENCY_MODES),
        )
        return "auto"

    @staticmethod
    def _has_conflicting_two_stage_memory_policy(server_args: ServerArgs) -> bool:
        if getattr(server_args, "use_fsdp_inference", False):
            return True
        if getattr(server_args, "dit_layerwise_offload", False):
            return True

        layerwise_components = normalize_layerwise_offload_components(
            getattr(server_args, "layerwise_offload_components", None)
        )
        if layerwise_components is None:
            return False

        return bool(
            {
                LAYERWISE_OFFLOAD_ALL_COMPONENTS,
                LAYERWISE_OFFLOAD_DIT_GROUP,
                "transformer",
                "transformer_2",
            }
            & set(layerwise_components)
        )

    def _should_use_two_stage_sequential_residency(
        self, server_args: ServerArgs
    ) -> tuple[bool, str]:
        mode = self._two_stage_residency_mode()
        if mode == "resident":
            return False, f"{self._TWO_STAGE_RESIDENCY_ENV}=resident"
        if mode == "sequential":
            return True, f"{self._TWO_STAGE_RESIDENCY_ENV}=sequential"

        if self._has_conflicting_two_stage_memory_policy(server_args):
            return False, "FSDP or DiT layerwise offload is active"

        performance_mode = str(
            getattr(server_args, "performance_mode", "auto") or "auto"
        ).lower()
        tp_size = getattr(server_args, "tp_size", 1) or 1
        cfg_parallel = bool(getattr(server_args, "enable_cfg_parallel", False))
        is_manual_memory_path = performance_mode in ("manual", "memory")
        is_tp_without_cfg_path = tp_size > 1 and not cfg_parallel

        if is_manual_memory_path or is_tp_without_cfg_path:
            return (
                True,
                "auto detected no FSDP/layerwise policy with "
                f"performance_mode={performance_mode}, tp_size={tp_size}, "
                f"enable_cfg_parallel={cfg_parallel}",
            )

        return False, "auto did not detect a high-risk residency combination"

    def _configure_two_stage_component_residency(
        self, server_args: ServerArgs
    ) -> None:
        """Avoid high-risk stage-1/refiner GPU residency overlap.

        SANA-WM's stage-2 refiner includes a large Gemma-3 text encoder and an
        LTX-2 transformer.  In manual TP runs without FSDP or layerwise
        offload, the default resident strategy can leave stage-1 text/DiT
        weights on GPU while the refiner starts loading, which exhausts H100
        memory even for small smoke inputs.  Auto mode applies this conservative
        residency only for those high-risk combinations; users can force either
        path with SGLANG_SANA_WM_TWO_STAGE_RESIDENCY=resident|sequential.
        """
        should_configure, reason = self._should_use_two_stage_sequential_residency(
            server_args
        )
        if not should_configure:
            logger.info(
                "SANA-WM two-stage sequential residency disabled: %s. "
                "Set %s=sequential to force the memory-safe path.",
                reason,
                self._TWO_STAGE_RESIDENCY_ENV,
            )
            return

        configured: list[str] = []
        for component_name in self._SEQUENTIAL_RESIDENCY_COMPONENTS:
            if component_name in self.component_residency_strategies:
                continue
            module = self.modules.get(component_name)
            if not isinstance(module, nn.Module):
                continue
            if is_fsdp_managed_module(module) or is_layerwise_offloaded_module(module):
                continue
            self.component_residency_strategies[component_name] = VanillaD2HStrategy()
            configured.append(component_name)

        if configured:
            logger.info(
                "SANA-WM two-stage sequential residency enabled (%s) for "
                "components: %s. Set %s=resident to force the GPU-resident path.",
                reason,
                configured,
                self._TWO_STAGE_RESIDENCY_ENV,
            )

    def _maybe_add_refiner_stage(self, server_args: ServerArgs) -> None:
        if sana_wm_skip_refiner_enabled():
            return
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
        if sana_wm_skip_refiner_enabled():
            return super()._add_decoding_stage()
        self.add_stage(
            SanaWMRefinerDecodingStage(
                vae=self.get_module("vae"),
                pipeline=self,
                component_name="vae",
            ),
            "decoding_stage",
        )


EntryClass = [SanaWMPipeline, SanaWMTwoStagePipeline]
