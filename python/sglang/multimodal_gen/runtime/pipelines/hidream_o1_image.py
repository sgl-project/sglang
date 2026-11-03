# SPDX-License-Identifier: Apache-2.0
"""HiDream-O1-Image pipeline — one class, one loader, one stage set.

All three checkpoints run through the same three-stage set (before-denoising
with prompt enhancement, TI2I conditioning and initial noise; the clean-image
denoising loop with CFG, co-batching, deterministic reroll and cache-dit; a
pure-unpatchify decoding stage); the variant resolved from the served model
path only selects the recipe — fixed 28-step flash / flow-match editing for
the Dev checkpoints in fp32 weights, 50-step FlowUniPC with CFG 5.0 for the
full checkpoint in bf16 — via the variant-aware configs.
"""

import json
import os

import torch
from sglang.multimodal_gen.configs.models.dits.hidream_o1_image import (
    HiDreamO1ImageDitConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.hidream_o1_image import (
    HiDreamO1ImagePipelineConfig,
)
from sglang.multimodal_gen.configs.sample.hidream_o1_image import (
    _VARIANT_DEFAULTS,
    HiDreamO1ImageSamplingParams,
    _resolve_variant,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.fsdp_load import maybe_load_fsdp_model
from sglang.multimodal_gen.runtime.loader.utils import _list_safetensors_files
from sglang.multimodal_gen.runtime.loader.weight_load_plan import WeightLoadPlan
from sglang.multimodal_gen.runtime.models.dits.hidream_o1_image import (
    HiDreamO1ImageTransformer,
    is_hidream_o1_image_weight,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import InputValidationStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hidream_o1_image import (
    HIDREAM_O1_TMS_TOKEN,
    HiDreamO1ImageBeforeDenoisingStage,
    HiDreamO1ImageDecodingStage,
    HiDreamO1ImageDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hidream_o1_image.generation import (
    DEFAULT_TIMESTEPS,
    FlowMatchScheduler,
    build_scheduler,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from transformers import AutoProcessor

logger = init_logger(__name__)

# Flow shift the reference implementation trains and samples with; a named
# constant because the tests pin the UniPC contract against it.
HIDREAM_O1_FLOW_SHIFT = _VARIANT_DEFAULTS["full"]["shift"]


class HiDreamO1ImagePipeline(ComposedPipelineBase):
    pipeline_name = "HiDreamO1ImagePipeline"
    pipeline_config_cls = HiDreamO1ImagePipelineConfig
    sampling_params_cls = HiDreamO1ImageSamplingParams
    _required_config_modules: list[str] = []

    def validate_disagg_role(self, role: RoleType) -> None:
        if role != RoleType.MONOLITHIC:
            raise ValueError(
                "HiDreamO1ImagePipeline has a single component that both encodes "
                "the prompt and denoises, so encoder/denoiser/decoder "
                "disaggregation has nothing to split. Run with the monolithic role."
            )

    @staticmethod
    def _validate_tms_token(tokenizer, dit_config) -> None:
        tms_ids = tokenizer.encode(HIDREAM_O1_TMS_TOKEN, add_special_tokens=False)
        expected_tms_id = dit_config.arch_config.tms_token_id
        if tms_ids != [expected_tms_id]:
            raise ValueError(
                f"{HIDREAM_O1_TMS_TOKEN} encodes to {tms_ids}, but the pixel head "
                f"reads the timestep from token id {expected_tms_id}."
            )

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, torch.nn.Module]:
        model_path = maybe_download_model(
            self.model_path, revision=server_args.revision
        )
        with open(os.path.join(model_dir, "config.json")) as config_file:
            hf_config = json.load(config_file)

        if loaded_modules is not None and _HIDREAM_O1_MODULES <= set(loaded_modules):
            self._validate_tms_token(loaded_modules["tokenizer"], dit_config)
            return loaded_modules

        dit_config = server_args.pipeline_config.dit_config
        variant = _resolve_variant(model_path)
        w_dtype = getattr(torch, _VARIANT_DEFAULTS[variant]["torch_dtype"])

        # The vision tower is built for every variant: the stage serves TI2I
        # editing for all of them, and the checkpoint ships model.visual.*.
        model = self._load_transformer(
            server_args, model_path, dit_config, hf_config, w_dtype
        )

        if variant != "full":
            # Dev variants keep fp32 weights + bf16 autocast per variant policy;
            # the dtype reaches the loader as param_dtype, so this is just the
            # eval switch (the loader may leave the model on CPU for residency).
            model = model.eval()

        processor = AutoProcessor.from_pretrained(model_path)

        tokenizer = (
            processor
            if isinstance(processor, PreTrainedTokenizerBase)
            else processor.tokenizer
        )
        self._validate_tms_token(tokenizer, dit_config)

        # The variant-default scheduler is built once at pipeline creation,
        # like the official pipeline builds its scheduler before denoising:
        # full -> FlowUniPCMultistepScheduler via the diffusion defaults
        # (recipe shift 3.0), Dev -> flash. The Dev editing override is also a
        # deployment constant (the variant recipe decides it, not the
        # request), so its flow_match template is built here too; the stage
        # only selects a template and re-arms its per-request step counter.
        variant_defaults = _VARIANT_DEFAULTS[variant] # TODO: check need to change
        scheduler = build_scheduler(
            num_inference_steps=server_args.pipeline_config.num_inference_steps,
            timesteps_list=None if variant == "full" else DEFAULT_TIMESTEPS,
            shift=variant_defaults["shift"],
            device=get_local_torch_device(),
            scheduler_name=variant_defaults["scheduler_name"],
        )
        editing_scheduler = FlowMatchScheduler(shift=variant_defaults["shift"])
        editing_scheduler.set_timesteps_list(
            DEFAULT_TIMESTEPS, get_local_torch_device()
        )

        return {
            "model": model,
            "processor": processor,
            "tokenizer": tokenizer,
            "scheduler": scheduler,
            "editing_scheduler": editing_scheduler,
        }

    @staticmethod
    def _load_transformer(
        server_args: ServerArgs,
        model_path: str,
        dit_config: HiDreamO1ImageDitConfig,
        hf_config: dict,
        param_dtype: torch.dtype,
    ) -> HiDreamO1ImageTransformer:
        safetensors_list = _list_safetensors_files(
            model_path,
            index_file="model.safetensors.index.json",
            key_filter=is_hidream_o1_image_weight,
        )
        if not safetensors_list:
            raise ValueError(f"No safetensors files found in {model_path}")

        local_torch_device = get_local_torch_device()
        transformer = maybe_load_fsdp_model(
            model_cls=HiDreamO1ImageTransformer,
            init_params={
                "config": dit_config,
                "hf_config": hf_config,
                # TI2I editing is a capability of every variant.
                "with_vision_tower": True,
            },
            weight_dir_list=safetensors_list,
            device=local_torch_device,
            hsdp_replicate_dim=server_args.hsdp_replicate_dim,
            hsdp_shard_dim=server_args.hsdp_shard_dim,
            component_starts_on_cpu=server_args.should_start_component_on_cpu(
                "transformer"
            ),
            pin_cpu_memory=server_args.pin_cpu_memory,
            fsdp_inference=server_args.should_use_fsdp_for_component("transformer"),
            param_dtype=param_dtype,
            reduce_dtype=torch.float32,
            output_dtype=None,
            strict=True,
            checkpoint_key_filter=is_hidream_o1_image_weight,
            weight_load_plan=WeightLoadPlan(checkpoint_load_device=local_torch_device),
        )
        server_args.model_paths["transformer"] = model_path
        logger.info(
            "HiDream %s transformer loaded to %s",
            _resolve_variant(model_path),
            local_torch_device,
        )

        # The TI2I two-pass loop drives decoder submodules directly, bypassing
        # FSDP2's per-layer unshard hooks; under FSDP inference the model must
        # take its dense 4D-mask branch instead (see _fsdp_wrapped).
        from torch.distributed.fsdp import FSDPModule

        transformer._fsdp_wrapped = isinstance(
            transformer, FSDPModule
        ) or any(isinstance(m, FSDPModule) for m in transformer.modules())
        if transformer._fsdp_wrapped:
            logger.info(
                "FSDP inference wrapping detected: TI2I requests take the "
                "dense attention branch."
            )
        return transformer

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        arch_config = server_args.pipeline_config.dit_config.arch_config
        processor = self.get_module("processor")
        self.add_stage(InputValidationStage())
        self.add_stage(
            HiDreamO1ImageBeforeDenoisingStage(
                tokenizer=self.get_module("tokenizer"),
                processor=processor,
                scheduler=self.get_module("scheduler"),
                editing_scheduler=self.get_module("editing_scheduler"),
                vision_start_token_id=arch_config.vision_start_token_id,
                tms_token_id=arch_config.tms_token_id,
            ),
            "hidream_o1_image_before_denoising_stage",
        )
        self.add_stage(
            HiDreamO1ImageDenoisingStage(
                model=self.get_module("model"),
                scheduler=self.get_module("scheduler"),
                processor=processor,
            ),
            "hidream_o1_image_denoising_stage",
        )
        self.add_stage(HiDreamO1ImageDecodingStage(), "hidream_o1_image_decoding_stage")


EntryClass = HiDreamO1ImagePipeline
