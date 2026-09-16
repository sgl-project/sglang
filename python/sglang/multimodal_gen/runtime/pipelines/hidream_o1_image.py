# SPDX-License-Identifier: Apache-2.0
"""HiDream-O1-Image text-to-image pipeline.

The checkpoint is a plain Qwen3-VL directory with no ``model_index.json``, so
the pipeline loads its single component itself instead of going through the
diffusers component loaders.
"""

import json
import os

import torch

from sglang.multimodal_gen.configs.pipeline_configs.hidream_o1_image import (
    HiDreamO1ImagePipelineConfig,
)
from sglang.multimodal_gen.configs.sample.hidream_o1_image import (
    HiDreamO1ImageSamplingParams,
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
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
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
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision import resolve_precision

logger = init_logger(__name__)

# Flow shift the reference implementation trains and samples with.
HIDREAM_O1_FLOW_SHIFT = 3.0
# Modules the stages read back through get_module; a caller-supplied dict has to
# cover all of them before load_modules can skip building them.
_HIDREAM_O1_MODULES = frozenset({"transformer", "processor", "tokenizer", "scheduler"})


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
        model_dir = maybe_download_model(self.model_path)
        with open(os.path.join(model_dir, "config.json")) as config_file:
            hf_config = json.load(config_file)

        # Read even when the modules are handed in: create_pipeline_stages and
        # the stages both need these, and neither is recoverable from a module.
        self._vision_start_token_id = hf_config["vision_start_token_id"]
        dit_config = server_args.pipeline_config.dit_config
        dit_config.update_model_arch(hf_config["text_config"])

        # The pixel head is not numerically stable in fp16, so this is a hard
        # rejection rather than a cast.
        param_dtype = resolve_precision(
            server_args, "dit", precision_attr="dit_precision"
        )
        if param_dtype != torch.bfloat16:
            raise ValueError(
                f"HiDream-O1-Image is only released in bfloat16; got {param_dtype}. "
                "Drop --dit-precision / --component-precisions overrides."
            )

        if loaded_modules is not None and _HIDREAM_O1_MODULES <= set(loaded_modules):
            self._validate_tms_token(loaded_modules["tokenizer"], dit_config)
            return loaded_modules

        safetensors_list = _list_safetensors_files(
            model_dir,
            index_file="model.safetensors.index.json",
            key_filter=is_hidream_o1_image_weight,
        )
        if not safetensors_list:
            raise ValueError(f"No safetensors files found in {model_dir}")

        local_torch_device = get_local_torch_device()
        transformer = maybe_load_fsdp_model(
            model_cls=HiDreamO1ImageTransformer,
            init_params={"config": dit_config, "hf_config": hf_config},
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
        server_args.model_paths["transformer"] = model_dir

        from transformers import AutoProcessor, PreTrainedTokenizerBase

        processor = AutoProcessor.from_pretrained(model_dir)
        tokenizer = (
            processor
            if isinstance(processor, PreTrainedTokenizerBase)
            else processor.tokenizer
        )
        self._validate_tms_token(tokenizer, dit_config)

        return {
            "transformer": transformer,
            "processor": processor,
            "tokenizer": tokenizer,
            "scheduler": FlowUniPCMultistepScheduler(
                num_train_timesteps=1000,
                shift=HIDREAM_O1_FLOW_SHIFT,
                prediction_type="flow_prediction",
                use_dynamic_shifting=False,
            ),
        }

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        # 1. Resolution defaults plus the per-output seeds the noise is drawn from.
        self.add_stage(InputValidationStage())

        # 2. Prompt tokens, mrope positions, hybrid attention mask, patchified
        #    noise and the timestep schedule.
        self.add_stage(
            HiDreamO1ImageBeforeDenoisingStage(
                tokenizer=self.get_module("tokenizer"),
                processor=self.get_module("processor"),
                scheduler=self.get_module("scheduler"),
                vision_start_token_id=self._vision_start_token_id,
                tms_token_id=server_args.pipeline_config.dit_config.arch_config.tms_token_id,
            )
        )

        # 3. Denoising loop, converting the predicted clean image to a velocity.
        self.add_stage(
            HiDreamO1ImageDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                pipeline=self,
            )
        )

        # 4. Unpatchify; there is no VAE to decode through.
        self.add_stage(HiDreamO1ImageDecodingStage())


EntryClass = HiDreamO1ImagePipeline
