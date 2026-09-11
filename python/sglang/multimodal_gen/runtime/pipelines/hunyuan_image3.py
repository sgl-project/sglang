import json
import os
from typing import Any

import torch
from torch.distributed import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from transformers import AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    component_attn_backend_context_manager,
)
from sglang.multimodal_gen.runtime.loader.fsdp_load import shard_model
from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
    resolve_transformer_checkpoint_files,
)
from sglang.multimodal_gen.runtime.loader.utils import set_default_torch_dtype
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    safetensors_weights_iterator,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    get_global_component_residency_manager,
)
from sglang.multimodal_gen.runtime.models.dits.hunyuan_image3 import (
    HunyuanImage3ForCausalMM,
    LightProjector,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_conv3d_hunyuan_image3 import (
    AutoencoderKLConv3D,
)
from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan_image3 import (
    HunyuanImage3AR,
    HunyuanImage3DecodingStage,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_hf_config,
    load_dict,
    maybe_download_model,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision_types import PRECISION_TO_TYPE
from sglang.multimodal_gen.runtime.utils.precision import set_mixed_precision_policy

logger = init_logger(__name__)


def _module_memory_gb(module: torch.nn.Module) -> float:
    total_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
    total_bytes += sum(b.numel() * b.element_size() for b in module.buffers())
    return total_bytes / (1024**3)


class HunyuanImage3Pipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "HunyuanImage3Pipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "processor",
        "transformer",
        "scheduler",
        "vision_model",
        "vision_aligner",
    ]

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        pipeline_config = server_args.pipeline_config

        model_path = maybe_download_model(self.model_path)
        self.model_path = model_path
        logger.info("Loading HunyuanImage-3 components from %s", model_path)

        hf_config = get_hf_config(
            model_path,
            trust_remote_code=True,
            revision=server_args.revision,
        )
        config_dict = hf_config.to_dict()

        pipeline_config.dit_config.update_model_arch(config_dict)
        vae_config_dict = dict(config_dict["vae"])
        pipeline_config.vae_config.update_model_arch(vae_config_dict)
        pipeline_config.vae_config.post_init()

        flow_shift = self._read_flow_shift(model_path)
        pipeline_config.flow_shift = flow_shift

        ar_model = self._load_ar_model(
            server_args, pipeline_config, model_path, config_dict
        )

        return {
            "transformer": ar_model,
            "text_encoder": ar_model,
            "vae": self._load_vae(
                server_args, pipeline_config, model_path, vae_config_dict
            ),
            "tokenizer": self._load_tokenizer(server_args, model_path),
            "scheduler": FlowMatchEulerDiscreteScheduler(shift=flow_shift),
            "processor": self._load_processor(server_args, model_path, hf_config),
            "vision_model": self._load_vision_model(model_path, config_dict),
            "vision_aligner": self._load_vision_aligner(model_path, config_dict),
        }

    def _load_ar_model(
        self,
        server_args: ServerArgs,
        pipeline_config: Any,
        model_path: str,
        config_dict: dict[str, Any],
    ) -> torch.nn.Module:
        # resolve_transformer_checkpoint_files handles the --transformer-
        # weights-path override (incl. GGUF/mixed-export selection) and the
        # plain component directory, returning the shard list plus an
        # optional override config path (unused here; the AR config comes
        # from the unified checkpoint's config.json).
        safetensors_list = list(
            resolve_transformer_checkpoint_files(server_args, model_path).safetensors
        )

        local_torch_device = get_local_torch_device()
        cpu_offload = bool(server_args.dit_cpu_offload)
        checkpoint_load_device = (
            torch.device("cpu") if cpu_offload else local_torch_device
        )
        fsdp_inference = bool(server_args.use_fsdp_inference)
        if fsdp_inference and current_platform.is_mps():
            logger.warning("Disabling FSDP for MPS platform as it's not compatible")
            fsdp_inference = False

        param_dtype = PRECISION_TO_TYPE[pipeline_config.dit_precision]
        logger.info(
            "Loading AR model: %d shard(s), %s",
            len(safetensors_list),
            param_dtype,
        )

        attn_backend, matched_backend_key = (
            server_args.resolve_component_attention_backend("transformer")
        )
        with component_attn_backend_context_manager(
            attn_backend, component_name=matched_backend_key or "transformer"
        ):
            with (
                set_default_torch_dtype(param_dtype),
                torch.device(checkpoint_load_device),
            ):
                model = HunyuanImage3ForCausalMM(
                    config=pipeline_config.dit_config,
                    hf_config=config_dict,
                )

            weights_to_load = {name for name, _ in model.named_parameters()}
            loaded_weights = model.load_weights(
                safetensors_weights_iterator(safetensors_list)
            )
            weights_not_loaded = weights_to_load - loaded_weights
            if weights_not_loaded:
                raise ValueError(
                    f"AR weights not initialized from checkpoint: "
                    f"{sorted(weights_not_loaded)}"
                )

            model.post_load_weights()
            for param in model.parameters():
                param.requires_grad = False

            if fsdp_inference:
                self._shard_ar_model(
                    model,
                    server_args=server_args,
                    cpu_offload=cpu_offload,
                    param_dtype=param_dtype,
                )
        model.eval()
        self.memory_usages["transformer"] = _module_memory_gb(model)
        return model

    def _shard_ar_model(
        self,
        model: torch.nn.Module,
        server_args: ServerArgs,
        cpu_offload: bool,
        param_dtype: torch.dtype,
    ) -> None:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=torch.float32,
            cast_forward_inputs=False,
        )
        set_mixed_precision_policy(
            param_dtype=param_dtype,
            reduce_dtype=torch.float32,
            mp_policy=mp_policy,
        )
        device_mesh = init_device_mesh(
            current_platform.device_type,
            mesh_shape=(server_args.hsdp_replicate_dim, server_args.hsdp_shard_dim),
            mesh_dim_names=("replicate", "shard"),
        )
        shard_model(
            model,
            cpu_offload=cpu_offload,
            reshard_after_forward=True,
            mp_policy=mp_policy,
            mesh=device_mesh,
            fsdp_shard_conditions=getattr(model, "_fsdp_shard_conditions", None),
            pin_cpu_memory=server_args.pin_cpu_memory,
        )

    def _load_vae(
        self,
        server_args: ServerArgs,
        pipeline_config: Any,
        model_path: str,
        vae_config_dict: dict[str, Any],
    ) -> torch.nn.Module:
        # config.json omits a few constructor args
        vae_params = dict(vae_config_dict)
        vae_params.setdefault("in_channels", 3)
        vae_params.setdefault("out_channels", 3)
        vae_params.setdefault("ffactor_temporal", 4)
        vae = AutoencoderKLConv3D(**vae_params)

        dtype = PRECISION_TO_TYPE[pipeline_config.vae_precision]
        vae.to(dtype=dtype)

        state_dict = self._collect_prefixed_weights(model_path, "vae.")
        missing_keys, _unexpected = vae.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logger.warning(
                "VAE missing %d key(s), e.g. %s",
                len(missing_keys),
                missing_keys[:3],
            )

        device = (
            torch.device("cpu")
            if server_args.should_cpu_offload_component("vae")
            else get_local_torch_device()
        )
        vae.to(device=device)
        vae.eval()
        if server_args.pipeline_config.vae_tiling:
            vae.enable_tiling()
        if server_args.pipeline_config.vae_slicing:
            vae.enable_slicing()
        self.memory_usages["vae"] = _module_memory_gb(vae)
        return vae

    @staticmethod
    def _load_tokenizer(server_args: ServerArgs, model_path: str):
        return AutoTokenizer.from_pretrained(
            model_path,
            revision=server_args.revision,
            trust_remote_code=True,
        )

    def _load_processor(self, server_args: ServerArgs, model_path: str, hf_config):
        processor_cls = get_class_from_dynamic_module(
            "image_processor.HunyuanImage3ImageProcessor",
            model_path,
            revision=server_args.revision,
        )
        return processor_cls(hf_config)

    def _load_vision_model(
        self,
        model_path: str,
        config_dict: dict[str, Any],
    ) -> torch.nn.Module:
        vit_config = config_dict["vit"]
        # Reference SigLIP2 ViT (plain PyTorch, F.sdpa packed attention)
        # instead of SRT Siglip2Model, matching the official cond-image path.
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan_image3.vision import (
            Siglip2VisionTransformer,
        )

        vision_model = Siglip2VisionTransformer(vit_config)

        state_dict = self._collect_prefixed_weights(model_path, "vision_model.")
        missing_keys, _unexpected = vision_model.load_state_dict(
            state_dict, strict=False
        )
        if missing_keys:
            logger.warning(
                "vision_model missing %d key(s), e.g. %s",
                len(missing_keys),
                sorted(missing_keys)[:3],
            )

        device = get_local_torch_device()
        vision_model.to(device=device)
        vision_model.eval()
        self.memory_usages["vision_model"] = _module_memory_gb(vision_model)
        logger.info(
            "Loaded vision_model (%.2f GiB)", self.memory_usages["vision_model"]
        )
        return vision_model

    def _load_vision_aligner(
        self,
        model_path: str,
        config_dict: dict[str, Any],
    ) -> torch.nn.Module:
        vision_aligner = LightProjector(config_dict["vit_aligner"])
        state_dict = self._collect_prefixed_weights(model_path, "vision_aligner.")
        missing_keys, _unexpected = vision_aligner.load_state_dict(
            state_dict, strict=False
        )
        if missing_keys:
            logger.warning(
                "vision_aligner missing %d key(s), e.g. %s",
                len(missing_keys),
                missing_keys[:3],
            )

        device = get_local_torch_device()
        vision_aligner.to(device=device)
        vision_aligner.eval()
        self.memory_usages["vision_aligner"] = _module_memory_gb(vision_aligner)
        logger.info(
            "Loaded vision_aligner (%.2f GiB)", self.memory_usages["vision_aligner"]
        )
        return vision_aligner

    @staticmethod
    def _collect_prefixed_weights(
        model_path: str, prefix: str
    ) -> dict[str, torch.Tensor]:
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shard_names = sorted(
            {shard for key, shard in weight_map.items() if key.startswith(prefix)}
        )
        shard_paths = [os.path.join(model_path, name) for name in shard_names]

        state_dict: dict[str, torch.Tensor] = {}
        for name, tensor in safetensors_weights_iterator(shard_paths):
            if name.startswith(prefix):
                state_dict[name[len(prefix) :]] = tensor
        return state_dict

    @staticmethod
    def _read_flow_shift(model_path: str) -> float:
        return float(
            load_dict(os.path.join(model_path, "generation_config.json"))["flow_shift"]
        )

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(
            HunyuanImage3AR(
                ar_model=self.get_module("transformer"),
                vae=self.get_module("vae"),
                tokenizer=self.get_module("tokenizer"),
                processor=self.get_module("processor"),
                scheduler=self.get_module("scheduler"),
                model_path=self.model_path,
                vision_model=self.get_module("vision_model"),
                vision_aligner=self.get_module("vision_aligner"),
                # Reuse the scheduler's group ceiling as the initial cond-encode
                # chunk cap; 1 (the no-grouping default) means unlimited.
                max_cond_encode_chunk=(
                    server_args.batching_max_size
                    if server_args.batching_max_size > 1
                    else None
                ),
            ),
            "hunyuan_image3_ar",
        )

        self.add_stage_factory(
            RoleType.DECODER,
            lambda: HunyuanImage3DecodingStage(
                vae=self.get_module("vae"), pipeline=self
            ),
            "decoding_stage",
        )

    def forward_batch(self, batches, server_args: ServerArgs):
        if len(batches) > 1 and self.executor.component_residency_manager is None:
            self.component_residency_manager = get_global_component_residency_manager(
                self, server_args
            )
            self.executor.component_residency_manager = self.component_residency_manager
        return super().forward_batch(batches, server_args)


EntryClass = [HunyuanImage3Pipeline]
