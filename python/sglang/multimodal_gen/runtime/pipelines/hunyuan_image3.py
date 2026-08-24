import json
import os
from itertools import chain
from typing import Any

import torch
from torch.distributed import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    component_attn_backend_context_manager,
)
from sglang.multimodal_gen.runtime.loader.fsdp_load import shard_model
from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
    resolve_transformer_safetensors_to_load,
)
from sglang.multimodal_gen.runtime.loader.utils import (
    _list_safetensors_files,
    set_default_torch_dtype,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    safetensors_weights_iterator,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.models.dits.hunyuan_image3 import (
    HunyuanImage3ForCausalMM,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan_image3 import (
    HunyuanImage3AR,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_hf_config,
    maybe_download_model,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision_types import PRECISION_TO_TYPE
from sglang.multimodal_gen.utils import set_mixed_precision_policy

logger = init_logger(__name__)

# flow-matching timestep shift used by the official HunyuanImage-3 scheduler
# (generation_config.json: flow_shift)
_DEFAULT_FLOW_SHIFT = 3.0


def _module_memory_gb(module: Any) -> float:
    """Approximate GPU memory footprint of a module's parameters/buffers (GiB)."""
    if not isinstance(module, torch.nn.Module):
        return 0.0
    total_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
    total_bytes += sum(b.numel() * b.element_size() for b in module.buffers())
    return total_bytes / (1024**3)


class HunyuanImage3Pipeline(LoRAPipeline, ComposedPipelineBase):
    """Pipeline for HunyuanImage-3 text-to-image generation."""

    pipeline_name = "HunyuanImage3Pipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "processor",
        "transformer",
        "scheduler",
    ]

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        """
        Load HunyuanImage-3 components from a transformers-format repository.

        HunyuanImage-3 is published as a single unified checkpoint (one
        config.json + sharded model-*.safetensors with custom code files),
        without model_index.json or per-component subfolders, so the default
        diffusers component loading cannot be used. All components are carved
        out of the same repository here:

        - transformer / text_encoder / vision_language_encoder: the unified
          AR backbone (HunyuanImage3ForCausalMM), loaded from the shared
          safetensors shards
        - vae: the repo's AutoencoderKLConv3D, loaded from the "vae.*" keys
        - tokenizer: the fast tokenizer shipped in the repo root
        - scheduler: flow-matching Euler scheduler with the repo's flow_shift
        - processor: the repo's HunyuanImage3ImageProcessor (best effort)
        """
        required = list(self.required_config_modules)
        modules: dict[str, Any] = {}
        if loaded_modules:
            for module_name in required:
                if module_name in loaded_modules:
                    logger.info("Using provided module: %s", module_name)
                    modules[module_name] = loaded_modules[module_name]
        if set(modules) == set(required):
            return modules

        pipeline_config = server_args.pipeline_config

        # 1. resolve the local model path and read the transformers config
        model_path = maybe_download_model(self.model_path)
        self.model_path = model_path
        logger.info("Loading HunyuanImage-3 components from %s", model_path)

        hf_config = get_hf_config(
            model_path,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
        )
        config_dict = hf_config.to_dict()

        # 2. feed architecture metadata back into the pipeline configs
        pipeline_config.dit_config.update_model_arch(config_dict)
        vae_config_dict = dict(config_dict.get("vae", {}))
        vae_config_dict.pop("_class_name", None)
        pipeline_config.vae_config.update_model_arch(vae_config_dict)
        if hasattr(pipeline_config.vae_config, "post_init"):
            pipeline_config.vae_config.post_init()

        flow_shift = self._read_flow_shift(model_path)
        pipeline_config.flow_shift = flow_shift

        # 3. load the heavy components
        ar_model = None
        if any(
            name not in modules
            for name in ("transformer", "text_encoder")
        ):
            ar_model = self._load_ar_model(
                server_args, pipeline_config, model_path, config_dict
            )
        if ar_model is None:
            # e.g. the transformer was provided via loaded_modules
            ar_model = modules.get("transformer")

        for module_name in required:
            if module_name in modules:
                continue
            if module_name == "transformer":
                modules["transformer"] = ar_model
            elif module_name == "text_encoder":
                # HunyuanImage-3 has no standalone text encoder: the unified
                # AR backbone provides text conditioning
                modules["text_encoder"] = ar_model
            elif module_name == "vae":
                modules["vae"] = self._load_vae(
                    server_args, pipeline_config, model_path, vae_config_dict
                )
            elif module_name == "tokenizer":
                modules["tokenizer"] = self._load_tokenizer(server_args, model_path)
            elif module_name == "scheduler":
                modules["scheduler"] = FlowMatchEulerDiscreteScheduler(
                    shift=flow_shift
                )
            elif module_name == "processor":
                modules["processor"] = self._load_processor(
                    server_args, model_path, hf_config
                )
            else:
                raise ValueError(f"Unknown required module: {module_name}")

        logger.debug("Memory usage of loaded modules (GiB): %s", self.memory_usages)
        return modules

    # --- component loaders ---------------------------------------------------

    def _load_ar_model(
        self,
        server_args: ServerArgs,
        pipeline_config: Any,
        model_path: str,
        config_dict: dict[str, Any],
    ) -> torch.nn.Module:
        """Load the unified AR backbone (also serves as the DiT).

        The official checkpoint ships fused/interleaved layouts (per-group
        interleaved QKV, [up; gate] fused projections, individual per-expert
        tensors) that the generic state-dict mapper does not understand, so
        the weights are loaded through the model's own vLLM-style
        ``load_weights``, which converts all of them natively.
        """
        safetensors_list = resolve_transformer_safetensors_to_load(
            server_args, model_path
        )
        server_args.model_paths["transformer"] = model_path

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
            "Loading HunyuanImage3ForCausalMM from %s safetensors file(s), param_dtype: %s",
            len(safetensors_list),
            param_dtype,
        )

        attn_backend, matched_backend_key = (
            server_args.resolve_component_attention_backend("transformer")
        )
        with component_attn_backend_context_manager(
            attn_backend, component_name=matched_backend_key or "transformer"
        ):
            with set_default_torch_dtype(param_dtype), torch.device(
                checkpoint_load_device
            ):
                model = HunyuanImage3ForCausalMM(
                    config=pipeline_config.dit_config,
                    hf_config=config_dict,
                )

            weights_to_load = {name for name, _ in model.named_parameters()}
            loaded_weights = model.load_weights(
                safetensors_weights_iterator(safetensors_list)
            )
            weights_not_loaded = weights_to_load - (loaded_weights or set())
            if weights_not_loaded:
                raise ValueError(
                    "Following HunyuanImage-3 AR weights were not initialized "
                    f"from checkpoint: {sorted(weights_not_loaded)}. This usually "
                    "indicates a checkpoint/model-arch mismatch or a broken "
                    "weight-name mapping."
                )

            # Post-load fixups normally performed by maybe_load_fsdp_model.
            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None and hasattr(
                    quant_method, "process_weights_after_loading"
                ):
                    quant_method.process_weights_after_loading(module)
            model.post_load_weights()
            for name, param in chain(model.named_parameters(), model.named_buffers()):
                if param.is_meta:
                    raise RuntimeError(
                        f"Unexpected param or buffer {name} on meta device."
                    )
                if isinstance(param, torch.nn.Parameter):
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
        """Apply FSDP sharding to the already-loaded AR backbone."""
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
        """Load the repo's AutoencoderKLConv3D and fill it with the "vae.*" weights."""
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        vae_cls = get_class_from_dynamic_module(
            "autoencoder_kl_3d.AutoencoderKLConv3D",
            model_path,
            revision=server_args.revision,
        )
        # the vae section of config.json omits a few constructor args
        vae_params = dict(vae_config_dict)
        vae_params.setdefault("in_channels", 3)
        vae_params.setdefault("out_channels", 3)
        vae_params.setdefault("ffactor_temporal", 4)
        vae = vae_cls(**vae_params)

        dtype = PRECISION_TO_TYPE.get(pipeline_config.vae_precision, torch.float32)
        vae.to(dtype=dtype)

        state_dict = self._collect_vae_weights(model_path)
        missing_keys, _unexpected = vae.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logger.warning(
                "Missing %d key(s) when loading HunyuanImage-3 VAE, e.g. %s",
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
        self.memory_usages["vae"] = _module_memory_gb(vae)
        return vae

    @staticmethod
    def _collect_vae_weights(model_path: str) -> dict[str, torch.Tensor]:
        """Extract the "vae.*" keys from the unified checkpoint, stripping the prefix."""
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                weight_map = json.load(f).get("weight_map", {})
            shard_names = sorted(
                {
                    shard
                    for key, shard in weight_map.items()
                    if key.startswith("vae.")
                }
            )
            shard_paths = [os.path.join(model_path, name) for name in shard_names]
        else:
            shard_paths = _list_safetensors_files(model_path)

        state_dict: dict[str, torch.Tensor] = {}
        for name, tensor in safetensors_weights_iterator(shard_paths):
            if name.startswith("vae."):
                state_dict[name[len("vae.") :]] = tensor
        return state_dict

    @staticmethod
    def _load_tokenizer(server_args: ServerArgs, model_path: str):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            model_path,
            revision=server_args.revision,
            trust_remote_code=server_args.trust_remote_code,
        )

    def _load_processor(self, server_args: ServerArgs, model_path: str, hf_config):
        """Best-effort load of the repo's HunyuanImage3ImageProcessor."""
        if not server_args.trust_remote_code:
            logger.warning(
                "trust_remote_code is disabled; skipping HunyuanImage3ImageProcessor loading"
            )
            return None
        try:
            from transformers.dynamic_module_utils import (
                get_class_from_dynamic_module,
            )

            processor_cls = get_class_from_dynamic_module(
                "image_processor.HunyuanImage3ImageProcessor",
                model_path,
                revision=server_args.revision,
            )
            return processor_cls(hf_config)
        except Exception as e:
            logger.warning(
                "Failed to load HunyuanImage3ImageProcessor from %s: %s. "
                "Continuing without it; the AR stage does not require it yet.",
                model_path,
                e,
            )
            return None

    @staticmethod
    def _read_flow_shift(model_path: str) -> float:
        """Read flow_shift from generation_config.json, falling back to the official default."""
        gen_config_path = os.path.join(model_path, "generation_config.json")
        if os.path.exists(gen_config_path):
            try:
                with open(gen_config_path) as f:
                    flow_shift = json.load(f).get("flow_shift")
                if flow_shift is not None:
                    return float(flow_shift)
            except Exception as e:
                logger.warning(
                    "Failed to read flow_shift from %s: %s", gen_config_path, e
                )
        return _DEFAULT_FLOW_SHIFT

    # --- pipeline stages -----------------------------------------------------

    def create_pipeline_stages(self, server_args: ServerArgs):
        # Stage 1: AR latent generation. Runs the native diffusion loop
        # with every backbone pass routed into the sglang backbone's
        # forward_block. Stops before VAE decode.
        self.add_stage(
            HunyuanImage3AR(
                ar_model=self.get_module("transformer"),
                vae=self.get_module("vae"),
                tokenizer=self.get_module("tokenizer"),
                processor=self.get_module("processor"),
                scheduler=self.get_module("scheduler"),
                model_path=self.model_path,
            ),
            "hunyuan_image3_ar",
        )

        # Stage 2: VAE decoding
        self.add_standard_decoding_stage()


EntryClass = [HunyuanImage3Pipeline]
