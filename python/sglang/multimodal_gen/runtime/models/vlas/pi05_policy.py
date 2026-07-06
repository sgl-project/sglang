# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy

from sglang.multimodal_gen.configs.pipeline_configs.pi05 import Pi05PipelineConfig
from sglang.multimodal_gen.runtime.cache.vla_prefix_cache import (
    PrefixContext,
    VLAPrefixCacheKey,
    VLAPrefixCacheManager,
)
from sglang.multimodal_gen.runtime.distributed.vla import get_vla_split_group
from sglang.multimodal_gen.runtime.loader.utils import (
    set_default_torch_dtype,
    skip_init_modules,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    safetensors_weights_iterator,
)
from sglang.multimodal_gen.runtime.models.vlas.pi05_core import Pi05CoreModel
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.vla_denoise_graph import (
    VLADenoiseGraphRunner,
    VLADenoiseShapeBucket,
)
from sglang.multimodal_gen.runtime.utils.vla_observation import (
    VLAObservationBatch,
    stable_tensor_sha256,
)
from sglang.multimodal_gen.utils import set_mixed_precision_policy

logger = init_logger(__name__)


@dataclass
class Pi05CheckpointManifest:
    model_path: str
    safetensor_files: list[str] = field(default_factory=list)
    component_keys: dict[str, list[str]] = field(default_factory=dict)
    skipped_lm_head_keys: list[str] = field(default_factory=list)


class Pi05ActionExpert(nn.Module):
    def __init__(self, config: Pi05PipelineConfig, core_model: Pi05CoreModel):
        super().__init__()
        self.config = config
        self.core_model = core_model

    def forward(
        self,
        prefix_context: PrefixContext,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        return self.core_model.denoise_step(
            prefix_context.prefix_pad_masks,
            prefix_context.past_key_values,
            x_t,
            timestep,
            bool(prefix_context.layout.get("full_attention", False)),
        )


class Pi05PolicyModel(nn.Module):
    _ROLE_COMPONENTS = {
        "all": None,
        "prefix": {"vision_tower", "paligemma", "multi_modal_projector"},
        "action": {"action_expert", "action_heads"},
        "idle": set(),
    }

    def __init__(
        self,
        config: Pi05PipelineConfig,
        *,
        model_path: str,
        device: torch.device,
        dtype: torch.dtype,
        manifest: Pi05CheckpointManifest,
    ):
        super().__init__()
        self.config = config
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.manifest = manifest
        self.runtime_role = self._resolve_runtime_role()
        mp_policy = MixedPrecisionPolicy(
            dtype,
            dtype,
            dtype,
            cast_forward_inputs=False,
        )
        set_mixed_precision_policy(
            param_dtype=dtype,
            reduce_dtype=dtype,
            output_dtype=dtype,
            mp_policy=mp_policy,
        )
        with set_default_torch_dtype(dtype), skip_init_modules():
            self.core_model = Pi05CoreModel(config, runtime_role=self.runtime_role)
        self.core_model.eval()
        if device.type == "cuda":
            if self._use_componentwise_empty_init():
                self._componentwise_empty_init()
            else:
                self._to_empty_preserve_buffers(self.core_model, device=device)
                self._set_prefix_output_device()
                self._move_offloaded_prefix_modules_to_empty_cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            self._load_weights()
            torch.cuda.empty_cache()
        else:
            self._load_weights()
            self.core_model.to(device)
            self._set_prefix_output_device()
        if self.runtime_role != "all":
            logger.info("Pi05 split runtime role on rank: %s", self.runtime_role)
        self.action_expert = Pi05ActionExpert(config, self.core_model)
        self.graph_runner = VLADenoiseGraphRunner(
            enabled=config.enable_action_cuda_graph
        )

    @staticmethod
    def _to_empty_preserve_buffers(module: nn.Module, *, device: torch.device) -> None:
        buffers = {
            name: buffer.detach().clone().to(device=device)
            for name, buffer in module.named_buffers(recurse=True)
        }
        module.to_empty(device=device)
        for name, buffer in buffers.items():
            if "." in name:
                parent_name, buffer_name = name.rsplit(".", 1)
                parent = module.get_submodule(parent_name)
            else:
                parent = module
                buffer_name = name
            parent._buffers[buffer_name] = buffer

    def _use_componentwise_empty_init(self) -> bool:
        prefix_offload = (
            self.config.offload_prefix_image_encoder
            or self.config.offload_prefix_image_encoder_after_embed
            or self.config.offload_prefix_token_embedding
            or self.config.offload_prefix_language_layers
            or self.config.offload_prefix_language_layers_after_prefix
            or self.config.offload_prefix_language_layer_count_after_prefix > 0
        )
        return (
            self.runtime_role in ("all", "prefix") and prefix_offload
        ) or self._offload_action_expert_between_requests()

    def _prefix_language_phase_offload_layer_count(self, layer_count: int) -> int:
        if self.config.offload_prefix_language_layers_after_prefix:
            return layer_count
        return min(
            max(self.config.offload_prefix_language_layer_count_after_prefix, 0),
            layer_count,
        )

    def _componentwise_empty_init(self) -> None:
        logger.info(
            "Pi05 componentwise empty init enabled for runtime role %s",
            self.runtime_role,
        )
        self._to_empty_preserve_buffers(self.core_model, device=torch.device("cpu"))
        paligemma = self.core_model.paligemma_with_expert.paligemma
        if paligemma is not None:
            language_model = paligemma.model.language_model
            self._to_empty_preserve_buffers(
                language_model.rotary_emb,
                device=self.device,
            )
            self._to_empty_preserve_buffers(language_model.norm, device=self.device)

            if (
                self.config.offload_prefix_image_encoder
                or self.config.offload_prefix_image_encoder_after_embed
            ):
                self._to_empty_preserve_buffers(
                    paligemma.model.vision_tower,
                    device=torch.device("cpu"),
                )
                self._to_empty_preserve_buffers(
                    paligemma.model.multi_modal_projector,
                    device=torch.device("cpu"),
                )
            else:
                self._to_empty_preserve_buffers(
                    paligemma.model.vision_tower,
                    device=self.device,
                )
                self._to_empty_preserve_buffers(
                    paligemma.model.multi_modal_projector,
                    device=self.device,
                )

            if self.config.offload_prefix_token_embedding:
                self._to_empty_preserve_buffers(
                    language_model.embed_tokens,
                    device=torch.device("cpu"),
                )
            else:
                self._to_empty_preserve_buffers(
                    language_model.embed_tokens,
                    device=self.device,
                )

            phase_layer_count = self._prefix_language_phase_offload_layer_count(
                len(language_model.layers)
            )
            if self.config.offload_prefix_language_layers:
                self._to_empty_preserve_buffers(
                    language_model.layers,
                    device=torch.device("cpu"),
                )
                language_model.configure_layerwise_cpu_offload(
                    compute_device=self.device,
                    empty_cache=self.config.offload_prefix_language_layers_empty_cache,
                )
            elif phase_layer_count:
                for i, layer in enumerate(language_model.layers):
                    self._to_empty_preserve_buffers(
                        layer,
                        device=(
                            torch.device("cpu")
                            if i < phase_layer_count
                            else self.device
                        ),
                    )
            else:
                self._to_empty_preserve_buffers(
                    language_model.layers,
                    device=self.device,
                )

            self._set_prefix_output_device()
        action_device = (
            torch.device("cpu")
            if self._offload_action_expert_between_requests()
            else self.device
        )
        self._move_action_modules_to_empty_device(action_device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _move_action_modules_to_empty_device(self, device: torch.device) -> None:
        gemma_expert = self.core_model.paligemma_with_expert.gemma_expert
        if gemma_expert is not None:
            self._to_empty_preserve_buffers(gemma_expert, device=device)
        for module in (
            self.core_model.action_in_proj,
            self.core_model.action_out_proj,
            self.core_model.time_mlp_in,
            self.core_model.time_mlp_out,
        ):
            if module is not None:
                self._to_empty_preserve_buffers(module, device=device)

    def _offload_action_expert_between_requests(self) -> bool:
        return (
            self.runtime_role == "all"
            and self.device.type == "cuda"
            and self.config.offload_action_expert_after_denoise
        )

    def _move_action_modules_to_device(self, device: torch.device) -> None:
        gemma_expert = self.core_model.paligemma_with_expert.gemma_expert
        if gemma_expert is not None:
            gemma_expert.to(device)
        for module in (
            self.core_model.action_in_proj,
            self.core_model.action_out_proj,
            self.core_model.time_mlp_in,
            self.core_model.time_mlp_out,
        ):
            if module is not None:
                module.to(device)

    def _set_prefix_output_device(self) -> None:
        paligemma_with_expert = self.core_model.paligemma_with_expert
        if paligemma_with_expert.paligemma is not None:
            paligemma_with_expert.set_prefix_output_device(self.device)

    def _move_offloaded_prefix_modules_to_empty_cpu(self) -> None:
        paligemma = self.core_model.paligemma_with_expert.paligemma
        if paligemma is None:
            return
        cpu = torch.device("cpu")
        if (
            self.config.offload_prefix_image_encoder
            or self.config.offload_prefix_image_encoder_after_embed
        ):
            self._to_empty_preserve_buffers(
                paligemma.model.vision_tower,
                device=cpu,
            )
            self._to_empty_preserve_buffers(
                paligemma.model.multi_modal_projector,
                device=cpu,
            )
        if self.config.offload_prefix_token_embedding:
            self._to_empty_preserve_buffers(
                paligemma.model.language_model.embed_tokens,
                device=cpu,
            )
        language_model = paligemma.model.language_model
        phase_layer_count = self._prefix_language_phase_offload_layer_count(
            len(language_model.layers)
        )
        if self.config.offload_prefix_language_layers:
            self._to_empty_preserve_buffers(language_model.layers, device=cpu)
            language_model.configure_layerwise_cpu_offload(
                compute_device=self.device,
                empty_cache=self.config.offload_prefix_language_layers_empty_cache,
            )
        elif phase_layer_count:
            for layer in language_model.layers[:phase_layer_count]:
                self._to_empty_preserve_buffers(layer, device=cpu)

    @staticmethod
    def _resolve_runtime_role() -> str:
        split = get_vla_split_group()
        if split is None:
            return "all"
        if split.is_prefix_rank:
            return "prefix"
        if split.is_action_rank:
            return "action"
        return "idle"

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Pi05PipelineConfig,
        *,
        dtype: torch.dtype | None = None,
    ) -> Pi05PolicyModel:
        local_path = maybe_download_model(
            model_path,
            force_diffusers_model=False,
            allow_patterns=["*.json", "*.model", "*.safetensors", "*.txt"],
        )
        cls._apply_checkpoint_config(local_path, config)
        device = torch.device(current_platform.device_type)
        dtype = dtype or cls._dtype_from_config(config.materialize_dtype)
        manifest = cls._inspect_checkpoint(local_path, config)
        logger.info(
            "Loaded Pi05 checkpoint manifest from %s (%d safetensors, %d skipped lm_head tensors)",
            local_path,
            len(manifest.safetensor_files),
            len(manifest.skipped_lm_head_keys),
        )
        return cls(
            config,
            model_path=local_path,
            device=device,
            dtype=dtype,
            manifest=manifest,
        )

    @staticmethod
    def _dtype_from_config(dtype_name: str) -> torch.dtype:
        name = (dtype_name or "bf16").lower()
        if name in ("bf16", "bfloat16"):
            return torch.bfloat16
        if name in ("fp16", "float16", "half"):
            return torch.float16
        if name in ("fp32", "float32"):
            return torch.float32
        raise ValueError(f"Unsupported Pi05 dtype: {dtype_name}")

    @staticmethod
    def _apply_checkpoint_config(
        model_path: str,
        config: Pi05PipelineConfig,
    ) -> None:
        config_path = Path(model_path) / "config.json"
        if not config_path.exists():
            return
        with open(config_path, encoding="utf-8") as f:
            payload = json.load(f)

        config.paligemma_variant = payload.get(
            "paligemma_variant", config.paligemma_variant
        )
        config.action_expert_variant = payload.get(
            "action_expert_variant", config.action_expert_variant
        )
        config.action_horizon = int(payload.get("chunk_size", config.action_horizon))
        config.n_action_steps = int(
            payload.get("n_action_steps", config.n_action_steps)
        )
        config.action_dim = int(payload.get("max_action_dim", config.action_dim))
        config.state_dim = int(payload.get("max_state_dim", config.state_dim))
        config.default_num_inference_steps = int(
            payload.get("num_inference_steps", config.default_num_inference_steps)
        )
        config.max_token_len = int(
            payload.get("tokenizer_max_length", config.max_token_len)
        )
        config.time_embedding_min_period = float(
            payload.get("min_period", config.time_embedding_min_period)
        )
        config.time_embedding_max_period = float(
            payload.get("max_period", config.time_embedding_max_period)
        )
        if "image_resolution" in payload:
            resolution = tuple(payload["image_resolution"])
            config.image_size = (int(resolution[0]), int(resolution[1]))
        input_features = payload.get("input_features") or {}
        image_keys = []
        for key, feature in input_features.items():
            if feature.get("type") == "VISUAL":
                image_keys.append(key.rsplit(".", 1)[-1])
            elif feature.get("type") == "STATE":
                shape = feature.get("shape") or []
                if shape:
                    config.state_dim = int(shape[0])
        config.empty_cameras = int(payload.get("empty_cameras", 0) or 0)
        image_keys.extend(f"empty_camera_{i}" for i in range(config.empty_cameras))
        if image_keys:
            config.image_keys = tuple(image_keys)

        output_features = payload.get("output_features") or {}
        action_feature = output_features.get("action") or {}
        action_shape = action_feature.get("shape") or []
        if action_shape:
            config.output_action_dim = int(action_shape[0])

    @staticmethod
    def _inspect_checkpoint(
        model_path: str,
        config: Pi05PipelineConfig,
    ) -> Pi05CheckpointManifest:
        path = Path(model_path)
        safetensor_files = sorted(str(p) for p in path.glob("*.safetensors"))
        component_keys = {name: [] for name in config.loader_component_map}
        skipped_lm_head_keys: list[str] = []

        try:
            from safetensors import safe_open
        except ImportError:
            return Pi05CheckpointManifest(
                model_path=model_path,
                safetensor_files=safetensor_files,
                component_keys=component_keys,
            )

        for filename in safetensor_files:
            with safe_open(filename, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if (
                        config.skip_unused_lm_head
                        and key == "paligemma_with_expert.gemma_expert.lm_head.weight"
                    ):
                        skipped_lm_head_keys.append(key)
                        continue
                    for component, prefixes in config.loader_component_map.items():
                        if any(key.startswith(prefix) for prefix in prefixes):
                            component_keys[component].append(key)
                            break

        return Pi05CheckpointManifest(
            model_path=model_path,
            safetensor_files=safetensor_files,
            component_keys=component_keys,
            skipped_lm_head_keys=skipped_lm_head_keys,
        )

    @staticmethod
    def _candidate_weight_keys(key: str) -> list[str]:
        if key.startswith("model."):
            key = key[len("model.") :]
        if key.startswith("PaligemmaWithExpert."):
            key = key.replace("PaligemmaWithExpert.", "paligemma_with_expert.", 1)
        if key.startswith("action_time_mlp_in."):
            key = key.replace("action_time_mlp_in.", "time_mlp_in.", 1)
        elif key.startswith("action_time_mlp_out."):
            key = key.replace("action_time_mlp_out.", "time_mlp_out.", 1)
        if key.startswith("state_proj."):
            return []
        if key == "paligemma_with_expert.gemma_expert.lm_head.weight":
            return []

        candidates = [key]
        replacements = {
            ".vision_tower.vision_model.": ".vision_tower.",
            ".paligemma.language_model.": ".paligemma.model.language_model.",
            ".paligemma.vision_tower.": ".paligemma.model.vision_tower.",
            ".paligemma.multi_modal_projector.": (
                ".paligemma.model.multi_modal_projector."
            ),
        }
        for old, new in replacements.items():
            if old in key:
                candidates.append(key.replace(old, new))

        if key in {
            "paligemma_with_expert.paligemma.lm_head.weight",
            "paligemma_with_expert.paligemma.model.lm_head.weight",
        }:
            candidates.append(
                "paligemma_with_expert.paligemma.model.language_model."
                "embed_tokens.weight"
            )
        return list(dict.fromkeys(candidates))

    def _component_for_source_key(self, key: str) -> str | None:
        if key in {
            "paligemma_with_expert.paligemma.lm_head.weight",
            "paligemma_with_expert.paligemma.model.lm_head.weight",
        }:
            return "paligemma"
        candidates = self._candidate_weight_keys(key)
        if not candidates:
            return None
        for component, prefixes in self.config.loader_component_map.items():
            if any(
                candidate.startswith(prefix)
                for candidate in candidates
                for prefix in prefixes
            ):
                return component
        return None

    def _should_load_source_key(self, key: str) -> bool:
        components = self._ROLE_COMPONENTS[self.runtime_role]
        if components is None:
            return True
        component = self._component_for_source_key(key)
        return component in components

    def _should_read_source_key(self, key: str) -> bool:
        return self._should_load_source_key(key) and bool(
            self._candidate_weight_keys(key)
        )

    def _resolve_target_key(
        self,
        source_key: str,
        target_state: dict[str, torch.Tensor],
    ) -> str | None:
        candidates = self._candidate_weight_keys(source_key)
        if not candidates:
            return None
        for candidate in candidates:
            if candidate in target_state:
                return candidate
        return None

    def _should_stream_weights_to_gpu(
        self,
        target_state: dict[str, torch.Tensor],
    ) -> bool:
        if self.device.type != "cuda" or self.runtime_role in ("action", "idle"):
            return False
        # Split prefix/action ranks need a distributed streamer protocol; the
        # direct GPU path is only safe for a single local loading process.
        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            return False

        has_weight = False
        for filename in self.manifest.safetensor_files:
            with safe_open(filename, framework="pt", device="cpu") as f:
                for source_key in f.keys():
                    if not self._should_read_source_key(source_key):
                        continue
                    target_key = self._resolve_target_key(source_key, target_state)
                    if target_key is None:
                        continue
                    has_weight = True
                    if target_state[target_key].device.type != "cuda":
                        return False
        return has_weight

    def _load_weights(self) -> None:
        target_state = self.core_model.state_dict()
        loaded_keys: set[str] = set()
        unexpected = 0
        mismatched = 0
        stream_to_gpu = self._should_stream_weights_to_gpu(target_state)
        if stream_to_gpu:
            logger.info(
                "Pi05 weight load streams safetensors directly to %s",
                self.device,
            )

        with torch.no_grad():
            if stream_to_gpu:
                weights = safetensors_weights_iterator(
                    self.manifest.safetensor_files,
                    to_cpu=False,
                    key_filter=self._should_read_source_key,
                    clone_streamed_tensors=False,
                )
                for source_key, tensor in weights:
                    target_key = self._resolve_target_key(source_key, target_state)
                    if target_key is None:
                        unexpected += 1
                        continue
                    target = target_state[target_key]
                    if tuple(target.shape) != tuple(tensor.shape):
                        mismatched += 1
                        continue
                    if tensor.dtype != target.dtype:
                        tensor = tensor.to(dtype=target.dtype)
                    target.copy_(tensor, non_blocking=True)
                    loaded_keys.add(target_key)
            else:
                for filename in self.manifest.safetensor_files:
                    with safe_open(filename, framework="pt", device="cpu") as f:
                        for source_key in f.keys():
                            if not self._should_read_source_key(source_key):
                                continue
                            target_key = self._resolve_target_key(
                                source_key,
                                target_state,
                            )
                            if target_key is None:
                                unexpected += 1
                                continue
                            tensor = f.get_tensor(source_key)
                            target = target_state[target_key]
                            if tuple(target.shape) != tuple(tensor.shape):
                                mismatched += 1
                                continue
                            if tensor.dtype != target.dtype:
                                tensor = tensor.to(dtype=target.dtype)
                            target.copy_(
                                tensor,
                                non_blocking=target.device.type == "cuda",
                            )
                            loaded_keys.add(target_key)

        missing = [key for key in target_state if key not in loaded_keys]
        if missing or unexpected or mismatched:
            logger.warning(
                "Pi05 weight load: %d loaded, %d missing, %d unexpected, %d mismatched",
                len(loaded_keys),
                len(missing),
                unexpected,
                mismatched,
            )
            if missing:
                logger.warning("First missing Pi05 weights: %s", missing[:8])
        else:
            logger.info("Pi05 weights loaded successfully")

    def build_prefix_cache_key(
        self,
        observation: VLAObservationBatch,
        options: dict[str, Any],
    ) -> VLAPrefixCacheKey:
        state_digest = None
        if observation.state is not None:
            state_digest = stable_tensor_sha256(observation.state)

        masks = {
            name: bool(mask.item()) for name, mask in observation.image_masks.items()
        }
        model_revision = os.path.basename(os.path.normpath(self.model_path))
        return VLAPrefixCacheManager.make_key(
            model_revision=model_revision,
            tokenizer_id=f"{self.config.paligemma_variant}:{self.config.max_token_len}",
            normalization_config=options.get("normalization_config"),
            discretization_config=options.get("discretization_config"),
            camera_order=tuple(observation.metadata.get("camera_order", ())),
            image_hashes=observation.metadata.get("image_hashes", {}),
            prompt=observation.prompt,
            token_digest=stable_tensor_sha256(observation.tokens),
            state_digest=state_digest,
            masks=masks,
            positions_version=self.config.prefix_cache_layout_version,
            dtype=str(self.dtype).replace("torch.", ""),
            adapter=options.get("adapter"),
            parallel_layout_version=self.config.parallel_layout_version,
            cache_namespace="pi05",
        )

    def encode_prefix(self, observation: VLAObservationBatch) -> PrefixContext:
        camera_order = tuple(observation.metadata.get("camera_order", ()))
        images = [
            observation.images[name].to(self.device, dtype=torch.float32)
            for name in camera_order
        ]
        image_masks = [
            observation.image_masks[name].to(self.device) for name in camera_order
        ]
        token_len = int(observation.token_masks.sum(dim=1).max().item())
        tokens_trimmed = token_len > 0
        if tokens_trimmed and token_len < observation.tokens.shape[1]:
            tokens_cpu = observation.tokens[:, :token_len]
            token_masks_cpu = observation.token_masks[:, :token_len]
        else:
            tokens_cpu = observation.tokens
            token_masks_cpu = observation.token_masks
        tokens = tokens_cpu.to(self.device)
        token_masks = token_masks_cpu.to(self.device)
        prefix_full_attention_hint = all(
            bool(observation.image_masks[name].all().item()) for name in camera_order
        ) and bool(token_masks_cpu.all().item())
        past_key_values, prefix_pad_masks, prefix_position_ids, full_attention = (
            self.core_model.encode_prefix(
                images,
                image_masks,
                tokens,
                token_masks,
                prefix_full_attention_hint=prefix_full_attention_hint,
                tokens_trimmed=tokens_trimmed,
            )
        )
        return PrefixContext(
            past_key_values=past_key_values,
            prefix_pad_masks=prefix_pad_masks,
            prefix_position_ids=prefix_position_ids,
            prefix_len=prefix_pad_masks.shape[1],
            dtype=self.dtype,
            device=self.device,
            layout={
                "camera_order": camera_order,
                "full_attention": full_attention,
                "parallel_layout_version": self.config.parallel_layout_version,
            },
        )

    def sample_noise(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return torch.randn(
            batch_size,
            self.config.action_horizon,
            self.config.action_dim,
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )

    def denoise_step(
        self,
        prefix_context: PrefixContext,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        *,
        use_cuda_graph: bool = True,
    ) -> torch.Tensor:
        if not bool(prefix_context.layout.get("full_attention", False)):
            use_cuda_graph = False
        if not use_cuda_graph:
            return self.action_expert(prefix_context, x_t, timestep)
        bucket = VLADenoiseShapeBucket(
            batch_size=x_t.shape[0],
            prefix_len=prefix_context.prefix_len,
            action_horizon=x_t.shape[1],
            action_dim=x_t.shape[2],
            dtype=str(x_t.dtype).replace("torch.", ""),
            parallel_layout=self.config.parallel_layout_version,
        )
        return self.graph_runner.capture_or_run(
            bucket,
            self.action_expert,
            prefix_context,
            x_t,
            timestep,
        )

    def sample_actions(
        self,
        observation: VLAObservationBatch,
        prefix_context: PrefixContext,
        *,
        noise: torch.Tensor | None,
        num_steps: int,
        use_cuda_graph: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        offload_action = self._offload_action_expert_between_requests()
        if offload_action:
            self._move_action_modules_to_device(self.device)
            use_cuda_graph = False

        x_t = noise
        if x_t is None:
            x_t = self.sample_noise(observation.batch_size, generator=generator)
        else:
            x_t = x_t.to(device=self.device, dtype=torch.float32).clone()

        dt = -1.0 / num_steps
        timesteps = torch.linspace(
            1.0,
            1.0 / num_steps,
            num_steps,
            dtype=torch.float32,
            device=self.device,
        )
        for timestep_value in timesteps:
            timestep = timestep_value.expand(observation.batch_size)
            velocity = self.denoise_step(
                prefix_context,
                x_t,
                timestep,
                use_cuda_graph=use_cuda_graph,
            )
            x_t.add_(velocity, alpha=dt)
        if offload_action:
            self._move_action_modules_to_device(torch.device("cpu"))
            torch.cuda.empty_cache()
        return x_t

    def warmup_actions(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            self.config.action_horizon,
            self.config.action_dim,
            device=self.device,
            dtype=torch.float32,
        )


__all__ = [
    "Pi05ActionExpert",
    "Pi05CheckpointManifest",
    "Pi05PolicyModel",
]
