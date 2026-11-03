# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from torch import nn

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.models.vlas.dm05_core import DM05CoreModel
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.runtime.vla.cuda_graph import VLATensorGraphRunner
from sglang.multimodal_gen.runtime.vla.observation import (
    VLAObservationBatch,
    tensor_fingerprint,
)
from sglang.multimodal_gen.runtime.vla.prefix_cache import (
    PrefixContext,
    VLADensePrefixCache,
)

if TYPE_CHECKING:
    from sglang.multimodal_gen.configs.pipeline_configs.dm05 import DM05PipelineConfig


@dataclass(frozen=True)
class DM05CheckpointManifest:
    model_path: str
    safetensor_files: tuple[str, ...]
    loaded_keys: tuple[str, ...]
    loaded_parameter_names: tuple[str, ...]
    config_sha256: str


def _target_name(source_name: str) -> str:
    # Transformers 5.19 flattened SigLIP's former vision_model container.
    return source_name.removeprefix("model.").replace(
        "vlm.model.vision_tower.vision_model.", "vlm.model.vision_tower.", 1
    )


def _set_parameter(module: nn.Module, name: str, value: nn.Parameter) -> None:
    parent, _, leaf = name.rpartition(".")
    owner = module.get_submodule(parent) if parent else module
    owner._parameters[leaf] = value


def load_dm05_weights(
    model: DM05CoreModel,
    model_path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> DM05CheckpointManifest:
    root = Path(model_path)
    index_path = root / "model.safetensors.index.json"
    index_map = None
    if index_path.exists():
        index_map = json.loads(index_path.read_text())["weight_map"]
        filenames = sorted(set(index_map.values()))
        if any(Path(name).name != name for name in filenames):
            raise ValueError("DM05 checkpoint shard names must be local filenames")
        files = tuple(root / name for name in filenames)
    else:
        files = tuple(sorted(root.glob("*.safetensors")))
    if not files:
        raise ValueError(f"No DM05 safetensors found in {model_path}")

    parameters = dict(model.named_parameters(remove_duplicate=False))
    alias_groups: dict[int, list[str]] = {}
    for name, parameter in parameters.items():
        alias_groups.setdefault(id(parameter), []).append(name)
    source_files: dict[str, Path] = {}
    target_sources: set[str] = set()
    for filename in files:
        with safe_open(filename, framework="pt", device="cpu") as reader:
            for source in reader.keys():
                if source in source_files:
                    raise ValueError(f"Duplicate DM05 checkpoint key: {source}")
                if index_map is not None and index_map.get(source) != filename.name:
                    raise ValueError(f"DM05 shard index mismatch for {source}")
                target = _target_name(source)
                if target in target_sources:
                    raise ValueError(f"Duplicate DM05 target tensor: {target}")
                if target not in parameters:
                    raise ValueError(f"Unexpected DM05 checkpoint key: {source}")
                shape = tuple(reader.get_slice(source).get_shape())
                if shape != tuple(parameters[target].shape):
                    raise ValueError(
                        f"DM05 tensor shape mismatch for {source}: "
                        f"{shape} != {tuple(parameters[target].shape)}"
                    )
                source_files[source] = filename
                target_sources.add(target)
    if index_map is not None and set(index_map) != set(source_files):
        raise ValueError("DM05 shard index contains missing tensors")
    supplied = {_target_name(name) for name in source_files}
    missing = [
        names for names in alias_groups.values() if not supplied.intersection(names)
    ]
    if missing:
        raise ValueError(f"Missing DM05 checkpoint tensors: {missing}")

    loaded_groups: dict[int, torch.Tensor] = {}
    for filename in files:
        with safe_open(filename, framework="pt", device="cpu") as reader:
            for source in reader.keys():
                target = _target_name(source)
                tensor = reader.get_tensor(source)
                if not tensor.is_floating_point():
                    raise ValueError(f"DM05 parameter must be floating point: {source}")
                group = id(parameters[target])
                aliases = alias_groups[group]
                if group in loaded_groups:
                    if not torch.equal(tensor, loaded_groups[group]):
                        raise ValueError(
                            f"Conflicting tied DM05 checkpoint tensor: {source}"
                        )
                    continue
                parameter = nn.Parameter(
                    tensor.to(device=device, dtype=dtype), requires_grad=False
                )
                for alias in aliases:
                    _set_parameter(model, alias, parameter)
                # Retain a CPU view only for parameters with checkpoint aliases.
                if len(aliases) > 1:
                    loaded_groups[group] = tensor
    for name, buffer in list(model.named_buffers()):
        if buffer.is_meta:
            raise RuntimeError(f"DM05 buffer was not initialized: {name}")
        parent, _, leaf = name.rpartition(".")
        owner = model.get_submodule(parent) if parent else model
        owner._buffers[leaf] = buffer.to(device=device)
    if any(parameter.is_meta for parameter in model.parameters()):
        raise RuntimeError("DM05 checkpoint left unmaterialized parameters")
    return DM05CheckpointManifest(
        model_path=str(root),
        safetensor_files=tuple(str(path) for path in files),
        loaded_keys=tuple(sorted(source_files)),
        loaded_parameter_names=tuple(sorted(parameters)),
        config_sha256=hashlib.sha256((root / "config.json").read_bytes()).hexdigest(),
    )


class DM05PolicyModel(nn.Module):
    def __init__(
        self,
        config: DM05PipelineConfig,
        core_model: DM05CoreModel,
        *,
        device: torch.device | str,
        model_path: str = "",
        manifest: DM05CheckpointManifest | None = None,
    ):
        super().__init__()
        self.config = config
        self.core_model = core_model.requires_grad_(False).eval()
        self.device = torch.device(device)
        self.model_path = model_path
        self.manifest = manifest
        self._execution_lock = threading.RLock()
        self.action_graph_runner = VLATensorGraphRunner(
            "dm05-action-loop",
            enabled=config.enable_action_cuda_graph,
            max_entries=config.action_cuda_graph_max_entries,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: DM05PipelineConfig,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        revision: str | None = None,
    ) -> DM05PolicyModel:
        local_path = maybe_download_model(
            model_path,
            revision=revision,
            force_diffusers_model=False,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"],
        )
        payload = json.loads((Path(local_path) / "config.json").read_text())
        if payload.get("model_type") != "dm05":
            raise ValueError("Expected a DM05 checkpoint config with model_type='dm05'")
        config.apply_checkpoint_config(local_path, payload)
        dtype = dtype or cls._dtype_from_config(config.materialize_dtype)
        device = torch.device(device or get_local_torch_device())
        if config.precision_policy == "fp32_mixed" and dtype != torch.float32:
            raise ValueError("DM05 fp32_mixed requires FP32 model parameters")
        # Keep nonpersistent RoPE/embedding buffers initialized on CPU; only the
        # parameters are meta until they are loaded, one tensor at a time.
        with init_empty_weights(include_buffers=False):
            core = DM05CoreModel(
                payload,
                precision_policy=config.precision_policy,
                language_attention=config.llm_attn_implementation,
                vision_attention=config.vision_attn_implementation,
                action_attention=config.action_attn_implementation,
            )
        core.vlm.tie_weights()
        manifest = load_dm05_weights(core, local_path, device=device, dtype=dtype)
        return cls(
            config, core, device=device, model_path=local_path, manifest=manifest
        )

    @staticmethod
    def _dtype_from_config(value: str) -> torch.dtype:
        if value in ("bf16", "bfloat16"):
            return torch.bfloat16
        if value in ("fp32", "float32"):
            return torch.float32
        raise ValueError(f"Unsupported DM05 parameter dtype: {value}")

    @contextmanager
    def _precision_context(self):
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.config.precision_policy == "fp32_mixed",
        ):
            yield

    def build_prefix_cache_key(
        self, observation: VLAObservationBatch, *, bucket_prompt: bool = False
    ) -> str:
        del bucket_prompt
        metadata = observation.metadata["dm05"]
        fingerprints = {
            "tokens": tensor_fingerprint(observation.tokens),
            "token_masks": tensor_fingerprint(observation.token_masks),
        }
        for name in (
            "pixel_values",
            "token_type_ids",
            "history_pixel_values",
            "history_mask",
        ):
            value = metadata.get(name)
            fingerprints[name] = None if value is None else tensor_fingerprint(value)
        identity = {
            "model_path": self.model_path,
            "config": None if self.manifest is None else self.manifest.config_sha256,
            "precision": self.config.precision_policy,
            "dtype": str(self.core_model.action_in_proj.weight.dtype),
            "language_attention": self.config.llm_attn_implementation,
            "vision_attention": self.config.vision_attn_implementation,
            "camera_order": observation.metadata.get("camera_order", ()),
            "inputs": fingerprints,
        }
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    @torch.inference_mode()
    def encode_prefix(
        self,
        observation: VLAObservationBatch,
        *,
        use_cuda_graph: bool = True,
        bucket_prompt: bool | None = None,
    ) -> PrefixContext:
        del use_cuda_graph, bucket_prompt
        metadata = observation.metadata["dm05"]
        vision_dtype = next(self.core_model.vlm.model.vision_tower.parameters()).dtype

        def move(name: str, *, dtype: torch.dtype | None = None):
            value = metadata.get(name)
            return None if value is None else value.to(device=self.device, dtype=dtype)

        with self._execution_lock, self._precision_context():
            return self.core_model.encode_prefix(
                observation.tokens.to(self.device),
                observation.token_masks.to(self.device),
                pixel_values=move("pixel_values", dtype=vision_dtype),
                token_type_ids=move("token_type_ids"),
                history_pixel_values=move("history_pixel_values", dtype=vision_dtype),
                history_mask=move("history_mask"),
            )

    def sample_noise(
        self, batch_size: int = 1, *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        generator_device = self.device if generator is None else generator.device
        return torch.randn(
            batch_size,
            self.config.action_horizon,
            self.config.action_dim,
            dtype=self.core_model.action_in_proj.weight.dtype,
            device=generator_device,
            generator=generator,
        ).to(self.device)

    def _sample_loop(
        self,
        prefix: PrefixContext,
        noise: torch.Tensor,
        action_mask: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        x_t = noise
        time_value = 1.0
        dt = -1.0 / num_steps
        for _ in range(num_steps):
            time = torch.full(
                (noise.shape[0],),
                time_value,
                device=noise.device,
                dtype=noise.dtype,
            )
            x_t = x_t * action_mask
            x_t = x_t + self.core_model.velocity(prefix, x_t, time) * dt
            time_value += dt
        return x_t

    @torch.inference_mode()
    def sample_actions(
        self,
        observation: VLAObservationBatch,
        prefix_context: PrefixContext,
        *,
        noise: torch.Tensor | None = None,
        num_steps: int | None = None,
        use_cuda_graph: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        num_steps = (
            self.config.default_num_inference_steps if num_steps is None else num_steps
        )
        if (
            isinstance(num_steps, bool)
            or not isinstance(num_steps, int)
            or num_steps <= 0
        ):
            raise ValueError("DM05 num_steps must be a positive integer")
        dtype = self.core_model.action_in_proj.weight.dtype
        if noise is None:
            noise = self.sample_noise(observation.batch_size, generator=generator)
        else:
            noise = noise.to(device=self.device, dtype=dtype)
        expected = (
            observation.batch_size,
            self.config.action_horizon,
            self.config.action_dim,
        )
        if tuple(noise.shape) != expected:
            raise ValueError(
                f"DM05 noise must have shape {expected}, got {tuple(noise.shape)}"
            )
        action_mask = observation.metadata["dm05"]["action_mask"].to(
            device=self.device, dtype=dtype
        )
        if tuple(action_mask.shape) not in ((expected[0], 1, expected[2]), expected):
            raise ValueError(
                "DM05 action mask must broadcast across the action horizon"
            )
        with self._execution_lock, self._precision_context():
            if not use_cuda_graph or not self.config.enable_action_cuda_graph:
                return self._sample_loop(prefix_context, noise, action_mask, num_steps)
            layers = tuple(prefix_context.past_key_values)
            inputs = (
                noise,
                action_mask,
                prefix_context.prefix_pad_masks,
                *(tensor for layer in layers for tensor in layer[:2]),
            )

            def execute(*values):
                current_noise, mask, visible, *kv = values
                prefix = PrefixContext(
                    past_key_values=VLADensePrefixCache(
                        (kv[i], kv[i + 1], None) for i in range(0, len(kv), 2)
                    ),
                    prefix_pad_masks=visible,
                    prefix_len=prefix_context.prefix_len,
                )
                return (self._sample_loop(prefix, current_noise, mask, num_steps),)

            result = self.action_graph_runner.run(
                execute,
                inputs,
                key=(
                    num_steps,
                    self.config.precision_policy,
                    self.config.action_attn_implementation,
                ),
            )[0]
            return result.clone()

    def warmup_actions(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            self.config.action_horizon,
            self.config.action_dim,
            device=self.device,
            dtype=torch.float32,
        )

    def should_run_action_denoise(self, prefix_context: PrefixContext | None) -> bool:
        return True

    def action_parallel_info(
        self, prefix_context: PrefixContext | None
    ) -> dict[str, Any]:
        return {
            "split_group": False,
            "world_size": 1,
            "action_sequence_parallel": False,
            "prefix_root": 0,
            "action_root": 0,
        }

    def graph_cache_info(self) -> dict[str, Any]:
        return {"action": asdict(self.action_graph_runner.cache_info()), "prefix": None}
