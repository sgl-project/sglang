# SPDX-License-Identifier: Apache-2.0
"""SGLang VLA stages backed by the released Motubrain2 raw observation policy."""

from __future__ import annotations

import base64
import hashlib
import os
import threading
from collections import OrderedDict
from contextlib import nullcontext
from copy import copy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.sample.motubrain2 import Motubrain2Observation
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _tensor_fingerprint(tensor: torch.Tensor) -> str:
    """Content hash for prefix-cache keys; CPU-independent and exact."""

    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _native_attention(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import (
        torch_default_sdpa_backends,
    )

    # SGLang disables cuDNN SDPA globally. RawPolicy uses PyTorch's native
    # selection for both the frozen encoders and the joint denoiser.
    return torch_default_sdpa_backends()


class _PreparedProcessor:
    """Pass the exact prefix-cache result to RawPolicy.predict without re-encoding."""

    def __init__(self, processor, example):
        self.processor = processor
        self.example = example

    def encode(self, observation):
        return self.example

    def __getattr__(self, name):
        return getattr(self.processor, name)


class _CachedTextEncoder:
    """LRU cache for instruction embeddings shared by one policy."""

    def __init__(self, inner: Any, max_entries: int = 64):
        if type(max_entries) is not int or max_entries < 0:
            raise ValueError("text_cache_size must be a nonnegative integer")
        self.inner = inner
        self.max_entries = max_entries
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def encode(self, prompt: str) -> dict[str, Any]:
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        if self.max_entries == 0:
            return self.inner.encode(prompt)
        with self._lock:
            cached = self._cache.get(prompt)
            if cached is not None:
                self._cache.move_to_end(prompt)
        if cached is None:
            cached = self.inner.encode(prompt)
            with self._lock:
                self._cache[prompt] = cached
                while len(self._cache) > self.max_entries:
                    self._cache.popitem(last=False)
        return {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in cached.items()
        }


def _validate_rtc_config(config: Any, action_steps: int) -> dict[str, Any] | None:
    if config is None:
        return None
    if not isinstance(config, dict):
        raise TypeError("rtc must be a mapping")
    if not bool(config.get("enabled", False)):
        return None
    execute_prefix = config.get("execute_prefix", 1)
    if type(execute_prefix) is not int or execute_prefix < 0:
        raise ValueError("rtc.execute_prefix must be a nonnegative integer")
    if execute_prefix >= action_steps:
        raise ValueError("rtc.execute_prefix must be smaller than the action horizon")
    delay = config.get("delay", action_steps - execute_prefix)
    if type(delay) is not int or delay < 1:
        raise ValueError("rtc.delay must be a positive integer")
    return {
        "enabled": True,
        "execute_prefix": execute_prefix,
        "delay": int(delay),
    }


def _build_rtc(last_action: torch.Tensor, config: dict[str, Any]):
    from motubrain2.inference.rtc import TrainRTC

    start = config["execute_prefix"]
    future = last_action[start : start + config["delay"]]
    return TrainRTC(anchor=None, future=future, delay=len(future))


@dataclass(frozen=True)
class _SessionState:
    action: torch.Tensor | None = None
    rng_state: torch.Tensor | None = None


def _check_native_dit_cache(args: SimpleNamespace) -> None:
    try:
        from motubrain2.inference.dit_cache import DiTCacheConfig
    except ImportError as exc:
        if exc.name != "motubrain2.inference.dit_cache":
            raise
    else:
        if hasattr(DiTCacheConfig, "for_recipe"):
            return

    # Older policies may enable caching solely from the threshold.
    if args.dit_cache_enabled is False:
        args.dit_cache_threshold = args.dit_cache_threshold_2 = None
        return

    from motubrain2.inference.sampling import sampling_options

    if (
        args.dit_cache_enabled is not None
        or args.dit_cache_threshold is not None
        or args.dit_cache_threshold_2 is not None
        or sampling_options(args.model)
        == dict(sampler="euler", steps=30, video_shift=6.0, action_shift=1.0)
    ):
        raise ImportError(
            "This cache configuration requires a Motubrain2 installation with "
            "DiTCacheConfig.for_recipe. Upgrade Motubrain2 or explicitly disable "
            "the cache with dit_cache_enabled=false."
        )


class Motubrain2PolicyModel(nn.Module):
    """SGLang worker model backed by the native Motubrain policy."""

    runtime_role = "monolithic"

    def __init__(self, model_path: str, config: Any, policy: Any):
        super().__init__()
        self.model_path = str(model_path)
        self.config = config
        self.policy = policy
        # Text caching is local to this adapter. Keep the supplied RawPolicy intact.
        self.processor = copy(policy.processor)

        self._device = torch.device(policy.device)
        model = getattr(policy, "model", None)
        if model is not None:
            # Register the denoiser so framework tooling sees its parameters.
            self.model = model
            try:
                parameter = next(model.parameters())
                self._dtype = parameter.dtype
            except StopIteration:  # pragma: no cover - defensive
                self._dtype = torch.float32
        else:  # pragma: no cover - native policy always has a model
            self._dtype = torch.float32

        self.action_horizon = int(self.processor.action_steps)
        self.canonical_action_dim = int(self.processor.config.action_dim)
        self.wire_dim = int(self.processor.actions.wire_dim)
        self.camera_keys = list(config.image_keys)

        cache_size = int(getattr(config, "text_cache_size", 0) or 0)
        if cache_size:
            self.processor.text = _CachedTextEncoder(self.processor.text, cache_size)

        rtc_config = getattr(config, "rtc_config", None)
        self.rtc_config = _validate_rtc_config(
            rtc_config() if callable(rtc_config) else rtc_config,
            self.action_horizon,
        )
        self.max_sessions = config.max_sessions
        if type(self.max_sessions) is not int or self.max_sessions < 1:
            raise ValueError("max_sessions must be a positive integer")
        self.sessions: OrderedDict[str, _SessionState] = OrderedDict()
        self._lock = threading.Lock()

        if self.processor.sparse_history is not None:
            logger.warning(
                "Motubrain2 sparse history is enabled; each observation must "
                "carry 'history_cameras' with %d RGB frames per camera",
                self.processor.sparse_history.rgb_frames,
            )
        logger.info(
            "Motubrain2PolicyModel ready: device=%s dtype=%s cameras=%s horizon=%d "
            "canonical_dim=%d wire_dim=%d sampler=%s steps=%s rtc=%s",
            self._device,
            self._dtype,
            self.camera_keys,
            self.action_horizon,
            self.canonical_action_dim,
            self.wire_dim,
            policy.sampling.get("sampler"),
            policy.sampling.get("steps"),
            self.rtc_config is not None,
        )

    def __repr__(self) -> str:
        return (
            f"Motubrain2PolicyModel(model_path={self.model_path!r}, "
            f"cameras={self.camera_keys}, horizon={self.action_horizon}, "
            f"wire_dim={self.wire_dim})"
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    # ------------------------------------------------------------- loading
    @classmethod
    def from_pretrained(cls, model_path: str, config: Any) -> Motubrain2PolicyModel:
        config.load_serving_profile(model_path)
        from motubrain2.inference.policy import RawPolicy

        if bool(getattr(config, "deterministic", False)):
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
        bundle = str(config.bundle_path(model_path))
        profile = str(
            getattr(config, "profile", None) or os.path.join(model_path, "profile.json")
        )
        if not os.path.isdir(bundle):
            raise ValueError(f"Motubrain2 bundle directory not found: {bundle}")
        for name in ("config.json", "model.safetensors"):
            if not os.path.isfile(os.path.join(bundle, name)):
                raise ValueError(f"Motubrain2 bundle is missing {name}: {bundle}")
        if not os.path.isfile(profile):
            raise ValueError(f"Motubrain2 robot profile not found: {profile}")

        device = _resolve_device()
        dtype = _materialize_dtype(config.materialize_dtype)
        bf16 = dtype == torch.bfloat16
        if config.video_mxfp8 and not bf16:
            raise ValueError("video_mxfp8 requires materialize_dtype=bf16")
        args = SimpleNamespace(
            model=bundle,
            profile=profile,
            wan=config.wan,
            wan_subfolder=getattr(config, "wan_subfolder", None),
            wan_config=getattr(config, "wan_config", None),
            text_encoder=config.text_encoder,
            da3=getattr(config, "da3", None),
            da3_config=getattr(config, "da3_config", None),
            action_stats=getattr(config, "action_stats", None),
            encoder_device=getattr(config, "encoder_device", None),
            device=str(device),
            bf16=bf16,
            seed=int(getattr(config, "inference_seed", 0)),
            compile=bool(getattr(config, "compile", False)),
            multistream=str(getattr(config, "multistream", "off")),
            async_io=False,
            video_mxfp8=bool(getattr(config, "video_mxfp8", False)),
            video_mxfp8_overlay=getattr(config, "video_mxfp8_overlay", None),
            sampler=config.native_sampler_override(),
            steps=getattr(config, "steps_override", None),
            video_shift=getattr(config, "video_shift_override", None),
            action_shift=getattr(config, "action_shift_override", None),
            dit_cache_enabled=config.dit_cache_enabled,
            dit_cache_threshold=config.dit_cache_threshold,
            dit_cache_threshold_2=config.dit_cache_threshold_2,
            dit_cache_skip_steps=config.dit_cache_skip_steps,
            dit_cache_skip_steps_2=config.dit_cache_skip_steps_2,
            dit_cache_damping=config.dit_cache_damping,
        )
        _check_native_dit_cache(args)
        logger.info(
            "Loading Motubrain2 policy: bundle=%s profile=%s device=%s bf16=%s",
            bundle,
            profile,
            device,
            bf16,
        )
        policy = RawPolicy.from_options(args)
        # RawPolicy retains the checkpoint dtype when bf16 is disabled.
        if dtype == torch.float32 and any(
            p.is_floating_point() and p.dtype != dtype
            for p in policy.model.parameters()
        ):
            policy.model.float()
            if policy.streams is not None:
                from motubrain2.inference.streams import InferenceStreams

                # The native stream scheduler registers each parameter's storage.
                with torch.inference_mode():
                    policy.streams = InferenceStreams(
                        policy.model, policy.device, args.multistream
                    )
        config.validate_loaded_profile(policy.processor, policy.sampling)
        return cls(model_path, config, policy)

    @torch.inference_mode()
    def encode_observation(self, observation: Motubrain2Observation):
        with _native_attention(self.device):
            return self.processor.encode(observation.inputs)

    @staticmethod
    def observation_key(observation: Motubrain2Observation) -> tuple:
        inputs = observation.inputs
        return (
            inputs["instruction"],
            tuple(
                (name, _tensor_fingerprint(clip))
                for name, clip in inputs["cameras"].items()
            ),
            tuple(
                (name, _tensor_fingerprint(clip))
                for name, clip in inputs.get("history_cameras", {}).items()
            ),
            tuple(
                (name, _tensor_fingerprint(inputs[name]))
                for name in ("state", "camera_extrinsic")
                if name in inputs
            ),
        )

    def _resolve_generator(
        self,
        generator: Any,
        seed: int | None,
    ) -> torch.Generator:
        if isinstance(generator, (list, tuple)):
            generator = generator[0] if generator else None
        if isinstance(generator, torch.Generator):
            if torch.device(generator.device) == self.device:
                return generator
            moved = torch.Generator(device=self.device)
            moved.manual_seed(int(generator.initial_seed()))
            return moved
        return torch.Generator(device=self.device).manual_seed(
            self.config.inference_seed if seed is None else int(seed)
        )

    @torch.inference_mode()
    def predict(
        self,
        observation: Motubrain2Observation,
        example: dict[str, Any],
        *,
        num_steps: int,
        generator: Any = None,
    ) -> tuple[torch.Tensor, dict | None]:
        session_id = observation.session_id
        if self.rtc_config is not None and (
            not isinstance(session_id, str) or not session_id.strip()
        ):
            raise ValueError("RTC requires a nonempty string session_id")
        seed = observation.seed

        with (
            self._lock,
            _native_attention(self.device),
            torch.compiler.set_stance("force_eager")
            if self.config.enforce_eager
            else nullcontext(),
        ):
            previous = (
                self.sessions.get(session_id, _SessionState())
                if not observation.reset
                else _SessionState()
            )
            gen = self._resolve_generator(None if seed is None else generator, seed)
            if seed is None and previous.rng_state is not None:
                gen.set_state(previous.rng_state)
            rtc = observation.rtc
            if (
                rtc is None
                and self.rtc_config is not None
                and previous.action is not None
            ):
                rtc = _build_rtc(previous.action, self.rtc_config)
            request_policy = copy(self.policy)
            request_policy.generator = gen
            request_policy.sampling = {**self.policy.sampling, "steps": int(num_steps)}
            request_policy.processor = _PreparedProcessor(self.processor, example)
            result = request_policy.predict(observation.inputs, rtc=rtc)
            decoded, valid = result["decoded_action"], result["decoded_valid"]
            return_native = observation.return_native_outputs
            native_outputs = None
            if valid.dtype != torch.bool or valid.shape != decoded.shape:
                raise ValueError("native validity mask does not match decoded actions")
            fully_valid = bool(valid.all())
            if not return_native and not fully_valid:
                raise ValueError(
                    "policy produced invalid controller action coordinates"
                )
            actions = decoded.detach().float()
            if tuple(actions.shape) != (self.action_horizon, self.wire_dim):
                raise ValueError(
                    f"policy produced actions shaped {tuple(actions.shape)}, expected "
                    f"{(self.action_horizon, self.wire_dim)}"
                )
            if not torch.isfinite(actions).all():
                raise ValueError("policy produced non-finite actions")
            if return_native:
                from safetensors.torch import save

                tensors = {
                    name: result[name].contiguous()
                    for name in ("video", "action", "decoded_action", "decoded_valid")
                }
                native_outputs = {
                    "format": "safetensors",
                    "encoding": "base64",
                    "data": base64.b64encode(save(tensors)).decode("ascii"),
                }
            if session_id is not None:
                # Publish both pieces of state only after successful validation and
                # serialization. A failed draw or reset leaves the episode retryable.
                rng_state = gen.get_state() if seed is None else previous.rng_state
                prefix = (
                    actions.cpu().clone() if self.rtc_config and fully_valid else None
                )
                if prefix is None and rng_state is None:
                    self.sessions.pop(session_id, None)
                else:
                    self.sessions[session_id] = _SessionState(prefix, rng_state)
                    self.sessions.move_to_end(session_id)
                    if len(self.sessions) > self.max_sessions:
                        self.sessions.popitem(last=False)
        return actions, native_outputs

    def release_session(self, session_id: str) -> bool:
        with self._lock:
            return self.sessions.pop(session_id, None) is not None


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _materialize_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if dtype == torch.bfloat16 or str(dtype).lower() in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype == torch.float32 or str(dtype).lower() in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported Motubrain2 materialize_dtype: {dtype!r}")
