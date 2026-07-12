# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.vla.observation import (
    collate_vla_observation_batches,
)
from sglang.multimodal_gen.runtime.vla.parallel import (
    broadcast_prefix_context,
    broadcast_tensor_from_rank,
    get_vla_split_group,
)
from sglang.multimodal_gen.runtime.vla.prefix_cache import (
    PrefixContext,
    VLAPrefixCacheManager,
    slice_prefix_context,
)


def vla_state(batch: Req) -> dict[str, Any]:
    """Per-request scratchpad shared by the VLA pipeline stages."""

    return batch.extra["vla"]


def vla_timings(batch: Req) -> dict[str, float]:
    return vla_state(batch).setdefault("timings", {})


def vla_options(batch: Req) -> dict[str, Any]:
    return vla_state(batch).get("options") or {}


def materialize_vla_action_batch(
    actions: Any,
    action_dim: int,
    output_format: str,
) -> Any:
    output_format = output_format.lower()
    if isinstance(actions, torch.Tensor):
        actions_out = actions[..., :action_dim].detach().float().cpu().numpy()
        if output_format != "numpy":
            actions_out = actions_out.tolist()
    elif isinstance(actions, np.ndarray):
        actions_out = actions[..., :action_dim].astype(np.float32, copy=False)
        if output_format != "numpy":
            actions_out = actions_out.tolist()
    else:
        actions_out = actions
    if not isinstance(actions_out, list):
        return actions_out
    if not actions_out:
        return []
    first = actions_out[0]
    if isinstance(first, list) and first and isinstance(first[0], list):
        actions_out = [[step[:action_dim] for step in sample] for sample in actions_out]
    else:
        actions_out = [[step[:action_dim] for step in actions_out]]
    if output_format == "numpy":
        return np.asarray(actions_out, dtype=np.float32)
    return actions_out


def synchronize_vla_action_tensor(actions: torch.Tensor | None) -> None:
    if actions is not None and actions.device.type == "cuda":
        torch.cuda.synchronize(actions.device)


def _effective_prefix_cache_enabled(
    batch: Req,
    server_args: ServerArgs,
) -> bool:
    options = vla_options(batch)
    return bool(options.get("enable_prefix_cache", True)) and bool(
        server_args.pipeline_config.enable_global_prefix_cache
    )


def _grouped_fingerprint(
    batch: Req,
    server_args: ServerArgs,
) -> tuple[Any, ...]:
    if (
        batch.is_warmup
        or get_vla_split_group() is not None
        or _effective_prefix_cache_enabled(batch, server_args)
        or batch.generator is not None
    ):
        return ("single", id(batch))

    observation = vla_state(batch).get("observation_batch")
    camera_order = tuple(observation.metadata.get("camera_order", ()))
    image_shapes = tuple(
        (
            name,
            tuple(observation.images[name].shape),
            bool(observation.image_masks[name].item()),
        )
        for name in camera_order
    )
    return (
        "grouped",
        camera_order,
        image_shapes,
        None if observation.state is None else tuple(observation.state.shape),
        None if observation.noise is None else tuple(observation.noise.shape),
        tuple(observation.tokens.shape),
        tuple(observation.token_masks.shape),
        batch.action_horizon,
        batch.action_dim,
        batch.num_inference_steps,
    )


class VLAObservationPreprocessStage(PipelineStage):
    def __init__(self, preprocessor: Any):
        super().__init__()
        self.preprocessor = preprocessor

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.ENCODER

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        start = time.perf_counter()
        state = vla_state(batch)
        raw_observation = dict(state.get("observation") or {})
        raw_observation.setdefault("prompt", batch.prompt)
        observation = self.preprocessor(raw_observation)
        state["observation_batch"] = observation
        vla_timings(batch)["preprocess_ms"] = (time.perf_counter() - start) * 1000
        return batch


class VLAPrefixEncodingStage(PipelineStage):
    def __init__(
        self,
        policy_model: Any,
        prefix_cache: VLAPrefixCacheManager,
    ):
        super().__init__()
        self.policy_model = policy_model
        self.prefix_cache = prefix_cache

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.ENCODER

    def run_grouped_requests(
        self,
        batches: list[Req],
        server_args: ServerArgs,
    ) -> list[Req]:
        results: list[Req | None] = [None] * len(batches)
        for fingerprint, group in self._group_requests_by_fingerprint(
            batches,
            lambda batch: _grouped_fingerprint(batch, server_args),
        ):
            group_batches = [batch for _, batch in group]
            if len(group_batches) == 1 or fingerprint[0] == "single":
                for index, batch in group:
                    results[index] = self(batch, server_args)
                continue

            prefix_start = time.perf_counter()
            observations = [
                vla_state(batch)["observation_batch"] for batch in group_batches
            ]
            grouped_observation = collate_vla_observation_batches(observations)
            prefix_context = self.policy_model.encode_prefix(grouped_observation)
            prefix_ms = (time.perf_counter() - prefix_start) * 1000

            for offset, (index, batch) in enumerate(group):
                state = vla_state(batch)
                state["observation_group"] = grouped_observation
                state["prefix_context_group"] = prefix_context
                state["prefix_context"] = slice_prefix_context(
                    prefix_context,
                    offset,
                )
                state["cache"] = {
                    "hit": False,
                    "scope": "request",
                    "prefix_len": prefix_context.prefix_len,
                    "grouped": True,
                    "batch_size": len(group_batches),
                }
                timings = vla_timings(batch)
                timings["cache_lookup_ms"] = 0.0
                timings["prefix_ms"] = prefix_ms
                results[index] = batch

            if (
                server_args.pipeline_config.empty_cache_after_prefix
                and torch.cuda.is_available()
            ):
                torch.cuda.empty_cache()

        return [result for result in results if result is not None]

    def _recv_prefix_result(self, batch: Req, split: Any) -> Req:
        state = vla_state(batch)
        state["prefix_context"] = broadcast_prefix_context(
            None,
            split,
            src=split.prefix_root,
        )
        state["cache"] = split.broadcast_object_from_rank(
            None,
            src=split.prefix_root,
        )
        timings = split.broadcast_object_from_rank(None, src=split.prefix_root)
        vla_timings(batch).update(timings)
        return batch

    def _send_prefix_result(
        self,
        batch: Req,
        split: Any,
        prefix_context: Any,
    ) -> None:
        broadcast_prefix_context(
            prefix_context,
            split,
            src=split.prefix_root,
        )
        split.broadcast_object_from_rank(
            vla_state(batch)["cache"],
            src=split.prefix_root,
        )
        split.broadcast_object_from_rank(
            vla_timings(batch),
            src=split.prefix_root,
        )

    def get_cached_context(
        self, batch: Req, server_args: ServerArgs, observation: Any
    ) -> tuple[str, PrefixContext]:
        """try querying the cache for PrefixContext with prefix cache key built from observations and other keys"""
        cache_enabled = _effective_prefix_cache_enabled(batch, server_args)
        if cache_enabled:
            cache_key = self.policy_model.build_prefix_cache_key(observation)
            cached_context = self.prefix_cache.get(cache_key)
        else:
            cache_key = None
            cached_context = None
        return cache_key, cached_context

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        state = vla_state(batch)
        if batch.is_warmup:
            state["prefix_context"] = None
            state["cache"] = {"hit": False, "warmup": True}
            return batch

        split = get_vla_split_group()
        if split is not None and not split.is_prefix_rank:
            return self._recv_prefix_result(batch, split)

        observation = state["observation_batch"]
        cache_start = time.perf_counter()
        cache_enabled = _effective_prefix_cache_enabled(batch, server_args)

        # 1. try querying the per-request LRU prefix kv cache
        cache_key, cached_context = self.get_cached_context(
            batch, server_args, observation
        )

        vla_timings(batch)["cache_lookup_ms"] = (
            time.perf_counter() - cache_start
        ) * 1000

        # 2. prepare VLAState
        if cached_context is not None:
            state["prefix_context"] = cached_context
            state["cache"] = {
                "hit": True,
                "scope": "global",
                "mode": "exact",
                "prefix_len": cached_context.prefix_len,
            }
            if split is not None:
                self._send_prefix_result(batch, split, cached_context)
            return batch

        prefix_start = time.perf_counter()

        # 3. run encoding
        prefix_context = self.policy_model.encode_prefix(observation)
        if cache_key is not None:
            prefix_context.cache_key_digest = cache_key

        vla_timings(batch)["prefix_ms"] = (time.perf_counter() - prefix_start) * 1000
        state["prefix_context"] = prefix_context
        state["cache"] = {
            "hit": False,
            "scope": "global" if cache_enabled else "request",
            "mode": "exact" if cache_enabled else "disabled",
            "prefix_len": prefix_context.prefix_len,
        }

        # 4. update prefix kv cache
        if cache_key is not None:
            self.prefix_cache.put(cache_key, prefix_context)
        if split is not None:
            self._send_prefix_result(batch, split, prefix_context)
        if (
            server_args.pipeline_config.empty_cache_after_prefix
            and torch.cuda.is_available()
        ):
            torch.cuda.empty_cache()
        return batch


class VLAActionDenoisingStage(PipelineStage):
    def __init__(self, policy_model: Any):
        super().__init__()
        self.policy_model = policy_model

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    def run_grouped_requests(
        self,
        batches: list[Req],
        server_args: ServerArgs,
    ) -> list[Req]:
        results: list[Req | None] = [None] * len(batches)

        def action_fingerprint(batch: Req) -> tuple[Any, ...]:
            prefix_context = vla_state(batch).get("prefix_context_group")
            if prefix_context is None:
                return ("single", id(batch))
            return (
                "grouped",
                id(prefix_context),
                batch.num_inference_steps,
                str(vla_options(batch).get("output_format") or "list"),
            )

        for _, group in self._group_requests_by_fingerprint(
            batches,
            action_fingerprint,
        ):
            group_batches = [batch for _, batch in group]
            prefix_context = vla_state(group_batches[0]).get("prefix_context_group")
            if len(group_batches) == 1 or prefix_context is None:
                for index, batch in group:
                    results[index] = self(batch, server_args)
                continue

            start = time.perf_counter()
            options = vla_options(group_batches[0])
            observation = vla_state(group_batches[0])["observation_group"]
            actions = self.policy_model.sample_actions(
                observation,
                prefix_context,
                noise=observation.noise,
                num_steps=group_batches[0].num_inference_steps,
                use_cuda_graph=bool(options.get("enable_cuda_graph", True)),
                generator=None,
            )
            synchronize_vla_action_tensor(actions)
            actions_out = materialize_vla_action_batch(
                actions,
                server_args.pipeline_config.output_action_dim,
                str(options.get("output_format") or "list"),
            )
            action_ms = (time.perf_counter() - start) * 1000
            parallel_info = self.policy_model.action_parallel_info(prefix_context)

            for offset, (index, batch) in enumerate(group):
                state = vla_state(batch)
                vla_timings(batch)["action_denoise_ms"] = action_ms
                state["parallel"] = parallel_info
                state["actions_output"] = actions_out[offset]
                results[index] = batch

        return [result for result in results if result is not None]

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        start = time.perf_counter()
        state = vla_state(batch)
        observation = state.get("observation_batch")
        split = get_vla_split_group()
        prefix_context = state.get("prefix_context")
        should_run_action = (
            split is None or self.policy_model.should_run_action_denoise(prefix_context)
        )
        parallel_info = self.policy_model.action_parallel_info(prefix_context)
        if batch.is_warmup:
            actions = (
                self.policy_model.warmup_actions(batch_size=1)
                if should_run_action
                else None
            )
        elif should_run_action:
            # broadcast PrefixContext from action root rank to action ranks
            options = vla_options(batch)
            noise = observation.noise if observation is not None else None
            actions = self.policy_model.sample_actions(
                observation,
                prefix_context,
                noise=noise,
                num_steps=batch.num_inference_steps,
                use_cuda_graph=bool(options.get("enable_cuda_graph", True)),
                generator=batch.generator,
            )
            synchronize_vla_action_tensor(actions)
        else:
            actions = None

        if split is not None:
            if should_run_action:
                vla_timings(batch)["action_denoise_ms"] = (
                    time.perf_counter() - start
                ) * 1000
            actions = broadcast_tensor_from_rank(
                actions,
                split,
                src=split.action_root,
                device=self.policy_model.device,
            )
            timings = split.broadcast_object_from_rank(
                vla_timings(batch) if should_run_action else None,
                src=split.action_root,
            )
            vla_timings(batch).update(timings)
            parallel_info = split.broadcast_object_from_rank(
                parallel_info if should_run_action else None,
                src=split.action_root,
            )
        else:
            vla_timings(batch)["action_denoise_ms"] = (
                time.perf_counter() - start
            ) * 1000
        state["parallel"] = parallel_info
        state["actions"] = actions
        return batch


class VLAActionPostprocessStage(PipelineStage):
    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        start = time.perf_counter()
        state = vla_state(batch)
        action_dim = server_args.pipeline_config.output_action_dim
        options = vla_options(batch)
        actions_out = state.get("actions_output")
        if actions_out is None:
            action_batch = materialize_vla_action_batch(
                state["actions"],
                action_dim,
                str(options.get("output_format") or "list"),
            )
            actions_out = (
                action_batch[0] if isinstance(action_batch, list) else action_batch
            )
            if isinstance(action_batch, np.ndarray):
                actions_out = action_batch[0]

        payload = {
            "request_id": batch.request_id,
            "actions": actions_out,
        }
        payload["parameters"] = {"num_inference_steps": batch.num_inference_steps}
        if options.get("return_timing", True):
            timings = dict(vla_timings(batch))
            timings["postprocess_ms"] = (time.perf_counter() - start) * 1000
            payload["timings"] = timings
        if not batch.is_warmup:
            payload["cache"] = state.get("cache", {})
        if state.get("parallel") is not None:
            payload["parallel"] = state["parallel"]

        return OutputBatch(
            output=[payload],
            metrics=batch.metrics,
        )
