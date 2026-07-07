# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from sglang.multimodal_gen.runtime.cache.vla_prefix_cache import (
    VLAPrefixCacheManager,
    slice_prefix_context,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed.vla import (
    broadcast_metadata,
    broadcast_optional_tensor,
    broadcast_prefix_context,
    broadcast_timing,
    get_vla_split_group,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.vla_observation import (
    collate_vla_observation_batches,
)


@dataclass(frozen=True)
class VLAStageKeys:
    raw_observation: str = "vla_observation"
    observation_batch: str = "vla_observation_batch"
    observation_group: str = "vla_observation_group"
    options: str = "vla_options"
    timings: str = "vla_timings"
    prefix_context: str = "vla_prefix_context"
    prefix_context_group: str = "vla_prefix_context_group"
    prefix_cache_key: str = "vla_prefix_cache_key"
    cache: str = "vla_cache"
    group_index: str = "vla_group_index"
    group_size: str = "vla_group_size"
    actions: str = "vla_actions"
    actions_output: str = "vla_actions_output"

    @classmethod
    def for_namespace(cls, namespace: str) -> VLAStageKeys:
        return cls(
            raw_observation=f"{namespace}_observation",
            observation_batch=f"{namespace}_observation_batch",
            observation_group=f"{namespace}_observation_group",
            options=f"{namespace}_options",
            timings=f"{namespace}_timings",
            prefix_context=f"{namespace}_prefix_context",
            prefix_context_group=f"{namespace}_prefix_context_group",
            prefix_cache_key=f"{namespace}_prefix_cache_key",
            cache=f"{namespace}_cache",
            group_index=f"{namespace}_group_index",
            group_size=f"{namespace}_group_size",
            actions=f"{namespace}_actions",
            actions_output=f"{namespace}_actions_output",
        )


def vla_timings(batch: Req, keys: VLAStageKeys) -> dict[str, float]:
    return batch.extra.setdefault(keys.timings, {})


def vla_options(batch: Req, keys: VLAStageKeys) -> dict[str, Any]:
    return batch.extra.get(keys.options) or {}


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
    keys: VLAStageKeys,
) -> bool:
    options = vla_options(batch, keys)
    return bool(options.get("enable_prefix_cache", True)) and bool(
        server_args.pipeline_config.enable_global_prefix_cache
    )


def _grouped_fingerprint(
    batch: Req,
    server_args: ServerArgs,
    keys: VLAStageKeys,
) -> tuple[Any, ...]:
    if (
        batch.is_warmup
        or get_vla_split_group() is not None
        or _effective_prefix_cache_enabled(batch, server_args, keys)
        or batch.generator is not None
    ):
        return ("single", id(batch))

    observation = batch.extra.get(keys.observation_batch)
    options = vla_options(batch, keys)
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
        int(options.get("action_horizon") or batch.action_horizon),
        int(options.get("action_dim") or batch.action_dim),
        int(options.get("num_inference_steps") or batch.num_inference_steps),
    )


class VLAObservationPreprocessStage(PipelineStage):
    def __init__(
        self,
        preprocessor: Any,
        *,
        keys: VLAStageKeys = VLAStageKeys(),
    ):
        super().__init__()
        self.preprocessor = preprocessor
        self.keys = keys

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.ENCODER

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        start = time.perf_counter()
        raw_observation = dict(batch.extra.get(self.keys.raw_observation) or {})
        raw_observation.setdefault("prompt", batch.prompt)
        observation = self.preprocessor(raw_observation)
        batch.extra[self.keys.observation_batch] = observation
        vla_timings(batch, self.keys)["preprocess_ms"] = (
            time.perf_counter() - start
        ) * 1000
        return batch


class VLAPrefixEncodingStage(PipelineStage):
    def __init__(
        self,
        policy_model: Any,
        prefix_cache: VLAPrefixCacheManager,
        *,
        keys: VLAStageKeys = VLAStageKeys(),
    ):
        super().__init__()
        self.policy_model = policy_model
        self.prefix_cache = prefix_cache
        self.keys = keys

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
            lambda batch: _grouped_fingerprint(batch, server_args, self.keys),
        ):
            group_batches = [batch for _, batch in group]
            if len(group_batches) == 1 or fingerprint[0] == "single":
                for index, batch in group:
                    results[index] = self(batch, server_args)
                continue

            prefix_start = time.perf_counter()
            observations = [
                batch.extra[self.keys.observation_batch] for batch in group_batches
            ]
            grouped_observation = collate_vla_observation_batches(observations)
            prefix_context = self.policy_model.encode_prefix(grouped_observation)
            prefix_ms = (time.perf_counter() - prefix_start) * 1000

            for offset, (index, batch) in enumerate(group):
                batch.extra[self.keys.observation_group] = grouped_observation
                batch.extra[self.keys.prefix_context_group] = prefix_context
                batch.extra[self.keys.prefix_context] = slice_prefix_context(
                    prefix_context,
                    offset,
                )
                batch.extra[self.keys.group_index] = offset
                batch.extra[self.keys.group_size] = len(group_batches)
                batch.extra[self.keys.cache] = {
                    "hit": False,
                    "match_len": 0,
                    "full_prefix_len": prefix_context.prefix_len,
                    "grouped": True,
                    "batch_size": len(group_batches),
                }
                timings = vla_timings(batch, self.keys)
                timings["cache_lookup_ms"] = 0.0
                timings["prefix_ms"] = prefix_ms
                results[index] = batch

            if (
                server_args.pipeline_config.empty_cache_after_prefix
                and torch.cuda.is_available()
            ):
                torch.cuda.empty_cache()

        return [result for result in results if result is not None]

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.is_warmup:
            batch.extra[self.keys.prefix_context] = None
            batch.extra[self.keys.cache] = {"hit": False, "warmup": True}
            return batch

        split = get_vla_split_group()
        if split is not None and not split.is_prefix_rank:
            prefix_context = broadcast_prefix_context(
                None,
                split,
                src=split.prefix_root,
                device=self.policy_model.device,
            )
            batch.extra[self.keys.prefix_context] = prefix_context
            batch.extra[self.keys.cache] = broadcast_metadata(
                None,
                split,
                src=split.prefix_root,
            )
            timings = broadcast_timing(None, split, src=split.prefix_root)
            vla_timings(batch, self.keys).update(timings)
            return batch

        observation = batch.extra[self.keys.observation_batch]
        options = vla_options(batch, self.keys)
        cache_key = self.policy_model.build_prefix_cache_key(observation, options)
        batch.extra[self.keys.prefix_cache_key] = cache_key

        cache_start = time.perf_counter()
        cache_enabled = bool(options.get("enable_prefix_cache", True))
        lookup = (
            self.prefix_cache.get(cache_key)
            if cache_enabled and server_args.pipeline_config.enable_global_prefix_cache
            else None
        )
        vla_timings(batch, self.keys)["cache_lookup_ms"] = (
            time.perf_counter() - cache_start
        ) * 1000

        if lookup is not None and lookup.hit:
            batch.extra[self.keys.prefix_context] = lookup.context
            batch.extra[self.keys.cache] = {
                "hit": True,
                "match_len": lookup.match_len,
                "full_prefix_len": lookup.full_prefix_len,
            }
            if split is not None:
                broadcast_prefix_context(
                    lookup.context,
                    split,
                    src=split.prefix_root,
                    device=self.policy_model.device,
                )
                broadcast_metadata(
                    batch.extra[self.keys.cache],
                    split,
                    src=split.prefix_root,
                )
                broadcast_timing(
                    vla_timings(batch, self.keys),
                    split,
                    src=split.prefix_root,
                )
            return batch

        prefix_start = time.perf_counter()
        prefix_context = self.policy_model.encode_prefix(observation)
        vla_timings(batch, self.keys)["prefix_ms"] = (
            time.perf_counter() - prefix_start
        ) * 1000
        batch.extra[self.keys.prefix_context] = prefix_context
        batch.extra[self.keys.cache] = {
            "hit": False,
            "match_len": 0 if lookup is None else lookup.match_len,
            "full_prefix_len": cache_key.full_prefix_len,
            "partial_rejected": False if lookup is None else lookup.partial_rejected,
        }

        if cache_enabled and server_args.pipeline_config.enable_global_prefix_cache:
            self.prefix_cache.put(cache_key, prefix_context)
        if split is not None:
            broadcast_prefix_context(
                prefix_context,
                split,
                src=split.prefix_root,
                device=self.policy_model.device,
            )
            broadcast_metadata(
                batch.extra[self.keys.cache],
                split,
                src=split.prefix_root,
            )
            broadcast_timing(
                vla_timings(batch, self.keys),
                split,
                src=split.prefix_root,
            )
        if (
            server_args.pipeline_config.empty_cache_after_prefix
            and torch.cuda.is_available()
        ):
            torch.cuda.empty_cache()
        return batch


class VLAActionDenoisingStage(PipelineStage):
    def __init__(
        self,
        policy_model: Any,
        *,
        keys: VLAStageKeys = VLAStageKeys(),
    ):
        super().__init__()
        self.policy_model = policy_model
        self.keys = keys

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
            prefix_context = batch.extra.get(self.keys.prefix_context_group)
            if prefix_context is None:
                return ("single", id(batch))
            return (
                "grouped",
                id(prefix_context),
                int(
                    vla_options(batch, self.keys).get("num_inference_steps")
                    or batch.num_inference_steps
                ),
                str(vla_options(batch, self.keys).get("output_format") or "list"),
            )

        for _, group in self._group_requests_by_fingerprint(
            batches,
            action_fingerprint,
        ):
            group_batches = [batch for _, batch in group]
            prefix_context = group_batches[0].extra.get(self.keys.prefix_context_group)
            if len(group_batches) == 1 or prefix_context is None:
                for index, batch in group:
                    results[index] = self(batch, server_args)
                continue

            start = time.perf_counter()
            options = vla_options(group_batches[0], self.keys)
            observation = group_batches[0].extra[self.keys.observation_group]
            actions = self.policy_model.sample_actions(
                observation,
                prefix_context,
                noise=observation.noise,
                num_steps=int(
                    options.get("num_inference_steps")
                    or group_batches[0].num_inference_steps
                ),
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

            for offset, (index, batch) in enumerate(group):
                vla_timings(batch, self.keys)["action_denoise_ms"] = action_ms
                batch.extra[self.keys.actions] = actions[offset : offset + 1]
                batch.extra[self.keys.actions_output] = actions_out[offset]
                results[index] = batch

        return [result for result in results if result is not None]

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        start = time.perf_counter()
        observation = batch.extra.get(self.keys.observation_batch)
        split = get_vla_split_group()
        should_run_action = split is None or split.is_action_rank
        if batch.is_warmup:
            actions = (
                self.policy_model.warmup_actions(batch_size=1)
                if should_run_action
                else None
            )
        elif should_run_action:
            options = vla_options(batch, self.keys)
            prefix_context = batch.extra[self.keys.prefix_context]
            noise = observation.noise if observation is not None else None
            actions = self.policy_model.sample_actions(
                observation,
                prefix_context,
                noise=noise,
                num_steps=int(
                    options.get("num_inference_steps") or batch.num_inference_steps
                ),
                use_cuda_graph=bool(options.get("enable_cuda_graph", True)),
                generator=batch.generator,
            )
            synchronize_vla_action_tensor(actions)
        else:
            actions = None

        if split is not None:
            if should_run_action:
                vla_timings(batch, self.keys)["action_denoise_ms"] = (
                    time.perf_counter() - start
                ) * 1000
            actions = broadcast_optional_tensor(
                actions,
                split,
                src=split.action_root,
                device=self.policy_model.device,
            )
            timings = broadcast_timing(
                vla_timings(batch, self.keys) if should_run_action else None,
                split,
                src=split.action_root,
            )
            vla_timings(batch, self.keys).update(timings)
        else:
            vla_timings(batch, self.keys)["action_denoise_ms"] = (
                time.perf_counter() - start
            ) * 1000
        batch.extra[self.keys.actions] = actions
        return batch


class VLAActionPostprocessStage(PipelineStage):
    def __init__(self, *, keys: VLAStageKeys = VLAStageKeys()):
        super().__init__()
        self.keys = keys

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        start = time.perf_counter()
        action_dim = server_args.pipeline_config.output_action_dim
        options = vla_options(batch, self.keys)
        actions_out = batch.extra.get(self.keys.actions_output)
        if actions_out is None:
            action_batch = materialize_vla_action_batch(
                batch.extra[self.keys.actions],
                action_dim,
                str(options.get("output_format") or "list"),
            )
            actions_out = (
                action_batch[0] if isinstance(action_batch, list) else action_batch
            )
            if isinstance(action_batch, np.ndarray):
                actions_out = action_batch[0]

        payload = {
            "actions": actions_out,
        }
        payload["parameters"] = {
            "num_inference_steps": int(
                options.get("num_inference_steps") or batch.num_inference_steps
            )
        }
        if options.get("return_timing", True):
            timings = dict(vla_timings(batch, self.keys))
            timings["postprocess_ms"] = (time.perf_counter() - start) * 1000
            payload["timings"] = timings
        if not batch.is_warmup:
            payload["cache"] = batch.extra.get(self.keys.cache, {})

        return OutputBatch(
            output=[payload],
            metrics=batch.metrics,
        )
