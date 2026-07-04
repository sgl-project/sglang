# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from typing import Any

import torch

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.models.pi05 import (
    Pi05PolicyModel,
    Pi05PrefixCacheManager,
    Pi05Preprocessor,
    broadcast_metadata,
    broadcast_optional_tensor,
    broadcast_prefix_context,
    broadcast_timing,
    collate_pi05_observation_batches,
    get_pi05_split_group,
    slice_prefix_context,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _timings(batch: Req) -> dict[str, float]:
    timings = batch.extra.setdefault("pi05_timings", {})
    return timings


def _options(batch: Req) -> dict[str, Any]:
    return batch.extra.get("pi05_options") or {}


def _effective_prefix_cache_enabled(batch: Req, server_args: ServerArgs) -> bool:
    options = _options(batch)
    return bool(options.get("enable_prefix_cache", True)) and bool(
        server_args.pipeline_config.enable_global_prefix_cache
    )


def _grouped_fingerprint(batch: Req, server_args: ServerArgs) -> tuple[Any, ...]:
    if (
        batch.is_warmup
        or get_pi05_split_group() is not None
        or _effective_prefix_cache_enabled(batch, server_args)
        or batch.generator is not None
    ):
        return ("single", id(batch))

    observation = batch.extra.get("pi05_observation_batch")
    options = _options(batch)
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


class Pi05PreprocessStage(PipelineStage):
    def __init__(self, preprocessor: Pi05Preprocessor):
        super().__init__()
        self.preprocessor = preprocessor

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.ENCODER

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        start = time.perf_counter()
        raw_observation = dict(batch.extra.get("pi05_observation") or {})
        raw_observation.setdefault("prompt", batch.prompt)
        observation = self.preprocessor(raw_observation)
        batch.extra["pi05_observation_batch"] = observation
        _timings(batch)["preprocess_ms"] = (time.perf_counter() - start) * 1000
        return batch


class Pi05PrefixStage(PipelineStage):
    def __init__(
        self,
        policy_model: Pi05PolicyModel,
        prefix_cache: Pi05PrefixCacheManager,
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
            batches, lambda batch: _grouped_fingerprint(batch, server_args)
        ):
            group_batches = [batch for _, batch in group]
            if len(group_batches) == 1 or fingerprint[0] == "single":
                for index, batch in group:
                    results[index] = self(batch, server_args)
                continue

            prefix_start = time.perf_counter()
            observations = [
                batch.extra["pi05_observation_batch"] for batch in group_batches
            ]
            grouped_observation = collate_pi05_observation_batches(observations)
            prefix_context = self.policy_model.encode_prefix(grouped_observation)
            prefix_ms = (time.perf_counter() - prefix_start) * 1000

            for offset, (index, batch) in enumerate(group):
                batch.extra["pi05_observation_group"] = grouped_observation
                batch.extra["pi05_prefix_context_group"] = prefix_context
                batch.extra["pi05_prefix_context"] = slice_prefix_context(
                    prefix_context,
                    offset,
                )
                batch.extra["pi05_group_index"] = offset
                batch.extra["pi05_group_size"] = len(group_batches)
                batch.extra["pi05_cache"] = {
                    "hit": False,
                    "match_len": 0,
                    "full_prefix_len": prefix_context.prefix_len,
                    "grouped": True,
                    "batch_size": len(group_batches),
                }
                timings = _timings(batch)
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
            batch.extra["pi05_prefix_context"] = None
            batch.extra["pi05_cache"] = {"hit": False, "warmup": True}
            return batch

        split = get_pi05_split_group()
        if split is not None and not split.is_prefix_rank:
            prefix_context = broadcast_prefix_context(
                None,
                split,
                src=split.prefix_root,
                device=self.policy_model.device,
            )
            batch.extra["pi05_prefix_context"] = prefix_context
            batch.extra["pi05_cache"] = broadcast_metadata(
                None,
                split,
                src=split.prefix_root,
            )
            timings = broadcast_timing(None, split, src=split.prefix_root)
            _timings(batch).update(timings)
            return batch

        observation = batch.extra["pi05_observation_batch"]
        options: dict[str, Any] = batch.extra.get("pi05_options") or {}
        cache_key = self.policy_model.build_prefix_cache_key(observation, options)
        batch.extra["pi05_prefix_cache_key"] = cache_key

        cache_start = time.perf_counter()
        cache_enabled = bool(options.get("enable_prefix_cache", True))
        lookup = (
            self.prefix_cache.get(cache_key)
            if cache_enabled and server_args.pipeline_config.enable_global_prefix_cache
            else None
        )
        _timings(batch)["cache_lookup_ms"] = (time.perf_counter() - cache_start) * 1000

        if lookup is not None and lookup.hit:
            batch.extra["pi05_prefix_context"] = lookup.context
            batch.extra["pi05_cache"] = {
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
                    batch.extra["pi05_cache"],
                    split,
                    src=split.prefix_root,
                )
                broadcast_timing(_timings(batch), split, src=split.prefix_root)
            return batch

        prefix_start = time.perf_counter()
        prefix_context = self.policy_model.encode_prefix(observation)
        _timings(batch)["prefix_ms"] = (time.perf_counter() - prefix_start) * 1000
        batch.extra["pi05_prefix_context"] = prefix_context
        batch.extra["pi05_cache"] = {
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
                batch.extra["pi05_cache"],
                split,
                src=split.prefix_root,
            )
            broadcast_timing(_timings(batch), split, src=split.prefix_root)
        if (
            server_args.pipeline_config.empty_cache_after_prefix
            and torch.cuda.is_available()
        ):
            torch.cuda.empty_cache()
        return batch


class Pi05ActionDenoisingStage(PipelineStage):
    def __init__(self, policy_model: Pi05PolicyModel):
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
            prefix_context = batch.extra.get("pi05_prefix_context_group")
            if prefix_context is None:
                return ("single", id(batch))
            return (
                "grouped",
                id(prefix_context),
                int(
                    _options(batch).get("num_inference_steps")
                    or batch.num_inference_steps
                ),
            )

        for _, group in self._group_requests_by_fingerprint(
            batches,
            action_fingerprint,
        ):
            group_batches = [batch for _, batch in group]
            prefix_context = group_batches[0].extra.get("pi05_prefix_context_group")
            if len(group_batches) == 1 or prefix_context is None:
                for index, batch in group:
                    results[index] = self(batch, server_args)
                continue

            start = time.perf_counter()
            options = _options(group_batches[0])
            observation = group_batches[0].extra["pi05_observation_group"]
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
            action_ms = (time.perf_counter() - start) * 1000

            for offset, (index, batch) in enumerate(group):
                _timings(batch)["action_denoise_ms"] = action_ms
                batch.extra["pi05_actions"] = actions[offset : offset + 1]
                results[index] = batch

        return [result for result in results if result is not None]

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        start = time.perf_counter()
        observation = batch.extra.get("pi05_observation_batch")
        split = get_pi05_split_group()
        should_run_action = split is None or split.is_action_rank
        if batch.is_warmup:
            actions = (
                self.policy_model.warmup_actions(batch_size=1)
                if should_run_action
                else None
            )
        elif should_run_action:
            options: dict[str, Any] = batch.extra.get("pi05_options") or {}
            prefix_context = batch.extra["pi05_prefix_context"]
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
        else:
            actions = None

        if split is not None:
            if should_run_action:
                _timings(batch)["action_denoise_ms"] = (
                    time.perf_counter() - start
                ) * 1000
            actions = broadcast_optional_tensor(
                actions,
                split,
                src=split.action_root,
                device=self.policy_model.device,
            )
            timings = broadcast_timing(
                _timings(batch) if should_run_action else None,
                split,
                src=split.action_root,
            )
            _timings(batch).update(timings)
        else:
            _timings(batch)["action_denoise_ms"] = (time.perf_counter() - start) * 1000
        batch.extra["pi05_actions"] = actions
        return batch


class Pi05PostprocessStage(PipelineStage):
    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        start = time.perf_counter()
        actions = batch.extra["pi05_actions"]
        if isinstance(actions, torch.Tensor):
            actions_out = actions.detach().float().cpu().numpy().tolist()
        else:
            actions_out = actions
        action_dim = server_args.pipeline_config.output_action_dim
        if isinstance(actions_out, list):
            actions_out = [
                [step[:action_dim] for step in sample] for sample in actions_out
            ]

        payload = {
            "actions": actions_out[0] if isinstance(actions_out, list) else actions_out,
        }
        options: dict[str, Any] = batch.extra.get("pi05_options") or {}
        payload["parameters"] = {
            "num_inference_steps": int(
                options.get("num_inference_steps") or batch.num_inference_steps
            )
        }
        if options.get("return_timing", True):
            timings = dict(_timings(batch))
            timings["postprocess_ms"] = (time.perf_counter() - start) * 1000
            payload["timings"] = timings
        if not batch.is_warmup:
            payload["cache"] = batch.extra.get("pi05_cache", {})

        return OutputBatch(
            output=[payload],
            metrics=batch.metrics,
        )
