# SPDX-License-Identifier: Apache-2.0

"""Motubrain2 robot action pipeline."""

from __future__ import annotations

from collections import OrderedDict
from time import perf_counter
from typing import Any

import numpy as np
import torch

from sglang.multimodal_gen.configs.pipeline_configs.motubrain2 import (
    Motubrain2PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.motubrain2 import Motubrain2SamplingParams
from sglang.multimodal_gen.runtime.cache.conditioning import conditioning_weights_epoch
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.models.motubrain2.observation import (
    Motubrain2Preprocessor,
)
from sglang.multimodal_gen.runtime.models.motubrain2.policy import Motubrain2PolicyModel
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class Motubrain2ActionStage(PipelineStage):
    def __init__(self, model: Motubrain2PolicyModel, config: Motubrain2PipelineConfig):
        super().__init__()
        self.model = model
        self.preprocess = Motubrain2Preprocessor(config)
        self.encoded = OrderedDict()

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        request = batch.extra["motubrain2"]
        options = request["options"]
        config = server_args.pipeline_config
        timings = {}
        payload = {
            "request_id": batch.request_id,
            "parameters": {"num_inference_steps": batch.num_inference_steps},
            "parallel": {
                "split_group": False,
                "runtime_role": "monolithic",
                "action_sequence_parallel": False,
            },
        }
        if batch.is_warmup:
            actions = np.zeros(
                (config.action_horizon, config.output_action_dim), dtype=np.float32
            )
        else:
            start = perf_counter()
            observation = self.preprocess(request["observation"])
            timings["preprocess_ms"] = (perf_counter() - start) * 1000

            start = perf_counter()
            cache_enabled = (
                options["enable_prefix_cache"]
                and config.enable_global_prefix_cache
                and config.prefix_cache_max_entries > 0
                and not server_args.disable_conditioning_cache
                and server_args.conditioning_cache_max_size_mb > 0
            )
            key = (
                (conditioning_weights_epoch(), self.model.observation_key(observation))
                if cache_enabled
                else None
            )
            example = self.encoded.get(key) if cache_enabled else None
            hit = example is not None
            timings["cache_lookup_ms"] = (perf_counter() - start) * 1000
            if hit:
                self.encoded.move_to_end(key)
            else:
                start = perf_counter()
                example = self.model.encode_observation(observation)
                timings["prefix_ms"] = (perf_counter() - start) * 1000
                if cache_enabled:
                    self.encoded[key] = example
                    if len(self.encoded) > config.prefix_cache_max_entries:
                        self.encoded.popitem(last=False)
                if config.empty_cache_after_prefix and self.model.device.type == "cuda":
                    torch.cuda.empty_cache()
            payload["cache"] = {
                "hit": hit,
                "scope": "global" if cache_enabled else "request",
                "mode": "exact" if cache_enabled else "disabled",
                "prefix_len": int(example["video"].shape[0]),
            }
            start = perf_counter()
            decoded, native_outputs = self.model.predict(
                observation,
                example,
                num_steps=batch.num_inference_steps,
                generator=batch.generator,
            )
            actions = decoded.cpu().numpy()
            timings["action_denoise_ms"] = (perf_counter() - start) * 1000
            if native_outputs is not None:
                payload["extras"] = {"native_outputs": native_outputs}
        start = perf_counter()
        payload["actions"] = (
            actions if options["output_format"] == "numpy" else actions.tolist()
        )
        if options["return_timing"]:
            timings["postprocess_ms"] = (perf_counter() - start) * 1000
            payload["timings"] = timings
        return OutputBatch(output=[payload], metrics=batch.metrics)


class Motubrain2Pipeline(ComposedPipelineBase):
    """SGLang Diffusion pipeline for Motubrain2 robot policies."""

    pipeline_name = "Motubrain2Pipeline"
    pipeline_config_cls = Motubrain2PipelineConfig
    sampling_params_cls = Motubrain2SamplingParams
    _required_config_modules: list[str] = []

    def validate_disagg_role(self, role: RoleType) -> None:
        if role != RoleType.MONOLITHIC:
            raise ValueError("Motubrain2Pipeline requires monolithic execution")

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if loaded_modules is not None:
            return loaded_modules
        config: Motubrain2PipelineConfig = server_args.pipeline_config
        model = Motubrain2PolicyModel.from_pretrained(self.model_path, config)
        logger.info("Loaded Motubrain2 policy backend: %r", model)
        return {"policy_model": model}

    def release_action_session(self, session_id: str) -> bool:
        return self.get_module("policy_model").release_session(session_id)

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stage(
            Motubrain2ActionStage(
                self.get_module("policy_model"), server_args.pipeline_config
            ),
            "motubrain2_actions",
        )


EntryClass = Motubrain2Pipeline
