# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import time

import torch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

from sglang.srt.models.yue2.protocol import (
    CODEC_OFFSET,
    GenerationConfig,
    Sampling,
    SongRequest,
    negative_prefix,
    token_prefixes,
)
from sglang.srt.models.yue2.sampling import generate_tokens


logger = init_logger(__name__)


class Yue2ARStage(PipelineStage):
    """Generate ABC planning and semantic codec tokens with the AR MoT path."""

    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        request: SongRequest = batch.extra["yue2_request"]
        model = self.model
        tokenizer = self.tokenizer

        generation_config = self._generation_config(batch.sampling_params)
        cfg_scale = request.cfg_scale if request.cfg_scale is not None else request.guidance

        started = time.perf_counter()
        if self._session_path_enabled():
            from sglang.srt.models.yue2.cuda_graph import default_session_pool
            from sglang.srt.models.yue2.song import generate_song_tokens

            negative = None
            if cfg_scale != 1.0:
                negative = negative_prefix(request, tokenizer)
            from sglang.srt.models.yue2.song import song_token_budget

            session = default_session_pool.acquire(
                model, [token_prefixes(request, tokenizer)] if cfg_scale == 1
                else [token_prefixes(request, tokenizer), negative],
                song_token_budget(generation_config.abc, generation_config.semantic))
            try:
                abc_ids, semantic_ids, timing, fed_codec = generate_song_tokens(
                    model, session, tokenizer, request,
                    abc_sampling=generation_config.abc,
                    semantic_sampling=generation_config.semantic,
                    cfg_scale=cfg_scale, negative_prefix_ids=negative)
            except Exception:
                default_session_pool.release(session)
                raise
            abc = tokenizer.decode(abc_ids) if abc_ids else None
            batch.extra.update(
                {
                    "yue2_abc": abc,
                    "yue2_abc_ids": list(abc_ids),
                    "yue2_semantic_ids": list(semantic_ids),
                    "yue2_semantic_prefix": token_prefixes(request, tokenizer, abc_ids=abc_ids),
                    "yue2_timing": timing,
                    "yue2_truncated": bool(timing["abc"].get("truncated")
                                           or timing["semantic"].get("truncated")),
                    "yue2_session": session,
                    "yue2_fed_codec": int(fed_codec),
                }
            )
            batch.extra["yue2_timing"]["total_seconds"] = time.perf_counter() - started
            return batch

        prefix = list(batch.extra["yue2_prefix"])
        abc_ids: list[int] = []
        abc: str | None = None
        if request.cot != "off":
            abc_ids, abc_timing, abc_truncated = generate_tokens(
                model,
                prefix,
                generation_config.abc,
                request.seed,
                phase="abc",
                cfg_scale=1.0,
                use_cuda_graph=True,
            )
            abc = tokenizer.decode(abc_ids)
            prefix = token_prefixes(request, tokenizer, abc_ids=abc_ids)

        sampling = generation_config.semantic
        negative = None
        if cfg_scale != 1.0:
            negative = negative_prefix(request, tokenizer, abc_ids=abc_ids)
        # Upstream fidelity: cot=off restores the historical (vLLM-era)
        # BF16 + keep-first-3 top-p arithmetic in the semantic phase.
        legacy_semantic = request.cot == "off"
        semantic_ids, semantic_timing, semantic_truncated = generate_tokens(
            model,
            prefix,
            sampling,
            request.seed,
            phase="semantic",
            negative=negative,
            cfg_scale=cfg_scale,
            use_cuda_graph=True,
            legacy_off=legacy_semantic,
            use_fast_sampler=not legacy_semantic,
        )

        batch.extra.update(
            {
                "yue2_abc": abc,
                "yue2_abc_ids": abc_ids,
                "yue2_semantic_ids": semantic_ids,
                "yue2_semantic_prefix": list(prefix),
                "yue2_timing": {
                    "abc": abc_timing,
                    "semantic": semantic_timing,
                    "total_seconds": time.perf_counter() - started,
                },
                "yue2_truncated": bool(abc_truncated or semantic_truncated),
            }
        )
        return batch

    @staticmethod
    def _session_path_enabled() -> bool:
        import os

        return os.environ.get("SGLANG_YUE2_SESSION", "1") == "1"

    @staticmethod
    def _generation_config(sampling_params) -> GenerationConfig:
        return GenerationConfig(
            abc=Sampling(
                temperature=0.7,
                top_p=0.9,
                top_k=30,
                repetition_penalty=1.005,
                penalty_window=100,
                min_tokens=min(
                    32,
                    int(getattr(sampling_params, "abc_max_tokens", 4096)),
                ),
                max_tokens=int(getattr(sampling_params, "abc_max_tokens", 4096)),
            ),
            semantic=Sampling(
                temperature=float(sampling_params.temperature or 1.0),
                top_p=float(sampling_params.top_p),
                top_k=int(sampling_params.top_k),
                repetition_penalty=float(sampling_params.repetition_penalty or 1.2),
                penalty_window=50,
                min_tokens=min(
                    200,
                    int(getattr(sampling_params, "semantic_max_tokens", 9000)),
                ),
                max_tokens=int(getattr(sampling_params, "semantic_max_tokens", 9000)),
            ),
            ode_steps=int(getattr(sampling_params, "ode_steps", 32)),
            ode_method="midpoint",
            context=24576,
        )
