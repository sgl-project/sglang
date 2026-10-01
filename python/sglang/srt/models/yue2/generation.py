# SPDX-License-Identifier: Apache-2.0
"""End-to-end YuE2 generation helpers used by the streaming server."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch

from .protocol import (
    CODEC_OFFSET,
    GenerationConfig,
    SongRequest,
    negative_prefix,
    token_prefixes,
)
from .sampling import generate_tokens
from .streaming import StreamingConfig, stream_audio


@dataclass
class CodecResult:
    prefix: list[int]
    abc: str | None
    abc_ids: list[int]
    codec_ids: list[int]
    timing: dict


def generate_codec_tokens(
    runtime,
    request: SongRequest,
    *,
    abc_max_tokens: int = 4096,
    semantic_max_tokens: int = 9000,
    use_cuda_graph: bool = True,
) -> CodecResult:
    config = GenerationConfig()
    prefix = token_prefixes(request, runtime.tokenizer)
    cfg_scale = request.cfg_scale if request.cfg_scale is not None else request.guidance

    abc_ids: list[int] = []
    abc: str | None = None
    if request.cot != "off":
        abc_sampling = config.abc
        abc_kwargs = {**abc_sampling.__dict__, "max_tokens": int(abc_max_tokens)}
        if abc_kwargs["min_tokens"] > abc_kwargs["max_tokens"]:
            abc_kwargs["min_tokens"] = abc_kwargs["max_tokens"]
        abc_sampling = type(abc_sampling)(**abc_kwargs)
        abc_ids, abc_timing, _ = generate_tokens(
            runtime.model,
            prefix,
            abc_sampling,
            request.seed,
            phase="abc",
            cfg_scale=1.0,
            use_cuda_graph=use_cuda_graph,
        )
        abc = runtime.tokenizer.decode(abc_ids)
        prefix = token_prefixes(request, runtime.tokenizer, abc_ids=abc_ids)

        semantic_sampling = config.semantic
        semantic_kwargs = {**semantic_sampling.__dict__, "max_tokens": int(semantic_max_tokens)}
        if semantic_kwargs["min_tokens"] > semantic_kwargs["max_tokens"]:
            semantic_kwargs["min_tokens"] = semantic_kwargs["max_tokens"]
        semantic_sampling = type(semantic_sampling)(**semantic_kwargs)
    negative = None
    if cfg_scale != 1.0:
        negative = negative_prefix(request, runtime.tokenizer, abc_ids=abc_ids)
    codec_ids, semantic_timing, _ = generate_tokens(
        runtime.model,
        prefix,
        semantic_sampling,
        request.seed,
        phase="semantic",
        negative=negative,
        cfg_scale=cfg_scale,
        use_cuda_graph=use_cuda_graph,
    )
    return CodecResult(
        prefix=list(prefix),
        abc=abc,
        abc_ids=list(abc_ids),
        codec_ids=[int(token) - CODEC_OFFSET for token in codec_ids],
        timing={"abc": abc_timing, "semantic": semantic_timing},
    )


def stream_generate_audio(
    runtime,
    request: SongRequest,
    *,
    config: StreamingConfig | None = None,
    abc_max_tokens: int = 4096,
    semantic_max_tokens: int = 9000,
) -> Iterator[tuple[torch.Tensor, int, int]]:
    """Generate AR codec tokens and stream NAR/VAE audio chunks."""
    result = generate_codec_tokens(
        runtime,
        request,
        abc_max_tokens=abc_max_tokens,
        semantic_max_tokens=semantic_max_tokens,
        use_cuda_graph=True,
    )
    config = config or StreamingConfig()
    yield from stream_audio(
        runtime.model,
        runtime.vae,
        prefix=result.prefix,
        codec=result.codec_ids,
        seed=request.seed,
        config=config,
    )
