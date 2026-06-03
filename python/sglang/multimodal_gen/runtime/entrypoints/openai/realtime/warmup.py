# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import shutil
import time
from typing import TYPE_CHECKING

from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
    GenerateSession,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.registry import (
    get_realtime_model_adapter,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    process_generation_batch,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import ReleaseRealtimeSessionReq
from sglang.multimodal_gen.runtime.postprocess.realesrgan_upscaler import (
    REALESRGAN_TORCH_COMPILE_ENV,
)
from sglang.multimodal_gen.runtime.postprocess.rife_interpolator import (
    RIFE_TORCH_COMPILE_ENV,
)
from sglang.multimodal_gen.runtime.server_warmup import (
    MINIMUM_PICTURE_BASE64_FOR_WARMUP,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
        RealtimeVideoGenerationsRequest,
    )
    from sglang.multimodal_gen.runtime.scheduler_client import AsyncSchedulerClient
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

logger = init_logger(__name__)

REALTIME_WARMUP_DEFAULT_CHUNKS = 2
REALTIME_WARMUP_DEFAULT_SIZES = ("832x480", "480x832")
REALTIME_WARMUP_SIZE_ENV = "SGLANG_REALTIME_WARMUP_SIZE"
REALTIME_WARMUP_SIZES_ENV = "SGLANG_REALTIME_WARMUP_SIZES"


def _compile_env_enabled(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


def _upscaling_compile_enabled() -> bool:
    return _compile_env_enabled(REALESRGAN_TORCH_COMPILE_ENV)


def _frame_interpolation_compile_enabled() -> bool:
    return _compile_env_enabled(RIFE_TORCH_COMPILE_ENV)


def realtime_warmup_enabled() -> bool:
    return _upscaling_compile_enabled() or _frame_interpolation_compile_enabled()


def _realtime_warmup_sizes() -> list[str]:
    value = os.environ.get(REALTIME_WARMUP_SIZES_ENV)
    if value is None:
        value = os.environ.get(REALTIME_WARMUP_SIZE_ENV)
    if value is None:
        return list(REALTIME_WARMUP_DEFAULT_SIZES)

    sizes = []
    for size in value.split(","):
        size = size.strip()
        if size and size not in sizes:
            sizes.append(size)
    if not sizes:
        raise ValueError(f"{REALTIME_WARMUP_SIZES_ENV} must contain at least one size")
    return sizes


def _build_realtime_warmup_request(
    server_args: ServerArgs,
    *,
    warmup_chunks: int,
    size: str,
) -> "RealtimeVideoGenerationsRequest":
    from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
        RealtimeVideoGenerationsRequest,
    )

    chunk_size = int(
        server_args.pipeline_config.dit_config.arch_config.num_frames_per_block
    )
    camera_actions = [[] for _ in range(chunk_size * warmup_chunks)]
    upscaling_enabled = _upscaling_compile_enabled()
    frame_interpolation_enabled = _frame_interpolation_compile_enabled()
    return RealtimeVideoGenerationsRequest(
        type="init",
        prompt="warmup",
        first_frame=MINIMUM_PICTURE_BASE64_FOR_WARMUP,
        condition_inputs={"camera_actions": camera_actions},
        max_chunks=warmup_chunks,
        size=size,
        seed=42,
        guidance_scale=1.0,
        realtime_output_format="raw",
        enable_upscaling=upscaling_enabled,
        upscaling_model_path=os.environ.get(
            "SGLANG_REALTIME_WARMUP_REALESRGAN_MODEL_PATH"
        ),
        upscaling_scale=4,
        enable_frame_interpolation=frame_interpolation_enabled,
        frame_interpolation_exp=1,
        frame_interpolation_scale=1.0,
        frame_interpolation_model_path=os.environ.get(
            "SGLANG_REALTIME_WARMUP_RIFE_MODEL_PATH"
        ),
    )


async def _release_realtime_warmup_session(
    scheduler_client: AsyncSchedulerClient,
    session_id: str,
) -> None:
    try:
        await scheduler_client.forward(ReleaseRealtimeSessionReq(session_id=session_id))
    except Exception as e:
        logger.warning(
            "failed to release realtime warmup session, session_id=%s, error=%s",
            session_id,
            e,
        )


async def _run_realtime_warmup_for_size(
    server_args: ServerArgs,
    scheduler_client: AsyncSchedulerClient,
    *,
    size: str,
    warmup_chunks: int,
) -> None:
    session = GenerateSession()
    scheduler_session_started = False
    start_time = time.perf_counter()
    logger.info(
        "Starting realtime warmup, session_id=%s, size=%s, chunks=%s",
        session.id,
        size,
        warmup_chunks,
    )
    try:
        adapter = get_realtime_model_adapter(server_args)
        session.set_adapter(adapter)
        realtime_req = _build_realtime_warmup_request(
            server_args,
            warmup_chunks=warmup_chunks,
            size=size,
        )
        await adapter.on_init(session, realtime_req)
        session.set_request(realtime_req)

        for chunk_idx in range(warmup_chunks):
            await adapter.wait_for_next_chunk(session)
            chunk = session.new_chunk()
            batch = adapter.prepare_next_request(session, server_args, chunk)
            batch.suppress_logs = True
            batch.extra["realtime_warmup"] = True

            scheduler_session_started = True
            _, result = await process_generation_batch(scheduler_client, batch)
            if result.error is not None:
                raise RuntimeError(result.error)
            adapter.on_chunk_complete(session, result)
            logger.info(
                "Realtime warmup chunk processed, session_id=%s, size=%s, chunk_idx=%s",
                session.id,
                size,
                chunk_idx,
            )

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Realtime warmup size finished, session_id=%s, size=%s, chunks=%s, elapsed=%.2fs",
            session.id,
            size,
            warmup_chunks,
            elapsed,
        )
    finally:
        if scheduler_session_started:
            await _release_realtime_warmup_session(scheduler_client, session.id)
        if session.input_temp_dir is not None:
            shutil.rmtree(session.input_temp_dir, ignore_errors=True)
        session.dispose()


async def run_realtime_warmup(
    server_args: ServerArgs,
    scheduler_client: AsyncSchedulerClient,
) -> None:
    if not realtime_warmup_enabled():
        return

    warmup_chunks = REALTIME_WARMUP_DEFAULT_CHUNKS
    sizes = _realtime_warmup_sizes()
    start_time = time.perf_counter()
    logger.info(
        "Starting realtime warmup, sizes=%s, chunks_per_size=%s",
        sizes,
        warmup_chunks,
    )
    try:
        get_realtime_model_adapter(server_args)
    except ValueError as e:
        logger.info("Skipping realtime warmup: %s", e)
        return

    for size in sizes:
        await _run_realtime_warmup_for_size(
            server_args,
            scheduler_client,
            size=size,
            warmup_chunks=warmup_chunks,
        )

    elapsed = time.perf_counter() - start_time
    logger.info(
        "Realtime warmup finished, sizes=%s, chunks_per_size=%s, elapsed=%.2fs",
        sizes,
        warmup_chunks,
        elapsed,
    )
