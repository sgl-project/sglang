# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
import dataclasses
import os
import time
from typing import Optional

from fastapi import UploadFile

from sglang.multimodal_gen.runtime.entrypoints.utils import post_process_sample
from sglang.multimodal_gen.runtime.utils.logging_utils import (
    init_logger,
    log_batch_completion,
    log_generation_timer,
)

logger = init_logger(__name__)


@dataclasses.dataclass
class SetLoraReq:
    lora_nickname: str
    lora_path: Optional[str] = None
    target: str = "all"  # "all", "transformer", "transformer_2", "critic"


@dataclasses.dataclass
class MergeLoraWeightsReq:
    target: str = "all"  # "all", "transformer", "transformer_2", "critic"


@dataclasses.dataclass
class UnmergeLoraWeightsReq:
    target: str = "all"  # "all", "transformer", "transformer_2", "critic"


def _parse_size(size: str) -> tuple[int, int] | tuple[None, None]:
    try:
        parts = size.lower().replace(" ", "").split("x")
        if len(parts) != 2:
            raise ValueError
        w, h = int(parts[0]), int(parts[1])
        return w, h
    except Exception:
        return None, None


# Helpers
async def _save_upload_to_path(upload: UploadFile, target_path: str) -> str:
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    content = await upload.read()
    with open(target_path, "wb") as f:
        f.write(content)
    return target_path


async def process_generation_batch(
    scheduler_client,
    batch,
):
    total_start_time = time.perf_counter()
    with log_generation_timer(logger, batch.prompt):
        result = await scheduler_client.forward([batch])

        if result.output is None:
            raise RuntimeError("Model generation returned no output.")

        save_file_path = str(os.path.join(batch.output_path, batch.output_file_name))
        post_process_sample(
            result.output[0],
            batch.data_type,
            batch.fps,
            batch.save_output,
            save_file_path,
        )

    total_time = time.perf_counter() - total_start_time
    log_batch_completion(logger, 1, total_time)

    return save_file_path
