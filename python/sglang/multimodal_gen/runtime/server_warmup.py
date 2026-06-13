# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
import tempfile
from typing import Any

from sglang.multimodal_gen.runtime.entrypoints.openai.utils import save_image_to_path
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

MINIMUM_PICTURE_BASE64_FOR_WARMUP = "data:image/jpg;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAbUlEQVRYhe3VsQ2AMAxE0Y/lIgNQULD/OqyCMgCihCKSG4yRuKuiNH6JLsoEbMACOGBcua9HOR7Y6w6swBwMy0qLTpkeI77qdEBpBFAHBBDAGH8WrwJKI4AAegUCfAKgEgpQDvh3CR3oQCuav58qlAw73kKCSgAAAABJRU5ErkJggg=="


def get_first_generation_req(req_or_group: Any) -> Req | None:
    """Extract the first req"""
    if isinstance(req_or_group, Req):
        return req_or_group
    if isinstance(req_or_group, list) and req_or_group:
        first_req = req_or_group[0]
        if isinstance(first_req, Req):
            return first_req
    return None


def is_warmup_req(req_or_group: Any) -> bool:
    """either server-based or req-based"""
    req = get_first_generation_req(req_or_group)
    return req.is_warmup if req is not None else False


def is_server_based_warmup(req_or_group: Any) -> bool:
    req = get_first_generation_req(req_or_group)
    return (
        req is not None and req.is_warmup and bool(req.extra.get("server_based_warmup"))
    )


def should_return_warmup_result(req_or_group: Any) -> bool:
    # server-based warmup needs to return to the http server to finish the startup
    req = get_first_generation_req(req_or_group)
    return (
        req is not None
        and req.is_warmup
        and bool(req.extra.get("return_warmup_result"))
    )


async def prepare_warmup_image_path(server_args: ServerArgs) -> str:
    if server_args.input_save_path is not None:
        uploads_dir = server_args.input_save_path
        os.makedirs(uploads_dir, exist_ok=True)
    else:
        uploads_dir = tempfile.mkdtemp(prefix="sglang_input_")

    warmup_image_base = os.path.join(uploads_dir, "warmup_image")
    return await save_image_to_path(
        MINIMUM_PICTURE_BASE64_FOR_WARMUP, warmup_image_base
    )


def prepare_warmup_image_path_sync(server_args: ServerArgs) -> str:
    return asyncio.run(prepare_warmup_image_path(server_args))
