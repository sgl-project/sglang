from typing import Any, Optional

from fastapi import APIRouter, Body, HTTPException

from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    MergeLoraWeightsReq,
    SetLoraReq,
    UnmergeLoraWeightsReq,
)
from sglang.multimodal_gen.runtime.scheduler_client import scheduler_client
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

router = APIRouter(prefix="/v1")
logger = init_logger(__name__)


async def _handle_lora_request(req: Any, success_msg: str, failure_msg: str):
    try:
        response = await scheduler_client.forward(req)
        if isinstance(response, dict) and response.get("status") == "ok":
            return {"status": "ok", "message": success_msg}
        else:
            error_msg = (
                response.get("message", "Unknown error")
                if isinstance(response, dict)
                else "Unknown response format"
            )
            raise HTTPException(status_code=500, detail=f"{failure_msg}: {error_msg}")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        logger.error(f"Error during '{failure_msg}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/set_lora")
async def set_lora(
    lora_nickname: str = Body(..., embed=True),
    lora_path: Optional[str] = Body(None, embed=True),
):
    req = SetLoraReq(lora_nickname=lora_nickname, lora_path=lora_path)
    return await _handle_lora_request(
        req,
        f"Successfully set LoRA adapter: {lora_nickname}",
        "Failed to set LoRA adapter",
    )


@router.post("/merge_lora_weights")
async def merge_lora_weights():
    req = MergeLoraWeightsReq()
    return await _handle_lora_request(
        req, "Successfully merged LoRA weights", "Failed to merge LoRA weights"
    )


@router.post("/unmerge_lora_weights")
async def unmerge_lora_weights():
    req = UnmergeLoraWeightsReq()
    return await _handle_lora_request(
        req, "Successfully unmerged LoRA weights", "Failed to unmerge LoRA weights"
    )
