from typing import Optional

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


@router.post("/set_lora")
async def set_lora(
    lora_nickname: str = Body(..., embed=True),
    lora_path: Optional[str] = Body(None, embed=True),
):
    """
    Set the LoRA adapter for the pipeline.
    """
    try:
        req = SetLoraReq(lora_nickname=lora_nickname, lora_path=lora_path)
        # Use the singleton scheduler client to forward the request
        response = await scheduler_client.forward(req)

        if isinstance(response, dict) and response.get("status") == "ok":
            return {
                "status": "ok",
                "message": f"Successfully set LoRA adapter: {lora_nickname}",
            }
        else:
            error_msg = (
                response.get("message", "Unknown error")
                if isinstance(response, dict)
                else "Unknown response format"
            )
            raise HTTPException(
                status_code=500, detail=f"Failed to set LoRA adapter: {error_msg}"
            )

    except Exception as e:
        logger.error(f"Error setting LoRA adapter: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/merge_lora_weights")
async def merge_lora_weights():
    """
    Merge LoRA weights into the base model.
    """
    try:
        req = MergeLoraWeightsReq()
        response = await scheduler_client.forward(req)

        if isinstance(response, dict) and response.get("status") == "ok":
            return {"status": "ok", "message": "Successfully merged LoRA weights"}
        else:
            error_msg = (
                response.get("message", "Unknown error")
                if isinstance(response, dict)
                else "Unknown response format"
            )
            raise HTTPException(
                status_code=500, detail=f"Failed to merge LoRA weights: {error_msg}"
            )
    except Exception as e:
        logger.error(f"Error merging LoRA weights: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/unmerge_lora_weights")
async def unmerge_lora_weights():
    """
    Unmerge LoRA weights from the base model.
    """
    try:
        req = UnmergeLoraWeightsReq()
        response = await scheduler_client.forward(req)

        if isinstance(response, dict) and response.get("status") == "ok":
            return {"status": "ok", "message": "Successfully unmerged LoRA weights"}
        else:
            error_msg = (
                response.get("message", "Unknown error")
                if isinstance(response, dict)
                else "Unknown response format"
            )
            raise HTTPException(
                status_code=500, detail=f"Failed to unmerge LoRA weights: {error_msg}"
            )
    except Exception as e:
        logger.error(f"Error unmerging LoRA weights: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
