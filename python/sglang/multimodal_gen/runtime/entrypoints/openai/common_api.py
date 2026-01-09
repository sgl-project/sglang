from typing import Any, Optional

from fastapi import APIRouter, Body, HTTPException

from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    MergeLoraWeightsReq,
    SetLoraReq,
    UnmergeLoraWeightsReq,
)
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

router = APIRouter(prefix="/v1")
logger = init_logger(__name__)


async def _handle_lora_request(req: Any, success_msg: str, failure_msg: str):
    try:
        response = await async_scheduler_client.forward(req)
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
    target: str = Body("all", embed=True),
    strength: float = Body(1.0, embed=True),
):
    """
    Set a LoRA adapter for the specified transformer(s).

    Args:
        lora_nickname: The nickname of the adapter.
        lora_path: Path to the LoRA adapter (local path or HF repo id).
        target: Which transformer(s) to apply the LoRA to. One of:
            - "all": Apply to all transformers (default)
            - "transformer": Apply only to the primary transformer (high noise for Wan2.2)
            - "transformer_2": Apply only to transformer_2 (low noise for Wan2.2)
            - "critic": Apply only to the critic model
        strength: LoRA strength for merge, default 1.0. Values < 1.0 reduce the effect,
            values > 1.0 amplify the effect.
    """
    req = SetLoraReq(
        lora_nickname=lora_nickname,
        lora_path=lora_path,
        target=target,
        strength=strength,
    )
    return await _handle_lora_request(
        req,
        f"Successfully set LoRA adapter: {lora_nickname} (target: {target}, strength: {strength})",
        "Failed to set LoRA adapter",
    )


@router.post("/merge_lora_weights")
async def merge_lora_weights(
    target: str = Body("all", embed=True),
    strength: float = Body(1.0, embed=True),
):
    """
    Merge LoRA weights into the base model.

    Args:
        target: Which transformer(s) to merge. One of "all", "transformer",
                "transformer_2", "critic".
        strength: LoRA strength for merge, default 1.0. Values < 1.0 reduce the effect,
            values > 1.0 amplify the effect.
    """
    req = MergeLoraWeightsReq(target=target, strength=strength)
    return await _handle_lora_request(
        req,
        f"Successfully merged LoRA weights (target: {target}, strength: {strength})",
        "Failed to merge LoRA weights",
    )


@router.post("/unmerge_lora_weights")
async def unmerge_lora_weights(
    target: str = Body("all", embed=True),
):
    """
    Unmerge LoRA weights from the base model.

    Args:
        target: Which transformer(s) to unmerge. One of "all", "transformer",
                "transformer_2", "critic".
    """
    req = UnmergeLoraWeightsReq(target=target)
    return await _handle_lora_request(
        req,
        f"Successfully unmerged LoRA weights (target: {target})",
        "Failed to unmerge LoRA weights",
    )


@router.get("/model_info")
async def model_info():
    """Get the model information."""
    server_args = get_global_server_args()
    if not server_args:
        raise HTTPException(status_code=500, detail="Server args not initialized")

    result = {
        "model_path": server_args.model_path,
    }
    return result
