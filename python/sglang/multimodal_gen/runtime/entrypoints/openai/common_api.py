import time
from typing import Any, List, Optional, Union

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

from sglang.multimodal_gen.registry import get_model_info
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    ListLorasReq,
    MergeLoraWeightsReq,
    SetLoraReq,
    UnmergeLoraWeightsReq,
    format_lora_message,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

router = APIRouter(prefix="/v1")
logger = init_logger(__name__)


class ModelCard(BaseModel):
    """Model cards."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "sglang"
    root: Optional[str] = None
    parent: Optional[str] = None
    max_model_len: Optional[int] = None


class DiffusionModelCard(ModelCard):
    """Extended ModelCard with diffusion-specific fields."""

    num_gpus: Optional[int] = None
    task_type: Optional[str] = None
    dit_precision: Optional[str] = None
    vae_precision: Optional[str] = None
    pipeline_name: Optional[str] = None
    pipeline_class: Optional[str] = None


async def _handle_lora_request(req: Any, success_msg: str, failure_msg: str):
    try:
        output: OutputBatch = await async_scheduler_client.forward(req)
        if output.error is None:
            return {"status": "ok", "message": success_msg}
        else:
            error_msg = output.error
            raise HTTPException(status_code=500, detail=f"{failure_msg}: {error_msg}")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        logger.error(f"Error during '{failure_msg}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/set_lora")
async def set_lora(
    lora_nickname: Union[str, List[str]] = Body(..., embed=True),
    lora_path: Optional[Union[str, List[Optional[str]]]] = Body(None, embed=True),
    target: Union[str, List[str]] = Body("all", embed=True),
    strength: Union[float, List[float]] = Body(1.0, embed=True),
):
    """
    Set LoRA adapter(s) for the specified transformer(s).
    Supports both single LoRA (backward compatible) and multiple LoRA adapters.

    Args:
        lora_nickname: The nickname(s) of the adapter(s). Can be a string or a list of strings.
        lora_path: Path(s) to the LoRA adapter(s) (local path or HF repo id).
            Can be a string, None, or a list of strings/None. Must match the length of lora_nickname.
        target: Which transformer(s) to apply the LoRA to. Can be a string or a list of strings.
            If a list, must match the length of lora_nickname. Valid values:
            - "all": Apply to all transformers (default)
            - "transformer": Apply only to the primary transformer (high noise for Wan2.2)
            - "transformer_2": Apply only to transformer_2 (low noise for Wan2.2)
            - "critic": Apply only to the critic model
        strength: LoRA strength(s) for merge, default 1.0. Can be a float or a list of floats.
            If a list, must match the length of lora_nickname. Values < 1.0 reduce the effect,
            values > 1.0 amplify the effect.
    """
    req = SetLoraReq(
        lora_nickname=lora_nickname,
        lora_path=lora_path,
        target=target,
        strength=strength,
    )
    nickname_str, target_str, strength_str = format_lora_message(
        lora_nickname, target, strength
    )

    return await _handle_lora_request(
        req,
        f"Successfully set LoRA adapter(s): {nickname_str} (target: {target_str}, strength: {strength_str})",
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


@router.get("/list_loras")
async def list_loras():
    """List loaded LoRA adapters and current application status per module."""
    try:
        req = ListLorasReq()
        output: OutputBatch = await async_scheduler_client.forward(req)
        if output.error is None:
            return output.output or {}
        else:
            raise HTTPException(status_code=500, detail=output.error)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        logger.error(f"Error during 'list_loras': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_class=ORJSONResponse)
async def available_models():
    """Show available models. OpenAI-compatible endpoint with extended diffusion info."""
    server_args = get_global_server_args()
    if not server_args:
        raise HTTPException(status_code=500, detail="Server args not initialized")

    model_info = get_model_info(server_args.model_path)

    card_kwargs = {
        "id": server_args.model_path,
        "root": server_args.model_path,
        # Extended diffusion-specific fields
        "num_gpus": server_args.num_gpus,
        "task_type": server_args.pipeline_config.task_type.name,
        "dit_precision": server_args.pipeline_config.dit_precision,
        "vae_precision": server_args.pipeline_config.vae_precision,
    }

    if model_info:
        card_kwargs["pipeline_name"] = model_info.pipeline_cls.pipeline_name
        card_kwargs["pipeline_class"] = model_info.pipeline_cls.__name__

    model_card = DiffusionModelCard(**card_kwargs)

    # Return dict directly to preserve extended fields (ModelList strips them)
    return {"object": "list", "data": [model_card.model_dump()]}


@router.get("/models/{model:path}", response_class=ORJSONResponse)
async def retrieve_model(model: str):
    """Retrieve a model instance. OpenAI-compatible endpoint with extended diffusion info."""
    server_args = get_global_server_args()
    if not server_args:
        raise HTTPException(status_code=500, detail="Server args not initialized")

    if model != server_args.model_path:
        return ORJSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": f"The model '{model}' does not exist",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        )

    model_info = get_model_info(server_args.model_path)

    card_kwargs = {
        "id": model,
        "root": model,
        "num_gpus": server_args.num_gpus,
        "task_type": server_args.pipeline_config.task_type.name,
        "dit_precision": server_args.pipeline_config.dit_precision,
        "vae_precision": server_args.pipeline_config.vae_precision,
    }

    if model_info:
        card_kwargs["pipeline_name"] = model_info.pipeline_cls.pipeline_name
        card_kwargs["pipeline_class"] = model_info.pipeline_cls.__name__

    # Return dict to preserve extended fields
    return DiffusionModelCard(**card_kwargs).model_dump()
