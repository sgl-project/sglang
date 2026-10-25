"""Where serving runs media preprocessing, and how many tokens an image takes."""

from typing import List, Optional

import torch
from transformers import BaseImageProcessor

from sglang.srt import platforms
from sglang.srt.runtime_context import get_exec
from sglang.srt.utils import is_cpu, is_npu, is_xpu

_is_cpu = is_cpu()
_is_npu = is_npu()
_is_xpu = is_xpu()


def preprocessing_competes_with_the_scheduler(
    processor, *, disable_fast_image_processor: bool
) -> bool:
    """Whether image preprocessing contends with the serving accelerator.

    The fast image processor runs inside the tokenizer process but may run on
    the same accelerator as the scheduler. A second preprocessing worker there
    adds device contention rather than CPU preprocessing parallelism.
    """
    if _is_cpu or get_exec().deterministic.rl_on_policy_target is not None:
        return False
    if disable_fast_image_processor:
        return False
    image_processor = getattr(processor, "image_processor", None)
    if not isinstance(image_processor, BaseImageProcessor):
        return False
    if _is_xpu or _is_npu:
        return True
    platform = platforms.current_platform
    return platform.is_cuda_alike()


def fast_image_processor_device(processor, *, base_gpu_id: int) -> Optional[str]:
    """The device for the fast image processor, or None to leave it unset."""
    if _is_cpu or get_exec().deterministic.rl_on_policy_target is not None:
        return "cpu"
    if _is_xpu:
        return "xpu"
    if not _is_npu:
        platform = platforms.current_platform
        if not platform.is_cuda_alike():
            return None
        return f"{platform.device_type}:{base_gpu_id}"
    if processor.__class__.__name__ == "MiniMaxVLProcessor":
        # MiniMax's image/video processors create 10-dim tensors during
        # patch extraction, exceeding the Ascend 8-dim limit; patch them
        # (same pattern as qwen-vl / GLM-4.6V) and run on NPU.
        from sglang.srt.hardware_backend.npu.modules.minimax_m3_processor import (
            npu_apply_minimax_m3_image_preprocess_patch,
            npu_apply_minimax_m3_video_preprocess_patch,
        )

        npu_apply_minimax_m3_image_preprocess_patch(processor.image_processor)
        if (
            hasattr(processor, "video_processor")
            and processor.video_processor is not None
        ):
            npu_apply_minimax_m3_video_preprocess_patch(processor.video_processor)
        return "npu"
    if processor.__class__.__name__ not in {"Glm4vProcessor", "Glm46VProcessor"}:
        # For qwen-vl, the processor hits a reshape issue from the Ascend
        # dims restriction.
        from sglang.srt.hardware_backend.npu.modules.qwen_vl_processor import (
            npu_apply_qwen_image_preprocess_patch,
        )

        npu_apply_qwen_image_preprocess_patch()
        return "npu"
    if processor.__class__.__name__ == "Glm46VProcessor":
        from sglang.srt.hardware_backend.npu.modules.glm46v_processor import (
            npu_apply_glm46v_image_preprocess_patch,
        )

        npu_apply_glm46v_image_preprocess_patch()
        return "npu"
    return None


def resolve_hf_image_token_counts(processor, images: List) -> List[int]:
    """Per-image token counts from the HF processor's own helper."""
    assert images is not None
    image_sizes = [
        (
            tuple(image.shape[-2:])
            if isinstance(image, torch.Tensor)
            else (image.height, image.width)
        )
        for image in images
    ]
    num_image_tokens = processor._get_num_multimodal_tokens(
        image_sizes=image_sizes
    ).num_image_tokens
    return [int(count) for count in num_image_tokens]
