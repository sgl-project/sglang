# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any

from sglang.srt.ug.runtime import (
    UGInterleavedMessage,
    UGSRTPreparedInput,
)

U1_IMG_START_TOKEN = "<img>"
U1_IMG_END_TOKEN = "</img>"
U1_IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
U1_IMAGE_PLACEHOLDER = "<image>"
U1_T2I_CFG_UNCONDITION_ROLE = "u1_t2i_cfg_uncondition"
U1_INTERLEAVE_TEXT_UNCONDITION_ROLE = "u1_interleave_text_uncondition"
U1_EDIT_IMG_CONDITION_ROLE = "u1_edit_img_condition"
U1_EDIT_UNCONDITION_ROLE = "u1_edit_uncondition"
U1_IMAGENET_MEAN = (0.485, 0.456, 0.406)
U1_IMAGENET_STD = (0.229, 0.224, 0.225)
U1_SYSTEM_MESSAGE_FOR_GEN = (
    "You are an image generation and editing assistant that accurately understands "
    "and executes user intent.\n\n"
    "You support two modes:\n\n"
    "1. Think Mode:\n"
    "If the task requires reasoning, you MUST start with a <think></think> block. "
    "Put all reasoning inside the block using plain text. DO NOT include any image "
    "tags. Keep it reasonable and directly useful for producing the final image.\n\n"
    "2. Non-Think Mode:\n"
    "If no reasoning is needed, directly produce the final image.\n\n"
    "Task Types:\n\n"
    "A. Text-to-Image Generation:\n"
    "- Generate a high-quality image based on the user's description.\n"
    "- Ensure visual clarity, semantic consistency, and completeness.\n"
    "- DO NOT introduce elements that contradict or override the user's intent.\n\n"
    "B. Image Editing:\n"
    "- Use the provided image(s) as input or reference for modification or "
    "transformation.\n"
    "- The result can be an edited image or a new image based on the reference(s).\n"
    "- Preserve all unspecified attributes unless explicitly changed.\n\n"
    "General Rules:\n"
    "- For any visible text in the image, follow the language specified for the "
    "rendered text in the user's description, not the language of the prompt. If no "
    "language is specified, use the user's input language."
)
U1_INTERLEAVE_SYSTEM_MESSAGE = (
    "You are a multimodal assistant capable of reasoning with both text and "
    "images. You support two modes:\n\n"
    "Think Mode: When reasoning is needed, you MUST start with a "
    "<think></think> block and place all reasoning inside it. You MUST "
    "interleave text with generated images using tags like <image1>, <image2>. "
    "Images can ONLY be generated between <think> and </think>, and may be "
    "referenced in the final answer.\n\n"
    "Non-Think Mode: When no reasoning is needed, directly provide the answer "
    "without reasoning. Do not use tags like <image1>, <image2>; present any "
    "images naturally alongside the text.\n\n"
    "After the think block, always provide a concise, user-facing final answer. "
    "The answer may include text, images, or both. Match the user's language in "
    "both reasoning and the final answer."
)


def _u1_grid_hw_metadata(grid_hw: Any) -> list[list[int]]:
    return [list(map(int, row)) for row in grid_hw.tolist()]


def _u1_multimodal_inputs(
    *,
    pixel_values: Any,
    grid_hw: Any,
    offsets: list[tuple[int, int]],
    position_ids: list[list[int]] | None = None,
    precomputed_embeddings: Any | None = None,
) -> Any:
    from sglang.srt.managers.schedule_batch import (
        Modality,
        MultimodalDataItem,
        MultimodalInputs,
    )

    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=pixel_values,
        precomputed_embeddings=precomputed_embeddings,
        model_specific_data={"image_grid_hws": grid_hw},
        offsets=offsets,
    )
    item.set_pad_value()
    mm_inputs = MultimodalInputs(mm_items=[item])
    if position_ids is not None:
        import torch

        mm_inputs.mrope_positions = torch.tensor(position_ids, dtype=torch.long).t()
        mm_inputs.mrope_position_delta = (
            mm_inputs.mrope_positions[:, -1:].max(dim=0, keepdim=True).values
        )
    return mm_inputs


def build_u1_native_vlm_prepared_input(
    *,
    tokenizer: Any,
    messages: list[UGInterleavedMessage],
    session: Any | None = None,
) -> UGSRTPreparedInput:
    image = _first_u1_image_content(messages)
    question = _u1_question_text(messages)
    pixel_values, grid_hw = load_u1_native_image(image)
    input_ids, image_offsets, prompt = build_u1_vlm_input_ids_and_offsets(
        tokenizer=tokenizer,
        grid_hw=grid_hw,
        question=question,
    )

    mm_inputs = _u1_multimodal_inputs(
        pixel_values=pixel_values,
        grid_hw=grid_hw,
        offsets=image_offsets,
    )
    return UGSRTPreparedInput(
        input_ids=input_ids,
        input_text=prompt,
        messages=list(messages),
        mm_inputs=mm_inputs,
        adapter_metadata={
            "u1": {
                "segment_type": "vlm",
                "source": "native_vlm_input",
                "image_grid_hw": _u1_grid_hw_metadata(grid_hw),
                "image_offsets": list(image_offsets),
            },
            "ug_model_state_updates": {
                "u1": {
                    "last_segment_type": "vlm",
                    "last_source": "native_vlm_input",
                    "native_vlm_prompt": True,
                    "session_id": _u1_session_id(session),
                }
            },
        },
    )


def build_u1_native_interleave_prepared_input(
    *,
    tokenizer: Any,
    messages: list[UGInterleavedMessage],
    session: Any | None = None,
    think_mode: bool = False,
    system_message: str = U1_INTERLEAVE_SYSTEM_MESSAGE,
) -> UGSRTPreparedInput:
    prompt_text = _u1_prompt_with_image_placeholders(
        _u1_question_text(messages),
        image_count=len(_u1_image_contents(messages)),
    )
    prompt = build_u1_interleave_prompt(
        prompt=prompt_text,
        system_message=system_message,
        think_mode=think_mode,
    )
    return _build_u1_native_interleave_like_prepared_input(
        tokenizer=tokenizer,
        prompt=prompt,
        messages=list(messages),
        images=_u1_image_contents(messages),
        session=session,
        role=None,
        source="native_interleave_prompt",
        segment_type="interleave",
        model_state_updates={
            "last_segment_type": "interleave",
            "last_source": "native_interleave_prompt",
            "native_interleave_prompt": True,
            "open_image_marker": False,
            "interleave_pending_image_marker": False,
            "interleave_image_count": 0,
        },
    )


def build_u1_native_interleave_text_uncondition_prepared_input(
    *,
    tokenizer: Any,
    messages: list[UGInterleavedMessage],
    session: Any | None = None,
) -> UGSRTPreparedInput:
    images = _u1_image_contents(messages)
    prompt = build_u1_t2i_plain_query(prompt=U1_IMAGE_PLACEHOLDER * len(images))
    return _build_u1_native_interleave_like_prepared_input(
        tokenizer=tokenizer,
        prompt=prompt,
        messages=[UGInterleavedMessage(type="text", content="")],
        images=images,
        session=session,
        role=U1_INTERLEAVE_TEXT_UNCONDITION_ROLE,
        source="native_interleave_text_uncondition_prompt",
        segment_type="interleave_text_uncondition",
        model_state_updates={
            "last_segment_type": "interleave_text_uncondition",
            "last_source": "native_interleave_text_uncondition_prompt",
            "native_interleave_text_uncondition_prompt": True,
            "open_image_marker": False,
        },
    )


def build_u1_native_interleave_text_uncondition_marker_prepared_input(
    *,
    tokenizer: Any,
    session: Any | None = None,
    logical_position: int,
) -> UGSRTPreparedInput:
    img_start_id = tokenizer.convert_tokens_to_ids(U1_IMG_START_TOKEN)
    next_position = int(logical_position) + 1
    return UGSRTPreparedInput(
        input_ids=[int(img_start_id)],
        input_text=U1_IMG_START_TOKEN,
        messages=[],
        position_ids=[[int(logical_position), 0, 0]],
        srt_sidecar_role=U1_INTERLEAVE_TEXT_UNCONDITION_ROLE,
        srt_sidecar_session_id=_u1_sidecar_session_id(
            session,
            U1_INTERLEAVE_TEXT_UNCONDITION_ROLE,
        ),
        adapter_metadata={
            "u1": {
                "segment_type": "interleave_text_uncondition_image_marker",
                "source": "native_interleave_text_uncondition_image_marker",
                "g_position_start": next_position,
            },
            "ug_model_state_updates": {
                "u1": {
                    "last_segment_type": "interleave_text_uncondition",
                    "last_source": "native_interleave_text_uncondition_image_marker",
                    "native_interleave_text_uncondition_prompt": True,
                    "open_image_marker": True,
                    "g_position_start": next_position,
                    "session_id": _u1_session_id(session),
                }
            },
        },
    )


def _build_u1_native_interleave_like_prepared_input(
    *,
    tokenizer: Any,
    prompt: str,
    messages: list[UGInterleavedMessage],
    images: list[Any],
    session: Any | None,
    role: str | None,
    source: str,
    segment_type: str,
    model_state_updates: dict[str, Any] | None,
) -> UGSRTPreparedInput:
    mm_inputs = None
    image_offsets: list[tuple[int, int]] = []
    g_position_start = None
    if images:
        import torch

        pixel_values_list = []
        grid_hw_list = []
        for image in images:
            pixel_values, grid_hw = load_u1_native_image(
                image,
                min_pixels=512 * 512,
                max_pixels=min(2048 * 2048, (4096 * 4096) // max(1, len(images))),
                upscale=False,
            )
            pixel_values_list.append(pixel_values)
            grid_hw_list.append(grid_hw)
        pixel_values = torch.cat(pixel_values_list, dim=0)
        grid_hw = torch.cat(grid_hw_list, dim=0)
        prompt = _replace_u1_image_placeholders(prompt, grid_hw)

    input_ids = _u1_tokenize_to_ids(
        tokenizer,
        prompt,
    )
    if not input_ids:
        raise RuntimeError(f"U1 native {segment_type} prompt produced no input ids")
    if images:
        from sglang.srt.models.sensenova_u1 import build_u1_vlm_thw_indexes

        context_id = tokenizer.convert_tokens_to_ids(U1_IMG_CONTEXT_TOKEN)
        image_offsets = _u1_image_context_offsets(
            input_ids,
            context_token_id=context_id,
        )
        if len(image_offsets) != len(images):
            raise RuntimeError(
                "U1 native interleave prompt image context count mismatch: "
                f"{len(image_offsets)} != {len(images)}"
            )
        positions = build_u1_vlm_thw_indexes(
            input_ids,
            grid_hw=grid_hw,
            img_start_token_id=tokenizer.convert_tokens_to_ids(U1_IMG_START_TOKEN),
            img_context_token_id=context_id,
        )
        g_position_start = int(positions[0].max().item()) + 1
        mm_inputs = _u1_multimodal_inputs(
            pixel_values=pixel_values,
            grid_hw=grid_hw,
            offsets=image_offsets,
        )
    else:
        g_position_start = len(input_ids)

    session_id = _u1_session_id(session)
    u1_metadata = {
        "segment_type": segment_type,
        "source": source,
        "g_position_start": g_position_start,
    }
    if image_offsets:
        u1_metadata["image_offsets"] = list(image_offsets)
        u1_metadata["image_count"] = len(images)
    adapter_metadata = {"u1": u1_metadata}
    if model_state_updates is not None:
        state_updates = dict(model_state_updates)
        state_updates["g_position_start"] = g_position_start
        state_updates["session_id"] = session_id
        adapter_metadata["ug_model_state_updates"] = {"u1": state_updates}
    return UGSRTPreparedInput(
        input_ids=input_ids,
        input_text=prompt,
        messages=messages,
        mm_inputs=mm_inputs,
        srt_sidecar_role=role,
        srt_sidecar_session_id=_u1_sidecar_session_id(session, role),
        adapter_metadata=adapter_metadata,
    )


def build_u1_native_t2i_prepared_input(
    *,
    tokenizer: Any,
    messages: list[UGInterleavedMessage],
    session: Any | None = None,
) -> UGSRTPreparedInput:
    prompt_text = _u1_question_text(messages)
    prompt = build_u1_t2i_prompt(prompt=prompt_text)
    input_ids = _u1_tokenize_to_ids(
        tokenizer,
        prompt,
        add_special_tokens=False,
    )
    if not input_ids:
        raise RuntimeError("U1 native T2I prompt produced no input ids")
    img_start_id = tokenizer.convert_tokens_to_ids(U1_IMG_START_TOKEN)
    if img_start_id not in input_ids:
        raise RuntimeError("U1 native T2I prompt did not contain <img> token")
    return UGSRTPreparedInput(
        input_ids=input_ids,
        input_text=prompt,
        messages=list(messages),
        adapter_metadata={
            "u1": {
                "segment_type": "t2i",
                "source": "native_t2i_prompt",
                "prompt_ends_with_image_marker": input_ids[-1] == img_start_id,
            },
            "ug_model_state_updates": {
                "u1": {
                    "last_segment_type": "t2i",
                    "last_source": "native_t2i_prompt",
                    "native_t2i_prompt": True,
                    "open_image_marker": input_ids[-1] == img_start_id,
                    "session_id": _u1_session_id(session),
                }
            },
        },
    )


def build_u1_native_edit_prepared_input(
    *,
    tokenizer: Any,
    messages: list[UGInterleavedMessage],
    session: Any | None = None,
) -> UGSRTPreparedInput:
    image = _first_u1_image_content(messages)
    prompt_text = _u1_question_text(messages)
    if U1_IMAGE_PLACEHOLDER not in prompt_text:
        prompt_text = f"{U1_IMAGE_PLACEHOLDER}\n{prompt_text}"
    pixel_values, grid_hw = load_u1_native_image(
        image,
        min_pixels=512 * 512,
        max_pixels=2048 * 2048,
        upscale=False,
    )
    prompt = build_u1_t2i_prompt(prompt=prompt_text)
    prompt = _replace_u1_image_placeholders(prompt, grid_hw)
    input_ids = _u1_tokenize_to_ids(
        tokenizer,
        prompt,
        add_special_tokens=False,
    )
    if not input_ids:
        raise RuntimeError("U1 native edit prompt produced no input ids")
    context_id = tokenizer.convert_tokens_to_ids(U1_IMG_CONTEXT_TOKEN)
    selected = [
        index for index, token_id in enumerate(input_ids) if token_id == context_id
    ]
    if not selected:
        raise RuntimeError("U1 native edit prompt did not contain image context tokens")

    from sglang.srt.models.sensenova_u1 import build_u1_vlm_thw_indexes

    positions = build_u1_vlm_thw_indexes(
        input_ids,
        grid_hw=grid_hw,
        img_start_token_id=tokenizer.convert_tokens_to_ids(U1_IMG_START_TOKEN),
        img_context_token_id=context_id,
    )
    g_position_start = int(positions[0].max().item()) + 1
    image_offsets = [(selected[0], selected[-1])]
    mm_inputs = _u1_multimodal_inputs(
        pixel_values=pixel_values,
        grid_hw=grid_hw,
        offsets=image_offsets,
    )
    return UGSRTPreparedInput(
        input_ids=input_ids,
        input_text=prompt,
        messages=list(messages),
        mm_inputs=mm_inputs,
        adapter_metadata={
            "u1": {
                "segment_type": "edit",
                "source": "native_edit_prompt",
                "image_grid_hw": _u1_grid_hw_metadata(grid_hw),
                "image_offsets": image_offsets,
                "g_position_start": g_position_start,
            },
            "ug_model_state_updates": {
                "u1": {
                    "last_segment_type": "edit",
                    "last_source": "native_edit_prompt",
                    "native_edit_prompt": True,
                    "g_position_start": g_position_start,
                    "session_id": _u1_session_id(session),
                }
            },
        },
    )


def build_u1_native_edit_img_condition_prepared_input(
    *,
    tokenizer: Any,
    messages: list[UGInterleavedMessage],
    session: Any | None = None,
) -> UGSRTPreparedInput:
    image = _first_u1_image_content(messages)
    pixel_values, grid_hw = load_u1_native_image(
        image,
        min_pixels=512 * 512,
        max_pixels=2048 * 2048,
        upscale=False,
    )
    prompt = build_u1_t2i_plain_query(
        prompt=U1_IMAGE_PLACEHOLDER,
        append_text=U1_IMG_START_TOKEN,
    )
    prompt = _replace_u1_image_placeholders(prompt, grid_hw)
    prepared = _build_u1_native_image_sidecar_prepared_input(
        tokenizer=tokenizer,
        prompt=prompt,
        image=image,
        pixel_values=pixel_values,
        grid_hw=grid_hw,
        role=U1_EDIT_IMG_CONDITION_ROLE,
        source="native_edit_img_condition_prompt",
        segment_type="edit_img_condition",
        session=session,
    )
    return prepared


def build_u1_native_edit_uncondition_prepared_input(
    *,
    tokenizer: Any,
    session: Any | None = None,
) -> UGSRTPreparedInput:
    return _build_u1_native_marker_sidecar_prepared_input(
        tokenizer=tokenizer,
        session=session,
        role=U1_EDIT_UNCONDITION_ROLE,
        source="native_edit_uncondition_prompt",
        segment_type="edit_uncondition",
    )


def _build_u1_native_image_sidecar_prepared_input(
    *,
    tokenizer: Any,
    prompt: str,
    image: Any,
    pixel_values: Any,
    grid_hw: Any,
    role: str,
    source: str,
    segment_type: str,
    session: Any | None,
) -> UGSRTPreparedInput:
    input_ids = _u1_tokenize_to_ids(
        tokenizer,
        prompt,
        add_special_tokens=False,
    )
    if not input_ids:
        raise RuntimeError(f"U1 native {segment_type} prompt produced no input ids")
    context_id = tokenizer.convert_tokens_to_ids(U1_IMG_CONTEXT_TOKEN)
    selected = [
        index for index, token_id in enumerate(input_ids) if token_id == context_id
    ]
    if not selected:
        raise RuntimeError(
            f"U1 native {segment_type} prompt did not contain image context tokens"
        )

    from sglang.srt.models.sensenova_u1 import build_u1_vlm_thw_indexes

    positions = build_u1_vlm_thw_indexes(
        input_ids,
        grid_hw=grid_hw,
        img_start_token_id=tokenizer.convert_tokens_to_ids(U1_IMG_START_TOKEN),
        img_context_token_id=context_id,
    )
    g_position_start = int(positions[0].max().item()) + 1
    image_offsets = [(selected[0], selected[-1])]
    mm_inputs = _u1_multimodal_inputs(
        pixel_values=pixel_values,
        grid_hw=grid_hw,
        offsets=image_offsets,
    )
    return UGSRTPreparedInput(
        input_ids=input_ids,
        input_text=prompt,
        messages=[UGInterleavedMessage(type="image", content=image)],
        mm_inputs=mm_inputs,
        srt_sidecar_role=role,
        srt_sidecar_session_id=_u1_sidecar_session_id(session, role),
        adapter_metadata={
            "u1": {
                "segment_type": segment_type,
                "source": source,
                "image_grid_hw": _u1_grid_hw_metadata(grid_hw),
                "image_offsets": image_offsets,
                "g_position_start": g_position_start,
            }
        },
    )


def build_u1_native_generated_image_commit_prepared_input(
    *,
    tokenizer: Any,
    image: Any,
    session: Any | None = None,
    patch_size: int = 16,
    downsample_ratio: float = 0.5,
) -> UGSRTPreparedInput:
    precomputed_embeddings = None
    if isinstance(image, dict) and image.get("precomputed_embeddings") is not None:
        import torch

        precomputed_embeddings = (
            torch.as_tensor(image["precomputed_embeddings"]).detach().cpu()
        )
        grid_hw_value = image.get("grid_hw", image.get("image_grid_hws"))
        pixel_values = None
        if grid_hw_value is None:
            if image.get("pixel_values") is None:
                raise ValueError(
                    "U1 generated image commit with precomputed embeddings "
                    "requires grid_hw/image_grid_hws or pixel_values"
                )
            pixel_values, loaded_grid_hw = load_u1_generated_image_for_commit(
                image,
                patch_size=patch_size,
            )
            grid_hw = loaded_grid_hw
        else:
            grid_hw = torch.as_tensor(grid_hw_value, dtype=torch.long)
    else:
        pixel_values, grid_hw = load_u1_generated_image_for_commit(
            image,
            patch_size=patch_size,
        )
    merge_size = _u1_merge_size_from_downsample_ratio(downsample_ratio)
    grid_h = int(grid_hw[0, 0])
    grid_w = int(grid_hw[0, 1])
    if grid_h % merge_size or grid_w % merge_size:
        raise ValueError(
            "U1 generated image patch grid must be divisible by merge size "
            f"{merge_size}, got {grid_h}x{grid_w}"
        )
    token_h = grid_h // merge_size
    token_w = grid_w // merge_size
    num_context_tokens = token_h * token_w
    if num_context_tokens <= 0:
        raise ValueError("U1 generated image commit requires image context tokens")
    if precomputed_embeddings is not None:
        flat_precomputed = precomputed_embeddings.reshape(
            -1, precomputed_embeddings.shape[-1]
        )
        if int(flat_precomputed.shape[0]) != num_context_tokens:
            raise ValueError(
                "U1 generated image commit precomputed embedding length must "
                f"match image context tokens, got {int(flat_precomputed.shape[0])} "
                f"vs {num_context_tokens}"
            )
        precomputed_embeddings = flat_precomputed

    img_start_id = tokenizer.convert_tokens_to_ids(U1_IMG_START_TOKEN)
    context_id = tokenizer.convert_tokens_to_ids(U1_IMG_CONTEXT_TOKEN)
    img_end_id = tokenizer.convert_tokens_to_ids(U1_IMG_END_TOKEN)
    omit_start = _u1_session_has_open_image_marker(session, img_start_id)
    prefix_len = _u1_session_logical_position(session)
    if prefix_len is None:
        prefix_len = _u1_session_context_length(session)

    input_ids: list[int] = []
    position_ids: list[list[int]] = []
    if omit_start:
        context_t = prefix_len
    else:
        input_ids.append(int(img_start_id))
        position_ids.append([prefix_len, 0, 0])
        context_t = prefix_len + 1

    context_start = len(input_ids)
    input_ids.extend([int(context_id)] * num_context_tokens)
    for h_idx in range(token_h):
        for w_idx in range(token_w):
            position_ids.append([context_t, h_idx, w_idx])
    context_end = len(input_ids) - 1
    end_t = context_t + 1
    input_ids.append(int(img_end_id))
    position_ids.append([end_t, 0, 0])

    mm_inputs = _u1_multimodal_inputs(
        pixel_values=pixel_values,
        grid_hw=grid_hw,
        offsets=[(context_start, context_end)],
        position_ids=position_ids,
        precomputed_embeddings=precomputed_embeddings,
    )

    message = UGInterleavedMessage(type="image", content=image)
    metadata = _u1_generated_image_commit_metadata(
        session=session,
        token_indices=list(range(context_start, context_end + 1)),
        grid_hw=grid_hw,
        omit_start=omit_start,
        g_position_start=end_t + 1,
    )
    return UGSRTPreparedInput(
        input_ids=input_ids,
        input_text="<u1:generated_image_commit>",
        messages=[message],
        position_ids=position_ids,
        mm_inputs=mm_inputs,
        adapter_metadata=metadata,
    )


def build_u1_native_t2i_cfg_uncondition_prepared_input(
    *,
    tokenizer: Any,
    session: Any | None = None,
) -> UGSRTPreparedInput:
    return _build_u1_native_marker_sidecar_prepared_input(
        tokenizer=tokenizer,
        session=session,
        role=U1_T2I_CFG_UNCONDITION_ROLE,
        source="native_t2i_cfg_uncondition_prompt",
        segment_type="t2i_cfg_uncondition",
    )


def _build_u1_native_marker_sidecar_prepared_input(
    *,
    tokenizer: Any,
    session: Any | None,
    role: str,
    source: str,
    segment_type: str,
) -> UGSRTPreparedInput:
    prompt = build_u1_t2i_uncondition_prompt()
    input_ids = _u1_tokenize_to_ids(
        tokenizer,
        prompt,
        add_special_tokens=False,
    )
    if not input_ids:
        raise RuntimeError(f"U1 native {segment_type} prompt produced no input ids")
    img_start_id = tokenizer.convert_tokens_to_ids(U1_IMG_START_TOKEN)
    if input_ids[-1] != img_start_id:
        raise RuntimeError(f"U1 native {segment_type} prompt must end with <img>")
    return UGSRTPreparedInput(
        input_ids=input_ids,
        input_text=prompt,
        messages=[UGInterleavedMessage(type="text", content="")],
        srt_sidecar_role=role,
        srt_sidecar_session_id=_u1_sidecar_session_id(session, role),
        adapter_metadata={
            "u1": {
                "segment_type": segment_type,
                "source": source,
                "prompt_ends_with_image_marker": True,
            }
        },
    )


def build_u1_t2i_prompt(*, prompt: str) -> str:
    return (
        f"<|im_start|>system\n{U1_SYSTEM_MESSAGE_FOR_GEN}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
        f"{U1_IMG_START_TOKEN}"
    )


def build_u1_interleave_prompt(
    *,
    prompt: str,
    system_message: str = U1_INTERLEAVE_SYSTEM_MESSAGE,
    think_mode: bool = False,
) -> str:
    query = (
        f"<|im_start|>system\n{system_message}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    )
    if not think_mode:
        query += "<think>\n\n</think>\n\n"
    return query


def build_u1_t2i_plain_query(*, prompt: str, append_text: str | None = None) -> str:
    query = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    if append_text is not None:
        query += append_text
    return query


def _replace_u1_image_placeholders(prompt: str, grid_hw: Any) -> str:
    for i in range(int(grid_hw.shape[0])):
        num_patch_token = int(grid_hw[i, 0] * grid_hw[i, 1] * 0.5**2)
        image_tokens = (
            U1_IMG_START_TOKEN
            + U1_IMG_CONTEXT_TOKEN * num_patch_token
            + U1_IMG_END_TOKEN
        )
        prompt = prompt.replace(U1_IMAGE_PLACEHOLDER, image_tokens, 1)
    if U1_IMAGE_PLACEHOLDER in prompt:
        raise RuntimeError("U1 prompt still contains unresolved <image> placeholders")
    return prompt


def build_u1_t2i_uncondition_prompt() -> str:
    return build_u1_t2i_plain_query(prompt="", append_text=U1_IMG_START_TOKEN)


def _u1_image_contents(messages: list[UGInterleavedMessage]) -> list[Any]:
    return [message.content for message in messages if message.type == "image"]


def _u1_prompt_with_image_placeholders(prompt: str, *, image_count: int) -> str:
    if image_count <= 0:
        return prompt
    placeholder_count = prompt.count(U1_IMAGE_PLACEHOLDER)
    if placeholder_count > image_count:
        raise ValueError(
            "U1 prompt contains more <image> placeholders than image inputs: "
            f"{placeholder_count} > {image_count}"
        )
    if placeholder_count < image_count:
        prompt = (
            f"{U1_IMAGE_PLACEHOLDER}\n" * (image_count - placeholder_count) + prompt
        )
    return prompt


def _u1_image_context_offsets(
    input_ids: list[int],
    *,
    context_token_id: int,
) -> list[tuple[int, int]]:
    offsets = []
    index = 0
    while index < len(input_ids):
        if int(input_ids[index]) != int(context_token_id):
            index += 1
            continue
        start = index
        while index + 1 < len(input_ids) and int(input_ids[index + 1]) == int(
            context_token_id
        ):
            index += 1
        offsets.append((start, index))
        index += 1
    return offsets


def build_u1_vlm_input_ids_and_offsets(
    *,
    tokenizer: Any,
    grid_hw: Any,
    question: str,
) -> tuple[list[int], list[tuple[int, int]], str]:
    prompt = build_u1_vlm_prompt(question=question)
    for i in range(int(grid_hw.shape[0])):
        num_patch_token = int(grid_hw[i, 0] * grid_hw[i, 1] * 0.5**2)
        image_tokens = (
            U1_IMG_START_TOKEN
            + U1_IMG_CONTEXT_TOKEN * num_patch_token
            + U1_IMG_END_TOKEN
        )
        prompt = prompt.replace(U1_IMAGE_PLACEHOLDER, image_tokens, 1)

    input_ids = _u1_tokenize_to_ids(tokenizer, prompt)
    context_token_id = tokenizer.convert_tokens_to_ids(U1_IMG_CONTEXT_TOKEN)
    selected = [
        index
        for index, token_id in enumerate(input_ids)
        if token_id == context_token_id
    ]
    if not selected:
        raise RuntimeError("U1 native VLM prompt did not contain image context tokens")
    return input_ids, [(selected[0], selected[-1])], prompt


def build_u1_vlm_prompt(*, question: str) -> str:
    return (
        f"<|im_start|>user\n{U1_IMAGE_PLACEHOLDER}\n{question}"
        "<|im_end|>\n<|im_start|>assistant\n"
    )


def _u1_tokenize_to_ids(
    tokenizer: Any,
    prompt: str,
    *,
    add_special_tokens: bool | None = None,
) -> list[int]:
    kwargs = {"return_tensors": "pt"}
    if add_special_tokens is not None:
        kwargs["add_special_tokens"] = add_special_tokens
    try:
        tokenized = tokenizer(prompt, **kwargs)
    except TypeError:
        tokenized = tokenizer(prompt, return_tensors="pt")
    input_ids = tokenized["input_ids"]
    if hasattr(input_ids, "tolist"):
        return input_ids[0].tolist()
    return list(input_ids[0])


def _u1_token_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None:
        raise RuntimeError(f"U1 tokenizer has no id for token {token!r}")
    return int(token_id)


def _u1_eos_token_ids(tokenizer: Any) -> set[int]:
    eos_ids = set()
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        eos_ids.add(int(eos_token_id))
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        for token in ("<|im_end|>", "</s>"):
            try:
                token_id = convert(token)
            except Exception:
                continue
            if token_id is not None:
                try:
                    eos_ids.add(int(token_id))
                except (TypeError, ValueError):
                    pass
    return eos_ids


def _u1_decode_token_ids(tokenizer: Any, token_ids: list[int]) -> str:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        return " ".join(str(token_id) for token_id in token_ids)
    try:
        return str(decode(token_ids, skip_special_tokens=True))
    except TypeError:
        return str(decode(token_ids))


def _u1_needs_text_cfg(sampling_params: Any | None) -> bool:
    if sampling_params is None:
        return False
    return float(getattr(sampling_params, "cfg_text_scale", 1.0)) > 1.0


def _u1_needs_any_cfg(sampling_params: Any | None) -> bool:
    if sampling_params is None:
        return False
    return not (
        float(getattr(sampling_params, "cfg_text_scale", 1.0)) == 1.0
        and float(getattr(sampling_params, "cfg_img_scale", 1.0)) == 1.0
    )


def load_u1_native_image(
    image: Any,
    *,
    patch_size: int = 16,
    downsample_ratio: float = 0.5,
    min_pixels: int = 65536,
    max_pixels: int = 4194304,
    upscale: bool = False,
):
    if isinstance(image, dict):
        pixel_values = image.get("pixel_values")
        grid_hw = image.get("grid_hw", image.get("image_grid_hws"))
        if pixel_values is not None and grid_hw is not None:
            import torch

            return (
                torch.as_tensor(pixel_values, dtype=torch.float32),
                torch.as_tensor(grid_hw, dtype=torch.long),
            )

    import numpy as np
    import torch
    from PIL import Image

    if not isinstance(image, Image.Image):
        image = Image.open(image)
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    else:
        image = image.convert("RGB")

    if upscale:
        image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)

    resized = _u1_dynamic_preprocess_native_resolution(
        image,
        size_factor=int(patch_size // downsample_ratio),
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    array = np.asarray(resized, dtype=np.float32) / 255.0
    pixel_values = torch.from_numpy(array).permute(2, 0, 1)
    return _u1_normalize_and_patchify(pixel_values, patch_size=patch_size)


def load_u1_generated_image_for_commit(
    image: Any,
    *,
    patch_size: int = 16,
):
    if isinstance(image, dict):
        pixel_values = image.get("pixel_values")
        if pixel_values is not None and image.get("value_range") == "minus_one_to_one":
            return _load_u1_generated_tensor_for_commit(
                pixel_values,
                patch_size=patch_size,
                minus_one_to_one=True,
            )
        grid_hw = image.get("grid_hw", image.get("image_grid_hws"))
        if pixel_values is not None and grid_hw is not None:
            import torch

            return (
                torch.as_tensor(pixel_values, dtype=torch.float32),
                torch.as_tensor(grid_hw, dtype=torch.long),
            )

    import numpy as np
    import torch
    from PIL import Image

    if torch.is_tensor(image):
        return _load_u1_generated_tensor_for_commit(
            image,
            patch_size=patch_size,
            minus_one_to_one=False,
        )
    else:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        pixel_values = torch.from_numpy(array).permute(2, 0, 1)

    return _u1_normalize_and_patchify_commit_image(
        pixel_values,
        patch_size=patch_size,
    )


def _load_u1_generated_tensor_for_commit(
    image: Any,
    *,
    patch_size: int,
    minus_one_to_one: bool,
):
    import torch

    pixel_values = torch.as_tensor(image).detach().cpu()
    pixel_values = (
        pixel_values.to(torch.bfloat16) if minus_one_to_one else pixel_values.float()
    )
    if pixel_values.ndim == 4:
        if int(pixel_values.shape[0]) != 1:
            raise ValueError(
                "U1 generated image commit expects a single image tensor, "
                f"got batch={int(pixel_values.shape[0])}"
            )
        pixel_values = pixel_values[0]
    if pixel_values.ndim != 3 or int(pixel_values.shape[0]) != 3:
        raise ValueError(
            "U1 generated image commit tensor must have shape [3,H,W] "
            f"or [1,3,H,W], got {tuple(pixel_values.shape)}"
        )
    if minus_one_to_one:
        pixel_values = pixel_values * 0.5 + 0.5
    elif float(pixel_values.min()) < 0.0:
        pixel_values = pixel_values * 0.5 + 0.5

    return _u1_normalize_and_patchify_commit_image(
        pixel_values,
        patch_size=patch_size,
    )


def _u1_session_id(session: Any | None) -> str | None:
    return getattr(getattr(session, "handle", None), "session_id", None)


def _u1_sidecar_session_id(session: Any | None, role: str | None) -> str | None:
    session_id = _u1_session_id(session)
    if role is None or session_id is None:
        return None
    return f"{session_id}:{role}"


def _u1_session_context_length(session: Any | None) -> int:
    handle = getattr(session, "handle", None)
    context_length = getattr(handle, "context_length", None)
    if context_length is not None:
        return int(context_length)
    return int(getattr(session, "srt_last_origin_input_len", 0) or 0)


def _u1_session_logical_position(session: Any | None) -> int | None:
    metadata = getattr(session, "metadata", {}) or {}
    model_state = metadata.get("ug_model_state") or {}
    u1_state = model_state.get("u1") or {}
    g_position_start = u1_state.get("g_position_start")
    if g_position_start is None:
        return None
    return int(g_position_start)


def _u1_merge_size_from_downsample_ratio(downsample_ratio: float) -> int:
    if downsample_ratio <= 0:
        raise ValueError(f"downsample_ratio must be > 0, got {downsample_ratio}")
    merge_size = int(1 / downsample_ratio)
    if merge_size <= 0 or abs((1 / merge_size) - downsample_ratio) > 1e-6:
        raise ValueError(
            "U1 downsample_ratio must be the reciprocal of an integer, "
            f"got {downsample_ratio}"
        )
    return merge_size


def _u1_session_has_open_image_marker(
    session: Any | None,
    img_start_token_id: int,
) -> bool:
    metadata = getattr(session, "metadata", {}) or {}
    model_state = metadata.get("ug_model_state") or {}
    u1_state = model_state.get("u1") or {}
    if bool(u1_state.get("open_image_marker")):
        return True
    last_output_ids = metadata.get("srt_last_u_decode_output_ids") or ()
    return bool(last_output_ids) and int(last_output_ids[-1]) == int(img_start_token_id)


def _u1_generated_image_commit_metadata(
    *,
    session: Any | None,
    token_indices: list[int],
    grid_hw: Any,
    omit_start: bool,
    g_position_start: int | None = None,
) -> dict[str, Any]:
    metadata = getattr(session, "metadata", {}) or {}
    model_state = metadata.get("ug_model_state") or {}
    previous_state = model_state.get("u1") or {}
    previous_segments = [
        dict(segment) for segment in previous_state.get("segments", [])
    ]
    u1_segment = {
        "segment_type": "image",
        "source": "native_generated_image_commit",
        "token_indices": list(token_indices),
        "attention_rows": [
            {
                "kind": "image",
                "attention": "hybrid",
                "start": min(token_indices) if token_indices else 0,
                "end": (max(token_indices) + 1) if token_indices else 0,
            }
        ],
        "generated_image_commit": True,
        "native_generated_image_commit": True,
        "omit_image_start": bool(omit_start),
        "image_grid_hw": [list(map(int, row)) for row in grid_hw.tolist()],
    }
    u1_state = {
        "segments": previous_segments + [u1_segment],
        "last_segment_type": "image",
        "last_source": "native_generated_image_commit",
        "last_generated_image_commit": True,
        "native_generated_image_commit": True,
        "open_image_marker": False,
    }
    if g_position_start is not None:
        u1_state["g_position_start"] = int(g_position_start)
    return {
        "u1": u1_segment,
        "ug_model_state_updates": {"u1": u1_state},
    }


def _u1_dynamic_preprocess_native_resolution(
    image: Any,
    *,
    size_factor: int,
    min_pixels: int,
    max_pixels: int,
):
    width, height = image.size
    resized_height, resized_width = _u1_smart_resize(
        height,
        width,
        factor=size_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return image.resize((resized_width, resized_height))


def _u1_smart_resize(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            "absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _u1_normalize_and_patchify(pixel_values: Any, *, patch_size: int):
    import torch

    mean = torch.tensor(U1_IMAGENET_MEAN, dtype=pixel_values.dtype).view(3, 1, 1)
    std = torch.tensor(U1_IMAGENET_STD, dtype=pixel_values.dtype).view(3, 1, 1)
    return _u1_preprocess_pixel_values(
        (pixel_values - mean) / std,
        patch_size=patch_size,
    )


def _u1_normalize_and_patchify_commit_image(pixel_values: Any, *, patch_size: int):
    height = int(pixel_values.shape[1])
    width = int(pixel_values.shape[2])
    if height % patch_size or width % patch_size:
        raise ValueError(
            "U1 generated image commit requires image size divisible by "
            f"patch_size={patch_size}, got {width}x{height}"
        )
    return _u1_normalize_and_patchify(pixel_values, patch_size=patch_size)


def _u1_preprocess_pixel_values(pixel_values: Any, *, patch_size: int):
    import torch

    c, h, w = pixel_values.shape
    grid_h = h // patch_size
    grid_w = w // patch_size
    flatten_pixel_values = (
        pixel_values.view(c, grid_h, patch_size, grid_w, patch_size)
        .permute(1, 3, 0, 2, 4)
        .reshape(grid_h * grid_w, c * patch_size**2)
    )
    grid_hw = torch.tensor([[grid_h, grid_w]], dtype=torch.long)
    return flatten_pixel_values.to(torch.float32), grid_hw


def _first_u1_image_content(messages: list[UGInterleavedMessage]) -> Any:
    for message in messages:
        if message.type != "image":
            continue
        return message.content
    raise ValueError("U1 VLM text generation requires an image message")


def _u1_question_text(messages: list[UGInterleavedMessage]) -> str:
    parts = [str(message.content) for message in messages if message.type == "text"]
    question = "\n".join(part for part in parts if part)
    if not question:
        raise ValueError("U1 VLM text generation requires a text question")
    return question
