from typing import Dict, List, Tuple, Union

from sglang.srt.managers.schedule_batch import Modality, MultimodalProcessorOutput


def expand_kimi_token_ids_for_images(
    processor, input_ids: List[int], images: List
) -> Tuple[List[int], List[Tuple[int, int]]]:
    image_token_id = processor.mm_tokens.image_token_id
    image_token_counts = [
        processor._processor.media_processor.media_tokens_calculator(
            {"type": "image", "image": image}
        )
        for image in images
    ]

    placeholder_count = sum(token_id == image_token_id for token_id in input_ids)
    if placeholder_count != len(image_token_counts):
        raise ValueError(
            "Kimi VLM token-id prompt has mismatched media placeholders and "
            f"images: placeholders={placeholder_count}, images={len(image_token_counts)}"
        )

    expanded_ids: List[int] = []
    image_offsets: List[Tuple[int, int]] = []
    image_index = 0
    for token_id in input_ids:
        if token_id != image_token_id:
            expanded_ids.append(token_id)
            continue

        num_tokens = image_token_counts[image_index]
        start = len(expanded_ids)
        expanded_ids.extend([image_token_id] * num_tokens)
        image_offsets.append((start, start + num_tokens - 1))
        image_index += 1

    return expanded_ids, image_offsets


async def process_kimi_token_ids_mm_data(
    processor,
    input_ids: List[int],
    image_data: List[Union[str, bytes, Dict]],
    **kwargs,
) -> MultimodalProcessorOutput:
    if not image_data:
        return MultimodalProcessorOutput(
            input_ids=list(input_ids),
            mm_items=[],
            im_token_id=processor.mm_tokens.image_token_id,
        )

    base_output = await processor.fast_load_mm_data(
        prompt="",
        image_data=image_data,
        multimodal_tokens=processor.mm_tokens,
    )
    images = base_output.images
    expanded_ids, image_offsets = expand_kimi_token_ids_for_images(
        processor=processor,
        input_ids=input_ids,
        images=images,
    )

    media_text_parts = []
    mediums = []
    for image in images:
        num_tokens = processor._processor.media_processor.media_tokens_calculator(
            {"type": "image", "image": image}
        )
        media_text_parts.append(processor.mm_tokens.image_token * num_tokens)
        mediums.append({"type": "image", "image": image})

    key = "_medias"[1:]  # bypass lint
    kwargs[key] = mediums
    ret = processor.process_mm_data(
        input_text="".join(media_text_parts),
        images=None,
        **kwargs,
    )

    collected_items = processor.collect_mm_items_from_processor_output(ret)
    for mm_item in collected_items:
        if mm_item.modality == Modality.IMAGE:
            mm_item.offsets = image_offsets

    from sglang.srt.managers.mm_utils import get_new_expanded_mm_items

    return MultimodalProcessorOutput(
        input_ids=expanded_ids,
        mm_items=get_new_expanded_mm_items(collected_items),
        im_token_id=processor.mm_tokens.image_token_id,
    )
