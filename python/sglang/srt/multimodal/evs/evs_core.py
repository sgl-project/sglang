# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/evs.py

import torch


def compute_retained_tokens_count(
    tokens_per_frame: int, num_frames: int, q: float
) -> int:
    """
    Compute the number of retained tokens for a given video.
    Method ensures that we retain all the tokens from the first frame
    regardless of the pruning rate.

    Args:
        tokens_per_frame: The number of tokens per frame.
        num_frames: The total number of frames.
        q: The pruning rate.

    Returns:
        The number of retained tokens.
    """
    total_tokens = tokens_per_frame * num_frames
    evs_num_tokens = int(total_tokens * (1 - q))
    min_num_tokens = tokens_per_frame
    return max(min_num_tokens, evs_num_tokens)


def compute_retention_mask(
    video_embeds: torch.Tensor,
    video_size_thw: torch.LongTensor | tuple[int, int, int],
    spatial_merge_size: int,
    q: float,
) -> torch.Tensor:
    """
    Computes the retention mask for input video embeddings.

    Args:
        video_embeds (`torch.Tensor`): The input video embeddings
            of shape `(T * H * W // spatial_merge_size ^ 2, hidden_size)`
        video_size_thw (`torch.LongTensor` of shape `(3)`):
            The temporal, height and width of video.
        spatial_merge_size: Size reduction for rows & cols dimensions.
        q: (`float`): Pruning rate factor [0,1)

    Returns:
        `torch.Tensor`: The retention mask for the video embeddings of
            `(T * H * W // spatial_merge_size ^ 2)` shape.
    """
    T, H, W = map(int, video_size_thw)

    # Use reshape instead of einops to avoid graph breaks
    video_embeds = video_embeds.reshape(
        T,
        H // spatial_merge_size,
        W // spatial_merge_size,
        video_embeds.size(-1),
    )
    tokens_per_frame = (H // spatial_merge_size) * (W // spatial_merge_size)
    # Core EVS
    similarity = torch.nn.functional.cosine_similarity(
        video_embeds[1:, ...], video_embeds[:-1, ...], dim=-1
    )
    dissimilarity = 1 - similarity

    # Always ensure we include all tokens from the first frame
    dissimilarity = torch.cat(
        [255 * torch.ones_like(video_embeds[:1, :, :, 0]), dissimilarity], dim=0
    )

    dissimilarity_flat = dissimilarity.view(-1)
    order = torch.argsort(dissimilarity_flat, dim=-1, descending=True, stable=True)
    retain_num_tokens = compute_retained_tokens_count(
        tokens_per_frame=tokens_per_frame, num_frames=T, q=q
    )
    topk_indices = order[:retain_num_tokens]

    retention_mask = torch.zeros_like(dissimilarity_flat, dtype=torch.bool)
    retention_mask[topk_indices] = True
    retention_mask = retention_mask.reshape(dissimilarity.size())

    mask = retention_mask.view(-1)  # "T H W -> (T H W)"
    return mask


# ▲ End of VLLM code


def redistribute_placeholder_tokens_by_tokens_per_frame(
    input_ids: torch.Tensor,
    *,
    frame_offsets_inclusive: list[tuple[int, int]],
    num_tokens_per_frame: list[int],
) -> torch.Tensor:
    """
    Adjust placeholder token spans in input_ids to match pruned token counts.

    After EVS pruning, each frame may have a different number of tokens than
    originally allocated in the prompt. This function rewrites the input_ids
    so that the placeholder token spans match the actual pruned counts, while
    preserving the total sequence length.

    For example, if frame 0 originally had 4 placeholder tokens at positions [2,6] (inclusive)
    but EVS pruned it to 2 tokens, this function shrinks that span to 2 tokens
    and redistributes the freed positions. See: test/srt/test_evs.py:test_redistribute_placeholder_tokens_by_tokens_per_frame

    Args:
        input_ids: The original input token IDs with placeholder spans.
        frame_offsets_inclusive: List of (start, end) positions for each frame's
            placeholder span in input_ids. Both start and end are inclusive.
        num_tokens_per_frame: The actual number of tokens per frame after pruning.

    Returns:
        Modified input_ids with placeholder spans adjusted to match pruned counts.
        The total length is asserted to have *remained unchanged*.
    """
    input_ids_list: list[int] = input_ids.tolist()
    filler_token_id = input_ids_list[frame_offsets_inclusive[0][0]]

    cursor = 0
    final = []
    for (start, end), num_tokens in zip(
        frame_offsets_inclusive, num_tokens_per_frame, strict=True
    ):
        final.extend(input_ids_list[cursor:start])
        final.extend([filler_token_id] * num_tokens)
        cursor = end + 1
    final.extend(input_ids_list[cursor:])
    assert len(final) == len(input_ids_list)

    final_tensor = torch.tensor(final, device=input_ids.device, dtype=input_ids.dtype)
    return final_tensor
