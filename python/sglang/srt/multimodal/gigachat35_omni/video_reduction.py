# Copyright 2026 SGLang Team
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
"""GigaChat3.5-Omni video token reduction.

A video keeps every token of its first frame and, over its later frames, the
`tokens_per_later_frame * (T - 1)` tokens that differ most from the same token
of the previous frame (mean absolute difference over the hidden dim), as the
checkpoint was trained. Later frames keep content-dependent counts: the
processor lays out an even split with the right total, and `rewrite_layout`
gives every frame its real count after encoding.
"""

import bisect
from dataclasses import dataclass
from typing import List, Tuple

import torch

from sglang.srt.managers.req_types import MultimodalDataItem
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult


@dataclass(kw_only=True)
class GigaChat35OmniVideoEmbedding(EmbeddingResult):
    """The kept tokens of a request's videos.

    Attributes:
        embedding: Kept embeddings of every video, in prompt order.
        frame_counts: Kept tokens per frame, one list per video.
    """

    frame_counts: List[List[int]]

    # Frames scored per step; bounds the temporary memory of long videos.
    FRAME_CHUNK = 128
    # Videos carry this `model_specific_data` key: they take the full encoder
    # path, and every prefill chunk that meets a video's span is rewritten.
    REWRITES_LAYOUT_KEY = "rewrites_layout"
    # `model_specific_data` keys: a soundtrack's video index (audio item); a
    # video's padded prompt span, its soundtrack item and its original offsets.
    SOUNDTRACK_OF_KEY = "soundtrack_of"
    SPAN_IDS_KEY = "span_ids"
    SOUNDTRACK_KEY = "soundtrack"
    SOUNDTRACK_OFFSETS_KEY = "soundtrack_offsets"

    @staticmethod
    def num_kept_tokens(
        num_frames: int, frame_tokens: int, tokens_per_later_frame: int
    ) -> int:
        """Counts the tokens a video keeps.

        Args:
            num_frames: Frames of the video.
            frame_tokens: Tokens of one frame.
            tokens_per_later_frame: Tokens kept per later frame on average.

        Returns:
            The first frame plus the later frames' budget, at most every token.
        """
        budget = frame_tokens + tokens_per_later_frame * (num_frames - 1)
        return min(frame_tokens * num_frames, budget)

    @classmethod
    def placeholder_counts(
        cls, num_frames: int, frame_tokens: int, tokens_per_later_frame: int
    ) -> List[int]:
        """Splits a video's kept tokens evenly over its frames, the rest on the last.

        Args:
            num_frames: Frames of the video.
            frame_tokens: Tokens of one frame.
            tokens_per_later_frame: Tokens kept per later frame on average.

        Returns:
            Placeholders per frame.
        """
        kept = cls.num_kept_tokens(num_frames, frame_tokens, tokens_per_later_frame)
        base, rest = divmod(kept, num_frames)
        return [base] * (num_frames - 1) + [base + rest]

    @classmethod
    def reduce(
        cls,
        embeds: torch.Tensor,
        grid: Tuple[int, int, int],
        tokens_per_later_frame: int,
    ) -> Tuple[torch.Tensor, List[int]]:
        """Keeps a video's first frame and the later tokens that changed most.

        One stable sort over the whole video picks the tokens, so ties keep the
        earlier token and a frame may keep any number of them.

        Args:
            embeds: Embeddings of every frame, `[T * H * W, hidden]`.
            grid: `(T, H, W)` of the video.
            tokens_per_later_frame: Tokens kept per later frame on average.

        Returns:
            The kept embeddings in prompt order and the kept tokens per frame.
        """
        num_frames, height, width = grid
        frames = embeds.reshape(num_frames, height, width, embeds.shape[-1])
        scores = frames.new_empty((num_frames - 1, height, width))

        for start in range(1, num_frames, cls.FRAME_CHUNK):
            end = min(start + cls.FRAME_CHUNK, num_frames)
            diff = frames[start:end] - frames[start - 1 : end - 1]
            scores[start - 1 : end - 1] = diff.abs().mean(dim=-1)

        first = torch.full_like(frames[:1, :, :, 0], torch.inf)
        order = torch.argsort(
            torch.cat([first, scores]).view(-1), descending=True, stable=True
        )
        kept = cls.num_kept_tokens(num_frames, height * width, tokens_per_later_frame)
        keep = torch.zeros_like(order, dtype=torch.bool)
        keep[order[:kept]] = True
        return embeds[keep], keep.reshape(num_frames, -1).sum(dim=-1).tolist()

    def rewrite_layout(
        self,
        input_ids: torch.Tensor,
        *,
        items: List[MultimodalDataItem],
        extend_prefix_len: int,
        extend_seq_len: int,
        extend_start: int,
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """Gives every frame of the request's videos its kept token count.

        The stamps and soundtrack runs between frames move with them. A video
        keeps its total, so nothing outside its span moves. The span is rebuilt
        from the ids `pad_input_ids` saved and the soundtrack's offsets from its
        saved ones, so every chunk gets the same layout.

        Args:
            input_ids: Token ids of the whole batch.
            items: The request's videos, in prompt order.
            extend_prefix_len: Tokens of the request already in the KV cache.
            extend_seq_len: Tokens of the request in this batch.
            extend_start: Where the request's tokens start in `input_ids`.

        Returns:
            A copy of `input_ids` with the chunk's part of every span rewritten,
            and the videos' frame runs in the rewritten prompt.
        """
        input_ids = input_ids.clone()
        chunk_end = extend_prefix_len + extend_seq_len
        # Prompt position p of the request is input_ids[p + batch_offset].
        batch_offset = extend_start - extend_prefix_len
        offsets = []

        for item, counts in zip(items, self.frame_counts, strict=True):
            # Frames without stamps share one run.
            if len(item.offsets) == 1:
                counts = [sum(counts)]

            num_placeholders = sum(end - start + 1 for start, end in item.offsets)
            assert sum(counts) == num_placeholders, (
                f"Video kept {sum(counts)} tokens for {num_placeholders} placeholders"
            )
            span_start = item.offsets[0][0]
            span_ids = item.model_specific_data[self.SPAN_IDS_KEY]
            # Ends of the old frame runs, and the running sum of their size changes.
            ends, shifts, span = [], [0], []
            cursor = span_start

            for (start, end), count in zip(item.offsets, counts, strict=True):
                span.extend(span_ids[cursor - span_start : start - span_start])
                span.extend([item.pad_value] * count)

                if count:
                    offsets.append((start + shifts[-1], start + shifts[-1] + count - 1))

                ends.append(end)
                shifts.append(shifts[-1] + count - (end - start + 1))
                cursor = end + 1

            first, last = max(span_start, extend_prefix_len), min(cursor, chunk_end)

            if first < last:
                input_ids[first + batch_offset : last + batch_offset] = torch.tensor(
                    span[first - span_start : last - span_start],
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )

            soundtrack = item.model_specific_data.get(self.SOUNDTRACK_KEY)

            if soundtrack is not None:
                moved = []

                for start, end in item.model_specific_data[self.SOUNDTRACK_OFFSETS_KEY]:
                    shift = shifts[bisect.bisect_left(ends, start)]
                    moved.append((start + shift, end + shift))

                soundtrack.offsets = moved

        return input_ids, offsets
