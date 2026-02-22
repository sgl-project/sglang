import logging
from typing import List, Union

import torch

from sglang.srt.models.glm_image import GlmImageForConditionalGeneration

logger = logging.getLogger(__name__)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens


class GlmImageProcessor(SGLangBaseProcessor):
    models = [GlmImageForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self.IMAGE_TOKEN = "<|image|>"
        self.IMAGE_START_TOKEN = "<|begin_of_image|>"
        self.IMAGE_END_TOKEN = "<|end_of_image|>"

        self.IM_TOKEN_ID = hf_config.image_token_id
        self.IMAGE_START_TOKEN_ID = hf_config.image_start_token_id
        self.IMAGE_END_TOKEN_ID = hf_config.image_end_token_id

        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.IMAGE_TOKEN,
            image_token_id=self.IM_TOKEN_ID,
        ).build(_processor)

    def _compute_glm_image_mrope_positions(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ):
        """Compute MRoPE positions for GlmImage (image generation model).

        For source images (prefill), creates 2D spatial encoding.
        For target image grids (decode), pre-computes 2D spatial positions
        so each generated token gets proper (temporal, height, width) coordinates.
        For text tokens, uses sequential positions across all 3 dims.

        The returned position_ids has shape (3, prefill_len + decode_len) where
        decode_len covers the target grid tokens. During decode, the model looks
        up positions by index (seq_len - 1) to get proper 2D spatial encoding.
        """
        seq_len = input_ids.shape[0]
        device = input_ids.device

        image_start_token_id = self.IMAGE_START_TOKEN_ID
        image_end_token_id = self.IMAGE_END_TOKEN_ID

        text_positions = torch.arange(seq_len, device=device).unsqueeze(0).repeat(3, 1)

        # Find image boundaries
        image_end_positions = torch.where(input_ids == image_end_token_id)[0]
        image_start_positions = torch.where(input_ids == image_start_token_id)[0] + 1

        current_pos = 0
        prev_image_end = 0
        position_id_parts = []

        num_complete_images = len(image_end_positions)

        for img_idx in range(min(num_complete_images, len(image_start_positions))):
            start = image_start_positions[img_idx].item()
            end = image_end_positions[img_idx].item()

            if image_grid_thw is None or img_idx >= len(image_grid_thw):
                break

            _, height, width = image_grid_thw[img_idx].tolist()
            height = int(height)
            width = int(width)

            # Text tokens before this image
            llm_pos_length = start - prev_image_end
            llm_position_ids = text_positions[
                :, current_pos : current_pos + llm_pos_length
            ]
            current_pos += llm_pos_length

            # Image tokens with 2D spatial encoding
            image_seq_length = height * width
            position_width = torch.arange(
                current_pos, current_pos + width, device=device
            ).repeat(height)
            position_height = torch.arange(
                current_pos, current_pos + height, device=device
            ).repeat_interleave(width)
            position_temporal = torch.full(
                (image_seq_length,), current_pos, device=device, dtype=torch.long
            )
            vision_position_ids = torch.stack(
                [position_temporal, position_height, position_width], dim=0
            )
            current_pos += max(height, width)

            prev_image_end = end
            position_id_parts.append(
                torch.cat([llm_position_ids, vision_position_ids], dim=-1)
            )

        # Remaining text tokens
        end_length = seq_len - prev_image_end
        llm_position_ids = text_positions[:, current_pos : current_pos + end_length]
        current_pos += end_length
        position_id_parts.append(llm_position_ids)

        # Prefill positions
        position_ids = torch.cat(position_id_parts, dim=-1)

        # --- Decode positions for target (incomplete) image grids ---
        # Target grids are those in image_grid_thw beyond the complete images.
        # These correspond to the image tokens the model will generate autoregressively.
        # Each generated token needs a 2D spatial position based on its row/col
        # in the target grid, matching HF's _cached_decode_position_ids logic.
        if image_grid_thw is not None:
            total_grids = len(image_grid_thw)
            num_decode_grids = total_grids - num_complete_images

            if num_decode_grids > 0:
                decode_pos = current_pos
                decode_parts = []

                # Iterate in reverse order to match HF's get_rope_index:
                # for i in range(1, num_decode_grids + 1): grid_idx = -i
                for i in range(1, num_decode_grids + 1):
                    grid_idx = -i
                    _, h, w = image_grid_thw[grid_idx].tolist()
                    h, w = int(h), int(w)
                    total_tokens = h * w

                    h_indices = (
                        torch.arange(h, device=device)
                        .unsqueeze(1)
                        .expand(h, w)
                        .flatten()
                    )
                    w_indices = (
                        torch.arange(w, device=device)
                        .unsqueeze(0)
                        .expand(h, w)
                        .flatten()
                    )

                    decode_temporal = torch.full(
                        (total_tokens,), decode_pos, device=device, dtype=torch.long
                    )
                    decode_height = decode_pos + h_indices
                    decode_width = decode_pos + w_indices

                    decode_parts.append(
                        torch.stack(
                            [decode_temporal, decode_height, decode_width], dim=0
                        )
                    )
                    decode_pos += max(h, w)

                # End marker for tokens after target grid
                end_marker = torch.full(
                    (3, 1), decode_pos, device=device, dtype=torch.long
                )
                decode_parts.append(end_marker)

                decode_positions = torch.cat(decode_parts, dim=1)
                position_ids = torch.cat([position_ids, decode_positions], dim=1)

        mrope_position_delta = torch.zeros([1], dtype=torch.long, device=device)
        return position_ids, mrope_position_delta

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        image_grid_thw = None

        # When input_text is a list of ints (pre-tokenized input_ids passed
        # directly via engine.generate(input_ids=...)), preserve them as-is
        # to avoid lossy decode→re-tokenize roundtrip.
        if (
            isinstance(input_text, list)
            and len(input_text)
            and isinstance(input_text[0], int)
        ):
            input_ids = torch.tensor(input_text, dtype=torch.long)
            mm_items = []
            if image_data:
                for img in image_data:
                    if not isinstance(img, dict):
                        continue
                    # Create proper mm_items from processor_output dicts
                    # so pixel_values reach the vision encoder.
                    # Only create items when actual pixel features are present.
                    if "pixel_values" in img:
                        items = self.collect_mm_items_from_processor_output(img)
                        for item in items:
                            if img.get("format") == "processor_output":
                                from sglang.srt.managers.schedule_batch import (
                                    MultimodalInputFormat,
                                )

                                item.format = MultimodalInputFormat.PROCESSOR_OUTPUT

                            # Filter image_grid_thw on mm_item to only include
                            # source grids that have corresponding pixel_values.
                            # Target generation grids (no pixels) must NOT go to
                            # vision encoder — they are only for MRoPE positions.
                            pv = getattr(item, "feature", None)
                            grid = getattr(item, "image_grid_thw", None)
                            if pv is not None and grid is not None:
                                total_pixels = pv.shape[0]
                                source_patches = 0
                                source_grid_count = 0
                                for gi in range(len(grid)):
                                    patches = int(grid[gi].prod().item())
                                    if source_patches + patches <= total_pixels:
                                        source_patches += patches
                                        source_grid_count += 1
                                    else:
                                        break
                                if source_grid_count < len(grid):
                                    item.image_grid_thw = grid[:source_grid_count]

                        mm_items.extend(items)
                    # Extract full image_grid_thw for MRoPE position computation
                    # (includes both source and target grids)
                    if "image_grid_thw" in img:
                        grid = img["image_grid_thw"]
                        if isinstance(grid, torch.Tensor):
                            image_grid_thw = grid

            # Add offsets to all mm_items (matching base_processor behavior).
            # Offsets tell the chunked prefill where image tokens are in input_ids.
            for mm_item in mm_items:
                mm_token_id = self.mm_tokens.get_token_id_by_modality(mm_item.modality)
                if mm_token_id is not None:
                    mm_item.offsets = self.get_mm_items_offset(
                        input_ids=input_ids,
                        mm_token_id=mm_token_id,
                    )
        else:
            base_output = self.load_mm_data(
                prompt=input_text,
                image_data=image_data,
                multimodal_tokens=self.mm_tokens,
            )

            mm_items, input_ids, ret = self.process_and_combine_mm_data(
                base_output, self.mm_tokens
            )

            input_ids = input_ids.flatten()

            # Get full image_grid_thw for MRoPE (includes target grids)
            image_grid_thw = getattr(ret, "image_grid_thw", None)

            # Filter mm_item grids to only source grids (with pixel_values).
            # Target generation grids must NOT go to vision encoder.
            for item in mm_items:
                pv = getattr(item, "feature", None)
                grid = getattr(item, "image_grid_thw", None)
                if pv is not None and grid is not None:
                    total_pixels = pv.shape[0]
                    source_patches = 0
                    source_grid_count = 0
                    for gi in range(len(grid)):
                        patches = int(grid[gi].prod().item())
                        if source_patches + patches <= total_pixels:
                            source_patches += patches
                            source_grid_count += 1
                        else:
                            break
                    if source_grid_count < len(grid):
                        item.image_grid_thw = grid[:source_grid_count]

        # Fallback: get image_grid_thw from mm_items or image_data dicts
        if image_grid_thw is None:
            grids = []
            for item in mm_items:
                g = getattr(item, "image_grid_thw", None)
                if g is not None:
                    grids.append(g if g.dim() == 2 else g.unsqueeze(0))
            if grids:
                image_grid_thw = torch.cat(grids, dim=0)
        if image_grid_thw is None and image_data:
            for img in image_data:
                if isinstance(img, dict) and "image_grid_thw" in img:
                    image_grid_thw = img["image_grid_thw"]
                    if isinstance(image_grid_thw, torch.Tensor):
                        break

        mrope_positions, mrope_position_delta = self._compute_glm_image_mrope_positions(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
        )

        mm_inputs = {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_token_id": self.mm_tokens.image_token_id,
            "mrope_positions": mrope_positions,
            "mrope_position_delta": mrope_position_delta,
        }

        return mm_inputs
