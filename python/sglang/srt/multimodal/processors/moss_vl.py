import asyncio
import os
import re
import tempfile
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import unquote, urlparse

import pybase64
import requests
import torch

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.moss_vl import MossVLForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    SGL_USE_CUDA_IPC,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (
    MultimodalSpecialTokens,
)
from sglang.srt.utils.cuda_ipc_transport_utils import CudaIpcTensorTransportProxy


class MossVLImageProcessor(SGLangBaseProcessor):
    models = [MossVLForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.image_only_mm_tokens = MultimodalSpecialTokens(
            image_token="<|image|>",
            image_token_regex=re.compile(re.escape("<|image|>")),
        ).build(_processor)
        self.image_token_id = getattr(hf_config, "image_token_id", None)
        self.vision_seq_pad_multiple = 1

    def _build_mm_items(
        self, processor_output: Dict, input_ids: torch.Tensor
    ) -> List[MultimodalDataItem]:
        pixel_values = processor_output.get("pixel_values")
        if pixel_values is None:
            return []

        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=pixel_values,
            model_specific_data={},
        )

        grid_thw = processor_output.get("grid_thw")
        if grid_thw is not None:
            item.set("grid_thw", grid_thw)

        return [item]

    def _build_vision_token_info(
        self,
        grid_thw: Optional[torch.Tensor],
        media_nums_per_sample: Optional[List[int]],
    ) -> List[dict]:
        if grid_thw is None:
            return []

        grid_thw = torch.as_tensor(grid_thw, dtype=torch.long)
        if grid_thw.ndim == 1:
            grid_thw = grid_thw.unsqueeze(0)
        if grid_thw.numel() == 0:
            return []

        tokens_per_media = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]) // (
            self.spatial_merge_size**2
        )

        if media_nums_per_sample is None:
            media_nums_per_sample = [grid_thw.shape[0]]

        batch_size = len(media_nums_per_sample)
        if batch_size == 1:
            total_len = 0
            for i in range(grid_thw.shape[0]):
                num_tokens = tokens_per_media[i].item()
                num_frames = grid_thw[i, 0].item()
                total_len += num_tokens + num_frames

            if total_len % self.vision_seq_pad_multiple != 0:
                max_seq_len = (
                    (total_len + self.vision_seq_pad_multiple - 1)
                    // self.vision_seq_pad_multiple
                    * self.vision_seq_pad_multiple
                )
            else:
                max_seq_len = total_len

            sample_info = {
                "medias": [],
                "total_length": total_len,
                "pad_start": total_len,
                "pad_end": max_seq_len,
            }

            current_seq_len = 0
            for media_idx in range(grid_thw.shape[0]):
                num_tokens = tokens_per_media[media_idx].item()
                t, h, w = grid_thw[media_idx].tolist()
                num_frames = t
                tokens_per_frame = num_tokens // num_frames
                chunk_len = num_frames * (tokens_per_frame + 1)

                sample_info["medias"].append(
                    {
                        "start": current_seq_len,
                        "end": current_seq_len + chunk_len,
                        "length": chunk_len,
                        "num_frames": num_frames,
                        "grid_h": h,
                        "grid_w": w,
                        "vision_tokens_per_frame": tokens_per_frame,
                        "has_separator": True,
                    }
                )
                current_seq_len += chunk_len

            return [sample_info]

        tokens_per_sample = []
        media_idx = 0
        for num_medias_in_sample in media_nums_per_sample:
            sample_tokens = 0
            for i in range(num_medias_in_sample):
                num_tokens = tokens_per_media[media_idx + i].item()
                num_frames = grid_thw[media_idx + i, 0].item()
                sample_tokens += num_tokens + num_frames
            tokens_per_sample.append(sample_tokens)
            media_idx += num_medias_in_sample

        max_seq_len = max(tokens_per_sample)
        if max_seq_len % self.vision_seq_pad_multiple != 0:
            max_seq_len = (
                (max_seq_len + self.vision_seq_pad_multiple - 1)
                // self.vision_seq_pad_multiple
                * self.vision_seq_pad_multiple
            )

        vision_token_info = []
        media_idx = 0
        for sample_idx, num_medias_in_sample in enumerate(media_nums_per_sample):
            sample_info = {
                "medias": [],
                "total_length": tokens_per_sample[sample_idx],
                "pad_start": tokens_per_sample[sample_idx],
                "pad_end": max_seq_len,
            }

            seq_offset = 0
            for _ in range(num_medias_in_sample):
                num_tokens = tokens_per_media[media_idx].item()
                t, h, w = grid_thw[media_idx].tolist()
                num_frames = t
                tokens_per_frame = num_tokens // num_frames
                media_length = num_tokens + num_frames

                sample_info["medias"].append(
                    {
                        "start": seq_offset,
                        "end": seq_offset + media_length,
                        "length": media_length,
                        "num_frames": num_frames,
                        "grid_h": h,
                        "grid_w": w,
                        "vision_tokens_per_frame": tokens_per_frame,
                        "has_separator": True,
                    }
                )

                seq_offset += media_length
                media_idx += 1

            vision_token_info.append(sample_info)

        return vision_token_info

    def _compute_position_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        is_image_token = input_ids == self.image_token_id
        if attention_mask is not None:
            is_padding = attention_mask == 0
        else:
            is_padding = torch.zeros_like(input_ids, dtype=torch.bool)

        is_regular_token = ~(is_image_token | is_padding)
        cumulative_regular = is_regular_token.long().cumsum(dim=1)
        base_position_ids = cumulative_regular - is_regular_token.long()
        base_position_ids = base_position_ids.masked_fill(is_padding, 0)
        return base_position_ids.unsqueeze(0).expand(3, -1, -1).clone()

    def _compute_vision_position_ids(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        vision_token_info: List[dict],
        max_vision_seq_len: int,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = input_ids.shape[0]
        device = input_ids.device

        image_token_indices = (input_ids == self.image_token_id).nonzero()

        flat_eff_h = []
        flat_eff_w = []
        flat_vis_starts = []

        for info in vision_token_info:
            medias = info.get("medias", [])
            for media in medias:
                num_frames = media["num_frames"]
                h, w = media["grid_h"], media["grid_w"]
                eh, ew = h // self.spatial_merge_size, w // self.spatial_merge_size
                start = media["start"]
                tok_per_frame = media["vision_tokens_per_frame"]
                stride = tok_per_frame + 1
                for f in range(num_frames):
                    flat_eff_h.append(eh)
                    flat_eff_w.append(ew)
                    flat_vis_starts.append(start + f * stride)

        vision_pos_ids = torch.zeros(
            (3, batch_size, max_vision_seq_len),
            dtype=torch.long,
            device=device,
        )

        if len(flat_eff_h) == 0 or len(image_token_indices) == 0:
            rope_deltas = (
                position_ids.max(dim=0).values.max(dim=-1).values
                + 1
                - input_ids.shape[1]
            )
            return vision_pos_ids, position_ids, rope_deltas

        num_matches = min(len(flat_eff_h), len(image_token_indices))
        flat_eff_h = torch.tensor(
            flat_eff_h[:num_matches], device=device, dtype=torch.long
        )
        flat_eff_w = torch.tensor(
            flat_eff_w[:num_matches], device=device, dtype=torch.long
        )
        flat_vis_starts = torch.tensor(
            flat_vis_starts[:num_matches], device=device, dtype=torch.long
        )

        target_indices = image_token_indices[:num_matches]
        batch_rows = target_indices[:, 0]
        text_cols = target_indices[:, 1]

        max_hw = torch.maximum(flat_eff_h, flat_eff_w)
        shifts = max_hw + 1

        shift_map = torch.zeros(
            (batch_size, input_ids.shape[1]), dtype=torch.long, device=device
        )
        shift_map[batch_rows, text_cols] = shifts
        cum_shifts = shift_map.cumsum(dim=1)

        orig_pos = position_ids[0, batch_rows, text_cols]
        shifts_before = cum_shifts[batch_rows, text_cols] - shifts
        t_vals = orig_pos + shifts_before

        new_pos_ids = position_ids + cum_shifts.unsqueeze(0)
        img_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        img_token_mask[batch_rows, text_cols] = True
        new_pos_ids[:, img_token_mask] -= 1

        if attention_mask is not None:
            padding_mask = (attention_mask == 0).unsqueeze(0)
            new_pos_ids.masked_fill_(padding_mask, 0)

        position_ids = new_pos_ids

        unique_shapes = torch.unique(
            torch.stack([flat_eff_h, flat_eff_w], dim=1), dim=0
        )
        for shape in unique_shapes:
            eh, ew = shape[0].item(), shape[1].item()
            mask = (flat_eff_h == eh) & (flat_eff_w == ew)

            sub_t_vals = t_vals[mask]
            sub_batch_rows = batch_rows[mask]
            sub_vis_starts = flat_vis_starts[mask]
            num_frames_sub = sub_t_vals.shape[0]
            if num_frames_sub == 0:
                continue

            y_grid = (
                torch.arange(eh, device=device)
                .view(1, eh, 1)
                .expand(num_frames_sub, -1, ew)
            )
            x_grid = (
                torch.arange(ew, device=device)
                .view(1, 1, ew)
                .expand(num_frames_sub, eh, -1)
            )
            t_grid = sub_t_vals.view(-1, 1, 1).expand(-1, eh, ew)

            h_grid = t_grid + y_grid
            w_grid = t_grid + x_grid

            flat_t = t_grid.reshape(-1)
            flat_h = h_grid.reshape(-1)
            flat_w = w_grid.reshape(-1)

            tokens_per_frame = eh * ew
            seq_offsets = torch.arange(tokens_per_frame, device=device).unsqueeze(0)
            abs_seq_offsets = seq_offsets + sub_vis_starts.unsqueeze(1)

            flat_seq_inds = abs_seq_offsets.reshape(-1)
            flat_batch_inds = (
                sub_batch_rows.unsqueeze(1).expand(-1, tokens_per_frame).reshape(-1)
            )

            valid_mask = flat_seq_inds < max_vision_seq_len
            if valid_mask.any():
                final_b = flat_batch_inds[valid_mask]
                final_s = flat_seq_inds[valid_mask]
                vision_pos_ids[0, final_b, final_s] = flat_t[valid_mask]
                vision_pos_ids[1, final_b, final_s] = flat_h[valid_mask]
                vision_pos_ids[2, final_b, final_s] = flat_w[valid_mask]

        sep_vals = t_vals + max_hw
        sep_indices = flat_vis_starts + (flat_eff_h * flat_eff_w)
        valid_sep_mask = sep_indices < max_vision_seq_len
        if valid_sep_mask.any():
            final_b = batch_rows[valid_sep_mask]
            final_s = sep_indices[valid_sep_mask]
            vals = sep_vals[valid_sep_mask]
            vision_pos_ids[0, final_b, final_s] = vals
            vision_pos_ids[1, final_b, final_s] = vals
            vision_pos_ids[2, final_b, final_s] = vals

        max_pos = position_ids.max(dim=0).values.max(dim=-1).values
        rope_deltas = max_pos + 1 - input_ids.shape[1]
        return vision_pos_ids, position_ids, rope_deltas

    def _compute_position_metadata(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        grid_thw: Optional[torch.Tensor],
        media_nums_per_sample: Optional[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], List[dict]]:
        position_ids = self._compute_position_ids(input_ids, attention_mask)

        if grid_thw is None:
            max_pos = position_ids.max(dim=0).values.max(dim=-1).values
            rope_deltas = (max_pos + 1 - input_ids.shape[1]).unsqueeze(1)
            return position_ids, rope_deltas, None, []

        vision_token_info = self._build_vision_token_info(
            grid_thw, media_nums_per_sample
        )
        max_vision_seq_len = 0
        if vision_token_info:
            max_vision_seq_len = max(
                info.get("pad_end", 0) for info in vision_token_info
            )

        if max_vision_seq_len == 0:
            max_pos = position_ids.max(dim=0).values.max(dim=-1).values
            rope_deltas = (max_pos + 1 - input_ids.shape[1]).unsqueeze(1)
            return position_ids, rope_deltas, None, vision_token_info

        vision_position_ids, position_ids, rope_deltas = (
            self._compute_vision_position_ids(
                input_ids=input_ids,
                position_ids=position_ids,
                vision_token_info=vision_token_info,
                max_vision_seq_len=max_vision_seq_len,
                attention_mask=attention_mask,
            )
        )
        return (
            position_ids,
            rope_deltas.unsqueeze(1),
            vision_position_ids,
            vision_token_info,
        )

    def _compute_visible_frame_counts(
        self, cross_attention_mask: Optional[Union[torch.Tensor, List]]
    ) -> Optional[torch.Tensor]:
        if cross_attention_mask is None:
            return None

        # HF Moss-VL processor outputs a bool mask with shape
        # (batch_size, 1, text_len, num_frames), where True means masked.
        cross_attention_mask = torch.as_tensor(cross_attention_mask, dtype=torch.bool)
        visible_frame_counts = (~cross_attention_mask).sum(dim=-1, dtype=torch.int32)
        return visible_frame_counts.reshape(-1)

    def _resolve_file_url(self, value: str) -> str:
        parsed = urlparse(value)
        path = unquote(parsed.path or "")
        if parsed.netloc and not path.startswith("/"):
            path = f"/{path}"
        return path

    def _write_video_bytes_to_tempfile(
        self, video_bytes: bytes, suffix: str = ".mp4"
    ) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(video_bytes)
            return f.name

    def _normalize_video_string(self, value: str) -> Tuple[str, Optional[str]]:
        if value.startswith("file://"):
            return self._resolve_file_url(value), None

        if os.path.isfile(value):
            return value, None

        if value.startswith(("http://", "https://")):
            timeout = int(os.getenv("REQUEST_TIMEOUT", "10"))
            response = requests.get(value, stream=True, timeout=timeout)
            response.raise_for_status()
            suffix = os.path.splitext(urlparse(value).path)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                return f.name, f.name

        if value.startswith("data:"):
            header, encoded = value.split(",", 1)
            mime = header.split(";", 1)[0]
            suffix = ".mp4"
            if "/" in mime:
                ext = mime.rsplit("/", 1)[-1]
                if ext:
                    suffix = f".{ext}"
            temp_path = self._write_video_bytes_to_tempfile(
                pybase64.b64decode(encoded, validate=True),
                suffix=suffix,
            )
            return temp_path, temp_path

        temp_path = self._write_video_bytes_to_tempfile(
            pybase64.b64decode(value, validate=True)
        )
        return temp_path, temp_path

    def _normalize_single_video_input(
        self, video_input: Union[str, Dict]
    ) -> Tuple[Union[str, Dict], List[str]]:
        temp_paths: List[str] = []
        if isinstance(video_input, dict):
            normalized = dict(video_input)
            video_path, temp_path = self._normalize_video_string(
                normalized["video_path"]
            )
            normalized["video_path"] = video_path
            if temp_path is not None:
                temp_paths.append(temp_path)
            return normalized, temp_paths

        normalized_path, temp_path = self._normalize_video_string(video_input)
        if temp_path is not None:
            temp_paths.append(temp_path)
        return normalized_path, temp_paths

    async def _normalize_video_inputs_async(
        self, video_data: Optional[List[Union[str, Dict]]]
    ) -> Tuple[Optional[List[Union[str, Dict]]], List[str]]:
        if not video_data:
            return video_data, []

        loop = asyncio.get_running_loop()
        futures = [
            loop.run_in_executor(
                self.io_executor, self._normalize_single_video_input, v
            )
            for v in video_data
        ]
        results = await asyncio.gather(*futures)

        normalized_inputs: List[Union[str, Dict]] = []
        temp_paths: List[str] = []
        for normalized_input, created_paths in results:
            normalized_inputs.append(normalized_input)
            temp_paths.extend(created_paths)
        return normalized_inputs, temp_paths

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, Dict]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        normalized_video_data, temp_video_paths = (
            await self._normalize_video_inputs_async(request_obj.video_data)
        )

        try:
            base_output = self.load_mm_data(
                prompt=input_text,
                image_data=image_data,
                multimodal_tokens=self.image_only_mm_tokens,
            )

            processor_output = self.process_mm_data(
                input_text=base_output.input_text,
                images=base_output.images,
                videos=normalized_video_data,
            )
            input_ids = torch.as_tensor(processor_output["input_ids"], dtype=torch.long)
            attention_mask = processor_output.get("attention_mask")
            if attention_mask is not None:
                attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)
            grid_thw = processor_output.get("grid_thw")
            if grid_thw is not None:
                grid_thw = torch.as_tensor(grid_thw, dtype=torch.long)
            media_nums_per_sample = processor_output.get("media_nums_per_sample")
            visible_frame_counts = self._compute_visible_frame_counts(
                processor_output.get("cross_attention_mask")
            )

            (
                mrope_positions,
                mrope_position_delta,
                vision_position_ids,
                vision_token_info,
            ) = self._compute_position_metadata(
                input_ids=input_ids,
                attention_mask=attention_mask,
                grid_thw=grid_thw,
                media_nums_per_sample=media_nums_per_sample,
            )

            input_ids = input_ids.flatten()
            mm_items = self._build_mm_items(processor_output, input_ids)
            if mm_items and vision_token_info:
                mm_items[0].set("vision_token_info", vision_token_info[0])

            if SGL_USE_CUDA_IPC:
                for item in mm_items:
                    if isinstance(item.feature, torch.Tensor) and item.feature.is_cuda:
                        sync_flag, available_slice = (
                            self.cudaipc_mmfeature_pool.return_a_slice_tensor_with_flag(
                                item.feature
                            )
                        )
                        if isinstance(available_slice, torch.Tensor):
                            available_slice.copy_(
                                item.feature.reshape(-1).view(torch.int8),
                                non_blocking=True,
                            )
                            item.feature = CudaIpcTensorTransportProxy(
                                data=available_slice,
                                info_data=item.feature,
                                sync_buffer_meta=sync_flag,
                            )
                    elif (
                        isinstance(item.precomputed_embeddings, torch.Tensor)
                        and item.precomputed_embeddings.is_cuda
                    ):
                        sync_flag, available_slice = (
                            self.cudaipc_mmfeature_pool.return_a_slice_tensor_with_flag(
                                item.precomputed_embeddings
                            )
                        )
                        if isinstance(available_slice, torch.Tensor):
                            flattened = item.precomputed_embeddings.reshape(-1)
                            available_slice.copy_(
                                flattened.view(torch.int8),
                                non_blocking=True,
                            )
                            item.precomputed_embeddings = CudaIpcTensorTransportProxy(
                                data=available_slice,
                                info_data=item.precomputed_embeddings,
                                sync_buffer_meta=sync_flag,
                            )

            return MultimodalProcessorOutput(
                input_ids=input_ids.tolist(),
                mm_items=mm_items,
                im_token_id=self.image_token_id,
                mrope_positions=mrope_positions.squeeze(1),
                mrope_position_delta=mrope_position_delta,
                media_nums_per_sample=media_nums_per_sample,
                vision_position_ids=(
                    vision_position_ids.squeeze(1)
                    if vision_position_ids is not None
                    else None
                ),
                visible_frame_counts=visible_frame_counts,
            )
        finally:
            for temp_path in temp_video_paths:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass
