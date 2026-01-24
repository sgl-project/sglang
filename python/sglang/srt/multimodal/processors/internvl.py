# Adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_intern_vit.py

import logging
from functools import lru_cache
from typing import List

import numpy as np
import torch
from decord import VideoReader, cpu, gpu
from PIL import Image

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.interns1 import InternS1ForConditionalGeneration
from sglang.srt.models.internvl import InternVLChatModel
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

logger = logging.getLogger(__name__)


class InternVLProcessor(BaseMultimodalProcessor):
    models = [InternVLChatModel, InternS1ForConditionalGeneration]

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    IMAGE_MAX_NUM = 12

    DEFAULT_VIDEO_NUM_FRAMES = 32
    VIDEO_MAX_NUM = 1
    VIDEO_USE_THUMBNAIL = False

    CONTEXT_FALLBACK = 40960
    CONTEXT_RESERVED = 256

    # OpenAI multimodal placeholder tokens
    IMAGE_PLACEHOLDER_TOKEN = "<image>"
    VIDEO_PLACEHOLDER_TOKEN = "<video>"

    IMG_START = "<img>"
    IMG_END = "</img>"
    IMG_CONTEXT = "<IMG_CONTEXT>"

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_normalize_tensors(device="cuda", dtype=torch.float32):
        mean = torch.tensor(
            InternVLProcessor.IMAGENET_MEAN, device=device, dtype=dtype
        ).view(-1, 1, 1)
        std = torch.tensor(
            InternVLProcessor.IMAGENET_STD, device=device, dtype=dtype
        ).view(-1, 1, 1)
        return mean, std

    def __init__(self, hf_config, server_args, _image_processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _image_processor, *args, **kwargs)

        image_size = (
            getattr(hf_config, "force_image_size", None)
            or hf_config.vision_config.image_size
        )
        patch_size = hf_config.vision_config.patch_size
        if isinstance(image_size, list):
            image_size = image_size[0]
        if isinstance(patch_size, list):
            patch_size = patch_size[0]

        if hasattr(self._processor, "tokenizer"):
            tokenizer = self._processor.tokenizer
        else:
            tokenizer = self._processor
        self.tokenizer = tokenizer

        llm_arch = hf_config.llm_config.architectures[0]
        self.llm_arch = llm_arch
        video_token_map = {
            "Qwen2ForCausalLM": "<|video_pad|>",
            "Qwen3ForCausalLM": "<|video_pad|>",
            "Qwen3MoeForCausalLM": "<|video_pad|>",
            "GptOssForCausalLM": "<|reserved_200000|>",
        }
        self.VIDEO_CONTEXT_TOKEN = video_token_map.get(llm_arch, None)
        self.video_token_id = (
            tokenizer.convert_tokens_to_ids(self.VIDEO_CONTEXT_TOKEN)
            if self.VIDEO_CONTEXT_TOKEN
            else None
        )

        self.image_token_id = (
            tokenizer.convert_tokens_to_ids(self.IMG_CONTEXT)
            if self.IMG_CONTEXT
            else None
        )
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (hf_config.downsample_ratio**2)
        )

        self.img_start_token_id = tokenizer.convert_tokens_to_ids(self.IMG_START)
        self.img_end_token_id = tokenizer.convert_tokens_to_ids(self.IMG_END)

        # Placeholder token use <image>/<video>
        # Offset token id use IMG_CONTEXT / VIDEO_CONTEXT
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.IMAGE_PLACEHOLDER_TOKEN,
            image_token_id=self.image_token_id,
            video_token=self.VIDEO_PLACEHOLDER_TOKEN,
            video_token_id=self.video_token_id,
        ).build(_image_processor)

        # Cache token id for IMG_CONTEXT (used by both branches)
        self.img_context_token_id = tokenizer.convert_tokens_to_ids(self.IMG_CONTEXT)

        # InternLM2 legacy multimodal tokens: use <IMG_CONTEXT> as placeholder
        self.mm_tokens_internlm2 = MultimodalSpecialTokens(
            image_token=self.IMG_CONTEXT,
            image_token_id=self.img_context_token_id,
        ).build(_image_processor)

        self.max_context_len = (
            getattr(server_args, "context_length", None)
            or getattr(server_args, "max_context_len", None)
            or getattr(hf_config, "max_position_embeddings", None)
            or getattr(
                getattr(hf_config, "llm_config", None), "max_position_embeddings", None
            )
            or self.CONTEXT_FALLBACK
        )

    @staticmethod
    def dynamic_preprocess(
        tensor, image_size=448, max_num=IMAGE_MAX_NUM, use_thumbnail=False
    ):
        # Tensor: (C,H,W) float on GPU
        C, H, W = tensor.shape
        aspect_ratio = W / H

        # Generate all possible aspect ratios
        target_ratios = set(
            (i, j)
            for n in range(1, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # Find closest ratio
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)

        for x, y in target_ratios:
            target_ar = x / y
            diff = abs(aspect_ratio - target_ar)
            blocks = x * y
            best_blocks = best_ratio[0] * best_ratio[1]

            if diff < best_ratio_diff:
                best_ratio_diff = diff
                best_ratio = (x, y)
            elif diff == best_ratio_diff and blocks > best_blocks:
                best_ratio = (x, y)

        target_w, target_h = image_size * best_ratio[0], image_size * best_ratio[1]
        blocks = best_ratio[0] * best_ratio[1]

        # Resize on GPU
        resized = torch.nn.functional.interpolate(
            tensor.unsqueeze(0),
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)

        # Split into tiles
        tiles = []
        for i in range(blocks):
            x = (i % best_ratio[0]) * image_size
            y = (i // best_ratio[0]) * image_size
            tile = resized[:, y : y + image_size, x : x + image_size]
            tiles.append(tile)

        # Add thumbnail if needed
        if use_thumbnail and len(tiles) > 1:
            thumb = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                size=(image_size, image_size),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)
            tiles.append(thumb)

        return torch.stack(tiles).to(torch.bfloat16)

    @staticmethod
    def _open_video_reader(path: str) -> VideoReader:
        try:
            return VideoReader(path, ctx=gpu(0), num_threads=1)
        except (RuntimeError, OSError) as e:
            logger.warning(
                "[internvl] VideoReader gpu decode failed (%s), fallback CPU", e
            )
            return VideoReader(path, ctx=cpu(0), num_threads=1)

    def _ensure_placeholders_before_assistant(
        self, prompt: str, placeholder: str, want: int
    ) -> str:
        if want <= 0:
            return prompt
        have = (prompt or "").count(placeholder)
        missing = want - have
        if missing <= 0:
            return prompt

        insert = "\n" + "\n".join([placeholder] * missing) + "\n"

        marker = "<|im_start|>assistant"
        idx = (prompt or "").rfind(marker)
        if idx != -1:
            return (prompt or "")[:idx] + insert + (prompt or "")[idx:]
        return (prompt or "") + insert

    def _token_len(self, text: str) -> int:
        try:
            ids = self.tokenizer(text, return_tensors="pt")["input_ids"].flatten()
            return int(ids.numel())
        except Exception:
            return 0

    def _resolve_video_num_frames(
        self, *, requested: int, num_videos: int, text_len: int, image_tile_cnt: int
    ) -> int:
        if num_videos <= 0:
            return 0
        if not self.VIDEO_CONTEXT_TOKEN or not self.video_token_id:
            return 0
        image_tokens = image_tile_cnt * self.num_image_token
        budget = (
            int(self.max_context_len)
            - int(text_len)
            - int(image_tokens)
            - int(self.CONTEXT_RESERVED)
        )
        if budget <= 0:
            return 1
        max_total_frames = max(1, budget // self.num_image_token)
        frames_per_video = max(1, max_total_frames // max(num_videos, 1))
        return max(1, min(int(requested), int(frames_per_video)))

    async def process_mm_data_async(
        self, image_data, input_text, request_obj, **kwargs
    ):

        is_internlm2 = self.llm_arch == "InternLM2ForCausalLM"

        if is_internlm2:
            return await self.process_internlm2_mm_data_async(
                image_data=image_data,
                input_text=input_text,
                request_obj=request_obj,
                **kwargs,
            )
        else:
            # Default branch uses OpenAI-style placeholders
            return await self.process_qwen_mm_data_async(
                image_data=image_data,
                input_text=input_text,
                request_obj=request_obj,
                **kwargs,
            )

    async def process_qwen_mm_data_async(
        self, image_data, input_text, request_obj, **kwargs
    ):

        img_max_num = (
            getattr(request_obj, "image_max_dynamic_patch", None)
            or getattr(request_obj, "max_dynamic_patch", None)
            or kwargs.get("image_max_dynamic_patch")
            or kwargs.get("max_dynamic_patch")
            or self.IMAGE_MAX_NUM
        )
        img_max_num = max(1, int(img_max_num))

        vid_max_num = (
            getattr(request_obj, "video_max_dynamic_patch", None)
            or getattr(request_obj, "max_dynamic_patch", None)
            or kwargs.get("video_max_dynamic_patch")
            or kwargs.get("max_dynamic_patch")
            or self.VIDEO_MAX_NUM
        )
        vid_max_num = max(1, int(vid_max_num))

        # Qwen/Qwen3 branch: OpenAI-style placeholders <image>/<video>
        prompt = input_text or ""
        video_data = getattr(request_obj, "video_data", None) or []

        if image_data:
            prompt = self._ensure_placeholders_before_assistant(
                prompt, self.IMAGE_PLACEHOLDER_TOKEN, len(image_data)
            )
        if video_data:
            prompt = self._ensure_placeholders_before_assistant(
                prompt, self.VIDEO_PLACEHOLDER_TOKEN, len(video_data)
            )

        logger.info(
            "[internvl][qwen] placeholders image=%d video=%d",
            prompt.count(self.IMAGE_PLACEHOLDER_TOKEN),
            prompt.count(self.VIDEO_PLACEHOLDER_TOKEN),
        )

        base_output = self.load_mm_data(
            prompt=prompt,
            image_data=image_data,
            video_data=video_data,
            multimodal_tokens=self.mm_tokens,  # expects <image>/<video>
            discard_alpha_channel=True,
        )

        logger.info(
            "[internvl][qwen] loaded images=%d videos=%d",
            len(base_output.images),
            len(base_output.videos),
        )

        mean, std = self._get_normalize_tensors(device="cuda")

        # ----- Images -> tiles -----
        num_patches_list: List[int] = []
        pixel_values_list: List[torch.Tensor] = []

        for image in base_output.images:
            if isinstance(image, Image.Image):
                img_np = np.array(image.convert("RGB"))
                tensor = (
                    torch.from_numpy(img_np).permute(2, 0, 1).cuda().float() / 255.0
                )
            else:
                tensor = image.cuda()

            tensor = (tensor - mean) / std
            tiles = self.dynamic_preprocess(
                tensor, image_size=448, max_num=img_max_num, use_thumbnail=True
            )
            pixel_values_list.append(tiles)
            num_patches_list.append(int(tiles.shape[0]))

        if image_data and not pixel_values_list:
            raise ValueError(
                "[internvl][qwen] image_data provided but no images parsed from prompt placeholders"
            )

        image_tensor = (
            torch.cat(pixel_values_list, dim=0) if pixel_values_list else None
        )

        # ----- Videos -> frame tiles (optional) -----
        video_tensor = None
        video_patch_lists = []
        video_pixel_values = []

        requested_frames = int(
            kwargs.get("video_num_frames", self.DEFAULT_VIDEO_NUM_FRAMES)
        )
        num_frames = self._resolve_video_num_frames(
            requested=requested_frames,
            num_videos=len(base_output.videos),
            text_len=self._token_len(base_output.input_text or prompt),
            image_tile_cnt=int(sum(num_patches_list)) if num_patches_list else 0,
        )

        if base_output.videos and num_frames > 0 and self.video_token_id is not None:
            for video in base_output.videos:
                vr = (
                    video
                    if isinstance(video, VideoReader)
                    else self._open_video_reader(str(video))
                )
                max_frame = len(vr) - 1
                frame_indices = (
                    [0]
                    if num_frames == 1
                    else np.linspace(0, max_frame, num=num_frames, dtype=int).tolist()
                )

                per_video_tiles = []
                per_video_patch_cnt = []
                for fi in frame_indices:
                    frame = vr[int(fi)]
                    img_np = (
                        frame.asnumpy()
                        if hasattr(frame, "asnumpy")
                        else np.array(frame)
                    )
                    frame_t = (
                        torch.from_numpy(img_np).permute(2, 0, 1).cuda().float() / 255.0
                    )
                    frame_t = (frame_t - mean) / std

                    tiles = self.dynamic_preprocess(
                        frame_t,
                        image_size=448,
                        max_num=vid_max_num,
                        use_thumbnail=self.VIDEO_USE_THUMBNAIL,
                    )
                    per_video_tiles.append(tiles)
                    per_video_patch_cnt.append(int(tiles.shape[0]))

                pv = torch.cat(per_video_tiles, dim=0)
                video_pixel_values.append(pv)
                video_patch_lists.append(per_video_patch_cnt)

            video_tensor = (
                torch.cat(video_pixel_values, dim=0) if video_pixel_values else None
            )

        # ----- Build prompt text with <img> + CONTEXT*n + </img> -----
        img_ph = "<<<__IMG_PLACEHOLDER__>>>"
        vid_ph = "<<<__VID_PLACEHOLDER__>>>"

        input_text_mid = base_output.input_text or prompt
        input_text_mid = input_text_mid.replace(self.IMAGE_PLACEHOLDER_TOKEN, img_ph)
        input_text_mid = input_text_mid.replace(self.IMG_CONTEXT, img_ph)

        if self.VIDEO_CONTEXT_TOKEN and self.video_token_id is not None:
            input_text_mid = input_text_mid.replace(
                self.VIDEO_PLACEHOLDER_TOKEN, vid_ph
            )
        else:
            input_text_mid = input_text_mid.replace(self.VIDEO_PLACEHOLDER_TOKEN, "")

        input_text_updated = input_text_mid

        # Expand images
        for num_patches in num_patches_list:
            image_tokens = (
                self.IMG_START
                + (self.IMG_CONTEXT * (self.num_image_token * int(num_patches)))
                + self.IMG_END
            )
            input_text_updated = input_text_updated.replace(img_ph, image_tokens, 1)

        # Expand videos (each frame is one <img>...</img>)
        if video_patch_lists and self.VIDEO_CONTEXT_TOKEN:
            for frame_patch_list in video_patch_lists:
                frame_lines = []
                for i, patch_cnt in enumerate(frame_patch_list):
                    ctx_cnt = int(self.num_image_token) * int(patch_cnt)
                    frame_tokens = (
                        self.IMG_START
                        + (self.VIDEO_CONTEXT_TOKEN * ctx_cnt)
                        + self.IMG_END
                    )
                    frame_lines.append(f"Frame {i+1}: {frame_tokens}")
                video_tokens = "\n".join(frame_lines) + "\n"
                input_text_updated = input_text_updated.replace(vid_ph, video_tokens, 1)

        # Tokenize
        input_ids_tensor = self.tokenizer(input_text_updated, return_tensors="pt")[
            "input_ids"
        ].flatten()
        input_ids = input_ids_tensor.tolist()

        # Offsets
        image_offsets = []
        if image_tensor is not None:
            image_offsets = self.get_mm_items_offset(
                input_ids=input_ids_tensor.to("cuda"),
                mm_token_id=self.img_context_token_id,
            )

        video_offsets = []
        if video_tensor is not None and self.video_token_id is not None:
            video_offsets = self.get_mm_items_offset(
                input_ids=input_ids_tensor.to("cuda"),
                mm_token_id=self.video_token_id,
            )

        items = []
        if image_tensor is not None:
            items.append(
                MultimodalDataItem(
                    feature=image_tensor, modality=Modality.IMAGE, offsets=image_offsets
                )
            )
        if video_tensor is not None:
            items.append(
                MultimodalDataItem(
                    feature=video_tensor, modality=Modality.VIDEO, offsets=video_offsets
                )
            )

        return {
            "input_ids": input_ids,
            "mm_items": items,
            "im_start_id": self.img_start_token_id,
            "im_end_id": self.img_end_token_id,
            "im_token_id": self.img_context_token_id,
            "video_token_id": self.video_token_id,
        }

    async def process_internlm2_mm_data_async(
        self, image_data, input_text, request_obj, **kwargs
    ):
        # InternLM2 branch: legacy placeholder <IMG_CONTEXT> (stable for InternLM2 prompt behavior)
        prompt = input_text or ""
        video_data = getattr(request_obj, "video_data", None) or []
        if video_data:
            logger.warning(
                "[internvl][internlm2] video input ignored for InternLM2 branch"
            )

        # Convert any OpenAI-style <image> into <IMG_CONTEXT>
        prompt = prompt.replace(self.IMAGE_PLACEHOLDER_TOKEN, self.IMG_CONTEXT)

        if image_data:
            prompt = self._ensure_placeholders_before_assistant(
                prompt, self.IMG_CONTEXT, len(image_data)
            )

        logger.info(
            "[internvl][internlm2] placeholders img_context=%d",
            prompt.count(self.IMG_CONTEXT),
        )

        base_output = self.load_mm_data(
            prompt=prompt,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens_internlm2,  # expects <IMG_CONTEXT>
            discard_alpha_channel=True,
        )

        mean, std = self._get_normalize_tensors(device="cuda")

        num_patches_list: List[int] = []
        pixel_values_list: List[torch.Tensor] = []

        for image in base_output.images:
            if isinstance(image, Image.Image):
                img_np = np.array(image.convert("RGB"))
                tensor = (
                    torch.from_numpy(img_np).permute(2, 0, 1).cuda().float() / 255.0
                )
            else:
                tensor = image.cuda()

            tensor = (tensor - mean) / std
            tiles = self.dynamic_preprocess(
                tensor, image_size=448, max_num=12, use_thumbnail=True
            )
            pixel_values_list.append(tiles)
            num_patches_list.append(int(tiles.shape[0]))

        if image_data and not pixel_values_list:
            raise ValueError(
                "[internvl][internlm2] image_data provided but no images parsed from prompt placeholders"
            )

        pixel_values = (
            torch.cat(pixel_values_list, dim=0) if pixel_values_list else None
        )

        # Expand each <IMG_CONTEXT> into <img> + <IMG_CONTEXT>*N + </img>
        ph = "<<<__IMG_CONTEXT_PLACEHOLDER__>>>"
        input_text_base = (base_output.input_text or prompt).replace(
            self.IMG_CONTEXT, ph
        )

        input_text_updated = input_text_base
        for num_patches in num_patches_list:
            image_tokens = (
                self.IMG_START
                + (self.IMG_CONTEXT * (self.num_image_token * int(num_patches)))
                + self.IMG_END
            )
            input_text_updated = input_text_updated.replace(ph, image_tokens, 1)

        # Tokenize
        input_ids_tensor = self.tokenizer(input_text_updated, return_tensors="pt")[
            "input_ids"
        ].flatten()
        input_ids = input_ids_tensor.tolist()

        # Offsets
        image_offsets = []
        if pixel_values is not None:
            image_offsets = self.get_mm_items_offset(
                input_ids=input_ids_tensor.to("cuda"),
                mm_token_id=self.img_context_token_id,
            )

        items = []
        if pixel_values is not None:
            items.append(
                MultimodalDataItem(
                    feature=pixel_values, modality=Modality.IMAGE, offsets=image_offsets
                )
            )

        return {
            "input_ids": input_ids,
            "mm_items": items,
            "im_start_id": self.img_start_token_id,
            "im_end_id": self.img_end_token_id,
            "im_token_id": self.img_context_token_id,
            "video_token_id": self.video_token_id,
        }
