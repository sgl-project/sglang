"""Standalone UNLIMITED-OCR configuration and HF processor."""

import math
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image, ImageOps
from transformers import (
    AutoConfig,
    AutoProcessor,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    ProcessorMixin,
)

from sglang.srt.configs.deepseek_ocr import (
    ImageTransform,
    MlpProjectorConfig,
    VisionEncoderConfig,
    VLChatProcessorOutput,
    find_closest_aspect_ratio,
)
from sglang.srt.multimodal.customized_mm_processor_utils import (
    register_customized_processor,
)


def dynamic_preprocess(
    image, min_num=2, max_num=32, image_size=640, use_thumbnail=False
):
    """Split an image into tiles based on the best-matching aspect ratio."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images, target_aspect_ratio


class UnlimitedOCRHFProcessor(ProcessorMixin):
    """HuggingFace-style processor for UNLIMITED-OCR (OCR mode)."""

    tokenizer_class = "PreTrainedTokenizerFast"
    attributes = ["tokenizer"]

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        candidate_resolutions: Tuple[Tuple[int, int]],
        patch_size: int,
        downsample_ratio: int,
        image_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize: bool = True,
        image_token: str = "<image>",
        pad_token: str = "<｜▁pad▁｜>",
        add_special_token: bool = False,
        sft_format: str = "unlimitedocr",
        mask_prompt: bool = True,
        ignore_id: int = -100,
        base_size: int = 1024,
        image_size: int = 640,
        crop_mode: bool = True,
        **kwargs,
    ):
        """Initialize tokenizer, image transform, and special tokens."""
        self.candidate_resolutions = candidate_resolutions
        self.base_size = base_size
        self.image_size = image_size
        self.crop_mode = crop_mode
        self.patch_size = patch_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.normalize = normalize
        self.downsample_ratio = downsample_ratio
        self.image_transform = ImageTransform(
            mean=image_mean, std=image_std, normalize=normalize
        )
        if type(tokenizer) is not PreTrainedTokenizerFast:
            tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer.name_or_path)
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"

        if tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": pad_token})

        image_token_id = self.tokenizer.vocab.get(image_token)
        if image_token_id is None:
            special_tokens = [image_token]
            special_tokens_dict = {"additional_special_tokens": special_tokens}
            self.tokenizer.add_special_tokens(special_tokens_dict)
        self.image_token_id = self.tokenizer.vocab.get(image_token)

        special_tokens = ["<|ref|>", "<|/ref|>", "<|det|>", "<|/det|>", "<|grounding|>"]
        special_tokens_dict = {"additional_special_tokens": special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)

        special_tokens = ["<|User|>", "<|Assistant|>"]
        special_tokens_dict = {"additional_special_tokens": special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)

        self.image_token = image_token
        self.pad_token = pad_token
        self.add_special_token = add_special_token
        self.sft_format = sft_format
        self.mask_prompt = mask_prompt
        self.ignore_id = ignore_id

        super().__init__(tokenizer, **kwargs)

    def format_messages_v2(
        self,
        messages: str,
        pil_images,
        max_req_input_len=-1,
        base_size: int = None,
        image_size: int = None,
        crop_mode: bool = None,
    ):
        """Tokenize messages with embedded images and return processed tensors."""
        base_size = base_size or self.base_size
        image_size = image_size or self.image_size
        crop_mode = crop_mode if crop_mode is not None else self.crop_mode

        tokenized_data = []
        masked_tokenized_data = []
        images_list = []
        images_seq_mask = []
        images_spatial_crop = []

        image_index = 0
        image_token_cnt = messages.count(self.image_token)
        (
            input_ids,
            images,
            images_crop,
            seq_mask,
            spatial_crop,
            num_image_tokens,
            image_shapes,
        ) = self.tokenize_with_images(
            messages,
            pil_images[image_index : image_index + image_token_cnt],
            bos=True,
            eos=True,
            cropping=crop_mode,
            base_size=base_size,
            image_size=image_size,
        )

        image_index = image_token_cnt
        images_list += images
        images_seq_mask += seq_mask
        images_spatial_crop = spatial_crop

        return (
            input_ids,
            masked_tokenized_data,
            images_list,
            images_seq_mask,
            images_spatial_crop,
            images_crop,
        )

    @property
    def bos_id(self):
        """Return the beginning-of-sequence token ID."""
        return self.tokenizer.bos_token_id

    @property
    def eos_id(self):
        """Return the end-of-sequence token ID."""
        return self.tokenizer.eos_token_id

    @property
    def pad_id(self):
        """Return the padding token ID."""
        return self.tokenizer.pad_token_id

    def encode(self, text: str, bos: bool = True, eos: bool = False):
        """Encode text into token IDs with optional BOS/EOS."""
        t = self.tokenizer.encode(text, add_special_tokens=False)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int], **kwargs) -> str:
        """Decode token IDs back into a string."""
        return self.tokenizer.decode(t, **kwargs)

    def process_one(
        self,
        prompt: str = None,
        conversations: List[Dict[str, str]] = None,
        images: List[Image.Image] = None,
        apply_sft_format: bool = False,
        inference_mode: bool = True,
        system_prompt: str = "",
        max_req_input_len: int = -1,
        base_size: int = None,
        image_size: int = None,
        crop_mode: bool = None,
        **kwargs,
    ):
        """Process a single prompt with images into model-ready tensors."""
        base_size = base_size or self.base_size
        image_size = image_size or self.image_size
        crop_mode = crop_mode if crop_mode is not None else self.crop_mode

        prompt = conversations or prompt
        (
            input_ids,
            masked_tokenized_str,
            images_list,
            images_seq_mask,
            images_spatial_crop,
            images_crop,
        ) = self.format_messages_v2(
            prompt,
            images,
            max_req_input_len,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
        )

        target_ids = torch.LongTensor(masked_tokenized_str)

        has_images = len(images_list) > 0
        has_local_crops = []
        if len(images_spatial_crop) > 0:
            has_local_crops = [
                (crop[0] > 1 or crop[1] > 1).item() for crop in images_spatial_crop
            ]

        if len(images_list) == 0:
            images = torch.zeros((1, 3, image_size, image_size))
        else:
            images = torch.stack(images_list, dim=0)

        images_spatial_crop = torch.stack([images_spatial_crop], dim=0)

        prepare = VLChatProcessorOutput(
            input_ids=input_ids,
            target_ids=target_ids,
            images_crop=images_crop,
            pixel_values=images,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
        )
        prepare.has_images = has_images
        prepare.has_local_crops = has_local_crops

        return prepare

    def __call__(
        self,
        *,
        prompt: str = None,
        conversations: List[Dict[str, str]] = None,
        images: List[Image.Image] = None,
        apply_sft_format: bool = False,
        inference_mode: bool = True,
        system_prompt: str = "",
        max_req_input_len: int = -1,
        text: list[str] = None,
        base_size: int = None,
        image_size: int = None,
        crop_mode: bool = None,
        **kwargs,
    ):
        """Call the processor to tokenize text and images for inference."""
        assert text is None or isinstance(text, list)
        if text is not None:
            text = text[0]

        prepare = self.process_one(
            prompt=prompt or text,
            conversations=conversations,
            images=images,
            apply_sft_format=apply_sft_format,
            inference_mode=inference_mode,
            system_prompt=system_prompt,
            max_req_input_len=max_req_input_len,
            base_size=base_size if base_size is not None else self.base_size,
            image_size=image_size if image_size is not None else self.image_size,
            crop_mode=crop_mode if crop_mode is not None else self.crop_mode,
        )
        return prepare

    def find_all_indices(self, messages, target_value):
        """Return all indices where target_value appears in messages."""
        indices = []
        for index, item in enumerate(messages):
            if item == target_value:
                indices.append(index)
        return indices

    def tokenize_with_images(
        self,
        conversation: str,
        images: List[Image.Image],
        bos: bool = True,
        eos: bool = True,
        cropping: bool = True,
        base_size: int = None,
        image_size: int = None,
    ):
        """Tokenize text with <image> tags (OCR mode)."""
        base_size = base_size or self.base_size
        image_size = image_size or self.image_size

        assert conversation.count(self.image_token) == len(images)
        text_splits: list[str] = conversation.split(self.image_token)
        images_list, images_crop_list, images_seq_mask, images_spatial_crop = (
            [],
            [],
            [],
            [],
        )
        image_shapes = []
        num_image_tokens = []
        tokenized_str = []

        for text_sep, image in zip(text_splits, images):
            tokenized_sep = self.encode(text_sep, bos=False, eos=False)
            tokenized_str += tokenized_sep
            images_seq_mask += [False] * len(tokenized_sep)
            image_shapes.append(image.size)

            if image.size[0] <= 640 and image.size[1] <= 640:
                crop_ratio = [1, 1]
            else:
                if cropping:
                    images_crop_raw, crop_ratio = dynamic_preprocess(
                        image, image_size=image_size
                    )
                else:
                    crop_ratio = [1, 1]

            if image_size <= 640 and not cropping:
                image = image.resize((image_size, image_size))
            if cropping:
                pad_size = base_size
            else:
                pad_size = image_size

            global_view = ImageOps.pad(
                image,
                (pad_size, pad_size),
                color=tuple(int(x * 255) for x in self.image_transform.mean),
            )
            images_list.append(self.image_transform(global_view))

            num_width_tiles, num_height_tiles = crop_ratio
            images_spatial_crop.append([num_width_tiles, num_height_tiles])

            if num_width_tiles > 1 or num_height_tiles > 1:
                for i in range(len(images_crop_raw)):
                    images_crop_list.append(self.image_transform(images_crop_raw[i]))

            num_queries = math.ceil(
                (image_size // self.patch_size) / self.downsample_ratio
            )
            num_queries_base = math.ceil(
                (base_size // self.patch_size) / self.downsample_ratio
            )
            if cropping:
                tokenized_image = (
                    [self.image_token_id] * num_queries_base + [self.image_token_id]
                ) * num_queries_base
                tokenized_image += [self.image_token_id]
                if num_width_tiles > 1 or num_height_tiles > 1:
                    tokenized_image += (
                        [self.image_token_id] * (num_queries * num_width_tiles)
                        + [self.image_token_id]
                    ) * (num_queries * num_height_tiles)
            else:
                tokenized_image = (
                    [self.image_token_id] * num_queries + [self.image_token_id]
                ) * num_queries
                tokenized_image += [self.image_token_id]

            tokenized_str += tokenized_image
            images_seq_mask += [True] * len(tokenized_image)
            num_image_tokens.append(len(tokenized_image))

        tokenized_sep = self.encode(text_splits[-1], bos=False, eos=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        if bos:
            tokenized_str = [self.bos_id] + tokenized_str
            images_seq_mask = [False] + images_seq_mask
        if eos:
            tokenized_str = tokenized_str + [self.eos_id]
            images_seq_mask = images_seq_mask + [False]

        assert len(tokenized_str) == len(images_seq_mask)

        masked_tokenized_str = []
        for token_index in tokenized_str:
            if token_index != self.image_token_id:
                masked_tokenized_str.append(token_index)
            else:
                masked_tokenized_str.append(self.ignore_id)

        assert len(tokenized_str) == len(images_seq_mask) == len(masked_tokenized_str)

        input_ids = torch.LongTensor(tokenized_str)
        target_ids = torch.LongTensor(masked_tokenized_str)
        images_seq_mask = torch.tensor(images_seq_mask, dtype=torch.bool)

        target_ids[(input_ids < 0) | (input_ids == self.image_token_id)] = (
            self.ignore_id
        )
        input_ids[input_ids < 0] = self.pad_id

        inference_mode = True
        if inference_mode:
            assert input_ids[-1] == self.eos_id
            input_ids = input_ids[:-1]
            target_ids = target_ids[:-1]
            images_seq_mask = images_seq_mask[:-1]

        if len(images_list) == 0:
            pixel_values = torch.zeros((1, 3, base_size, base_size))
            images_spatial_crop = torch.zeros((1, 1), dtype=torch.long)
            images_crop = torch.zeros((1, 3, image_size, image_size)).unsqueeze(0)
        else:
            pixel_values = torch.stack(images_list, dim=0)
            images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long)
            if images_crop_list:
                images_crop = torch.stack(images_crop_list, dim=0).unsqueeze(0)
            else:
                images_crop = torch.zeros(
                    (len(images_list), 3, image_size, image_size)
                ).unsqueeze(1)

        input_ids = input_ids.unsqueeze(0)
        return (
            input_ids,
            pixel_values,
            images_crop,
            images_seq_mask,
            images_spatial_crop,
            num_image_tokens,
            image_shapes,
        )


class UnlimitedLanguageConfig(PretrainedConfig):
    """Configuration for the UNLIMITED language model backbone."""

    model_type = "unlimited_language"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=102400,
        hidden_size=4096,
        intermediate_size=11008,
        moe_intermediate_size=1407,
        num_hidden_layers=30,
        num_attention_heads=32,
        num_key_value_heads=32,
        n_shared_experts=None,
        n_routed_experts=None,
        ep_size=1,
        routed_scaling_factor=1.0,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        v_head_dim=128,
        qk_nope_head_dim=128,
        topk_method="gready",
        n_group=None,
        topk_group=None,
        num_experts_per_tok=None,
        moe_layer_freq=1,
        first_k_dense_replace=0,
        norm_topk_prob=False,
        scoring_func="softmax",
        aux_loss_alpha=0.001,
        seq_aux=True,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=100000,
        eos_token_id=100001,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        use_mla=True,
        **kwargs,
    ):
        """Initialize language model configuration parameters."""
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.ep_size = ep_size
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = float(rms_norm_eps)
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_mla = use_mla

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


@register_customized_processor(processor_class=UnlimitedOCRHFProcessor)
class UnlimitedVLConfig(PretrainedConfig):
    """Top-level vision-language config for UNLIMITED-OCR models."""

    model_type = "unlimited-ocr"
    vision_config: VisionEncoderConfig = None
    projector_config: MlpProjectorConfig = None

    tile_tag: str = "2D"
    global_view_pos: str = "head"
    candidate_resolutions: tuple[tuple[int, int]] = ((384, 384),)
    customized_processor_type: type[Any] = UnlimitedOCRHFProcessor

    def __init__(
        self,
        tile_tag: str = "tile_tag",
        global_view_pos: str = "head",
        candidate_resolutions: tuple[tuple[int, int]] = ((384, 384),),
        **kwargs,
    ):
        """Initialize UNLIMITED VL config with vision, projector, and language sub-configs."""
        super().__init__(**kwargs)

        vision_config = kwargs.get("vision_config", {})
        self.vision_config = VisionEncoderConfig(**vision_config)

        projector_config = kwargs.get("projector_config", {})
        self.projector_config = MlpProjectorConfig(**projector_config)

        language_config = kwargs.get("language_config", {})
        self.text_config = UnlimitedLanguageConfig(**language_config)

        self.tile_tag = tile_tag
        self.global_view_pos = global_view_pos
        self.candidate_resolutions = candidate_resolutions
        self.vocab_size = self.text_config.vocab_size
        self.hidden_size = self.text_config.hidden_size


AutoProcessor.register(UnlimitedVLConfig, UnlimitedOCRHFProcessor)

try:
    AutoConfig.register("unlimited-ocr", UnlimitedVLConfig)
except ValueError:
    pass
