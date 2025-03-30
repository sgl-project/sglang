import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageOps
from transformers import (
    AutoProcessor,
    LlamaTokenizerFast,
    PretrainedConfig,
    ProcessorMixin,
)


def select_best_resolution(image_size, candidate_resolutions):
    # used for cropping
    original_width, original_height = image_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in candidate_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(
            original_height * scale
        )
        effective_resolution = min(
            downscaled_width * downscaled_height, original_width * original_height
        )
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution
            and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


class DictOutput(object):
    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value


@dataclass
class VLChatProcessorOutput(DictOutput):
    input_ids: torch.LongTensor
    target_ids: torch.LongTensor
    images: torch.Tensor
    images_seq_mask: torch.BoolTensor
    images_spatial_crop: torch.LongTensor

    def __len__(self):
        return len(self.input_ids)


class ImageTransform(object):
    def __init__(
        self,
        mean: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        std: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        normalize: bool = True,
    ):
        self.mean = mean
        self.std = std
        self.normalize = normalize

        # only load torchvision.transforms when needed
        try:
            import torchvision.transforms as T

            # FIXME: add version check for gguf
        except ImportError as err:
            raise ImportError(
                "Please install torchvision via `pip install torchvision` to use Deepseek-VL2."
            ) from err

        transform_pipelines = [T.ToTensor()]

        if normalize:
            transform_pipelines.append(T.Normalize(mean, std))

        self.transform = T.Compose(transform_pipelines)

    def __call__(self, pil_img: Image.Image):
        x = self.transform(pil_img)
        return x


class DeepseekVLV2Processor(ProcessorMixin):
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")
    attributes = ["tokenizer"]

    def __init__(
        self,
        tokenizer: LlamaTokenizerFast,
        candidate_resolutions: Tuple[Tuple[int, int]],
        patch_size: int,
        downsample_ratio: int,
        image_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize: bool = True,
        image_token: str = "<image>",
        pad_token: str = "<｜▁pad▁｜>",
        add_special_token: bool = False,
        sft_format: str = "deepseek",
        mask_prompt: bool = True,
        ignore_id: int = -100,
        **kwargs,
    ):

        self.candidate_resolutions = candidate_resolutions
        self.image_size = candidate_resolutions[0][0]
        self.patch_size = patch_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.normalize = normalize
        self.downsample_ratio = downsample_ratio

        self.image_transform = ImageTransform(
            mean=image_mean, std=image_std, normalize=normalize
        )
        self.tokenizer = tokenizer
        # must set this，padding side with make a difference in batch inference
        self.tokenizer.padding_side = "left"

        # add the pad_token as special token to use 'tokenizer.pad_token' and 'tokenizer.pad_token_id'
        if tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": pad_token})

        # add image token
        image_token_id = self.tokenizer.vocab.get(image_token)
        if image_token_id is None:
            special_tokens = [image_token]
            special_tokens_dict = {"additional_special_tokens": special_tokens}
            self.tokenizer.add_special_tokens(special_tokens_dict)
        self.image_token_id = self.tokenizer.vocab.get(image_token)

        # add five special tokens for grounding-related tasks
        # <|ref|>, <|/ref|>, <|det|>, <|/det|>, <|grounding|>
        special_tokens = ["<|ref|>", "<|/ref|>", "<|det|>", "<|/det|>", "<|grounding|>"]
        special_tokens_dict = {"additional_special_tokens": special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)

        # add special tokens for SFT data
        special_tokens = ["<|User|>", "<|Assistant|>"]
        special_tokens_dict = {"additional_special_tokens": special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)

        self.image_token = image_token
        self.pad_token = pad_token
        self.add_special_token = add_special_token
        self.sft_format = sft_format
        self.mask_prompt = mask_prompt
        self.ignore_id = ignore_id

        super().__init__(
            tokenizer,
            **kwargs,
        )

    def format_messages_v2(self, messages, pil_images, max_req_input_len=-1):
        """play the role of format_messages_v2 and get_images_info in the last version"""
        tokenized_data = []
        masked_tokenized_data = []  # labels
        images_list = []
        images_seq_mask = []
        images_spatial_crop = []

        image_index = 0
        image_token_cnt = messages.count(self.image_token)
        tokenized_str, images, seq_mask, spatial_crop = self.tokenize_with_images(
            messages,
            pil_images[image_index : image_index + image_token_cnt],
            bos=False,
            eos=True,
            cropping=len(pil_images) <= 2,
            max_req_input_len=max_req_input_len,
        )

        image_index = image_token_cnt
        tokenized_data += tokenized_str
        if self.mask_prompt:
            masked_tokenized_data += [self.ignore_id] * len(tokenized_str)
        else:
            masked_tokenized_data += tokenized_str
        images_list += images
        images_seq_mask += seq_mask
        images_spatial_crop += spatial_crop

        assert len(tokenized_data) == len(
            images_seq_mask
        ), f"format_messages_v2: tokenized_str's length {len(tokenized_str)} is not equal to imags_seq_mask's length {len(images_seq_mask)}"

        return (
            tokenized_data,
            masked_tokenized_data,
            images_list,
            images_seq_mask,
            images_spatial_crop,
        )

    @property
    def bos_id(self):
        return self.tokenizer.bos_token_id

    @property
    def eos_id(self):
        return self.tokenizer.eos_token_id

    @property
    def pad_id(self):
        return self.tokenizer.pad_token_id

    def encode(self, text: str, bos: bool = True, eos: bool = False):
        t = self.tokenizer.encode(text, add_special_tokens=False)

        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]

        return t

    def decode(self, t: List[int], **kwargs) -> str:
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
        **kwargs,
    ):
        """

        Args:
            prompt (str): the formatted prompt;
            conversations (List[Dict]): conversations with a list of messages;
            images (List[ImageType]): the list of images;
            apply_sft_format (bool): if prompt is not None, then apply the SFT format to prompt;
                if conversations is not None, then it will always apply the SFT format to conversations;
            inference_mode (bool): if True, then remove the last eos token;
            system_prompt (str): the system prompt;
            **kwargs:

        Returns:
            outputs (BaseProcessorOutput): the output of the processor,
                - input_ids (torch.LongTensor): [N + image tokens]
                - target_ids (torch.LongTensor): [N + image tokens]
                - images (torch.FloatTensor): [n_images, 3, H, W]
                - image_id (int): the id of the image token
                - num_image_tokens (List[int]): the number of image tokens
        """

        assert (
            prompt is None or conversations is None
        ), "prompt and conversations cannot be used at the same time."

        (
            tokenized_str,
            masked_tokenized_str,
            images_list,
            images_seq_mask,
            images_spatial_crop,
        ) = self.format_messages_v2(conversations, images, max_req_input_len)

        assert (
            len(tokenized_str) == len(images_seq_mask) == len(masked_tokenized_str)
        ), (
            f"tokenized_str's length {len(tokenized_str)}, input_ids' length {len(masked_tokenized_str)}, "
            f"imags_seq_mask's length {len(images_seq_mask)}, are not equal"
        )

        input_ids = torch.LongTensor(tokenized_str)
        target_ids = torch.LongTensor(masked_tokenized_str)
        images_seq_mask = torch.tensor(images_seq_mask, dtype=torch.bool)

        # set input_ids < 0 | input_ids == self.image_token_id as ignore_id
        target_ids[(input_ids < 0) | (input_ids == self.image_token_id)] = (
            self.ignore_id
        )
        input_ids[input_ids < 0] = self.pad_id

        if inference_mode:
            assert input_ids[-1] == self.eos_id
            input_ids = input_ids[:-1]
            target_ids = target_ids[:-1]
            images_seq_mask = images_seq_mask[:-1]

        if len(images_list) == 0:
            images = torch.zeros((1, 3, self.image_size, self.image_size))
            images_spatial_crop = torch.zeros((1, 2), dtype=torch.long)
        else:
            images = torch.stack(images_list, dim=0)
            images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long)

        prepare = VLChatProcessorOutput(
            input_ids=input_ids,
            target_ids=target_ids,
            images=images,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
        )

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
        **kwargs,
    ):
        prepare = self.process_one(
            prompt=prompt,
            conversations=conversations,
            images=images,
            apply_sft_format=apply_sft_format,
            inference_mode=inference_mode,
            system_prompt=system_prompt,
            max_req_input_len=max_req_input_len,
        )

        return prepare

    def find_all_indices(self, messages, target_value):
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
        max_req_input_len: int = -1,
    ):
        """Tokenize text with <image> tags."""
        images_list, images_seq_mask, images_spatial_crop = [], [], []
        text_splits = conversation.split(self.image_token)
        tokenized_str = []
        for text_sep, image in zip(text_splits, images):
            """encode text_sep"""
            tokenized_sep = self.encode(text_sep, bos=False, eos=False)
            tokenized_str += tokenized_sep
            images_seq_mask += [False] * len(tokenized_sep)

            """select best resolution for anyres"""
            if cropping:
                best_width, best_height = select_best_resolution(
                    image.size, self.candidate_resolutions
                )
            else:
                best_width, best_height = self.image_size, self.image_size
            # print(image.size, (best_width, best_height)) # check the select_best_resolutions func

            """process the global view"""
            global_view = ImageOps.pad(
                image,
                (self.image_size, self.image_size),
                color=tuple(int(x * 255) for x in self.image_transform.mean),
            )
            images_list.append(self.image_transform(global_view))

            """process the local views"""
            local_view = ImageOps.pad(
                image,
                (best_width, best_height),
                color=tuple(int(x * 255) for x in self.image_transform.mean),
            )
            for i in range(0, best_height, self.image_size):
                for j in range(0, best_width, self.image_size):
                    images_list.append(
                        self.image_transform(
                            local_view.crop(
                                (j, i, j + self.image_size, i + self.image_size)
                            )
                        )
                    )

            """record height / width crop num"""
            num_width_tiles, num_height_tiles = (
                best_width // self.image_size,
                best_height // self.image_size,
            )
            images_spatial_crop.append([num_width_tiles, num_height_tiles])

            """add image tokens"""
            h = w = math.ceil(
                (self.image_size // self.patch_size) / self.downsample_ratio
            )
            # global views tokens h * (w + 1), 1 is for line seperator
            tokenized_image = [self.image_token_id] * h * (w + 1)
            # add a seperator between global and local views
            tokenized_image += [self.image_token_id]
            # local views tokens, (num_height_tiles * h) * (num_width_tiles * w + 1)
            tokenized_image += (
                [self.image_token_id]
                * (num_height_tiles * h)
                * (num_width_tiles * w + 1)
            )

            tokenized_str += tokenized_image
            images_seq_mask += [True] * len(tokenized_image)
            # print(width_crop_num, height_crop_num, len(tokenized_image)) # test the correctness of the number of image-related tokens

        """process the last text split"""
        tokenized_sep = self.encode(text_splits[-1], bos=False, eos=False)
        # deal with video, limit with request len
        if max_req_input_len > -1:
            if max_req_input_len < len(tokenized_sep) + len(tokenized_str) - 1:
                rest = max_req_input_len - len(tokenized_sep) - 1 - 1024
                tokenized_str = tokenized_str[:rest]
                images_seq_mask = images_seq_mask[:rest]
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        """add the bos and eos tokens"""
        if bos:
            tokenized_str = [self.bos_id] + tokenized_str
            images_seq_mask = [False] + images_seq_mask
        if eos:
            tokenized_str = tokenized_str + [self.eos_id]
            images_seq_mask = images_seq_mask + [False]

        assert len(tokenized_str) == len(
            images_seq_mask
        ), f"tokenize_with_images func: tokenized_str's length {len(tokenized_str)} is not equal to imags_seq_mask's length {len(images_seq_mask)}"

        return tokenized_str, images_list, images_seq_mask, images_spatial_crop


class DeepseekVL2VisionEncoderConfig(PretrainedConfig):
    model_type: str = "vision"

    model_name: str = "siglip_large_patch16_384"
    image_size: int = 384
    patch_size: int = 16
    width: int = 1024
    layers: int = 24
    heads: int = 16
    mlp_ratio: int = 4
    global_pool: str = "map"
    ignore_head: bool = True
    class_token: bool = False
    num_classes: int = 0
    use_checkpoint: bool = False
    weight_init: str = "skip"
    deterministic: bool = False
    num_recomputing_layers: int = 0

    def __init__(
        self,
        model_name: str = "siglip_large_patch16_384",
        image_size: int = 384,
        patch_size: int = 16,
        width: int = 1024,
        layers: int = 24,
        heads: int = 16,
        mlp_ratio: int = 4,
        global_pool: str = "map",
        ignore_head: bool = True,
        class_token: bool = False,
        num_classes: int = 0,
        use_checkpoint: bool = False,
        **kwargs,
    ):
        self.model_name = model_name
        self.image_size = image_size
        self.patch_size = patch_size
        self.width = width
        self.layers = layers
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.global_pool = global_pool
        self.ignore_head = ignore_head
        self.class_token = class_token
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint

        super().__init__(**kwargs)


class DeepseekVL2MlpProjectorConfig(PretrainedConfig):
    model_type = "mlp_projector"
    projector_type: str = "downsample_mlp_gelu"
    input_dim: int = 1152
    n_embed: int = 2048
    depth: int = 2
    mlp_ratio: int = 1
    downsample_ratio: int = 2
    token_pooling: bool = False

    def __init__(
        self,
        projector_type: str = "downsample_mlp_gelu",
        input_dim: int = 1152,
        n_embed: int = 2048,
        depth: int = 2,
        mlp_ratio: int = 1,
        downsample_ratio: int = 2,
        **kwargs,
    ):
        self.projector_type = projector_type
        self.input_dim = input_dim
        self.n_embed = n_embed
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.downsample_ratio = downsample_ratio

        super().__init__(**kwargs)


class DeepseekV2Config(PretrainedConfig):

    model_type = "deepseek_v2"
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
        # for backward compatibility
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


class DeepseekVL2Config(PretrainedConfig):
    model_type = "deepseek_vl_v2"
    vision_config: DeepseekVL2VisionEncoderConfig
    projector_config: DeepseekVL2MlpProjectorConfig
    language_config: DeepseekV2Config

    tile_tag: str = "2D"
    global_view_pos: str = "head"
    candidate_resolutions: Tuple[Tuple[int, int]] = ((384, 384),)

    def __init__(
        self,
        tile_tag: str = "tile_tag",
        global_view_pos: str = "head",
        candidate_resolutions: Tuple[Tuple[int, int]] = ((384, 384),),
        **kwargs,
    ):
        super().__init__(**kwargs)

        vision_config = kwargs.get("vision_config", {})
        self.vision_config = DeepseekVL2VisionEncoderConfig(**vision_config)

        projector_config = kwargs.get("projector_config", {})
        self.projector_config = DeepseekVL2MlpProjectorConfig(**projector_config)

        language_config = kwargs.get("language_config", {})
        if isinstance(language_config, DeepseekV2Config):
            self.language_config = language_config
        else:
            self.language_config = DeepseekV2Config(**language_config)

        self.tile_tag = tile_tag
        self.global_view_pos = global_view_pos
        self.candidate_resolutions = candidate_resolutions
        self.architectures = ["DeepseekVL2ForCausalLM"]


AutoProcessor.register(DeepseekVL2Config, DeepseekVLV2Processor)
