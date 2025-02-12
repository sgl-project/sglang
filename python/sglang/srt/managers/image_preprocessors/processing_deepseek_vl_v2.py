# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from dataclasses import dataclass
from typing import Dict, Tuple, List, Literal, Optional
import math

import torch
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as T
from transformers import LlamaTokenizerFast
from transformers.processing_utils import ProcessorMixin
from PIL import Image, ImageOps


def select_best_resolution(image_size, candidate_resolutions):
    # used for cropping
    original_width, original_height = image_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in candidate_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(
            original_width * scale), int(original_height * scale)
        effective_resolution = min(
            downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
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
    sft_format: str
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
            normalize: bool = True
    ):
        self.mean = mean
        self.std = std
        self.normalize = normalize

        transform_pipelines = [
            T.ToTensor()
        ]

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
            mean=image_mean, std=image_std, normalize=normalize)
        self.tokenizer = tokenizer
        # must set this，padding side with make a difference in batch inference
        self.tokenizer.padding_side = 'left'

        # add the pad_token as special token to use 'tokenizer.pad_token' and 'tokenizer.pad_token_id'
        if tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': pad_token})

        # add image token
        image_token_id = self.tokenizer.vocab.get(image_token)
        if image_token_id is None:
            special_tokens = [image_token]
            special_tokens_dict = {"additional_special_tokens": special_tokens}
            self.tokenizer.add_special_tokens(special_tokens_dict)
        self.image_token_id = self.tokenizer.vocab.get(image_token)

        # add five special tokens for grounding-related tasks
        # <|ref|>, <|/ref|>, <|det|>, <|/det|>, <|grounding|>
        special_tokens = ['<|ref|>', '<|/ref|>',
                          '<|det|>', '<|/det|>', '<|grounding|>']
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

    def format_messages_v2(self, messages, pil_images, systems=None):
        """play the role of format_messages_v2 and get_images_info in the last version"""
        tokenized_data = []
        masked_tokenized_data = []  # labels
        images_list = []
        images_seq_mask = []
        images_spatial_crop = []

        image_index = 0
        image_token_cnt = messages.count(self.image_token_id)
        tokenized_str, images, seq_mask, spatial_crop = self.tokenize_with_images(
            messages,
            pil_images[image_index:image_index+image_token_cnt],
            bos=False,
            eos=True,
            cropping=len(pil_images) <= 2
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
            images_seq_mask), f"format_messages_v2: tokenized_str's length {len(tokenized_str)} is not equal to imags_seq_mask's length {len(images_seq_mask)}"

        return tokenized_data, masked_tokenized_data, images_list, images_seq_mask, images_spatial_crop

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

        tokenized_str, masked_tokenized_str, images_list, images_seq_mask, images_spatial_crop = self.format_messages_v2(
            conversations, images)

        assert len(tokenized_str) == len(images_seq_mask) == len(masked_tokenized_str), \
            (f"tokenized_str's length {len(tokenized_str)}, input_ids' length {len(masked_tokenized_str)}, "
             f"imags_seq_mask's length {len(images_seq_mask)}, are not equal")

        input_ids = torch.LongTensor(tokenized_str)
        target_ids = torch.LongTensor(masked_tokenized_str)
        images_seq_mask = torch.tensor(images_seq_mask, dtype=torch.bool)

        # set input_ids < 0 | input_ids == self.image_token_id as ignore_id
        target_ids[(input_ids < 0) | (
            input_ids == self.image_token_id)] = self.ignore_id
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
            images_spatial_crop = torch.tensor(
                images_spatial_crop, dtype=torch.long)

        prepare = VLChatProcessorOutput(
            sft_format=sft_format,
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
            **kwargs,
    ):
        prepare = self.process_one(
            prompt=prompt,
            conversations=conversations,
            images=images,
            apply_sft_format=apply_sft_format,
            inference_mode=inference_mode,
            system_prompt=system_prompt
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
    ):
        """Tokenize text with <image> tags."""
        images_list, images_seq_mask, images_spatial_crop = [], [], []
        tokenized_str = []
        split_idx = self.find_all_indices(conversation, 100003)
        for idx, image in enumerate(images):
            """encode text_sep"""
            if idx == 0:
                tokenized_sep = conversation[0:split_idx[idx]]
            else:
                tokenized_sep = conversation[split_idx[idx-1]+1:split_idx[idx]]
            tokenized_str += tokenized_sep
            images_seq_mask += [False] * len(tokenized_sep)

            """select best resolution for anyres"""
            if cropping:
                best_width, best_height = select_best_resolution(
                    image.size, self.candidate_resolutions)
            else:
                best_width, best_height = self.image_size, self.image_size
            # print(image.size, (best_width, best_height)) # check the select_best_resolutions func

            """process the global view"""
            global_view = ImageOps.pad(image, (self.image_size, self.image_size),
                                       color=tuple(int(x * 255) for x in self.image_transform.mean))
            images_list.append(self.image_transform(global_view))

            """process the local views"""
            local_view = ImageOps.pad(image, (best_width, best_height),
                                      color=tuple(int(x * 255) for x in self.image_transform.mean))
            for i in range(0, best_height, self.image_size):
                for j in range(0, best_width, self.image_size):
                    images_list.append(
                        self.image_transform(local_view.crop((j, i, j + self.image_size, i + self.image_size))))

            """record height / width crop num"""
            num_width_tiles, num_height_tiles = best_width // self.image_size, best_height // self.image_size
            images_spatial_crop.append([num_width_tiles, num_height_tiles])

            """add image tokens"""
            h = w = math.ceil(
                (self.image_size // self.patch_size) / self.downsample_ratio)
            # global views tokens h * (w + 1), 1 is for line seperator
            tokenized_image = [self.image_token_id] * h * (w + 1)
            # add a seperator between global and local views
            tokenized_image += [self.image_token_id]
            # local views tokens, (num_height_tiles * h) * (num_width_tiles * w + 1)
            tokenized_image += [self.image_token_id] * \
                (num_height_tiles * h) * (num_width_tiles * w + 1)

            tokenized_str += tokenized_image
            images_seq_mask += [True] * len(tokenized_image)
            # print(width_crop_num, height_crop_num, len(tokenized_image)) # test the correctness of the number of image-related tokens

        """process the last text split"""
        # tokenized_sep = self.encode(text_splits[-1], bos=False, eos=False)
        tokenized_sep = conversation[split_idx[idx-1]+1:]
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
            images_seq_mask), f"tokenize_with_images func: tokenized_str's length {len(tokenized_str)} is not equal to imags_seq_mask's length {len(images_seq_mask)}"

        return tokenized_str, images_list, images_seq_mask, images_spatial_crop
