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
# Adapted from https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/blob/cb5a65ff10232128389d882d805fa609427544f1/configuration.py

from typing import Any

from transformers.configuration_utils import PretrainedConfig

from sglang.srt.configs.nemotron_h import NemotronHConfig
from sglang.srt.configs.radio import RadioConfig
from sglang.srt.multimodal.internvl_utils import IMAGENET_MEAN, IMAGENET_STD


def float_triplet(seq: Any):
    a, b, c = tuple(seq)
    assert (
        isinstance(a, float) and isinstance(b, float) and isinstance(c, float)
    ), "expected three floats"
    return a, b, c


class NemotronH_Nano_VL_V2_Config(PretrainedConfig):
    model_type = "NemotronH_Nano_VL_V2"
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        llm_config=None,
        force_image_size: int = 512,
        patch_size: int = 16,
        downsample_ratio=0.5,
        template=None,
        ps_version="v2",
        image_tag_type="internvl",
        projector_hidden_size=4096,
        vit_hidden_size=1280,
        video_pruning_rate: float = 0.0,
        video_context_token: str = "<video>",
        img_context_token: str = "<image>",
        img_start_token: str = "<img>",
        img_end_token: str = "</img>",
        norm_mean: tuple[float, float, float] | list[float] = IMAGENET_MEAN,
        norm_std: tuple[float, float, float] | list[float] = IMAGENET_STD,
        use_thumbnail: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Handle both cases: when loading from JSON (llm_config is dict) and when called internally by transformers (llm_config; vision_config are None)
        if llm_config is not None:
            self.llm_config = NemotronHConfig(**llm_config)
            assert isinstance(vision_config, dict), "vision_config must be a dictionary"
            self.raw_vision_config = vision_config
        else:
            assert vision_config is None
            self.llm_config = NemotronHConfig()
            self.raw_vision_config = {}

        # Assign configuration values
        vision_image_size = self.raw_vision_config.get("image_size", force_image_size)
        vision_patch_size = self.raw_vision_config.get("patch_size", patch_size)
        self.image_size = int(
            vision_image_size[0]
            if isinstance(vision_image_size, list)
            else vision_image_size
        )
        self.patch_size = int(
            vision_patch_size[0]
            if isinstance(vision_patch_size, list)
            else vision_patch_size
        )

        self.downsample_ratio = downsample_ratio
        self.video_context_token = video_context_token
        self.img_context_token = img_context_token
        self.template = template  # TODO move out of here and into the tokenizer
        self.ps_version = ps_version  # Pixel shuffle version
        self.image_tag_type = image_tag_type  # TODO: into the tokenizer too?
        self.projector_hidden_size = projector_hidden_size
        self.vit_hidden_size = vit_hidden_size
        self.video_pruning_rate = video_pruning_rate

        self.norm_mean = float_triplet(norm_mean)
        self.norm_std = float_triplet(norm_std)
        self.use_thumbnail = use_thumbnail
        self.img_start_token = img_start_token
        self.img_end_token = img_end_token

    def create_radio_config(self):
        config = self.raw_vision_config
        model_name = config["args"]["model"]
        reg_tokens = config["args"].get("register_multiple")
        image_size = config.get("preferred_resolution", [224])[0]
        radio_config = RadioConfig(
            patch_size=self.patch_size,
            norm_mean=self.norm_mean,
            norm_std=self.norm_std,
            model_name=model_name,
            reg_tokens=reg_tokens,
            image_size=image_size,
        )
        return radio_config
