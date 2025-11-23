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

from transformers.configuration_utils import PretrainedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.utils import logging

from sglang.srt.configs.nemotron_h import NemotronHConfig
from sglang.srt.multimodal.internvl_utils import IMAGENET_MEAN, IMAGENET_STD

logger = logging.get_logger(__name__)


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
        attn_implementation="flash_attention_2",
        video_pruning_rate: float = 0.0,
        video_context_token: str = "<video>",
        img_context_token: str = "<image>",
        img_start_token: str = "<img>",
        img_end_token: str = "</img>",
        norm_mean: tuple[float, float, float] | list[float] = IMAGENET_MEAN,
        norm_std: tuple[float, float, float] | list[float] = IMAGENET_STD,
        use_thumbnail: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        if vision_config is not None:
            vision_auto_config = get_class_from_dynamic_module(
                *vision_config["auto_map"]["AutoConfig"].split("--")[::-1]
            )
            self.vision_config = vision_auto_config(**vision_config)
        else:
            self.vision_config = PretrainedConfig()

        # Handle both cases: when loading from JSON (llm_config is dict) and when called internally by transformers (llm_config is None)
        if llm_config is not None:
            self.llm_config = NemotronHConfig(**llm_config)
        else:
            self.llm_config = NemotronHConfig()

        # Assign configuration values
        vision_image_size = getattr(self.vision_config, "image_size", force_image_size)
        vision_patch_size = getattr(self.vision_config, "patch_size", patch_size)
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

        self._attn_implementation = attn_implementation
        self.vision_config.use_flash_attn = (
            self._attn_implementation is not None
            and "flash_attention" in self._attn_implementation
        )
        self.norm_mean = tuple(norm_mean)
        assert len(self.norm_mean) == 3, "norm_mean must be a tuple of 3 elements"
        self.norm_std = tuple(norm_std)
        assert len(self.norm_std) == 3, "norm_std must be a tuple of 3 elements"
        self.use_thumbnail = use_thumbnail
        self.llm_config._attn_implementation = self._attn_implementation

        self.img_start_token = img_start_token
        self.img_end_token = img_end_token
