# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
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
from transformers.configuration_utils import PretrainedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.utils import logging

# from .configuration_radio import RADIOConfig  # todo afrimi  fix it!
from sglang.srt.configs.nemotron_h import NemotronHConfig

logger = logging.get_logger(__name__)


class NemotronH_Nano_VL_V2_Config(PretrainedConfig):
    model_type = "NemotronH_Nano_VL_V2"
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        llm_config=None,
        force_image_size=None,
        downsample_ratio=0.5,
        template=None,
        ps_version="v1",
        image_tag_type="internvl",
        projector_hidden_size=4096,
        vit_hidden_size=1280,
        attn_implementation="flash_attention_2",
        video_pruning_rate: float = 0.0,
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
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
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
        self.llm_config._attn_implementation = self._attn_implementation
