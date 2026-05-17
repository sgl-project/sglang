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


import torch
from transformers import PretrainedConfig

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.utils import logger

from .evs_core import tokens_per_frame
from .evs_module import EVS, EVSConfig, EVSDataItem, VideoEVSDataItem


def _non_evs_data_items(
    *,
    image: torch.Tensor | None,
    image_offsets: list[tuple[int, int]],
    video: torch.Tensor | None,
    video_offsets: list[tuple[int, int]],
    input_ids_list: list[int],
):
    items: list[MultimodalDataItem] = []
    if image is not None:
        item = MultimodalDataItem(
            modality=Modality.IMAGE, feature=image, offsets=image_offsets
        )
        items.append(item)
    if video is not None:
        item = MultimodalDataItem(
            modality=Modality.VIDEO, feature=video, offsets=video_offsets
        )
        items.append(item)
    return items


class EVSProcessor:
    """
    This processor handles prompt construction with the correct number of
    placeholder tokens per frame. When EVS is active, it allocates fewer
    placeholders based on the pruning rate. When inactive, it uses the full
    token count.
    """

    def __init__(
        self,
        hf_config: PretrainedConfig,
        config_to_evs_model: dict[type[PretrainedConfig], type[EVS]],
    ):
        assert len(config_to_evs_model) > 0
        assert all(issubclass(model, EVS) for model in config_to_evs_model.values())

        self.evs_config: EVSConfig | None = None

        config_name = hf_config.__class__.__name__
        evs_model = config_to_evs_model.get(hf_config.__class__)
        if evs_model is None:
            logger.info(
                f"[EVS] no model matches {config_name} in {config_to_evs_model}"
            )
            return
        evs_config = evs_model.create_evs_config(hf_config)
        logger.info(
            f"""[EVS] {evs_config} {'enabled' if evs_config.video_pruning_rate > 0.0 else 'disabled'} for model={evs_model.__name__}; model_config={config_name}"""
        )
        if evs_config.video_pruning_rate > 0.0:
            self.evs_config = evs_config

    def static_size_data_items(
        self, *, frames_per_video: list[int], num_images: int, rows: int, cols: int
    ):
        """helper function to create data items for models with static image and video tokens per frame"""

        frame_num_tokens = rows * cols

        if self.evs_config is None:
            tpf = [[frame_num_tokens] * num_frames for num_frames in frames_per_video]
            return _non_evs_data_items, tpf

        def create_evs_data_items(
            *,
            input_ids_list: list[int],
            image: torch.Tensor | None,
            image_offsets: list[tuple[int, int]],
            video: torch.Tensor | None,
            video_offsets: list[tuple[int, int]],
        ) -> list[MultimodalDataItem]:
            items = []
            if image is not None:
                image_thw_grids = [(1, rows, cols)] * num_images
                item = EVSDataItem(
                    modality=Modality.IMAGE,
                    feature=image,
                    offsets=image_offsets,
                    thw_grids=image_thw_grids,
                )
                items.append(item)
            if video is not None:
                video_thw_grids = [
                    (num_frames, rows, cols) for num_frames in frames_per_video
                ]
                item = VideoEVSDataItem(
                    modality=Modality.VIDEO,
                    feature=video,
                    offsets=video_offsets,
                    thw_grids=video_thw_grids,
                    pre_chunked_input_ids=input_ids_list,
                )
                items.append(item)
            return items

        tpf = [
            tokens_per_frame(
                q=self.evs_config.video_pruning_rate,
                num_frames=num_frames,
                frame_num_tokens=frame_num_tokens,
            )
            for num_frames in frames_per_video
        ]

        return create_evs_data_items, tpf
