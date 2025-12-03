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

"""
EVS (Efficient Video Sampling) module integration for SGLang.
Integration requires two components:
1. EVS: A model mixin that wraps get_video_feature() to apply pruning
2. EVSProcessor: A processor mixin that adjusts token counts in prompt construction
"""

import dataclasses
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from transformers import PretrainedConfig

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult
from sglang.srt.multimodal.evs import (
    compute_retained_tokens_count,
    compute_retention_mask,
)
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.utils import logger


@dataclasses.dataclass(kw_only=True)
class VideoEVSDataItem(MultimodalDataItem):
    modality: Modality = Modality.VIDEO
    frames_per_video: list[int]

    def __post_init__(self):
        assert self.is_video()


@dataclass(kw_only=True)
class EVSEmbeddingResult(EmbeddingResult):
    """
    Embedding result that includes per-frame token counts after EVS pruning.

    After pruning, each frame retains a different number of tokens based on its
    dissimilarity to the previous frame. This metadata is needed downstream to
    adjust the input_ids placeholder spans to match the actual embedding sizes.

    Attributes:
        embedding: The pruned video embeddings tensor.
        num_tokens_per_frame: Actual retained token count for each frame.
            For example, [256, 180, 195, 256] means frame 0 kept all 256 tokens
            (first frame is never pruned), while frames 1-2 were pruned.
    """

    num_tokens_per_frame: list[int]


@dataclass(frozen=True, kw_only=True)
class EVSConfig:
    video_pruning_rate: float
    full_frame_num_tokens: int

    def __post_init__(self):
        assert (
            self.video_pruning_rate >= 0.0 and self.video_pruning_rate < 1.0
        ), f"Video pruning rate must be between 0.0 and 1.0, got {self.video_pruning_rate=}"
        assert self.full_frame_num_tokens > 0

    def tokens_per_frame(self, num_frames: int) -> list[int]:
        """
        Estimate token distribution across frames for prompt construction.

        This is called by the processor BEFORE embeddings are computed in `process_mm_data_async` to allocate
        placeholder tokens in the prompt. It provides an even distribution estimate;
        the actual per-frame counts after pruning may differ slightly due to the
        content-aware nature of EVS, although the *total* number of tokens is guaranteed to be the same.
        The distribution isn't interesting really - it can just be random if we wanted to.
        """
        retained = compute_retained_tokens_count(
            tokens_per_frame=self.full_frame_num_tokens,
            num_frames=num_frames,
            q=self.video_pruning_rate,
        )
        base = retained // num_frames
        rem = retained % num_frames
        return [base] * (num_frames - 1) + [base + rem]


class EVS(torch.nn.Module, ABC):
    """
    Base class for video models that support EVS pruning.

    Subclass this alongside your model class and implement the static `create_evs_config`.
    On initialization, if video_pruning_rate > 0, this mixin replaces the model's
    get_video_feature() method with a wrapper that applies EVS pruning.

    The model must define a get_video_feature(items) method that returns video
    embeddings as a tensor of shape (total_frames, tokens_per_frame, hidden_dim).

    Example: See `NemotronH_Nano_VL_V2`
    """

    @staticmethod
    @abstractmethod
    def create_evs_config(config: PretrainedConfig) -> EVSConfig:
        """Extract EVS parameters from model config. Must be implemented by subclass."""
        raise NotImplementedError

    def __init__(
        self,
        config: PretrainedConfig,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> None:
        super().__init__()
        model_name = self.__class__.__name__
        if not hasattr(self, "get_video_feature"):
            raise AttributeError(
                f"Expected Model {model_name} to have a `get_video_feature` method"
            )
        self.evs_config = self.create_evs_config(config)
        video_pruning_rate = self.evs_config.video_pruning_rate
        self.original_get_video_feature = self.get_video_feature
        if video_pruning_rate > 0.0:
            logger.info(f"[EVS] enabled for {model_name} [{video_pruning_rate=}]")
            self.get_video_feature = self.evs
        else:
            logger.info(
                f"[EVS] requested on model {model_name} but is disabled for pruning_rate == 0.0."
            )

    def evs(self, items: list[MultimodalDataItem]) -> EVSEmbeddingResult:
        """
        Apply EVS pruning to video embeddings.

        Args:
            items: List containing a single VideoEVSDataItem with video features.

        Returns:
            EVSEmbeddingResult with pruned embeddings and actual token counts per frame.
        """
        q = self.evs_config.video_pruning_rate
        logger.debug(f"[EVS] beginning for model {self.__class__.__name__} [{q=}]")
        rows = cols = int(self.evs_config.full_frame_num_tokens**0.5)
        if len(items) > 1:
            raise MultimodalDataItem.MultimodalDataItemContainsAllItemsOfSameModality
        item = items[0]
        assert isinstance(item, VideoEVSDataItem)
        videos_features = self.original_get_video_feature([item])

        final_embeddings: list[torch.Tensor] = []
        num_tokens_per_frame: list[int] = []

        for single_video in videos_features.split(item.frames_per_video):
            num_frames = single_video.shape[0]

            video_size_thw = (num_frames, rows, cols)

            retention_mask = compute_retention_mask(
                single_video,
                video_size_thw=video_size_thw,
                spatial_merge_size=1,
                q=q,
            ).view(num_frames, -1)

            preserved = single_video[retention_mask]
            final_embeddings.append(preserved)
            retention_mask_thw = retention_mask.reshape(video_size_thw)
            num_tokens_per_frame.extend(
                retention_mask_thw.sum(dim=(1, 2)).long().tolist()
            )
        final_embeddings_tensor = torch.cat(final_embeddings)
        return EVSEmbeddingResult(
            embedding=final_embeddings_tensor,
            num_tokens_per_frame=num_tokens_per_frame,
        )


@dataclass(frozen=True, kw_only=True)
class NonEVSConfig:
    """
    Fallback configuration for when EVS is not active.

    Used by EVSProcessor when the model doesn't support EVS or when pruning
    rate is 0. Provides the fixed token count per frame.
    """

    frame_num_tokens: int


class EVSProcessor(BaseMultimodalProcessor, ABC):
    """
    Base processor class for models that may support EVS.

    This processor handles prompt construction with the correct number of
    placeholder tokens per frame. When EVS is active, it allocates fewer
    placeholders based on the pruning rate. When inactive, it uses the full
    token count.

    Subclass this and implement create_non_evs_config() to specify the
    unpruned token count. The processor will automatically detect if the
    model supports EVS by checking processor.models for EVS subclasses.
    """

    @staticmethod
    @abstractmethod
    def create_non_evs_config(hf_config: PretrainedConfig) -> NonEVSConfig:
        """
        Return fallback config for non-EVS models.

        This is required because a processor may serve multiple model variants,
        some of which don't support EVS.
        """
        raise NotImplementedError

    def __init__(self, hf_config: PretrainedConfig, *args, **kwargs):
        super().__init__(hf_config, *args, **kwargs)
        self.non_evs_config = self.create_non_evs_config(hf_config)

        config_name = hf_config.__class__.__name__
        processor_name = self.__class__.__name__
        model_name = hf_config.model_type

        assert isinstance(model_name, str)

        evs_models = {
            model.__name__: model for model in self.models if issubclass(model, EVS)
        }

        if len(evs_models) == 0:
            logger.warning(
                f"[EVS] No EVS models found for processor.models={self.models}"
            )

        identity = f"processor={processor_name} model={model_name} config={config_name}"
        self.evs_config = None
        if model_name in evs_models:
            evs_model = evs_models[model_name]
            evs_config = evs_model.create_evs_config(hf_config)
            assert (
                self.non_evs_config.frame_num_tokens == evs_config.full_frame_num_tokens
            )
            logger.info(f"[EVS] {evs_config} resolved for triplet {identity}")
            if evs_config.video_pruning_rate > 0.0:
                self.evs_config = evs_config
        else:
            logger.info(f"[EVS] no config found for triplet {identity}")

    def tokens_per_frame(self, num_frames: int) -> list[int]:
        """
        Get the number of placeholder tokens to allocate per frame.

        Called during prompt construction to determine how many placeholder
        tokens to insert for each video frame.

        Returns:
            List of token counts. If EVS is active, returns estimated pruned
            counts. Otherwise, returns the full token count for each frame.
        """
        if self.evs_config is not None:
            return self.evs_config.tokens_per_frame(num_frames)
        else:
            return [self.non_evs_config.frame_num_tokens] * num_frames

    def data_item(
        self,
        frames_per_video: list[int],
        *,
        feature: torch.Tensor,
        offsets: list[tuple[int, int]],
    ) -> MultimodalDataItem:
        """
        Create the appropriate data item based on EVS status.

        Returns VideoEVSDataItem (with frame counts) if EVS is active,
        otherwise returns a standard MultimodalDataItem.
        """
        if self.evs_config is not None:
            return VideoEVSDataItem(
                frames_per_video=frames_per_video, feature=feature, offsets=offsets
            )
        else:
            return MultimodalDataItem(
                modality=Modality.VIDEO, feature=feature, offsets=offsets
            )
