# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sglang.multimodal_gen.runtime.cache.vla_prefix_cache import VLAPrefixCacheManager
from sglang.multimodal_gen.runtime.models.vlas import Pi05PolicyModel
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.pi05_preprocess import (
    Pi05Preprocessor,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.vla import (
    VLAActionDenoisingStage,
    VLAActionPostprocessStage,
    VLAObservationPreprocessStage,
    VLAPrefixStage,
    VLAStageKeys,
)

PI05_STAGE_KEYS = VLAStageKeys.for_namespace("pi05")


class Pi05PreprocessStage(VLAObservationPreprocessStage):
    def __init__(self, preprocessor: Pi05Preprocessor):
        super().__init__(preprocessor, keys=PI05_STAGE_KEYS)


class Pi05PrefixStage(VLAPrefixStage):
    def __init__(
        self,
        policy_model: Pi05PolicyModel,
        prefix_cache: VLAPrefixCacheManager,
    ):
        super().__init__(policy_model, prefix_cache, keys=PI05_STAGE_KEYS)


class Pi05ActionDenoisingStage(VLAActionDenoisingStage):
    def __init__(self, policy_model: Pi05PolicyModel):
        super().__init__(policy_model, keys=PI05_STAGE_KEYS)


class Pi05PostprocessStage(VLAActionPostprocessStage):
    def __init__(self):
        super().__init__(keys=PI05_STAGE_KEYS)
