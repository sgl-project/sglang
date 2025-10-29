# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
from abc import abstractmethod
from typing import Any, Dict, Optional

import mindspore as ms

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class MindSporeModelBase(ms.nn.Cell):
    @abstractmethod
    def construct(self, **model_inputs) -> ms.Tensor:
        raise NotImplementedError

    def prepare_inputs(
        self, forward_batch: ForwardBatch, model_inputs: Dict
    ) -> ms.Tensor:
        return model_inputs
