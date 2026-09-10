# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from sglang.srt.layers.quantization.base_scheme import BaseLinearScheme, BaseMoEScheme

__all__ = ["ModelSlimLinearScheme", "ModelSlimMoEScheme"]


class ModelSlimLinearScheme(BaseLinearScheme):
    """
    Base class for the linear schemes supported by ModelSlim. The contract is
    BaseLinearScheme's.
    """


class ModelSlimMoEScheme(BaseMoEScheme):
    """
    Base class for the MoE schemes supported by ModelSlim. The contract is
    BaseMoEScheme's.
    """
