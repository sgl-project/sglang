# SPDX-License-Identifier: Apache-2.0

import io
import pickle

import pytest
import torch
from sglang.multimodal_gen.runtime.managers.worker_bootstrap import _ConfigPickler


class _ConfigTensor(torch.Tensor):
    pass


@pytest.mark.parametrize(
    "tensor_type", [torch.Tensor, torch.nn.Parameter, _ConfigTensor]
)
def test_config_pickler_preserves_cpu_tensor_types(tensor_type):
    tensor = torch.arange(4, dtype=torch.float32).as_subclass(tensor_type)
    buffer = io.BytesIO()
    _ConfigPickler(buffer, protocol=pickle.HIGHEST_PROTOCOL).dump(tensor)
    restored = pickle.loads(buffer.getvalue())
    assert type(restored) is tensor_type
    torch.testing.assert_close(restored, tensor, rtol=0, atol=0)
