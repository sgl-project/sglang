"""The retained torch.compile entry point is independent of TCPCG."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.compilation import torch_compile_decoration as decoration
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class Model(torch.nn.Module):
    def forward(self, x):
        return x + 2


def test_disabled_compile_keeps_raw_forward():
    model = Model()
    with patch.object(torch, "compile") as compile_model:
        with decoration.patch_model(
            model, False, 4, SimpleNamespace(ca_comm=None)
        ) as forward:
            assert forward(torch.tensor(3)).item() == 5
        compile_model.assert_not_called()


@pytest.mark.parametrize("raises", [False, True])
def test_compile_restores_module_policy_and_communicator(raises):
    model = Model()
    communicator = object()
    group = SimpleNamespace(ca_comm=communicator)
    with (
        patch.object(decoration, "_to_torch") as convert,
        patch.object(
            torch, "compile", side_effect=lambda forward, **kwargs: forward
        ) as compile_model,
    ):
        try:
            with decoration.patch_model(model, True, 4, group) as forward:
                assert forward(torch.tensor(3)).item() == 5
                group.ca_comm = None
                if raises:
                    raise RuntimeError("forward failed")
        except RuntimeError:
            assert raises
        assert group.ca_comm is communicator
        assert convert.call_args_list[0].kwargs == {"reverse": False, "num_tokens": 4}
        assert convert.call_args_list[1].kwargs == {"reverse": True, "num_tokens": 4}
        compile_model.assert_called_once()
