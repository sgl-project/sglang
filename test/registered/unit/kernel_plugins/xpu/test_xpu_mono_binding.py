# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.kernel_plugins.amd import binding
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, stage="base-a", runner_config="cpu")


def test_select_xpu_monokernel():
    args = ServerArgs(model_path="dummy", enable_xpu_monokernel=True)
    assert binding.selected_monokernel(args) == ("xpu", "--enable-xpu-monokernel")
    assert binding.selected_monokernel(ServerArgs(model_path="dummy")) is None
    args.enable_amd_monokernel = True
    with pytest.raises(ValueError, match="other monokernel flags"):
        binding.selected_monokernel(args)


def test_xpu_device_does_not_require_cuda():
    with patch.object(binding.torch.xpu, "is_available", return_value=False):
        assert binding.device_isa("xpu") is None


def test_xpu_flag_requires_xpu_server():
    args = SimpleNamespace(
        enable_xpu_monokernel=True,
        enable_amd_monokernel=False,
        enable_nvidia_megakernel=False,
        device="cpu",
    )
    with patch.object(binding, "resolving_view", return_value=args):
        with pytest.raises(ValueError, match="requires an XPU server"):
            binding.handle_monokernel(args)


@pytest.mark.parametrize("allow_fallback", [False, True])
def test_unsupported_xpu_draft_can_keep_the_standard_forward(allow_fallback):
    spec = SimpleNamespace(
        name="test draft",
        api_version=1,
        requires=SimpleNamespace(hf_config={}),
        load_tensor=None,
        model_disable_reasons=lambda *args: [],
        devices={"xe2": SimpleNamespace(resident_blocks=32)},
        allow_model_fallback=allow_fallback,
        build=Mock(),
    )
    model = torch.nn.Identity()
    forward = model.forward
    device = SimpleNamespace(
        current_device=lambda: 0,
        get_device_properties=lambda _: SimpleNamespace(gpu_subslice_count=20),
    )
    config = SimpleNamespace(architectures=["TestDraft"], model_type="test")
    with (
        patch.object(binding, "_backend", return_value="xpu"),
        patch.object(binding, "device_isa", return_value="xe2"),
        patch.object(binding, "find_spec", return_value=(spec, "")),
        patch.object(binding, "spec_registered", return_value=True),
        patch.object(binding, "_gather_tp", side_effect=lambda result: [result]),
        patch.object(binding, "_device_module", return_value=device),
        patch.object(binding, "_flag", return_value="--enable-xpu-monokernel"),
    ):
        if allow_fallback:
            binding.install_monokernel_forward(
                model=model, hf_config=config, capture_tail_hooks=[]
            )
        else:
            with pytest.raises(ValueError, match="32"):
                binding.install_monokernel_forward(
                    model=model, hf_config=config, capture_tail_hooks=[]
                )
    assert model.forward == forward
    spec.build.assert_not_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
