"""CPU preprocessing must not accidentally allocate image tensors on CUDA."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors import base_processor


def make_processor(monkeypatch):
    monkeypatch.setattr(base_processor, "_is_cpu", False)
    monkeypatch.setattr(base_processor, "_is_xpu", False)
    monkeypatch.setattr(base_processor, "_is_npu", False)
    monkeypatch.setattr(
        base_processor,
        "get_exec",
        lambda: SimpleNamespace(
            deterministic=SimpleNamespace(rl_on_policy_target=None)
        ),
    )
    monkeypatch.setattr(
        base_processor.platforms,
        "current_platform",
        SimpleNamespace(is_cuda_alike=lambda: True, device_type="cuda"),
    )
    cls = base_processor.BaseMultimodalProcessor
    with patch.object(cls, "__abstractmethods__", set()):
        proc = object.__new__(cls)
    proc.server_args = SimpleNamespace(base_gpu_id=3)
    proc.disable_fast_image_processor = False
    return proc


@pytest.mark.parametrize("cpu_only,expected", [(True, "cpu"), (False, "cuda:3")])
def test_preprocess_device_override_preserves_default(monkeypatch, cpu_only, expected):
    proc = make_processor(monkeypatch)
    with envs.SGLANG_MM_PREPROCESS_ON_CPU.override(cpu_only):
        assert proc._fast_image_processor_device(MagicMock()) == expected


@pytest.mark.parametrize("cpu_only,gpu_decode", [(True, False), (False, True)])
def test_cpu_override_also_disables_gpu_jpeg_decode(monkeypatch, cpu_only, gpu_decode):
    image = Image.new("RGB", (8, 8))
    loader = MagicMock(return_value=(image, image.size))
    monkeypatch.setattr(base_processor, "load_image", loader)
    with envs.SGLANG_MM_PREPROCESS_ON_CPU.override(cpu_only):
        result = base_processor.BaseMultimodalProcessor._load_single_item(
            b"encoded image", Modality.IMAGE
        )
    assert result is image
    loader.assert_called_once_with(b"encoded image", gpu_decode)
