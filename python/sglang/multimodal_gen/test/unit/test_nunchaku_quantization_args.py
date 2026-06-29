# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sglang.multimodal_gen.configs.quantization import nunchaku
from sglang.multimodal_gen.configs.quantization.nunchaku import NunchakuSVDQuantArgs


def _patch_nunchaku_cuda(monkeypatch, capabilities):
    monkeypatch.setattr(nunchaku.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(nunchaku, "is_nunchaku_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(capabilities))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: capabilities[i])


def test_nunchaku_nvfp4_rejects_sm89(monkeypatch):
    _patch_nunchaku_cuda(monkeypatch, [(8, 9)])
    args = NunchakuSVDQuantArgs(
        enable_svdquant=True,
        transformer_weights_path="svdq-fp4_r128-z-image-turbo.safetensors",
        quantization_precision="nvfp4",
        quantization_rank=128,
    )

    with pytest.raises(ValueError, match="SVDQuant FP4"):
        args._validate()


@pytest.mark.parametrize("capability", [(10, 0), (12, 0)])
def test_nunchaku_nvfp4_allows_blackwell(monkeypatch, capability):
    _patch_nunchaku_cuda(monkeypatch, [capability])
    args = NunchakuSVDQuantArgs(
        enable_svdquant=True,
        transformer_weights_path="svdq-fp4_r128-z-image-turbo.safetensors",
        quantization_precision="nvfp4",
        quantization_rank=128,
    )

    args._validate()


def test_nunchaku_int4_keeps_sm89_support(monkeypatch):
    _patch_nunchaku_cuda(monkeypatch, [(8, 9)])
    args = NunchakuSVDQuantArgs(
        enable_svdquant=True,
        transformer_weights_path="svdq-int4_r128-z-image-turbo.safetensors",
        quantization_precision="int4",
        quantization_rank=128,
    )

    args._validate()
