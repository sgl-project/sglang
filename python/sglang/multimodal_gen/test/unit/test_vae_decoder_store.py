"""The decode-dtype weights live in a file-backed mapping, not anonymous memory.

What matters: the store round-trips the exact rounded bytes, a later start
adopts it without paying the cast, a mismatched store is discarded rather than
adopted, and the kill switch keeps everything in memory.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.loader.component_loaders.vae_loader import (
    _decode_dtype_store_path,
    _hold_decoder_weights_in_decode_dtype,
)


class _TinyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])
        self.head = nn.Linear(8, 8)  # stays fp32, like the output projection
        self.prepare_calls = 0

    def prepare_decoder_autocast_weights(self, dtype) -> int:
        self.prepare_calls += 1
        converted = 0
        for block in self.blocks:
            if block.weight.dtype != dtype:
                block.to(dtype=dtype)
                converted += 1
        return converted


def _server_args():
    return SimpleNamespace(
        component_precisions={},
        pipeline_config=SimpleNamespace(
            vae_decode_precision="fp16",
            vae_precision="fp32",
        ),
        disable_autocast=False,
    )


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    from sglang.multimodal_gen.runtime.utils import precision

    monkeypatch.setattr(precision.current_platform, "is_amp_supported", lambda: True)
    monkeypatch.setenv("SGLANG_DIFFUSION_CACHE_ROOT", str(tmp_path / "cache"))


def test_the_cast_weights_end_up_file_backed(tmp_path):
    vae = _TinyVAE()
    model_path = tmp_path / "ckpt"
    model_path.mkdir()
    _hold_decoder_weights_in_decode_dtype(
        vae, _server_args(), "video_vae", str(model_path)
    )

    path = _decode_dtype_store_path(str(model_path), "video_vae", torch.float16)
    import os

    assert os.path.exists(path)
    assert all(b.weight.dtype == torch.float16 for b in vae.blocks)
    assert vae.head.weight.dtype == torch.float32

    from safetensors.torch import load_file

    stored = load_file(path)
    for name, tensor in stored.items():
        assert torch.equal(tensor, vae.state_dict()[name])


def test_a_second_start_adopts_the_store_without_casting(tmp_path):
    model_path = tmp_path / "ckpt"
    model_path.mkdir()
    first = _TinyVAE()
    _hold_decoder_weights_in_decode_dtype(
        first, _server_args(), "video_vae", str(model_path)
    )

    second = _TinyVAE()
    second.load_state_dict(
        {
            k: v.to(torch.float32) if v.dtype == torch.float16 else v
            for k, v in first.state_dict().items()
        }
    )
    _hold_decoder_weights_in_decode_dtype(
        second, _server_args(), "video_vae", str(model_path)
    )
    assert second.prepare_calls == 0
    for name in first.state_dict():
        assert torch.equal(second.state_dict()[name], first.state_dict()[name])


def test_a_mismatched_store_is_discarded_and_the_cast_kept(tmp_path):
    model_path = tmp_path / "ckpt"
    model_path.mkdir()
    path = _decode_dtype_store_path(str(model_path), "video_vae", torch.float16)
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    from safetensors.torch import save_file

    save_file({"blocks.0.weight": torch.zeros(4, 4, dtype=torch.float16)}, path)

    vae = _TinyVAE()
    _hold_decoder_weights_in_decode_dtype(
        vae, _server_args(), "video_vae", str(model_path)
    )
    assert vae.prepare_calls == 1
    assert all(b.weight.dtype == torch.float16 for b in vae.blocks)
    assert not os.path.exists(path)


def test_the_store_kill_switch_keeps_the_copies_in_memory(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_DIFFUSION_DISABLE_VAE_DECODER_STORE", "1")
    model_path = tmp_path / "ckpt"
    model_path.mkdir()
    vae = _TinyVAE()
    _hold_decoder_weights_in_decode_dtype(
        vae, _server_args(), "video_vae", str(model_path)
    )

    path = _decode_dtype_store_path(str(model_path), "video_vae", torch.float16)
    import os

    assert not os.path.exists(path)
    assert all(b.weight.dtype == torch.float16 for b in vae.blocks)
