# SPDX-License-Identifier: Apache-2.0
"""Alpha resolution for adapters that carry it in the safetensors header.

An adapter published as a loose `.safetensors` file has no sibling
`adapter_config.json` and no per-layer `.alpha` tensors, so alpha falls back to
rank. For a distilled adapter trained at alpha 8 with rank 128 that scales the
delta 16x while still loading and generating.
"""

import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import (
    _lora_alpha_from_safetensors_header,
)


def _write_adapter(path, metadata=None):
    save_file({"blocks.0.attn.qkv_proj.lora_A": torch.zeros(2, 2)}, path, metadata)
    return str(path)


def test_alpha_is_read_from_the_safetensors_header(tmp_path):
    path = _write_adapter(tmp_path / "turbo.safetensors", {"alpha": "8"})
    assert _lora_alpha_from_safetensors_header(path) == 8


def test_published_h3_turbo_alphas_are_per_checkpoint(tmp_path):
    """lightx2v/Minimax-h3-Turbo ships alpha 8 and alpha 128 side by side.

    Both are rank 128, so no repo-level default can serve them: the ref2v and
    8-step adapters want scale 8/128 while the 768p ones want scale 1.
    """
    for alpha in ("8", "128"):
        path = _write_adapter(tmp_path / f"turbo_{alpha}.safetensors", {"alpha": alpha})
        assert _lora_alpha_from_safetensors_header(path) == int(alpha)


def test_float_spelled_alpha_is_accepted(tmp_path):
    path = _write_adapter(tmp_path / "float.safetensors", {"alpha": "8.0"})
    assert _lora_alpha_from_safetensors_header(path) == 8


def test_header_without_alpha_defers_to_the_caller(tmp_path):
    path = _write_adapter(tmp_path / "bare.safetensors", {"format": "pt"})
    assert _lora_alpha_from_safetensors_header(path) is None
    path = _write_adapter(tmp_path / "nometa.safetensors")
    assert _lora_alpha_from_safetensors_header(path) is None


def test_unusable_alpha_defers_instead_of_raising(tmp_path):
    for spelling in ("", "none", "0", "-8", "8.5"):
        path = _write_adapter(tmp_path / "bad.safetensors", {"alpha": spelling})
        assert _lora_alpha_from_safetensors_header(path) is None


def test_non_safetensors_adapter_is_not_probed(tmp_path):
    """`.bin` adapters are still accepted by the loader and have no header."""
    path = tmp_path / "adapter.bin"
    path.write_bytes(b"not safetensors")
    assert _lora_alpha_from_safetensors_header(str(path)) is None
