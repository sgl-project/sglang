"""CPU coverage for W8A8 dequantization of MTP weights on the unquantized
(NPU) draft path.

On NPU the MTP draft is forced to ``quant_config=None`` while ModelSlim
checkpoints may store the MTP projections as W8A8 (int8 + per-row
``weight_scale``). Without dequantization the raw int8 codes are copied
into the bf16 parameters and the draft model is numerically broken
(spec-decode accept rate collapses to ~0). These tests cover scale
discovery and the ``load_weights`` dequant branch without an NPU.
"""

import json
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from sglang.srt.models import qwen3_5_mtp
from sglang.srt.models.qwen3_5_mtp import (
    Qwen3_5ForCausalLMMTP,
    _load_mtp_w8a8_dequant_scales,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

Q_PROJ = "mtp.layers.0.self_attn.q_proj"


def _write_ckpt(tmp_path, tensors):
    save_file(tensors, str(tmp_path / "shard_00001.safetensors"))
    index = {
        "metadata": {"total_size": 0},
        "weight_map": {name: "shard_00001.safetensors" for name in tensors},
    }
    (tmp_path / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps(index)
    )
    return str(tmp_path)


def _stub_model(monkeypatch, ckpt, quant_config=None):
    """Bypass __init__ (needs runtime context); set only what load_weights reads."""
    model = Qwen3_5ForCausalLMMTP.__new__(Qwen3_5ForCausalLMMTP)
    torch.nn.Module.__init__(model)
    model.quant_config = quant_config
    model.config = SimpleNamespace(torch_dtype=torch.bfloat16)
    monkeypatch.setattr(
        qwen3_5_mtp,
        "get_spec",
        lambda: SimpleNamespace(speculative_draft_model_path=ckpt),
    )
    monkeypatch.setattr(
        qwen3_5_mtp, "get_model", lambda: SimpleNamespace(model_path=ckpt)
    )
    return model


class _CaptureParam:
    def __init__(self):
        self.calls = []

    def weight_loader(self, param, loaded_weight, shard_id):
        self.calls.append((shard_id, loaded_weight))


def test_scale_discovery_keys_by_weight_name(tmp_path):
    q = torch.randint(-127, 128, (16, 8), dtype=torch.int8)
    s = (torch.rand(16, 1) * 0.01 + 0.001).float()
    ckpt = _write_ckpt(
        tmp_path,
        {
            f"{Q_PROJ}.weight": q,
            f"{Q_PROJ}.weight_scale": s,
            # FLOAT (unquantized) MTP tensors carry no scale
            "mtp.fc.weight": torch.randn(8, 32, dtype=torch.bfloat16),
            # non-mtp scales must be ignored
            "model.language_model.layers.0.mlp.gate_proj.weight_scale": torch.rand(
                4, 1
            ),
        },
    )
    scales = _load_mtp_w8a8_dequant_scales(ckpt)
    assert list(scales) == [f"{Q_PROJ}.weight"]
    assert torch.equal(scales[f"{Q_PROJ}.weight"], s)


def test_scale_discovery_without_index(tmp_path):
    assert _load_mtp_w8a8_dequant_scales(str(tmp_path)) == {}


def test_scale_discovery_missing_shard_does_not_raise(tmp_path):
    (tmp_path / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {f"{Q_PROJ}.weight_scale": "missing.safetensors"}}
        )
    )
    assert _load_mtp_w8a8_dequant_scales(str(tmp_path)) == {}


def test_load_weights_dequantizes_int8_mtp_projection(monkeypatch, tmp_path):
    q = torch.randint(-127, 128, (16, 8), dtype=torch.int8)
    s = (torch.rand(16, 1) * 0.01 + 0.001).float()
    ckpt = _write_ckpt(
        tmp_path, {f"{Q_PROJ}.weight": q, f"{Q_PROJ}.weight_scale": s}
    )
    model = _stub_model(monkeypatch, ckpt)
    capture = _CaptureParam()
    model.named_parameters = lambda *a, **k: [
        ("model.layers.0.qkv_proj.weight", capture)
    ]

    model.load_weights(iter([(f"{Q_PROJ}.weight", q.clone())]))

    assert len(capture.calls) == 1
    shard_id, loaded = capture.calls[0]
    assert shard_id == "q"
    assert loaded.dtype == torch.bfloat16
    expected = (q.float() * s.float()).to(torch.bfloat16)
    assert torch.equal(loaded, expected)


def test_load_weights_passes_int8_through_when_quantized(monkeypatch, tmp_path):
    q = torch.randint(-127, 128, (16, 8), dtype=torch.int8)
    s = (torch.rand(16, 1) * 0.01 + 0.001).float()
    ckpt = _write_ckpt(
        tmp_path, {f"{Q_PROJ}.weight": q, f"{Q_PROJ}.weight_scale": s}
    )
    model = _stub_model(monkeypatch, ckpt, quant_config=SimpleNamespace())
    capture = _CaptureParam()
    model.named_parameters = lambda *a, **k: [
        ("model.layers.0.qkv_proj.weight", capture)
    ]

    model.load_weights(iter([(f"{Q_PROJ}.weight", q.clone())]))

    shard_id, loaded = capture.calls[0]
    assert shard_id == "q"
    assert loaded.dtype == torch.int8
    assert torch.equal(loaded, q)


def test_load_weights_scale_shape_mismatch_is_noop(monkeypatch, tmp_path):
    q = torch.randint(-127, 128, (16, 8), dtype=torch.int8)
    s = (torch.rand(4, 1) * 0.01 + 0.001).float()  # wrong row count
    ckpt = _write_ckpt(
        tmp_path, {f"{Q_PROJ}.weight": q, f"{Q_PROJ}.weight_scale": s}
    )
    model = _stub_model(monkeypatch, ckpt)
    capture = _CaptureParam()
    model.named_parameters = lambda *a, **k: [
        ("model.layers.0.qkv_proj.weight", capture)
    ]

    model.load_weights(iter([(f"{Q_PROJ}.weight", q.clone())]))

    _, loaded = capture.calls[0]
    assert loaded.dtype == torch.int8
    assert torch.equal(loaded, q)
