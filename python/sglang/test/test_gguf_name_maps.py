import json
from types import SimpleNamespace

import gguf
import numpy as np
import torch
from transformers import PretrainedConfig

from sglang.srt.model_loader.gguf_name_maps import (
    build_gemma4_name_map,
    build_qwen35_name_map,
)
from sglang.srt.model_loader.weight_utils import gguf_quant_weights_iterator
from sglang.srt.utils.hf_transformers import config as config_utils
from sglang.srt.utils.hf_transformers.config import (
    _maybe_gguf_gemma4_text_only,
    _maybe_gguf_text_only,
)


class _NameMap:
    def __init__(self, names):
        self.names = names

    def get_name(self, name):
        return self.names.get(name)


class _GGUF:
    def __init__(self, names):
        self.names = names

    def get_tensor_name_map(self, arch, layers):
        assert layers == 2
        return _NameMap(self.names)


def _write_index(tmp_path, names):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model-00001.safetensors" for name in names}})
    )


def test_qwen35_name_map_uses_sidecar_namespace(tmp_path):
    names = [
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.1.linear_attn.A_log",
        "model.language_model.layers.1.linear_attn.dt_bias",
        "model.visual.blocks.0.weight",
    ]
    _write_index(tmp_path, names)
    result = build_qwen35_name_map(
        SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=2)),
        _GGUF({"model.layers.0.self_attn.q_proj": "blk.0.attn_q"}),
        "QWEN35",
        str(tmp_path / "model.gguf"),
    )
    assert result == {
        "blk.0.attn_q.weight": "model.layers.0.self_attn.q_proj.weight",
        "blk.1.ssm_a": "model.layers.1.linear_attn.A_log",
        "blk.1.ssm_dt.bias": "model.layers.1.linear_attn.dt_bias",
    }


def test_gemma4_name_map_handles_router_and_suffixless_tensors(tmp_path):
    names = [
        "model.language_model.layers.0.router.scale",
        "model.language_model.layers.0.router.per_expert_scale",
        "model.language_model.layers.0.layer_scalar",
        "model.language_model.layers.0.self_attn.q_proj.qweight",
        "model.language_model.layers.0.self_attn.q_proj.weight_scale",
    ]
    _write_index(tmp_path, names)
    result = build_gemma4_name_map(
        SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=2)),
        _GGUF(
            {
                "model.layers.0.layer_scalar": "blk.0.layer_scalar",
                "model.layers.0.self_attn.q_proj": "blk.0.attn_q",
            }
        ),
        "GEMMA4",
        str(tmp_path / "model.gguf"),
    )
    assert result == {
        "blk.0.ffn_gate_inp.scale": names[0],
        "blk.0.ffn_down_exps.scale": names[1],
        "blk.0.layer_scalar.weight": names[2],
        "blk.0.attn_q.weight": (
            "model.language_model.layers.0.self_attn.q_proj.weight"
        ),
    }


def test_gemma4_sidecar_config_uses_text_tower():
    text_config = PretrainedConfig()
    config = SimpleNamespace(
        model_type="gemma4",
        text_config=text_config,
        bos_token_id=2,
        eos_token_id=[1, 3],
        pad_token_id=0,
    )
    assert _maybe_gguf_gemma4_text_only(config) is text_config
    assert text_config.architectures == ["Gemma4ForCausalLM"]
    assert (
        text_config.bos_token_id,
        text_config.eos_token_id,
        text_config.pad_token_id,
    ) == (
        2,
        [1, 3],
        0,
    )


def test_qwen35_sidecar_config_uses_text_tower():
    text_config = PretrainedConfig()
    config = SimpleNamespace(
        model_type="qwen3_5",
        text_config=text_config,
        architectures=["Qwen3_5ForConditionalGeneration"],
        bos_token_id=151643,
        eos_token_id=151645,
        pad_token_id=151643,
    )
    assert _maybe_gguf_text_only(config) is text_config
    assert text_config.architectures == ["Qwen3_5ForCausalLM"]


def test_get_config_uses_colocated_gguf_sidecar(tmp_path, monkeypatch):
    gguf_path = tmp_path / "model-Q4_K_M.gguf"
    gguf_path.write_bytes(b"GGUF")
    (tmp_path / "config.json").write_text("{}")

    text_config = PretrainedConfig()
    parent_config = SimpleNamespace(
        model_type="qwen3_5",
        text_config=text_config,
        architectures=["Qwen3_5ForConditionalGeneration"],
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )

    class _Parser:
        def parse(self, model, **_kwargs):
            assert model == tmp_path
            return parent_config

    monkeypatch.setattr(config_utils, "_ensure_gguf_version", lambda: None)
    monkeypatch.setattr(
        config_utils, "get_model_config_parser", lambda parser_name: _Parser()
    )

    result = config_utils.get_config(str(gguf_path), trust_remote_code=True)

    assert result is text_config
    assert result.architectures == ["Qwen3_5ForCausalLM"]


def test_gemma4_packed_moe_iterator_uses_mapped_subtree(monkeypatch):
    tensors = [
        SimpleNamespace(
            name="blk.0.ffn_down_exps.weight",
            tensor_type=gguf.GGMLQuantizationType.Q5_1,
            data=np.zeros((2, 6, 4), dtype=np.uint8),
        ),
        SimpleNamespace(
            name="blk.0.ffn_gate_up_exps.weight",
            tensor_type=gguf.GGMLQuantizationType.Q4_K,
            data=np.arange(2 * 8 * 3, dtype=np.uint8).reshape(2, 8, 3),
        ),
        SimpleNamespace(
            name="blk.0.ffn_down_exps.scale",
            tensor_type=gguf.GGMLQuantizationType.F32,
            data=np.ones((2,), dtype=np.float32),
        ),
    ]
    monkeypatch.setattr(gguf, "GGUFReader", lambda _: SimpleNamespace(tensors=tensors))
    name_map = {
        "blk.0.ffn_down_exps.weight": (
            "model.language_model.layers.0.experts.down_proj"
        ),
        "blk.0.ffn_gate_up_exps.weight": (
            "model.language_model.layers.0.experts.gate_up_proj"
        ),
        "blk.0.ffn_down_exps.scale": (
            "model.language_model.layers.0.router.per_expert_scale"
        ),
    }

    weights = dict(gguf_quant_weights_iterator("unused.gguf", name_map))
    prefix = "model.language_model.layers.0.experts.0"
    assert f"{prefix}.down_proj.qweight_type" in weights
    assert f"{prefix}.gate_proj.qweight_type" in weights
    assert f"{prefix}.up_proj.qweight_type" in weights
    assert weights[f"{prefix}.down_proj.qweight"].shape == (6, 4)
    torch.testing.assert_close(
        weights[f"{prefix}.gate_proj.qweight"],
        torch.arange(12, dtype=torch.uint8).reshape(4, 3),
    )
    torch.testing.assert_close(
        weights[f"{prefix}.up_proj.qweight"],
        torch.arange(12, 24, dtype=torch.uint8).reshape(4, 3),
    )
    assert "model.language_model.layers.0.router.per_expert_scale" in weights
