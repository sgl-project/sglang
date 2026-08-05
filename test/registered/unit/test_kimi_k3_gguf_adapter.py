import json
import mmap
import sys
import types
from enum import IntEnum
from pathlib import Path

import numpy as np
import pytest
import torch
from sglang.srt.model_loader.kimi_k3_gguf_adapter import (
    KimiK3GGUFAdapter,
    _dense_tensor,
    _find_arch,
    _MMapRangeReleaser,
    _Transform,
    is_kimi_k3_gguf_config,
)


class _Type(IntEnum):
    F32 = 0
    BF16 = 1
    IQ2_XXS = 2


class _Field:
    def __init__(self, value):
        self.value = value

    def contents(self):
        return self.value


class _Tensor:
    def __init__(self, name, data, tensor_type=_Type.F32):
        self.name = name
        self.data = np.asarray(data)
        self.tensor_type = tensor_type


class _MappedFile:
    def __init__(self, size):
        self._size = size
        self.advice = []

    def size(self):
        return self._size

    def madvise(self, advice, start, length):
        self.advice.append((advice, start, length))


class _NameMap:
    def get_name(self, name, try_suffixes=()):
        if name == "model.embed_tokens.weight":
            return "token_embd.weight"
        return None


def _source_names():
    names = ["language_model.model.embed_tokens.weight"]
    layer_tails = [
        "self_attn.A_log",
        "self_attn.dt_bias",
        "self_attn.q_conv1d.weight",
        "self_attn.k_conv1d.weight",
        "self_attn.v_conv1d.weight",
        "self_attn.g_proj.weight",
        "self_attention_res_norm.weight",
        "self_attention_res_proj.weight",
        "mlp_res_norm.weight",
        "mlp_res_proj.weight",
        "block_sparse_moe.routed_expert_down_proj.weight",
        "block_sparse_moe.routed_expert_up_proj.weight",
        "block_sparse_moe.routed_expert_norm.weight",
        "block_sparse_moe.gate.e_score_correction_bias",
    ]
    names.extend(f"language_model.model.layers.0.{tail}" for tail in layer_tails)
    names.extend(
        [
            "language_model.model.output_attn_res_norm.weight",
            "language_model.model.output_attn_res_proj.weight",
        ]
    )
    for expert in range(2):
        for projection in ("w1", "w2", "w3"):
            for suffix in ("packed", "scale"):
                names.append(
                    "language_model.model.layers.0.block_sparse_moe.experts."
                    f"{expert}.{projection}.weight_{suffix}"
                )
    return names


def _tensors():
    dense = {
        "token_embd.weight": np.ones((4, 4), dtype=np.float32),
        "blk.0.ssm_a": -np.exp(np.array([0.25, 0.5], dtype=np.float32)),
        "blk.0.ssm_dt.bias": np.ones(2, dtype=np.float32),
        "blk.0.ssm_conv1d_q.weight": np.ones((1, 2, 1, 2), dtype=np.float32),
        "blk.0.ssm_conv1d_k.weight": np.ones((1, 2, 1, 2), dtype=np.float32),
        "blk.0.ssm_conv1d_v.weight": np.ones((1, 2, 1, 2), dtype=np.float32),
        "blk.0.ssm_g.weight": np.ones((2, 4), dtype=np.float32),
        "blk.0.attn_res_score.weight": np.arange(4, dtype=np.float32),
        "blk.0.ffn_res_score.weight": np.arange(4, dtype=np.float32),
        "output_res_score.weight": np.arange(4, dtype=np.float32),
        "blk.0.ffn_routed_down.weight": np.ones((2, 4), dtype=np.float32),
        "blk.0.ffn_routed_up.weight": np.ones((4, 2), dtype=np.float32),
        "blk.0.ffn_routed_norm.weight": np.ones(2, dtype=np.float32),
        "blk.0.exp_probs_b.bias": np.ones(2, dtype=np.float32),
    }
    tensors = [_Tensor(name, value) for name, value in dense.items()]
    for role in ("gate", "down", "up"):
        tensors.append(
            _Tensor(
                f"blk.0.ffn_{role}_exps.weight",
                np.arange(32, dtype=np.uint8).reshape(2, 4, 4),
                _Type.IQ2_XXS,
            )
        )
    return tensors


def _model_config(path: Path):
    text = types.SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=4,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        linear_attn_config={"full_attn_layers": []},
        num_experts=2,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        num_key_value_heads=2,
        kv_lora_rank=2,
        qk_nope_head_dim=2,
        v_head_dim=2,
    )
    root = types.SimpleNamespace(
        model_type="kimi_k3",
        architectures=["KimiK3ForConditionalGeneration"],
        language_only=True,
    )
    return types.SimpleNamespace(
        model_path=str(path), hf_config=root, hf_text_config=text
    )


def _install_fake_gguf(monkeypatch, tensors):
    fields = {
        "general.architecture": _Field("kimi-k3"),
        "kimi-k3.block_count": _Field(1),
        "kimi-k3.activation.situ_beta": _Field(4.0),
        "kimi-k3.activation.situ_linear_beta": _Field(25.0),
    }

    cursor = 2 * mmap.PAGESIZE + 137
    for tensor in tensors:
        tensor.data_offset = cursor
        tensor.n_bytes = tensor.data.nbytes
        cursor += tensor.n_bytes + mmap.PAGESIZE
    mapped = _MappedFile(cursor + mmap.PAGESIZE)

    class Reader:
        def __init__(self, _path):
            self.fields = fields
            self.tensors = tensors
            self.data = types.SimpleNamespace(_mmap=mapped)

    module = types.SimpleNamespace(
        MODEL_ARCH_NAMES={1: "kimi-linear"},
        GGUFReader=Reader,
        get_tensor_name_map=lambda _arch, _layers: _NameMap(),
        mapped=mapped,
    )
    monkeypatch.setitem(sys.modules, "gguf", module)
    return module


def _make_adapter(tmp_path, monkeypatch, tensors=None):
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"synthetic")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {name: "model.safetensors" for name in _source_names()}}
        )
    )
    _install_fake_gguf(monkeypatch, tensors or _tensors())
    return KimiK3GGUFAdapter(str(gguf_path), _model_config(gguf_path))


def test_architecture_falls_back_to_complete_kimi_linear_base_map():
    module = types.SimpleNamespace(MODEL_ARCH_NAMES={7: "kimi-linear"})
    assert _find_arch(module) == 7


def test_adapter_requires_language_only(tmp_path, monkeypatch):
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"synthetic")
    config = _model_config(gguf_path)
    config.hf_config.language_only = False
    _install_fake_gguf(monkeypatch, _tensors())
    with pytest.raises(ValueError, match="--language-only"):
        KimiK3GGUFAdapter(str(gguf_path), config)


def test_adapter_rejects_unplanned_text_tensor(tmp_path, monkeypatch):
    tensors = _tensors() + [_Tensor("blk.0.silent_corruption", [1.0])]
    with pytest.raises(ValueError, match="extra=.*silent_corruption"):
        _make_adapter(tmp_path, monkeypatch, tensors)


def test_iterator_orders_qtypes_and_inverts_k3_transforms(tmp_path, monkeypatch):
    adapter = _make_adapter(tmp_path, monkeypatch)
    rows = list(adapter.weights_iterator())

    # Three projections x two experts, all before any packed tensor.
    assert all(name.endswith("qweight_type") for name, _ in rows[:6])
    first_qweight = next(
        i for i, (name, _) in enumerate(rows) if name.endswith("qweight")
    )
    assert first_qweight > 6

    by_name = {name: value for name, value in rows if not name.endswith("qweight_type")}
    assert torch.allclose(
        by_name["language_model.model.layers.0.self_attn.A_log"],
        torch.tensor([0.25, 0.5]),
    )
    assert by_name["language_model.model.layers.0.self_attn.q_conv1d.weight"].shape == (
        2,
        1,
        2,
    )
    assert "language_model.model.layers.0.self_attention_res_score_gguf" in by_name
    assert (
        "language_model.model.layers.0.block_sparse_moe.experts.0.w1.qweight" in by_name
    )
    assert (
        "language_model.model.layers.0.block_sparse_moe.experts.0.w2.qweight" in by_name
    )
    assert (
        "language_model.model.layers.0.block_sparse_moe.experts.0.w3.qweight" in by_name
    )


def test_kimi_config_detection_is_exact(tmp_path):
    path = tmp_path / "model.gguf"
    config = _model_config(path)
    assert is_kimi_k3_gguf_config(config)
    config.hf_config.model_type = "kimi_linear"
    assert not is_kimi_k3_gguf_config(config)


def test_bf16_reader_bytes_are_reinterpreted_without_dequantization():
    expected = torch.tensor([[1.0, -2.5]], dtype=torch.bfloat16)
    raw = expected.view(torch.uint8).numpy()
    actual = _dense_tensor(_Tensor("protected.weight", raw, _Type.BF16))
    assert actual.dtype == torch.bfloat16
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


def _aligned_release(mapped, offset, length):
    start = offset - offset % mmap.PAGESIZE
    end = min(
        mapped.size(),
        ((offset + length + mmap.PAGESIZE - 1) // mmap.PAGESIZE) * mmap.PAGESIZE,
    )
    return (mmap.MADV_DONTNEED, start, end - start)


def test_dense_ranges_release_only_after_synchronous_consumer_resumes(
    tmp_path, monkeypatch
):
    adapter = _make_adapter(tmp_path, monkeypatch)
    mapped = adapter.reader.data._mmap
    iterator = adapter.weights_iterator()

    # Qtypes do not touch payload pages.
    for _ in range(6):
        assert next(iterator)[0].endswith("qweight_type")
    assert mapped.advice == []

    token_name, token_value = next(iterator)
    assert token_name == "language_model.model.embed_tokens.weight"
    token_copy = token_value.clone()  # Model loader's synchronous device-copy boundary.
    assert torch.equal(token_copy, token_value)
    assert mapped.advice == []

    a_log_name, a_log_value = next(iterator)
    token = adapter.tensors["token_embd.weight"]
    assert mapped.advice == [_aligned_release(mapped, token.data_offset, token.n_bytes)]
    assert a_log_name.endswith("self_attn.A_log")
    a_log_copy = a_log_value.clone()
    assert torch.equal(a_log_copy, a_log_value)

    next(iterator)
    a_log = adapter.tensors["blk.0.ssm_a"]
    assert mapped.advice[-1] == _aligned_release(
        mapped, a_log.data_offset, a_log.n_bytes
    )
    iterator.close()


def test_each_expert_range_releases_after_its_zero_copy_view_is_consumed(
    tmp_path, monkeypatch
):
    adapter = _make_adapter(tmp_path, monkeypatch)
    mapped = adapter.reader.data._mmap
    iterator = adapter.weights_iterator()

    while True:
        name, value = next(iterator)
        if name.endswith(".qweight"):
            break
    assert name.endswith("experts.0.w1.qweight")
    before = len(mapped.advice)
    owned_copy = value.clone()
    assert torch.equal(owned_copy, value)
    assert len(mapped.advice) == before

    next_name, _ = next(iterator)
    assert next_name.endswith("experts.1.w1.qweight")
    gate = adapter.tensors["blk.0.ffn_gate_exps.weight"]
    assert mapped.advice[-1] == _aligned_release(
        mapped,
        gate.data_offset,
        gate.data[0].nbytes,
    )
    iterator.close()


def test_kv_transform_defers_both_source_ranges_until_combined_value_is_consumed(
    tmp_path,
):
    path = tmp_path / "mapped.gguf"
    path.write_bytes(b"synthetic")
    k_b = _Tensor("blk.0.attn_k_b.weight", np.ones((2, 2, 2), dtype=np.float32))
    v_b = _Tensor("blk.0.attn_v_b.weight", np.ones((2, 2, 2), dtype=np.float32))
    for index, tensor in enumerate((v_b, k_b), start=1):
        tensor.data_offset = index * mmap.PAGESIZE + 73
        tensor.n_bytes = tensor.data.nbytes
    mapped = _MappedFile(4 * mmap.PAGESIZE)

    adapter = object.__new__(KimiK3GGUFAdapter)
    adapter.reader = types.SimpleNamespace(tensors=[v_b, k_b])
    adapter.regular = {}
    adapter.transforms = {
        k_b.name: _Transform("kv_b", "language_model.model.layers.0.kv", v_b.name),
        v_b.name: _Transform(
            "kv_b_partner", "language_model.model.layers.0.kv", k_b.name
        ),
    }
    adapter.tensors = {tensor.name: tensor for tensor in (v_b, k_b)}
    adapter.text_config = types.SimpleNamespace(
        num_key_value_heads=2,
        kv_lora_rank=2,
        qk_nope_head_dim=2,
        v_head_dim=2,
    )
    adapter._range_releaser = _MMapRangeReleaser(
        str(path), types.SimpleNamespace(data=types.SimpleNamespace(_mmap=mapped))
    )

    iterator = adapter._iter_dense()
    name, combined = next(iterator)
    assert name == "language_model.model.layers.0.kv"
    assert combined.shape == (8, 2)
    combined_copy = combined.clone()
    assert torch.equal(combined_copy, combined)
    assert mapped.advice == []
    with pytest.raises(StopIteration):
        next(iterator)
    assert mapped.advice == [
        _aligned_release(mapped, k_b.data_offset, k_b.n_bytes),
        _aligned_release(mapped, v_b.data_offset, v_b.n_bytes),
    ]


def test_shared_inode_releasers_never_use_global_fadvise(tmp_path, monkeypatch):
    path = tmp_path / "mapped.gguf"
    path.write_bytes(b"x" * (2 * mmap.PAGESIZE))
    first_mapping = _MappedFile(path.stat().st_size)
    second_mapping = _MappedFile(path.stat().st_size)

    def reject_global_fadvise(*_args):
        raise AssertionError("shared-inode fadvise must never be called")

    monkeypatch.setattr("os.posix_fadvise", reject_global_fadvise, raising=False)
    monkeypatch.setattr("os.POSIX_FADV_DONTNEED", 4, raising=False)
    first = _MMapRangeReleaser(
        str(path),
        types.SimpleNamespace(data=types.SimpleNamespace(_mmap=first_mapping)),
    )
    second = _MMapRangeReleaser(
        str(path),
        types.SimpleNamespace(data=types.SimpleNamespace(_mmap=second_mapping)),
    )

    # Model two ranks at different offsets in the same inode. Each release is
    # confined to its own mapping, regardless of the other rank's progress.
    first.release(17, 23)
    second.release(mmap.PAGESIZE + 31, 29)
    first.close()
    second.close()

    assert first_mapping.advice == [_aligned_release(first_mapping, 17, 23)]
    assert second_mapping.advice == [
        _aligned_release(second_mapping, mmap.PAGESIZE + 31, 29)
    ]


def test_adapter_fails_closed_without_madvise_support(tmp_path):
    path = tmp_path / "mapped.gguf"
    path.write_bytes(b"synthetic")
    reader = types.SimpleNamespace(
        data=types.SimpleNamespace(_mmap=types.SimpleNamespace(size=lambda: 9))
    )
    with pytest.raises(RuntimeError, match="MADV_DONTNEED"):
        _MMapRangeReleaser(str(path), reader)
