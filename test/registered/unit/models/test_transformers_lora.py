from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.lora.torch_ops import sgemm_lora_a_fwd, sgemm_lora_b_fwd
from sglang.srt.models.transformers.lora import (
    TransformersLoRAMixin,
    adapt_transformers_lora,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TorchOpAdapter(nn.Module):
    def __init__(self, weight, a, b, offsets, batch_info, bias=None):
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.a, self.b, self.offsets = a, b, torch.tensor(offsets)
        self.batch_info = batch_info
        self.bias = bias

    def forward(self, x):
        projected = nn.functional.linear(x, self.weight)
        hidden = sgemm_lora_a_fwd(x, self.a, self.batch_info, len(self.offsets) - 1)
        return sgemm_lora_b_fwd(
            hidden, self.b, self.batch_info, self.offsets, projected
        ), self.bias


def metadata():
    return SimpleNamespace(
        use_cuda_graph=False,
        weight_indices_cpu=torch.tensor([0, 1]),
        seg_lens_cpu=torch.tensor([3, 2]),
        lora_ranks_cpu=torch.tensor([2, 1]),
        scalings_cpu=torch.tensor([0.7, 1.1]),
    )


@pytest.mark.parametrize("parts", [[8], [4, 2, 2], [4, 4]])
@pytest.mark.parametrize("packed", [False, True])
def test_tensor_wrapper_runs_segmented_native_torch_lora(parts, packed):
    torch.manual_seed(7)
    x = torch.randn(5, 6)
    weight = torch.randn(sum(parts), 6)
    bias = torch.randn(sum(parts))
    a = torch.randn(2, len(parts) * 2, 6)
    b = torch.randn(2, sum(parts), 2)
    batch = metadata()
    offsets = [0, *torch.tensor(parts).cumsum(0).tolist()]
    adapter = TorchOpAdapter(weight, a, b, offsets, batch, bias)
    wrapped = adapt_transformers_lora(adapter)
    assert wrapped is adapter
    assert isinstance(wrapped, TorchOpAdapter)
    expected = nn.functional.linear(x, weight, bias)
    token_start = 0
    for index, length in enumerate([3, 2]):
        rank = int(batch.lora_ranks_cpu[index])
        output_start = 0
        for shard, width in enumerate(parts):
            delta = (
                x[token_start : token_start + length]
                @ a[index, shard * rank : (shard + 1) * rank].T
            )
            delta = delta @ b[index, output_start : output_start + width, :rank].T
            expected[
                token_start : token_start + length, output_start : output_start + width
            ] += delta * batch.scalings_cpu[index]
            output_start += width
        token_start += length
    value = x[None] if packed else x
    actual = wrapped(value)
    assert isinstance(actual, torch.Tensor)
    torch.testing.assert_close(actual, expected[None] if packed else expected)
    with pytest.raises(ValueError, match="one packed"):
        wrapped(torch.zeros(2, 3, 6))


class TensorEmbedding(nn.Embedding):
    pass


def test_embedding_wrapper_preserves_packed_ids_and_module_identity():
    original = TensorEmbedding(12, 4)
    ids = torch.tensor([[2, 5, 7]])
    expected = original(ids)
    wrapped = adapt_transformers_lora(original, embedding=True)
    assert isinstance(wrapped, TensorEmbedding)
    torch.testing.assert_close(wrapped(ids), expected)
    torch.testing.assert_close(wrapped(ids[0]), expected[0])
    with pytest.raises(ValueError, match="one packed"):
        wrapped(ids.expand(2, -1))


class RuntimeLinear(nn.Linear):
    _hf_returns_tensor = True

    def __init__(self, input_size, parts, *, local_input=None, local_parts=None):
        super().__init__(input_size, sum(parts), bias=False)
        self.input_size, self.output_size = input_size, sum(parts)
        self.output_sizes = parts
        if local_input is not None:
            self.input_size_per_partition = local_input
        if local_parts is not None:
            self.output_partition_sizes = local_parts


class RuntimeModel(TransformersLoRAMixin, nn.Module):
    def __init__(self, fused=True):
        super().__init__()
        self.model = nn.Module()
        layer = nn.Module()
        layer.attn = nn.Module()
        layer.mlp = nn.Module()
        layer.mlp.down_proj = RuntimeLinear(8, [6], local_input=4)
        self.model.layers = nn.ModuleList([layer])
        self._stacked_mapping = {}
        self.packed_modules_mapping = {}
        if fused:
            layer.attn.qkv_proj = RuntimeLinear(6, [4, 2, 2], local_parts=[2, 2, 2])
            self.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
            self._stacked_mapping = {
                f"model.layers.0.attn.{name}": ("model.layers.0.attn.qkv_proj", shard)
                for name, shard in zip(["q_proj", "k_proj", "v_proj"], ["q", "k", "v"])
            }
        else:
            layer.attn.q_proj = RuntimeLinear(6, [4])
            layer.attn.k_proj = RuntimeLinear(6, [2])
            layer.attn.v_proj = RuntimeLinear(6, [2])


def test_target_resolution_uses_actual_fusion_and_local_shapes():
    fused, unfused = RuntimeModel(), RuntimeModel(False)
    assert fused.get_lora_target_modules(["q_proj"]) == {"attn.qkv_proj"}
    assert unfused.get_lora_target_modules(["q_proj"]) == {"attn.q_proj"}
    assert fused.get_hidden_dim("attn.qkv_proj", 0) == (6, 8)
    assert fused.get_lora_buffer_shape("A", "attn.qkv_proj", 0, 4, 2) == (2, 12, 6)
    assert fused.get_lora_buffer_shape("B", "attn.qkv_proj", 0, 4, 2) == (2, 6, 4)
    assert fused.get_lora_buffer_shape("A", "mlp.down_proj", 0, 4, 2) == (2, 4, 4)
    assert fused.get_lora_buffer_shape("B", "mlp.down_proj", 0, 4, 2) == (2, 6, 4)
    assert fused.get_lora_target_modules("all-linear") == {
        "attn.qkv_proj",
        "mlp.down_proj",
    }
    with pytest.raises(ValueError, match="does not match"):
        fused.get_lora_target_modules(["unused_proj"])


def test_partial_qkv_adapter_gets_zero_factors_for_absent_slices():
    model = RuntimeModel()
    a, b = torch.randn(2, 6), torch.randn(2, 2)
    weights = {
        "base_model.model.layers.0.attn.k_proj.lora_A.default.weight": a,
        "base_model.model.layers.0.attn.k_proj.lora_B.default.weight": b,
    }
    model.normalize_lora_weights(weights, 0)
    merged_a = weights["model.layers.0.attn.qkv_proj.lora_A.weight"]
    merged_b = weights["model.layers.0.attn.qkv_proj.lora_B.weight"]
    torch.testing.assert_close(
        merged_a, torch.cat([torch.zeros_like(a), a, torch.zeros_like(a)])
    )
    torch.testing.assert_close(
        merged_b, torch.cat([torch.zeros(4, 2), b, torch.zeros(2, 2)])
    )


def test_unfused_adapter_is_not_silently_fused():
    model = RuntimeModel(False)
    a, b = torch.randn(2, 6), torch.randn(4, 2)
    weights = {
        "base_model.model.layers.0.attn.q_proj.lora_A.weight": a,
        "base_model.model.layers.0.attn.q_proj.lora_B.weight": b,
    }
    model.normalize_lora_weights(weights, 0)
    assert set(weights) == {
        "model.layers.0.attn.q_proj.lora_A.weight",
        "model.layers.0.attn.q_proj.lora_B.weight",
    }
    torch.testing.assert_close(weights["model.layers.0.attn.q_proj.lora_A.weight"], a)


def test_fused_checkpoint_repeats_a_for_native_packed_buffers():
    model = RuntimeModel()
    a, b = torch.randn(2, 6), torch.randn(8, 2)
    weights = {
        "base_model.model.layers.0.attn.qkv_proj.lora_A.weight": a,
        "base_model.model.layers.0.attn.qkv_proj.lora_B.weight": b,
    }
    model.normalize_lora_weights(weights, 0)
    torch.testing.assert_close(
        weights["model.layers.0.attn.qkv_proj.lora_A.weight"], a.repeat(3, 1)
    )
    torch.testing.assert_close(weights["model.layers.0.attn.qkv_proj.lora_B.weight"], b)


def test_missing_factor_and_unknown_target_raise():
    model = RuntimeModel()
    with pytest.raises(ValueError, match="Both LoRA factors"):
        model.normalize_lora_weights(
            {"model.layers.0.attn.q_proj.lora_A.weight": torch.randn(2, 6)}, 0
        )
    with pytest.raises(ValueError, match="no matching"):
        model.normalize_lora_weights(
            {"model.layers.0.attn.unknown.lora_A.weight": torch.randn(2, 6)}, 0
        )


def test_encoder_layer_indices_and_embedding_aliases():
    model = RuntimeModel()
    assert (
        model.get_lora_layer_id(
            "base_model.model.bert.encoder.layer.3.attention.self.query.lora_A.weight"
        )
        == 3
    )
    assert (
        model.normalize_lora_weight_name(
            "base_model.model.bert.embeddings.word_embeddings.lora_A.weight"
        )
        == "base_model.model.bert.embeddings.embed_tokens.lora_A.weight"
    )


def test_real_manager_wraps_native_linear_and_runs_active_lora():
    from sglang.srt.lora.backend.torch_backend import TorchNativeLoRABackend
    from sglang.srt.lora.layers import BaseLayerWithLoRA
    from sglang.srt.lora.lora_manager import LoRAManager
    from sglang.srt.models.transformers.layers import HFCompatibleReplicatedLinear
    from sglang.srt.runtime_context import get_parallel

    class Model(TransformersLoRAMixin, nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            layer = nn.Module()
            layer.proj = HFCompatibleReplicatedLinear(6, 8, bias=False)
            self.model.layers = nn.ModuleList([layer])
            self.model.get_input_embeddings = lambda: None
            self._stacked_mapping = {}
            self.packed_modules_mapping = {}

    torch.manual_seed(12)
    with get_parallel().override(tp_size=1, tp_rank=0):
        model = Model()
        manager = LoRAManager.__new__(LoRAManager)
        manager.base_model = model
        manager.base_hf_config = SimpleNamespace(num_hidden_layers=1)
        manager.target_modules = {"proj"}
        manager.lora_backend = TorchNativeLoRABackend(2, torch.device("cpu"))
        manager.init_lora_modules()
        wrapper = model.model.layers[0].proj
        assert isinstance(wrapper, BaseLayerWithLoRA)
        weight = torch.randn(8, 6)
        wrapper.weight.data.copy_(weight)
        a, b = torch.randn(2, 2, 6), torch.randn(2, 8, 2)
        wrapper.set_lora_info(a, b)
        manager.lora_backend.batch_info = metadata()
        x = torch.randn(1, 5, 6)
        expected = nn.functional.linear(x, weight)
        expected[:, :3] += (x[:, :3] @ a[0].T @ b[0].T) * 0.7
        expected[:, 3:] += (x[:, 3:] @ a[1, :1].T @ b[1, :, :1].T) * 1.1
        torch.testing.assert_close(wrapper(x), expected)
        manager.lora_backend.batch_info = None
        torch.testing.assert_close(wrapper(x), nn.functional.linear(x, weight))


def test_real_adapter_loader_uses_model_fusion_mapping():
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.lora.backend.torch_backend import TorchNativeLoRABackend
    from sglang.srt.lora.lora import LoRAAdapter
    from sglang.srt.lora.lora_config import LoRAConfig

    model = RuntimeModel()
    config = LoRAConfig.from_dict(
        dict(peft_type="LORA", target_modules=["k_proj"], r=2, lora_alpha=2)
    )
    adapter = LoRAAdapter(
        "test",
        config,
        SimpleNamespace(num_hidden_layers=1),
        LoadConfig(),
        TorchNativeLoRABackend(1, torch.device("cpu")),
        model,
    )
    a, b = torch.randn(2, 6), torch.randn(2, 2)
    adapter.initialize_weights_from_tensors(
        {
            "base_model.model.layers.0.attn.k_proj.lora_A.weight": a,
            "base_model.model.layers.0.attn.k_proj.lora_B.weight": b,
        }
    )
    torch.testing.assert_close(
        adapter.layers[0].weights["model.layers.0.attn.qkv_proj.lora_A.weight"][2:4], a
    )
    torch.testing.assert_close(
        adapter.layers[0].weights["model.layers.0.attn.qkv_proj.lora_B.weight"][4:6], b
    )


@pytest.mark.parametrize("kind", ["gemma", "bart"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_scaled_embedding_native_lora_matches_peft(kind, dtype):
    pytest.importorskip("peft", minversion="0.21.0")
    from peft import LoraConfig
    from peft.tuners.lora.layer import Embedding
    from transformers.models.bart.modeling_bart import BartScaledWordEmbedding
    from transformers.models.gemma3.modeling_gemma3 import Gemma3TextScaledWordEmbedding

    from sglang.srt.lora.backend.torch_backend import TorchNativeLoRABackend
    from sglang.srt.lora.layers import VocabParallelEmbeddingWithLoRA
    from sglang.srt.models.transformers.embedding import (
        ScaledVocabParallelEmbedding,
        scaled_embedding_contract,
    )
    from sglang.srt.runtime_context import get_parallel

    torch.manual_seed(17)
    cls = Gemma3TextScaledWordEmbedding if kind == "gemma" else BartScaledWordEmbedding
    base_reference = cls(12, 4, 0, embed_scale=1.234567)
    base_reference.weight = nn.Parameter(base_reference.weight.to(dtype))
    reference = Embedding(
        base_reference, "default", LoraConfig(r=2, lora_alpha=2), r=2, lora_alpha=2
    ).eval()
    a, b = torch.randn(2, 2, 12).to(dtype), torch.randn(2, 4, 2).to(dtype)
    reference.lora_embedding_A["default"].data.copy_(a[0])
    reference.lora_embedding_B["default"].data.copy_(b[0])
    ids = torch.tensor([[2, 5, 7, 3, 9]])
    with (
        get_parallel().override(
            tp_size=1,
            tp_rank=0,
            tp_group=SimpleNamespace(world_size=1, rank_in_group=0),
        ),
        torch.no_grad(),
    ):
        base = ScaledVocabParallelEmbedding(12, 4, params_dtype=dtype)
        base.weight[:12].copy_(base_reference.weight)
        base.set_scale(
            base_reference.embed_scale, scaled_embedding_contract(base_reference)
        )
        backend = TorchNativeLoRABackend(2, torch.device("cpu"))
        wrapped = adapt_transformers_lora(
            VocabParallelEmbeddingWithLoRA(base, backend), embedding=True
        )
        wrapped.set_lora_info(None, a, b)
        batch = metadata()
        batch.lora_ranks_cpu = torch.tensor([2, 0])
        batch.scalings_cpu = torch.tensor([1.0, 0.0])
        backend.batch_info = batch
        expected = reference(ids[0], adapter_names=["default"] * 3 + ["__base__"] * 2)
        actual = wrapped(ids)
        assert actual.dtype == dtype
        torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)
        backend.batch_info = None
        torch.testing.assert_close(wrapped(ids), base_reference(ids), rtol=0, atol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
