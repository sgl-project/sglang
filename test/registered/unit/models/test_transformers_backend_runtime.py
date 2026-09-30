# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    BertConfig,
    Qwen3Config,
    RobertaConfig,
)

from sglang.srt.configs.embedding_model_spec import resolve_embedding_model_spec
from sglang.srt.configs.transformers_backend import transformers_requires_full_sequence
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.models.transformers import (
    TransformersEmbeddingModel,
    TransformersForSequenceClassification,
)
from sglang.srt.models.transformers.attention import AttentionDescriptor
from sglang.srt.models.transformers.multimodal_utils import (
    flatten_encoder_features,
    multimodal_fingerprint,
    validate_multimodal_offsets,
)
from sglang.srt.models.transformers.speculative import AuxiliaryHiddenStateCapture
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


class ReferenceAttentionBackend:
    def forward(self, q, k, v, layer, batch, save_cache, **kwargs):
        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        outputs, offset = [], 0
        for length in batch.extend_seq_lens.tolist():
            query = q[offset : offset + length].transpose(0, 1)
            key = k[offset : offset + length].transpose(0, 1)
            value = v[offset : offset + length].transpose(0, 1)
            result = nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                is_causal=layer.attn_type == AttentionType.DECODER,
                scale=layer.scaling,
                enable_gqa=True,
            )
            outputs.append(result.transpose(0, 1).flatten(1))
            offset += length
        return torch.cat(outputs)


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setattr("sglang.srt.models.transformers.base.get_device", lambda: "cpu")
    group = SimpleNamespace(
        world_size=1, rank_in_group=0, is_first_rank=True, is_last_rank=True
    )
    with (
        get_context().override_server_args(device="cpu"),
        get_parallel().override(
            tp_size=1,
            tp_rank=0,
            attn_tp_size=1,
            attn_tp_rank=0,
            pp_group=group,
            tp_group=group,
        ),
    ):
        yield


def config_for(kind):
    kwargs = dict(
        vocab_size=48,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        pad_token_id=0,
        num_labels=1,
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
        classifier_dropout=0,
    )
    if kind == "qwen3":
        return Qwen3Config(**kwargs, num_key_value_heads=2, head_dim=4)
    return {"bert": BertConfig, "roberta": RobertaConfig}[kind](**kwargs)


def model_settings(config, *, classify=False, pooling="mean"):
    config.architectures = [
        f"{type(config).__name__.removesuffix('Config')}{'ForSequenceClassification' if classify else 'Model'}"
    ]
    if not classify:
        config.pooling_type = pooling
        config.normalize = False
    return SimpleNamespace(
        hf_config=config,
        hf_text_config=config,
        model_path="",
        revision=None,
        trust_remote_code=False,
        embedding_model_spec=resolve_embedding_model_spec(
            config.architectures,
            is_embedding_requested=True,
            is_embedding_gemma=False,
            model_type=config.model_type,
        ),
        is_multimodal=False,
        is_matryoshka=False,
        transformers_embedding_plan=None,
    )


def packed_batch():
    ids = torch.tensor([3, 4, 5, 6, 7, 8, 9])
    return SimpleNamespace(
        input_ids=ids,
        extend_seq_lens=torch.tensor([4, 3]),
        forward_mode=ForwardMode.EXTEND,
        extend_seq_lens_cpu=[4, 3],
        token_type_ids=torch.tensor([0, 0, 1, 1, 0, 1, 1]),
        dimensions=None,
        return_pooled_hidden_states=True,
        is_prefill_only=True,
        token_indices_to_pool=None,
        multi_item_delimiter_indices=None,
    )


@pytest.mark.parametrize("kind", ["bert", "roberta", "qwen3"])
@pytest.mark.parametrize("classify", [False, True])
def test_complete_wrapper_loading_and_packed_outputs(runtime, kind, classify):
    torch.manual_seed(11)
    config = config_for(kind)
    settings = model_settings(config, classify=classify)
    cls = AutoModelForSequenceClassification if classify else AutoModel
    reference = cls.from_config(
        copy.deepcopy(config), attn_implementation="eager"
    ).eval()
    wrapper_cls = (
        TransformersForSequenceClassification
        if classify
        else TransformersEmbeddingModel
    )
    wrapper = wrapper_cls(config=copy.deepcopy(config), model_config=settings)
    loaded = wrapper.load_weights(reference.state_dict().items())
    assert loaded
    assert isinstance(wrapper.attention_instances, nn.ModuleDict)
    batch = packed_batch()
    if kind != "bert":
        batch.token_type_ids = None
    positions = torch.tensor([0, 1, 2, 3, 0, 1, 2])
    expected, start = [], 0
    with torch.no_grad():
        for length in batch.extend_seq_lens.tolist():
            kwargs = {"input_ids": batch.input_ids[start : start + length][None]}
            if batch.token_type_ids is not None:
                kwargs["token_type_ids"] = batch.token_type_ids[start : start + length][
                    None
                ]
            output = reference(**kwargs)
            expected.append(
                output.logits if classify else output.last_hidden_state.mean(1)
            )
            start += length
        with forward_context(ForwardContext(ReferenceAttentionBackend())):
            actual = wrapper(
                batch.input_ids, positions, batch, get_embedding=True
            ).embeddings
    torch.testing.assert_close(actual, torch.cat(expected), rtol=2e-5, atol=2e-6)


def test_initial_checkpoint_requires_all_weights_and_allows_hot_reload(runtime):
    config = config_for("qwen3")
    settings = model_settings(config)
    reference = AutoModel.from_config(
        copy.deepcopy(config), attn_implementation="eager"
    )
    wrapper = TransformersEmbeddingModel(config=config, model_config=settings)
    weights = reference.state_dict()
    missing = "embed_tokens.weight"
    with pytest.raises(ValueError, match="missing required parameters.*embed_tokens"):
        wrapper.load_weights(
            (name, value) for name, value in weights.items() if name != missing
        )
    assert not wrapper._weights_loaded
    wrapper.load_weights(weights.items())
    replacement = torch.randn_like(weights[missing])
    wrapper.load_weights([(missing, replacement)])
    torch.testing.assert_close(
        wrapper.model.get_input_embeddings().weight[: config.vocab_size], replacement
    )


@pytest.mark.parametrize("tied", [False, True])
def test_causal_checkpoint_loading_preserves_tied_embeddings(runtime, tied):
    from transformers import AutoModelForCausalLM

    from sglang.srt.models.transformers import TransformersForCausalLM

    config = config_for("qwen3")
    config.tie_word_embeddings = tied
    reference = AutoModelForCausalLM.from_config(
        copy.deepcopy(config), attn_implementation="eager"
    )
    wrapper = TransformersForCausalLM(config=config)
    weights = reference.state_dict()
    if tied:
        weights.pop("lm_head.weight")
    wrapper.load_weights(weights.items())
    assert (
        wrapper.lm_head.weight is wrapper.model.get_input_embeddings().weight
    ) == tied
    torch.testing.assert_close(
        wrapper.lm_head.weight[: config.vocab_size], reference.lm_head.weight
    )


def test_attention_kv_head_replication_and_encoder_semantics():
    config = config_for("qwen3")
    config.num_attention_heads = 8
    config.num_key_value_heads = 2
    module = SimpleNamespace(
        config=config, layer_idx=0, is_causal=False, head_dim=4, sliding_window=65
    )
    desc = AttentionDescriptor.from_module(module, config, 4)
    assert (desc.num_heads, desc.num_kv_heads, desc.sliding_window, desc.causal) == (
        2,
        1,
        64,
        False,
    )


@pytest.mark.parametrize(
    "kind,pooling,expected",
    [
        ("bert", "last", True),
        ("qwen3", "last", False),
        ("qwen3", "mean", True),
        ("qwen3", "cls", True),
    ],
)
def test_complete_sequence_policy(kind, pooling, expected):
    settings = model_settings(config_for(kind), pooling=pooling)
    assert transformers_requires_full_sequence(settings) == expected


def test_multimodal_features_keep_every_image():
    features = (torch.randn(3, 8), torch.randn(5, 8))
    output = SimpleNamespace(pooler_output=features)
    torch.testing.assert_close(flatten_encoder_features(output), torch.cat(features))


def test_multimodal_hash_includes_layout_revision_and_order():
    feature = torch.arange(12).view(3, 4)
    key = multimodal_fingerprint("revision-a", "image", feature, {"grid": [1, 2, 3]})
    assert key == multimodal_fingerprint(
        "revision-a", "image", feature.clone(), {"grid": [1, 2, 3]}
    )
    assert key != multimodal_fingerprint(
        "revision-b", "image", feature, {"grid": [1, 2, 3]}
    )
    assert key != multimodal_fingerprint(
        "revision-a", "image", feature, {"grid": [1, 3, 2]}
    )
    assert key != multimodal_fingerprint(
        "revision-a", "image", feature.flip(0), {"grid": [1, 2, 3]}
    )


def test_multimodal_offset_validation_rejects_overlap():
    items = [
        SimpleNamespace(modality="image", offsets=[(1, 2)]),
        SimpleNamespace(modality="image", offsets=[(2, 3)]),
    ]
    with pytest.raises(ValueError, match="overlap"):
        validate_multimodal_offsets(items, torch.tensor([0, 9, 9, 9]), {"image": 9})


def test_auxiliary_capture_selected_layers_and_compile():
    class Block(nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale
            self._sglang_attention_key = str(scale)

        def forward(self, x):
            return x * self.scale

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block(2), Block(3), Block(5)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    model = Model()
    capture = AuxiliaryHiddenStateCapture(model, [0, 2], 3)

    def run(x):
        capture.reset()
        output = model(x)
        return output, capture.collect()

    compiled = torch.compile(run, backend="eager", fullgraph=True)
    for length in [2, 5]:
        x = torch.randn(1, length, 4)
        output, states = compiled(x)
        torch.testing.assert_close(output, x * 30)
        torch.testing.assert_close(states[0], x[0] * 2)
        torch.testing.assert_close(states[1], x[0] * 30)
    capture.close()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_mean_pooling_is_independent_of_preceding_requests(dtype):
    from sglang.srt.layers.pooler import PoolingType, pool_hidden_states

    first = torch.full((512, 8), 512.0, dtype=dtype)
    second = torch.arange(24, dtype=torch.float32).reshape(3, 8).to(dtype) / 128
    batch = SimpleNamespace(extend_seq_lens=torch.tensor([512, 3]))
    result = pool_hidden_states(PoolingType.MEAN, torch.cat([first, second]), batch)
    torch.testing.assert_close(
        result[1], second.float().mean(0).to(dtype), rtol=0, atol=0
    )


def test_roberta_position_ids_preserve_padding_and_sequence_boundaries(runtime):
    config = config_for("roberta")
    settings = model_settings(config)
    wrapper = TransformersEmbeddingModel(config=config, model_config=settings)
    ids = torch.tensor([3, 0, 4, 0, 5])
    positions = torch.tensor([0, 1, 2, 0, 1])
    assert wrapper._format_position_ids(positions, ids).tolist() == [[1, 0, 2, 0, 1]]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_scaled_embedding_preserves_hf_dtype_and_scale(runtime, dtype):
    from transformers.models.gemma3.modeling_gemma3 import Gemma3TextScaledWordEmbedding

    from sglang.srt.models.transformers.base import TransformersBase
    from sglang.srt.models.transformers.embedding import ScaledVocabParallelEmbedding

    reference = Gemma3TextScaledWordEmbedding(32, 8, 0, embed_scale=8**0.5).to(
        dtype=dtype
    )

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = reference

        def get_input_embeddings(self):
            return self.embed_tokens

        def set_input_embeddings(self, module):
            self.embed_tokens = module

    owner = TransformersBase.__new__(TransformersBase)
    nn.Module.__init__(owner)
    owner.model = Backbone()
    owner.vocab_size = 32
    owner.embed_scale = None
    owner.replace_vocab_embed_class(owner.model)
    replacement = owner.model.embed_tokens.to(dtype=dtype)
    assert isinstance(replacement, ScaledVocabParallelEmbedding)
    replacement.weight.weight_loader(replacement.weight, reference.weight)
    ids = torch.tensor([1, 3, 7, 12])
    torch.testing.assert_close(replacement(ids), reference(ids), rtol=0, atol=0)


@pytest.mark.parametrize(
    "rope_type,expected",
    [
        ("default", True),
        ("linear", True),
        ("dynamic", False),
        ("dynamic_ntk", False),
        ("longrope", False),
    ],
)
def test_graph_capability_handles_mixed_nested_rope_metadata(rope_type, expected):
    from sglang.srt.models.transformers import can_enable_torch_compile

    config = SimpleNamespace(
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000},
            "full_attention": {"rope_type": rope_type, "factor": 8},
            "rope_type": "default",
            "rope_theta": 1000000,
        }
    )
    assert can_enable_torch_compile(config) is expected


@pytest.mark.parametrize("disabled", ["qkv", "mlp", "norm", "residual", "rope"])
def test_individual_fusion_ablation(runtime, monkeypatch, disabled):
    monkeypatch.setenv("SGLANG_TRANSFORMERS_DISABLED_FUSIONS", disabled)
    config = config_for("qwen3")
    wrapper = TransformersEmbeddingModel(
        config=config, model_config=model_settings(config)
    )
    assert wrapper.transformers_fusion_counts[disabled] == 0
    assert sum(wrapper.transformers_fusion_counts.values()) > 0


def test_unknown_fusion_ablation_is_rejected(runtime, monkeypatch):
    monkeypatch.setenv("SGLANG_TRANSFORMERS_DISABLED_FUSIONS", "qk_v")
    config = config_for("qwen3")
    with pytest.raises(ValueError, match="Unknown Transformers fusions"):
        TransformersEmbeddingModel(config=config, model_config=model_settings(config))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
