import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers import (
    AutoModelForSequenceClassification,
    BertConfig,
    Qwen2Config,
    RobertaConfig,
)

from sglang.srt.configs.embedding_model_spec import (
    EmbeddingTask,
    PoolingStrategy,
    resolve_embedding_model_spec,
)
from sglang.srt.configs.transformers_task import resolve_embedding_pipeline
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.models.transformers.pooling import (
    ClassificationMixin,
    PackedClassificationHead,
    TransformersPooler,
    task_checkpoint_prefixes,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def batch(lengths, **kwargs):
    return SimpleNamespace(
        extend_seq_lens=torch.tensor(lengths),
        input_ids=torch.zeros(sum(lengths), dtype=torch.long),
        dimensions=None,
        token_indices_to_pool=None,
        multi_item_delimiter_indices=None,
        is_prefill_only=True,
        return_pooled_hidden_states=True,
        **kwargs,
    )


def tiny_config(kind, labels=3):
    options = dict(
        vocab_size=48,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_labels=labels,
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
        classifier_dropout=0,
        pad_token_id=0,
    )
    if kind == "qwen2":
        return Qwen2Config(**options, num_key_value_heads=2)
    return {"bert": BertConfig, "roberta": RobertaConfig}[kind](**options)


@pytest.mark.parametrize("kind", ["bert", "roberta", "qwen2"])
@pytest.mark.parametrize("labels", [1, 3])
def test_packed_classification_matches_hf(kind, labels):
    torch.manual_seed(42)
    model = AutoModelForSequenceClassification.from_config(
        tiny_config(kind, labels)
    ).eval()
    ids = torch.tensor([[4, 5, 6, 7, 8], [9, 10, 0, 0, 0], [11, 12, 13, 14, 0]])
    attention_mask = ids.ne(0)
    arguments = dict(input_ids=ids, attention_mask=attention_mask)
    if kind == "bert":
        arguments["token_type_ids"] = torch.tensor(
            [[0, 0, 1, 1, 1], [0, 1, 0, 0, 0], [0, 0, 1, 1, 0]]
        )
    with torch.no_grad():
        reference = model(**arguments).logits
        hidden = model.base_model(**arguments).last_hidden_state
        packed = hidden[attention_mask]
        owner = SimpleNamespace(
            task_head=PackedClassificationHead(model).eval(),
            model=model.base_model,
            pooler=Pooler(
                PoolingType.LAST if kind == "qwen2" else PoolingType.CLS, False
            ),
        )
        result = ClassificationMixin._pool_output(owner, packed, batch([5, 2, 4]))
        torch.testing.assert_close(result.embeddings, reference)
        assert result.embeddings.shape == (3, labels)
        assert result.pooled_hidden_states.shape == (3, 16)
        for order in ([2, 0, 1], [1]):
            reordered = torch.cat(
                [hidden[index, attention_mask[index]] for index in order]
            )
            result = ClassificationMixin._pool_output(
                owner,
                reordered,
                batch([int(attention_mask[index].sum()) for index in order]),
            )
            torch.testing.assert_close(result.embeddings, reference[order])


def test_classifier_requires_head_and_bert_pooler_weights():
    model = AutoModelForSequenceClassification.from_config(tiny_config("bert"))
    owner = SimpleNamespace(
        task_head=PackedClassificationHead(model), model=model.base_model
    )
    required = {f"task_head.{name}" for name, _ in owner.task_head.named_parameters()}
    required |= {
        f"model.pooler.{name}" for name, _ in model.base_model.pooler.named_parameters()
    }
    ClassificationMixin._validate_task_weights(owner, required)
    for name in sorted(required):
        with pytest.raises(ValueError, match="missing required tensors"):
            ClassificationMixin._validate_task_weights(owner, required - {name})


def make_metadata(tmp_path, *, pooling="mean_tokens", normalize=True, dense=True):
    modules = [
        dict(idx=0, path="", type="sentence_transformers.models.Transformer"),
        dict(idx=1, path="1_Pooling", type="sentence_transformers.models.Pooling"),
    ]
    (tmp_path / "1_Pooling").mkdir(exist_ok=True)
    (tmp_path / "1_Pooling/config.json").write_text(
        json.dumps({"word_embedding_dimension": 4, f"pooling_mode_{pooling}": True})
    )
    if dense:
        modules.append(
            dict(idx=2, path="2_Dense", type="sentence_transformers.models.Dense")
        )
        (tmp_path / "2_Dense").mkdir(exist_ok=True)
        (tmp_path / "2_Dense/config.json").write_text(
            json.dumps(
                dict(
                    in_features=4,
                    out_features=3,
                    bias=True,
                    activation_function="torch.nn.Tanh",
                )
            )
        )
    if normalize:
        modules.append(
            dict(
                idx=len(modules),
                path="3_Normalize",
                type="sentence_transformers.models.Normalize",
            )
        )
    (tmp_path / "modules.json").write_text(json.dumps(modules))
    return modules


def spec():
    return resolve_embedding_model_spec(
        ["BertModel"], is_embedding_requested=True, is_embedding_gemma=False
    )


def test_mean_dense_normalize_projection_and_dimensions(tmp_path):
    make_metadata(tmp_path)
    plan = resolve_embedding_pipeline(
        SimpleNamespace(hidden_size=4), spec(), str(tmp_path)
    )
    assert plan.pooling == PoolingStrategy.MEAN
    pooler = TransformersPooler(plan, supports_dimensions=True)
    weight = torch.tensor([[1.0, 0, 0, 1], [0, 1, 1, 0], [1, -1, 1, -1]])
    bias = torch.tensor([0.1, 0.2, 0.3])
    save_file(
        {"linear.weight": weight, "linear.bias": bias},
        str(tmp_path / "2_Dense/model.safetensors"),
    )
    loaded = pooler.load_projection_weights(str(tmp_path))
    assert loaded == {"pooler.stages.0.0.weight", "pooler.stages.0.0.bias"}
    hidden = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
    expected = torch.stack([hidden[:2].mean(0), hidden[2:].mean(0)])
    projected = torch.tanh(torch.nn.functional.linear(expected, weight, bias))
    context = batch([2, 4])
    result = pooler(hidden, context)
    torch.testing.assert_close(
        result.embeddings, torch.nn.functional.normalize(projected, dim=-1)
    )
    torch.testing.assert_close(result.pooled_hidden_states, expected)
    context.dimensions = [2, 3]
    result = pooler(hidden, context)
    for row, actual, dimension in zip(projected, result.embeddings, context.dimensions):
        torch.testing.assert_close(
            actual, torch.nn.functional.normalize(row[:dimension], dim=-1)
        )
    with pytest.raises(ValueError, match="Matryoshka"):
        TransformersPooler(plan)(hidden, context)


@pytest.mark.parametrize("pooling", ["CLS", "MEAN", "LAST"])
def test_explicit_pooling_and_normalization(tmp_path, pooling):
    config = SimpleNamespace(hidden_size=4, pooling_type=pooling, normalize=False)
    plan = resolve_embedding_pipeline(config, spec(), str(tmp_path))
    assert plan.pooling.name == pooling
    assert not plan.normalize
    hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    expected = {
        "CLS": hidden[[0, 2]],
        "LAST": hidden[[1, 4]],
        "MEAN": torch.stack([hidden[:2].mean(0), hidden[2:].mean(0)]),
    }[pooling]
    torch.testing.assert_close(
        TransformersPooler(plan)(hidden, batch([2, 3])).embeddings, expected
    )


@pytest.mark.parametrize("max_seq_length", [None, 128, "long"])
def test_sentence_transformers_max_seq_length_is_read(tmp_path, max_seq_length):
    make_metadata(tmp_path)
    if max_seq_length is not None:
        (tmp_path / "sentence_bert_config.json").write_text(
            json.dumps({"max_seq_length": max_seq_length, "do_lower_case": False})
        )
    if max_seq_length == "long":
        with pytest.raises(ValueError, match="max_seq_length"):
            resolve_embedding_pipeline(
                SimpleNamespace(hidden_size=4), spec(), str(tmp_path)
            )
        return
    plan = resolve_embedding_pipeline(
        SimpleNamespace(hidden_size=4), spec(), str(tmp_path)
    )
    assert plan.max_seq_length == max_seq_length


@pytest.mark.parametrize(
    "head_dim, bidirectional, window, expected",
    [
        (128, False, None, "trtllm_mha"),
        (32, False, None, "fa4"),
        (64, True, 64, "fa4"),
        (64, True, None, "trtllm_mha"),
    ],
)
def test_sm100_default_backend_avoids_trtllm_for_unsupported_encoders(
    monkeypatch, head_dim, bidirectional, window, expected
):
    from sglang.srt.arg_groups import model_override_base as base

    platform = SimpleNamespace(
        is_out_of_tree=lambda: False,
        is_hopper_with_cuda_12_3=False,
        is_sm100=True,
        is_hip=False,
        has_flashinfer=True,
    )
    monkeypatch.setattr(base, "current_platform", platform)
    monkeypatch.setattr(base, "get_platform", lambda: platform)
    monkeypatch.setattr(base, "is_no_spec_infer_or_topk_one", lambda view: True)
    monkeypatch.setattr(base, "resolving_view", lambda args: args)
    monkeypatch.setattr(base, "resolved_view", lambda args: args)
    server_args = SimpleNamespace(
        speculative_algorithm=None, speculative_eagle_topk=None
    )
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["BertModel"]),
        has_asymmetric_kv=False,
        head_dim=head_dim,
        embedding_model_spec=SimpleNamespace(bidirectional_attention=bidirectional),
        sliding_window_size=window,
    )
    assert base.get_default_attn_backend(server_args, False, model_config) == expected


def test_mean_pooling_ignores_graph_padding_rows():
    from sglang.srt.layers.pooler import PoolingType, pool_hidden_states

    lengths = torch.tensor([3, 1, 2], dtype=torch.int64)
    hidden = torch.randn(int(lengths.sum()) + 5, 4)
    batch = SimpleNamespace(extend_seq_lens=lengths)
    pooled = pool_hidden_states(PoolingType.MEAN, hidden, batch)
    expected = torch.stack(
        [hidden[0:3].mean(0), hidden[3:4].mean(0), hidden[4:6].mean(0)]
    )
    torch.testing.assert_close(pooled, expected)


@pytest.mark.parametrize(
    "malformation",
    ["unknown", "missing", "duplicate", "traversal", "multimode", "inconsistent"],
)
def test_invalid_sentence_transformers_metadata_rejected(tmp_path, malformation):
    modules = make_metadata(tmp_path)
    if malformation == "unknown":
        modules[-1]["type"] = "custom.Module"
    elif malformation == "missing":
        modules = modules[:1]
    elif malformation == "duplicate":
        modules[-1]["idx"] = 1
    elif malformation == "traversal":
        modules[1]["path"] = "../pooling"
    elif malformation == "multimode":
        (tmp_path / "1_Pooling/config.json").write_text(
            json.dumps(dict(pooling_mode_mean_tokens=True, pooling_mode_cls_token=True))
        )
    elif malformation == "inconsistent":
        (tmp_path / "2_Dense/config.json").write_text(
            json.dumps(dict(in_features=8, out_features=3))
        )
    (tmp_path / "modules.json").write_text(json.dumps(modules))
    with pytest.raises(ValueError):
        resolve_embedding_pipeline(
            SimpleNamespace(hidden_size=4), spec(), str(tmp_path)
        )


def test_missing_projection_weight_does_not_silently_drop_dense(tmp_path):
    make_metadata(tmp_path)
    pooler = TransformersPooler(
        resolve_embedding_pipeline(
            SimpleNamespace(hidden_size=4), spec(), str(tmp_path)
        )
    )
    with pytest.raises(FileNotFoundError):
        pooler.load_projection_weights(str(tmp_path))


@pytest.mark.parametrize(
    "architecture",
    [
        "RobertaForSequenceClassification",
        "ModernBertForSequenceClassification",
        "FutureModelForSequenceClassification",
    ],
)
def test_task_resolution_recognizes_sequence_classifiers(architecture):
    resolved = resolve_embedding_model_spec(
        [architecture], is_embedding_requested=False, is_embedding_gemma=False
    )
    assert resolved.task == EmbeddingTask.CLASSIFY
    assert not resolved.normalize


def test_pooling_typos_are_errors(tmp_path):
    with pytest.raises(ValueError, match="pooling_type"):
        resolve_embedding_pipeline(
            SimpleNamespace(hidden_size=4, pooling_type="MEN"), spec(), str(tmp_path)
        )


@pytest.mark.parametrize("kind", ["bert", "roberta", "qwen2"])
def test_hf_checkpoint_prefixes_load_all_trained_tensors(kind, tmp_path):
    from safetensors.torch import load_file

    torch.manual_seed(71)
    source = AutoModelForSequenceClassification.from_config(tiny_config(kind)).eval()
    source.save_pretrained(tmp_path, safe_serialization=True)
    target = AutoModelForSequenceClassification.from_config(tiny_config(kind)).eval()
    owner = torch.nn.Module()
    owner.model = target.base_model
    owner.task_head = PackedClassificationHead(target)
    prefixes = task_checkpoint_prefixes(owner.model, classification=True)
    prefixes[""] = "model."
    mapped = {}
    for name, tensor in load_file(str(tmp_path / "model.safetensors")).items():
        prefix = next(
            prefix
            for prefix in sorted(prefixes, key=len, reverse=True)
            if name.startswith(prefix)
        )
        mapped[prefixes[prefix] + name[len(prefix) :]] = tensor
    owner.load_state_dict(mapped, strict=True)
    ClassificationMixin._validate_task_weights(owner, set(mapped))
    ids = torch.tensor([[4, 5, 6], [7, 8, 0]])
    with torch.no_grad():
        expected = source(input_ids=ids, attention_mask=ids.ne(0)).logits
        actual = target(input_ids=ids, attention_mask=ids.ne(0)).logits
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "value", [[], [1, 2], [True], [float("nan")], [float("inf")], "0.5", None]
)
def test_rerank_rejects_embedding_vectors_and_invalid_scalars(value):
    from sglang.srt.entrypoints.openai.serving_rerank import OpenAIServingRerank

    request = SimpleNamespace(
        documents=["document"], return_documents=False, top_n=None
    )
    handler = OpenAIServingRerank.__new__(OpenAIServingRerank)
    with pytest.raises(ValueError):
        handler._build_rerank_response([dict(embedding=value)], request)


def test_rerank_accepts_a_single_classification_score():
    from sglang.srt.entrypoints.openai.serving_rerank import OpenAIServingRerank

    request = SimpleNamespace(
        documents=["document"], return_documents=False, top_n=None
    )
    handler = OpenAIServingRerank.__new__(OpenAIServingRerank)
    responses = handler._build_rerank_response([dict(embedding=[0.7])], request)
    assert responses[0].score == 0.7


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
