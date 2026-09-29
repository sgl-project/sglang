# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team

import copy
from types import SimpleNamespace

import pytest
import torch
from test_transformers_backend_runtime import model_settings
from test_transformers_backend_runtime import runtime as runtime_fixture
from transformers import AutoModel, ModernBertConfig

from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.models.transformers import TransformersEmbeddingModel
from sglang.test.ci.ci_register import register_cpu_ci

runtime = runtime_fixture
register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TensorKVPool:
    def __init__(self, size):
        self.size = size
        self.buffers = {}

    def set_kv_buffer(self, layer, locations, key, value):
        if layer.layer_id not in self.buffers:
            self.buffers[layer.layer_id] = [
                x.new_zeros((self.size, *x.shape[1:])) for x in (key, value)
            ]
        for buffer, tensor in zip(self.buffers[layer.layer_id], (key, value)):
            buffer[locations.loc] = tensor

    def get_key_buffer(self, layer_id):
        return self.buffers[layer_id][0]

    def get_value_buffer(self, layer_id):
        return self.buffers[layer_id][1]


def native_batch(lengths):
    count = sum(lengths)
    locations = torch.arange(count).flip(0)
    request_tokens = torch.zeros(len(lengths), max(lengths), dtype=torch.long)
    offset = 0
    for index, length in enumerate(lengths):
        request_tokens[index, :length] = locations[offset : offset + length]
        offset += length
    batch = SimpleNamespace(
        input_ids=torch.arange(count) + 3,
        forward_mode=ForwardMode.EXTEND,
        seq_lens=torch.tensor(lengths),
        extend_seq_lens=torch.tensor(lengths),
        extend_seq_lens_cpu=lengths,
        extend_prefix_lens=torch.zeros(len(lengths), dtype=torch.long),
        req_pool_indices=torch.arange(len(lengths)),
        out_cache_loc=locations,
        encoder_lens=None,
        token_type_ids=None,
        dimensions=None,
        return_pooled_hidden_states=True,
        is_prefill_only=True,
        token_indices_to_pool=None,
        multi_item_delimiter_indices=None,
    )
    backend = TorchNativeAttnBackend(
        SimpleNamespace(
            device="cpu",
            req_to_token_pool=SimpleNamespace(req_to_token=request_tokens),
            token_to_kv_pool=TensorKVPool(count),
        )
    )
    return batch, backend


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("window", [None, 0, 2])
def test_paged_extend_window_matches_dense_oracle(runtime, causal, window):
    torch.manual_seed(24)
    batch, backend = native_batch([9, 6])
    query = torch.randn(15, 4, 8)
    key, value = torch.randn(15, 2, 8), torch.randn(15, 2, 8)
    layer = RadixAttention(
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        scaling=0.3,
        layer_id=0,
        sliding_window_size=-1 if window is None else window,
        attn_type=AttentionType.DECODER if causal else AttentionType.ENCODER_ONLY,
    )
    actual = backend.forward_extend(query.flatten(1), key, value, layer, batch)
    expected, start = [], 0
    for length in batch.extend_seq_lens_cpu:
        q, k, v = [
            x[start : start + length].transpose(0, 1) for x in (query, key, value)
        ]
        scores = q @ k.repeat_interleave(2, dim=0).transpose(-1, -2) * layer.scaling
        positions = torch.arange(length)
        distance = positions[:, None] - positions[None, :]
        mask = torch.ones(length, length, dtype=torch.bool)
        if causal:
            mask &= distance >= 0
        if window is not None:
            mask &= distance.abs() <= window
        scores.masked_fill_(~mask, -torch.inf)
        expected.append(
            (scores.softmax(-1) @ v.repeat_interleave(2, dim=0))
            .transpose(0, 1)
            .flatten(1)
        )
        start += length
    torch.testing.assert_close(actual, torch.cat(expected), atol=2e-6, rtol=2e-5)


def modernbert_wrapper():
    config = ModernBertConfig(
        vocab_size=48,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        local_attention=4,
        global_attn_every_n_layers=2,
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=1,
        sep_token_id=2,
        attention_dropout=0,
        embedding_dropout=0,
        mlp_dropout=0,
    )
    settings = model_settings(config)
    reference = AutoModel.from_config(
        copy.deepcopy(config), attn_implementation="eager"
    ).eval()
    wrapper = TransformersEmbeddingModel(
        config=copy.deepcopy(config), model_config=settings
    )
    wrapper.load_weights(reference.state_dict().items())
    return reference, wrapper


def test_modernbert_packed_window_matches_huggingface(runtime):
    torch.manual_seed(31)
    reference, wrapper = modernbert_wrapper()
    assert [
        layer.sliding_window_size for layer in wrapper.attention_instances.values()
    ] == [-1, 2]
    batch, backend = native_batch([9, 6])
    positions = torch.cat(
        [torch.arange(length) for length in batch.extend_seq_lens_cpu]
    )
    expected, start = [], 0
    with torch.no_grad():
        for length in batch.extend_seq_lens_cpu:
            expected.append(
                reference(
                    batch.input_ids[start : start + length][None]
                ).last_hidden_state.mean(1)
            )
            start += length
        with forward_context(ForwardContext(backend)):
            actual = wrapper(
                batch.input_ids, positions, batch, get_embedding=True
            ).embeddings
    torch.testing.assert_close(actual, torch.cat(expected), atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize(
    "backend", ["torch_native", "fa3", "fa4", "flashinfer", "triton", "trtllm_mha"]
)
def test_bidirectional_window_backend_capability(runtime, backend):
    _, wrapper = modernbert_wrapper()
    if backend in {"torch_native", "fa3", "fa4"}:
        wrapper.validate_attention_backend(backend, backend)
    else:
        with pytest.raises(ValueError, match="bidirectional sliding-window"):
            wrapper.validate_attention_backend(backend, backend)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
