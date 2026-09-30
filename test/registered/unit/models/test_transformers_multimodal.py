# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team

import copy
from array import array
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch import nn
from transformers import (
    AutoModel,
    AutoModelForImageTextToText,
    CLIPImageProcessor,
    CLIPVisionConfig,
    LlamaConfig,
    LlavaConfig,
    LlavaProcessor,
    PreTrainedTokenizerFast,
    Qwen2_5_VLConfig,
    Qwen2VLConfig,
)

from sglang.srt.managers.mm_utils import init_mm_embedding_cache
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.models.transformers import TransformersMultiModalForCausalLM
from sglang.srt.models.transformers.multimodal import MultiModalMixin
from sglang.srt.models.transformers.multimodal_utils import (
    flatten_encoder_features,
    placeholder_spans,
)
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
from sglang.srt.multimodal.processors.transformers_auto import (
    TransformersAutoMultimodalProcessor,
    _uses_mrope,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setattr("sglang.srt.models.transformers.base.get_device", lambda: "cpu")
    group = SimpleNamespace(
        world_size=1, rank_in_group=0, is_first_rank=True, is_last_rank=True
    )
    with (
        get_context().override_server_args(
            device="cpu",
            mm_process_config={},
            tokenizer_worker_num=1,
            mm_io_worker_num=1,
            mm_processor_worker_num=1,
        ),
        get_parallel().override(
            tp_size=1,
            tp_rank=0,
            attn_tp_size=1,
            attn_tp_rank=0,
            pp_group=group,
            tp_group=group,
        ),
    ):
        init_mm_embedding_cache(1024 * 1024)
        yield
        init_mm_embedding_cache(0)


def tiny_llava():
    return LlavaConfig(
        text_config=LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            pad_token_id=0,
        ),
        vision_config=CLIPVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=8,
            patch_size=4,
        ),
        image_token_index=2,
        image_seq_length=4,
        vision_feature_layer=-2,
        vision_feature_select_strategy="default",
        architectures=["LlavaForConditionalGeneration"],
    )


def tiny_llava_processor(config):
    tokenizer_backend = Tokenizer(
        WordLevel(
            {"<pad>": 0, "<unk>": 1, "<image>": 2, "start": 3, "end": 4},
            unk_token="<unk>",
        )
    )
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        pad_token="<pad>",
        unk_token="<unk>",
        additional_special_tokens=["<image>"],
    )
    hf_processor = LlavaProcessor(
        image_processor=CLIPImageProcessor(
            size={"shortest_edge": 8}, crop_size={"height": 8, "width": 8}
        ),
        tokenizer=tokenizer,
        patch_size=4,
        vision_feature_select_strategy="default",
        num_additional_image_tokens=1,
    )
    return TransformersAutoMultimodalProcessor(
        config,
        SimpleNamespace(base_gpu_id=0, tp_size=1),
        hf_processor,
        transport_mode="default",
        skip_mm_pool=True,
    )


class ReferencePagedAttention:
    def __init__(self):
        self.cache = {}

    def forward(self, q, k, v, layer, batch, save_cache, **kwargs):
        q = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)
        prefix = batch.extend_prefix_lens_cpu[0]
        if prefix:
            old_k, old_v = self.cache[layer.layer_id]
            k = torch.cat((old_k[:prefix], k))
            v = torch.cat((old_v[:prefix], v))
        self.cache[layer.layer_id] = (k, v)
        mask = (
            torch.arange(k.shape[0])[None, :]
            <= prefix + torch.arange(q.shape[0])[:, None]
        )
        output = nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1),
            k.transpose(0, 1),
            v.transpose(0, 1),
            attn_mask=mask,
            scale=layer.scaling,
            enable_gqa=True,
        )
        return output.transpose(0, 1).flatten(1)


def batch_for(mm_inputs, length, prefix=0):
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        mm_inputs=[mm_inputs],
        extend_prefix_lens_cpu=[prefix],
        extend_seq_lens_cpu=[length],
        extend_seq_lens=torch.tensor([length]),
        token_type_ids=None,
        mrope_positions=None,
        contains_mm_inputs=lambda: True,
    )


def test_llava_real_processor_wrapper_cache_and_chunk_boundary(runtime):
    torch.manual_seed(83)
    config = tiny_llava()
    processor = tiny_llava_processor(config)
    images = [
        Image.new("RGB", (8, 8), color=(250, 20, 40)),
        Image.new("RGB", (8, 8), color=(20, 240, 10)),
    ]
    processed = processor._apply_hf_processor(
        "start <image> end <image> end", images=images
    )
    ids = processed["input_ids"].flatten()
    items = processor._build_mm_items(processed, ids)
    assert len(items) == 1 and len(items[0].offsets) == 2
    mm_inputs = MultimodalInputs(mm_items=items, im_token_id=2)
    reference = AutoModelForImageTextToText.from_config(
        copy.deepcopy(config), attn_implementation="eager"
    ).eval()
    wrapper = TransformersMultiModalForCausalLM(config=copy.deepcopy(config))
    wrapper.load_weights(reference.state_dict().items())
    padded_ids = torch.tensor(
        wrapper.pad_input_ids(array("q", ids.tolist()), mm_inputs)
    )
    assert not torch.equal(padded_ids, ids)
    counts = []
    handle = wrapper.model.vision_tower.register_forward_hook(
        lambda *args: counts.append(1)
    )
    positions = torch.arange(ids.numel())
    with torch.no_grad():
        expected = reference.model(**processed).last_hidden_state[0]
        with forward_context(ForwardContext(ReferencePagedAttention())):
            whole = wrapper._forward_hidden_states(
                padded_ids, positions, batch_for(mm_inputs, ids.numel())
            )
        assert len(counts) == 1
        with forward_context(ForwardContext(ReferencePagedAttention())):
            cached = wrapper._forward_hidden_states(
                padded_ids, positions, batch_for(mm_inputs, ids.numel())
            )
        assert len(counts) == 1
        init_mm_embedding_cache(1024 * 1024)
        split = 3
        with forward_context(ForwardContext(ReferencePagedAttention())):
            first = wrapper._forward_hidden_states(
                padded_ids[:split], positions[:split], batch_for(mm_inputs, split)
            )
            second = wrapper._forward_hidden_states(
                padded_ids[split:],
                positions[split:],
                batch_for(mm_inputs, ids.numel() - split, split),
            )
        assert len(counts) == 2
    handle.remove()
    processor.io_executor.shutdown(wait=False)
    processor.cpu_executor.shutdown(wait=False)
    torch.testing.assert_close(whole, expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(cached, expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        torch.cat((first, second)), expected, rtol=2e-5, atol=2e-6
    )


class EncoderAdapter(MultiModalMixin, nn.Module):
    def __init__(self, model):
        nn.Module.__init__(self)
        self.model = model
        self.config = model.config
        self.text_config = model.config.text_config


@pytest.mark.parametrize("kind", ["qwen2", "qwen2_5"])
def test_qwen_visual_encoder_preserves_all_unequal_image_features(kind):
    text = dict(
        vocab_size=32,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        rope_parameters={"rope_type": "default", "mrope_section": [2, 2, 2]},
    )
    if kind == "qwen2":
        config = Qwen2VLConfig(
            text_config=text,
            vision_config=dict(
                depth=2,
                embed_dim=16,
                hidden_size=24,
                mlp_ratio=2,
                num_heads=4,
                patch_size=2,
                spatial_merge_size=2,
                temporal_patch_size=2,
            ),
        )
    else:
        config = Qwen2_5_VLConfig(
            text_config=text,
            vision_config=dict(
                depth=2,
                hidden_size=16,
                out_hidden_size=24,
                intermediate_size=32,
                num_heads=4,
                patch_size=2,
                spatial_merge_size=2,
                temporal_patch_size=2,
                fullatt_block_indexes=[0, 1],
                window_size=8,
            ),
        )
    torch.manual_seed(79)
    model = AutoModel.from_config(config, attn_implementation="eager").eval()
    adapter = EncoderAdapter(model)
    grid = torch.tensor([[1, 2, 2], [1, 2, 4]])
    pixels = torch.randn(12, 24)
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=pixels,
        offsets=[(1, 1), (3, 4)],
        model_specific_data={"image_grid_thw": grid},
    )
    with torch.no_grad():
        expected = flatten_encoder_features(
            model.get_image_features(pixels, image_grid_thw=grid)
        )
        actual = adapter.get_image_feature([item])
    assert actual.shape == (3, 24)
    torch.testing.assert_close(actual, expected)
    assert adapter._uses_mrope_positions() and _uses_mrope(config)
    item.offsets = [(1, 2)]
    with pytest.raises(ValueError, match="tokens for.*placeholders"):
        adapter.get_image_feature([item])


@pytest.mark.parametrize(
    "ids,expected",
    [
        ([2, 2, 3, 2], [(0, 1), (3, 3)]),
        ([2, 2], [(0, 1)]),
        ([3, 2, 2], [(1, 2)]),
        ([2, 3], [(0, 0)]),
        ([3, 4], []),
        ([], []),
    ],
)
def test_placeholder_boundaries_are_not_circular(ids, expected):
    assert placeholder_spans(torch.tensor(ids), 2) == expected


def test_precomputed_processor_items_have_distinct_cache_identities(runtime):
    processor = tiny_llava_processor(tiny_llava())
    ids = torch.tensor([2, 2])
    first = processor._build_mm_items(
        {"precomputed_embeddings": torch.zeros(2, 16)}, ids
    )
    second = processor._build_mm_items(
        {"precomputed_embeddings": torch.ones(2, 16)}, ids
    )
    assert first[0].hash != second[0].hash
    assert first[0].offsets == second[0].offsets == [(0, 1)]
    processor.io_executor.shutdown(wait=False)
    processor.cpu_executor.shutdown(wait=False)


def test_qwen_video_positions_include_frame_timing():
    processor = TransformersAutoMultimodalProcessor.__new__(
        TransformersAutoMultimodalProcessor
    )
    processor.mm_tokens = MultimodalSpecialTokens(image_token_id=2, video_token_id=8)
    processor._spatial_merge_size = 2
    processor._tokens_per_second = 4
    processor._vision_start_token_id = 7
    processor._model_type = "qwen2_5_vl"
    input_ids = [3, 7, 8, 8, 9, 4]
    grid = torch.tensor([[2, 2, 2]])
    fast, _ = processor._compute_mrope_positions(
        input_ids, video_grid_thw=grid, second_per_grid_ts=torch.tensor([0.5])
    )
    slow, _ = processor._compute_mrope_positions(
        input_ids, video_grid_thw=grid, second_per_grid_ts=torch.tensor([2.0])
    )
    assert fast[0, 3] - fast[0, 2] == 2
    assert slow[0, 3] - slow[0, 2] == 8


def test_target_verify_does_not_reencode_prompt_images(monkeypatch):
    model = AutoModel.from_config(tiny_llava(), attn_implementation="eager").eval()
    adapter = EncoderAdapter(model)
    adapter._mm_cache_enabled = True
    batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        mm_inputs=[object()],
        mrope_positions=None,
        contains_mm_inputs=lambda: True,
    )
    monkeypatch.setattr(
        adapter,
        "_run_hf_backbone",
        lambda **kwargs: kwargs["input_embeds"],
        raising=False,
    )
    ids = torch.tensor([3, 4])
    actual = adapter._forward_hidden_states(ids, torch.arange(2), batch)
    torch.testing.assert_close(actual, model.get_input_embeddings()(ids))


@pytest.mark.parametrize("kind", ["qwen2", "qwen2_5"])
def test_qwen_complete_wrapper_cached_and_chunked_images(runtime, kind):
    text = dict(
        vocab_size=32,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        rope_parameters={"rope_type": "default", "mrope_section": [2, 2, 2]},
    )
    if kind == "qwen2":
        config = Qwen2VLConfig(
            text_config=text,
            vision_config=dict(
                depth=2,
                embed_dim=16,
                hidden_size=24,
                mlp_ratio=2,
                num_heads=4,
                patch_size=2,
                spatial_merge_size=2,
                temporal_patch_size=2,
            ),
        )
        config.architectures = ["Qwen2VLForConditionalGeneration"]
    else:
        config = Qwen2_5_VLConfig(
            text_config=text,
            vision_config=dict(
                depth=2,
                hidden_size=16,
                out_hidden_size=24,
                intermediate_size=32,
                num_heads=4,
                patch_size=2,
                spatial_merge_size=2,
                temporal_patch_size=2,
                fullatt_block_indexes=[0, 1],
                window_size=8,
            ),
        )
        config.architectures = ["Qwen2_5_VLForConditionalGeneration"]
    config.image_token_id, config.video_token_id = 2, 8
    config.vision_start_token_id, config.vision_end_token_id = 7, 9
    torch.manual_seed(97)
    reference = AutoModelForImageTextToText.from_config(
        copy.deepcopy(config), attn_implementation="eager"
    ).eval()
    wrapper = TransformersMultiModalForCausalLM(config=copy.deepcopy(config))
    wrapper.load_weights(reference.state_dict().items())
    ids = torch.tensor([3, 7, 2, 9, 4, 7, 2, 2, 9, 4])
    grid = torch.tensor([[1, 2, 2], [1, 2, 4]])
    pixels = torch.randn(12, 24)
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=pixels,
        offsets=[(2, 2), (6, 7)],
        model_specific_data={"image_grid_thw": grid},
    )
    item.set_hash(3751)
    mm_inputs = MultimodalInputs(mm_items=[item], im_token_id=2)
    padded = torch.tensor(wrapper.pad_input_ids(array("q", ids.tolist()), mm_inputs))
    positions, _ = reference.model.get_rope_index(
        ids[None], mm_token_type_ids=(ids == 2).long()[None], image_grid_thw=grid
    )
    positions = positions[:, 0]
    calls = []
    handle = wrapper.model.visual.register_forward_hook(lambda *args: calls.append(1))
    with torch.no_grad():
        expected = reference.model(
            input_ids=ids[None],
            pixel_values=pixels,
            image_grid_thw=grid,
            position_ids=positions[:, None],
        ).last_hidden_state[0]
        whole_batch = batch_for(mm_inputs, len(ids))
        whole_batch.mrope_positions = positions
        with forward_context(ForwardContext(ReferencePagedAttention())):
            actual = wrapper._forward_hidden_states(
                padded, torch.arange(len(ids)), whole_batch
            )
        split = 7
        with forward_context(ForwardContext(ReferencePagedAttention())):
            first_batch = batch_for(mm_inputs, split)
            first_batch.mrope_positions = positions[:, :split]
            first = wrapper._forward_hidden_states(
                padded[:split], torch.arange(split), first_batch
            )
            second_batch = batch_for(mm_inputs, len(ids) - split, split)
            second_batch.mrope_positions = positions[:, split:]
            second = wrapper._forward_hidden_states(
                padded[split:], torch.arange(split, len(ids)), second_batch
            )
    handle.remove()
    assert len(calls) == 1
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        torch.cat((first, second)), expected, rtol=2e-5, atol=2e-6
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
