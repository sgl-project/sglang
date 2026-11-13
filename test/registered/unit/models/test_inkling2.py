"""Config, checkpoint loading, logits, vision, and multimodal serving."""

import base64
import copy
import io
import json
import sys
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from safetensors.torch import load, save
from torch import nn

from sglang.srt.configs.inkling2 import Inkling2MMConfig, Inkling2ModelConfig
from sglang.srt.configs.model_config import get_hybrid_layer_ids, is_hybrid_swa_model
from sglang.srt.hf_transformers import get_config
from sglang.srt.layers import logits_processor as logits_module
from sglang.srt.layers.logits_processor import LogitsMetadata
from sglang.srt.layers.quantization.inkling2 import Inkling2QuantConfig
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.models.inkling2 import (
    Inkling2ForConditionalGeneration,
    Inkling2LogitsProcessor,
)
from sglang.srt.models.inkling2_vision import Inkling2VisionEncoder
from sglang.srt.models.inkling_common import inkling2_moe as moe
from sglang.srt.multimodal.inkling2 import Inkling2ImageProcessor
from sglang.srt.parser.inkling_renderer import render_inkling_messages
from sglang.srt.parser.inkling_tokenizer import IMAGE_TOKEN_ID
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.inkling2_test_utils import precision_group

register_cpu_ci(est_time=60, suite="base-a-test-cpu")
register_cuda_ci(est_time=37, stage="base-c", runner_config="4-gpu-gb300")

# The released checkpoint's config.json.
_RELEASE_CONFIG = {
    "architectures": ["Inkling2ForConditionalGeneration"],
    "model_type": "inkling2",
    "eos_token_id": 200006,
    "text_config": {
        "model_max_length": 1048576,
        "torch_dtype": "bfloat16",
        "hidden_size": 4608,
        "num_hidden_layers": 48,
        "vocab_size": 201024,
        "num_attention_heads": 48,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "d_rel": 16,
        "rel_extent": 1024,
        "q_bias": False,
        "o_bias": False,
        "log_scaling_n_floor": 128000,
        "log_scaling_alpha": 0.1,
        "rms_norm_eps": 1e-06,
        "pre_norm_pre_affine": False,
        "final_norm_pre_affine": False,
        "use_value_norm_no_gamma": True,
        "qk_l2_clip_threshold": 22.627416997969522,
        "rel_l2_clip_threshold": 8.0,
        "local_layer_ids": [
            0,
            1,
            2,
            3,
            4,
            6,
            7,
            8,
            9,
            10,
            12,
            13,
            14,
            15,
            16,
            18,
            19,
            20,
            21,
            22,
            24,
            25,
            26,
            27,
            28,
            30,
            31,
            32,
            33,
            34,
            36,
            37,
            38,
            39,
            40,
            42,
            43,
            44,
            45,
            46,
        ],
        "sconv_kernel_size": 4,
        "unpadded_vocab_size": 200058,
        "final_logit_softcapping": 30.0,
        "swa_head_dim": 128,
        "swa_num_attention_heads": 48,
        "swa_num_key_value_heads": 8,
        "sliding_window_size": 1024,
        "n_layers_per_attn_res_block": 6,
        "attn_res_normalize_values": True,
        "n_routed_experts": 384,
        "num_experts_per_tok": 8,
        "n_shared_experts": 0,
        "intermediate_size": 2304,
        "mlp_gate_softcap": 4.0,
        "mlp_up_softcap": 25.0,
        "route_scale": 8.0,
        "use_gate_bias": True,
    },
    "vision_config": {
        "temporal_patch_size": 2,
        "native_patch_size": 20,
        "adapter_merge_size": 2,
        "projection_hidden_dim": 4096,
        "projector_bias": False,
        "hidden_size": 1024,
        "num_hidden_layers": 23,
        "num_attention_heads": 16,
        "mlp_ratio": 4.0,
        "posemb_grid_size": 32,
        "layer_norm_eps": 1e-05,
    },
    "mtp_config": {
        "hidden_size": 4608,
        "num_hidden_layers": 5,
        "num_attention_heads": 48,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 18432,
        "max_position_embeddings": 1048576,
        "torch_dtype": "bfloat16",
        "q_bias": False,
        "o_bias": False,
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0,
        "local_layer_ids": [0, 1, 3, 4],
        "sliding_window_size": 1024,
        "block_size": 8,
        "markov_rank": 256,
        "markov_head_type": "rnn_hidden",
        "target_layer_ids": [5, 11, 17, 23, 29, 35, 41, 48],
    },
}


@pytest.fixture
def release_config():
    return copy.deepcopy(_RELEASE_CONFIG)


def test_public_config_load_and_roundtrip(tmp_path, release_config):
    (tmp_path / "config.json").write_text(json.dumps(release_config))
    config = get_config(str(tmp_path), trust_remote_code=False)
    assert isinstance(config, Inkling2MMConfig)
    text = config.get_text_config()
    assert text.padded_vocab_size == release_config["text_config"]["vocab_size"]
    assert text.vocab_size == release_config["text_config"]["unpadded_vocab_size"]
    assert text.max_position_embeddings == 1048576
    assert is_hybrid_swa_model(config.architectures, text)
    local, full = get_hybrid_layer_ids(config.architectures, text)
    assert local == release_config["text_config"]["local_layer_ids"]
    assert sorted(local + full) == list(range(text.num_hidden_layers))
    reloaded = Inkling2MMConfig(**config.to_dict())
    assert reloaded.text_config.to_dict() == text.to_dict()
    assert reloaded.vision_config == release_config["vision_config"]
    assert reloaded.mtp_config == release_config["mtp_config"]


def test_precision_file_is_detected_without_launch_flag(tmp_path, release_config):
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.model_loader.weight_utils import get_quant_config

    precision = dict(
        quant_method="inkling2",
        model=dict(
            parameter_groups=[
                precision_group("mxfp8", ["model.llm.layers.0.mlp.experts.w13_weight"])
            ]
        ),
    )
    (tmp_path / "config.json").write_text(json.dumps(release_config))
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(precision))
    config = ModelConfig(str(tmp_path), trust_remote_code=False)
    assert config.quantization == "inkling2"
    assert config.is_hybrid_swa
    assert config.is_hybrid_swa_compress
    assert not config.is_multimodal
    loaded = get_quant_config(config, LoadConfig(), packed_modules_mapping={})
    assert loaded.quantizes_input("model.llm.layers.0.mlp.experts.w13_weight")


def _target(
    monkeypatch,
    tp_size=1,
    tp_rank=0,
    precision=("mxfp8", "mxfp8"),
    ep_size=1,
    ep_rank=0,
):
    monkeypatch.setattr(
        moe,
        "get_parallel",
        lambda: SimpleNamespace(
            moe_tp_rank=tp_rank,
            moe_tp_size=tp_size,
            moe_ep_rank=ep_rank,
            moe_ep_size=ep_size,
        ),
    )
    monkeypatch.setattr(moe.Inkling2Experts, "prepare_routed_exchange", lambda _: None)
    names = [f"model.llm.layers.0.mlp.experts.{name}_weight" for name in ("w13", "w2")]
    precision = Inkling2QuantConfig(
        dict(
            quant_method="inkling2",
            model=dict(
                parameter_groups=[
                    precision_group(precision[i % 2], [name])
                    for i, name in enumerate(names)
                ]
            ),
        )
    )
    config = SimpleNamespace(
        intermediate_size=128,
        hidden_size=64,
        mlp_gate_softcap=0,
        mlp_up_softcap=0,
        num_experts_per_tok=2,
        route_scale=8.0,
    )
    target = Inkling2ForConditionalGeneration.__new__(Inkling2ForConditionalGeneration)
    nn.Module.__init__(target)
    target.visual, target.quant_config = None, precision
    target.llm = nn.Module()
    layer = nn.Module()
    layer.mlp = nn.Module()
    target.llm.layers = nn.ModuleList([layer])
    with set_default_torch_dtype(torch.bfloat16):
        layer.mlp.experts = moe.Inkling2Experts(
            config, 4, "model.llm.layers.0.mlp.experts.", precision
        )
    return target


def _weights(target):
    generator = torch.Generator().manual_seed(91)
    weights = {}
    for name in sorted(
        target.quant_config.mxfp8_weights | target.quant_config.nvfp4_weights
    ):
        experts = 4
        shape = (experts, 256, 64) if "w13" in name else (experts, 64, 128)
        if name in target.quant_config.nvfp4_weights:
            weights[name] = torch.randint(
                0,
                256,
                (*shape[:-1], shape[-1] // 2),
                dtype=torch.uint8,
                generator=generator,
            )
            weights[name + "_scale"] = torch.rand(
                *shape[:-1], shape[-1] // 16, generator=generator
            ).to(torch.float8_e4m3fn)
            weights[name + "_amax"] = torch.rand(experts, generator=generator)
            weights[name + "_outer_scale_denominator"] = torch.tensor([2688.0])
            continue
        weights[name] = torch.randn(shape, generator=generator).to(torch.float8_e4m3fn)
        weights[name + "_scale"] = torch.randint(
            120,
            130,
            (*shape[:-1], shape[-1] // 32),
            dtype=torch.uint8,
            generator=generator,
        )
    return weights


@pytest.mark.parametrize("scales_first", [False, True])
@pytest.mark.parametrize(
    "precision,tp_size,ep_size",
    [
        (("mxfp8", "mxfp8"), 1, 1),
        (("mxfp8", "mxfp8"), 2, 1),
        (("mxfp8", "mxfp8"), 4, 1),
        (("mxfp8", "mxfp8"), 1, 2),
        (("mxfp8", "mxfp8"), 1, 4),
        (("nvfp4", "nvfp4"), 1, 1),
        (("nvfp4", "nvfp4"), 1, 2),
        (("nvfp4", "nvfp4"), 1, 4),
        (("nvfp4", "mxfp8"), 1, 1),
        (("nvfp4", "mxfp8"), 1, 2),
        (("nvfp4", "mxfp8"), 1, 4),
    ],
)
def test_packed_checkpoint_loads_bitwise_without_requantizing(
    monkeypatch, tp_size, ep_size, scales_first, precision
):
    for rank in range(max(tp_size, ep_size)):
        target = _target(
            monkeypatch,
            tp_size=tp_size,
            tp_rank=rank if tp_size > 1 else 0,
            ep_size=ep_size,
            ep_rank=rank if ep_size > 1 else 0,
            precision=precision,
        )
        weights = load(save(_weights(target)))
        weights = sorted(
            weights.items(), key=lambda item: ("_weight_" in item[0]) != scales_first
        )
        loaded = target.load_weights(weights)
        assert loaded == set(dict(target.named_parameters()))
        for name, value in weights:
            leaf = name.rsplit(".", 1)[1].replace("_weight_", "_")
            actual = getattr(target.llm.layers[0].mlp.experts, leaf)
            expected = value
            if ep_size > 1 and not leaf.endswith("outer_scale_denominator"):
                expected = value.chunk(ep_size, dim=0)[rank]
            if tp_size > 1 and value.ndim == 3:
                expected = value.chunk(tp_size, dim=1 if leaf.startswith("w13") else 2)[
                    rank
                ]
            assert actual.dtype == value.dtype
            assert torch.equal(
                actual.view(torch.uint8), expected.contiguous().view(torch.uint8)
            )


@pytest.fixture
def logits_runtime(monkeypatch):
    monkeypatch.setattr(
        logits_module,
        "get_parallel",
        lambda: SimpleNamespace(
            enable_dp_lm_head=False,
            enable_tp_lm_head_all_to_all=False,
            tp_size=1,
            attn_dp_size=1,
        ),
    )

    monkeypatch.setattr(
        logits_module,
        "get_exec",
        lambda: SimpleNamespace(
            features=SimpleNamespace(
                enable_fp32_lm_head=False,
                enable_mis=False,
                aux_hidden_state_capture=None,
                return_aux_hidden_state_layer_ids=None,
            ),
            deterministic=SimpleNamespace(rl_on_policy_target=None),
        ),
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("buffered", [False, True])
@pytest.mark.parametrize("vocab,padded", [(6, 8), (200058, 201024)])
def test_logits_preserve_padding_for_normalization(
    logits_runtime, device, buffered, vocab, padded
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
    config = Inkling2ModelConfig(
        vocab_size=vocab, padded_vocab_size=padded, final_logit_softcapping=None
    )
    processor = Inkling2LogitsProcessor(config)
    generator = torch.Generator().manual_seed(7)
    hidden = torch.randn(2, 4, generator=generator).to(device)
    head = torch.nn.Linear(4, padded, bias=False, device=device)
    head.weight.data.copy_(torch.randn(padded, 4, generator=generator).to(device))
    buffer = torch.empty(2, padded, device=device) if buffered else None
    metadata = LogitsMetadata(
        forward_mode=ForwardMode.DECODE, next_token_logits_buffer=buffer
    )
    output = processor(
        torch.zeros(2, dtype=torch.long, device=device), hidden, head, metadata
    )
    expected = hidden @ head.weight.T
    expected[:, vocab:] = float("-inf")
    assert output.next_token_logits.shape == (2, padded)
    assert torch.equal(
        output.next_token_logits.view(torch.uint8), expected.view(torch.uint8)
    )
    if buffered:
        assert output.next_token_logits.data_ptr() == buffer.data_ptr()
    expected_logprobs = torch.log_softmax(expected, dim=-1)
    actual_logprobs = torch.log_softmax(output.next_token_logits, dim=-1)
    assert torch.equal(
        actual_logprobs.view(torch.uint8), expected_logprobs.view(torch.uint8)
    )
    assert (output.next_token_logits.argmax(-1) < vocab).all()
    assert config.vocab_size == vocab
    assert processor.input_logprob_processor.vocab_size == padded


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("softcap", [None, 15.0])
def test_mtp_masks_padding_after_softcap(softcap):
    from sglang.kernels.ops.speculative.dspark.markov_sample import MarkovSampleStep

    base = torch.tensor(
        [[0.0, 1, 2, 3, 4, 5, 1000, 1000]], device="cuda", dtype=torch.bfloat16
    )
    expected = base.clone()
    if softcap:
        from sglang.kernels.ops.activation.softcap import softcap_inplace_logits

        softcap_inplace_logits(expected, softcap)
    logits = torch.empty_like(base)
    token = MarkovSampleStep.execute(
        base_logits=base,
        bias=torch.zeros_like(base),
        temperatures=torch.zeros(1, device="cuda"),
        key=0,
        offset=torch.zeros(1, device="cuda", dtype=torch.int64),
        step=0,
        step_dtype=torch.bfloat16,
        logits_out=logits,
        softcap=softcap,
        vocab_limit=6,
    )
    assert torch.equal(
        logits[:, :6].view(torch.uint8), expected[:, :6].view(torch.uint8)
    )
    assert torch.isneginf(logits[:, 6:]).all()
    assert token.item() == 5
    assert torch.isfinite(logits[:, :6]).all()


@pytest.mark.parametrize(
    "token_id,expected", [(5, 5), (6, None), (7, None), (1000, None)]
)
def test_chunked_prefill_excludes_padded_next_prompt_tokens(token_id, expected):
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    batch = ScheduleBatch.init_new(
        reqs=[],
        req_to_token_pool=SimpleNamespace(device="cpu"),
        token_to_kv_pool_allocator=None,
        tree_cache=None,
        model_config=SimpleNamespace(vocab_size=8, token_id_vocab_size=6),
        enable_overlap=False,
        spec_algorithm=None,
        chunked_req=SimpleNamespace(extend_end=1, origin_input_ids=[1, token_id]),
    )
    assert batch.chunked_req_next_prompt_token == expected


@pytest.fixture
def encoder():
    torch.manual_seed(7)
    model = Inkling2VisionEncoder(
        dict(
            temporal_patch_size=2,
            native_patch_size=2,
            adapter_merge_size=2,
            projection_hidden_dim=32,
            projector_bias=True,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            mlp_ratio=2.0,
            posemb_grid_size=4,
            layer_norm_eps=1e-5,
        ),
        text_hidden_size=24,
    ).eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("gamma"):
                param.fill_(0.1)
            elif "ln_" in name and name.endswith("weight"):
                param.fill_(1)
            else:
                param.normal_(std=0.1)
    return model


def test_native_patch_order_and_cls_positions(encoder):
    patches = torch.arange(6 * 3 * 4 * 4, dtype=torch.float32).reshape(6, 3, 4, 4)
    patches = patches.to(torch.bfloat16)
    captured = []
    handle = encoder.backbone.embeddings.register_forward_pre_hook(
        lambda module, args: captured.append(args[0])
    )
    with torch.inference_mode():
        hidden, positions, lengths, cumulative = encoder._embed_images(
            patches, [[1, 2, 2], [1, 1, 2]]
        )
    handle.remove()
    expected = []
    for image, height, width in ((patches[:4], 2, 2), (patches[4:], 1, 2)):
        expected.append(torch.zeros(1, 12, dtype=torch.bfloat16))
        for y in range(height * 2):
            for x in range(width * 2):
                block = image[(y // 2) * width + x // 2]
                patch = block[:, y % 2 * 2 : y % 2 * 2 + 2, x % 2 * 2 : x % 2 * 2 + 2]
                expected.append(patch.permute(1, 2, 0).reshape(1, -1))
    assert torch.equal(captured[0], torch.cat(expected))
    assert lengths == [17, 9]
    assert cumulative.tolist() == [0, 17, 26]
    assert positions[[0, 1, 4, 16, 17, 18, 25]].tolist() == [
        [0, 0],
        [1, 1],
        [1, 4],
        [4, 4],
        [0, 0],
        [1, 1],
        [2, 4],
    ]
    assert torch.equal(hidden[0], hidden[17])


def test_multiple_images_have_independent_attention(encoder):
    patches = torch.randn(3, 2, 4, 4, 3, dtype=torch.bfloat16)
    grid = torch.tensor([[1, 1, 1], [1, 1, 2]])
    with torch.inference_mode():
        expected = encoder(patches, grid)
        modified = patches.clone()
        modified[1:] *= -2
        actual = encoder(modified, grid)
    assert expected.shape == (3, 24)
    assert expected.dtype == torch.bfloat16
    assert torch.isfinite(expected).all()
    assert torch.equal(expected[:1], actual[:1])
    assert not torch.equal(expected[1:], actual[1:])


def _png(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize(
    "width,height,count", [(1, 1, 1), (40, 40, 1), (80, 80, 4), (41, 39, 2), (0, 1, 0)]
)
def test_grid_and_token_count(width, height, count):
    processor = Inkling2ImageProcessor()
    assert processor.count_tokens_for_image(width, height) == count
    assert processor.resolve_virtual_dims((width, height)) == (width, height)
    if count:
        result = processor(Image.new("RGB", (width, height)))
        assert result.vision_patches_bthwc.shape == (count, 2, 40, 40, 3)
        assert result.vision_grid_thw.prod().item() == count
        assert result.vision_original_thw.tolist() == [[1, height, width]]
        assert result.num_tokens == [count]
        assert torch.equal(
            result.vision_patches_bthwc[:, 0], result.vision_patches_bthwc[:, 1]
        )


def test_training_hydration_dimension_and_token_contracts(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(_png(Image.new("RGB", (41, 40))))
    image = {
        "location": str(path),
        "size": (41, 40),
        "num_tokens": 2,
        "image_token_id": 200005,
    }
    processor = Inkling2ImageProcessor()
    result = processor.preprocess_image_for_training(**image)
    assert result.int_tokens == [200005, 200005]
    for changes, message in [
        ({"size": (41, 41)}, "size mismatch"),
        ({"virtual_size": (0, 40)}, "positive integers"),
        ({"num_tokens": 1}, "token count"),
    ]:
        with pytest.raises(ValueError, match=message):
            processor.preprocess_image_for_training(**(image | changes))
    path.write_bytes(b"not an image")
    with pytest.raises(OSError):
        processor(path)


def _processor():
    from sglang.srt.multimodal.inkling2.processor import Inkling2MultimodalProcessor

    processor = Inkling2MultimodalProcessor.__new__(Inkling2MultimodalProcessor)
    processor.image_processor = Inkling2ImageProcessor()
    processor.IMAGE_TOKEN_ID = IMAGE_TOKEN_ID
    return processor


def _data_uri(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def test_image_expansion_matches_training_hydration():
    """Serving expands each rendered placeholder exactly as training hydrates
    the same image, so prompts and pixels agree across the two pipelines."""
    image_processor = Inkling2ImageProcessor()
    content, sources, hydrated = [], [], []
    for index, size in enumerate([(41, 39), (80, 81)]):
        uri = _data_uri(Image.new("RGBA", size, (30, 70, 110, 128)))
        content += [
            {"type": "text", "text": f"Image {index}:"},
            {"type": "image_url", "image_url": {"url": uri}},
        ]
        sources.append(SimpleNamespace(url=uri))
        hydrated.append(
            image_processor.preprocess_image_for_training(
                location=uri,
                size=size,
                num_tokens=image_processor.count_tokens_for_image(*size),
                image_token_id=IMAGE_TOKEN_ID,
            )
        )
    prompt = render_inkling_messages(
        [{"role": "user", "content": content}], spec="inkling2"
    )
    expected_tokens = []
    images = iter(hydrated)
    for token in prompt:
        expected_tokens += (
            next(images).int_tokens if token == IMAGE_TOKEN_ID else [token]
        )
    result = _processor().assemble(prompt, sources)
    assert result.input_ids == expected_tokens
    assert result.im_token_id == IMAGE_TOKEN_ID
    assert len(result.mm_items) == 2
    for item, expected in zip(result.mm_items, hydrated):
        assert torch.equal(item.feature, expected.vision_patches_bthwc)
        assert torch.equal(item.vision_grid_thw, expected.vision_grid_thw)
        assert torch.equal(item.vision_original_thw, expected.vision_original_thw)
        start, end = item.offsets[0]
        assert result.input_ids[start : end + 1] == expected.int_tokens


@pytest.mark.parametrize(
    "tokens,images", [([IMAGE_TOKEN_ID], []), ([], [Image.new("RGB", (1, 1))])]
)
def test_image_count_must_match(tokens, images):
    with pytest.raises(ValueError, match="placeholder count"):
        _processor().assemble(tokens, images)


def test_epd_rebuild_matches_colocated_expansion():
    """A language-only server rebuilds the colocated prompt and image spans from grids."""
    from sglang.srt.multimodal.inkling2.processor import build_encoder_inputs
    from sglang.srt.multimodal.modality import Modality

    processor = _processor()
    images = [Image.new("RGB", (41, 39), "red"), Image.new("RGB", (80, 81), "blue")]
    prompt = [5, IMAGE_TOKEN_ID, 6, 7, IMAGE_TOKEN_ID, 8]
    colocated = processor.assemble(prompt, images)
    encoded = build_encoder_inputs(images, Modality.IMAGE)
    for item, reference in zip(encoded.mm_items, colocated.mm_items):
        assert torch.equal(item.feature, reference.feature)
        assert torch.equal(item.vision_grid_thw, reference.vision_grid_thw)
    rows = sum(
        end - start + 1 for item in colocated.mm_items for start, end in item.offsets
    )
    embeddings = torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2)
    rebuilt = processor.get_validated_mm_data(
        prompt,
        {Modality.IMAGE: embeddings},
        img_grid_thw=encoded["image_grid_thw"],
    )
    assert rebuilt.input_ids == colocated.input_ids
    assert [item.offsets for item in rebuilt.mm_items] == [
        item.offsets for item in colocated.mm_items
    ]
    assert torch.equal(
        torch.cat([item.precomputed_embeddings for item in rebuilt.mm_items]),
        embeddings,
    )
    with pytest.raises(ValueError, match="encoded image count"):
        processor.get_mm_data(
            prompt,
            {Modality.IMAGE: embeddings},
            img_grid_thw=encoded["image_grid_thw"][:1],
        )
    with pytest.raises(ValueError, match="images only"):
        build_encoder_inputs(images, Modality.VIDEO)


def test_server_chat_encoding_dispatch():
    from sglang.srt.entrypoints.openai.chat_encoding import (
        resolve_chat_encoding_spec,
        spec_renders_prompt_ids,
    )

    assert (
        resolve_chat_encoding_spec(
            hf_config=SimpleNamespace(
                architectures=["Inkling2ForConditionalGeneration"],
                model_type="inkling2",
            ),
            tokenizer=SimpleNamespace(chat_template=None),
        )
        == "inkling2"
    )
    assert spec_renders_prompt_ids("inkling2")


def test_multimodal_requires_explicit_enable(tmp_path):

    from sglang.srt.configs.model_config import ModelConfig, is_multimodal_model

    (tmp_path / "config.json").write_text(json.dumps(_RELEASE_CONFIG))
    assert is_multimodal_model(["Inkling2ForConditionalGeneration"])
    assert not ModelConfig(str(tmp_path), trust_remote_code=False).is_multimodal
    enabled = ModelConfig(
        str(tmp_path), trust_remote_code=False, enable_multimodal=True
    )
    assert enabled.is_multimodal
    assert not enabled.is_audio_model
    assert enabled.hf_config.vision_config == _RELEASE_CONFIG["vision_config"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
