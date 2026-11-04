# SPDX-License-Identifier: Apache-2.0

import json
import os
import tempfile
from contextlib import nullcontext
from types import SimpleNamespace

import PIL.Image
import pytest
import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.lingbot_video_moe import (
    LingBotVideoMoEPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.lingbot_video_moe import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_NEGATIVE_PROMPT_IMAGE,
    REFINER_BASE_HEIGHT,
    REFINER_BASE_WIDTH,
    REFINER_PIPELINE_NAME,
    LingBotVideoMoESamplingParams,
)
from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.registry import _get_config_info, get_model_info
from sglang.multimodal_gen.runtime.distributed import sp_shard_utils as sps
from sglang.multimodal_gen.runtime.layers.moe import (
    LingBotVideoGroupedExperts,
    LingBotVideoRouter,
)
from sglang.multimodal_gen.runtime.managers.forward_context import (
    get_forward_context,
)
from sglang.multimodal_gen.runtime.models.dits import (
    lingbot_video_moe as dits_lingbot_video_moe,
)
from sglang.multimodal_gen.runtime.models.dits.lingbot_video_moe import (
    LingBotVideoAttention,
    LingBotVideoBlock,
    LingBotVideoTransformer3DModel,
    _joint_position_ids,
    make_joint_position_ids,
)
from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import (
    VaeFastPathGate,
    register_vae_fast_path_gate,
)
from sglang.multimodal_gen.runtime.pipelines import (
    lingbot_video_moe as pipeline_lingbot_video_moe,
)
from sglang.multimodal_gen.runtime.pipelines.lingbot_video_moe import (
    LingBotVideoPipeline,
)
from sglang.multimodal_gen.runtime.pipelines.lingbot_video_moe_refiner import (
    LingBotVideoRefinerPipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    decoding as decoding_stages_module,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe import (
    LingBotVideoDenoisingStage,
    LingBotVideoRefinementStage,
    LingBotVideoRefinerUpscaleStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe import (
    refiner_stages as refiner_stages_module,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.auto_negative import (
    LingBotVideoAutoNegativeStage,
    prune_negative,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.i2v import (
    COND_LATENT_KEY,
    VLM_IMAGE_KEY,
    VLM_PATCH_FACTOR,
    apply_first_frame_prefix,
    pixel_to_vlm_image,
    preprocess_condition_image,
    smart_resize,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.refiner import (
    compute_refiner_sigmas,
    prepare_refiner_latent,
    resize_video_pixels,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.rewriter import (
    LingBotVideoPromptRewriteStage,
    build_expand_prompt,
    build_map_prompt,
    needs_rewrite,
    parse_caption,
    resolve_mode,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.rewriter_backends import (
    HTTPRewriterBackend,
    TransformersRewriterBackend,
    build_rewriter_backend,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.rewriter_prompts import (
    VIDEO_DURATION_EN,
    VIDEO_DURATION_ZH,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.text_encoding import (
    IMG_PROMPT_TEMPLATE,
    PROMPT_TEMPLATE,
    LingBotVideoTextEncodingStage,
)

_LINGBOT_MODULE_SUBDIRS = (
    "scheduler",
    "text_encoder",
    "processor",
    "transformer",
    "vae",
)


def test_moe_path_resolves_moe_configs():
    get_model_info.cache_clear()
    _get_config_info.cache_clear()
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = os.path.join(tmpdir, "lingbot-video-moe-30b-a3b")
        os.makedirs(model_dir)
        with open(
            os.path.join(model_dir, "model_index.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(
                {"_class_name": "LingBotVideoPipeline", "_diffusers_version": "0.37.1"},
                f,
            )
        for subdir in _LINGBOT_MODULE_SUBDIRS:
            os.mkdir(os.path.join(model_dir, subdir))
        info = get_model_info(model_dir, backend="sglang")

    assert info.pipeline_cls.__name__ == "LingBotVideoPipeline"
    assert info.pipeline_config_cls is LingBotVideoMoEPipelineConfig
    assert info.sampling_param_cls is LingBotVideoMoESamplingParams


def test_router_bias_shifts_selection_but_not_gate_weights():
    router = LingBotVideoRouter(
        hidden_size=4,
        num_experts=4,
        top_k=2,
        score_func="sigmoid",
        norm_topk_prob=False,
        n_group=None,
        topk_group=None,
        route_scale=1.0,
    )
    with torch.no_grad():
        router.weight.copy_(
            torch.tensor(
                [
                    [4.0, 0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0, 0.0],
                    [-2.0, 0.0, 0.0, 0.0],
                    [-4.0, 0.0, 0.0, 0.0],
                ]
            )
        )
        router.e_score_correction_bias.copy_(torch.tensor([0.0, 0.0, 0.0, 10.0]))

    top_indices, top_scores = router(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))

    assert set(top_indices[0].tolist()) == {0, 3}
    raw = torch.sigmoid(torch.tensor([4.0, -4.0]))
    picked = {
        int(idx): float(score.detach())
        for idx, score in zip(top_indices[0], top_scores[0])
    }
    assert abs(picked[0] - float(raw[0])) < 1e-5
    assert abs(picked[3] - float(raw[1])) < 1e-5


def _sdpa(q, k, v, attn_mask=None, attn_mask_meta=None):
    q_, k_, v_ = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if attn_mask is not None and attn_mask.dim() == 2:
        attn_mask = attn_mask[:, None, None, :]
    out = torch.nn.functional.scaled_dot_product_attention(
        q_, k_, v_, attn_mask=attn_mask
    )
    return out.transpose(1, 2)


def _real_attention(num_heads, head_dim):
    attn = object.__new__(LingBotVideoAttention)
    attn.local_num_heads = num_heads
    attn.head_dim = head_dim
    attn.to_q = attn.to_k = attn.to_v = attn.to_out = lambda x: (x, None)
    attn.norm_q = attn.norm_k = lambda t: t
    attn.attn = _sdpa
    return attn


def test_attention_isolates_samples_across_batch(monkeypatch):
    monkeypatch.setattr(
        dits_lingbot_video_moe, "_apply_rotary_emb", lambda t, *a, **k: t
    )
    num_heads, head_dim, batch, seq_len = 4, 8, 3, 8
    attn = _real_attention(num_heads, head_dim)
    hidden = num_heads * head_dim
    torch.manual_seed(0)
    x = torch.randn(batch, seq_len, hidden)
    freqs = torch.zeros(batch * seq_len, head_dim // 2)

    valid = [seq_len, seq_len - 2, seq_len - 5]
    mask = torch.zeros(batch, seq_len, dtype=torch.bool)
    for i, length in enumerate(valid):
        mask[i, :length] = True

    batched = attn.forward(x, (freqs, freqs), mask)

    for i, length in enumerate(valid):
        solo = attn.forward(
            x[i : i + 1],
            (freqs[i * seq_len : (i + 1) * seq_len],) * 2,
            mask[i : i + 1],
        )
        torch.testing.assert_close(batched[i : i + 1, :length], solo[:, :length])

    # Flattening the batch into one sequence lets sample 0 attend across the
    # boundary; its output must differ from the isolated per-sample result.
    flat = attn.forward(x.reshape(1, batch * seq_len, hidden), (freqs, freqs), None)
    flat = flat.reshape(batch, seq_len, hidden)
    assert (flat[0, : valid[0]] - batched[0, : valid[0]]).abs().max() > 1e-3


def test_attention_forwards_2d_mask_and_varlen_metadata(monkeypatch):
    monkeypatch.setattr(
        dits_lingbot_video_moe, "_apply_rotary_emb", lambda t, *a, **k: t
    )
    num_heads, head_dim, batch, seq_len = 4, 8, 2, 6
    attn = _real_attention(num_heads, head_dim)
    hidden = num_heads * head_dim
    captured = {}

    def capture_attention(q, k, v, attn_mask=None, attn_mask_meta=None):
        captured["mask"] = attn_mask
        captured["meta"] = attn_mask_meta
        return _sdpa(q, k, v, attn_mask=attn_mask)

    attn.attn = capture_attention
    x = torch.randn(batch, seq_len, hidden)
    freqs = torch.zeros(batch * seq_len, head_dim // 2)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    metadata = {"max_seqlen": seq_len}

    attn.forward(x, (freqs, freqs), mask, metadata)

    assert captured["mask"] is mask
    assert captured["meta"] is metadata


def test_attention_single_sample_matches_direct_attention(monkeypatch):
    monkeypatch.setattr(
        dits_lingbot_video_moe, "_apply_rotary_emb", lambda t, *a, **k: t
    )
    num_heads, head_dim, seq_len = 4, 8, 6
    attn = _real_attention(num_heads, head_dim)
    hidden = num_heads * head_dim
    torch.manual_seed(0)
    x = torch.randn(1, seq_len, hidden)
    freqs = torch.zeros(seq_len, head_dim // 2)

    out = attn.forward(x, (freqs, freqs), attention_mask=None)

    qkv = x.unflatten(2, (num_heads, head_dim))
    expected = _sdpa(qkv, qkv, qkv).flatten(2)
    torch.testing.assert_close(out, expected)


def test_attention_rejects_heads_not_divisible_by_ulysses_degree(monkeypatch):
    monkeypatch.setattr(dits_lingbot_video_moe, "get_tp_world_size", lambda: 1)
    monkeypatch.setattr(
        dits_lingbot_video_moe, "get_ulysses_parallel_world_size", lambda: 3
    )
    with pytest.raises(ValueError, match="ulysses_degree"):
        LingBotVideoAttention(
            hidden_size=32, num_heads=4, norm_eps=1e-6, qkv_bias=True, out_bias=True
        )


def test_block_casts_activations_to_the_residual_not_the_weight_dtype():
    # fp8 weights are storage, not a dtype activations can carry: quantizing the
    # input again leaves the kernel without a scale for it.
    hidden = 8
    seen = {}

    def capture(name):
        def call(tensor, *args, **kwargs):
            seen[name] = tensor.dtype
            return tensor

        return call

    block = object.__new__(LingBotVideoBlock)
    torch.nn.Module.__init__(block)
    block.scale_shift_table = torch.nn.Parameter(torch.zeros(1, 6 * hidden))
    block.norm1 = block.norm2 = lambda t: t
    block.norm_post_attn = block.norm_post_ffn = lambda t: t
    block.attn = capture("attn")
    block.attn.to_q = SimpleNamespace(
        weight=torch.zeros(hidden, hidden, dtype=torch.float8_e4m3fn)
    )
    block.ffn = capture("ffn")

    x = torch.randn(1, 4, hidden, dtype=torch.bfloat16)
    block(x, torch.zeros(1, 6 * hidden), (torch.zeros(4, 4), torch.zeros(4, 4)))

    assert seen["attn"] == torch.bfloat16
    assert seen["ffn"] == torch.bfloat16


# float32 takes the early return the loop below used to skip; bf16 takes the loop.
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_to_leaves_quantized_weights_at_their_own_dtype(dtype):
    model = object.__new__(LingBotVideoTransformer3DModel)
    torch.nn.Module.__init__(model)
    model.proj_out = torch.nn.Linear(4, 4, bias=False)
    model.proj_out.weight = torch.nn.Parameter(
        torch.zeros(4, 4, dtype=torch.float8_e4m3fn), requires_grad=False
    )
    model.patch_embedder = torch.nn.Linear(4, 4, bias=False)
    model.norm1 = torch.nn.Linear(4, 4, bias=False)

    model.to(dtype)

    assert model.proj_out.weight.dtype == torch.float8_e4m3fn
    assert model.patch_embedder.weight.dtype == dtype
    assert model.norm1.weight.dtype == torch.float32


def test_shard_sequence_is_off_without_sequence_parallelism():
    # sp_size must short-circuit first: get_forward_context() asserts when unset.
    model = object.__new__(LingBotVideoTransformer3DModel)
    model.sp_size = 1
    assert model._shard_sequence() is False


@pytest.mark.parametrize("sp_size", [2, 3, 4])
def test_joint_shard_reassembles_the_global_sequence(monkeypatch, sp_size):
    gt, gh, gw, text_len = 3, 2, 2, 7
    joint_seq_len = gt * gh * gw + text_len
    dev = torch.device("cpu")
    positions = _joint_position_ids(torch.tensor([text_len]), gt, gh, gw, text_len, dev)
    hidden = torch.arange(joint_seq_len, dtype=torch.float32).view(1, joint_seq_len, 1)

    monkeypatch.setattr(sps, "get_sp_world_size", lambda: sp_size)
    chunks, pos_chunks = [], []
    for rank in range(sp_size):
        monkeypatch.setattr(sps, "get_sp_parallel_rank", lambda rank=rank: rank)
        shard = sps.build_shard_plan(joint_seq_len)
        chunks.append(sps.shard_like(hidden, shard))
        pos_chunks.append(
            sps.shard_like(
                positions.reshape(1, joint_seq_len, 3), shard, pad_mode="repeat_last"
            ).reshape(-1, 3)
        )

    gathered = torch.cat(chunks, dim=1)[:, :joint_seq_len]
    torch.testing.assert_close(gathered, hidden)
    torch.testing.assert_close(torch.cat(pos_chunks)[:joint_seq_len], positions)


class _FakeBatchEncoding(dict):
    def to(self, _device):
        return self


class _FakeQwenProcessor:
    def __init__(self, prompt_width, prefix_width, true_len):
        self.prompt_width = prompt_width
        self.prefix_width = prefix_width
        self.true_len = true_len

    def __call__(self, **kwargs):
        if "max_length" in kwargs:
            width = self.prompt_width
            mask = torch.zeros(1, width, dtype=torch.long)
            mask[0, : self.true_len] = 1
        else:
            width = self.prefix_width
            mask = torch.ones(1, width, dtype=torch.long)
        return _FakeBatchEncoding(
            input_ids=torch.zeros(1, width, dtype=torch.long),
            attention_mask=mask,
        )


class _RecordingProcessor:
    def __init__(self, width=8):
        self.width = width
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "input_ids": torch.zeros(1, self.width, dtype=torch.long),
            "attention_mask": torch.ones(1, self.width, dtype=torch.long),
        }


def _text_encoding_stage(processor, encoder=None):
    stage = object.__new__(LingBotVideoTextEncodingStage)
    stage._component_residency_manager = None
    stage.text_encoders = [encoder]
    stage.tokenizers = [processor]
    stage.token_length = 128
    stage.hidden_state_skip_layer = 0
    stage.prompt_template = PROMPT_TEMPLATE
    stage._crop_start = None
    return stage


def test_text_encoding_crops_template_then_trims_padding():
    prompt_width, prefix_width, true_len, channels = 10, 3, 8, 4
    hidden = torch.arange(prompt_width, dtype=torch.float32)
    hidden = hidden.view(1, prompt_width, 1).expand(1, prompt_width, channels)

    def encoder(**kwargs):
        return SimpleNamespace(hidden_states=[hidden])

    stage = _text_encoding_stage(
        _FakeQwenProcessor(prompt_width, prefix_width, true_len), encoder
    )
    embeds, mask = stage._encode_prompt(
        "a structured caption", torch.device("cpu"), torch.float32
    )

    assert tuple(embeds.shape) == (1, true_len - prefix_width, channels)
    torch.testing.assert_close(embeds, hidden[:, prefix_width:true_len])
    assert int(mask.sum()) == true_len - prefix_width
    assert stage._compute_crop_start() == prefix_width


def test_text_encoding_sets_forward_context_for_native_encoder():
    prompt_width, prefix_width, true_len, channels = 6, 2, 5, 4
    hidden = torch.zeros(1, prompt_width, channels)

    class NativeEncoder:
        uses_sglang_forward_context = True

        def __call__(self, **kwargs):
            context = get_forward_context()
            assert context.current_timestep == 0
            assert context.attn_metadata is None
            assert kwargs["output_hidden_states"] is True
            return SimpleNamespace(hidden_states=[hidden])

    stage = _text_encoding_stage(
        _FakeQwenProcessor(prompt_width, prefix_width, true_len), NativeEncoder()
    )
    embeds, mask = stage._encode_prompt(
        "a structured caption", torch.device("cpu"), torch.float32
    )

    assert tuple(embeds.shape) == (1, true_len - prefix_width, channels)
    assert tuple(mask.shape) == (1, true_len - prefix_width)


def test_condition_image_prepends_vision_block_before_caption():
    processor = _RecordingProcessor()
    stage = _text_encoding_stage(processor)
    image = PIL.Image.new("RGB", (64, 64))

    stage._build_prompt_inputs("a structured caption", [image])

    call = processor.calls[0]
    assert call["images"] == [image]
    text = call["text"][0]
    assert IMG_PROMPT_TEMPLATE + "a structured caption" in text
    assert text.index(IMG_PROMPT_TEMPLATE) < text.index("a structured caption")
    assert text.index("<|im_start|>user") < text.index(IMG_PROMPT_TEMPLATE)


def test_text_only_prompt_keeps_no_vision_block():
    processor = _RecordingProcessor()
    stage = _text_encoding_stage(processor)

    stage._build_prompt_inputs("a structured caption")

    call = processor.calls[0]
    assert call["images"] is None
    assert IMG_PROMPT_TEMPLATE not in call["text"][0]


def test_crop_start_is_measured_on_the_text_only_prefix():
    processor = _RecordingProcessor()
    stage = _text_encoding_stage(processor)

    assert stage._compute_crop_start() == processor.width
    assert processor.calls[0]["images"] is None


def test_vlm_images_read_the_condition_frame_from_batch_extra():
    image = PIL.Image.new("RGB", (32, 32))
    read = LingBotVideoTextEncodingStage._vlm_images
    assert read(SimpleNamespace(extra={})) is None
    assert read(SimpleNamespace(extra={VLM_IMAGE_KEY: image})) == [image]


def test_check_inputs_enforces_frame_and_size_contract():
    check = LingBotVideoTextEncodingStage.check_inputs
    check(480, 832, 1)
    check(480, 832, 81)
    with pytest.raises(ValueError, match=r"4n\+1"):
        check(480, 832, 83)
    with pytest.raises(ValueError, match="multiples of 16"):
        check(480, 840, 81)


def test_decode_scale_and_shift_invert_vae_normalization():
    config = LingBotVideoMoEPipelineConfig()
    scale, shift = config.get_decode_scale_and_shift(
        torch.device("cpu"), torch.float32, vae=None
    )
    arch = config.vae_config.arch_config
    std = torch.tensor(arch.latents_std, dtype=torch.float32).view(1, -1, 1, 1, 1)
    mean = torch.tensor(arch.latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1)
    torch.testing.assert_close(scale, 1.0 / std)
    torch.testing.assert_close(shift, mean)


def test_latents_stay_fp32_under_bf16_precision():
    config = LingBotVideoMoEPipelineConfig()
    assert config.get_latent_dtype(torch.bfloat16) == torch.float32


def test_grouped_experts_store_packed_w13_weight():
    experts = LingBotVideoGroupedExperts(
        num_experts=2, hidden_size=4, intermediate_size=3
    )
    names = {n for n, _ in experts.named_parameters()}
    assert "w13_weight" in names and "w2" in names
    assert "w1" not in names and "w3" not in names
    assert tuple(experts.w13_weight.shape) == (2, 6, 4)  # [E, 2I, H]


def test_preprocess_packs_w1_w3_into_w13_weight():
    pack = LingBotVideoTransformer3DModel.preprocess_loaded_state_dict
    E, I, H = 2, 3, 4
    w1 = torch.arange(E * I * H, dtype=torch.float32).reshape(E, I, H)
    w2 = torch.arange(E * H * I, dtype=torch.float32).reshape(E, H, I)
    w3 = torch.arange(E * I * H, dtype=torch.float32).reshape(E, I, H) + 100.0
    # block 0: w1 before w3; block 1: w3 before w1 (order-independence).
    src = [
        ("blocks.0.ffn.experts.w1", w1),
        ("blocks.0.ffn.experts.w2", w2),
        ("blocks.0.ffn.experts.w3", w3),
        ("blocks.0.ffn.router.weight", torch.zeros(E, H)),
        ("blocks.1.ffn.experts.w3", w3.clone()),
        ("blocks.1.ffn.experts.w2", w2.clone()),
        ("blocks.1.ffn.experts.w1", w1.clone()),
    ]
    out = dict(pack(None, iter(src)))
    assert set(out.keys()) == {
        "blocks.0.ffn.experts.w13_weight",
        "blocks.0.ffn.experts.w2",
        "blocks.0.ffn.router.weight",
        "blocks.1.ffn.experts.w13_weight",
        "blocks.1.ffn.experts.w2",
    }
    packed = torch.cat((w1, w3), dim=1)  # gate then up, dim-1
    torch.testing.assert_close(out["blocks.0.ffn.experts.w13_weight"], packed)
    torch.testing.assert_close(out["blocks.1.ffn.experts.w13_weight"], packed)
    torch.testing.assert_close(out["blocks.0.ffn.experts.w2"], w2)


def test_joint_position_ids_match_reference_and_cover_padding():
    dev = torch.device("cpu")
    gt, gh, gw = 2, 3, 4
    n_video = gt * gh * gw

    # B==1, no padding: byte-identical to the per-sample reference.
    vec = _joint_position_ids(torch.tensor([5]), gt, gh, gw, 5, dev)
    torch.testing.assert_close(vec, make_joint_position_ids(5, gt, gh, gw, dev))

    # B==1 with padding: real tokens match the text_len=4 reference; the extra
    # padding row is (0,0,0). vec has n_video+L rows (matches q for B*S).
    vec_p = _joint_position_ids(torch.tensor([4]), gt, gh, gw, 5, dev)
    torch.testing.assert_close(
        vec_p[: n_video + 4], make_joint_position_ids(4, gt, gh, gw, dev)
    )
    torch.testing.assert_close(
        vec_p[n_video + 4 :], torch.zeros((1, 3), dtype=torch.int32)
    )

    # B>1 with padding: covers B*S rows; each sample's real tokens match its ref.
    text_lens = [5, 3, 6]
    B, L = len(text_lens), 6
    vec_b = _joint_position_ids(torch.tensor(text_lens), gt, gh, gw, L, dev)
    assert vec_b.shape[0] == B * (n_video + L)
    for i, t in enumerate(text_lens):
        start = i * (n_video + L)
        real = n_video + t
        torch.testing.assert_close(
            vec_b[start : start + real], make_joint_position_ids(t, gt, gh, gw, dev)
        )


def test_image_requests_are_split_into_one_request_per_output():
    params = LingBotVideoMoESamplingParams(
        request_id="rid",
        prompt='{"comprehensive_description": "a fox"}',
        image_path="fox.png",
        output_path="/tmp",
        output_file_name="video.mp4",
        num_outputs_per_prompt=2,
        seed=[7, 8],
    )

    outputs = params.expand_video_request_outputs_for_queue(Req(sampling_params=params))

    assert [req.num_outputs_per_prompt for req in outputs] == [1, 1]
    assert [req.batch_size for req in outputs] == [1, 1]
    assert [req.seed for req in outputs] == [7, 8]
    assert all(req.image_path == "fox.png" for req in outputs)


def test_single_output_requests_are_left_unsplit():
    params = LingBotVideoMoESamplingParams(
        request_id="rid",
        prompt='{"comprehensive_description": "a fox"}',
        output_path="/tmp",
        output_file_name="video.mp4",
    )
    batch = Req(sampling_params=params)

    assert params.expand_video_request_outputs_for_queue(batch) == [batch]


class _StubVae:
    # The fast-path gate registry is keyed by weak reference, so this cannot be a
    # SimpleNamespace.
    def __init__(self):
        self.use_tiling = False


class _StubScheduler:
    sigma_max = 1.0
    sigma_min = 0.0

    def __init__(self):
        self.timesteps = torch.zeros(1)

    def set_timesteps(self, device, sigmas, shift):
        self.timesteps = torch.as_tensor(sigmas)


def test_condition_image_is_cover_resized_then_center_cropped():
    height, width = 32, 64
    # A portrait source is scaled 2x on width to 64x256, so the centered crop
    # keeps source rows 56-71, all inside the green band.
    array = torch.zeros(128, 32, 3, dtype=torch.uint8)
    array[:43, :, 0] = 255
    array[43:86, :, 1] = 255
    array[86:, :, 2] = 255
    source = PIL.Image.fromarray(array.numpy(), mode="RGB")

    pixel = preprocess_condition_image(source, height, width)

    assert pixel.dtype == torch.float32
    green = torch.tensor([0.0, 1.0, 0.0]).view(1, 3, 1, 1, 1)
    torch.testing.assert_close(pixel, green.expand(1, 3, 1, height, width))


def test_condition_image_preserves_pixel_values_when_already_sized():
    array = torch.randint(0, 256, (16, 24, 3), dtype=torch.uint8)
    image = PIL.Image.fromarray(array.numpy(), mode="RGB")

    pixel = preprocess_condition_image(image, 16, 24)

    expected = array.permute(2, 0, 1).float().div(255.0)[None, :, None]
    torch.testing.assert_close(pixel, expected)


def test_smart_resize_aligns_to_the_patch_factor():
    height, width = smart_resize(468, 832, factor=VLM_PATCH_FACTOR)
    assert (height, width) == (480, 832)

    height, width = smart_resize(30, 30, factor=VLM_PATCH_FACTOR)
    assert height % VLM_PATCH_FACTOR == 0 and width % VLM_PATCH_FACTOR == 0
    assert height * width >= 4 * VLM_PATCH_FACTOR**2


def test_vlm_image_is_patch_aligned():
    pixel = preprocess_condition_image(PIL.Image.new("RGB", (100, 60)), 48, 80)

    image = pixel_to_vlm_image(pixel)

    assert image.mode == "RGB"
    assert image.width % VLM_PATCH_FACTOR == 0
    assert image.height % VLM_PATCH_FACTOR == 0


def test_first_frame_prefix_overwrites_only_the_conditioned_frames():
    latents = torch.zeros(1, 16, 5, 2, 3)
    cond = torch.arange(16 * 2 * 3, dtype=torch.float32).reshape(1, 16, 1, 2, 3)

    out = apply_first_frame_prefix(latents, cond)

    torch.testing.assert_close(out[:, :, :1], cond)
    assert float(out[:, :, 1:].abs().max()) == 0.0


def test_first_frame_prefix_follows_the_latent_dtype():
    latents = torch.zeros(1, 16, 2, 1, 1, dtype=torch.float32)
    cond = torch.ones(1, 16, 1, 1, 1, dtype=torch.bfloat16)

    out = apply_first_frame_prefix(latents, cond)

    assert out.dtype == torch.float32
    assert float(out[:, :, 0].min()) == 1.0


def test_denoising_reapplies_the_condition_latent_after_each_step():
    stage = object.__new__(LingBotVideoDenoisingStage)
    cond = torch.full((1, 16, 1, 2, 2), 7.0)
    batch = SimpleNamespace(extra={COND_LATENT_KEY: cond})
    latents = torch.zeros(1, 16, 3, 2, 2)

    out = stage.post_forward_for_ti2v_task(batch, None, None, latents, None)

    torch.testing.assert_close(out[:, :, :1], cond)
    assert float(out[:, :, 1:].abs().max()) == 0.0


def test_denoising_leaves_latents_untouched_without_a_condition_latent():
    stage = object.__new__(LingBotVideoDenoisingStage)
    latents = torch.randn(1, 16, 3, 2, 2)

    out = stage.post_forward_for_ti2v_task(
        SimpleNamespace(extra={}), None, None, latents, None
    )

    torch.testing.assert_close(out, latents)


def test_config_takes_an_optional_image_and_loads_the_vae_encoder():
    config = LingBotVideoMoEPipelineConfig()

    assert config.task_type == ModelTaskType.TI2V
    assert config.task_type.accepts_image_input()
    assert not config.task_type.requires_image_input()
    assert config.skip_input_image_preprocess
    assert config.vae_config.load_encoder and config.vae_config.load_decoder
    # The base class excludes TI2V; the text-only path must keep batching.
    assert config.supports_dynamic_batching()


def test_single_frame_request_outputs_an_image():
    params = LingBotVideoMoESamplingParams(prompt="a caption", num_frames=1)

    params._set_output_file_name()

    assert params.data_type == DataType.IMAGE


def _adjusted(num_frames, explicit_fields, pipeline_class_name=None, **kwargs):
    params = LingBotVideoMoESamplingParams(
        prompt="a caption", num_frames=num_frames, **kwargs
    )
    params._explicit_fields = set(explicit_fields)
    params._adjust(
        SimpleNamespace(
            pipeline_config=LingBotVideoMoEPipelineConfig(),
            pipeline_class_name=pipeline_class_name,
            num_gpus=4,
            output_path=None,
            comfyui_mode=False,
        )
    )
    return params


def test_single_frame_request_swaps_in_the_image_negative_prompt():
    params = _adjusted(1, [])

    assert params.negative_prompt == DEFAULT_NEGATIVE_PROMPT_IMAGE
    assert "temporal_and_motion_stability" not in params.negative_prompt


def test_explicit_negative_prompt_survives_the_single_frame_default():
    params = _adjusted(1, ["negative_prompt"], negative_prompt="keep me")

    assert params.negative_prompt == "keep me"


def test_single_frame_survives_multi_gpu_frame_alignment():
    params = _adjusted(1, [])

    assert params.num_frames == 1


def test_refiner_rejects_single_frame_requests():
    with pytest.raises(ValueError, match="single-frame"):
        _adjusted(1, [], pipeline_class_name=REFINER_PIPELINE_NAME)


def test_refiner_defaults_the_base_pass_to_its_own_aspect():
    params = _adjusted(81, [], pipeline_class_name=REFINER_PIPELINE_NAME)

    assert params.data_type == DataType.VIDEO
    assert (params.height, params.width) == (REFINER_BASE_HEIGHT, REFINER_BASE_WIDTH)


def test_refiner_keeps_an_explicit_base_geometry_of_its_aspect():
    params = _adjusted(
        81,
        ["height", "width"],
        pipeline_class_name=REFINER_PIPELINE_NAME,
        height=544,
        width=960,
    )

    assert (params.height, params.width) == (544, 960)


def test_refiner_rejects_a_base_geometry_it_would_stretch():
    with pytest.raises(ValueError, match="aspect ratio"):
        _adjusted(
            81,
            ["height", "width"],
            pipeline_class_name=REFINER_PIPELINE_NAME,
            height=480,
            width=480,
        )


@pytest.mark.parametrize(
    "pipeline_class_name, warmup_resolutions, num_frames, expected",
    [
        (REFINER_PIPELINE_NAME, None, 81, (REFINER_BASE_HEIGHT, REFINER_BASE_WIDTH)),
        (REFINER_PIPELINE_NAME, ["480x480"], 81, (480, 480)),
        (None, None, 81, (480, 480)),
    ],
)
def test_refiner_warmup_runs_at_the_base_geometry_real_requests_get(
    pipeline_class_name, warmup_resolutions, num_frames, expected
):
    params = LingBotVideoMoESamplingParams()
    req = SimpleNamespace(height=480, width=480, num_frames=num_frames)

    params.prepare_synthetic_warmup_request_for_queue(
        req,
        SimpleNamespace(
            pipeline_class_name=pipeline_class_name,
            warmup_resolutions=warmup_resolutions,
        ),
    )

    assert (req.height, req.width) == expected


def test_base_video_request_keeps_its_geometry_and_video_negative_prompt():
    params = _adjusted(81, [])

    assert (params.height, params.width) == (480, 480)
    assert params.negative_prompt == DEFAULT_NEGATIVE_PROMPT
    assert params.data_type == DataType.VIDEO


def test_sequence_shard_is_on_and_frames_stay_as_requested():
    # The DiT shards the joint sequence itself, so latent frames must not be
    # rounded up to num_gpus: 9 frames would otherwise become 13.
    params = _adjusted(9, [])

    assert params.enable_sequence_shard
    assert not params.adjust_frames
    assert params.num_frames == 9


def test_sequence_shard_can_be_turned_off_by_the_request():
    params = _adjusted(81, ["enable_sequence_shard"], enable_sequence_shard=False)

    assert not params.enable_sequence_shard


def test_refiner_sigmas_start_at_the_threshold_and_descend():
    sigmas = compute_refiner_sigmas(
        sigma_max=1.0,
        sigma_min=0.0,
        num_inference_steps=8,
        shift=3.0,
        t_thresh=0.85,
        tail_steps=2,
    )

    assert abs(float(sigmas[0]) - 0.85) < 1e-6
    assert all(b < a for a, b in zip(sigmas, sigmas[1:]))
    assert float(sigmas.min()) >= 0.0 and float(sigmas.max()) <= 1.0


def test_refiner_sigma_tail_steps_extend_the_schedule():
    kwargs = dict(
        sigma_max=1.0,
        sigma_min=0.0,
        num_inference_steps=8,
        shift=3.0,
        t_thresh=0.85,
    )

    plain = compute_refiner_sigmas(tail_steps=0, **kwargs)
    tailed = compute_refiner_sigmas(tail_steps=2, **kwargs)

    assert len(tailed) == len(plain) + 2


@pytest.mark.parametrize("t_thresh", [0.0, 1.5])
def test_refiner_rejects_a_threshold_outside_the_unit_range(t_thresh):
    with pytest.raises(ValueError, match="t_thresh"):
        compute_refiner_sigmas(
            sigma_max=1.0,
            sigma_min=0.0,
            num_inference_steps=8,
            shift=3.0,
            t_thresh=t_thresh,
        )


def test_refiner_latent_interpolates_toward_noise_at_the_threshold():
    upscaled = torch.zeros(1, 16, 2, 2, 2)
    noise = torch.ones(1, 16, 2, 2, 2)

    blended = prepare_refiner_latent(upscaled, noise, 0.85)

    torch.testing.assert_close(blended, torch.full_like(upscaled, 0.85))


def test_pixel_resize_keeps_the_clip_layout():
    pixels = torch.rand(1, 3, 5, 24, 32)

    resized = resize_video_pixels(pixels, 48, 64)

    assert tuple(resized.shape) == (1, 3, 5, 48, 64)
    assert 0.0 <= float(resized.min()) and float(resized.max()) <= 1.0


def test_refiner_weights_resolve_to_the_refiner_subfolder():
    pipeline = object.__new__(LingBotVideoRefinerPipeline)
    pipeline.model_path = "/models/lingbot"
    server_args = SimpleNamespace(component_paths={})

    path = pipeline._resolve_component_path(
        server_args, "transformer_2", "transformer_2"
    )

    assert path == "/models/lingbot/refiner"


def test_transformer_2_path_override_wins_over_the_refiner_subfolder(tmp_path):
    override = tmp_path / "custom_refiner"
    override.mkdir()
    pipeline = object.__new__(LingBotVideoRefinerPipeline)
    pipeline.model_path = "/models/lingbot"
    server_args = SimpleNamespace(component_paths={"transformer_2": str(override)})

    path = pipeline._resolve_component_path(
        server_args, "transformer_2", "transformer_2"
    )

    assert path == str(override)


def test_refiner_reseeds_every_output_of_a_batched_request(monkeypatch):
    monkeypatch.setattr(
        refiner_stages_module, "get_local_torch_device", lambda: torch.device("cpu")
    )
    batch = SimpleNamespace(seeds=[11, 22])

    generators = refiner_stages_module._refiner_generators(
        batch, device=torch.device("cpu")
    )
    batched = randn_tensor(
        (2, 1, 2, 2), generator=generators, device=torch.device("cpu")
    )

    for index, seed in enumerate(batch.seeds):
        alone = randn_tensor(
            (1, 1, 2, 2),
            generator=[torch.Generator().manual_seed(seed)],
            device=torch.device("cpu"),
        )
        torch.testing.assert_close(batched[index : index + 1], alone)


def _refiner_prepared(monkeypatch, *, base_scale, refiner_scale):
    monkeypatch.setattr(
        refiner_stages_module, "get_local_torch_device", lambda: torch.device("cpu")
    )
    monkeypatch.setattr(
        LingBotVideoDenoisingStage,
        "_prepare_denoising_loop",
        lambda self, batch, server_args: batch,
    )
    stage = object.__new__(LingBotVideoRefinementStage)
    config = LingBotVideoMoEPipelineConfig()
    config.refiner_guidance_scale = refiner_scale
    batch = SimpleNamespace(
        scheduler=_StubScheduler(),
        timesteps=None,
        num_inference_steps=None,
        guidance_scale=base_scale,
        do_classifier_free_guidance=base_scale > 1.0,
        prompt_embeds=[torch.ones(1, 4, 8)],
        prompt_attention_mask=torch.ones(1, 4, dtype=torch.long),
        negative_prompt_embeds=None,
        negative_attention_mask=None,
        extra={},
    )
    stage._prepare_denoising_loop(batch, SimpleNamespace(pipeline_config=config))
    return batch


def test_refiner_guidance_turns_cfg_on_when_the_base_pass_ran_without_it(monkeypatch):
    batch = _refiner_prepared(monkeypatch, base_scale=1.0, refiner_scale=3.0)

    assert batch.guidance_scale == 3.0
    assert batch.do_classifier_free_guidance
    torch.testing.assert_close(
        batch.negative_prompt_embeds[0], torch.zeros_like(batch.prompt_embeds[0])
    )


def test_refiner_guidance_turns_cfg_off_when_the_refiner_scale_is_one(monkeypatch):
    batch = _refiner_prepared(monkeypatch, base_scale=6.0, refiner_scale=1.0)

    assert not batch.do_classifier_free_guidance
    assert batch.negative_prompt_embeds is None


def test_refiner_upscale_runs_on_every_rank_under_cfg_parallel(monkeypatch):
    # Every rank denoises the refiner latents, so a rank that skips the
    # re-encode carries the base latents into the second pass.
    monkeypatch.setattr(
        decoding_stages_module,
        "get_global_server_args",
        lambda: SimpleNamespace(enable_cfg_parallel=True),
    )

    assert (
        object.__new__(DecodingStage).parallelism_type
        == StageParallelismType.MAIN_RANK_ONLY
    )
    assert (
        object.__new__(LingBotVideoRefinerUpscaleStage).parallelism_type
        == StageParallelismType.REPLICATED
    )


def _upscale_stage(monkeypatch, vae):
    monkeypatch.setattr(
        refiner_stages_module, "get_local_torch_device", lambda: torch.device("cpu")
    )
    stage = object.__new__(LingBotVideoRefinerUpscaleStage)
    stage.component_name = "vae"
    stage.vae = vae
    stage.use_declared_component = lambda **kwargs: nullcontext(vae)
    return stage


def _upscale_inputs(quality):
    config = LingBotVideoMoEPipelineConfig()
    config.refiner_height, config.refiner_width = 8, 16
    server_args = SimpleNamespace(
        pipeline_config=config, disable_autocast=False, component_precisions={}
    )
    batch = SimpleNamespace(
        latents=torch.zeros(1, 4, 1, 2, 2),
        extra={},
        seeds=[],
        sampling_params=SimpleNamespace(quality=quality),
        height=480,
        width=832,
    )
    return server_args, batch


def test_refiner_round_trip_matches_the_declared_decode_precision(monkeypatch):
    stage = _upscale_stage(monkeypatch, _StubVae())
    seen = {}

    def decode(latents, server_args, *, vae_dtype):
        seen["decode"] = vae_dtype
        return torch.zeros(1, 3, 1, 4, 4)

    def encode(pixels, server_args, *, vae_dtype, generators):
        seen["encode"] = vae_dtype
        return torch.zeros(1, 4, 1, 2, 2)

    stage.decode = decode
    stage._encode = encode
    server_args, batch = _upscale_inputs("lossless")
    # The decode-only override must win over vae_precision on both halves.
    server_args.pipeline_config.vae_decode_precision = "fp32"

    stage.forward(batch, server_args)

    declared = stage.component_uses(server_args)[0].target_dtype
    assert declared == torch.float32
    assert seen["decode"] == declared
    assert seen["encode"] == declared
    assert (batch.height, batch.width) == (8, 16)


@pytest.mark.parametrize(
    "quality, fast_path", [("exact", False), ("lossless", True), ("high", True)]
)
def test_refiner_decode_follows_the_request_quality(monkeypatch, quality, fast_path):
    # The base pass opens the VAE fast path from lossless up; the refiner decode
    # must use the same tier and resolve its precision for the same quality.
    vae = _StubVae()
    gate = VaeFastPathGate()
    register_vae_fast_path_gate(vae, gate)
    stage = _upscale_stage(monkeypatch, vae)
    seen = {}

    def resolve(server_args, component_name, *, quality):
        seen["quality"] = quality
        return torch.float32

    def decode(latents, server_args, *, vae_dtype):
        seen["fast_path"] = gate.enabled
        return torch.zeros(1, 3, 1, 4, 4)

    monkeypatch.setattr(refiner_stages_module, "resolve_decode_precision", resolve)
    stage.decode = decode
    stage._encode = lambda pixels, server_args, *, vae_dtype, generators: torch.zeros(
        1, 4, 1, 2, 2
    )
    server_args, batch = _upscale_inputs(quality)

    stage.forward(batch, server_args)

    assert seen == {"quality": quality, "fast_path": fast_path}


def test_refiner_condition_frame_keeps_the_base_framing():
    # A 3:1 image covers a 2:1 base and a 3:2 refiner frame with different crops;
    # frame 0 must show what the upscaled base clip shows.
    array = torch.zeros(32, 96, 3, dtype=torch.uint8)
    array[..., 0] = (torch.arange(96) // 8 * 20).to(torch.uint8)
    image = PIL.Image.fromarray(array.numpy(), mode="RGB")
    stage = object.__new__(LingBotVideoRefinerUpscaleStage)
    stage._encode = lambda pixels, server_args, *, vae_dtype, generators: pixels
    batch = SimpleNamespace(condition_image=[image], height=16, width=32)

    frame = stage._encode_condition_frame(
        batch,
        None,
        height=32,
        width=48,
        vae_dtype=torch.float32,
        generators=None,
    )

    base = preprocess_condition_image(image, height=16, width=32)
    torch.testing.assert_close(frame, resize_video_pixels(base, height=32, width=48))
    # Cropping at the refiner size shifts the edges by one 20/255 band.
    cropped_at_refiner = preprocess_condition_image(image, height=32, width=48)
    assert float((frame - cropped_at_refiner).abs().max()) > 0.05


def test_rewrite_mode_follows_the_request_shape():
    assert resolve_mode(SimpleNamespace(num_frames=1, image_path=None)) == "t2i"
    assert resolve_mode(SimpleNamespace(num_frames=121, image_path=None)) == "t2v"
    assert resolve_mode(SimpleNamespace(num_frames=121, image_path="f.png")) == "ti2v"


def test_structured_captions_skip_rewriting():
    assert needs_rewrite("a red fox in the snow")
    assert not needs_rewrite('{"comprehensive_description": "..."}')
    assert not needs_rewrite('  {"caption": {}}')


def test_expand_prompt_carries_the_duration_for_video_modes():
    video = build_expand_prompt("t2v", "a red fox", 5)
    assert "Video Duration: 5 seconds" in video
    assert video.endswith("Video Duration: 5 seconds")

    assert "Duration" not in build_expand_prompt("t2i", "a glass of tea", 5)


def test_expand_prompt_matches_the_caption_language():
    cjk_prompt = VIDEO_DURATION_ZH.format(duration=3)
    expanded = build_expand_prompt("t2v", cjk_prompt, 5)

    assert VIDEO_DURATION_ZH.format(duration=5) in expanded
    assert VIDEO_DURATION_EN.format(duration=5) not in expanded


def test_map_prompt_feeds_the_expansion_back():
    mapped = build_map_prompt("t2v", "A red fox trots through snow.", 5)

    assert "DETAILED CAPTION:\nA red fox trots through snow." in mapped
    assert mapped.endswith("Output the JSON now.")


def test_caption_parsing_tolerates_fences_and_prose():
    assert parse_caption('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_caption('Here you go: {"a": 1} hope that helps') == {"a": 1}
    assert parse_caption("no json here") is None
    assert parse_caption('["not", "an", "object"]') is None


class _FakeBackend:
    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def generate(self, text, image, use_lora):
        self.calls.append((use_lora, image))
        return self.replies.pop(0)


def _rewrite_stage(replies):
    stage = object.__new__(LingBotVideoPromptRewriteStage)
    stage.backend = _FakeBackend(replies)
    return stage


def test_rewriting_replaces_the_prompt_with_a_compact_caption():
    stage = _rewrite_stage(["A red fox trots through snow.", '{"b": 2, "a": 1}'])
    batch = SimpleNamespace(prompt="a red fox", num_frames=121, fps=24, image_path=None)

    stage.forward(batch, None)

    assert batch.prompt == '{"b":2,"a":1}'
    # Only the mapping turn enables the adapter.
    assert [use_lora for use_lora, _ in stage.backend.calls] == [False, True]
    assert all(image is None for _, image in stage.backend.calls)


def test_rewriting_refuses_to_fall_back_to_the_free_text_prompt():
    stage = _rewrite_stage(["expanded", "sorry, I cannot"])
    batch = SimpleNamespace(prompt="a red fox", num_frames=121, fps=24, image_path=None)

    with pytest.raises(ValueError, match="structured caption"):
        stage.forward(batch, None)


def test_rewriting_leaves_a_structured_caption_alone():
    stage = _rewrite_stage([])
    caption = '{"comprehensive_description": "..."}'
    batch = SimpleNamespace(prompt=caption, num_frames=121, fps=24, image_path=None)

    stage.forward(batch, None)

    assert batch.prompt == caption
    assert stage.backend.calls == []


def test_rewriter_backend_is_off_until_a_url_or_local_model_is_set():
    config = LingBotVideoMoEPipelineConfig()
    assert build_rewriter_backend(config) is None

    config.rewriter_url = "http://host:30000"
    assert isinstance(build_rewriter_backend(config), HTTPRewriterBackend)

    local = LingBotVideoMoEPipelineConfig()
    local.rewriter_model_path = "/models/base"
    with pytest.raises(ValueError, match="rewriter_adapter_path"):
        build_rewriter_backend(local)

    local.rewriter_adapter_path = "/models/adapter"
    assert isinstance(build_rewriter_backend(local), TransformersRewriterBackend)


def _added_stages(**config_overrides) -> list:
    config = LingBotVideoMoEPipelineConfig()
    for name, value in config_overrides.items():
        setattr(config, name, value)
    added = []
    pipeline = object.__new__(LingBotVideoPipeline)
    pipeline.add_stage = lambda stage, *a, **k: added.append(stage)
    pipeline.get_module = lambda name, default=None: SimpleNamespace(
        modules=lambda: iter(())
    )
    pipeline.add_standard_latent_preparation_stage = lambda *a, **k: None
    pipeline.add_standard_timestep_preparation_stage = lambda *a, **k: None
    pipeline.add_standard_decoding_stage = lambda *a, **k: None

    pipeline.create_pipeline_stages(SimpleNamespace(pipeline_config=config))
    return added


def _added_stage_names(**config_overrides) -> list[str]:
    return [type(stage).__name__ for stage in _added_stages(**config_overrides)]


def test_pipeline_skips_the_rewriter_stages_by_default(monkeypatch):
    # add_stage_if evaluates its argument eagerly, so the guard has to come first.
    def refuse(*args, **kwargs):
        raise AssertionError("the rewriter stages must not be constructed")

    monkeypatch.setattr(
        pipeline_lingbot_video_moe, "LingBotVideoPromptRewriteStage", refuse
    )
    monkeypatch.setattr(
        pipeline_lingbot_video_moe, "LingBotVideoAutoNegativeStage", refuse
    )
    assert LingBotVideoMoEPipelineConfig().rewriter_url is None

    added = _added_stage_names()

    assert "InputValidationStage" in added


def test_auto_negative_follows_the_rewriter_only_when_asked():
    served = dict(rewriter_url="http://host:30000")

    assert "LingBotVideoPromptRewriteStage" in _added_stage_names(**served)
    assert "LingBotVideoAutoNegativeStage" not in _added_stage_names(**served)

    both = _added_stage_names(**served, rewriter_auto_negative=True)
    assert both.index("LingBotVideoAutoNegativeStage") == (
        both.index("LingBotVideoPromptRewriteStage") + 1
    )


def test_rewriter_stages_share_one_backend():
    stages = _added_stages(
        rewriter_model_path="/models/base",
        rewriter_adapter_path="/models/adapter",
        rewriter_auto_negative=True,
    )

    backends = {
        id(stage.backend)
        for stage in stages
        if isinstance(
            stage, (LingBotVideoPromptRewriteStage, LingBotVideoAutoNegativeStage)
        )
    }
    assert len(backends) == 1


_DEFAULT_NEGATIVE = {
    "universal_negative": {
        "visual_quality": ["low quality", "underexposed", "crushed blacks"],
        "artistic_style": ["painting", "cartoon"],
        "physical_plausibility": ["objects defying gravity"],
        "temporal_and_motion_stability": ["motion blur", "warping"],
    }
}


def test_auto_negative_keeps_the_default_order_and_drops_only_deletions():
    pruned = {
        "universal_negative": {
            # Reordered, with an invented term the model must not be able to add.
            "visual_quality": ["invented", "low quality"],
            "artistic_style": ["painting", "cartoon"],
            "physical_plausibility": ["objects defying gravity"],
            "temporal_and_motion_stability": ["warping", "motion blur"],
        }
    }

    out = prune_negative(_DEFAULT_NEGATIVE, pruned, "a bright meadow at noon")

    assert out["universal_negative"]["visual_quality"] == ["low quality"]
    assert out["universal_negative"]["temporal_and_motion_stability"] == [
        "motion blur",
        "warping",
    ]


def test_auto_negative_restores_a_block_the_caption_never_asked_to_drop():
    emptied = {
        "universal_negative": {"physical_plausibility": [], "artistic_style": []}
    }

    live_action = prune_negative(_DEFAULT_NEGATIVE, emptied, "a cinematic car chase")
    assert live_action["universal_negative"]["physical_plausibility"] == [
        "objects defying gravity"
    ]
    assert live_action["universal_negative"]["artistic_style"] == [
        "painting",
        "cartoon",
    ]

    surreal = prune_negative(
        _DEFAULT_NEGATIVE, emptied, "a surreal dreamlike watercolor painting"
    )
    assert surreal["universal_negative"]["physical_plausibility"] == []
    assert surreal["universal_negative"]["artistic_style"] == []


def test_auto_negative_deletes_terms_the_caption_clearly_wants():
    kept = {"universal_negative": dict(_DEFAULT_NEGATIVE["universal_negative"])}

    out = prune_negative(
        _DEFAULT_NEGATIVE, kept, "a moody night scene with deep shadows"
    )

    assert out["universal_negative"]["visual_quality"] == ["low quality"]


def test_auto_negative_turns_dynamic_batching_off():
    # A merged request carries one negative for several captions, so pruning it
    # per request is only possible while merging is off.
    config = LingBotVideoMoEPipelineConfig()
    assert config.supports_dynamic_batching()

    config.rewriter_auto_negative = True
    assert not config.supports_dynamic_batching()


def test_auto_negative_needs_a_rewriter_backend():
    config = LingBotVideoMoEPipelineConfig()
    config.check_pipeline_config()

    config.rewriter_auto_negative = True
    with pytest.raises(ValueError, match="rewriter_auto_negative"):
        config.check_pipeline_config()

    config.rewriter_url = "http://host:30000"
    config.check_pipeline_config()

    local = LingBotVideoMoEPipelineConfig()
    local.rewriter_auto_negative = True
    local.rewriter_model_path = "/models/base"
    local.rewriter_adapter_path = "/models/adapter"
    local.check_pipeline_config()


@pytest.mark.parametrize(
    "negative_prompt",
    [
        "blurry, low quality",
        '{"universal_negative": []}',
        '{"universal_negative": "bad"}',
        '{"universal_negative": {"visual_quality": "not a list"}}',
        '{"universal_negative": {"visual_quality": [1, 2]}}',
    ],
)
def test_auto_negative_skips_a_negative_it_cannot_prune(negative_prompt):
    # A request may send any string here, so an unprunable one is passed through
    # rather than indexed as categories.
    stage = object.__new__(LingBotVideoAutoNegativeStage)
    stage.backend = _FakeBackend([])
    batch = SimpleNamespace(
        prompt='{"comprehensive_description": "a night scene"}',
        negative_prompt=negative_prompt,
        num_frames=121,
        fps=24,
        image_path=None,
    )

    stage.forward(batch, None)

    assert batch.negative_prompt == negative_prompt
    assert stage.backend.calls == []


def test_auto_negative_hints_match_whole_words_only():
    default = {
        "universal_negative": {
            "visual_quality": [
                "low quality",
                "underexposed",
                "subject hidden in darkness",
                "crushed blacks",
            ]
        }
    }

    def survivors(caption):
        return prune_negative(default, default, caption)["universal_negative"][
            "visual_quality"
        ]

    # "knight" contains "night" and "three-dimensional" contains "dim".
    assert (
        survivors("A knight in bright daylight")
        == default["universal_negative"]["visual_quality"]
    )
    assert (
        survivors("A bright three-dimensional sculpture")
        == default["universal_negative"]["visual_quality"]
    )

    assert survivors("A dim night scene") == ["low quality"]
    assert survivors("dimly lit room") == ["low quality"]
    # Multi-word and hyphenated hints still match.
    assert survivors("shot in low-light") == ["low quality"]
    assert survivors("a low light interior") == ["low quality"]
