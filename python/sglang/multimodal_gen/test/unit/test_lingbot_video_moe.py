# SPDX-License-Identifier: Apache-2.0

import json
import logging
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import PIL.Image
import pytest
import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.kernels.ops.diffusion.sites.gated_residual_site import (
    LINGBOT_VIDEO_GATED_RESIDUAL,
)
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
from sglang.multimodal_gen.configs.task_type import DataType, ModelTaskType
from sglang.multimodal_gen.registry import get_model_info
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
    LingBotVideoRMSNorm,
    LingBotVideoTransformer3DModel,
    _joint_position_ids,
)
from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import (
    VaeFastPathGate,
    register_vae_fast_path_gate,
)
from sglang.multimodal_gen.runtime.pipelines.lingbot_video_moe import (
    LingBotVideoPipeline,
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
    image_conditioning,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe import (
    refiner_stages as refiner_stages_module,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.i2v import (
    COND_LATENT_KEY,
    VLM_IMAGE_KEY,
    condition_frame,
    pixel_to_vlm_image,
    preprocess_condition_image,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.refiner import (
    compute_refiner_sigmas,
    prepare_refiner_latent,
    resize_video_pixels,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.text_encoding import (
    IMG_PROMPT_TEMPLATE,
    PROMPT_TEMPLATE,
    LingBotVideoTextEncodingStage,
)


@pytest.mark.parametrize(
    "pipeline_cls", [LingBotVideoPipeline, LingBotVideoRefinerPipeline]
)
def test_moe_path_resolves_moe_configs(tmp_path, pipeline_cls):
    model_dir = tmp_path / "lingbot-video-moe-30b-a3b"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text(
        json.dumps(
            {"_class_name": pipeline_cls.pipeline_name, "_diffusers_version": "0.37.1"}
        )
    )
    for subdir in ("scheduler", "text_encoder", "processor", "transformer", "vae"):
        (model_dir / subdir).mkdir()
    info = get_model_info(str(model_dir), backend="sglang")
    assert info.pipeline_cls is pipeline_cls
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


class _LinearAttn(torch.nn.Linear):
    to_q = property(lambda self: self)

    def forward(self, x, *args, **kwargs):
        return super().forward(x)


def test_block_per_sample_modulation_matches_per_token():
    batch, seq_len, hidden = 2, 5, 16
    torch.manual_seed(0)
    block = object.__new__(LingBotVideoBlock)
    torch.nn.Module.__init__(block)
    block.scale_shift_table = torch.nn.Parameter(torch.randn(1, 6 * hidden))
    for name in ("norm1", "norm_post_attn", "norm2", "norm_post_ffn"):
        setattr(block, name, LingBotVideoRMSNorm(hidden))
    block.attn = _LinearAttn(hidden, hidden)
    block.ffn = torch.nn.Linear(hidden, hidden)
    x = torch.randn(batch, seq_len, hidden)
    temb6 = torch.randn(batch, 6 * hidden)

    out = block(x, temb6, None)
    # The enabled gated site expands modulation to per-token (B, S, 6D) views;
    # on CPU the fused kernels decline, so this is the per-token reference.
    setattr(block, LINGBOT_VIDEO_GATED_RESIDUAL.enabled_attr, True)
    expected = block(x, temb6, None)

    assert out.shape == x.shape
    torch.testing.assert_close(out, expected)


def test_failed_norm_modulate_kernel_is_reported_and_not_retried(monkeypatch, caplog):
    """A failing fused RMSNorm+modulate kernel is reported once and the site
    then stays on the reference chain instead of relaunching the kernel on
    every call."""
    module = dits_lingbot_video_moe
    launches = []

    def failing_kernel(*args):
        launches.append(args)
        raise RuntimeError("kernel build failed")

    # restores the process-wide latch after the test
    monkeypatch.setattr(module, "_norm_modulate_fusion_failed", False, raising=False)
    monkeypatch.setattr(
        module, "can_use_rmsnorm_scale_shift_per_token", lambda *args: True
    )
    monkeypatch.setattr(module, "rmsnorm_scale_shift_per_token", failing_kernel)
    site = torch.nn.Module()
    module.LINGBOT_VIDEO_GATED_RESIDUAL.mark(site)
    assert module.LINGBOT_VIDEO_GATED_RESIDUAL.mount(site)
    torch.manual_seed(0)
    norm = module.LingBotVideoRMSNorm(8)
    x, scale, shift = torch.randn(3, 2, 4, 8)
    reference = norm(x) * (1.0 + scale) + shift

    with caplog.at_level(logging.WARNING, logger=module.logger.name):
        for _ in range(2):
            out = module._lingbot_norm_modulate(
                site, norm, x, scale, shift, torch.float32
            )
            torch.testing.assert_close(out, reference, rtol=0, atol=0)

    assert len(launches) == 1
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, warnings
    assert "kernel build failed" in warnings[0]


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


def _one_prompt_position_ids(text_len, grid_t, grid_h, grid_w, device):
    """The per-sample (video; text) positions: video t after the prompt, text on t."""
    tt = torch.arange(grid_t, device=device, dtype=torch.int32) + (text_len + 1)
    hh = torch.arange(grid_h, device=device, dtype=torch.int32)
    ww = torch.arange(grid_w, device=device, dtype=torch.int32)
    grid = torch.stack(torch.meshgrid(tt, hh, ww, indexing="ij"), dim=-1).flatten(0, 2)
    text_t = torch.arange(text_len, device=device, dtype=torch.int32) + 1
    text_pos = torch.stack(
        [text_t, torch.zeros_like(text_t), torch.zeros_like(text_t)], dim=-1
    )
    return torch.cat([grid, text_pos], dim=0)


def test_joint_position_ids_match_reference_and_cover_padding():
    dev = torch.device("cpu")
    gt, gh, gw = 2, 3, 4
    n_video = gt * gh * gw

    # B==1, no padding: byte-identical to the per-sample reference.
    vec = _joint_position_ids(torch.tensor([5]), gt, gh, gw, 5, dev)
    torch.testing.assert_close(vec, _one_prompt_position_ids(5, gt, gh, gw, dev))

    # B==1 with padding: real tokens match the text_len=4 reference; the extra
    # padding row is (0,0,0). vec has n_video+L rows (matches q for B*S).
    vec_p = _joint_position_ids(torch.tensor([4]), gt, gh, gw, 5, dev)
    torch.testing.assert_close(
        vec_p[: n_video + 4], _one_prompt_position_ids(4, gt, gh, gw, dev)
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
            vec_b[start : start + real], _one_prompt_position_ids(t, gt, gh, gw, dev)
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


@pytest.mark.parametrize("batch_size", [1, 2])
def test_base_and_refiner_encode_sample_each_posterior_once(monkeypatch, batch_size):
    monkeypatch.setattr(
        image_conditioning, "get_local_torch_device", lambda: torch.device("cpu")
    )
    vae = torch.nn.Identity()
    vae.encode = lambda x: SimpleNamespace(
        sample=lambda gen: x + randn_tensor(x.shape, generator=gen)
    )
    config = SimpleNamespace(get_decode_scale_and_shift=lambda **_: (2.0, 0.25))
    args = SimpleNamespace(pipeline_config=config, disable_autocast=True)
    pixels = torch.full((batch_size, 3, 1, 2, 2), 0.75)

    def generators():
        seeds = [torch.Generator().manual_seed(i) for i in range(batch_size)]
        return seeds[0] if batch_size == 1 else seeds

    actual = image_conditioning.encode_pixels(
        vae, pixels, args, torch.float32, generators()
    )
    expected = (0.5 + randn_tensor(pixels.shape, generator=generators()) - 0.25) * 2.0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_condition_frame_requires_one_reference():
    image = PIL.Image.new("RGB", (32, 32))
    assert condition_frame(image) is condition_frame([image]) is image
    with pytest.raises(ValueError, match="exactly one"):
        condition_frame([image, image])


@pytest.mark.parametrize(
    "height,width,expected",
    [(468, 832, (832, 480)), (30, 30, (64, 64)), (48, 80, (64, 64))],
)
def test_vlm_image_is_patch_aligned(height, width, expected):
    image = pixel_to_vlm_image(torch.ones(1, 3, 1, height, width))
    assert image.mode == "RGB"
    assert image.size == expected


@pytest.mark.parametrize("conditioned", [False, True])
def test_denoising_reapplies_only_the_conditioned_prefix(conditioned):
    stage = object.__new__(LingBotVideoDenoisingStage)
    latents = torch.randn(1, 16, 3, 2, 2, dtype=torch.float32)
    cond = torch.full_like(latents[:, :, :1], 7.0, dtype=torch.bfloat16)
    batch = SimpleNamespace(extra={COND_LATENT_KEY: cond} if conditioned else {})
    out = stage.post_forward_for_ti2v_task(batch, None, None, latents, None)
    if conditioned:
        torch.testing.assert_close(out[:, :, :1], cond.float())
        torch.testing.assert_close(out[:, :, 1:], latents[:, :, 1:])
    else:
        assert out is latents


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


def test_single_frame_keeps_frame_count_and_uses_image_negative_prompt():
    params = _adjusted(1, [])

    assert params.num_frames == 1
    assert params.negative_prompt == DEFAULT_NEGATIVE_PROMPT_IMAGE
    assert "temporal_and_motion_stability" not in params.negative_prompt


def test_explicit_negative_prompt_survives_the_single_frame_default():
    params = _adjusted(1, ["negative_prompt"], negative_prompt="keep me")

    assert params.negative_prompt == "keep me"


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


def test_refiner_sigma_tail_extends_the_descending_base_schedule():
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
    np.testing.assert_array_equal(tailed[:-2], plain)
    assert tailed[0] == pytest.approx(0.85)
    assert (np.diff(tailed) < 0).all()
    assert 0 <= tailed[-1] <= tailed[0] <= 1


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


@pytest.mark.parametrize("use_override", [False, True])
def test_refiner_weights_resolve_component_path(tmp_path, use_override):
    pipeline = object.__new__(LingBotVideoRefinerPipeline)
    pipeline.model_path = str(tmp_path)
    override = tmp_path / "custom_refiner"
    override.mkdir()
    args = SimpleNamespace(
        component_paths={"transformer_2": str(override)} if use_override else {}
    )
    assert pipeline._resolve_component_path(
        args, "transformer_2", "transformer_2"
    ) == str(override if use_override else tmp_path / "refiner")


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

    vae = SimpleNamespace(use_parallel_decode=False)
    assert DecodingStage(vae).parallelism_type == StageParallelismType.MAIN_RANK_ONLY
    assert (
        LingBotVideoRefinerUpscaleStage(vae).parallelism_type
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

    def encode(vae, pixels, server_args, vae_dtype, generators):
        seen["encode"] = vae_dtype
        return torch.zeros(1, 4, 1, 2, 2)

    stage.decode = decode
    monkeypatch.setattr(refiner_stages_module, "encode_pixels", encode)
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
    monkeypatch.setattr(
        refiner_stages_module, "encode_pixels", lambda *args: torch.zeros(1, 4, 1, 2, 2)
    )
    server_args, batch = _upscale_inputs(quality)

    stage.forward(batch, server_args)

    assert seen == {"quality": quality, "fast_path": fast_path}


def test_refiner_condition_frame_keeps_the_base_framing(monkeypatch):
    # A 3:1 image covers a 2:1 base and a 3:2 refiner frame with different crops;
    # frame 0 must show what the upscaled base clip shows.
    array = torch.zeros(32, 96, 3, dtype=torch.uint8)
    array[..., 0] = (torch.arange(96) // 8 * 20).to(torch.uint8)
    image = PIL.Image.fromarray(array.numpy(), mode="RGB")
    stage = _upscale_stage(monkeypatch, _StubVae())
    server_args, batch = _upscale_inputs("exact")
    batch.height, batch.width = 16, 32
    (
        server_args.pipeline_config.refiner_height,
        server_args.pipeline_config.refiner_width,
    ) = 32, 48
    batch.condition_image = [image]
    batch.extra[COND_LATENT_KEY] = torch.ones(1, 3, 1, 16, 32)
    stage.decode = lambda *a, **kw: torch.zeros(1, 3, 2, 16, 32)
    monkeypatch.setattr(
        refiner_stages_module, "encode_pixels", lambda vae, pixels, *args: pixels
    )
    stage.forward(batch, server_args)
    frame = batch.extra[COND_LATENT_KEY]
    base = preprocess_condition_image(image, height=16, width=32)
    torch.testing.assert_close(frame, resize_video_pixels(base, height=32, width=48))
    # Cropping at the refiner size shifts the edges by one 20/255 band.
    cropped_at_refiner = preprocess_condition_image(image, height=32, width=48)
    assert float((frame - cropped_at_refiner).abs().max()) > 0.05
