"""Config, checkpoint loading, logits, vision, and multimodal serving."""

import base64
import copy
import io
import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load, save
from torch import nn

from sglang.srt.configs.inkling2 import (
    INKLING2_ARCHS,
    Inkling2LoopMMConfig,
    Inkling2MMConfig,
    Inkling2ModelConfig,
)
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
from sglang.srt.models.inkling_common.inkling2_attn import (
    SLICED_QKVR_MIN_TOKENS,
    AttentionSources,
    Inkling2Attention,
)
from sglang.srt.multimodal.inkling2 import Inkling2ImageProcessor
from sglang.srt.multimodal.media_io import ImageData
from sglang.srt.parser.inkling_renderer import render_inkling_messages
from sglang.srt.parser.inkling_tokenizer import IMAGE_TOKEN_ID
from sglang.srt.runtime_context import SpawnRanks, override_platform, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.device import is_host_cpu_arm64
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.inkling2_test_utils import precision_group
from sglang.test.parallel_groups import publish

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


def _loop_text_config(release_config, *, is_varloop):
    loop = dict(
        core_start=6, core_end=18, num_passes=3, gate_rank=64, is_varloop=is_varloop
    )
    return Inkling2ModelConfig(**release_config["text_config"], loop_config=loop)


def test_loop_cache_layout(release_config):
    """KV layers follow the pass count unless shared; conv states stay per visit."""
    text = _loop_text_config(release_config, is_varloop=True)
    assert text.num_attention_layers == text.num_layer_visits == 72
    assert [text.cache_layer_id(6, p) for p in range(3)] == [6, 48, 60]

    text.configure_loop(num_passes=2, share_cache=True)
    assert text.num_layer_visits == 60 and text.num_attention_layers == 48
    assert text.full_attention_layer_ids == list(range(48))
    assert text.conv_layer_ids == list(range(60))
    assert [text.cache_layer_id(17, p) for p in range(2)] == [17, 17]
    assert [text.conv_layer_id(17, p) for p in range(2)] == [17, 59]
    assert [text.conv_layer_id(i, 1) for i in range(6, 18)] == list(range(48, 60))

    text.configure_loop(num_passes=1, share_cache=False)
    assert text.num_attention_layers == text.num_layer_visits == 48


def test_loop_config_loads_through_auto_config(tmp_path, release_config):
    """A looped checkpoint must load as the loop config class with a typed text config."""
    release_config["model_type"] = "inkling2_loop"
    release_config["architectures"] = ["Inkling2LoopForConditionalGeneration"]
    release_config["text_config"]["loop_config"] = dict(
        core_start=6, core_end=18, num_passes=3, gate_rank=64, is_varloop=False
    )
    (tmp_path / "config.json").write_text(json.dumps(release_config))
    config = get_config(str(tmp_path), trust_remote_code=False)
    assert isinstance(config, Inkling2LoopMMConfig)
    assert isinstance(config.text_config, Inkling2ModelConfig)
    assert config.text_config.num_layer_visits == 72


@pytest.mark.parametrize("arch", INKLING2_ARCHS)
def test_every_target_arch_name_reaches_inkling2_paths(arch):
    """Each config.json architecture name must resolve to the model, its image
    processor and the Inkling2 chat renderer; a missed site falls back silently."""
    from sglang.srt.arg_groups.model_override_base import _MODEL_OVERRIDE_FNS
    from sglang.srt.arg_groups.model_overrides.inkling import _inkling_overrides
    from sglang.srt.entrypoints.openai.chat_encoding import resolve_chat_encoding_spec
    from sglang.srt.models.registry import ModelRegistry
    from sglang.srt.multimodal.processors.inkling2 import Inkling2MultimodalProcessor

    assert _inkling_overrides in _MODEL_OVERRIDE_FNS[arch]
    model_cls, _ = ModelRegistry.resolve_model_cls([arch])
    assert issubclass(model_cls, Inkling2ForConditionalGeneration)
    assert arch in [m.__name__ for m in Inkling2MultimodalProcessor.models]
    hf_config = SimpleNamespace(architectures=[arch], model_type="inkling2")
    assert resolve_chat_encoding_spec(hf_config=hf_config, tokenizer=None) == "inkling2"


def test_loop_overrides_need_varloop(release_config):
    fixed = _loop_text_config(release_config, is_varloop=False)
    with pytest.raises(ValueError, match="is_varloop=true"):
        fixed.configure_loop(num_passes=None, share_cache=True)
    with pytest.raises(ValueError, match="at least 2 passes"):
        loop = {**fixed.loop_config, "num_passes": 1}
        Inkling2ModelConfig(**release_config["text_config"], loop_config=loop)
    varloop = _loop_text_config(release_config, is_varloop=True)
    with pytest.raises(ValueError, match="at least 1 passes"):
        varloop.configure_loop(num_passes=0, share_cache=False)
    with pytest.raises(ValueError, match="model_type inkling2_loop"):
        Inkling2MMConfig(text_config=varloop)


@pytest.mark.parametrize(
    "hub_stub",
    [
        None,
        dict(quant_method="modelopt", quant_algo="NVFP4"),
        dict(
            quant_method="fp8",
            activation_scheme="dynamic",
            weight_block_size=[1, 32],
            scale_fmt="ue8m0",
            modules_to_not_convert=["model.llm.unembed", "model.visual"],
        ),
    ],
    ids=["none", "nvfp4", "mxfp8"],
)
def test_precision_file_is_detected_without_launch_flag(
    tmp_path, release_config, hub_stub
):
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
    if hub_stub:
        # The published checkpoints add these stubs so the Hub counts parameters.
        release_config["quantization_config"] = hub_stub
    (tmp_path / "config.json").write_text(json.dumps(release_config))
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(precision))
    config = ModelConfig(str(tmp_path), trust_remote_code=False)
    assert config.quantization == "inkling2"
    assert config.hf_quant_config == {}
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


def test_mega_nvfp4_padding_matches_padded_checkpoint():
    """Intermediate 1152 runs NVFP4 MegaMoE at 1280: zero gate/up rows, zero w2
    columns and zero scales, laid out as if the checkpoint were 1280 wide."""
    from sglang.srt.layers.quantization.utils import swizzle_blockscale

    E, H, I, pad = 2, 256, 1152, 128
    gen = torch.Generator().manual_seed(0)

    def rand(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, generator=gen)

    w13, w2 = rand(E, 2 * I, H // 2), rand(E, H, I // 2)
    s13, s2 = rand(E, 2 * I, H // 16), rand(E, H, I // 16)
    experts = moe.Inkling2Experts.__new__(moe.Inkling2Experts)
    nn.Module.__init__(experts)
    experts.w13_weight, experts.w2_weight = w13, w2
    experts.w13_scale = swizzle_blockscale(s13).flatten(1)
    experts.w2_scale = swizzle_blockscale(s2).flatten(1)
    experts._mega_padded = None
    experts._empty_weight = lambda shape, dtype: torch.empty(shape, dtype=dtype)
    pad_rows = torch.nn.functional.pad
    expected = (
        pad_rows(w13, (0, 0, 0, 2 * pad)),
        swizzle_blockscale(pad_rows(s13, (0, 0, 0, 2 * pad))).flatten(1),
        pad_rows(w2, (0, pad // 2)),
        swizzle_blockscale(pad_rows(s2, (0, pad // 16))).flatten(1),
    )
    first = experts._mega_nvfp4_weights()
    for got, want in zip(first, expected):
        assert torch.equal(got, want)
    # A reload rewrites the same storage (captured graphs hold the addresses).
    experts.w2_weight.zero_()
    again = experts._mega_nvfp4_weights()
    assert [t.data_ptr() for t in again] == [t.data_ptr() for t in first]
    assert not again[2].any()


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
    # Served weights never require grad; out= matmuls reject ones that do.
    head = torch.nn.Linear(4, padded, bias=False, device=device).requires_grad_(False)
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


def test_multiple_images_have_independent_attention(encoder):
    patches = torch.randn(3, 2, 4, 4, 3, dtype=torch.bfloat16)
    grid = torch.tensor([[1, 1, 1], [1, 1, 2]])
    # Workaround for #53277: the 4-gpu-gb300 host g++ rejects Inductor's CPU
    # -march=armv9-a; drop once that runner has GCC >= 12.
    stance = "force_eager" if is_host_cpu_arm64() else "default"
    with torch.compiler.set_stance(stance), torch.inference_mode():
        expected = encoder(patches, grid)
        modified = patches.clone()
        modified[1:] *= -2
        actual = encoder(modified, grid)
    assert expected.shape == (3, 24)
    assert expected.dtype == torch.bfloat16
    assert torch.isfinite(expected).all()
    assert torch.equal(expected[:1], actual[:1])
    assert not torch.equal(expected[1:], actual[1:])


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
        sources.append(ImageData(url=uri))
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


def test_project_sources_gates_3src_gemv_to_captured_decode_and_verify(monkeypatch):
    """Keep eager short extends/decodes and captured extends on the stacked path."""
    from sglang.kernels.ops.gemm import sm12x_bf16_gemv
    from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode
    from sglang.srt.models.inkling_common import inkling2_attn
    from sglang.srt.models.inkling_common.inkling2_attn import (
        AttentionSources,
        Inkling2Attention,
    )

    split_sizes = [32, 32, 32, 32, 32, 32]
    total_n = sum(split_sizes)
    calls = []

    class _StubQKVR:
        def __init__(self):
            self.weight = torch.zeros(total_n, 64, dtype=torch.bfloat16)
            self.bias = None

        def __call__(self, x):
            calls.append("stacked")
            return torch.zeros(x.shape[0], total_n, dtype=x.dtype), None

    attn = Inkling2Attention.__new__(Inkling2Attention)
    nn.Module.__init__(attn)
    attn.split_sizes = split_sizes
    attn.qkvr = _StubQKVR()
    monkeypatch.setattr(inkling2_attn, "MergedColumnParallelLinear", _StubQKVR)
    monkeypatch.setattr(attn, "_can_use_qkvr_fast_path", lambda: True)
    monkeypatch.setattr(
        inkling2_attn, "get_platform", lambda: SimpleNamespace(is_sm120=True)
    )
    monkeypatch.setattr(
        inkling2_attn, "current_arch", lambda: SimpleNamespace(is_sm=lambda *_: True)
    )
    monkeypatch.setattr(sm12x_bf16_gemv, "use_sm12x_bf16_gemv_3src", lambda **_: True)

    def _fake_3src(*, xq, weight, **_):
        calls.append("gemv_3src")
        return torch.zeros(xq.shape[0], weight.shape[0], dtype=xq.dtype)

    monkeypatch.setattr(sm12x_bf16_gemv, "sm12x_bf16_gemv_3src", _fake_3src)

    sources = AttentionSources(
        torch.zeros(4, 64, dtype=torch.bfloat16),
        torch.zeros(4, 64, dtype=torch.bfloat16),
        torch.zeros(4, 64, dtype=torch.bfloat16),
    )

    for mode in (ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
        calls.clear()
        attn._project_sources(sources, mode)
        assert calls == ["stacked"], f"eager {mode} should stay on stacked path"

    with model_capture_mode():
        for mode in (ForwardMode.EXTEND, ForwardMode.MIXED):
            calls.clear()
            attn._project_sources(sources, mode)
            assert calls == ["stacked"], f"captured {mode} should stay on stacked path"
        for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
            calls.clear()
            attn._project_sources(sources, mode)
            assert calls == ["gemv_3src"], f"captured {mode} should use 3-source GEMV"


@contextmanager
def _qkvr_target(
    *,
    tp_rank: int = 0,
    tp_size: int = 1,
    q_bias: bool = False,
    d_rel: int = 1,
):
    config = Inkling2ModelConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        d_rel=d_rel,
        rel_extent=16,
        sliding_window_size=16,
        q_bias=q_bias,
        o_bias=False,
        local_layer_ids=[],
        n_layers_per_attn_res_block=1,
        n_routed_experts=2,
        num_experts_per_tok=1,
    )
    reset_context()
    try:
        publish(
            ServerArgs(model_path="dummy", device="cpu", tp_size=tp_size),
            role="test",
            ranks=SpawnRanks(world_rank=tp_rank),
        )
        with (
            override_platform(is_sm120=False, is_sm90=False),
            set_default_torch_dtype(torch.float32),
        ):
            attn = Inkling2Attention(
                config, layer_id=0, is_local=False, prefix="llm.layers.0.attn"
            )
        target = Inkling2ForConditionalGeneration.__new__(
            Inkling2ForConditionalGeneration
        )
        nn.Module.__init__(target)
        target.visual, target.quant_config = None, None
        target.llm = nn.Module()
        layer = nn.Module()
        layer.attn = nn.Module()
        layer.attn.qkvr = attn.qkvr
        target.llm.layers = nn.ModuleList([layer])
        yield target, attn
    finally:
        reset_context()


def _dyadic(shape, *, seed: int, denom: float):
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(-7, 8, shape, generator=gen, dtype=torch.float32) / denom


def _raw_qkvr_tensors(*, d_rel: int = 1, seed: int = 0):
    specs = (
        ("wq_du", 16),
        ("wk_dv", 8),
        ("wv_dv", 8),
        ("wr_du", 4 * d_rel),
        ("gate_attn", 16),
        ("attn_sink_proj", 4),
    )
    weights = {
        name: _dyadic((rows, 16), seed=seed + i + 1, denom=4.0)
        for i, (name, rows) in enumerate(specs)
    }
    biases = {
        name: _dyadic((rows,), seed=seed + 100 + i + 1, denom=8.0)
        for i, (name, rows) in enumerate(specs)
    }
    return specs, weights, biases


def _reference_project_sources(
    sources, specs, weights, biases, *, tp_rank: int, tp_size: int, active_biases
):
    src_by_proj = {
        "wq_du": sources.q,
        "wk_dv": sources.k,
        "wv_dv": sources.v,
        "wr_du": sources.q,
        "gate_attn": sources.q,
        "attn_sink_proj": sources.q,
    }
    expected = []
    for name, _ in specs:
        w_rank = weights[name].chunk(tp_size, dim=0)[tp_rank]
        b_rank = (
            biases[name].chunk(tp_size, dim=0)[tp_rank]
            if name in active_biases
            else None
        )
        expected.append(F.linear(src_by_proj[name], w_rank, b_rank))
    return tuple(expected)


def _named_qkvr_entries(specs, weights, biases, *, q_bias: bool):
    entries = [
        (f"model.llm.layers.0.attn.{name}.weight", weights[name]) for name, _ in specs
    ]
    if q_bias:
        entries.append(("model.llm.layers.0.attn.wq_du.bias", biases["wq_du"]))
    return entries


def _legacy_packed_qkvr_entries(
    specs,
    weights,
    biases,
    *,
    q_bias: bool,
    pad_rows: int = 0,
    q_bias_only: bool = False,
):
    packed_w = torch.cat([weights[name] for name, _ in specs], dim=0)
    if pad_rows:
        packed_w = torch.cat(
            [packed_w, torch.full((pad_rows, packed_w.shape[1]), 99.0)], dim=0
        )
    entries = [("model.llm.layers.0.attn.qkvr.weight", packed_w)]
    if q_bias:
        if q_bias_only:
            packed_b = biases["wq_du"]
        else:
            packed_b = torch.cat([biases[name] for name, _ in specs], dim=0)
            if pad_rows:
                packed_b = torch.cat([packed_b, torch.full((pad_rows,), 99.0)], dim=0)
        entries.append(("model.llm.layers.0.attn.qkvr.bias", packed_b))
    return entries


def _assert_project_sources_matches(
    attn,
    sources,
    specs,
    weights,
    biases,
    *,
    tp_rank: int,
    tp_size: int,
    local_stride: int,
    active_biases,
):
    with override_platform(is_sm120=False), torch.inference_mode():
        actual = attn._project_sources(sources)
        expected = _reference_project_sources(
            sources,
            specs,
            weights,
            biases,
            tp_rank=tp_rank,
            tp_size=tp_size,
            active_biases=active_biases,
        )
    assert all(t.stride() == (local_stride, 1) for t in actual)
    for got, want in zip(actual, expected, strict=True):
        assert torch.equal(got, want)


@pytest.mark.parametrize(
    "tp_size,d_rel,checkpoint_format,q_bias",
    [
        (1, 1, "named", False),
        (1, 1, "named", True),
        (2, 1, "named", True),
        (2, 1, "legacy_unpadded", False),
        (1, 1, "legacy_unpadded", True),
        (2, 1, "legacy_unpadded", True),
        (2, 1, "legacy_padded", True),
        (2, 4, "legacy_tp1_padded", True),
        (1, 1, "legacy_tp2_padded", True),
    ],
)
def test_qkvr_load_and_project_sources_matches_per_projection_linear(
    tp_size, d_rel, checkpoint_format, q_bias
):
    specs, weights, biases = _raw_qkvr_tensors(d_rel=d_rel, seed=11)
    sum_6 = sum(rows for _, rows in specs)
    local_sum = sum_6 // tp_size
    local_pad = -local_sum % 8
    local_stride = local_sum + local_pad
    if checkpoint_format == "named":
        entries = _named_qkvr_entries(specs, weights, biases, q_bias=q_bias)
        active_biases = {"wq_du"} if q_bias else set()
    else:
        pad_rows = (
            0
            if checkpoint_format == "legacy_unpadded"
            else -sum_6 % 8
            if checkpoint_format == "legacy_tp1_padded"
            else 2 * (-(sum_6 // 2) % 8)
            if checkpoint_format == "legacy_tp2_padded"
            else local_pad * tp_size
        )
        entries = _legacy_packed_qkvr_entries(
            specs, weights, biases, q_bias=q_bias, pad_rows=pad_rows
        )
        active_biases = {name for name, _ in specs} if q_bias else set()

    expected_loaded = {"llm.layers.0.attn.qkvr.weight"} | (
        {"llm.layers.0.attn.qkvr.bias"} if q_bias else set()
    )
    xq = _dyadic((5, 16), seed=501, denom=2.0)
    xk = _dyadic((5, 16), seed=502, denom=2.0)
    xv = _dyadic((5, 16), seed=503, denom=2.0)
    empty = torch.empty((0, 16), dtype=torch.float32)

    for tp_rank in range(tp_size):
        with _qkvr_target(
            tp_rank=tp_rank, tp_size=tp_size, q_bias=q_bias, d_rel=d_rel
        ) as (target, attn):
            assert target.load_weights(entries) == expected_loaded
            if local_pad:
                assert not attn.qkvr.weight[-local_pad:].any()
                if q_bias:
                    assert not attn.qkvr.bias[-local_pad:].any()

            for sources in (
                AttentionSources(xq, xk, xv),
                AttentionSources(xq, xq, xq),
                AttentionSources(xq, xq, xv),
                AttentionSources(empty, empty.clone(), empty.clone()),
            ):
                _assert_project_sources_matches(
                    attn,
                    sources,
                    specs,
                    weights,
                    biases,
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                    local_stride=local_stride,
                    active_biases=active_biases,
                )

            if q_bias and checkpoint_format == "legacy_unpadded":
                w_ptr, b_ptr = attn.qkvr.weight.data_ptr(), attn.qkvr.bias.data_ptr()
                sources = AttentionSources(xq, xk, xv)
                _, w2, b2 = _raw_qkvr_tensors(d_rel=d_rel, seed=211)
                target.load_weights(_named_qkvr_entries(specs, w2, b2, q_bias=True))
                assert attn.qkvr.weight.data_ptr() == w_ptr
                assert attn.qkvr.bias.data_ptr() == b_ptr
                _assert_project_sources_matches(
                    attn,
                    sources,
                    specs,
                    w2,
                    b2,
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                    local_stride=local_stride,
                    active_biases={"wq_du"},
                )
                target.load_weights(entries)
                _, w3, b3 = _raw_qkvr_tensors(d_rel=d_rel, seed=311)
                target.load_weights(
                    _legacy_packed_qkvr_entries(
                        specs, w3, b3, q_bias=True, q_bias_only=True
                    )
                )
                assert attn.qkvr.bias.data_ptr() == b_ptr
                _assert_project_sources_matches(
                    attn,
                    sources,
                    specs,
                    w3,
                    b3,
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                    local_stride=local_stride,
                    active_biases={"wq_du"},
                )


def test_qkvr_load_weights_rejects_invalid_and_mixed_checkpoints():
    """Malformed qkvr checkpoints fail loudly instead of silently padding."""
    specs, weights, biases = _raw_qkvr_tensors(d_rel=1, seed=31)
    named = _named_qkvr_entries(specs, weights, biases, q_bias=False)
    packed = _legacy_packed_qkvr_entries(specs, weights, biases, q_bias=False)
    sum_6 = packed[0][1].shape[0]
    q_full = weights["wq_du"].shape[0]
    qkvr_w = "model.llm.layers.0.attn.qkvr.weight"
    wq_w = "model.llm.layers.0.attn.wq_du.weight"

    for bad_entries, match in (
        ([named[0], packed[0]], "mixes packed and unpacked"),
        ([packed[0], named[0]], "mixes packed and unpacked"),
        ([packed[0], packed[0]], "Repeated checkpoint tensor"),
        ([named[0], named[0]], "Repeated checkpoint tensor"),
        (named[:-1], "Missing checkpoint tensors"),
        ([(qkvr_w, torch.tensor(0.0))], "expected ndim"),
        ([(qkvr_w, torch.zeros(sum_6, 15))], "expected ndim"),
        ([(qkvr_w, torch.zeros(sum_6 - 1, 16))], "unexpected packed qkvr rows"),
        ([(wq_w, torch.zeros(q_full - 1, 16))], "expected shape"),
        ([(wq_w, torch.zeros(q_full, 15))], "expected shape"),
    ):
        with _qkvr_target(tp_rank=0, tp_size=2, q_bias=False) as (target, _):
            with pytest.raises(ValueError, match=match):
                target.load_weights(bad_entries)

    with _qkvr_target(tp_rank=0, tp_size=2, q_bias=True) as (target, _):
        with pytest.raises(ValueError, match="Missing checkpoint tensors"):
            target.load_weights(
                [*named, ("model.llm.layers.0.attn.wk_dv.bias", biases["wk_dv"])]
            )

    with _qkvr_target(tp_rank=0, tp_size=2, q_bias=True) as (target, _):
        with pytest.raises(ValueError, match="expected shape"):
            target.load_weights(
                [
                    *named,
                    ("model.llm.layers.0.attn.wq_du.bias", torch.zeros(q_full - 1)),
                ]
            )

    with _qkvr_target(tp_rank=0, tp_size=1, q_bias=False, d_rel=4) as (target, _):
        with pytest.raises(ValueError, match="unexpected packed qkvr rows"):
            target.load_weights([(qkvr_w, torch.zeros(96, 16))])


def _assert_cuda_3slice_prefill(target, attn, specs, biases, monkeypatch):
    int_weights = {
        name: torch.randint(
            -2,
            3,
            (rows, 16),
            generator=torch.Generator().manual_seed(700 + i),
            dtype=torch.float32,
        )
        for i, (name, rows) in enumerate(specs)
    }
    target.load_weights(_named_qkvr_entries(specs, int_weights, biases, q_bias=False))
    attn.to(device="cuda", dtype=torch.bfloat16)
    w_cuda = attn.qkvr.weight
    m_min = SLICED_QKVR_MIN_TOKENS
    xq_c = torch.randint(
        -1, 2, (m_min, 16), generator=torch.Generator().manual_seed(710)
    ).to(device="cuda", dtype=torch.bfloat16)
    xk_c = torch.randint(
        -1, 2, (m_min, 16), generator=torch.Generator().manual_seed(711)
    ).to(device="cuda", dtype=torch.bfloat16)
    xv_c = torch.randint(
        -1, 2, (m_min, 16), generator=torch.Generator().manual_seed(712)
    ).to(device="cuda", dtype=torch.bfloat16)
    src_min = AttentionSources(xq_c, xk_c, xv_c)
    src_below = AttentionSources(
        xq_c[: m_min - 1], xk_c[: m_min - 1], xv_c[: m_min - 1]
    )

    def _fail_forward(*args, **kwargs):
        raise AssertionError("must use sliced prefill")

    with torch.inference_mode():
        assert attn._can_use_qkvr_fast_path()
        handle = nn.modules.module.register_module_forward_hook(lambda m, i, o: o)
        try:
            assert not attn._can_use_qkvr_fast_path()
        finally:
            handle.remove()
        assert attn._can_slice_qkvr_prefill(
            sources=src_min, weight=w_cuda, bias=None, v_lo=4, q_lo=8
        )
        assert not attn._can_slice_qkvr_prefill(
            sources=src_below, weight=w_cuda, bias=None, v_lo=4, q_lo=8
        )
        assert not attn._can_slice_qkvr_prefill(
            sources=src_min,
            weight=w_cuda,
            bias=torch.zeros(w_cuda.shape[0], device="cuda", dtype=torch.bfloat16),
            v_lo=4,
            q_lo=8,
        )
        assert not attn._can_slice_qkvr_prefill(
            sources=AttentionSources(xq_c.t().contiguous().t(), xk_c, xv_c),
            weight=w_cuda,
            bias=None,
            v_lo=4,
            q_lo=8,
        )
        assert not attn._can_slice_qkvr_prefill(
            sources=AttentionSources(xq_c[:1].expand(m_min, 16), xk_c, xv_c),
            weight=w_cuda,
            bias=None,
            v_lo=4,
            q_lo=8,
        )
        with torch.autocast("cuda", dtype=torch.float16):
            assert not attn._can_slice_qkvr_prefill(
                sources=src_min, weight=w_cuda, bias=None, v_lo=4, q_lo=8
            )
        with override_platform(is_sm120=True), monkeypatch.context() as mp:
            mp.setattr(attn.qkvr, "forward", _fail_forward)
            actual = attn._project_sources(src_min)
        w_cuda_ref = {
            name: w.to(device="cuda", dtype=torch.bfloat16)
            for name, w in int_weights.items()
        }
        expected = _reference_project_sources(
            src_min,
            specs,
            w_cuda_ref,
            biases,
            tp_rank=1,
            tp_size=2,
            active_biases=set(),
        )
        assert all(t.stride() == (w_cuda.shape[0], 1) for t in actual)
        for got, want in zip(actual, expected, strict=True):
            assert torch.equal(got, want)
    with torch.enable_grad():
        assert not attn._can_slice_qkvr_prefill(
            sources=AttentionSources(xq_c.clone().requires_grad_(True), xk_c, xv_c),
            weight=w_cuda,
            bias=None,
            v_lo=4,
            q_lo=8,
        )


def test_qkvr_3slice_prefill_parity_and_guards(monkeypatch):
    specs, weights, biases = _raw_qkvr_tensors(d_rel=1, seed=41)
    entries = _named_qkvr_entries(specs, weights, biases, q_bias=False)
    with _qkvr_target(tp_rank=1, tp_size=2, q_bias=False) as (target, attn):
        target.load_weights(entries)
        sources = AttentionSources(
            _dyadic((5, 20), seed=601, denom=2.0)[:, :16],
            _dyadic((5, 20), seed=611, denom=2.0)[:, :16],
            _dyadic((5, 20), seed=621, denom=2.0)[:, :16],
        )
        with torch.inference_mode():
            projected = attn._project_qkvr_3src_sliced(
                sources=sources, weight=attn.qkvr.weight, v_lo=4, q_lo=8
            )
            actual = attn._split_qkvr(projected)
            expected = _reference_project_sources(
                sources,
                specs,
                weights,
                biases,
                tp_rank=1,
                tp_size=2,
                active_biases=set(),
            )
        assert all(t.stride() == (projected.shape[1], 1) for t in actual)
        for got, want in zip(actual, expected, strict=True):
            assert torch.equal(got, want)

        cpu_src = AttentionSources(
            torch.zeros((SLICED_QKVR_MIN_TOKENS, 16), dtype=torch.bfloat16),
            torch.zeros((SLICED_QKVR_MIN_TOKENS, 16), dtype=torch.bfloat16),
            torch.zeros((SLICED_QKVR_MIN_TOKENS, 16), dtype=torch.bfloat16),
        )
        assert not attn._can_slice_qkvr_prefill(
            sources=cpu_src,
            weight=attn.qkvr.weight.to(torch.bfloat16),
            bias=None,
            v_lo=4,
            q_lo=8,
        )

        if torch.cuda.is_available():
            _assert_cuda_3slice_prefill(target, attn, specs, biases, monkeypatch)


def test_sharded_state_loader_distinguishes_inkling2_qkvr_layout(tmp_path, monkeypatch):
    """sharded_state tags new kvqr dumps and rejects pre-change dumps cleanly."""
    from safetensors.torch import safe_open, save_file

    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.model_loader import loader as loader_mod
    from sglang.srt.model_loader.loader import ShardedStateLoader

    monkeypatch.setattr(loader_mod, "_checkpoint_shard_rank", lambda: 0)
    monkeypatch.setattr(loader_mod, "_get_quantization_config", lambda *_: None)
    monkeypatch.setattr(loader_mod, "post_load_weights", lambda *_: None)

    specs, weights, biases = _raw_qkvr_tensors(d_rel=1, seed=53)
    entries = _named_qkvr_entries(specs, weights, biases, q_bias=False)

    new_dir = tmp_path / "new_sharded"
    new_dir.mkdir()
    with _qkvr_target(tp_rank=0, tp_size=1, q_bias=False) as (target, attn):
        target.load_weights(entries)
        target.llm.layers[0].attn = attn
        expected_weight = attn.qkvr.weight.detach().clone()
        ShardedStateLoader.save_model(target, str(new_dir), max_size=2500)

    shard_files = sorted(new_dir.glob("*.safetensors"))
    assert len(shard_files) >= 2
    for shard in shard_files:
        with safe_open(str(shard), framework="pt") as f:
            assert f.metadata() == {"inkling2_qkvr_layout": "kvqr"}

    loader = ShardedStateLoader(LoadConfig(load_format="sharded_state"))
    device_cfg = SimpleNamespace(device="cpu")

    with _qkvr_target(tp_rank=0, tp_size=1, q_bias=False) as (reloaded, reloaded_attn):
        reloaded.llm.layers[0].attn = reloaded_attn
        reloaded_attn.qkvr.weight.data.zero_()
        monkeypatch.setattr(loader_mod, "_initialize_model", lambda *_: reloaded)
        model_cfg = SimpleNamespace(
            model_path=str(new_dir), revision=None, dtype=torch.float32
        )
        loader.load_model(model_config=model_cfg, device_config=device_cfg)
        assert torch.equal(reloaded_attn.qkvr.weight, expected_weight)

    legacy_dir = tmp_path / "legacy_sharded"
    legacy_dir.mkdir()
    with _qkvr_target(tp_rank=0, tp_size=1, q_bias=False) as (target, attn):
        target.load_weights(entries)
        target.llm.layers[0].attn = attn
        legacy_state = ShardedStateLoader._filter_subtensors(target.state_dict())
        save_file(
            legacy_state,
            str(legacy_dir / ShardedStateLoader.DEFAULT_PATTERN.format(rank=0, part=0)),
        )

    with _qkvr_target(tp_rank=0, tp_size=1, q_bias=False) as (reloaded, reloaded_attn):
        reloaded.llm.layers[0].attn = reloaded_attn
        monkeypatch.setattr(loader_mod, "_initialize_model", lambda *_: reloaded)
        model_cfg = SimpleNamespace(
            model_path=str(legacy_dir), revision=None, dtype=torch.float32
        )
        with pytest.raises(ValueError, match="save_sharded_model"):
            loader.load_model(model_config=model_cfg, device_config=device_cfg)

    plain_dir = tmp_path / "plain_sharded"
    plain_dir.mkdir()
    plain_src = nn.Linear(8, 8, bias=False)
    ShardedStateLoader.save_model(plain_src, str(plain_dir))
    plain_file = plain_dir / ShardedStateLoader.DEFAULT_PATTERN.format(rank=0, part=0)
    with safe_open(str(plain_file), framework="pt") as f:
        assert f.metadata() is None
    plain_dst = nn.Linear(8, 8, bias=False)
    plain_dst.weight.data.zero_()
    monkeypatch.setattr(loader_mod, "_initialize_model", lambda *_: plain_dst)
    loader.load_model(
        model_config=SimpleNamespace(
            model_path=str(plain_dir), revision=None, dtype=torch.float32
        ),
        device_config=device_cfg,
    )
    assert torch.equal(plain_dst.weight, plain_src.weight)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
