import argparse
import contextlib
import dataclasses
import json
import os
import re
import socket
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map
from transformers import LlamaConfig

from sglang.kernels.ops.layernorm import hy4_ihc
from sglang.kernels.ops.quantization import act_quant_triton as dsa_act_quant
from sglang.kernels.ops.speculative.dflash import (
    compute_dflash_correct_drafts_and_bonus,
)
from sglang.srt.arg_groups.cuda_graph_hook import handle_cuda_graph_config
from sglang.srt.arg_groups.memory_hook import _req_to_token_table_mb
from sglang.srt.arg_groups.model_hook import handle_language_model_only
from sglang.srt.arg_groups.model_overrides.deepseek_v2 import (
    _reject_unsupported_hyv4_parallelism,
)
from sglang.srt.arg_groups.overrides import (
    collect_model_override_declarations,
    resolution_result,
)
from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.configs.hy_v4 import HYV4Config, HYV4VLConfig, HYV4VLVisionConfig
from sglang.srt.configs.hy_v4_dfly import (
    build_hyv4_dfly_draft_config,
    check_hyv4_dfly_quant_config,
    check_hyv4_dfly_serving,
    resolve_hyv4_dfly_spec,
)
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import (
    AttentionArch,
    ModelConfig,
    dsa_layer_skips_topk,
    is_deepseek_dsa,
    is_multimodal_breakable_cuda_graph_supported,
)
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention import vision as vision_attn
from sglang.srt.layers.attention.dsa import dsa_indexer, dsa_prefill_cuda_graph
from sglang.srt.layers.attention.dsa.utils import compute_dsa_seqlens
from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    DSAMetadata,
)
from sglang.srt.layers.cp.base import init_cp_strategy
from sglang.srt.layers.cp.interleave import InterleaveCPStrategy
from sglang.srt.layers.cp.padding import pad_logical_token_to_physical
from sglang.srt.layers.linear import LinearBase, ReplicatedLinear
from sglang.srt.layers.logits_processor import (
    LogitsProcessorOutput,
    autotune_dummy_run_mode,
)
from sglang.srt.layers.moe.utils import resolve_shared_experts_config
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsFusedMoEMethod,
    CompressedTensorsLinearMethod,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW4A16Mxfp4MoE,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.rotary_embedding import RotaryEmbedding
from sglang.srt.layers.rotary_embedding import base as rope_base
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.managers.overlap_utils import (
    FutureMap,
    RelayPayload,
    resolve_forward_inputs,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.allocation_sizing import (
    get_alloc_reserve_per_decode,
    get_req_to_token_extra_context_len,
)
from sglang.srt.mem_cache.kv_cache_configurator import (
    MIN_DEFAULT_MAX_RUNNING_REQUESTS,
    KVCacheConfigurator,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.cuda_graph_buffer_registry import build_decode_registry
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    get_attn_backend,
)
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode, ForwardMode
from sglang.srt.model_executor.runner import eager_runner, flashinfer_autotune
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    use_breakable_capture_recorder,
)
from sglang.srt.model_loader.loader import _get_quantization_config
from sglang.srt.models import hunyuan_v4, hunyuan_v4_dfly, hunyuan_v4_vl
from sglang.srt.models.deepseek_common.attention_forward_methods import (
    AttnForwardMethod,
    forward_mla,
)
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
from sglang.srt.models.hunyuan_v4 import HYV4MRotaryEmbedding
from sglang.srt.models.hunyuan_v4_dfly import HYV4ForCausalLMDFly
from sglang.srt.models.hunyuan_v4_nextn import HYV4ForCausalLMNextN
from sglang.srt.runtime_context import (
    create_context,
    get_context,
    get_exec,
    get_model,
    get_parallel,
    get_schedule,
    override_platform,
    reset_context,
    use_context,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative import base_spec_worker, dfly_worker_v2
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker, HiCacheDraftPlan
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import commit_accept
from sglang.srt.speculative.dfly_flash_worker_v2 import DFlyFlashWorkerV2
from sglang.srt.speculative.dfly_utils import (
    DFLY_DISCARD_SLOT,
    DFlyProducerCapture,
    VLDFlyRequestState,
    build_dfly_sparse_indices,
    dfly_draft_topk_planes,
    dfly_position_ids,
    dfly_producer_layers,
    dfly_query_positions,
    dfly_source_rows,
    dfly_window_positions,
    plan_dfly_block,
    split_dfly_drafts_for_verify,
)
from sglang.srt.speculative.dfly_validation import (
    dfly_targets_hy4_flash,
    validate_dfly_request,
)
from sglang.srt.speculative.dfly_worker_v2 import (
    DFlyWorkerV2,
    _attach_topk_capture,
    _make_draft_input,
)
from sglang.srt.speculative.draft_worker_common import make_draft_block_spec_info
from sglang.srt.speculative.pp_draft_embedding import (
    SAFETENSORS_INDEX_NAME,
    resolve_draft_embed_and_head,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_tp_sync import SpecTpSync
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import parallel_scope
from sglang.test.test_utils import publish_build_topology

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

TEXT_OVERRIDES = dict(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=1,
    num_attention_heads=4,
    num_key_value_heads=4,
    q_lora_rank=16,
    kv_lora_rank=8,
    qk_nope_head_dim=8,
    qk_rope_head_dim=8,
    v_head_dim=16,
    index_topk=8,
    index_head_dim=8,
    index_n_heads=2,
    max_position_embeddings=512,
)
# The default [12, 10, 10] sections need qk_rope_head_dim=64; a wider index
# head gives the indexer a non-rope part, as in the checkpoint (128 vs 64).
VL_TEXT_OVERRIDES = TEXT_OVERRIDES | {
    "rope_parameters": {"mrope_section": [2, 1, 1]},
    "index_head_dim": 16,
}

VISION_OVERRIDES = dict(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=1,
    num_attention_heads=4,
    max_image_size=64,
)


@pytest.fixture(scope="module")
def gloo_world():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="gloo",
    )
    publish_build_topology(tp_size=1)
    initialize_model_parallel(backend="gloo")
    yield
    destroy_model_parallel()
    destroy_distributed_environment()
    reset_context()


@pytest.fixture
def single_rank(monkeypatch, gloo_world):
    # sdpa runs everywhere; the production backend choice is a runtime-context
    # decision that a one-rank CPU process cannot resolve.
    monkeypatch.setattr(
        vision_attn.VisionAttention,
        "_determine_attention_backend",
        lambda self, passed_backend: "sdpa",
    )
    # RotaryEmbedding only reaches for a vendor kernel on platforms it
    # recognizes; on a bare CPU host the import target does not exist.
    monkeypatch.setattr(rope_base, "_is_cpu", True)
    monkeypatch.setattr(rope_base, "publish_role", lambda: None)
    with (
        get_parallel().override(tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0),
        get_context().override_server_args(),
    ):
        yield


def _tiny_vl_config(**vision_overrides):
    return HYV4VLConfig(
        text_config=VL_TEXT_OVERRIDES,
        vision_config=VISION_OVERRIDES | vision_overrides,
    )


def _fill_deterministic(module, seed=0, std=0.02):
    torch.manual_seed(seed)
    for param in module.parameters():
        torch.nn.init.normal_(param, std=std)


def test_vl_config_defaults():
    config = HYV4VLConfig()
    assert config.vision_config.out_hidden_size == config.text_config.hidden_size
    assert config.architectures == ["HYV4VLForConditionalGeneration"]
    assert (config.image_start_token_id, config.image_end_token_id) == (120118, 120119)
    assert (config.image_token_id, config.image_newline_token_id) == (120120, 120121)
    assert (config.video_start_token_id, config.video_end_token_id) == (120122, 120123)
    assert config.video_token_id == 120124
    rope = config.text_config.rope_parameters
    assert rope["rope_type"] == "default"
    assert rope["mrope_interleaved"] is True
    assert rope["mrope_section"] == [12, 10, 10]


def test_vl_config_nested_text_wins_and_flat_fields_fill_gaps():
    config = HYV4VLConfig(
        text_config={
            "hidden_size": 128,
            "rope_parameters": {"rope_theta": 7000.0, "mrope_section": [4, 2, 2]},
        },
        hidden_size=64,
        num_hidden_layers=2,
        rope_scaling={"rope_theta": 5000.0},
    )
    assert config.text_config.hidden_size == 128
    assert config.text_config.num_hidden_layers == 2
    assert config.vision_config.out_hidden_size == 128
    rope = config.text_config.rope_parameters
    assert rope["rope_theta"] == 7000.0
    assert rope["mrope_section"] == [4, 2, 2]
    assert rope["rope_type"] == "default"
    assert rope["mrope_interleaved"] is True


@pytest.mark.parametrize("architectures", [["LlamaForCausalLM"], []])
def test_vl_config_rejects_non_hyv4_text_architecture(architectures):
    """A nested text architecture other than HYV4ForCausalLM turns DSA off, so the
    text model would build no indexers and silently drop their weights."""
    with pytest.raises(ValueError, match="text_config.architectures"):
        HYV4VLConfig(
            text_config=TEXT_OVERRIDES | {"architectures": architectures},
            vision_config=VISION_OVERRIDES,
        )
    with pytest.raises(ValueError, match="text_config.architectures"):
        HYV4VLConfig(
            text_config=HYV4Config(**TEXT_OVERRIDES, architectures=architectures),
            vision_config=VISION_OVERRIDES,
        )


def test_vl_config_accepts_the_vl_architecture_as_text_architecture():
    """A checkpoint may name the VL wrapper in text_config.architectures; the text
    model still resolves to HYV4ForCausalLM, which builds the DSA indexers."""
    config = HYV4VLConfig(
        text_config=TEXT_OVERRIDES
        | {"architectures": ["HYV4VLForConditionalGeneration"]},
        vision_config=VISION_OVERRIDES,
    )
    assert config.text_config.architectures == ["HYV4ForCausalLM"]
    assert is_deepseek_dsa(config.text_config)


def test_vl_config_text_architecture_survives_serialization():
    """Serialization keeps the nested text model on DSA."""
    config = HYV4VLConfig(text_config=TEXT_OVERRIDES, vision_config=VISION_OVERRIDES)
    restored = HYV4VLConfig.from_dict(json.loads(config.to_json_string()))
    assert restored.text_config.architectures == ["HYV4ForCausalLM"]
    assert is_deepseek_dsa(restored.text_config)


def test_vl_config_with_deepseek_sparse_attention_layers_saves_and_reloads(tmp_path):
    """Checkpoints spell DSA layers deepseek_sparse_attention; save_pretrained must
    pass Transformers' layer_types validation and reload on DSA."""
    layer_types = ["deepseek_sparse_attention"] * TEXT_OVERRIDES["num_hidden_layers"]
    config = HYV4VLConfig(
        text_config=TEXT_OVERRIDES | {"layer_types": layer_types},
        vision_config=VISION_OVERRIDES,
    )
    config.save_pretrained(tmp_path)
    restored = HYV4VLConfig.from_pretrained(tmp_path)
    assert restored.text_config.layer_types == config.text_config.layer_types
    assert is_deepseek_dsa(restored.text_config)


def test_vendor_soft_logits_capping_feeds_the_logits_softcap(single_rank):
    """The vendor soft_logits_capping gate and value reach the logits softcap,
    also after VL serialization; a gate-off value stays off, a conflict fails."""
    vendor = {"soft_logits_capping": True, "soft_logits_capping_logits": 30.0}
    config = HYV4VLConfig(
        text_config=VL_TEXT_OVERRIDES, vision_config=VISION_OVERRIDES, **vendor
    )
    restored = HYV4VLConfig.from_dict(json.loads(config.to_json_string()))
    for vl_config in (config, restored):
        model = hunyuan_v4_vl.HYV4VLForConditionalGeneration(vl_config)
        assert model.language_model.logits_processor.final_logit_softcapping == 30.0

    gate_off = vendor | {"soft_logits_capping": False}
    assert HYV4Config(**TEXT_OVERRIDES, **gate_off).final_logit_softcapping is None
    with pytest.raises(ValueError, match="conflicts with soft_logits_capping_logits"):
        HYV4Config(**TEXT_OVERRIDES, **vendor, final_logit_softcapping=20.0)
    with pytest.raises(ValueError, match="positive soft_logits_capping_logits"):
        HYV4Config(**TEXT_OVERRIDES, soft_logits_capping=True)


def test_vl_config_rejects_unsupported_fields():
    with pytest.raises(ValueError, match="projector_type"):
        HYV4VLVisionConfig(projector_type="mlp")
    with pytest.raises(ValueError, match="2D RoPE"):
        HYV4VLVisionConfig(use_2d_rope=True, rope_2d_percent=0.5)
    with pytest.raises(ValueError, match="biased patch embedding"):
        HYV4VLVisionConfig(add_patchemb_bias=False)
    for name in ("video_spatial_compression", "video_temporal_compression"):
        with pytest.raises(ValueError, match="video token compression"):
            HYV4VLVisionConfig(**{name: True})
    with pytest.raises(ValueError, match="hidden_size"):
        HYV4VLConfig(
            text_config=TEXT_OVERRIDES,
            vision_config=VISION_OVERRIDES | {"out_hidden_size": 32},
        )
    with pytest.raises(ValueError, match="tie_word_embeddings=True"):
        HYV4VLConfig(
            text_config=TEXT_OVERRIDES,
            vision_config=VISION_OVERRIDES,
            tie_word_embeddings=True,
        )


def test_hf_to_sglang_mapper_covers_checkpoint_prefixes():
    mapper = hunyuan_v4_vl.HYV4VLForConditionalGeneration.hf_to_sglang_mapper
    cases = {
        "vit.embeddings.patch_embedding.weight": "visual.embeddings.patch_embedding.weight",
        "vit.embeddings.position_embedding.weight": "visual.embeddings.position_embedding.weight",
        "vit.layers.0.self_attn.q_proj.weight": "visual.layers.0.self_attn.q_proj.weight",
        "vit.perceive.proj.0.weight": "visual.perceive.proj.0.weight",
        "vit.vit.layers.0.mlp.dense_h_to_4h.weight": "visual.layers.0.mlp.dense_h_to_4h.weight",
        "model.embed_tokens.weight": "language_model.model.embed_tokens.weight",
        "model.layers.1.mlp.gate.wg.weight": "language_model.model.layers.1.mlp.gate.weight",
        "model.layers.1.mlp.gate.e_score_correction_bias": "language_model.model.layers.1.mlp.gate.e_score_correction_bias",
        "model.mtp_layers.0.eh_proj.weight": "language_model.model.mtp_layers.0.eh_proj.weight",
        "lm_head.weight": "language_model.lm_head.weight",
    }
    mapped = mapper.apply_dict({name: None for name in cases})
    assert set(mapped) == set(cases.values())


@pytest.mark.parametrize(
    "projector_type, use_2d_rope", [(None, False), ("rmsnorm_mlp_adapter", True)]
)
def test_vision_batch_preserves_media_boundaries(
    single_rank, projector_type, use_2d_rope
):
    config = _tiny_vl_config(
        projector_type=projector_type, use_2d_rope=use_2d_rope
    ).vision_config
    tower = hunyuan_v4_vl.HYV4VisionTransformer(config)
    # At std 0.02 attention is near-uniform and cross-item leakage stays below tolerance.
    _fill_deterministic(tower, std=0.2)
    grids = [[1, 4, 4], [3, 4, 4]]
    torch.manual_seed(1)
    pixels = torch.randn(4 * 16, 3 * 16 * 16)
    with torch.no_grad():
        together = tower(pixels, grids)
        separate = [
            tower(pixels[:16], grids[:1])[0],
            tower(pixels[16:], grids[1:])[0],
        ]
    # merge=2 plus a newline column: [1,4,4] -> 2*3 tokens, [3,4,4] -> 3*2*3.
    assert together[0].shape == (6, config.out_hidden_size)
    assert together[1].shape == (18, config.out_hidden_size)
    for actual, expected in zip(together, separate):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_vision_newline_and_extra_tokens_change_token_count(single_rank):
    grids = [[1, 4, 4]]
    pixels = torch.randn(16, 3 * 16 * 16)
    for overrides, expected in [
        (dict(use_image_newline_token=False), 4),
        (dict(cat_extra_token=1), 8),
    ]:
        config = _tiny_vl_config(**overrides).vision_config
        tower = hunyuan_v4_vl.HYV4VisionTransformer(config)
        _fill_deterministic(tower)
        with torch.no_grad():
            (out,) = tower(pixels, grids)
        assert out.shape[0] == expected


def test_vision_rejects_pixel_grid_mismatch(single_rank):
    config = _tiny_vl_config().vision_config
    tower = hunyuan_v4_vl.HYV4VisionTransformer(config)
    with pytest.raises(ValueError, match="grid_thw"):
        tower(torch.randn(15, 3 * 16 * 16), [[1, 4, 4]])


def _reference_vision_checkpoint(num_layers):
    # Written from the checkpoint layout, not from this port's parameters,
    # for VISION_OVERRIDES with a 64-wide text model and the default projector.
    shapes = {
        "embeddings.patch_embedding.weight": (64, 3, 16, 16),
        "embeddings.patch_embedding.bias": (64,),
        "embeddings.position_embedding.weight": (17, 64),
        "perceive.proj.0.weight": (128, 64, 2, 2),
        "perceive.proj.0.bias": (128,),
        "perceive.proj.2.weight": (256, 128, 1, 1),
        "perceive.proj.2.bias": (256,),
        "perceive.mlp.weight": (64, 256),
        "perceive.mlp.bias": (64,),
        "perceive.image_sep": (64,),
        "perceive.image_newline": (256,),
        "perceive.before_rms.weight": (64,),
        "perceive.after_rms.weight": (64,),
    }
    for i in range(num_layers):
        layer = {
            "input_layernorm.weight": (64,),
            "input_layernorm.bias": (64,),
            "post_attention_layernorm.weight": (64,),
            "post_attention_layernorm.bias": (64,),
            "self_attn.o_proj.weight": (64, 64),
            "self_attn.o_proj.bias": (64,),
            "mlp.dense_h_to_4h.weight": (128, 64),
            "mlp.dense_h_to_4h.bias": (128,),
            "mlp.dense_4h_to_h.weight": (64, 128),
            "mlp.dense_4h_to_h.bias": (64,),
        }
        for source in ("q_proj", "k_proj", "v_proj"):
            layer[f"self_attn.{source}.weight"] = (64, 64)
            layer[f"self_attn.{source}.bias"] = (64,)
        shapes.update({f"layers.{i}.{name}": shape for name, shape in layer.items()})
    return shapes


def test_vision_weights_pack_qkv_and_cover_every_param(single_rank):
    """Every checkpoint tensor lands in a tower parameter, Q/K/V in shard order."""
    tower = hunyuan_v4_vl.HYV4VisionTransformer(
        _tiny_vl_config(num_hidden_layers=2).vision_config
    )
    torch.manual_seed(0)
    hf_weights = {}
    for name, shape in _reference_vision_checkpoint(num_layers=2).items():
        source = name.rsplit(".", 2)[-2]
        if source in ("q_proj", "k_proj", "v_proj"):
            fill = ("q_proj", "k_proj", "v_proj").index(source) + 1
            hf_weights[name] = torch.full(shape, float(fill))
        else:
            hf_weights[name] = torch.randn(shape)

    loaded = tower.load_weights(iter(hf_weights.items()))

    assert loaded == set(dict(tower.named_parameters()))
    for layer in tower.layers:
        qkv = layer.self_attn.qkv_proj
        width = qkv.weight.shape[0] // 3
        for tensor in (qkv.weight, qkv.bias):
            for shard_id in range(3):
                shard = tensor[shard_id * width : (shard_id + 1) * width]
                assert torch.all(shard == float(shard_id + 1))


class _RecordingLogitsProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, input_ids, hidden_states, lm_head, forward_batch, *_):
        self.calls.append((input_ids, hidden_states))
        return "logits"


def _stub_text_body(monkeypatch, model):
    """Replace the transformer body and logits head; keep the composition between."""
    body_calls = []

    def body(
        input_ids, positions, forward_batch, input_embeds=None, pp_proxy_tensors=None
    ):
        body_calls.append(input_embeds)
        return torch.full((positions.shape[-1], 64), 3.0)

    monkeypatch.setattr(model.language_model.model, "forward", body)
    logits = _RecordingLogitsProcessor()
    monkeypatch.setattr(model.language_model, "logits_processor", logits)
    return body_calls, logits.calls


def _decode_batch():
    return SimpleNamespace(forward_mode=ForwardMode.DECODE, input_embeds=None)


def test_vl_forward_applies_logits_processor_once(single_rank, monkeypatch):
    """The VL forward must hand the logits head hidden states, never logits."""
    config = HYV4VLConfig(
        text_config=HYV4Config(**TEXT_OVERRIDES), vision_config=VISION_OVERRIDES
    )
    model = hunyuan_v4_vl.HYV4VLForConditionalGeneration(config)
    _fill_deterministic(model)
    body_calls, logits_calls = _stub_text_body(monkeypatch, model)
    input_ids = torch.tensor([1, 2, 3])

    output = model(input_ids, torch.arange(3), _decode_batch())

    assert output == "logits"
    torch.testing.assert_close(
        body_calls[0], model.language_model.model.embed_tokens(input_ids)
    )
    ((logits_input_ids, hidden_states),) = logits_calls
    assert logits_input_ids is input_ids
    assert torch.all(hidden_states == 3.0)


def _mrope_batch(mrope_positions, contains_mm_inputs, mode=ForwardMode.EXTEND):
    return SimpleNamespace(
        forward_mode=mode,
        contains_mm_inputs=lambda: contains_mm_inputs,
        mrope_positions=mrope_positions,
        input_embeds=None,
    )


def test_vl_forward_hands_mrope_positions_to_text_model(single_rank, monkeypatch):
    """Extend passes the batch's [3, N] mRoPE positions, decode and verify pass
    row 0 as 1-D positions; an image batch without mRoPE positions fails loudly."""
    model = hunyuan_v4_vl.HYV4VLForConditionalGeneration(_tiny_vl_config())
    seen_positions = []

    def body(
        input_ids, positions, forward_batch, input_embeds=None, pp_proxy_tensors=None
    ):
        seen_positions.append(positions)
        return torch.zeros(positions.shape[-1], 64)

    monkeypatch.setattr(model.language_model.model, "forward", body)
    monkeypatch.setattr(
        model.language_model, "logits_processor", _RecordingLogitsProcessor()
    )
    input_ids = torch.tensor([1, 2, 3])
    mrope_positions = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 1]])
    decode_positions = torch.tensor([7, 12, 300])

    model(input_ids, torch.arange(3), _mrope_batch(mrope_positions, False))
    for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
        model(
            input_ids,
            torch.arange(3),
            _mrope_batch(decode_positions.repeat(3, 1), True, mode),
        )

    extend_positions, *equal_row_positions = seen_positions
    assert torch.equal(extend_positions, mrope_positions)
    for row in equal_row_positions:
        assert row.ndim == 1 and torch.equal(row, decode_positions)
    with pytest.raises(ValueError, match=r"\(3, num_tokens\)"):
        model(input_ids, torch.arange(3), _mrope_batch(None, True))


def _interleaved_mrope_reference(cos_sin_cache, positions, x, mrope_section):
    """Reference semantics: rotary lane i takes the T, H or W position by
    i % 3 within each section, then rotates (even, odd) element pairs."""
    cos, sin = cos_sin_cache[positions].chunk(2, dim=-1)
    lane = torch.arange(cos.shape[-1])
    axis = torch.zeros_like(lane)
    axis[(lane % 3 == 1) & (lane < 3 * mrope_section[1])] = 1
    axis[(lane % 3 == 2) & (lane < 3 * mrope_section[2])] = 2
    cos = cos.gather(0, axis.expand_as(cos[:1])).squeeze(0).unsqueeze(-2)
    sin = sin.gather(0, axis.expand_as(sin[:1])).squeeze(0).unsqueeze(-2)
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack(
        (even * cos - odd * sin, odd * cos + even * sin), dim=-1
    ).flatten(-2)


class _ForbiddenRotary(nn.Module):
    def forward(self, positions, query, key):
        raise AssertionError("1-D positions reached the mRoPE rotary")


def test_vl_attention_and_indexer_rotate_mrope_sections(single_rank, monkeypatch):
    """Both the MLA rotary and the DSA indexer rotary apply interleaved T/H/W
    sections to [3, N] positions, while 1-D positions keep the text rotary."""
    model = hunyuan_v4_vl.HYV4VLForConditionalGeneration(_tiny_vl_config())
    attention = model.language_model.model.layers[0].self_attn
    rope_dim, nope_dim = (
        TEXT_OVERRIDES["qk_rope_head_dim"],
        TEXT_OVERRIDES["qk_nope_head_dim"],
    )
    index_head_dim = VL_TEXT_OVERRIDES["index_head_dim"]
    torch.manual_seed(0)
    positions = torch.tensor([[0, 1, 1, 1, 4], [0, 1, 1, 2, 4], [0, 1, 2, 1, 4]])
    num_tokens = positions.shape[1]
    # Callers rotate strided views into the projected q, the latent cache, and
    # the indexer q/k; every call slices fresh views out of copied backings.
    q = torch.randn(num_tokens, 4, nope_dim + rope_dim)
    latent = torch.randn(num_tokens, TEXT_OVERRIDES["kv_lora_rank"] + rope_dim)
    indexer_q = torch.randn(num_tokens, TEXT_OVERRIDES["index_n_heads"], index_head_dim)
    indexer_k = torch.randn(num_tokens, index_head_dim)
    dtypes = (torch.float32, torch.bfloat16)
    cases = (
        (
            attention.rotary_emb,
            lambda dtype: (
                q.to(dtype, copy=True)[..., nope_dim:],
                latent.to(dtype, copy=True)[..., -rope_dim:].unsqueeze(1),
            ),
        ),
        (
            attention.indexer.rotary_emb,
            lambda dtype: (
                indexer_q.to(dtype, copy=True)[..., :rope_dim],
                indexer_k.to(dtype, copy=True)[..., :rope_dim],
            ),
        ),
    )

    for rotary, views in cases:
        query, key = views(torch.float32)
        assert not query.is_contiguous() and not key.is_contiguous()
        cos_sin_cache = rotary.mrope.cos_sin_cache
        expected_q, expected_k = (
            _interleaved_mrope_reference(
                cos_sin_cache, positions, x.view(num_tokens, -1, rope_dim), [2, 1, 1]
            ).view_as(x)
            for x in (query, key)
        )
        actual_q, actual_k = rotary(positions, *views(torch.float32))
        torch.testing.assert_close(actual_q, expected_q)
        torch.testing.assert_close(actual_k, expected_k)

        # Decode feeds the equal rows as 1-D positions, so this must hold bitwise
        # in the serving dtype too, not only within float tolerance.
        text_positions = positions[0]
        text = {dtype: rotary.rope(text_positions, *views(dtype)) for dtype in dtypes}
        for dtype in dtypes:
            same_rows = rotary(text_positions.repeat(3, 1), *views(dtype))
            assert all(map(torch.equal, same_rows, text[dtype])), dtype

        monkeypatch.setattr(rotary, "mrope", _ForbiddenRotary())
        for dtype in dtypes:
            one_d = rotary(text_positions, *views(dtype))
            assert all(map(torch.equal, one_d, text[dtype])), dtype


class _TupleLinear(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x):
        return nn.functional.linear(x, self.weight), None


def _install_fused_layout(indexer, monkeypatch):
    """Fold wk and weights_proj into wk_weights_proj, as the CUDA fusion layout
    loads them, and forbid Hadamard, which that layout never applies."""
    wk, weights_proj = indexer.wk, indexer.weights_proj
    indexer.wk_weights_proj = _TupleLinear(
        torch.cat([wk.weight, weights_proj.weight.to(wk.weight.dtype)])
    )
    del indexer.wk
    del indexer.weights_proj
    monkeypatch.setattr(indexer, "use_dsa_indexer_fusion", True)

    def forbid_hadamard(x):
        raise AssertionError("the fused layout stores index Q/K without Hadamard")

    monkeypatch.setattr(dsa_indexer, "rotate_activation", forbid_hadamard)
    return wk, weights_proj


@torch.no_grad()
def test_fused_layout_indexer_rotates_mrope_outside_fused_kernels(
    single_rank, monkeypatch
):
    """Under the CUDA fusion weight layout (wk folded into wk_weights_proj),
    [3, N] positions take the unfused rotary in no-Hadamard space, while a 1-D
    K-only batch keeps the fused writer."""
    model = hunyuan_v4_vl.HYV4VLForConditionalGeneration(_tiny_vl_config())
    _fill_deterministic(model)
    indexer = model.language_model.model.layers[0].self_attn.indexer
    wk, weights_proj = _install_fused_layout(indexer, monkeypatch)
    stored_keys, fused_writes = [], []
    monkeypatch.setattr(
        indexer,
        "_store_index_k_cache",
        lambda **kwargs: stored_keys.append(kwargs["key"]),
    )
    monkeypatch.setattr(
        indexer,
        "_fused_k_prepare_and_store",
        lambda key, positions, *args, **kwargs: fused_writes.append(positions),
    )
    torch.manual_seed(1)
    rope_dim, num_tokens = TEXT_OVERRIDES["qk_rope_head_dim"], 5
    positions = torch.tensor([[0, 1, 1, 1, 4], [0, 1, 1, 2, 4], [0, 1, 2, 1, 4]])
    x = torch.randn(num_tokens, TEXT_OVERRIDES["hidden_size"])
    q_lora = torch.randn(num_tokens, TEXT_OVERRIDES["q_lora_rank"])
    cos_sin_cache = indexer.rotary_emb.mrope.cos_sin_cache

    def rotate_leading_rope(x):
        rope = _interleaved_mrope_reference(
            cos_sin_cache,
            positions,
            x[..., :rope_dim].reshape(num_tokens, -1, rope_dim),
            [2, 1, 1],
        )
        return torch.cat([rope.view_as(x[..., :rope_dim]), x[..., rope_dim:]], dim=-1)

    expected_q = rotate_leading_rope(
        indexer.wq_b(q_lora)[0].view(num_tokens, indexer.n_heads, indexer.head_dim)
    )
    expected_k = rotate_leading_rope(indexer.k_norm(wk(x)[0]))

    query, key, weights_raw = indexer._get_q_k_bf16(
        q_lora,
        x,
        positions,
        False,
        forward_batch=SimpleNamespace(attn_cp_metadata=None),
    )
    torch.testing.assert_close(query, expected_q)
    torch.testing.assert_close(key, expected_k)
    torch.testing.assert_close(
        weights_raw, nn.functional.linear(x, weights_proj.weight.to(x.dtype))
    )

    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_extend_without_speculative=lambda: True,
            is_decode_or_idle=lambda: False,
        ),
        out_cache_loc=torch.arange(num_tokens),
    )
    for k_only_positions in (positions, positions[0]):
        indexer._forward_cuda_k_only(
            x, k_only_positions, batch, 0, None, None, return_indices=False
        )
    (stored_key,) = stored_keys
    torch.testing.assert_close(stored_key, expected_k)
    (fused_positions,) = fused_writes
    assert fused_positions.ndim == 1


def _fake_act_quant(x, block_size, scale_fmt):
    return x, x.abs().amax(dim=-1, keepdim=True) + 1.0


def _run_prefill_split_op(monkeypatch, indexer, positions, x, q_lora):
    num_tokens = x.shape[0]
    forward_batch = SimpleNamespace(
        extend_num_tokens=num_tokens,
        out_cache_loc=torch.arange(num_tokens),
        attn_cp_metadata=None,
        forward_mode=SimpleNamespace(is_context_parallel_extend=lambda: False),
    )
    monkeypatch.setattr(dsa_prefill_cuda_graph, "_is_cuda", True)
    monkeypatch.setattr(
        dsa_prefill_cuda_graph,
        "get_tc_piecewise_forward_context",
        lambda: SimpleNamespace(forward_batch=forward_batch, dsa_indexers=[indexer]),
    )
    monkeypatch.setattr(
        dsa_prefill_cuda_graph,
        "get_attn_backend",
        lambda: SimpleNamespace(get_indexer_metadata=lambda layer_id, batch: None),
    )
    monkeypatch.setattr(dsa_act_quant, "act_quant", _fake_act_quant)
    monkeypatch.setattr(torch._dynamo.config, "disable", True)
    monkeypatch.setattr(indexer, "_should_skip_logits_computation", lambda batch: False)
    seen = SimpleNamespace(stored_keys=[], topk_inputs=[], fused_positions=[])
    monkeypatch.setattr(
        indexer,
        "_store_index_k_cache",
        lambda **kwargs: seen.stored_keys.append(kwargs["key"]),
    )
    monkeypatch.setattr(
        indexer,
        "_get_topk_ragged",
        lambda _cp, _batch, _layer, q_fp8, weights, *rest: seen.topk_inputs.append(
            (q_fp8, weights)
        ),
    )

    def fused_q_prepare_and_store(x, q_lora, positions, *args, **kwargs):
        seen.fused_positions.append(positions)
        return "fused_q_fp8", "fused_weights"

    monkeypatch.setattr(
        indexer, "_fused_q_prepare_and_store", fused_q_prepare_and_store
    )
    # The op registers only a CUDA kernel; dispatch to it directly with CPU tensors.
    torch.ops.sglang.pcg_dsa_indexer_prefill_split.default.redispatch(
        torch._C.DispatchKeySet("CUDA"),
        layer_id=0,
        x=x,
        q_lora=q_lora,
        positions=positions,
        topk_result=torch.full((num_tokens, indexer.index_topk), -1, dtype=torch.int32),
    )
    return seen


@torch.no_grad()
@pytest.mark.parametrize("layout", ["fused_mrope", "fused_1d", "unfused_1d"])
def test_prefill_graph_indexer_split_op_routes_by_positions_and_layout(
    single_rank, monkeypatch, layout
):
    """Under a prefill CUDA graph, [3, N] positions must leave the 1-D fused
    rope kernels and gate heads from the fused projection's weights_raw, while
    1-D batches keep their fused or unfused path unchanged."""
    model = hunyuan_v4_vl.HYV4VLForConditionalGeneration(_tiny_vl_config())
    _fill_deterministic(model)
    indexer = model.language_model.model.layers[0].self_attn.indexer
    weights_proj = indexer.weights_proj
    if layout != "unfused_1d":
        _install_fused_layout(indexer, monkeypatch)
    else:
        monkeypatch.setattr(dsa_indexer, "rotate_activation", lambda x: x)
        # The CPU fallback of the bf16-in/fp32-out gate GEMM needs matching dtypes.
        weights_proj.weight.data = weights_proj.weight.data.float()
    torch.manual_seed(2)
    num_tokens = 5
    positions = torch.tensor([[0, 1, 1, 1, 4], [0, 1, 1, 2, 4], [0, 1, 2, 1, 4]])
    if layout != "fused_mrope":
        positions = positions[0]
    x = torch.randn(num_tokens, TEXT_OVERRIDES["hidden_size"])
    q_lora = torch.randn(num_tokens, TEXT_OVERRIDES["q_lora_rank"])

    seen = _run_prefill_split_op(monkeypatch, indexer, positions, x, q_lora)

    if layout == "fused_1d":
        assert seen.fused_positions == [positions] and seen.stored_keys == []
        assert seen.topk_inputs == [("fused_q_fp8", "fused_weights")]
        return
    assert seen.fused_positions == []
    query, key, _ = indexer._get_q_k_bf16(
        q_lora,
        x,
        positions,
        False,
        forward_batch=SimpleNamespace(attn_cp_metadata=None),
    )
    (stored_key,) = seen.stored_keys
    torch.testing.assert_close(stored_key, key)
    _, q_scale = _fake_act_quant(query, None, None)
    gate = nn.functional.linear(x, weights_proj.weight.to(x.dtype)).float()
    expected_weights = (
        (gate * indexer.n_heads**-0.5).unsqueeze(-1) * q_scale * indexer.softmax_scale
    )
    ((q_fp8, weights),) = seen.topk_inputs
    torch.testing.assert_close(q_fp8, query)
    torch.testing.assert_close(weights, expected_weights)


@pytest.mark.parametrize("flag_source", ["config", "cli"])
def test_trtllm_fused_rope_follows_the_model_rotary_mode(
    single_rank, monkeypatch, flag_source
):
    """Language-only VL serving keeps the fused trtllm rope even when a decode
    graph batch carries its always-allocated mRoPE buffer; an mRoPE model
    never fuses."""
    monkeypatch.setattr(
        forward_mla.DeepseekMLAForwardMixin,
        "_fuse_rope_for_trtllm_mla",
        lambda self, forward_batch: True,
    )
    if flag_source == "config":
        config = HYV4VLConfig(
            text_config=VL_TEXT_OVERRIDES,
            vision_config=VISION_OVERRIDES,
            language_model_only=True,
        )
    else:
        # ModelConfig installs --language-model-only on the outer config only.
        config = _tiny_vl_config()
        config.language_model_only = True
        assert not getattr(config.text_config, "language_model_only", False)
    language_only = hunyuan_v4_vl.HYV4VLForConditionalGeneration(config)
    multimodal = hunyuan_v4_vl.HYV4VLForConditionalGeneration(_tiny_vl_config())
    eager_batch = SimpleNamespace(mrope_positions=None)
    graph_batch = SimpleNamespace(mrope_positions=torch.zeros(3, 2, dtype=torch.long))

    for layer in language_only.language_model.model.layers:
        assert layer.self_attn._fuse_rope_for_trtllm_mla(eager_batch)
        assert layer.self_attn._fuse_rope_for_trtllm_mla(graph_batch)
    for layer in multimodal.language_model.model.layers:
        assert not layer.self_attn._fuse_rope_for_trtllm_mla(graph_batch)


def test_text_config_without_mrope_keeps_text_rotary(single_rank):
    """A text config without mRoPE metadata leaves the HYV4 rotary untouched."""
    config = HYV4VLConfig(
        text_config=HYV4Config(**TEXT_OVERRIDES), vision_config=VISION_OVERRIDES
    )
    model = hunyuan_v4_vl.HYV4VLForConditionalGeneration(config)
    attention = model.language_model.model.layers[0].self_attn
    assert not model.is_mrope_enabled
    for rotary in (attention.rotary_emb, attention.indexer.rotary_emb):
        assert isinstance(rotary, RotaryEmbedding)


def test_vl_model_language_model_only(single_rank, monkeypatch):
    """A language-only flag declared in the config skips the vision tower and
    sends text straight to the causal LM."""
    config = HYV4VLConfig(
        text_config=VL_TEXT_OVERRIDES,
        vision_config=VISION_OVERRIDES,
        language_model_only=True,
    )
    model = hunyuan_v4_vl.HYV4VLForConditionalGeneration(config)
    assert model.visual is None
    with pytest.raises(RuntimeError, match="language_model_only"):
        model.get_image_feature([])

    body_calls, logits_calls = _stub_text_body(monkeypatch, model)
    model(torch.tensor([1, 2]), torch.arange(2), _decode_batch())
    assert body_calls == [None]
    assert len(logits_calls) == 1


def test_vl_config_json_resolves_hyv4_serving(tmp_path):
    """A config.json-loaded VL model resolves MLA/DSA, language-only, and
    quantization the way the text-only HYV4 model does."""
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["HYV4VLForConditionalGeneration"],
                "model_type": "hy_v4_vl",
                "text_config": TEXT_OVERRIDES,
                "vision_config": VISION_OVERRIDES,
                "language_model_only": True,
                "quantization_config": {
                    "quant_method": "fp8",
                    "weight_block_size": [128, 128],
                },
            }
        )
    )
    model_config = ModelConfig(str(tmp_path))
    assert model_config.is_lm_only
    assert model_config.quantization == "fp8"
    assert model_config.attention_arch == AttentionArch.MLA
    assert model_config.index_head_dim == TEXT_OVERRIDES["index_head_dim"]
    assert is_deepseek_dsa(model_config.hf_config)


def test_language_model_only_flag_accepts_the_vl_architecture(tmp_path):
    """Text-only serving of the VL checkpoint goes through --language-model-only."""
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["HYV4VLForConditionalGeneration"],
                "model_type": "hy_v4_vl",
                "text_config": VL_TEXT_OVERRIDES,
                "vision_config": VISION_OVERRIDES,
            }
        )
    )
    server_args = ServerArgs(
        model_path=str(tmp_path), language_model_only=True, device="cpu"
    )
    handle_language_model_only(server_args)


@pytest.mark.parametrize(
    "source,refused", [("flag", False), ("checkpoint", False), ("none", True)]
)
def test_pp_mtp_prefill_graph_refusal_follows_resolved_language_only(
    tmp_path, source, refused
):
    """PP + MTP with a prefill CUDA graph is refused only for media-serving VL:
    text-only serving, from the flag or the checkpoint, has no media hand-off."""
    config = {
        "architectures": ["HYV4VLForConditionalGeneration"],
        "model_type": "hy_v4_vl",
        "text_config": VL_TEXT_OVERRIDES,
        "vision_config": VISION_OVERRIDES,
    }
    if source == "checkpoint":
        config["language_model_only"] = True
    (tmp_path / "config.json").write_text(json.dumps(config))
    model_config = ModelConfig(str(tmp_path), language_model_only=source == "flag")
    cfg = SimpleNamespace(
        enable_prefill_cp=False,
        dcp_size=1,
        pp_size=2,
        speculative_algorithm="EAGLE",
        disaggregation_mode="null",
        cuda_graph_config=SimpleNamespace(
            prefill=SimpleNamespace(backend=Backend.BREAKABLE)
        ),
        _model_config=model_config,
    )
    with envs.SGLANG_ENABLE_PP_SPEC.override(True):
        if refused:
            with pytest.raises(ValueError, match="HYV4 VL multimodal serving"):
                _reject_unsupported_hyv4_parallelism(
                    cfg, "HYV4VLForConditionalGeneration"
                )
        else:
            _reject_unsupported_hyv4_parallelism(cfg, "HYV4VLForConditionalGeneration")


def test_vl_model_config_keeps_breakable_prefill_cuda_graph(tmp_path):
    """A multimodal-serving VL config stays on the breakable prefill CUDA graph
    that the generic multimodal rule would otherwise disable."""
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["HYV4VLForConditionalGeneration"],
                "model_type": "hy_v4_vl",
                "text_config": VL_TEXT_OVERRIDES,
                "vision_config": VISION_OVERRIDES,
            }
        )
    )
    model_config = ModelConfig(str(tmp_path))
    assert model_config.is_multimodal
    assert model_config.is_multimodal_breakable_cuda_graph_supported
    assert not is_multimodal_breakable_cuda_graph_supported(
        ["UnknownVisionForConditionalGeneration"]
    )


def test_vl_draft_model_config_resolves_hyv4_nextn(tmp_path):
    """A VL checkpoint loaded as the speculative draft resolves to the HYV4
    MTP architecture with one predict layer."""
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["HYV4VLForConditionalGeneration"],
                "model_type": "hy_v4_vl",
                "text_config": VL_TEXT_OVERRIDES,
                "vision_config": VISION_OVERRIDES,
            }
        )
    )
    model_config = ModelConfig(str(tmp_path), is_draft_model=True)
    assert model_config.hf_config.architectures == ["HYV4ForCausalLMNextN"]
    assert model_config.hf_config.num_nextn_predict_layers == 1


def _tiny_nextn_draft(monkeypatch, config):
    with get_context().override_server_args(disable_shared_experts_fusion=False):
        shared_experts = resolve_shared_experts_config(
            HYV4ForCausalLMNextN, config, None
        )
        with get_model().override(shared_experts=shared_experts):
            return HYV4ForCausalLMNextN(config)


DRAFT_MOE_OVERRIDES = dict(
    moe_intermediate_size=32, n_routed_experts=4, num_experts_per_tok=2
)


def test_nextn_draft_rejects_an_mtp_quant_override(single_rank, monkeypatch):
    """An absent or NONE mtp_quant_algo inherits the target quantization, as
    does BF16 on an unquantized target; any other override must fail."""

    def draft_config(mtp_quant_algo):
        return HYV4Config(
            **TEXT_OVERRIDES,
            **DRAFT_MOE_OVERRIDES,
            architectures=["HYV4ForCausalLMNextN"],
            mtp_quant_algo=mtp_quant_algo,
        )

    fp8 = Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
    for mtp_quant_algo, quant_config in (("FP8", None), ("FP8", fp8), ("bf16", fp8)):
        with pytest.raises(ValueError, match=f"mtp_quant_algo='{mtp_quant_algo}'"):
            HYV4ForCausalLMNextN(draft_config(mtp_quant_algo), quant_config)
    for mtp_quant_algo in ("none", "BF16"):
        _tiny_nextn_draft(monkeypatch, draft_config(mtp_quant_algo))


def test_vl_nextn_draft_rotates_like_the_target(single_rank, monkeypatch):
    """A draft built from the VL config rotates [3, N] positions exactly like
    the target's MLA and indexer rotaries; a text config keeps the text rotary."""
    draft = _tiny_nextn_draft(
        monkeypatch,
        HYV4VLConfig(
            text_config=VL_TEXT_OVERRIDES | DRAFT_MOE_OVERRIDES,
            vision_config=VISION_OVERRIDES,
        ),
    )
    target = hunyuan_v4_vl.HYV4VLForConditionalGeneration(_tiny_vl_config())
    draft_attention = draft.model.decoder.self_attn
    target_attention = target.language_model.model.layers[0].self_attn
    rope_dim = TEXT_OVERRIDES["qk_rope_head_dim"]
    torch.manual_seed(0)
    positions = torch.tensor([[0, 1, 1, 1, 4], [0, 1, 1, 2, 4], [0, 1, 2, 1, 4]])
    query = torch.randn(positions.shape[1], 4, rope_dim)
    key = torch.randn(positions.shape[1], 1, rope_dim)

    assert draft.is_mrope_enabled
    for draft_rotary, target_rotary in (
        (draft_attention.rotary_emb, target_attention.rotary_emb),
        (draft_attention.indexer.rotary_emb, target_attention.indexer.rotary_emb),
    ):
        assert isinstance(draft_rotary, HYV4MRotaryEmbedding)
        drafted = draft_rotary(positions, query.clone(), key.clone())
        expected = target_rotary(positions, query.clone(), key.clone())
        assert all(map(torch.equal, drafted, expected))

    text_draft = _tiny_nextn_draft(
        monkeypatch,
        HYV4Config(
            **TEXT_OVERRIDES,
            **DRAFT_MOE_OVERRIDES,
            architectures=["HYV4ForCausalLMNextN"],
        ),
    )
    text_attention = text_draft.model.decoder.self_attn
    assert not text_draft.is_mrope_enabled
    for rotary in (text_attention.rotary_emb, text_attention.indexer.rotary_emb):
        assert type(rotary) is RotaryEmbedding


def test_vl_nextn_draft_reads_mrope_positions_per_forward_mode(
    single_rank, monkeypatch
):
    """1-D positions lack the mRoPE delta: the draft reads [3, N] rows in
    extend, row 0 in equal-row modes, and keeps 1-D when no rows exist."""
    vl_draft = _tiny_nextn_draft(
        monkeypatch,
        HYV4VLConfig(
            text_config=VL_TEXT_OVERRIDES | DRAFT_MOE_OVERRIDES,
            vision_config=VISION_OVERRIDES,
        ),
    )
    text_draft = _tiny_nextn_draft(
        monkeypatch,
        HYV4Config(
            **TEXT_OVERRIDES,
            **DRAFT_MOE_OVERRIDES,
            architectures=["HYV4ForCausalLMNextN"],
        ),
    )
    seen_positions = []

    def body(input_ids, positions, forward_batch):
        seen_positions.append(positions)

    for draft in (vl_draft, text_draft):
        monkeypatch.setattr(draft.model, "forward", body)
        monkeypatch.setattr(draft, "logits_processor", _RecordingLogitsProcessor())
    input_ids = torch.tensor([1, 2, 3])
    batch_positions = torch.tensor([4, 5, 6])
    sectioned = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 1]]) + 100
    equal_rows = (batch_positions + 100).repeat(3, 1)
    cases = (
        (ForwardMode.EXTEND, sectioned, sectioned),
        (ForwardMode.DECODE, equal_rows, equal_rows[0]),
        (ForwardMode.DRAFT_EXTEND_V2, equal_rows, equal_rows[0]),
        (ForwardMode.TARGET_VERIFY, equal_rows, equal_rows[0]),
        (ForwardMode.IDLE, None, batch_positions),
        (ForwardMode.EXTEND, None, batch_positions),
    )

    for mode, mrope_positions, expected in cases:
        batch = SimpleNamespace(forward_mode=mode, mrope_positions=mrope_positions)
        vl_draft(input_ids, batch_positions, batch)
        text_draft(input_ids, batch_positions, batch)
        vl_positions, text_positions = seen_positions[-2:]
        assert torch.equal(vl_positions, expected), mode
        assert text_positions is batch_positions, mode


def test_vl_nextn_draft_embeds_target_media_features(single_rank, monkeypatch):
    """Prefill draft extend keeps the target's shifted media embeddings and
    embeds only each request's sampled tail token; other modes embed ids."""
    draft = _tiny_nextn_draft(
        monkeypatch,
        HYV4VLConfig(
            text_config=VL_TEXT_OVERRIDES | DRAFT_MOE_OVERRIDES,
            vision_config=VISION_OVERRIDES,
        ),
    )
    _fill_deterministic(draft)
    embed_tokens = draft.model.embed_tokens
    vocab_size = TEXT_OVERRIDES["vocab_size"]
    # Two requests; media rows carry hashed pad ids past the vocab.
    input_ids = torch.tensor([5, vocab_size + 7, vocab_size + 9, 7, 9, 11])
    last_indices = [3, 5]
    target_embeds = torch.randn(6, TEXT_OVERRIDES["hidden_size"])
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        contains_mm_inputs=lambda: True,
        mm_input_embeds=target_embeds.clone(),
        extend_start_loc=torch.tensor([0, 4]),
        extend_seq_lens=torch.tensor([4, 2]),
    )

    embeds = draft.model.embed_input_ids(input_ids, batch)

    expected = target_embeds.clone()
    expected[last_indices] = embed_tokens(input_ids[last_indices])
    assert torch.equal(embeds, expected)
    text_ids = torch.tensor([5, 7, 9])
    for mode in (ForwardMode.DECODE, ForwardMode.DRAFT_EXTEND_V2):
        batch = SimpleNamespace(
            forward_mode=mode, contains_mm_inputs=lambda: True, mm_input_embeds=None
        )
        assert torch.equal(
            draft.model.embed_input_ids(text_ids, batch), embed_tokens(text_ids)
        )


def test_vl_rejects_hyv4_context_parallel_outside_interleave():
    """HYV4 prefill-CP restrictions reach the VL architecture through the override path."""
    server_args = SimpleNamespace(
        enable_prefill_cp=True,
        cp_strategy="zigzag",
        dcp_size=1,
        pp_size=1,
        speculative_algorithm=None,
        disaggregation_mode="null",
    )
    with pytest.raises(ValueError, match="--enable-prefill-cp.*cp_strategy='zigzag'"):
        collect_model_override_declarations(
            "HYV4VLForConditionalGeneration", server_args, _tiny_vl_config()
        )


def _prefill_cp_rank(*, tp_size, attn_tp_size, cp_rank):
    attn_cp_size = tp_size // attn_tp_size
    return parallel_scope(
        tp_size=tp_size,
        tp_rank=cp_rank * attn_tp_size,
        attn_tp_size=attn_tp_size,
        attn_tp_rank=0,
        attn_cp_size=attn_cp_size,
        attn_cp_rank=cp_rank,
        attn_cp_group=SimpleNamespace(rank_in_group=cp_rank, world_size=attn_cp_size),
        attn_dp_size=1,
        attn_dp_rank=0,
        moe_tp_size=tp_size,
        moe_ep_size=1,
        moe_ep_rank=0,
        moe_dp_size=1,
    )


def test_vision_tower_reduces_over_attention_tp_under_prefill_cp(single_rank):
    """Prefill CP narrows attention TP below TP with attention DP off. Every
    attention-TP group embeds the same media, so a full-TP reduce would sum
    identical partial results instead of one shard set."""
    with _prefill_cp_rank(tp_size=4, attn_tp_size=2, cp_rank=0):
        block = hunyuan_v4_vl.HYV4VisionBlock(_tiny_vl_config().vision_config, None)
    assert block.self_attn.proj.use_dp_attention_reduce
    assert block.mlp.dense_4h_to_h.use_dp_attention_reduce


def test_cp_extend_pairs_full_sequence_embeddings_with_mrope_columns(
    single_rank, monkeypatch
):
    """The CP runner embeds the whole prompt before sharding, then each rank's text
    body gets the token rows and (3, N) mRoPE columns of the same global tokens,
    and the VL language model's head produces the logits."""
    model = hunyuan_v4_vl.HYV4VLForConditionalGeneration(_tiny_vl_config())
    _fill_deterministic(model)
    input_ids = torch.tensor([3, 1, 4, 1, 5, 9, 2])
    num_tokens = input_ids.numel()
    # Distinct rows, as image tokens have; a 1-D or row-sharded split loses them.
    mrope_positions = torch.stack(
        [torch.arange(num_tokens) * 3 + row for row in range(3)]
    )
    full_embeds = model.language_model.model.embed_tokens(input_ids)

    embedded_ids = []
    real_prepare_mm_inputs = hunyuan_v4_vl.prepare_mm_inputs

    def recording_prepare_mm_inputs(**kwargs):
        embedded_ids.append(kwargs["input_ids"])
        return real_prepare_mm_inputs(**kwargs)

    monkeypatch.setattr(hunyuan_v4_vl, "prepare_mm_inputs", recording_prepare_mm_inputs)
    body_calls = []

    def body(
        input_ids, positions, forward_batch, input_embeds=None, pp_proxy_tensors=None
    ):
        body_calls.append((input_ids, positions, input_embeds))
        return input_embeds

    monkeypatch.setattr(model.language_model.model, "forward", body)
    logits = _RecordingLogitsProcessor()
    monkeypatch.setattr(model.language_model, "logits_processor", logits)
    monkeypatch.setattr(eager_runner, "cp_gather_after_forward", lambda x, *_: x)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_: None)
    runner = object.__new__(eager_runner.EagerRunner)
    runner.model_runner = SimpleNamespace(model=model)

    init_cp_strategy(enable_prefill_cp=True, cp_size=2, cp_strategy="interleave")
    try:
        for rank in range(2):
            with _prefill_cp_rank(tp_size=2, attn_tp_size=1, cp_rank=rank):
                metadata = InterleaveCPStrategy(cp_size=2).build_metadata(
                    num_tokens=num_tokens,
                    seqs_len=[num_tokens],
                    extend_seqs_len=[num_tokens],
                )
                pad_logical_token_to_physical(metadata)
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.EXTEND,
                    input_ids=input_ids,
                    positions=torch.arange(num_tokens),
                    mrope_positions=mrope_positions,
                    extend_seq_lens_cpu=[num_tokens],
                    contains_mm_inputs=lambda: False,
                    input_embeds=None,
                    spec_info=None,
                    attn_cp_metadata=metadata,
                )
                assert runner._execute_extend_cp(batch, {}) == "logits"

            local_ids, local_positions, local_embeds = body_calls[rank]
            logical_len = metadata.per_rank_logical_token[rank]
            assert torch.equal(embedded_ids[rank], input_ids)
            assert local_positions.shape == (3, metadata.per_rank_actual_token[rank])
            assert torch.equal(
                local_positions[:, :logical_len], mrope_positions[:, rank::2]
            )
            assert torch.equal(local_ids[:logical_len], input_ids[rank::2])
            torch.testing.assert_close(local_embeds[:logical_len], full_embeds[rank::2])
            ((logits_input_ids, hidden_states),) = logits.calls[rank : rank + 1]
            assert logits_input_ids is input_ids
            assert hidden_states is local_embeds
    finally:
        init_cp_strategy(enable_prefill_cp=False, cp_size=1, cp_strategy="interleave")


def _pp_stage(rank):
    group = SimpleNamespace(
        world_size=2,
        rank_in_group=rank,
        is_first_rank=rank == 0,
        is_last_rank=rank == 1,
    )
    return get_parallel().override(pp_size=2, pp_rank=rank, pp_group=group)


def _three_token_decode_batch():
    return ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=3,
        input_ids=torch.tensor([1, 2, 3]),
        positions=torch.arange(3),
        out_cache_loc=torch.zeros(3, dtype=torch.long),
        req_pool_indices=torch.zeros(3, dtype=torch.long),
        seq_lens=torch.ones(3, dtype=torch.int32),
        seq_lens_cpu=torch.ones(3, dtype=torch.int32),
        seq_lens_sum=3,
        spec_info=None,
    )


def _stub_layer_compute(model, monkeypatch):
    """Replace attention and MLP math; keep the iHC streams and stage boundaries.
    A shared-indexer layer fails unless it receives the previous top-k."""
    config = model.config
    for layer_id in range(model.start_layer, model.end_layer):
        layer = model.model.layers[layer_id]
        reuses_topk = dsa_layer_skips_topk(config, layer_id)

        def attn(
            positions,
            hidden_states,
            forward_batch,
            zero_allocator,
            prev_topk_indices=None,
            layer_id=layer_id,
            reuses_topk=reuses_topk,
        ):
            if reuses_topk:
                assert prev_topk_indices is not None, layer_id
                topk = prev_topk_indices
            else:
                topk = (hidden_states[:, :8].abs() * 100).to(torch.int32) + layer_id
            return hidden_states * 0.5 + topk.float().mean(-1, keepdim=True), topk

        monkeypatch.setattr(layer.self_attn, "forward", attn)
        monkeypatch.setattr(
            layer.mlp, "forward", lambda x, *args, **kwargs: torch.tanh(x) * 0.25
        )


@pytest.mark.parametrize("num_layers", [3, 4])
def test_pp_two_stages_match_one_stage(single_rank, monkeypatch, num_layers):
    """A pipeline handoff carries the widened iHC streams and, when the next
    stage starts on a shared-indexer layer, the DSA top-k: two stages must
    reproduce the single-stage hidden states."""
    config = HYV4Config(
        **(TEXT_OVERRIDES | dict(num_hidden_layers=num_layers)),
        architectures=["HYV4ForCausalLM"],
    )
    whole = hunyuan_v4.HYV4ForCausalLM(config)
    _fill_deterministic(whole, std=0.2)
    _stub_layer_compute(whole, monkeypatch)
    input_ids = torch.tensor([1, 2, 3])
    expected = whole.model(input_ids, torch.arange(3), _three_token_decode_batch())

    stages = []
    for rank in range(2):
        with _pp_stage(rank):
            stage = hunyuan_v4.HYV4ForCausalLM(config)
        missing, _ = stage.load_state_dict(whole.state_dict(), strict=False)
        assert not missing
        _stub_layer_compute(stage, monkeypatch)
        stages.append(stage)
    with _pp_stage(0):
        proxy = stages[0].model(input_ids, torch.arange(3), _three_token_decode_batch())
    assert proxy["hidden_states"].shape == (3, config.hc_mult * config.hidden_size)
    assert ("topk_indices" in proxy.tensors) == dsa_layer_skips_topk(
        config, stages[1].start_layer
    )
    with _pp_stage(1):
        actual = stages[1].model(
            None, torch.arange(3), _three_token_decode_batch(), pp_proxy_tensors=proxy
        )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("num_layers", [3, 4])
def test_pp_stages_pass_an_empty_idle_batch(single_rank, monkeypatch, num_layers):
    """Attention-DP scheduling hands a locally empty rank an IDLE batch; every
    stage must forward it with zero tokens."""
    config = HYV4Config(
        **(TEXT_OVERRIDES | dict(num_hidden_layers=num_layers)),
        architectures=["HYV4ForCausalLM"],
    )
    batch = ForwardBatch(
        forward_mode=ForwardMode.IDLE,
        batch_size=0,
        input_ids=torch.empty(0, dtype=torch.long),
        positions=torch.empty(0, dtype=torch.long),
        out_cache_loc=torch.empty(0, dtype=torch.long),
        req_pool_indices=torch.empty(0, dtype=torch.long),
        seq_lens=torch.empty(0, dtype=torch.int32),
        seq_lens_cpu=torch.empty(0, dtype=torch.int32),
        seq_lens_sum=0,
        spec_info=None,
    )
    stages = []
    for rank in range(2):
        with _pp_stage(rank):
            stage = hunyuan_v4.HYV4ForCausalLM(config)
        _fill_deterministic(stage, std=0.2)
        _stub_layer_compute(stage, monkeypatch)
        stages.append(stage)
    with _pp_stage(0):
        proxy = stages[0].model(batch.input_ids, batch.positions, batch)
    assert proxy["hidden_states"].shape == (0, config.hc_mult * config.hidden_size)
    with _pp_stage(1):
        out = stages[1].model(None, batch.positions, batch, pp_proxy_tensors=proxy)
    assert out.shape == (0, config.hidden_size)


@pytest.mark.parametrize(
    "forward_mode, is_decode",
    [
        (ForwardMode.EXTEND, False),
        (ForwardMode.TARGET_VERIFY, False),
        (ForwardMode.DECODE, True),
    ],
    ids=["extend", "target_verify", "decode"],
)
def test_ihc_forward_mode_spans_every_layer_and_the_head(
    single_rank, monkeypatch, forward_mode, is_decode
):
    """The iHC backend policy reads the forward mode from a context the model
    sets; a speculative verify step must select like extend, through the head."""
    config = HYV4Config(
        **(TEXT_OVERRIDES | dict(num_hidden_layers=2)),
        architectures=["HYV4ForCausalLM"],
    )
    model = hunyuan_v4.HYV4ForCausalLM(config)
    _fill_deterministic(model, std=0.2)
    _stub_layer_compute(model, monkeypatch)
    seen = []

    def record(module):
        forward = module.forward

        def recorded(*args, **kwargs):
            seen.append(hy4_ihc._ihc_decode_forward.get())
            return forward(*args, **kwargs)

        monkeypatch.setattr(module, "forward", recorded)

    for layer in model.model.layers:
        record(layer.self_attn)
    record(model.model.hc_head)
    batch = _three_token_decode_batch()
    batch.forward_mode = forward_mode
    model.model(batch.input_ids, batch.positions, batch)
    assert seen == [is_decode] * (config.num_hidden_layers + 1)
    assert hy4_ihc._ihc_decode_forward.get() is False


def test_pp_stages_load_only_their_own_weights(single_rank):
    """Each stage holds and loads only its part of the checkpoint: the first
    stage the vision tower and embedding, the last the head and final norm."""
    config = HYV4VLConfig(
        text_config=VL_TEXT_OVERRIDES | dict(num_hidden_layers=2),
        vision_config=VISION_OVERRIDES,
    )
    embed_rows = torch.arange(256.0).unsqueeze(1).expand(256, 64)
    checkpoint = [
        ("vit.embeddings.patch_embedding.weight", torch.full((64, 3, 16, 16), 6.0)),
        ("model.embed_tokens.weight", embed_rows),
        ("model.layers.0.input_layernorm.weight", torch.full((64,), 2.0)),
        ("model.layers.1.input_layernorm.weight", torch.full((64,), 3.0)),
        ("model.norm.weight", torch.full((64,), 4.0)),
        ("model.hc_head.hc_head_scale", torch.full((1,), 5.0)),
        ("lm_head.weight", torch.full((256, 64), 8.0)),
    ]
    for rank, owned in (
        (
            0,
            {
                "model.embed_tokens.weight": embed_rows,
                "model.layers.0.input_layernorm.weight": 2.0,
            },
        ),
        (
            1,
            {
                "model.layers.1.input_layernorm.weight": 3.0,
                "model.norm.weight": 4.0,
                "model.hc_head.hc_head_scale": 5.0,
                "lm_head.weight": 8.0,
            },
        ),
    ):
        with _pp_stage(rank):
            model = hunyuan_v4_vl.HYV4VLForConditionalGeneration(config)
            model.load_weights(iter(checkpoint))
        assert (model.visual is not None) == (rank == 0)
        if model.visual is not None:
            assert torch.all(model.visual.embeddings.patch_embedding.weight == 6.0)
        params = dict(model.language_model.named_parameters())
        assert ("model.embed_tokens.weight" in params) == (rank == 0)
        assert ("lm_head.weight" in params) == (rank == 1)
        assert ("model.hc_head.hc_head_scale" in params) == (rank == 1)
        for name, value in owned.items():
            assert torch.all(params[name] == value), name


def test_pp_mtp_draft_takes_the_last_stage_head_and_the_checkpoint_embedding(
    single_rank, monkeypatch, tmp_path
):
    """Under PP the MTP draft runs on the last stage, which holds the target
    head but not its embedding: the draft shares the head and loads the
    checkpoint embedding into its own parameter."""
    config = HYV4VLConfig(
        text_config=VL_TEXT_OVERRIDES | DRAFT_MOE_OVERRIDES | dict(num_hidden_layers=2),
        vision_config=VISION_OVERRIDES,
    )
    embed_rows = torch.arange(256.0).unsqueeze(1).expand(256, 64).contiguous()
    save_file({"model.embed_tokens.weight": embed_rows}, tmp_path / "shard.safetensors")
    (tmp_path / SAFETENSORS_INDEX_NAME).write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": "shard.safetensors"}})
    )

    whole = hunyuan_v4_vl.HYV4VLForConditionalGeneration(config)
    embed, head = whole.get_embed_and_head()
    assert embed is whole.language_model.model.embed_tokens.weight
    assert head is whole.language_model.lm_head.weight

    with _pp_stage(0):
        first = hunyuan_v4_vl.HYV4VLForConditionalGeneration(config)
    assert first.get_embed_and_head()[1] is None
    with _pp_stage(1):
        last = hunyuan_v4_vl.HYV4VLForConditionalGeneration(config)
    draft = _tiny_nextn_draft(monkeypatch, config)
    embed, head = resolve_draft_embed_and_head(
        target_model=last,
        draft_model=draft,
        model_path=str(tmp_path),
        revision=None,
        load_config=LoadConfig(),
    )
    draft.set_embed_and_head(embed, head)
    assert draft.lm_head.weight is last.language_model.lm_head.weight
    assert torch.equal(draft.model.embed_tokens.weight, embed_rows)


@pytest.mark.parametrize("speculative", [True, False])
def test_pp_relays_media_embeddings_to_the_drafting_stage(
    single_rank, monkeypatch, speculative
):
    """The first stage embeds media; under EAGLE the last stage needs the
    merged embeddings for the draft prefill, so they ride the hand-off."""
    config = HYV4Config(
        **(TEXT_OVERRIDES | dict(num_hidden_layers=4)),
        architectures=["HYV4ForCausalLM"],
    )
    algorithm = SpeculativeAlgorithm.EAGLE if speculative else SpeculativeAlgorithm.NONE
    stages = []
    for rank in range(2):
        with _pp_stage(rank):
            stage = hunyuan_v4.HYV4ForCausalLM(config)
        _fill_deterministic(stage, std=0.2)
        _stub_layer_compute(stage, monkeypatch)
        stages.append(stage)
    media = torch.randn(3, config.hidden_size)
    sender = _three_token_decode_batch()
    sender.spec_algorithm = algorithm
    sender.mm_input_embeds = media
    with _pp_stage(0):
        proxy = stages[0].model(sender.input_ids, sender.positions, sender)
    assert ("mm_input_embeds" in proxy.tensors) == speculative
    assert "topk_indices" in proxy.tensors

    receiver = _three_token_decode_batch()
    receiver.spec_algorithm = algorithm
    with _pp_stage(1):
        stages[1].model(None, receiver.positions, receiver, pp_proxy_tensors=proxy)
    if speculative:
        assert receiver.mm_input_embeds is media
    else:
        assert receiver.mm_input_embeds is None


def test_vl_model_disables_shared_expert_fusion(single_rank, monkeypatch):
    """Shared experts must stay unfused, or they inherit the routed SwiGLU limit."""
    config = HYV4VLConfig(
        text_config=VL_TEXT_OVERRIDES
        | dict(
            num_hidden_layers=2,
            moe_intermediate_size=32,
            n_routed_experts=4,
            num_experts_per_tok=2,
        ),
        vision_config=VISION_OVERRIDES,
    )
    with get_context().override_server_args(disable_shared_experts_fusion=False):
        shared_experts = resolve_shared_experts_config(
            hunyuan_v4_vl.HYV4VLForConditionalGeneration, config, None
        )
        with get_model().override(shared_experts=shared_experts):
            model = hunyuan_v4_vl.HYV4VLForConditionalGeneration(config)
    moe = model.language_model.model.layers[1].mlp
    assert isinstance(moe, DeepseekV2MoE)
    assert moe.num_fused_shared_experts == 0


def test_vl_vision_tower_stays_unquantized_under_mxfp8(single_rank):
    """An MXFP8 checkpoint quantizes the text model only; the vision tower
    keeps bf16 linears, matching the reference split."""
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True, weight_block_size=[1, 32], use_mxfp8=True
    )
    model = hunyuan_v4_vl.HYV4VLForConditionalGeneration(
        _tiny_vl_config(), quant_config=quant_config
    )

    def linear_methods(module):
        return {
            type(layer.quant_method)
            for layer in module.modules()
            if isinstance(layer, LinearBase)
        }

    assert Fp8LinearMethod in linear_methods(model.language_model)
    assert linear_methods(model.visual) == {UnquantizedLinearMethod}


def test_vl_rejects_a_quantized_vision_tower(single_rank):
    """The vision tower is built unquantized, so a vision quantization config or
    quantized vision tensors must fail instead of loading as plain bf16."""
    with pytest.raises(ValueError, match="vision_config.quantization_config"):
        HYV4VLVisionConfig(quantization_config={"quant_method": "fp8"})
    model = hunyuan_v4_vl.HYV4VLForConditionalGeneration(_tiny_vl_config())
    weight = "layers.0.mlp.dense_h_to_4h.weight"
    for name, tensor in (
        (weight, torch.zeros(128, 64, dtype=torch.float8_e4m3fn)),
        (f"{weight}_scale_inv", torch.ones(1, 1)),
    ):
        with pytest.raises(ValueError, match=f"quantized vision tensor visual.{name}"):
            model.load_weights([(f"vit.{name}", tensor)])


def test_vl_model_load_weights_routes_and_renames(single_rank):
    """Checkpoint names route to the right tower while streaming; DSA indexer
    weights load, and norm names without a known destination are rejected."""
    model = hunyuan_v4_vl.HYV4VLForConditionalGeneration(_tiny_vl_config())
    torch.manual_seed(0)
    vision_params = dict(model.visual.named_parameters())
    lm_params = dict(model.language_model.named_parameters())
    hf_weights = {
        "vit.embeddings.patch_embedding.weight": torch.randn(
            vision_params["embeddings.patch_embedding.weight"].shape
        ),
        "model.embed_tokens.weight": torch.randn(
            lm_params["model.embed_tokens.weight"].shape
        ),
        "model.layers.0.self_attn.indexer.weights_proj.weight": torch.full(
            (TEXT_OVERRIDES["index_n_heads"], TEXT_OVERRIDES["hidden_size"]), 4.0
        ),
        "model.mtp_layers.0.eh_proj.weight": torch.zeros(64, 128),
        "lm_head.weight": torch.randn(lm_params["lm_head.weight"].shape),
    }
    vision_loaded_while_streaming = []

    def checkpoint():
        for name, tensor in hf_weights.items():
            yield name, tensor
            if name == "vit.embeddings.patch_embedding.weight":
                vision_loaded_while_streaming.append(
                    torch.equal(model.visual.embeddings.patch_embedding.weight, tensor)
                )

    model.load_weights(checkpoint())

    assert vision_loaded_while_streaming == [True]
    torch.testing.assert_close(
        lm_params["model.embed_tokens.weight"], hf_weights["model.embed_tokens.weight"]
    )
    assert torch.all(
        lm_params["model.layers.0.self_attn.indexer.weights_proj.weight"] == 4.0
    )
    torch.testing.assert_close(
        lm_params["lm_head.weight"], hf_weights["lm_head.weight"]
    )

    with pytest.raises(ValueError, match="Unmapped"):
        model.load_weights(iter([("unexpected.weight", torch.zeros(1))]))
    for name in (
        "model.layers.0.self_attn.query_layernorm.weight",
        "model.layers.0.self_attn.key_layernorm.weight",
    ):
        with pytest.raises(ValueError, match="Unsupported HYV4 VL checkpoint weight"):
            model.load_weights(iter([(name, torch.zeros(16))]))


FP4_QUANT_CONFIG = {
    "quant_method": "fp8",
    "fmt": "e4m3",
    "weight_block_size": [128, 128],
    "activation_scheme": "dynamic",
    "scale_fmt": "ue8m0",
    "expert_dtype": "mxfp4",
}
# 128-row q_lora and expert widths keep the FP8 blocks and MXFP4 groups aligned
# as in the checkpoint (fused q_a/kv_a scales concatenate at a block edge).
FP4_TEXT_OVERRIDES = (
    VL_TEXT_OVERRIDES
    | DRAFT_MOE_OVERRIDES
    | dict(num_hidden_layers=2, q_lora_rank=128, moe_intermediate_size=128)
)


def _write_fp4_config(path, quantization_config, nested=False, **text_overrides):
    text_config = FP4_TEXT_OVERRIDES | text_overrides
    config = {
        "architectures": ["HYV4VLForConditionalGeneration"],
        "model_type": "hy_v4_vl",
        "text_config": text_config,
        "vision_config": VISION_OVERRIDES,
    }
    if nested:
        text_config["quantization_config"] = quantization_config
    else:
        config["quantization_config"] = quantization_config
    (path / "config.json").write_text(json.dumps(config))
    return str(path)


def _ue8m0(*shape):
    return torch.randint(118, 131, shape, dtype=torch.uint8)


def _fp8_weight(out, inp):
    return (torch.randn(out, inp) * 0.5).to(torch.float8_e4m3fn)


def _hyv4_checkpoint(text, quantized, experts):
    """Names and shapes of a tencent/Hy4 checkpoint: bf16 except the linears
    `quantized(name, out, inp, mtp)` stores and the routed experts."""
    torch.manual_seed(0)
    hidden, heads = text.hidden_size, text.num_attention_heads
    tensors = {}

    def dense(name, *shape, dtype=torch.bfloat16):
        tensors[name] = (torch.randn(shape) * 0.05).to(dtype)

    def linear(name, out, inp, mtp):
        stored = quantized(name, out, inp, mtp)
        if stored is None:
            return dense(f"{name}.weight", out, inp)
        tensors.update(stored)

    def layer(prefix, layer_id, mtp):
        attn = f"{prefix}.self_attn"
        qk_head_dim = text.qk_nope_head_dim + text.qk_rope_head_dim
        linear(f"{attn}.q_a_proj", text.q_lora_rank, hidden, mtp)
        linear(f"{attn}.q_b_proj", heads * qk_head_dim, text.q_lora_rank, mtp)
        linear(
            f"{attn}.kv_a_proj_with_mqa",
            text.kv_lora_rank + text.qk_rope_head_dim,
            hidden,
            mtp,
        )
        linear(
            f"{attn}.kv_b_proj",
            heads * (text.qk_nope_head_dim + text.v_head_dim),
            text.kv_lora_rank,
            mtp,
        )
        linear(f"{attn}.o_proj", hidden, heads * text.v_head_dim, mtp)
        linear(f"{attn}.linear_gate", heads * text.v_head_dim, hidden, mtp)
        dense(f"{attn}.q_a_layernorm.weight", text.q_lora_rank)
        dense(f"{attn}.kv_a_layernorm.weight", text.kv_lora_rank)
        dense(f"{attn}.learnable_sink_param", heads, dtype=torch.float32)
        if mtp or not dsa_layer_skips_topk(text, layer_id):
            index = f"{attn}.indexer"
            linear(
                f"{index}.wq_b",
                text.index_n_heads * text.index_head_dim,
                text.q_lora_rank,
                mtp,
            )
            linear(f"{index}.wk", text.index_head_dim, hidden, mtp)
            dense(f"{index}.weights_proj.weight", text.index_n_heads, hidden)
            dense(f"{index}.k_norm.weight", text.index_head_dim)
            dense(f"{index}.k_norm.bias", text.index_head_dim)
        dense(f"{prefix}.input_layernorm.weight", hidden)
        dense(f"{prefix}.post_attention_layernorm.weight", hidden)
        mlp = f"{prefix}.mlp"
        if not mtp and text.mlp_layer_types[layer_id] == "dense":
            for proj, out, inp in (
                ("gate_proj", text.intermediate_size, hidden),
                ("up_proj", text.intermediate_size, hidden),
                ("down_proj", hidden, text.intermediate_size),
            ):
                linear(f"{mlp}.{proj}", out, inp, mtp)
            return
        dense(f"{mlp}.gate.weight", text.n_routed_experts, hidden)
        dense(
            f"{mlp}.gate.e_score_correction_bias",
            text.n_routed_experts,
            dtype=torch.float32,
        )
        shared = text.moe_intermediate_size * text.n_shared_experts
        linear(f"{mlp}.shared_experts.gate_proj", shared, hidden, mtp)
        linear(f"{mlp}.shared_experts.up_proj", shared, hidden, mtp)
        linear(f"{mlp}.shared_experts.down_proj", hidden, shared, mtp)
        inter = text.moe_intermediate_size
        for e in range(text.n_routed_experts):
            for proj, out, inp in (
                ("gate_proj", inter, hidden),
                ("up_proj", inter, hidden),
                ("down_proj", hidden, inter),
            ):
                tensors.update(experts(f"{mlp}.experts.{e}.{proj}", out, inp))

    hc_width = text.hc_mult * hidden
    for i in range(text.num_hidden_layers):
        layer(f"model.layers.{i}", i, mtp=False)
        for hc in ("hc_attn_layer", "hc_mlp_layer"):
            pre = f"model.layers.{i}.{hc}.hc_pre"
            dense(f"{pre}.hc_fn", 2 * text.hc_mult, hc_width, dtype=torch.float32)
            dense(f"{pre}.hc_base", 2 * text.hc_mult, dtype=torch.float32)
            dense(f"{pre}.hc_scale", 2, dtype=torch.float32)
    mtp = "model.mtp_layers.0"
    layer(mtp, 0, mtp=True)
    for norm in ("enorm", "hnorm", "final_layernorm"):
        dense(f"{mtp}.{norm}.weight", hidden)
    dense(f"{mtp}.eh_proj.weight", hidden, 2 * hidden)
    dense("model.hc_head.hc_head_fn", text.hc_mult, hc_width, dtype=torch.float32)
    dense("model.hc_head.hc_head_base", text.hc_mult, dtype=torch.float32)
    dense("model.hc_head.hc_head_scale", 1, dtype=torch.float32)
    dense("model.embed_tokens.weight", text.vocab_size, hidden)
    dense("model.norm.weight", hidden)
    dense("lm_head.weight", text.vocab_size, hidden)
    for name, shape in _reference_vision_checkpoint(num_layers=1).items():
        dense(f"vit.{name}", *shape)
    return tensors


def _fp4_checkpoint(text):
    """MXFP4 routed experts and a block-FP8 MTP layer use different scale layouts."""

    def quantized(name, out, inp, mtp):
        if not mtp:
            return None
        return {
            f"{name}.weight": _fp8_weight(out, inp),
            f"{name}.scale": _ue8m0(-(-out // 128), -(-inp // 128)),
        }

    def experts(name, out, inp):
        return {
            f"{name}.weight": torch.randint(1, 256, (out, inp // 2), dtype=torch.uint8),
            f"{name}.weight_scale": _ue8m0(out, inp // 32),
        }

    return _hyv4_checkpoint(text, quantized, experts)


def _modules_to_not_convert(checkpoint):
    return sorted(
        {
            name.removesuffix(".weight").removesuffix(".bias")
            for name, tensor in checkpoint.items()
            if tensor.dtype in (torch.bfloat16, torch.float32)
            and ".experts." not in name
        }
    )


@pytest.mark.parametrize("nested", [False, True], ids=["top_level", "text_config"])
def test_fp4_checkpoint_resolves_fp8_with_mxfp4_routed_experts(tmp_path, nested):
    """The checkpoint's FP8 config with expert_dtype mxfp4 resolves to fp8 with
    MXFP4 routed experts for the target and the MTP draft, also when nested
    under text_config; MXFP4 groups other than 32 are refused."""
    path = _write_fp4_config(tmp_path, FP4_QUANT_CONFIG, nested)
    for is_draft_model in (False, True):
        model_config = ModelConfig(path, is_draft_model=is_draft_model)
        assert model_config.quantization == "fp8"
        assert model_config.is_fp4_experts

    (tmp_path / "group16").mkdir()
    path = _write_fp4_config(
        tmp_path / "group16", FP4_QUANT_CONFIG | {"group_size": 16}, nested
    )
    with pytest.raises(ValueError, match="group_size=32"):
        ModelConfig(path)


@pytest.mark.parametrize("sm90", [True, False], ids=["sm90", "sm100"])
def test_fp4_serving_defaults_per_platform(tmp_path, monkeypatch, sm90):
    """MXFP4 experts default to flashinfer_mxfp4 on SM90 and SM100, where auto
    fails at warmup on SM90."""
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda *_: (9, 0) if sm90 else (10, 0)
    )
    path = _write_fp4_config(tmp_path, FP4_QUANT_CONFIG)
    with override_platform(is_cuda=True, is_sm90=sm90, is_sm100=not sm90):
        server_args = ServerArgs(model_path=path, device="cuda")
        server_args.resolve_once()
    assert resolution_result(server_args, "moe_runner_backend") == "flashinfer_mxfp4"


def test_fp4_scale_aliases_decode_ue8m0_only_for_float_scale_params():
    """ ".scale" and ".weight_scale" load as ".weight_scale_inv"; UE8M0 bytes
    become 2**(byte - 127) for FP8-block params, but stay raw bytes under MXFP8,
    whose scale params are uint8."""
    raw = torch.tensor([[0, 126, 127, 128, 254]], dtype=torch.uint8)
    block_fp8 = Fp8Config.from_config(FP4_QUANT_CONFIG)
    mxfp8 = Fp8Config.from_config(
        {"quant_method": "mxfp8", "activation_scheme": "dynamic"}
    )
    for alias in (".scale", ".weight_scale", ".weight_scale_inv"):
        name, scale = hunyuan_v4.normalize_hyv4_weight_scale(
            f"model.layers.1.mlp.experts.0.up_proj{alias}", raw, block_fp8
        )
        assert name == "model.layers.1.mlp.experts.0.up_proj.weight_scale_inv"
        torch.testing.assert_close(scale, torch.exp2(raw.float() - 127))
        _, scale = hunyuan_v4.normalize_hyv4_weight_scale(
            f"model.layers.1.mlp.experts.0.up_proj{alias}", raw, mxfp8
        )
        assert scale is raw
    for name in (
        "model.hc_head.hc_head_scale",
        "model.layers.0.hc_attn_layer.hc_pre.hc_scale",
    ):
        assert hunyuan_v4.normalize_hyv4_weight_scale(name, raw, block_fp8) == (
            name,
            raw,
        )


def test_fp4_indexer_permutation_keeps_whole_head_block_scales():
    """The indexer's in-head rope permutation moves weight rows but leaves a
    block scale whose block spans whole heads; a block that splits a head is
    refused instead of mis-scaling the moved rows."""
    config = SimpleNamespace(index_n_heads=2, index_head_dim=4, qk_rope_head_dim=2)
    name = "model.mtp_layers.0.self_attn.indexer.wq_b.weight_scale_inv"
    scale = torch.ones(1, 3)
    assert hunyuan_v4.permute_hyv4_indexer_weight(name, scale, config) is scale
    with pytest.raises(ValueError, match="index_head_dim"):
        hunyuan_v4.permute_hyv4_indexer_weight(name, torch.ones(4, 3), config)


def _poison(model):
    for param in model.parameters():
        if param.dtype.is_floating_point:
            param.data.fill_(float("nan"))
        else:
            param.data.zero_()


def _unwritten(model, skip=()):
    # FP8 configs give attention KV-cache scales, which no HYV4 checkpoint carries.
    return [
        name
        for name, param in model.named_parameters()
        if name not in skip
        and not name.endswith((".k_scale", ".v_scale"))
        and (
            param.float().isnan().any()
            if param.dtype.is_floating_point
            else (param == 0).any()
        )
    ]


def _fused_wk_weights_proj(indexer, text):
    del indexer.wk, indexer.weights_proj
    indexer.wk_weights_proj = ReplicatedLinear(
        text.hidden_size,
        text.index_head_dim + text.index_n_heads,
        bias=False,
        params_dtype=torch.bfloat16,
    )


def test_fp4_checkpoint_loads_every_tensor_at_its_checkpoint_precision(
    single_rank, monkeypatch, tmp_path
):
    """Every target and MTP-draft parameter loads at checkpoint precision: bf16
    linears (fused q_a/kv_a too) unquantized, MTP block FP8, experts MXFP4."""
    text = HYV4VLConfig(
        text_config=FP4_TEXT_OVERRIDES, vision_config=VISION_OVERRIDES
    ).text_config
    checkpoint = _fp4_checkpoint(text)
    quantization_config = FP4_QUANT_CONFIG | {
        "modules_to_not_convert": _modules_to_not_convert(checkpoint)
    }
    path = _write_fp4_config(tmp_path, quantization_config)
    target_config = ModelConfig(path)
    quant_config = _get_quantization_config(target_config, LoadConfig())
    with get_context().override_server_args(disable_shared_experts_fusion=False):
        shared_experts = resolve_shared_experts_config(
            hunyuan_v4_vl.HYV4VLForConditionalGeneration,
            target_config.hf_config,
            quant_config,
        )
        with get_model().override(shared_experts=shared_experts):
            target = hunyuan_v4_vl.HYV4VLForConditionalGeneration(
                target_config.hf_config, quant_config
            )
    for name, module in target.named_modules():
        if isinstance(module, LinearBase):
            assert isinstance(module.quant_method, UnquantizedLinearMethod), name
    _poison(target)
    target.load_weights(iter(checkpoint.items()))
    assert _unwritten(target) == []

    draft_config = ModelConfig(path, is_draft_model=True)
    draft_quant_config = _get_quantization_config(draft_config, LoadConfig())
    shared_experts = resolve_shared_experts_config(
        HYV4ForCausalLMNextN, draft_config.hf_config, draft_quant_config
    )
    with get_model().override(shared_experts=shared_experts):
        draft = HYV4ForCausalLMNextN(draft_config.hf_config, draft_quant_config)
    indexer = draft.model.decoder.self_attn.indexer
    _fused_wk_weights_proj(indexer, text)
    for name, module in draft.model.decoder.named_modules():
        if isinstance(module, LinearBase):
            expected = (
                UnquantizedLinearMethod
                if name.endswith("weights_proj")
                else Fp8LinearMethod
            )
            assert isinstance(module.quant_method, expected), name
    _poison(draft)
    draft.load_weights(iter(checkpoint.items()))
    assert _unwritten(draft, skip={"model.embed_tokens.weight", "lm_head.weight"}) == []

    mtp = "model.mtp_layers.0"
    for prefix, moe in (
        ("model.layers.1", target.language_model.model.layers[1].mlp),
        (mtp, draft.model.decoder.mlp),
    ):
        experts = moe.experts
        # flashinfer_mxfp4 routes FP4 experts by standard top-k ids and weights.
        assert moe.topk.is_fp4_experts
        assert experts.w13_weight.dtype == torch.int8
        gate = f"{prefix}.mlp.experts.0.gate_proj"
        torch.testing.assert_close(
            experts.w13_weight[0, : text.moe_intermediate_size].view(torch.uint8),
            checkpoint[f"{gate}.weight"],
        )
        torch.testing.assert_close(
            experts.w13_weight_scale_inv[0, : text.moe_intermediate_size],
            torch.exp2(checkpoint[f"{gate}.weight_scale"].float() - 127),
        )
    torch.testing.assert_close(
        draft.model.decoder.self_attn.fused_qkv_a_proj_with_mqa.weight_scale_inv,
        torch.exp2(
            torch.cat(
                [
                    checkpoint[f"{mtp}.self_attn.q_a_proj.scale"],
                    checkpoint[f"{mtp}.self_attn.kv_a_proj_with_mqa.scale"],
                ]
            ).float()
            - 127
        ),
    )
    wk = hunyuan_v4.permute_hyv4_indexer_weight(
        f"{mtp}.self_attn.indexer.wk.weight",
        checkpoint[f"{mtp}.self_attn.indexer.wk.weight"].float(),
        text,
    ) * torch.exp2(checkpoint[f"{mtp}.self_attn.indexer.wk.scale"].float() - 127)
    torch.testing.assert_close(
        indexer.wk_weights_proj.weight,
        torch.cat(
            [
                wk.bfloat16(),
                checkpoint[f"{mtp}.self_attn.indexer.weights_proj.weight"],
            ]
        ),
    )


FP8_QUANT_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "float-quantized",
    "quantization_status": "compressed",
    "global_compression_ratio": None,
    "kv_cache_scheme": None,
    "config_groups": {
        "block_fp8_w8a8": {
            "weights": {
                "num_bits": 8,
                "type": "float",
                "strategy": "block",
                "block_structure": [128, 128],
                "symmetric": True,
                "dynamic": False,
                "observer": "minmax",
            },
            "input_activations": {
                "num_bits": 8,
                "type": "float",
                "strategy": "group",
                "group_size": 128,
                "symmetric": True,
                "dynamic": True,
                "observer": None,
            },
            "output_activations": None,
            "targets": [
                r"re:(?:language_model\.)?model\.layers\.\d+\.self_attn\.(?:o_proj|linear_gate)$",
                r"re:(?:language_model\.)?model\.layers\.\d+\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)$",
                r"re:(?:language_model\.)?model\.layers\.78\.self_attn\.(?:q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj)$",
                r"re:(?:language_model\.)?model\.layers\.78\.self_attn\.indexer\.wq_b$",
                r"re:(?:language_model\.)?model\.layers\.78\.mlp\.shared_experts\.(?:gate_proj|up_proj|down_proj)$",
            ],
        }
    },
    "ignore": [
        r"re:.*lm_head$",
        r"re:(?:visual|vit)\..*",
        r"re:(?:language_model\.)?model\.layers\.0\.self_attn\..*",
        r"re:(?:language_model\.)?model\.layers\.\d+\.self_attn\.indexer\.(?:wk|weights_proj)$",
    ],
}


def _fp8_quant_config(num_hidden_layers):
    # The checkpoint names its MTP layer model.layers.<num_hidden_layers> (78).
    text = json.dumps(FP8_QUANT_CONFIG).replace(
        r"layers\\.78", rf"layers\\.{num_hidden_layers}"
    )
    return json.loads(text)


def _fp8_checkpoint(text):
    """Block-FP8 weights use float32 scales except for the documented BF16 tensors."""

    def block_fp8(name, out, inp):
        return {
            f"{name}.weight": _fp8_weight(out, inp),
            f"{name}.weight_scale": torch.rand(-(-out // 128), -(-inp // 128)) + 0.5,
        }

    def quantized(name, out, inp, mtp):
        attn_out = name.endswith((".o_proj", ".linear_gate"))
        if (attn_out and not name.startswith("model.layers.0.")) or (
            mtp and not name.endswith(".indexer.wk")
        ):
            return block_fp8(name, out, inp)
        return None

    return _hyv4_checkpoint(text, quantized, block_fp8)


def _fp8_target(monkeypatch, tmp_path, quantization_config=None):
    # The FP8 schemes probe the GPU generation; the checkpoint serves on SM100.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (10, 0))
    text = HYV4VLConfig(
        text_config=FP4_TEXT_OVERRIDES, vision_config=VISION_OVERRIDES
    ).text_config
    if quantization_config is None:
        quantization_config = _fp8_quant_config(text.num_hidden_layers)
    path = _write_fp4_config(tmp_path, quantization_config)
    target_config = ModelConfig(path)
    quant_config = _get_quantization_config(target_config, LoadConfig())
    with get_context().override_server_args(disable_shared_experts_fusion=False):
        shared_experts = resolve_shared_experts_config(
            hunyuan_v4_vl.HYV4VLForConditionalGeneration,
            target_config.hf_config,
            quant_config,
        )
        with get_model().override(shared_experts=shared_experts):
            target = hunyuan_v4_vl.HYV4VLForConditionalGeneration(
                target_config.hf_config, quant_config
            )
    return target, text, path


def test_fp8_checkpoint_loads_every_tensor_at_its_checkpoint_precision(
    single_rank, monkeypatch, tmp_path
):
    """The compressed-tensors FP8 checkpoint names only its quantized modules in
    `targets`: every target and MTP-draft parameter loads at checkpoint
    precision, with scales under their checkpoint names."""
    target, text, path = _fp8_target(monkeypatch, tmp_path)
    checkpoint = _fp8_checkpoint(text)
    quantized = {
        "language_model.model.layers.1.self_attn.o_proj",
        "language_model.model.layers.1.self_attn.linear_gate",
    }
    for name, module in target.named_modules():
        if isinstance(module, LinearBase):
            expected = (
                CompressedTensorsLinearMethod
                if name in quantized
                else UnquantizedLinearMethod
            )
            assert isinstance(module.quant_method, expected), name
    layer = target.language_model.model.layers[1]
    assert isinstance(layer.mlp.experts.quant_method, CompressedTensorsFusedMoEMethod)
    _poison(target)
    target.load_weights(iter(checkpoint.items()))
    assert _unwritten(target) == []
    torch.testing.assert_close(
        layer.self_attn.o_proj.weight_scale,
        checkpoint["model.layers.1.self_attn.o_proj.weight_scale"],
    )

    draft_config = ModelConfig(path, is_draft_model=True)
    draft_quant_config = _get_quantization_config(draft_config, LoadConfig())
    shared_experts = resolve_shared_experts_config(
        HYV4ForCausalLMNextN, draft_config.hf_config, draft_quant_config
    )
    with get_model().override(shared_experts=shared_experts):
        draft = HYV4ForCausalLMNextN(draft_config.hf_config, draft_quant_config)
    decoder = draft.model.decoder
    _fused_wk_weights_proj(decoder.self_attn.indexer, text)
    for name, module in decoder.named_modules():
        if isinstance(module, LinearBase):
            expected = (
                UnquantizedLinearMethod
                if name.endswith("weights_proj")
                else CompressedTensorsLinearMethod
            )
            assert isinstance(module.quant_method, expected), name
    assert isinstance(decoder.mlp.experts.quant_method, CompressedTensorsFusedMoEMethod)
    _poison(draft)
    draft.load_weights(iter(checkpoint.items()))
    assert _unwritten(draft, skip={"model.embed_tokens.weight", "lm_head.weight"}) == []
    mtp = "model.mtp_layers.0.self_attn"
    torch.testing.assert_close(
        decoder.self_attn.fused_qkv_a_proj_with_mqa.weight_scale,
        torch.cat(
            [
                checkpoint[f"{mtp}.q_a_proj.weight_scale"],
                checkpoint[f"{mtp}.kv_a_proj_with_mqa.weight_scale"],
            ]
        ),
    )


def test_fp8_checkpoint_scale_outside_the_targets_fails(
    single_rank, monkeypatch, tmp_path
):
    """A module the config misses must stay loud: plain compressed-tensors still
    refuses it, and HYV4 fails the load once the checkpoint stores its scale."""
    target, text, _ = _fp8_target(monkeypatch, tmp_path)
    shared = "model.layers.1.mlp.shared_experts.down_proj"
    plain = CompressedTensorsConfig.from_config(
        _fp8_quant_config(text.num_hidden_layers)
    )
    with pytest.raises(ValueError, match="Unable to find matching target"):
        plain.get_scheme_dict(nn.Linear(1, 1), f"language_model.{shared}")
    with pytest.raises(ValueError, match=f"leaves language_model.{shared} unquantized"):
        target.load_weights(iter([(f"{shared}.weight_scale", torch.ones(1, 1))]))


@pytest.mark.parametrize(
    "experts", [r"[1-9]\d*", "0"], ids=["expert0_unselected", "expert0_only"]
)
def test_fp8_checkpoint_with_partially_selected_experts_fails(
    single_rank, monkeypatch, tmp_path, experts
):
    """A fused expert bank serves one scheme: targets that select only some of
    its experts must fail construction instead of loading the others with the
    bank's precision."""
    config = _fp8_quant_config(FP4_TEXT_OVERRIDES["num_hidden_layers"])
    targets = config["config_groups"]["block_fp8_w8a8"]["targets"]
    targets[1] = targets[1].replace(r"experts\.\d+", rf"experts\.{experts}")
    with pytest.raises(ValueError, match="same quantization scheme"):
        _fp8_target(monkeypatch, tmp_path, config)


def _fp8_draft(path):
    draft_config = ModelConfig(path, is_draft_model=True)
    quant_config = _get_quantization_config(draft_config, LoadConfig())
    shared_experts = resolve_shared_experts_config(
        HYV4ForCausalLMNextN, draft_config.hf_config, quant_config
    )
    with get_model().override(shared_experts=shared_experts):
        return HYV4ForCausalLMNextN(draft_config.hf_config, quant_config)


@pytest.mark.parametrize(
    ("experts", "unselected"),
    [("0", "1"), (r"[1-9]\d*", "0")],
    ids=["zero_only", "zero_unselected"],
)
def test_fp8_draft_with_partially_selected_experts_fails(
    single_rank, monkeypatch, tmp_path, experts, unselected
):
    """The strict MTP draft must also refuse an expert bank its targets select
    only partly, instead of loading the other experts at the bank's precision."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (10, 0))
    text = HYV4VLConfig(
        text_config=FP4_TEXT_OVERRIDES, vision_config=VISION_OVERRIDES
    ).text_config
    config = _fp8_quant_config(text.num_hidden_layers)
    targets = config["config_groups"]["block_fp8_w8a8"]["targets"]
    targets[1] = targets[1].replace(r"experts\.\d+", rf"experts\.{experts}")
    path = _write_fp4_config(tmp_path, config)
    checkpoint = _fp8_checkpoint(text)
    mtp_experts = "model.mtp_layers.0.mlp.experts."
    for name in [n for n in checkpoint if n.startswith(mtp_experts)]:
        expert0 = name.removeprefix(mtp_experts).startswith("0.")
        if expert0 == (unselected == "0"):
            if name.endswith(".weight_scale"):
                del checkpoint[name]
            else:
                checkpoint[name] = checkpoint[name].bfloat16()
    mtp_expert = f"model.layers.{text.num_hidden_layers}.mlp.experts.{unselected}"
    with pytest.raises(ValueError, match=f"matching target for {mtp_expert}.gate_proj"):
        _fp8_draft(path).load_weights(iter(checkpoint.items()))


# Draft depth, block and target layers are fitted to a two-layer target.
DFLY_KEYS = dict(
    dfly_num_hidden_layers=1,
    dfly_block_size=7,
    dfly_mask_token_id=250,
    dfly_target_index_layer=[1],
    dfly_target_kv_layers=[1],
    dfly_target_index_topk=TEXT_OVERRIDES["index_topk"],
    dfly_context_local_window=None,
    dfly_enable_ihc=False,
    dfly_mlp_type="dense",
    dfly_attn_causal=True,
    dfly_mtp_fusion="slot0",
    dfly_enable_hidden_correction=True,
    # The checkpoint leaves this at hidden_size; 128 keeps FP8 blocks whole here.
    dfly_hidden_correction_intermediate_size=128,
)
DFLY_FP8_TARGETS = [
    r"re:(?:language_model\.)?model\.dfly_layers\.\d+\.mlp\.down_proj$",
    r"re:(?:language_model\.)?model\.dfly_layers\.\d+\.mlp\.gate_proj$",
    r"re:(?:language_model\.)?model\.dfly_layers\.\d+\.mlp\.up_proj$",
    r"re:(?:language_model\.)?model\.dfly_layers\.\d+\.self_attn\.kv_a_proj_with_mqa$",
    r"re:(?:language_model\.)?model\.dfly_layers\.\d+\.self_attn\.kv_b_proj$",
    r"re:(?:language_model\.)?model\.dfly_layers\.\d+\.self_attn\.linear_gate$",
    r"re:(?:language_model\.)?model\.dfly_layers\.\d+\.self_attn\.o_proj$",
    r"re:(?:language_model\.)?model\.dfly_layers\.\d+\.self_attn\.q_a_proj$",
    r"re:(?:language_model\.)?model\.dfly_layers\.\d+\.self_attn\.q_b_proj$",
    r"re:(?:language_model\.)?model\.dfly_layers\.hidden_correction\.down_proj$",
    r"re:(?:language_model\.)?model\.dfly_layers\.hidden_correction\.gate_proj$",
    r"re:(?:language_model\.)?model\.dfly_layers\.hidden_correction\.up_proj$",
]
DFLY_FP8_IGNORE = [
    r"re:(?:language_model\.)?model\.dfly_layers\.confidence_head\.proj$",
    r"re:(?:language_model\.)?model\.dfly_layers\.mtp_fusion\.eh_proj$",
]


def _block_fp8(name, out, inp):
    return {
        f"{name}.weight": _fp8_weight(out, inp),
        f"{name}.weight_scale": torch.rand(-(-out // 128), -(-inp // 128)) + 0.5,
    }


_ROUTED_EXPERT_TARGET = (
    r"re:(?:language_model\.)?model\.layers\.\d+\.mlp\.experts\.\d+\."
    r"(?:gate_proj|up_proj|down_proj)$"
)


def _mxfp4_dfly_quant_config(
    num_hidden_layers, checkpoint_format="mixed-precision", mxfp4_draft_targets=()
):
    """The block FP8 DFly checkpoint with its routed experts moved to a
    weight-only MXFP4 group, the layout vLLM-Hy3-preview df59666f accepts."""
    config = _fp8_quant_config(num_hidden_layers)
    config["format"] = checkpoint_format
    config["ignore"] += DFLY_FP8_IGNORE
    fp8 = config["config_groups"]["block_fp8_w8a8"]
    fp8["format"] = "float-quantized"
    fp8["targets"] = [
        target
        for target in fp8["targets"] + DFLY_FP8_TARGETS
        if target != _ROUTED_EXPERT_TARGET and target not in mxfp4_draft_targets
    ]
    config["config_groups"]["mxfp4_a16"] = {
        "format": "mxfp4-pack-quantized",
        "weights": {
            "num_bits": 4,
            "type": "float",
            "strategy": "group",
            "group_size": 32,
            "symmetric": True,
            "dynamic": False,
            "scale_dtype": "torch.uint8",
        },
        "input_activations": None,
        "output_activations": None,
        "targets": [_ROUTED_EXPERT_TARGET, *mxfp4_draft_targets],
    }
    return config


def _mxfp4_dfly_checkpoint(text):
    """_fp8_checkpoint's names with compressed-tensors MXFP4 routed experts
    (uint8 weight_packed and E8M0 weight_scale) and a block FP8 DFly draft."""

    def quantized(name, out, inp, mtp):
        attn_out = name.endswith((".o_proj", ".linear_gate"))
        if attn_out and not name.startswith("model.layers.0."):
            return _block_fp8(name, out, inp)
        return None

    def experts(name, out, inp):
        return {
            f"{name}.weight_packed": torch.randint(
                1, 256, (out, inp // 2), dtype=torch.uint8
            ),
            f"{name}.weight_scale": _ue8m0(out, inp // 32),
        }

    def draft(name, out, inp):
        return None if name.startswith("mtp_fusion.") else _block_fp8(name, out, inp)

    return _hyv4_checkpoint(text, quantized, experts) | _dfly_checkpoint(text, draft)


def _dfly_checkpoint(text, quantized=lambda name, out, inp: None):
    """Names and dtypes of a bundled DFly draft as the reference loader reads
    them: BF16 (sinks included) except the linears `quantized` stores."""
    torch.manual_seed(1)
    hidden, heads = text.hidden_size, text.num_attention_heads
    tensors = {}

    def dense(name, *shape):
        tensors[f"model.dfly_layers.{name}"] = (torch.randn(shape) * 0.05).bfloat16()

    def linear(name, out, inp):
        stored = quantized(name, out, inp)
        if stored is None:
            return dense(f"{name}.weight", out, inp)
        tensors.update({f"model.dfly_layers.{k}": v for k, v in stored.items()})

    for i in range(text.dfly_num_hidden_layers):
        attn = f"{i}.self_attn"
        qk_head_dim = text.qk_nope_head_dim + text.qk_rope_head_dim
        rope_kv = text.kv_lora_rank + text.qk_rope_head_dim
        linear(f"{attn}.q_a_proj", text.q_lora_rank, hidden)
        linear(f"{attn}.q_b_proj", heads * qk_head_dim, text.q_lora_rank)
        linear(f"{attn}.kv_a_proj_with_mqa", rope_kv, hidden)
        linear(
            f"{attn}.kv_b_proj",
            heads * (text.qk_nope_head_dim + text.v_head_dim),
            text.kv_lora_rank,
        )
        linear(f"{attn}.o_proj", hidden, heads * text.v_head_dim)
        linear(f"{attn}.linear_gate", heads * text.v_head_dim, hidden)
        dense(f"{attn}.q_a_layernorm.weight", text.q_lora_rank)
        dense(f"{attn}.kv_a_layernorm.weight", text.kv_lora_rank)
        dense(f"{attn}.learnable_sink_param", heads)
        dense(f"{i}.input_layernorm.weight", hidden)
        dense(f"{i}.post_attention_layernorm.weight", hidden)
        linear(f"{i}.mlp.gate_proj", text.intermediate_size, hidden)
        linear(f"{i}.mlp.up_proj", text.intermediate_size, hidden)
        linear(f"{i}.mlp.down_proj", hidden, text.intermediate_size)
    correction = text.dfly_hidden_correction_intermediate_size
    for norm in ("final_norm", "mtp_fusion.enorm", "mtp_fusion.hnorm"):
        dense(f"{norm}.weight", hidden)
    for norm in ("hidden_correction.hidden_norm", "hidden_correction.embed_norm"):
        dense(f"{norm}.weight", hidden)
    linear("mtp_fusion.eh_proj", hidden, 2 * hidden)
    dense("mtp_fusion.gate", hidden)
    linear("hidden_correction.gate_proj", correction, 2 * hidden)
    linear("hidden_correction.up_proj", correction, 2 * hidden)
    linear("hidden_correction.down_proj", hidden, correction)
    dense("confidence_head.proj.weight", 1, hidden)
    dense("confidence_head.proj.bias", 1)
    return tensors


def _dfly_model_path(
    tmp_path,
    quantization_config=None,
    *,
    vision_config=VISION_OVERRIDES,
    **text_overrides,
):
    text_config = FP4_TEXT_OVERRIDES | DFLY_KEYS | text_overrides
    config = {
        "architectures": ["HYV4VLForConditionalGeneration"],
        "model_type": "hy_v4_vl",
        "text_config": text_config,
        "vision_config": vision_config,
    }
    if quantization_config is not None:
        config["quantization_config"] = quantization_config
    (tmp_path / "config.json").write_text(json.dumps(config))
    return str(tmp_path)


def _dfly_target_and_draft(path):
    target_config = ModelConfig(path)
    quant_config = _get_quantization_config(target_config, LoadConfig())
    with get_context().override_server_args(disable_shared_experts_fusion=False):
        shared_experts = resolve_shared_experts_config(
            hunyuan_v4_vl.HYV4VLForConditionalGeneration,
            target_config.hf_config,
            quant_config,
        )
        with get_model().override(shared_experts=shared_experts):
            target = hunyuan_v4_vl.HYV4VLForConditionalGeneration(
                target_config.hf_config, quant_config
            )
    draft = HYV4ForCausalLMDFly(target_config.hf_config, quant_config)
    return target, draft, target_config.hf_config.text_config


def test_dfly_checkpoint_loads_target_and_draft_at_checkpoint_precision(
    single_rank, tmp_path
):
    """A checkpoint bundling DFly layers loads both ways: the target skips every
    model.dfly_layers tensor (a dense draft gate_proj used to reach the target's
    stacked-param path and raise), and the draft consumes exactly that set with
    q_a/kv_a and gate/up fused, storing KV in its target layer."""
    target, draft, text = _dfly_target_and_draft(_dfly_model_path(tmp_path))
    checkpoint = _hyv4_checkpoint(
        text,
        lambda *_: None,
        lambda name, out, inp: {f"{name}.weight": torch.randn(out, inp).bfloat16()},
    ) | _dfly_checkpoint(text)

    _poison(target)
    target.load_weights(iter(checkpoint.items()))
    assert _unwritten(target) == []

    _poison(draft)
    draft.load_weights(iter(checkpoint.items()))
    assert _unwritten(draft, skip={"model.embed_tokens.weight", "lm_head.weight"}) == []
    assert not draft.has_own_embed_tokens
    layer = draft.model.layers[0]
    assert layer.self_attn.attn_mqa.layer_id == DFLY_KEYS["dfly_target_kv_layers"][0]
    torch.testing.assert_close(
        layer.mlp.gate_up_proj.weight,
        torch.cat(
            [
                checkpoint["model.dfly_layers.0.mlp.gate_proj.weight"],
                checkpoint["model.dfly_layers.0.mlp.up_proj.weight"],
            ]
        ).float(),
    )


def test_dfly_fp8_draft_loads_block_scales_under_checkpoint_names(
    single_rank, monkeypatch, tmp_path
):
    """Draft linears resolve their compressed-tensors scheme under the
    model.dfly_layers checkpoint names: selected ones load block FP8 with the
    fused q_a/kv_a scales concatenated, an ignored one stays BF16."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (10, 0))
    config = _fp8_quant_config(FP4_TEXT_OVERRIDES["num_hidden_layers"])
    config["config_groups"]["block_fp8_w8a8"]["targets"] += DFLY_FP8_TARGETS
    config["ignore"] += DFLY_FP8_IGNORE
    target, draft, text = _dfly_target_and_draft(_dfly_model_path(tmp_path, config))
    for name, module in draft.model.named_modules():
        if isinstance(module, LinearBase):
            expected = (
                UnquantizedLinearMethod
                if name == "mtp_fusion.eh_proj"
                else CompressedTensorsLinearMethod
            )
            assert isinstance(module.quant_method, expected), name

    def quantized(name, out, inp):
        return None if name.startswith("mtp_fusion.") else _block_fp8(name, out, inp)

    dfly = _dfly_checkpoint(text, quantized)
    checkpoint = _fp8_checkpoint(text) | dfly
    _poison(target)
    target.load_weights(iter(checkpoint.items()))
    assert _unwritten(target) == []
    _poison(draft)
    draft.load_weights(iter(checkpoint.items()))
    assert _unwritten(draft, skip={"model.embed_tokens.weight", "lm_head.weight"}) == []
    attn = "model.dfly_layers.0.self_attn"
    torch.testing.assert_close(
        draft.model.layers[0].self_attn.fused_qkv_a_proj_with_mqa.weight_scale,
        torch.cat(
            [
                dfly[f"{attn}.q_a_proj.weight_scale"],
                dfly[f"{attn}.kv_a_proj_with_mqa.weight_scale"],
            ]
        ),
    )


@pytest.mark.parametrize(
    "checkpoint_format", ["mixed-precision", "mxfp4-pack-quantized"]
)
def test_dfly_mxfp4_checkpoint_loads_mxfp4_experts_and_a_block_fp8_draft(
    single_rank, monkeypatch, tmp_path, checkpoint_format
):
    """A compressed-tensors checkpoint with MXFP4 routed experts serves DFly:
    the target routes its experts to the split-expert scheme that registers the
    loader's per-expert weight_packed names (the fused MXFP4 method leaves them
    unloaded), and the draft loads only block FP8 and BF16."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (10, 0))
    config = _mxfp4_dfly_quant_config(
        FP4_TEXT_OVERRIDES["num_hidden_layers"], checkpoint_format
    )
    check_hyv4_dfly_quant_config(config)
    target, draft, text = _dfly_target_and_draft(_dfly_model_path(tmp_path, config))
    experts = target.language_model.model.layers[1].mlp.experts
    assert isinstance(experts.scheme, CompressedTensorsW4A16Mxfp4MoE)
    checkpoint = _mxfp4_dfly_checkpoint(text)
    _poison(target)
    target.load_weights(iter(checkpoint.items()))
    assert _unwritten(target) == []
    torch.testing.assert_close(
        experts.w2_weight_packed[0],
        checkpoint["model.layers.1.mlp.experts.0.down_proj.weight_packed"],
    )
    _poison(draft)
    draft.load_weights(iter(checkpoint.items()))
    assert _unwritten(draft, skip={"model.embed_tokens.weight", "lm_head.weight"}) == []


def _mxfp4_placement(
    checkpoint_format="mixed-precision", ignore_head=True, ignore=(), **edits
):
    """_mxfp4_dfly_quant_config with the fp8/mxfp4 group targets replaced, or
    extended by an "_extra" key."""
    config = _mxfp4_dfly_quant_config(2, checkpoint_format)
    if not ignore_head:
        config["ignore"] = [
            entry for entry in config["ignore"] if "lm_head" not in entry
        ]
    config["ignore"] += list(ignore)
    groups = config["config_groups"]
    for key, targets in edits.items():
        group = groups["block_fp8_w8a8" if key.startswith("fp8") else "mxfp4_a16"]
        group["targets"] = (
            group["targets"] + targets if key.endswith("_extra") else targets
        )
    return config


_LAYER_1 = r"re:(?:language_model\.)?model\.layers\.1\."
_ROUTER_1 = "language_model.model.layers.1.mlp.gate"
_ALL_BUT_ROUTERS = (
    r"re:(?!(?:language_model\.)?model\.layers\.\d+\.mlp\.gate(?:\.wg)?$)"
)
_ALL_BUT_ROUTERS_AND_EXPERTS = (
    r"re:(?!(?:language_model\.)?model\.layers\.\d+\.mlp\."
    r"(?:experts(?:\.|$)|gate(?:\.wg)?$))"
)
_ONLY_EXPERTS_AND_KV_B_1 = (
    r"re:(?!(?:language_model\.)?model\.layers\."
    r"(?:\d+\.mlp\.experts(?:\.|$)|1\.self_attn\.kv_b_proj$))"
)
_PLAIN_ROUTER_1 = r"re:model\.layers\.1\.mlp\.gate$"
_ALL_BUT_HEAD = r"re:(?!(?:language_model\.)?lm_head$)"


@pytest.mark.parametrize(
    ("config", "selected"),
    [
        (_mxfp4_placement(), None),
        (_mxfp4_placement("mxfp4-pack-quantized"), None),
        (_mxfp4_placement(fp8=["Linear"]), None),
        (_mxfp4_placement(mxfp4=["FusedMoE"]), None),
        (
            _mxfp4_placement(mxfp4_extra=[_LAYER_1 + r"self_attn\.kv_b_proj$"]),
            "language_model.model.layers.1.self_attn.kv_b_proj",
        ),
        (
            _mxfp4_placement(mxfp4_extra=["model.layers.1.self_attn.kv_b_proj"]),
            "language_model.model.layers.1.self_attn.kv_b_proj",
        ),
        (
            _mxfp4_placement(
                mxfp4_extra=[r"re:model\.layers\.1\.self_attn\.kv_b_proj$"]
            ),
            "language_model.model.layers.1.self_attn.kv_b_proj",
        ),
        (
            _mxfp4_placement(
                ignore=[_ONLY_EXPERTS_AND_KV_B_1],
                fp8_extra=["model.layers.1.self_attn.kv_b_proj"],
                mxfp4_extra=["Linear"],
            ),
            None,
        ),
        (
            _mxfp4_placement(
                ignore=[_ONLY_EXPERTS_AND_KV_B_1],
                fp8_extra=[r"re:model\.layers\.1\.self_attn\.kv_b_proj$"],
                mxfp4_extra=["Linear"],
            ),
            "language_model.model.layers.1.self_attn.kv_b_proj",
        ),
        (
            _mxfp4_placement(mxfp4_extra=["Linear"]),
            "language_model.model.layers.0.hc_attn_layer.hc_pre.hc_fn",
        ),
        (
            _mxfp4_placement(mxfp4_extra=[_LAYER_1 + r"self_attn\.q_a_proj$"]),
            "language_model.model.layers.1.self_attn.q_a_proj",
        ),
        (
            _mxfp4_placement(
                mxfp4_extra=[_LAYER_1 + r"mlp\.shared_experts\.(?:gate|up|down)_proj$"]
            ),
            "language_model.model.layers.1.mlp.shared_experts.gate_up_proj",
        ),
        (
            _mxfp4_placement(ignore_head=False, mxfp4_extra=["lm_head"]),
            "language_model.lm_head",
        ),
        (
            _mxfp4_placement(
                ignore_head=False, ignore=[_ALL_BUT_HEAD], mxfp4_extra=["Linear"]
            ),
            "language_model.lm_head",
        ),
        (
            _mxfp4_dfly_quant_config(2, mxfp4_draft_targets=DFLY_FP8_TARGETS[:3]),
            "model.dfly_layers.0.mlp.gate_up_proj",
        ),
        (_mxfp4_placement(mxfp4_extra=["model.layers.1.mlp.gate.wg"]), _ROUTER_1),
        (_mxfp4_placement(mxfp4_extra=[_LAYER_1 + r"mlp\.gate$"]), _ROUTER_1),
        (_mxfp4_placement(mxfp4_extra=[_LAYER_1 + r"mlp\.gate\.wg$"]), _ROUTER_1),
        (
            _mxfp4_placement(ignore=[_ALL_BUT_ROUTERS], mxfp4_extra=["Linear"]),
            _ROUTER_1,
        ),
        (
            _mxfp4_placement(
                ignore=[_ALL_BUT_ROUTERS_AND_EXPERTS, "model.layers.1.mlp.gate.wg"],
                mxfp4_extra=["Linear"],
            ),
            None,
        ),
        (
            _mxfp4_placement(
                ignore=[_ALL_BUT_ROUTERS_AND_EXPERTS, _LAYER_1 + r"mlp\.gate\.wg$"],
                mxfp4_extra=["Linear"],
            ),
            None,
        ),
        (
            _mxfp4_placement(
                ignore=[_ALL_BUT_ROUTERS_AND_EXPERTS, "model.layers.1.mlp.gate.wg"],
                mxfp4_extra=[_PLAIN_ROUTER_1],
            ),
            _ROUTER_1,
        ),
    ],
    ids=[
        "mixed_precision_experts",
        "mxfp4_pack_experts",
        "fp8_generic_linear",
        "mxfp4_fused_moe_class",
        "kv_b_proj_regex",
        "kv_b_proj_name",
        "kv_b_proj_checkpoint_regex",
        "fp8_checkpoint_name_with_mxfp4_linear",
        "fp8_checkpoint_regex_with_mxfp4_linear",
        "mxfp4_generic_linear",
        "fused_q_a_shard",
        "shared_experts",
        "lm_head_suffix",
        "lm_head_generic_linear",
        "draft_mlp",
        "router_name",
        "router_regex",
        "router_wg_regex",
        "router_generic_linear",
        "router_alias_ignored",
        "router_alias_regex_ignored",
        "plain_router_with_alias_ignored",
    ],
)
def test_dfly_mxfp4_placement_is_resolved_before_loading(
    monkeypatch, tmp_path, config, selected
):
    """Weight-only MXFP4 is admitted on routed experts only. Placement is
    resolved with compressed-tensors' own name, module-type and fused-shard
    matching during argument resolution: an MXFP4 kv_b_proj would otherwise
    load and then break MLA absorption, which reads its `.weight`, and an MXFP4
    router would load nothing and leave the routing matrix uninitialized. A
    module is matched under the loader's name against the mapped config and
    under every checkpoint name the loader accepts against the unmapped config;
    an entry naming one checkpoint name decides the module over type targets."""
    _assert_mxfp4_placement(monkeypatch, _dfly_model_path(tmp_path, config), selected)


@pytest.mark.parametrize("n_shared_experts", [1, 0])
def test_dfly_mxfp4_placement_refuses_a_plain_router_checkpoint_name(
    monkeypatch, tmp_path, n_shared_experts
):
    """The VL loader also accepts a router stored as mlp.gate: a checkpoint-only
    regex putting it in the MXFP4 group, next to a generic FP8 fallback, would
    otherwise load nothing into the router, with or without shared experts."""
    config = _mxfp4_placement(
        ignore=[_ALL_BUT_ROUTERS_AND_EXPERTS],
        fp8=["Linear"],
        mxfp4_extra=[_PLAIN_ROUTER_1],
    )
    path = _dfly_model_path(tmp_path, config, n_shared_experts=n_shared_experts)
    _assert_mxfp4_placement(monkeypatch, path, _ROUTER_1)


def test_dfly_mxfp4_placement_checks_every_checkpoint_name_of_a_module():
    """The VL weight mapper is many-to-one; placement must query every checkpoint
    name that loads into a module, not one inverse of the mapping."""
    mapper = hunyuan_v4_vl.HYV4VLForConditionalGeneration.hf_to_sglang_mapper
    expected = {
        _ROUTER_1: [
            _ROUTER_1,
            f"{_ROUTER_1}.wg",
            "model.layers.1.mlp.gate",
            "model.layers.1.mlp.gate.wg",
        ],
        "language_model.model.layers.1.self_attn.kv_b_proj": [
            "language_model.model.layers.1.self_attn.kv_b_proj",
            "model.layers.1.self_attn.kv_b_proj",
        ],
        "language_model.lm_head": ["language_model.lm_head", "lm_head"],
    }
    for name, names in expected.items():
        assert hunyuan_v4_dfly._checkpoint_names(name, mapper=mapper) == names


def _assert_mxfp4_placement(monkeypatch, path, selected):
    loads = []
    monkeypatch.setattr(
        DeepseekV2WeightLoaderMixin,
        "do_load_weights",
        lambda self, weights: loads.append(self),
    )
    server_args = _dfly_server_args(path)
    if selected is None:
        handle_speculative_decoding(server_args)
        assert resolution_result(server_args, "speculative_algorithm") == "DFLY"
    else:
        named = rf"routed experts.*(?<![\w.]){re.escape(selected)}(?![\w.])"
        with pytest.raises(ValueError, match=named):
            handle_speculative_decoding(server_args)
    assert loads == []


@pytest.mark.parametrize("n_shared_experts", [1, 0])
def test_dfly_mxfp4_placement_covers_every_quantizable_projection(
    single_rank, tmp_path, n_shared_experts
):
    """The placement check enumerates projections by name: any target or draft
    module owning a weight matrix that it misses could select MXFP4 unchecked.
    Routed experts are the admitted placement, embeddings are not projections,
    and the draft binds the target's embedding and head."""
    target, draft, text = _dfly_target_and_draft(
        _dfly_model_path(tmp_path, n_shared_experts=n_shared_experts)
    )
    built_target = {
        (name, module_cls)
        for name, module_cls in _weight_matrix_owners(target)
        if name.startswith("language_model.")
        and not name.endswith((".embed_tokens", ".mlp.experts"))
    }
    assert built_target == set(
        hunyuan_v4_dfly._hyv4_target_projections(text, prefix="language_model.")
    )
    built_draft = {
        (name, module_cls)
        for name, module_cls in _weight_matrix_owners(draft)
        if name not in ("model.embed_tokens", "lm_head")
    }
    assert built_draft == set(hunyuan_v4_dfly._dfly_draft_projections(draft.spec))


def _weight_matrix_owners(model):
    return {
        (name, type(module))
        for name, module in model.named_modules()
        if any(param.dim() >= 2 for param in module.parameters(recurse=False))
    }


def test_dfly_draft_refuses_an_mxfp4_draft_linear(single_rank, monkeypatch, tmp_path):
    """MXFP4 may quantize target modules only: a draft linear that the
    checkpoint puts in its MXFP4 group fails at draft construction instead of
    running weight-only FP4."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (10, 0))
    config = _mxfp4_dfly_quant_config(
        FP4_TEXT_OVERRIDES["num_hidden_layers"],
        mxfp4_draft_targets=DFLY_FP8_TARGETS[:3],
    )
    check_hyv4_dfly_quant_config(config)
    target_config = ModelConfig(_dfly_model_path(tmp_path, config))
    quant_config = _get_quantization_config(target_config, LoadConfig())
    with pytest.raises(ValueError, match="mlp.gate_up_proj must be BF16 or block FP8"):
        HYV4ForCausalLMDFly(target_config.hf_config, quant_config)


def _post_proj_norm_model(kind, declared):
    num_layers = 3 if kind == "dfly" else 2
    options = (
        TEXT_OVERRIDES
        | DFLY_KEYS
        | dict(
            num_hidden_layers=num_layers,
            mlp_layer_types=["dense"] * num_layers,
            indexer_types=["full"] * num_layers,
            n_routed_experts=0,
            n_shared_experts=0,
            num_nextn_predict_layers=0,
            dfly_num_hidden_layers=2,
            dfly_target_index_layer=[1, 2],
            dfly_target_kv_layers=[1, 2],
            architectures=["HYV4ForCausalLM"],
        )
    )
    field = "dfly_attn_post_proj_norm" if kind == "dfly" else "attn_post_proj_norm"
    if kind == "dfly":
        options["attn_post_proj_norm"] = True
    if declared != "absent":
        options[field] = declared
    config = HYV4Config(**options)
    if kind == "dfly":
        model = HYV4ForCausalLMDFly(config)
        checkpoint = _dfly_checkpoint(config)
        prefix = "model.dfly_layers"
    else:
        model = hunyuan_v4.HYV4ForCausalLM(config)
        checkpoint = {
            name: weight
            for name, weight in _hyv4_checkpoint(
                config, lambda *args: None, lambda *args: {}
            ).items()
            if not name.startswith(("vit.", "model.mtp_layers."))
        }
        prefix = "model.layers"
    norms = {
        f"{prefix}.{i}.self_attn.post_proj_norm.weight": (
            torch.linspace(0.25, 1.5, config.hidden_size) * (i + 1)
        ).bfloat16()
        for i in range(2)
    }
    return model, checkpoint, norms


@pytest.mark.parametrize("kind", ["base", "dfly"])
@pytest.mark.parametrize("declared", ["absent", None, False, True])
@pytest.mark.parametrize("present", [False, True])
def test_post_proj_norm_checkpoint_contract(single_rank, kind, declared, present):
    """Checkpoint-driven DFly norms must never be dropped or partly initialized."""
    model, checkpoint, norms = _post_proj_norm_model(kind, declared)
    if present:
        checkpoint.update(norms)
    infer = kind == "dfly" and declared in ("absent", None)
    if present and not (declared is True or infer):
        with pytest.raises(ValueError, match="Unexpected.*post.proj"):
            model.load_weights(iter(checkpoint.items()))
        return
    if not present and declared is True:
        with pytest.raises(ValueError, match="missing.*(post.proj|draft weights)"):
            model.load_weights(iter(checkpoint.items()))
        return
    _poison(model)
    model.load_weights(iter(checkpoint.items()))
    for i, layer in enumerate(model.model.layers):
        norm = layer.self_attn.post_proj_norm
        if present:
            torch.testing.assert_close(norm.weight, list(norms.values())[i].float())
        else:
            assert norm is None
            assert layer.attn_boundary.declaration.output_transform is None
    skip = {"model.embed_tokens.weight", "lm_head.weight"} if kind == "dfly" else set()
    assert _unwritten(model, skip=skip) == []


@pytest.mark.parametrize(
    "kind,declared", [("base", True), ("dfly", True), ("dfly", None)]
)
def test_post_proj_norm_rejects_partial_layers(single_rank, kind, declared):
    """One norm missing from a multilayer checkpoint must fail before serving."""
    model, checkpoint, norms = _post_proj_norm_model(kind, declared)
    checkpoint.update(dict(list(norms.items())[:1]))
    with pytest.raises(ValueError, match="(missing post-projection|Inconsistent)"):
        model.load_weights(iter(checkpoint.items()))


@pytest.mark.parametrize("kind", ["base", "dfly"])
@pytest.mark.parametrize("bad", ["duplicate", "shape", "unknown_layer"])
def test_post_proj_norm_strict_loader(single_rank, kind, bad):
    """Norm weights must use an exact layer name and a full hidden-width vector."""
    model, checkpoint, norms = _post_proj_norm_model(kind, True)
    name, weight = next(iter(norms.items()))
    checkpoint.update(norms)
    entries = list(checkpoint.items())
    if bad == "duplicate":
        entries.append((name, weight))
        error = "Duplicate"
    elif bad == "shape":
        entries = [(n, w[:1] if n == name else w) for n, w in entries]
        error = "shape|size"
    else:
        entries.append((name.replace(".0.", ".2."), weight))
        error = "Unexpected"
    with pytest.raises(ValueError, match=error):
        model.load_weights(iter(entries))


def test_dfly_post_proj_norm_rejects_wrong_dtype(single_rank):
    model, checkpoint, norms = _post_proj_norm_model("dfly", None)
    checkpoint.update({name: weight.float() for name, weight in norms.items()})
    with pytest.raises(ValueError, match="must be torch.bfloat16"):
        model.load_weights(iter(checkpoint.items()))


@pytest.mark.parametrize("value", [0, 1, "true"])
def test_dfly_post_proj_norm_rejects_non_boolean_config(value):
    with pytest.raises(ValueError, match="must be boolean or None"):
        resolve_hyv4_dfly_spec(_dfly_text_config(dfly_attn_post_proj_norm=value))


@pytest.mark.parametrize(
    "kind,declared,present",
    [
        ("base", True, True),
        ("dfly", True, True),
        ("dfly", None, True),
        ("base", "absent", False),
        ("base", False, False),
        ("dfly", None, False),
        ("dfly", False, False),
    ],
)
def test_post_proj_norm_forward_oracle(
    single_rank, monkeypatch, kind, declared, present
):
    """Normalize the projected, gated contribution before iHC or residual addition."""
    model, checkpoint, norms = _post_proj_norm_model(kind, declared)
    if present:
        checkpoint.update(norms)
    model.load_weights(iter(checkpoint.items()))
    layer = model.model.layers[0]
    attn = layer.self_attn
    torch.manual_seed(17)
    hidden = torch.randn(3, model.config.hidden_size)
    context = torch.randn(3, attn.num_local_heads * attn.v_head_dim)
    topk = torch.zeros(3, 8, dtype=torch.int32)

    def sparse_attention(positions, x, forward_batch, zero_allocator, **kwargs):
        return forward_mla._finish_absorb_output(
            attn, context, attn.prepare_attention_output_gate(x), topk
        )

    monkeypatch.setattr(attn, "forward", sparse_attention)
    ffn_inputs = []
    handle = layer.mlp.register_forward_pre_hook(
        lambda module, args: ffn_inputs.append(args[0].clone())
    )
    batch = _three_token_decode_batch()
    hunyuan_v4.residual_batch.start(batch)
    try:
        layer(batch.positions, hidden.clone(), batch, None, topk)
    finally:
        handle.remove()

    if kind == "base":
        residual = layer.hc_attn_layer.prepare_input(hidden)
        x, post, residual = layer.hc_attn_layer.pre(residual, layer.input_layernorm)
    else:
        residual = hidden
        x = layer.input_layernorm(hidden)
    gated = context * torch.sigmoid(nn.functional.linear(x, attn.linear_gate.weight))
    projected = nn.functional.linear(gated, attn.o_proj.weight)
    normalized = nn.functional.rms_norm(
        projected.float(),
        (model.config.hidden_size,),
        next(iter(norms.values())).float(),
        model.config.rms_norm_eps,
    )
    if not present:
        normalized = projected
    if kind == "base":
        updated = layer.hc_attn_layer.post(normalized, residual, post)
        expected, _, _ = layer.hc_mlp_layer.pre(updated, layer.post_attention_layernorm)
    else:
        expected = layer.post_attention_layernorm(normalized + residual)
    torch.testing.assert_close(ffn_inputs[0], expected, rtol=2e-5, atol=2e-6)


def test_dfly_draft_ffn_width_follows_dfly_intermediate_size(single_rank, tmp_path):
    """The draft FFN is dfly_intermediate_size wide, by default the target's
    intermediate_size. A checkpoint FFN of another width must be declared:
    otherwise the row and column loaders would keep a slice of it silently."""
    width = 2 * FP4_TEXT_OVERRIDES["intermediate_size"]
    (tmp_path / "declared").mkdir()
    _, draft, text = _dfly_target_and_draft(
        _dfly_model_path(tmp_path / "declared", dfly_intermediate_size=width)
    )
    wide = HYV4Config(**text.to_dict() | {"intermediate_size": width})
    checkpoint = _dfly_checkpoint(wide)
    _poison(draft)
    draft.load_weights(iter(checkpoint.items()))
    assert _unwritten(draft, skip={"model.embed_tokens.weight", "lm_head.weight"}) == []
    assert draft.model.layers[0].mlp.down_proj.weight.shape == (text.hidden_size, width)

    _, default_draft, _ = _dfly_target_and_draft(_dfly_model_path(tmp_path))
    with pytest.raises(ValueError, match="dfly_intermediate_size must declare"):
        default_draft.load_weights(iter(checkpoint.items()))


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (
            lambda c: c.update(
                {"model.dfly_layers.0.self_attn.indexer.wk.weight": torch.zeros(8, 64)}
            ),
            "Unexpected HYV4 DFly checkpoint weight",
        ),
        (
            lambda c: c.pop("model.dfly_layers.hidden_correction.down_proj.weight"),
            "missing draft weights",
        ),
        (
            lambda c: c.update(
                {"model.dfly_layers.0.self_attn.o_proj.weight": _fp8_weight(64, 64)}
            ),
            "must be torch.bfloat16",
        ),
    ],
    ids=["unexpected", "missing", "dtype"],
)
def test_dfly_draft_load_rejects_a_checkpoint_off_the_manifest(
    single_rank, tmp_path, edit, match
):
    _, draft, text = _dfly_target_and_draft(_dfly_model_path(tmp_path))
    checkpoint = _dfly_checkpoint(text)
    edit(checkpoint)
    with pytest.raises(ValueError, match=match):
        draft.load_weights(iter(checkpoint.items()))


def _dfly_text_config(**dfly_overrides):
    # Six layers: indexer_types full, full, shared, shared, shared, full. No
    # architectures: the indexer layout is read from the config itself.
    keys = dict(
        dfly_num_hidden_layers=2,
        dfly_block_size=4,
        dfly_mask_token_id=250,
        dfly_target_index_layer=[1, 5],
        dfly_target_kv_layers=[3, 4],
        dfly_context_local_window=64,
    )
    return HYV4Config(
        **TEXT_OVERRIDES | {"num_hidden_layers": 6} | keys | dfly_overrides
    )


@pytest.mark.parametrize("nested", [False, True], ids=["top_level", "text_config"])
def test_dfly_spec_follows_the_reference_contract(nested):
    """dfly_* keys resolve from either config level; the draft width counts the
    anchor query and the block-aligned top-k holds history, window and block,
    while the target config stays untouched."""
    text = _dfly_text_config().to_dict()
    dfly = {k: text.pop(k) for k in list(text) if k.startswith("dfly_")}
    vl = HYV4VLConfig(
        text_config=text | dfly if nested else text,
        vision_config=VISION_OVERRIDES,
        **({} if nested else dfly),
    )
    spec = resolve_hyv4_dfly_spec(vl.text_config)
    assert spec.num_speculative_tokens == 3
    assert spec.target_index_layers == (1, 5)
    assert spec.target_kv_layers == (3, 4)
    # index_topk 8 + window 64 + 4 queries, rounded up to the 128-index block.
    assert spec.draft_index_topk == 128
    assert (
        resolve_hyv4_dfly_spec(
            vl.text_config, speculative_num_draft_tokens=8
        ).num_speculative_tokens
        == 7
    )
    scalar = _dfly_text_config(dfly_target_index_layer=5)
    assert resolve_hyv4_dfly_spec(scalar).target_index_layers == (5, 5)

    draft = build_hyv4_dfly_draft_config(vl.text_config, spec)
    assert draft.num_hidden_layers == 2 and not draft.enable_ihc
    assert all(dsa_layer_skips_topk(draft, i) for i in range(2))
    assert vl.text_config.num_hidden_layers == 6
    assert vl.text_config.index_topk == 8 and vl.text_config.enable_ihc
    with pytest.raises(ValueError, match="bundled only with HYV4 checkpoints"):
        resolve_hyv4_dfly_spec(SimpleNamespace(**_dfly_text_config().to_dict()))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"dfly_target_index_layer": [1, 2]}, "indexer_types entry is 'full'"),
        ({"dfly_target_kv_layers": [0, 4]}, r"target layer in \[1, 6\)"),
        ({"dfly_target_kv_layers": [3, 3]}, "distinct target KV layer"),
        ({"dfly_mask_token_id": 256}, "outside the vocabulary"),
        ({"dfly_block_size": 18}, "dfly_block_size"),
        ({"dfly_mlp_type": "moe"}, "dfly_mlp_type"),
        ({"dfly_enable_ihc": True}, "dfly_enable_ihc"),
        ({"dfly_target_index_topk": 16}, "equal to the target index_topk"),
        ({"index_topk": 0, "dfly_target_index_topk": 0}, "positive int"),
        ({"index_topk": -1, "dfly_target_index_topk": -1}, "positive int"),
        ({"dfly_block_size": 1}, r"dfly_block_size \(anchor plus drafts\)"),
    ],
    ids=[
        "shared_index_layer",
        "layer_zero",
        "duplicate_kv_layer",
        "mask_outside_vocab",
        "block_too_wide",
        "moe_draft",
        "ihc_draft",
        "topk_mismatch",
        "zero_topk",
        "negative_topk",
        "block_without_drafts",
    ],
)
def test_dfly_spec_rejects_configs_outside_the_reference_contract(overrides, match):
    with pytest.raises(ValueError, match=match):
        resolve_hyv4_dfly_spec(_dfly_text_config(**overrides))


def _mxfp4_group_edit(edit):
    config = _mxfp4_dfly_quant_config(2)
    edit(config["config_groups"])
    return config


@pytest.mark.parametrize(
    "config",
    [
        FP8_QUANT_CONFIG | {"kv_cache_scheme": {"num_bits": 8, "type": "float"}},
        FP4_QUANT_CONFIG,
        json.loads(json.dumps(FP8_QUANT_CONFIG).replace('"block"', '"channel"')),
        json.loads(
            json.dumps(FP8_QUANT_CONFIG).replace('"dynamic": true', '"dynamic": false')
        ),
        _mxfp4_group_edit(
            lambda groups: groups["mxfp4_a16"].update(
                input_activations=groups["block_fp8_w8a8"]["input_activations"]
            )
        ),
        _mxfp4_group_edit(lambda groups: groups["block_fp8_w8a8"].pop("format")),
    ],
    ids=[
        "kv_cache_scheme",
        "fp8_mxfp4_experts",
        "channel_weights",
        "static_activations",
        "mxfp4_activations",
        "group_format_unresolved",
    ],
)
def test_dfly_quantization_follows_the_reference_contract(config):
    """Checkpoint groups are block W8A8 FP8 or weight-only MXFP4, each in its
    own format when the top level is mixed-precision (vLLM-Hy3-preview
    df59666f); FP8 MXFP4-expert checkpoints stay outside the contract."""
    check_hyv4_dfly_quant_config(None)
    check_hyv4_dfly_quant_config(FP8_QUANT_CONFIG)
    for checkpoint_format in ("mixed-precision", "mxfp4-pack-quantized"):
        check_hyv4_dfly_quant_config(_mxfp4_dfly_quant_config(2, checkpoint_format))
    with pytest.raises(ValueError, match="compressed-tensors W8A8 FP8"):
        check_hyv4_dfly_quant_config(config)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"attn_dp_size": 2}, "attention DP size=2"),
        ({"pp_size": 2}, "pp_size=2"),
        ({"dcp_size": 2}, "dcp_size=2"),
        ({"disaggregation_mode": "decode"}, "disaggregation_mode"),
        ({"kv_cache_dtype": "fp8_e4m3"}, "kv_cache_dtype='fp8_e4m3'"),
        ({"draft_kv_cache_dtype": "fp8_e4m3"}, "draft kv_cache_dtype"),
        ({"draft_quantization": "fp8"}, "draft quantization"),
        ({"speculative_eagle_topk": 4}, "speculative_eagle_topk=4"),
        ({"tp_size": 8}, "num_attention_heads=4 not divisible"),
        (
            {"prefill_cuda_graph_backend": Backend.FULL},
            "'full' prefill CUDA graph backend",
        ),
        (
            {"decode_cuda_graph_backend": Backend.BREAKABLE},
            "'breakable' decode CUDA graph backend",
        ),
        ({"enable_mixed_chunk": True}, "mixed chunked prefill"),
    ],
    ids=[
        "dp_attention",
        "pp",
        "dcp",
        "pd",
        "fp8_kv",
        "draft_kv_dtype",
        "draft_quant",
        "topk",
        "tp_divisibility",
        "prefill_graph",
        "breakable_decode_graph",
        "mixed_chunk",
    ],
)
def test_dfly_serving_envelope_fails_loud(override, match):
    text = _dfly_text_config()
    spec = resolve_hyv4_dfly_spec(text)
    supported = dict(
        spec=spec,
        draft_config=build_hyv4_dfly_draft_config(text, spec),
        dtype="auto",
        tp_size=2,
        attn_dp_size=1,
        pp_size=1,
        attn_cp_size=1,
        dcp_size=1,
        disaggregation_mode="null",
        enable_lora=False,
        enable_two_batch_overlap=False,
        kv_cache_dtype="auto",
        draft_kv_cache_dtype=None,
        draft_quantization=None,
        draft_attention_backend=None,
        speculative_eagle_topk=None,
        decode_cuda_graph_backend=Backend.FULL,
        prefill_cuda_graph_backend=Backend.DISABLED,
        enable_mixed_chunk=False,
    )
    check_hyv4_dfly_serving(**supported)
    check_hyv4_dfly_serving(**supported | {"decode_cuda_graph_backend": "disabled"})
    check_hyv4_dfly_serving(
        **supported | {"prefill_cuda_graph_backend": Backend.BREAKABLE}
    )
    # BF16 spellings name the same KV cache.
    check_hyv4_dfly_serving(**supported | {"draft_kv_cache_dtype": "bfloat16"})
    with pytest.raises(ValueError, match=match):
        check_hyv4_dfly_serving(**supported | override)


def _reference_dfly_rows(target_rows, anchors, query_width, window, active):
    """The reference's logical index rows (build_reuse_indices)."""
    bs, topk = target_rows.shape
    rows = torch.full((bs, query_width, topk + window + query_width), -1)
    starts = anchors[:, None] - window
    history = torch.where((target_rows >= 0) & (target_rows < starts), target_rows, -1)
    rows[:, :, :topk] = history[:, None, :]
    if window:
        rows[:, :, topk : topk + window] = (starts + torch.arange(window)).clamp(
            min=-1
        )[:, None, :]
    offsets = torch.arange(query_width)
    causal = offsets[None, :] <= offsets[:, None]
    rows[:, :, topk + window :] = torch.where(
        causal[None], (anchors[:, None] + offsets)[:, None, :], -1
    )
    return rows.masked_fill(~active[:, None, None], -1)


@pytest.mark.parametrize("physical", [False, True], ids=["positions", "slots"])
@pytest.mark.parametrize("window", [0, 5])
def test_dfly_sparse_indices_match_the_reference_rows(window, physical):
    """Sparse rows equal the reference's logical rows, mapped through
    req_to_token when the DSA top-k is fused, including window/history overlap,
    padding, a window starting before the prompt, prefix slots shared by two
    requests, and an inactive request."""
    torch.manual_seed(0)
    max_context_len, topk, query_width = 40, 6, 4
    slots = torch.randperm(1000)[: 4 * max_context_len].view(4, max_context_len)
    # Request 1's window covers prefix slots that are history for request 0.
    slots[1, :16] = slots[0, :16]
    req_to_token = torch.full((6, max_context_len), -7)
    req_pool_indices = torch.tensor([3, 0, 5, 2])
    req_to_token[req_pool_indices] = slots
    committed = torch.tensor([20, 18, 3, 37])
    positions, active = dfly_query_positions(
        anchors=committed,
        sampled=torch.ones(4, dtype=torch.bool),
        query_width=query_width,
        max_context_len=max_context_len,
    )
    assert active.tolist() == [True, True, True, False]
    target_rows = torch.stack(
        [torch.randint(0, max(int(p), 1), (topk,)) for p in committed]
    )
    target_rows[0, :5] = torch.tensor([19, 18, 17, 14, 13])
    target_rows[1, -2:] = -1

    def ids(logical):
        return dfly_position_ids(
            positions=logical,
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            physical=physical,
        )

    rows = build_dfly_sparse_indices(
        target_rows=ids(target_rows),
        window_ids=ids(
            dfly_window_positions(anchors=committed, context_local_window=window)
        ),
        block_ids=ids(positions),
        active=active,
        index_topk=128,
    ).view(4, query_width, 128)
    logical = _reference_dfly_rows(target_rows, committed, query_width, window, active)
    expected = torch.where(
        logical >= 0,
        slots[torch.arange(4)[:, None, None], logical.clamp(min=0)],
        -1,
    )
    width = logical.shape[-1]
    assert torch.equal(rows[..., :width].long(), expected if physical else logical)
    assert (rows[..., width:] == -1).all()


def _verify_dsa_metadata(*, anchors, query_width, index_topk):
    """DSA metadata of a TARGET_VERIFY block as the backend plans it: each row
    may scan the logical sequence length, clipped to the top-k budget."""
    expanded = (
        (anchors[:, None] + 1 + torch.arange(query_width)).flatten().to(torch.int32)
    )
    lengths = compute_dsa_seqlens(expanded, dsa_index_topk=index_topk)
    zeros = torch.zeros(1, dtype=torch.int32)
    return DSAMetadata(
        page_size=1,
        cache_seqlens_int32=zeros,
        max_seq_len_q=1,
        max_seq_len_k=1,
        cu_seqlens_q=zeros,
        cu_seqlens_k=zeros,
        page_table_1=None,
        real_page_table=zeros,
        dsa_cache_seqlens_int32=lengths,
        dsa_cu_seqlens_q=zeros,
        dsa_cu_seqlens_k=zeros,
        dsa_extend_seq_lens_list=[1] * expanded.numel(),
        dsa_seqlens_expanded=expanded,
    )


@pytest.mark.parametrize("budget", ["draft", "target"])
def test_dfly_draft_rows_reach_the_sparse_kernel_whole(
    single_rank, tmp_path, monkeypatch, budget
):
    """The FlashMLA sparse kernel scans a row only up to its top-k length,
    which the backend derives from the logical sequence length clipped to its
    top-k budget. External DFly rows keep block entries past a short context
    and window/block entries past the target's budget, so the draft forward
    must hand the kernel lengths covering every valid entry."""
    _, draft, text = _dfly_target_and_draft(_dfly_model_path(tmp_path))
    spec = draft.spec
    width, query_width = spec.draft_index_topk, spec.query_width
    anchors = torch.tensor([3, 40, 400])
    positions, active = dfly_query_positions(
        anchors=anchors,
        sampled=torch.ones(3, dtype=torch.bool),
        query_width=query_width,
        max_context_len=402,
    )
    assert active.tolist() == [True, True, False]
    # A short request has fewer committed positions than the indexer selects.
    target_rows = torch.full((3, spec.target_index_topk), -1)
    for row, anchor in zip(target_rows, anchors.tolist()):
        history = torch.randperm(anchor)[: spec.target_index_topk]
        row[: history.numel()] = history
    no_ids = dict(req_to_token=None, req_pool_indices=None, physical=False)
    rows = build_dfly_sparse_indices(
        target_rows=target_rows,
        window_ids=dfly_position_ids(
            positions=dfly_window_positions(
                anchors=anchors, context_local_window=spec.context_local_window
            ),
            **no_ids,
        ),
        block_ids=dfly_position_ids(positions=positions, **no_ids),
        active=active,
        index_topk=width,
    )
    index_topk = width if budget == "draft" else spec.target_index_topk
    metadata = _verify_dsa_metadata(
        anchors=anchors, query_width=query_width, index_topk=index_topk
    )
    calls = []

    def kernel(**kwargs):
        calls.append(kwargs)
        return kwargs["q"][..., :512], None, None

    kernel_owner = SimpleNamespace(
        device_sm_major=9,
        dsa_index_topk=index_topk,
        _call_flash_mla_sparse_fwd=kernel,
    )

    def layers(input_ids, positions, forward_batch, target_hidden, topk_indices):
        # The draft attention reaches the sparse kernel with the backend's metadata.
        for page_table in topk_indices:
            DeepseekSparseAttnBackend._forward_flashmla_sparse(
                kernel_owner,
                q_all=torch.zeros(input_ids.shape[0], 64, 576),
                kv_cache=torch.zeros(1, 1, 576),
                v_head_dim=512,
                page_table_1=page_table,
                sm_scale=1.0,
                topk_length=get_attn_backend().forward_metadata.dsa_cache_seqlens_int32,
            )
        return torch.zeros(input_ids.shape[0], text.hidden_size)

    monkeypatch.setattr(draft.model, "forward", layers)
    draft.stage_block(
        target_hidden=torch.zeros(3, text.hidden_size), topk_indices=[rows]
    )
    with forward_context(
        ForwardContext(attn_backend=SimpleNamespace(forward_metadata=metadata))
    ):
        draft.forward(
            positions.flatten(),
            positions.flatten(),
            SimpleNamespace(mrope_positions=None),
        )
    (call,) = calls
    indices = call["indices"][:, 0]
    scanned = torch.arange(width)[None, :] < call["topk_length"][:, None]
    assert not (indices.ge(0) & ~scanned).any()
    assert indices.ge(0).sum() == rows.ge(0).sum()
    assert (indices.view(3, query_width, width)[2] == -1).all()


def test_dfly_block_writes_and_reads_only_for_active_requests():
    """A request without a sampled bonus token or whose block crosses the
    context writes only the discard slot and reads no KV, and its verify rows
    stay inside the RoPE table; an active request, also one whose block ends
    exactly at the context, writes exactly its verify slots."""
    max_context_len, query_width = 32, 4
    verify_slots = torch.arange(101, 101 + 4 * query_width).view(4, query_width)
    block = plan_dfly_block(
        bonus_tokens=torch.tensor([7, 8, 9, 6]),
        anchors=torch.tensor([10, 10, 30, 28]),
        sampled=torch.tensor([True, False, True, True]),
        verify_slots=verify_slots,
        mask_token_id=250,
        max_context_len=max_context_len,
    )
    assert block.active.tolist() == [True, False, False, True]
    writes = block.write_slots.view(4, query_width)
    assert torch.equal(writes[block.active], verify_slots[block.active])
    assert (writes[~block.active] == DFLY_DISCARD_SLOT).all()
    input_ids = block.input_ids.view(4, query_width)
    assert input_ids[:, 0].tolist() == [7, 0, 0, 6]
    assert (input_ids[:, 1:] == 250).all()
    assert block.positions.max() < max_context_len
    assert block.verify_positions.max() < max_context_len
    assert block.verify_positions[[0, 3]].tolist() == [
        [10, 11, 12, 13],
        [28, 29, 30, 31],
    ]
    rows = build_dfly_sparse_indices(
        target_rows=torch.zeros(4, 3, dtype=torch.int64),
        window_ids=torch.empty(4, 0, dtype=torch.int64),
        block_ids=block.positions,
        active=block.active,
        index_topk=128,
    ).view(4, query_width, 128)
    assert (rows[~block.active] == -1).all()


def test_dfly_next_step_state_comes_from_the_last_committed_row():
    """After a verify, each request's next anchor is its committed length and
    its target hidden and top-k come from the row of its last committed token
    (the reference's query_start_loc[1:] - num_rejected - 1). An inactive
    request's -1 drafts commit only the bonus token even against a target that
    predicts the clamped verify input."""
    query_width = 4
    bonus = torch.tensor([5, 5, 5])
    drafts = torch.tensor([[1, 2, 3], [1, 9, 3], [-1, -1, -1]])
    candidates, verify_input = split_dfly_drafts_for_verify(
        bonus_tokens=bonus, draft_tokens=drafts
    )
    assert verify_input.min() >= 0
    target_predict = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4], [0, 0, 0, 4]])
    accept_len, next_bonus = compute_dflash_correct_drafts_and_bonus(
        candidates=candidates, target_predict=target_predict
    )
    out_tokens, commit_lens = commit_accept(candidates, accept_len, next_bonus)
    assert commit_lens.tolist() == [4, 2, 1]
    assert out_tokens[2, 0] == 0 and next_bonus.tolist() == [4, 2, 0]

    source_rows = dfly_source_rows(
        row_starts=torch.arange(3) * query_width, commit_lens=commit_lens
    )
    query_start_loc = torch.arange(4) * query_width
    assert torch.equal(
        source_rows, query_start_loc[1:] - (query_width - commit_lens) - 1
    )
    anchors = torch.tensor([10, 20, 30])
    verify_positions = (anchors[:, None] + torch.arange(query_width)).flatten()
    assert torch.equal(verify_positions[source_rows] + 1, anchors + commit_lens)

    capture = DFlyProducerCapture(
        producer_layers=(5, 9), max_rows=12, topk=1, device="cpu"
    )
    topk = {i: torch.arange(12).view(12, 1) * 10 + i for i in (5, 7, 9)}
    capture.record(5, topk[5] + 1000)
    capture.begin(num_rows=12)
    for layer_id in (5, 7, 9):
        capture.record(layer_id, topk[layer_id])
    rows = capture.finish()[:, source_rows]
    assert rows[:, :, 0].tolist() == [[35, 55, 85], [39, 59, 89]]
    # A producer that skipped this forward keeps last forward's rows at a
    # valid address; that staleness must still refuse the capture.
    capture.begin(num_rows=2, keep_rows=torch.tensor([2, 7]))
    capture.record(5, topk[5])
    with pytest.raises(RuntimeError, match=r"skipped index layers \[9\]"):
        capture.finish()


def test_dfly_request_state_follows_requests_through_batch_reshaping():
    """The state a decode step drafts from is what each request's last prefill
    or verify committed, wherever filter, merge, retraction and pool-row reuse
    moved the request: it equals the per-batch-row state carried through the
    same reshaping, including draft layers that share one producer. A reused
    row before its prefill, a length that moved without a commit, or a cache
    flush leaves the request without live state; the padding row is never
    written."""
    gen = torch.Generator().manual_seed(0)
    hidden_size, topk = 4, 3
    index_layers = (5, 2, 5)
    producers = dfly_producer_layers(index_layers)
    pool = ReqToTokenPool(
        size=5, max_context_len=8, device="cpu", enable_memory_saver=False
    )
    state = VLDFlyRequestState(
        num_rows=pool.req_to_token.shape[0],
        hidden_size=hidden_size,
        hidden_dtype=torch.float32,
        num_producers=len(producers),
        topk=topk,
        device="cpu",
    )

    def commit(rows, lens):
        """One forward's committed state; returns it in the batch-carried layout."""
        rows = torch.tensor(rows)
        hidden = torch.randn(len(rows), hidden_size, generator=gen)
        produced = {
            layer: torch.randint(0, 1 << 20, (len(rows), topk), generator=gen).to(
                torch.int32
            )
            for layer in producers
        }
        state.write(
            rows_cpu=rows,
            rows=rows,
            generations=pool.req_generation[rows],
            hidden=hidden,
            topk=torch.stack([produced[layer] for layer in producers]),
            committed_len=torch.tensor(lens),
        )
        return DFlashDraftInputV2(
            topk_p=torch.empty(len(rows), 0),
            topk_index=torch.stack(
                [produced[layer] for layer in index_layers], dim=1
            ).flatten(1),
            bonus_tokens=torch.zeros(len(rows), dtype=torch.int64),
            new_seq_lens=torch.tensor(lens),
            hidden_states=hidden,
        )

    def live(rows, lens):
        rows = torch.tensor(rows)
        return state.live(
            rows_cpu=rows,
            rows=rows,
            generations=pool.req_generation[rows],
            committed_len=torch.tensor(lens),
        )

    def check(batch, carried):
        rows = torch.tensor([row for row, _ in batch])
        assert live(rows.tolist(), [n for _, n in batch]).all()
        assert torch.equal(state.hidden[rows], carried.hidden_states)
        per_layer = carried.topk_index.view(len(batch), len(index_layers), topk)
        for layer, plane in enumerate(dfly_draft_topk_planes(index_layers)):
            assert torch.equal(state.topk[plane, rows], per_layer[:, layer])

    batch = [(row, 3) for row in pool.alloc_rows(3)]
    carried = commit([r for r, _ in batch], [n for _, n in batch])
    for step in range(40):
        check(batch, carried)
        op = step % 4
        if op == 0:
            batch = [(row, n + 1 + step % 3) for row, n in batch]
            carried = commit([r for r, _ in batch], [n for _, n in batch])
        elif op == 1 and len(batch) > 1:
            keep = torch.randperm(len(batch), generator=gen)[: len(batch) - 1]
            dropped = set(range(len(batch))) - set(keep.tolist())
            pool.free_rows([batch[i][0] for i in dropped])
            batch = [batch[i] for i in keep.tolist()]
            carried.filter_batch(keep)
        else:
            joining = [(row, 2 + step % 5) for row in pool.alloc_rows(1) or []]
            if joining:
                carried.merge_batch(
                    commit([r for r, _ in joining], [n for _, n in joining])
                )
                batch = batch + joining
    check(batch, carried)

    freed, length = batch[0]
    pool.free_rows([freed])
    (reused,) = pool.alloc_rows(1)
    assert reused == freed and not live([reused], [length]).any()
    commit([reused], [length])
    assert live([reused], [length]).all()
    assert not live([reused], [length + 1]).any()
    state.invalidate_all()
    assert not live([row for row, _ in batch], [n for _, n in batch]).any()
    with pytest.raises(RuntimeError, match="padding row"):
        commit([0], [1])
    assert state.committed_len[0] == -1 and not state.hidden[0].any()


def _dfly_extend_batch(seq_lens):
    num_tokens = sum(seq_lens)
    lens = torch.tensor(seq_lens, dtype=torch.int32)
    return ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=len(seq_lens),
        input_ids=torch.arange(1, num_tokens + 1),
        positions=torch.cat([torch.arange(n) for n in seq_lens]),
        out_cache_loc=torch.zeros(num_tokens, dtype=torch.long),
        req_pool_indices=torch.arange(len(seq_lens)),
        seq_lens=lens,
        seq_lens_cpu=lens,
        seq_lens_sum=num_tokens,
        extend_seq_lens=lens,
        extend_seq_lens_cpu=list(seq_lens),
        extend_prefix_lens_cpu=[0] * len(seq_lens),
        spec_info=None,
    )


_DSA_BACKEND_STUB = SimpleNamespace(
    use_mha=False,
    prefill_attention_backend_str="dsa",
    decode_attention_backend_str="dsa",
)


def _stub_dfly_producer_model(monkeypatch):
    """A text target whose index layers emit top-k from their hidden states
    into ``produced`` and whose attention core records what it returns."""
    config = _dfly_text_config(architectures=["HYV4ForCausalLM"])
    spec = resolve_hyv4_dfly_spec(config)
    model = hunyuan_v4.HYV4ForCausalLM(config)
    _fill_deterministic(model, std=0.2)
    produced, received, returned = {}, {}, {}

    def stub_prepare(attn):
        def prepare(
            positions,
            hidden_states,
            forward_batch,
            zero_allocator,
            llama_4_scaling,
            prev_topk_indices,
        ):
            if attn.should_run_indexer(prev_topk_indices):
                topk = (hidden_states[:, :8].abs() * 100).to(torch.int32)
                produced[attn.layer_id] = topk + attn.layer_id
            else:
                received[attn.layer_id] = prev_topk_indices
            return (
                None,
                None,
                None,
                None,
                forward_batch,
                zero_allocator,
                positions,
                produced.get(attn.layer_id, prev_topk_indices),
                None,
            )

        return prepare

    for layer in model.model.layers:
        monkeypatch.setattr(
            layer.self_attn, "forward_absorb_prepare", stub_prepare(layer.self_attn)
        )
        monkeypatch.setattr(
            layer.mlp, "forward", lambda x, *args, **kwargs: torch.tanh(x) * 0.25
        )

    def core(
        self,
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        forward_batch,
        zero_allocator,
        positions,
        topk_indices,
        llama_4_scaling,
        *args,
    ):
        attn_out = torch.zeros(
            positions.shape[0], self.num_local_heads * self.v_head_dim
        )
        output = forward_mla._finish_absorb_output(self, attn_out, None, topk_indices)
        returned[self.layer_id] = output[1]
        return output

    monkeypatch.setattr(
        forward_mla.DeepseekMLAForwardMixin, "forward_absorb_core", core
    )
    return model, spec, produced, received, returned


def _run_dfly_prefill(model, capture, seq_lens, keep_rows):
    batch = _dfly_extend_batch(seq_lens)
    capture.begin(num_rows=len(keep_rows), keep_rows=keep_rows)
    with forward_context(ForwardContext(attn_backend=_DSA_BACKEND_STUB)):
        model.model(batch.input_ids, batch.positions, batch)
    return capture.finish().clone()


def test_dfly_capture_reads_each_index_layer_at_its_producer(single_rank, monkeypatch):
    """Regression: the last selected index layer has no sharing successor, so
    its attention output drops its top-k; the capture must still hold the rows
    its indexer produced, while shared layers keep receiving the producer's."""
    model, spec, produced, received, returned = _stub_dfly_producer_model(monkeypatch)
    last = model.config.num_hidden_layers - 1
    assert spec.target_index_layers == (1, last)
    producers = dfly_producer_layers(spec.target_index_layers)
    capture = DFlyProducerCapture(
        producer_layers=producers, max_rows=2, topk=8, device="cpu"
    )
    _attach_topk_capture(model=model, capture=capture, layer_ids=producers)
    keep_rows = torch.tensor([2, 4])
    rows = _run_dfly_prefill(model, capture, [3, 2], keep_rows)

    assert returned[last] is None
    assert sorted(received) == [2, 3, 4]
    assert all(topk is produced[1] for topk in received.values())
    assert rows.dtype == torch.int32
    for plane, layer_id in enumerate(producers):
        assert torch.equal(rows[plane], produced[layer_id][keep_rows])


def test_dfly_breakable_prefill_replay_keeps_the_live_forward_rows(
    single_rank, monkeypatch
):
    """A breakable prefill graph replays without Python, and each index layer's
    top-k keeps its capture-time buffer. Every replay must still keep the last
    prefill rows of the forward begun before it: a replay equals the eager
    forward with the same top-k, follows a new request layout, and interleaves
    with eager forwards of batches above the capture ceiling."""
    model, spec, produced, _, _ = _stub_dfly_producer_model(monkeypatch)
    producers = dfly_producer_layers(spec.target_index_layers)
    capture = DFlyProducerCapture(
        producer_layers=producers, max_rows=3, topk=8, device="cpu"
    )
    _attach_topk_capture(model=model, capture=capture, layer_ids=producers)
    eager_calls = []
    recorder = SimpleNamespace(
        end_current_segment=lambda: None,
        begin_next_segment=lambda: None,
        register_eager_call=eager_calls.append,
        retain_for_replay=lambda value: value,
    )
    capture_batch = _dfly_extend_batch([5])
    with (
        forward_context(ForwardContext(attn_backend=_DSA_BACKEND_STUB)),
        use_breakable_capture_recorder(recorder),
    ):
        model.model(capture_batch.input_ids, capture_batch.positions, capture_batch)
    graph_topk = dict(produced)
    assert len(eager_calls) == len(producers)

    def replay(keep_rows, topk):
        capture.begin(num_rows=len(keep_rows), keep_rows=keep_rows)
        for layer_id in producers:
            graph_topk[layer_id].copy_(topk[layer_id])
        for call in eager_calls:
            call()
        return capture.finish().clone()

    keep_rows = torch.tensor([2, 4])
    eager_rows = _run_dfly_prefill(model, capture, [3, 2], keep_rows)
    assert torch.equal(replay(keep_rows, produced), eager_rows)

    for seq_lens in ([1, 4], [1, 1, 3]):
        keep_rows = torch.tensor(seq_lens).cumsum(0) - 1
        topk = {layer_id: graph_topk[layer_id] + 7 for layer_id in producers}
        rows = replay(keep_rows, topk)
        for plane, layer_id in enumerate(producers):
            assert torch.equal(rows[plane], topk[layer_id][keep_rows])

    keep_rows = torch.tensor([3, 6])
    rows = _run_dfly_prefill(model, capture, [4, 3], keep_rows)
    for plane, layer_id in enumerate(producers):
        assert torch.equal(rows[plane], produced[layer_id][keep_rows])
    assert len(eager_calls) == len(producers)
    keep_rows = torch.tensor([4])
    topk = {layer_id: graph_topk[layer_id] - 3 for layer_id in producers}
    rows = replay(keep_rows, topk)
    for plane, layer_id in enumerate(producers):
        assert torch.equal(rows[plane], topk[layer_id][keep_rows])


class _RecordingDFlyExecution:
    def __init__(self, events):
        self.events = events

    def draft_into_verify(self, *, input_ids, forward_batch):
        self.events.append(("draft", input_ids.clone()))
        input_ids.view(-1, 4)[:, 1:] = 7


@pytest.mark.parametrize("top_level", ["vl", "text"])
def test_dfly_verify_drafts_once_before_the_target_embeds(
    single_rank, monkeypatch, top_level
):
    """A verify forward drafts exactly once, at the top-level target and before
    any embedding, so the media/text embedding and the transformer read the
    drafted ids; the VL wrapper's language model never drafts again, and a
    prefill never drafts."""
    if top_level == "vl":
        model = hunyuan_v4_vl.HYV4VLForConditionalGeneration(_tiny_vl_config())
        text_model = model.language_model
    else:
        model = hunyuan_v4.HYV4ForCausalLM(HYV4Config(**TEXT_OVERRIDES))
        text_model = model
    _fill_deterministic(model)
    events = []
    model.dfly_execution = _RecordingDFlyExecution(events)
    real_prepare_mm_inputs = hunyuan_v4_vl.prepare_mm_inputs

    def recording_prepare_mm_inputs(**kwargs):
        events.append(("embed", kwargs["input_ids"].clone()))
        return real_prepare_mm_inputs(**kwargs)

    def body(
        input_ids, positions, forward_batch, input_embeds=None, pp_proxy_tensors=None
    ):
        events.append(("body", input_ids.clone()))
        return torch.zeros(input_ids.shape[0], TEXT_OVERRIDES["hidden_size"])

    monkeypatch.setattr(hunyuan_v4_vl, "prepare_mm_inputs", recording_prepare_mm_inputs)
    monkeypatch.setattr(text_model.model, "forward", body)
    monkeypatch.setattr(text_model, "logits_processor", _RecordingLogitsProcessor())
    placeholders = torch.tensor([5, 0, 0, 0, 6, 0, 0, 0])
    drafted = torch.tensor([5, 7, 7, 7, 6, 7, 7, 7])
    embeds = ["embed"] if top_level == "vl" else []
    for mode, ids in (
        (ForwardMode.EXTEND, placeholders),
        (ForwardMode.TARGET_VERIFY, drafted),
    ):
        events.clear()
        positions = torch.arange(8)
        batch = _mrope_batch(torch.stack([positions] * 3), False, mode=mode)
        model(placeholders.clone(), positions, batch)
        drafts = ["draft"] if mode.is_target_verify() else []
        assert [name for name, _ in events] == drafts + embeds + ["body"]
        assert all(torch.equal(seen, ids) for name, seen in events if name != "draft")


def test_dfly_hicache_plans_no_draft_pool():
    """DFly draft KV lives in target layers; the draft runner's pool is only a
    loader placeholder, so HiCache must neither pack nor sidecar it."""
    target = SimpleNamespace(spec_algorithm=SpeculativeAlgorithm.DFLY)
    placeholder = SimpleNamespace(
        token_to_kv_pool=object(),
        runner_ids=SimpleNamespace(is_fused_draft=False),
    )
    worker = SimpleNamespace(
        target_worker=SimpleNamespace(model_runner=target),
        draft_worker=SimpleNamespace(
            model_runner=placeholder, draft_runners=[placeholder]
        ),
    )
    worker._draft_model_runners = lambda: BaseSpecWorker._draft_model_runners(worker)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            base_spec_worker,
            "get_memory",
            lambda: SimpleNamespace(enable_hierarchical_cache=True),
        )
        plan = BaseSpecWorker._build_hicache_draft_plan(worker)
    assert plan == HiCacheDraftPlan()
    assert target.mtp_draft_device_pools == ()
    assert BaseSpecWorker.primary_draft_kv_pool.fget(worker) is None


def test_dfly_autotune_dummy_verify_runs_the_target_alone(
    single_rank, monkeypatch, tmp_path
):
    """Regression: the target runner's FlashInfer autotune warmup runs a
    TARGET_VERIFY dummy forward after the DFly worker installs its draft
    adapter, outside any DFly decode step, so startup failed on the unstaged
    adapter. That dummy must run the target alone and leave request state and
    draft candidates untouched; any other verify without a staged DFly step
    still fails."""
    target, draft, text = _dfly_target_and_draft(_dfly_model_path(tmp_path))
    width = draft.spec.query_width
    body_calls, logits_calls = _stub_text_body(monkeypatch, target)
    producers = dfly_producer_layers(draft.spec.target_index_layers)
    state = VLDFlyRequestState(
        num_rows=3,
        hidden_size=text.hidden_size,
        hidden_dtype=torch.float32,
        num_producers=len(producers),
        topk=draft.spec.target_index_topk,
        device="cpu",
    )
    live_row = torch.tensor([1])
    state.write(
        rows_cpu=live_row,
        rows=live_row,
        generations=torch.zeros(1, dtype=torch.int64),
        hidden=torch.randn(1, text.hidden_size),
        topk=torch.randint(0, 9, (len(producers), 1, draft.spec.target_index_topk)).to(
            torch.int32
        ),
        committed_len=torch.tensor([9]),
        mrope_delta=torch.tensor([4]),
    )
    before = [
        tensor.clone()
        for tensor in (state.hidden, state.topk, state.committed_len, state.mrope_delta)
    ]
    req_to_token = torch.arange(3 * 32, dtype=torch.int32).view(3, 32)

    worker = object.__new__(DFlyWorkerV2)
    worker.draft_model, worker.spec = draft, draft.spec
    worker._state = state
    worker._draft_topk_planes = dfly_draft_topk_planes(draft.spec.target_index_layers)
    worker._physical_ids = False
    worker._max_context_len = 32
    worker._draft_block_spec_info = make_draft_block_spec_info(
        draft_token_num=width, device="cpu"
    )
    worker.device = "cpu"
    worker._draft_in_target_forward = True
    worker.model_runner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        runner_ids=SimpleNamespace(bind_own_plan=lambda forward_batch: None),
        model=target,
    )
    worker._init_execution()
    assert target.dfly_execution is worker._execution
    candidates = worker._execution.candidates(2).clone()

    model_runner = SimpleNamespace(
        device="cpu",
        is_generation=True,
        is_draft_worker=False,
        spec_algorithm=SpeculativeAlgorithm.DFLY,
        decode_num_tokens_per_req=lambda num_draft_tokens=None: width,
        model=target,
        lora_manager=None,
        canary_manager=None,
        model_config=SimpleNamespace(
            hidden_size=text.hidden_size,
            vocab_size=text.vocab_size,
            dtype=torch.float32,
            is_encoder_decoder=False,
            hf_config=text,
        ),
        attn_backend=SimpleNamespace(
            get_cuda_graph_seq_len_fill_value=lambda: 1,
            init_forward_metadata=lambda forward_batch: None,
        ),
        ngram_embedding_manager=SimpleNamespace(enabled=False, table=None),
        get_pp_proxy_topk_size=lambda: None,
        get_pp_proxy_residual_num_blocks=lambda: None,
        runner_ids=SimpleNamespace(bind_runner_slots=lambda forward_batch: None),
        prepare_dummy_forward_batch=lambda forward_batch: forward_batch,
    )
    runner = object.__new__(eager_runner.EagerRunner)
    runner.model_runner = model_runner
    runner._eager_max_bs = 2
    buffers, batch_size = runner._autotune_buffers()

    @contextlib.contextmanager
    def flag_only_autotune(model_runner, *, run_lm_head):
        # FlashInfer's tuner is unavailable on CPU; keep only the dummy-run flag.
        with autotune_dummy_run_mode(run_lm_head=run_lm_head):
            yield

    monkeypatch.setattr(
        flashinfer_autotune, "flashinfer_autotune_context", flag_only_autotune
    )
    with get_context().override_server_args(speculative_num_draft_tokens=width):
        runner._flashinfer_autotune(buffers=buffers, batch_size=batch_size)
        assert len(body_calls) == len(logits_calls) == 1
        assert logits_calls[0][0].shape == (batch_size * width,)
        for tensor, saved in zip(
            (state.hidden, state.topk, state.committed_len, state.mrope_delta),
            before,
            strict=True,
        ):
            assert torch.equal(tensor, saved)
        assert torch.equal(worker._execution.candidates(2), candidates)

        with pytest.raises(RuntimeError, match="no staged draft step"):
            runner._dummy_run(batch_size=batch_size, buffers=buffers)
    assert len(body_calls) == 1


class _AgreeingLogitsProcessor(nn.Module):
    """Target logits that confirm the first ``agree[r]`` drafts of the request
    in pool row r."""

    def __init__(self, agree, query_width):
        super().__init__()
        self.agree, self.query_width = agree, query_width

    def forward(self, input_ids, hidden_states, lm_head, forward_batch, *_):
        width = self.query_width
        logits = (hidden_states @ lm_head.weight.T).view(
            -1, width, lm_head.weight.shape[0]
        )
        drafts = input_ids.view(-1, width).roll(-1, dims=1)
        agree = self.agree[forward_batch.req_pool_indices]
        confirm = torch.arange(width)[None, :] < agree[:, None]
        logits = logits.scatter_add(2, drafts[..., None], confirm[..., None] * 100.0)
        return LogitsProcessorOutput(
            next_token_logits=logits.flatten(0, 1), hidden_states=hidden_states
        )


_GRAPH_UNSAFE_OPS = frozenset(
    {
        torch.ops.aten._local_scalar_dense.default,
        torch.ops.aten.is_nonzero.default,
        torch.ops.aten.nonzero.default,
        torch.ops.aten.masked_select.default,
    }
)
_GRAPH_INDEX_OPS = frozenset(
    {
        torch.ops.aten.index.Tensor,
        torch.ops.aten.index_put.default,
        torch.ops.aten.index_put_.default,
    }
)


class _CpuGraph(TorchDispatchMode):
    """CPU stand-in for a captured CUDA graph: records the tensor ops of one
    forward and replays them on the same storage without running its Python
    again. Ops a CUDA graph cannot capture (host reads of device values,
    data-dependent shapes) fail the capture."""

    def __init__(self):
        super().__init__()
        self._ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        boolean_index = func in _GRAPH_INDEX_OPS and any(
            isinstance(index, torch.Tensor) and index.dtype == torch.bool
            for index in args[1]
        )
        if func in _GRAPH_UNSAFE_OPS or boolean_index:
            raise AssertionError(f"{func} cannot run inside a CUDA graph")
        out = func(*args, **kwargs)
        self._ops.append((func, args, kwargs, out))
        return out

    def replay(self):
        """Replay the recorded ops; returns the lookup from a tensor the
        capture produced to the one this replay produced."""
        replayed = {}

        def current(value):
            if isinstance(value, torch.Tensor):
                return replayed.get(id(value), value)
            return value

        with torch.no_grad():
            for func, args, kwargs, out in self._ops:
                fresh = func(*tree_map(current, args), **tree_map(current, kwargs))
                for old, new in zip(tree_flatten(out)[0], tree_flatten(fresh)[0]):
                    if isinstance(old, torch.Tensor):
                        replayed[id(old)] = new
        return current


def _dfly_parity_models():
    # A window shorter than the prompts keeps history top-k rows in the draft rows.
    config = _dfly_text_config(
        architectures=["HYV4ForCausalLM"],
        dfly_context_local_window=4,
        rope_parameters={
            "rope_theta": 10000.0,
            "rope_type": "default",
            "mrope_section": [2, 1, 1],
        },
    )
    target = hunyuan_v4.HYV4ForCausalLM(config)
    draft = HYV4ForCausalLMDFly(config)
    assert draft.is_mrope_enabled
    _fill_deterministic(target, seed=4, std=0.2)
    _fill_deterministic(draft, seed=3, std=0.2)
    draft.set_embed_and_head(*target.get_embed_and_head())
    return target, draft


_ALL_FOUR_REQUESTS = ((0, 1, 2, 3),) * 3


def _dfly_decode_steps(
    monkeypatch,
    *,
    target,
    draft,
    draft_in_target,
    physical,
    schedule=_ALL_FOUR_REQUESTS,
    graph_buckets=(),
    recapture_before=(),
    events=None,
    overlap=False,
):
    """Run DFly decode steps through the worker with the attention cores
    replaced by a reference that reads and writes a shared KV dict; returns
    each step's observable results.

    ``schedule`` lists each step's batch as indices into four requests. With
    ``graph_buckets`` the target runner captures one CPU graph per bucket at
    startup and replays every batch that fits one, padding it like the decode
    graph runner; ``recapture_before`` recaptures them before those steps.
    ``events`` collects each verify's HiCache wait and forward."""
    spec = draft.spec
    width = spec.query_width
    max_context_len = 40
    rows = torch.tensor([1, 2, 3, 4])
    seq_lens = torch.tensor([20, 37, 12, 25])
    gen = torch.Generator().manual_seed(7)
    req_to_token = torch.zeros(5, 48, dtype=torch.int32)
    req_to_token[1:] = (
        torch.arange(1, 5)[:, None] * 100 + torch.arange(48)[None, :] + 1
    ).to(torch.int32)
    pool = SimpleNamespace(
        req_to_token=req_to_token, req_generation=torch.zeros(5, dtype=torch.int64)
    )
    kv = SimpleNamespace(
        kv={layer: torch.randn(500, 16, generator=gen) for layer in range(6)}
    )
    mix = torch.randn(16, TEXT_OVERRIDES["hidden_size"], generator=gen) * 0.5

    def rotary(positions, dim):
        return torch.sin(positions[:, None] * 0.37 * torch.arange(1, dim + 1))

    producers = dfly_producer_layers(spec.target_index_layers)
    state = VLDFlyRequestState(
        num_rows=5,
        hidden_size=TEXT_OVERRIDES["hidden_size"],
        hidden_dtype=torch.float32,
        num_producers=len(producers),
        topk=spec.target_index_topk,
        device="cpu",
    )
    history = torch.stack(
        [
            torch.randint(0, 12, (4, spec.target_index_topk), generator=gen)
            for _ in producers
        ]
    )
    if physical:
        history = req_to_token[rows].gather(1, history.flatten(1).view(4, -1).long())
        history = history.view(4, len(producers), -1).transpose(0, 1)
    state.write(
        rows_cpu=rows,
        rows=rows,
        generations=pool.req_generation[rows],
        hidden=torch.randn(4, TEXT_OVERRIDES["hidden_size"], generator=gen),
        topk=history.to(torch.int32),
        committed_len=seq_lens,
        mrope_delta=torch.tensor([0, 5, -3, 11]),
    )
    pool.req_generation[3] += 1
    capture = DFlyProducerCapture(
        producer_layers=producers,
        max_rows=4 * width,
        topk=spec.target_index_topk,
        device="cpu",
    )
    target_backend = SimpleNamespace(token_to_kv_pool=kv, forward_metadata=None)
    draft_backend = SimpleNamespace(token_to_kv_pool=None, forward_metadata=None)
    observed, draft_positions = [], []

    def draft_layer(self, positions, hidden_states, forward_batch, zero_alloc, topk):
        if self.self_attn.attn_mqa.layer_id == spec.target_kv_layers[0]:
            draft_positions.append(positions.clone())
        backend = get_attn_backend()
        cache = backend.token_to_kv_pool.kv[self.self_attn.attn_mqa.layer_id]
        lengths = backend.forward_metadata.dsa_cache_seqlens_int32
        ids = torch.where(
            torch.arange(topk.shape[1])[None, :] < lengths[:, None], topk.long(), -1
        )
        if not physical:
            req_rows = forward_batch.req_pool_indices.repeat_interleave(width)
            ids = torch.where(
                ids >= 0, req_to_token[req_rows[:, None], ids.clamp(min=0)].long(), -1
            )
        cache[forward_batch.out_cache_loc] = hidden_states[:, :16] + rotary(
            positions, 16
        )
        read = torch.where(ids[..., None] >= 0, cache[ids.clamp(min=0)], 0.0).sum(1)
        # Every block row also reads its anchor row, which carries the target hidden.
        anchors = hidden_states.view(-1, width, hidden_states.shape[1])[:, :1]
        return (
            hidden_states
            + anchors.expand(-1, width, -1).flatten(0, 1)
            + torch.tanh(read @ mix)
            + rotary(positions, hidden_states.shape[1])
        )

    def target_body(
        input_ids, positions, forward_batch, input_embeds=None, pp_proxy_tensors=None
    ):
        assert get_attn_backend().forward_metadata is target_backend.planned
        observed.append(input_ids.clone())
        hidden = target.model.embed_tokens(input_ids)
        for layer, cache in kv.kv.items():
            cache[forward_batch.out_cache_loc] = (
                hidden[:, :16] * (layer + 1) + positions[:, None] * 0.02
            )
        slots = req_to_token[forward_batch.req_pool_indices.repeat_interleave(width)]
        reach = torch.arange(slots.shape[1])[None, :] <= positions[:, None]
        read = torch.where(reach[..., None], kv.kv[3][slots.long()], 0.0).sum(1)
        hidden = hidden + torch.tanh(read @ mix / (positions[:, None] + 1))
        for layer in producers:
            ids = (hidden[:, : spec.target_index_topk].abs() * 1000).long() % (
                positions[:, None] + 1
            )
            if physical:
                ids = slots.long().gather(1, ids)
            capture.record(layer, ids.to(torch.int32))
        return hidden

    def plan_verify(seq_lens):
        return _verify_dsa_metadata(
            anchors=seq_lens, query_width=width, index_topk=spec.target_index_topk
        )

    model_runner = SimpleNamespace(
        req_to_token_pool=pool,
        token_to_kv_pool=kv,
        runner_ids=SimpleNamespace(bind_own_plan=lambda forward_batch: None),
        model=target,
        capture_head_hooks=[],
        capture_tail_hooks=[],
    )
    registry = (
        build_decode_registry(
            device=torch.device("cpu"),
            max_bs=max(graph_buckets),
            max_num_token=max(graph_buckets) * width,
            seq_len_fill_value=1,
            cache_loc_dtype=torch.int64,
            share_pool=False,
        )
        if graph_buckets
        else None
    )

    def counter_event(index):
        return SimpleNamespace(
            wait=lambda layer: events.append(("restore_wait", index, layer))
        )

    class TargetWorker:
        graphs = {}
        hicache_layer_transfer_counter = SimpleNamespace(
            num_layers=6, events=[counter_event(i) for i in range(2)]
        )

        def forward_batch_generation(self, batch, forward_batch, **kwargs):
            # The model runner plans attention at the shape it runs.
            assert kwargs == {"is_verify": True}
            assert forward_batch.needs_forward_metadata_init()
            if events is not None:
                events.append(("verify", forward_batch.batch_size))
            raw_bs = forward_batch.batch_size
            bucket = min((b for b in graph_buckets if b >= raw_bs), default=None)
            if bucket is None:
                return self._eager(forward_batch)
            return self._replay(forward_batch, raw_bs=raw_bs, bucket=bucket)

        def _eager(self, forward_batch):
            loaded = dataclasses.replace(
                forward_batch, input_ids=forward_batch.input_ids.clone()
            )
            target_backend.planned = target_backend.forward_metadata = plan_verify(
                loaded.seq_lens
            )
            with forward_context(
                ForwardContext(attn_backend=target_backend), forward_batch=loaded
            ):
                output = target(loaded.input_ids, loaded.positions, loaded)
            assert target_backend.forward_metadata is target_backend.planned
            self.last = dict(
                verify_ids=observed[-1],
                draft_positions=draft_positions[-1],
                hidden=output.hidden_states.clone(),
                padded_candidates=torch.empty(0, width, dtype=torch.int64),
            )
            return SimpleNamespace(logits_output=output, can_run_cuda_graph=False)

        def capture(self):
            for bucket in sorted(graph_buckets, reverse=True):
                self.graphs[bucket] = self._capture(bucket)

        def _capture(self, bucket):
            registry.get_slot("seq_lens").buffer.fill_(1)
            registry.get_slot("seq_lens_cpu").buffer.fill_(1)
            for name in ("req_pool_indices", "out_cache_loc", "positions"):
                registry.get_slot(name).buffer.zero_()
            num_tokens = bucket * width

            def slot(name):
                return registry.get_slot(name).slice_for(bucket, num_tokens)

            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.TARGET_VERIFY,
                batch_size=bucket,
                input_ids=slot("input_ids"),
                req_pool_indices=slot("req_pool_indices"),
                seq_lens=slot("seq_lens"),
                seq_lens_cpu=slot("seq_lens_cpu"),
                out_cache_loc=slot("out_cache_loc"),
                positions=slot("positions"),
                seq_lens_sum=bucket,
                spec_info=DFlashVerifyInput(
                    draft_token=None,
                    positions=None,
                    draft_token_num=width,
                    custom_mask=None,
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                ),
                capture_hidden_mode=CaptureHiddenMode.FULL,
            )
            metadata = plan_verify(forward_batch.seq_lens)

            def run_once():
                target_backend.planned = target_backend.forward_metadata = metadata
                for hook in model_runner.capture_head_hooks:
                    hook(None, forward_batch)
                with forward_context(
                    ForwardContext(attn_backend=target_backend),
                    forward_batch=forward_batch,
                ):
                    out = target(
                        forward_batch.input_ids, forward_batch.positions, forward_batch
                    )
                for hook in model_runner.capture_tail_hooks:
                    hook(None, out, forward_batch, num_tokens)
                return out

            before = [tensor.clone() for tensor in _dfly_state_tensors(state)]
            run_once()
            graph = _CpuGraph()
            with graph:
                output = run_once()
            # Capture runs on dummy rows: it drafts nothing and commits nothing.
            assert all(
                torch.equal(tensor, saved)
                for tensor, saved in zip(_dfly_state_tensors(state), before)
            )
            if draft_in_target:
                assert (worker._execution.candidates(bucket)[:, 1:] == -1).all()
            return SimpleNamespace(
                graph=graph,
                output=output,
                metadata=metadata,
                observed=observed[-1],
                draft_positions=draft_positions[-1],
            )

        def _replay(self, forward_batch, *, raw_bs, bucket):
            captured = self.graphs[bucket]
            raw_tokens, num_tokens = raw_bs * width, bucket * width
            registry.fill_from(
                forward_batch,
                raw_bs=raw_bs,
                padded_bs=bucket,
                raw_num_tokens=raw_tokens,
                padded_num_tokens=num_tokens,
            )
            planned = plan_verify(registry.get_slot("seq_lens").slice_for(bucket, 0))
            for name in ("dsa_cache_seqlens_int32", "dsa_seqlens_expanded"):
                getattr(captured.metadata, name).copy_(getattr(planned, name))
            replayed = captured.graph.replay()
            hidden = replayed(captured.output.hidden_states)[:raw_tokens]
            self.last = dict(
                verify_ids=replayed(captured.observed)[:raw_tokens],
                draft_positions=replayed(captured.draft_positions)[:raw_tokens],
                hidden=hidden.clone(),
                padded_candidates=worker._execution._candidates[raw_bs:bucket].clone(),
            )
            logits_output = LogitsProcessorOutput(
                next_token_logits=replayed(captured.output.next_token_logits)[
                    :raw_tokens
                ],
                hidden_states=hidden,
            )
            return SimpleNamespace(logits_output=logits_output, can_run_cuda_graph=True)

    class DraftRunner:
        attn_backend = draft_backend

        def forward(self, forward_batch):
            draft_backend.forward_metadata = _verify_dsa_metadata(
                anchors=forward_batch.seq_lens,
                query_width=width,
                index_topk=spec.draft_index_topk,
            )
            with forward_context(
                ForwardContext(attn_backend=draft_backend), forward_batch=forward_batch
            ):
                return SimpleNamespace(
                    logits_output=draft(
                        forward_batch.input_ids, forward_batch.positions, forward_batch
                    )
                )

    def prepare_for_verify(self, batch, target_worker, *, defer_metadata_init=False):
        assert defer_metadata_init
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=batch.batch_size(),
            input_ids=self.draft_token,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            out_cache_loc=batch.out_cache_loc,
            seq_lens_sum=batch.seq_lens_sum,
            seq_lens_cpu=batch.seq_lens_cpu,
            positions=self.positions,
            spec_info=self,
            capture_hidden_mode=self.capture_hidden_mode,
        )
        return forward_batch, False

    def cache_locs(*, req_pool_indices, req_to_token, start_offset, **_):
        offsets = start_offset[:, None] + torch.arange(width)[None, :]
        return req_to_token[req_pool_indices[:, None], offsets].flatten().long()

    monkeypatch.setattr(hunyuan_v4_dfly.HYV4DFlyDecoderLayer, "forward", draft_layer)
    monkeypatch.setattr(target.model, "forward", target_body)
    monkeypatch.setattr(
        target,
        "logits_processor",
        _AgreeingLogitsProcessor(torch.tensor([0, width - 1, 0, 0, 2]), width),
    )
    monkeypatch.setattr(DFlashVerifyInput, "prepare_for_verify", prepare_for_verify)
    monkeypatch.setattr(dfly_worker_v2, "assign_extend_cache_locs_func", cache_locs)

    worker = object.__new__(DFlyWorkerV2)
    BaseSpecWorker.__init__(worker)
    target_worker = TargetWorker()
    worker._target_worker = target_worker
    worker.model_runner = model_runner
    worker.device = "cpu"
    worker.draft_model, worker.draft_model_runner = draft, DraftRunner()
    worker.spec, worker.block_size = spec, width
    worker._tp_sync = SpecTpSync(SimpleNamespace(world_size=1, rank_in_group=0))
    worker._producer_layers = producers
    worker._draft_topk_planes = dfly_draft_topk_planes(spec.target_index_layers)
    worker._state, worker._capture = state, capture
    worker._draft_block_spec_info = make_draft_block_spec_info(
        draft_token_num=width, device="cpu"
    )
    worker._max_context_len, worker._physical_ids = max_context_len, physical
    worker._draft_in_target_forward = draft_in_target
    target.dfly_execution = None
    target_params = dict(target.named_parameters())
    worker._init_execution()
    worker._install_capture_hooks()
    # Weight loading and updates must not reach the draft through the target.
    assert dict(target.named_parameters()).keys() == target_params.keys()
    _attach_topk_capture(model=target, capture=capture, layer_ids=producers)
    if graph_buckets:
        target_worker.capture()

    bonus = torch.tensor([11, 12, 13, 14])
    if overlap:
        future_map = FutureMap(
            device=torch.device("cpu"),
            spec_algo=SpeculativeAlgorithm.DFLY,
            req_to_token_pool=pool,
            needs_cpu_seq_lens=False,
        )
        future_map.publish(rows, seq_lens)
        future_map.stash(rows, RelayPayload(bonus_tokens=bonus))
    steps = []
    for index, selected in enumerate(schedule):
        if index in recapture_before:
            target_worker.capture()
        selected = torch.tensor(selected)
        step_lens = seq_lens[selected]
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            is_extend_in_batch=False,
            spec_info=_make_draft_input(
                bonus_tokens=bonus[selected], new_seq_lens=step_lens
            ),
            batch_size=lambda n=len(selected): n,
            seq_lens=step_lens,
            seq_lens_cpu=step_lens.clone(),
            seq_lens_sum=int(step_lens.sum()),
            req_pool_indices=rows[selected],
            req_pool_indices_cpu=rows[selected],
            hicache_consumer_index=-1 if events is None else index % 2,
            out_cache_loc=None,
            enable_overlap=overlap,
            spec_algorithm=SpeculativeAlgorithm.DFLY,
            prefill_input_ids_cpu=None,
            input_ids=None,
        )
        if overlap:
            batch.spec_info.future_indices = batch.req_pool_indices
            batch.seq_lens = torch.full_like(step_lens, -1)
            batch.spec_info.bonus_tokens.fill_(-1)
            future_map.resolve_seq_lens_cpu(batch)
            resolve_forward_inputs(batch, future_map)
            assert torch.equal(batch.seq_lens, step_lens)
            assert torch.equal(batch.spec_info.bonus_tokens, bonus[selected])

        published = []

        def on_publish(new_seq_lens):
            assert torch.equal(
                state.committed_len[batch.req_pool_indices], new_seq_lens
            )
            published.append(new_seq_lens.clone())
            future_map.publish(batch.req_pool_indices, new_seq_lens)

        planes = capture._rows
        result = worker.forward_batch_generation(
            batch, on_publish=on_publish if overlap else None
        )
        if overlap:
            assert len(published) == 1
            assert torch.equal(published[0], result.new_seq_lens)
            future_map.relay(
                batch.req_pool_indices,
                RelayPayload.from_draft_input(result.next_draft_input),
                draft_input=result.next_draft_input,
                accepted_lengths=result.accept_lens,
            )
        last = target_worker.last
        steps.append(
            dict(
                draft_positions=last["draft_positions"],
                verify_ids=last["verify_ids"],
                logits=result.logits_output.next_token_logits,
                tokens=result.next_token_ids,
                accept_lens=result.accept_lens,
                seq_lens=result.new_seq_lens,
                hidden=state.hidden.clone(),
                topk=state.topk.clone(),
                committed_len=state.committed_len.clone(),
                kv={layer: cache.clone() for layer, cache in kv.kv.items()},
                can_run_cuda_graph=result.can_run_cuda_graph,
                padded_candidates=last["padded_candidates"],
                verify_hidden=last["hidden"],
                verify_topk=planes[:, : len(selected) * width].clone(),
                rows=rows[selected],
            )
        )
        seq_lens[selected] = result.new_seq_lens
        bonus[selected] = result.next_draft_input.bonus_tokens
    return steps


def _dfly_state_tensors(state):
    return (state.hidden, state.topk, state.committed_len, state.mrope_delta)


def _assert_same_steps(expected_steps, actual_steps, *, skip=(), kv_from=0):
    for expected, actual in zip(expected_steps, actual_steps, strict=True):
        for key, value in expected.items():
            if key in skip:
                continue
            if key == "kv":
                for layer, cache in value.items():
                    assert torch.equal(
                        actual["kv"][layer][kv_from:], cache[kv_from:]
                    ), (key, layer)
            elif isinstance(value, torch.Tensor):
                assert torch.equal(actual[key], value), key
            else:
                assert actual[key] == value, key


@pytest.mark.parametrize("physical", [False, True], ids=["local_ids", "physical_ids"])
def test_dfly_target_verify_drafts_as_the_draft_runner_did(
    single_rank, monkeypatch, physical
):
    """Drafting inside the target's verify forward matches the separate
    draft-runner forward step for step: verify input, target logits, accepted
    lengths and tokens, committed request state and every KV byte, across a
    full accept, an interior accept, a block past the context and a request
    whose state went stale, with per-request mRoPE deltas. The draft reads and
    writes KV through the target pool, scans its sparse rows in full, and
    leaves the verify the target's own attention metadata."""
    target, draft = _dfly_parity_models()
    runs = [
        _dfly_decode_steps(
            monkeypatch,
            target=target,
            draft=draft,
            draft_in_target=draft_in_target,
            physical=physical,
        )
        for draft_in_target in (False, True)
    ]
    width = draft.spec.query_width
    first = runs[1][0]
    assert first["accept_lens"].tolist() == [width, 1, 1, 3]
    assert (first["verify_ids"].view(4, width)[0, 1:] > 0).all()
    # Active requests 0 and 3 rotate at p + j plus their own mRoPE delta;
    # requests 1 (block past the context) and 2 (stale state) at j.
    offsets = torch.arange(width)
    assert torch.equal(
        first["draft_positions"].view(4, width),
        torch.stack([20 + offsets, offsets, offsets, 25 + 11 + offsets]),
    )
    _assert_same_steps(*runs)


# Shrinks, grows, permutes and pads batches across buckets (2, 3), with an
# eager fallback at four requests; request 2's state is stale at its first step.
_GRAPH_SCHEDULE = ((0, 2, 3), (3, 1, 0, 2), (1,), (2, 3), (0, 1, 2), (3,), (1, 0))


def test_dfly_prefill_publishes_state_before_overlap_filter_and_merge():
    pool = SimpleNamespace(
        req_to_token=torch.zeros(4, 32, dtype=torch.int32),
        req_generation=torch.zeros(4, dtype=torch.int64),
    )
    worker = object.__new__(DFlyWorkerV2)
    BaseSpecWorker.__init__(worker)
    worker.device = "cpu"
    worker.model_runner = SimpleNamespace(req_to_token_pool=pool)
    worker.draft_model = SimpleNamespace(is_mrope_enabled=False)
    worker._state = VLDFlyRequestState(
        num_rows=4,
        hidden_size=3,
        hidden_dtype=torch.float32,
        num_producers=1,
        topk=2,
        device="cpu",
    )
    worker._capture = DFlyProducerCapture(
        producer_layers=(0,), max_rows=8, topk=2, device="cpu"
    )
    hidden = torch.arange(6, dtype=torch.float32).view(2, 3)
    topk = torch.arange(12, dtype=torch.int32).view(6, 2)

    def target_forward(batch, capture_hidden_mode):
        assert capture_hidden_mode == CaptureHiddenMode.LAST
        worker._capture.record(0, topk)
        return GenerationBatchResult(
            logits_output=SimpleNamespace(hidden_states=hidden[: batch.batch_size()]),
            next_token_ids=batch.req_pool_indices + 10,
        )

    worker._target_worker = SimpleNamespace(forward_batch_generation=target_forward)
    with get_context().override_server_args(speculative_algorithm="DFLY"):
        relay = FutureMap(
            device=torch.device("cpu"),
            spec_algo=SpeculativeAlgorithm.DFLY,
            req_to_token_pool=pool,
        )

        def prefill(rows, lengths):
            rows, lengths = torch.tensor(rows), torch.tensor(lengths)
            batch = SimpleNamespace(
                forward_mode=ForwardMode.EXTEND,
                is_extend_in_batch=False,
                req_pool_indices=rows,
                req_pool_indices_cpu=rows,
                seq_lens=lengths,
                extend_lens=lengths.tolist(),
                batch_size=lambda: len(rows),
            )
            published = []

            def publish(new_seq_lens):
                assert worker._live_rows(batch).all()
                assert torch.equal(worker._state.hidden[rows], hidden[: len(rows)])
                assert torch.equal(
                    worker._state.topk[0, rows], topk[lengths.cumsum(0) - 1]
                )
                published.append(new_seq_lens.clone())
                relay.publish(rows, new_seq_lens)

            result = worker.forward_batch_generation(batch, on_publish=publish)
            assert len(published) == 1
            assert torch.equal(published[0], result.next_draft_input.new_seq_lens)
            relay.stash(rows, RelayPayload.from_draft_input(result.next_draft_input))
            result.next_draft_input.future_indices = rows
            return result.next_draft_input

        carried = prefill([1, 3], [2, 4])
        saved_hidden = worker._state.hidden[3].clone()
        saved_topk = worker._state.topk[:, 3].clone()
        carried.filter_batch(torch.tensor([1]), new_indices_cpu=[1])
        hidden.add_(100)
        topk.add_(20)
        carried.merge_batch(prefill([2], [3]))
        batch = SimpleNamespace(
            spec_info=carried,
            req_pool_indices=carried.future_indices,
            req_pool_indices_cpu=carried.future_indices,
            enable_overlap=True,
            spec_algorithm=SpeculativeAlgorithm.DFLY,
            prefill_input_ids_cpu=None,
            input_ids=None,
        )
        relay.resolve_seq_lens_cpu(batch)
        resolve_forward_inputs(batch, relay)
        assert batch.seq_lens.tolist() == [4, 3]
        assert batch.seq_lens_cpu.tolist() == [4, 3]
        assert carried.bonus_tokens.tolist() == [13, 12]
        assert worker._live_rows(batch).tolist() == [True, True]
        assert torch.equal(worker._state.hidden[3], saved_hidden)
        assert torch.equal(worker._state.topk[:, 3], saved_topk)
        pool.req_generation[3] += 1
        assert worker._live_rows(batch).tolist() == [False, True]
        prefill([3], [5])
        batch.seq_lens = torch.tensor([5, 3])
        assert worker._live_rows(batch).tolist() == [True, True]


def test_dfly_overlap_relay_preserves_composite_decode_state(single_rank, monkeypatch):
    target, draft = _dfly_parity_models()
    with get_context().override_server_args(speculative_algorithm="DFLY"):
        runs = [
            _dfly_decode_steps(
                monkeypatch,
                target=target,
                draft=draft,
                draft_in_target=True,
                physical=True,
                schedule=_GRAPH_SCHEDULE,
                graph_buckets=(2, 3),
                overlap=overlap,
            )
            for overlap in (False, True)
        ]
    _assert_same_steps(*runs)


@pytest.mark.parametrize("has_reserved_lengths", [False, True])
def test_dfly_overlap_verify_without_cpu_sequence_lengths(has_reserved_lengths):
    rows = torch.tensor([1, 2])
    lengths = torch.tensor([5, 9])
    pool = SimpleNamespace(req_to_token=torch.zeros(3, 32, dtype=torch.int32))
    runner = SimpleNamespace(
        device="cpu",
        is_draft_worker=False,
        model_config=SimpleNamespace(
            requires_mm_token_modalities=False, model_is_mrope=True
        ),
        runner_ids=SimpleNamespace(
            own_plan=lambda _: SimpleNamespace(bind=lambda *args, **kwargs: None)
        ),
        ngram_embedding_manager=SimpleNamespace(enabled=False),
        lora_manager=None,
        decode_cuda_graph_runner=None,
    )
    worker = object.__new__(DFlyWorkerV2)
    BaseSpecWorker.__init__(worker)
    worker.device, worker.block_size, worker._max_context_len = "cpu", 3, 32
    worker._target_worker = SimpleNamespace(model_runner=runner)
    draft_input = _make_draft_input(
        bonus_tokens=torch.tensor([10, 11]), new_seq_lens=lengths
    )
    draft_input.future_indices = rows
    if has_reserved_lengths:
        draft_input.nxt_kv_lens_cpu = lengths + 2 * worker.block_size
        draft_input.nxt_kv_lens_sum = int(draft_input.nxt_kv_lens_cpu.sum())
    batch = ScheduleBatch(
        reqs=[],
        device="cpu",
        forward_mode=ForwardMode.DECODE,
        spec_algorithm=SpeculativeAlgorithm.DFLY,
        spec_info=draft_input,
        req_pool_indices=rows,
        req_pool_indices_cpu=rows,
        seq_lens=lengths,
        seq_lens_cpu=lengths.clone(),
        seq_lens_sum=int(lengths.sum()),
        multimodal_inputs=[None, None],
    )
    with get_context().override_server_args(device="cpu", speculative_algorithm="DFLY"):
        relay = FutureMap(
            device=torch.device("cpu"),
            spec_algo=SpeculativeAlgorithm.DFLY,
            req_to_token_pool=pool,
            needs_cpu_seq_lens=False,
        )
        relay.publish(rows, lengths)
        relay.resolve_seq_lens_cpu(batch)
        assert batch.seq_lens_cpu is None and batch.seq_lens_sum is None
        verify = worker._prepare_verify(
            batch, verify_ids=torch.arange(6), verify_slots=torch.arange(6).view(2, 3)
        )
    assert verify.forward_mode == ForwardMode.TARGET_VERIFY
    assert verify.positions.tolist() == [5, 6, 7, 9, 10, 11]
    assert torch.equal(verify.mrope_positions, verify.positions.expand(3, -1))
    assert torch.equal(verify.seq_lens, lengths)
    if has_reserved_lengths:
        assert torch.equal(verify.seq_lens_cpu, lengths + 6)
        assert verify.seq_lens_sum == 26
    else:
        assert verify.seq_lens_cpu is None and verify.seq_lens_sum is None
    assert batch.seq_lens_cpu is None and batch.seq_lens_sum is None


@pytest.mark.parametrize("physical", [False, True], ids=["local_ids", "physical_ids"])
def test_dfly_decode_graph_replays_the_draft_and_verify(
    single_rank, monkeypatch, physical
):
    """One captured decode graph per bucket holds the draft, its sequential
    hidden correction and the target verify. Replaying it with no Python of
    the forward reproduces the eager composite step for step through padded,
    shrunk, regrown and permuted batches, an eager fallback for a batch past
    every bucket and a recapture between steps: verify input, logits, accepted
    tokens, committed state and every KV slot but the discard slot. Each
    request's state comes from its last committed verify row, padded rows never
    draft, row 0 is never written, the target restore wait precedes every
    verify, and capture leaves request state untouched."""
    target, draft = _dfly_parity_models()
    width = draft.spec.query_width
    common = dict(
        target=target,
        draft=draft,
        draft_in_target=True,
        physical=physical,
        schedule=_GRAPH_SCHEDULE,
    )
    eager = _dfly_decode_steps(monkeypatch, **common)
    events = []
    graph = _dfly_decode_steps(
        monkeypatch,
        **common,
        graph_buckets=(2, 3),
        recapture_before={3},
        events=events,
    )

    assert [step["can_run_cuda_graph"] for step in graph] == [
        True,
        False,
        True,
        True,
        True,
        True,
        True,
    ]
    assert not any(step["can_run_cuda_graph"] for step in eager)
    assert graph[0]["accept_lens"].tolist() == [width, 1, 3]
    _assert_same_steps(
        eager, graph, skip=("can_run_cuda_graph", "padded_candidates"), kv_from=1
    )
    assert [len(step["padded_candidates"]) for step in graph] == [0, 0, 1, 0, 0, 1, 0]
    for step in graph:
        assert (step["padded_candidates"][:, 1:] == -1).all()
        source = torch.arange(len(step["rows"])) * width + step["accept_lens"] - 1
        rows = step["rows"]
        assert torch.equal(step["hidden"][rows], step["verify_hidden"][source])
        assert torch.equal(step["topk"][:, rows], step["verify_topk"][:, source])
        assert torch.equal(step["committed_len"][rows], step["seq_lens"])
        assert (step["hidden"][0] == 0).all() and (step["topk"][:, 0] == -1).all()
        assert step["committed_len"][0] == -1
    assert events == [
        event
        for index, selected in enumerate(_GRAPH_SCHEDULE)
        for event in (("restore_wait", index % 2, 5), ("verify", len(selected)))
    ]


def test_dfly_producer_stamps_prove_each_replay():
    """A replayed graph runs no Python, so producer freshness must be visible on
    the device: a replay that ran every producer, the last index layer
    included, passes; one whose graph lacks a producer is refused even though
    that plane still holds the previous replay's rows at a valid address."""
    capture = DFlyProducerCapture(
        producer_layers=(1, 5, 77), max_rows=8, topk=2, device="cpu"
    )
    produced = torch.zeros(8, 2, dtype=torch.int32)

    def forward(layers):
        for layer in layers:
            capture.record(layer, produced + layer)

    capture.arm(num_rows=8)
    complete, partial = _CpuGraph(), _CpuGraph()
    with complete:
        forward((1, 5, 77))
    with partial:
        forward((1, 77))
    capture.disarm()

    produced.fill_(3)
    capture.begin(num_rows=6)
    complete.replay()
    rows = capture.finish()
    assert rows[:, :, 0].tolist() == [[4] * 6, [8] * 6, [80] * 6]

    capture.begin(num_rows=6)
    partial.replay()
    with pytest.raises(RuntimeError, match=r"skipped index layers \[5\]"):
        capture.finish()


def test_dfly_kv_budget_charges_the_buffers_the_worker_allocates(tmp_path, monkeypatch):
    """The KV pool is sized after charging the DFly request state, the producer
    capture and the verify buffers, so the charge must cover every device byte
    the worker then allocates for that request pool; a buffer added without
    its charge fails here. The verify buffers hold one block per request, every
    graph bucket included, and a larger verify is refused."""
    hf_config = ModelConfig(_dfly_model_path(tmp_path)).hf_config
    text = hf_config.text_config
    max_running_requests = 5
    with get_context().override_server_args(
        speculative_algorithm="DFLY",
        speculative_num_draft_tokens=DFLY_KEYS["dfly_block_size"],
        max_running_requests=max_running_requests,
    ):
        configurator = object.__new__(KVCacheConfigurator)
        configurator.is_draft_worker = False
        configurator.spec_algorithm = SpeculativeAlgorithm.DFLY
        configurator.model_config = SimpleNamespace(hf_config=hf_config)
        configurator.attn_dp_size = 1
        configurator.model_dtype = torch.bfloat16
        charged = configurator._hyv4_dfly_buffer_bytes()

    spec = resolve_hyv4_dfly_spec(text)
    monkeypatch.setattr(dfly_worker_v2, "_attach_topk_capture", lambda **_: None)
    pool = SimpleNamespace(
        req_to_token=torch.zeros(max_running_requests + 1, 4, dtype=torch.int32)
    )
    worker = object.__new__(DFlyWorkerV2)
    worker._draft_worker = SimpleNamespace(alloc_memory_pool=lambda **_: None)
    worker._target_model = SimpleNamespace(config=text)
    worker.spec, worker.block_size = spec, spec.query_width
    worker._producer_layers = dfly_producer_layers(spec.target_index_layers)
    worker._draft_topk_planes = dfly_draft_topk_planes(spec.target_index_layers)
    worker.device = "meta"
    worker.draft_model = SimpleNamespace(spec=spec)
    worker._draft_block_spec_info = None
    worker._max_context_len, worker._physical_ids = 64, False
    worker._draft_in_target_forward = False
    worker.model_runner = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        req_to_token_pool=pool,
        runner_ids=None,
    )
    worker.alloc_memory_pool(
        memory_pool_config=SimpleNamespace(max_running_requests=max_running_requests),
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=None,
    )
    worker._init_execution()
    allocated = {
        id(tensor): tensor.nbytes
        for owner in (worker._state, worker._capture, worker._execution)
        for tensor in vars(owner).values()
        if isinstance(tensor, torch.Tensor) and tensor.is_meta
    }
    assert charged == sum(allocated.values()) > 0

    execution = worker._execution
    with pytest.raises(RuntimeError, match="buffers hold 5 requests"):
        execution.stage(live=torch.ones(6, dtype=torch.bool, device="meta"))
    with pytest.raises(RuntimeError, match="buffers hold 5 requests"):
        execution.stage_capture(batch_size=6)
    execution.stage_capture(batch_size=5)
    with pytest.raises(RuntimeError, match="overlaps a staged decode step"):
        execution.stage_capture(batch_size=5)


def test_dfly_target_prefill_runs_the_sparse_mla_path(single_rank):
    """A short BF16 prefill qualifies for one-shot MHA, whose attention never
    reaches the DFly snapshot; the DFly target plans and dispatches it as MLA."""
    backend = SimpleNamespace(
        _is_in_breakable_cuda_graph=lambda: False,
        _is_in_tc_piecewise_cuda_graph=lambda: False,
        _get_device_sm=lambda: 90,
        _is_blackwell=lambda: False,
        supports_mha_one_shot=True,
        token_to_kv_pool=SimpleNamespace(dtype=torch.bfloat16),
        hisparse_coordinator=None,
        enable_auto_select_prefill_impl=True,
        dsa_prefill_impl="flashmla_auto",
        dsa_kv_cache_store_fp8=False,
        use_mha=False,
        prefill_attention_backend_str="dsa",
        decode_attention_backend_str="dsa",
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        seq_lens_cpu=torch.tensor([5, 3]),
        get_max_chunk_capacity=lambda: 8192,
    )
    DeepseekSparseAttnBackend.set_dsa_prefill_impl(backend, batch)
    assert backend.use_mha

    worker = SimpleNamespace(
        _draft_worker=SimpleNamespace(init_attention_backends=lambda: None),
        _check_attention_backends=lambda: False,
        _check_target_kv_layers=lambda: None,
        _init_execution=lambda: None,
        _install_capture_hooks=lambda: None,
        model_runner=SimpleNamespace(attn_backend=backend),
    )
    DFlyWorkerV2.init_attention_backends(worker)
    DeepseekSparseAttnBackend.set_dsa_prefill_impl(backend, batch)
    assert not backend.use_mha
    assert backend.dsa_prefill_impl == "flashmla_sparse"

    backend.use_mha = True
    config = _dfly_text_config(architectures=["HYV4ForCausalLM"])
    attn = hunyuan_v4.HYV4Attention(config, layer_id=config.num_hidden_layers - 1)
    with forward_context(ForwardContext(attn_backend=backend)):
        assert attn.dispatch_attn_forward_method(batch) == AttnForwardMethod.MLA
    assert not backend.use_mha


@pytest.mark.parametrize("adapted", [False, True], ids=["raw", "target_adapted"])
@pytest.mark.parametrize(
    ("edit", "unmatched"),
    [
        (lambda c: c["ignore"].remove(DFLY_FP8_IGNORE[1]), "mtp_fusion.eh_proj"),
        (
            lambda c: c["config_groups"]["block_fp8_w8a8"]["targets"].remove(
                DFLY_FP8_TARGETS[0]
            ),
            "mlp.down_proj",
        ),
    ],
    ids=["unlisted_fusion_projection", "missing_fp8_target"],
)
def test_dfly_draft_quant_lookup_stays_strict(
    single_rank, monkeypatch, tmp_path, adapted, edit, unmatched
):
    """Every draft linear must be targeted or ignored by the checkpoint's
    quantization config, also when the draft receives the target's adapted
    config, whose unmatched layers would otherwise load BF16."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (10, 0))
    config = _fp8_quant_config(FP4_TEXT_OVERRIDES["num_hidden_layers"])
    config["config_groups"]["block_fp8_w8a8"]["targets"] += DFLY_FP8_TARGETS
    config["ignore"] += DFLY_FP8_IGNORE
    edit(config)
    target_config = ModelConfig(_dfly_model_path(tmp_path, config))
    quant_config = _get_quantization_config(target_config, LoadConfig())
    if adapted:
        quant_config = hunyuan_v4.hyv4_quant_config(quant_config)
        assert quant_config.ignore_unmatched_layers
    with pytest.raises(ValueError, match=unmatched):
        HYV4ForCausalLMDFly(target_config.hf_config, quant_config)


def test_dfly_draft_model_config_keeps_the_target_mla_cache_geometry(tmp_path):
    """The DFLY draft ModelConfig resolves the DFly architecture with the
    target's MLA latent, RoPE and DSA index geometry, the layout of the target
    KV it writes; other algorithms keep the NextN draft."""
    path = _dfly_model_path(tmp_path)
    target = ModelConfig(path)
    draft = ModelConfig(path, is_draft_model=True, speculative_algorithm="DFLY")
    assert draft.hf_config.architectures == ["HYV4ForCausalLMDFly"]
    assert draft.attention_arch == target.attention_arch == AttentionArch.MLA
    geometry = (
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "head_dim",
        "index_head_dim",
        "scaling",
    )
    assert {k: getattr(draft, k) for k in geometry} == {
        k: getattr(target, k) for k in geometry
    }
    nextn = ModelConfig(path, is_draft_model=True, speculative_algorithm="EAGLE")
    assert nextn.hf_config.architectures == ["HYV4ForCausalLMNextN"]


def test_dfly_draft_rejects_per_layer_topk_that_drops_layers(single_rank, tmp_path):
    _, draft, text = _dfly_target_and_draft(_dfly_model_path(tmp_path))
    good = torch.full(
        (draft.spec.query_width, draft.spec.draft_index_topk), -1, dtype=torch.int32
    )
    hidden = torch.zeros(1, text.hidden_size)
    for bad in ([], [good, good], [good[:, :-1]], [good.long()]):
        with pytest.raises(ValueError, match="one per draft layer"):
            draft.stage_block(target_hidden=hidden, topk_indices=bad)
    with pytest.raises(RuntimeError, match="staged draft block"):
        draft.forward(None, None, None)


def _dfly_server_args(path, **overrides):
    fields = dict(
        model_path=path,
        revision=None,
        device="cuda",
        dtype="auto",
        quantization=None,
        kv_cache_dtype="auto",
        tp_size=2,
        attn_dp_size=1,
        pp_size=1,
        attn_cp_size=1,
        dcp_size=1,
        disaggregation_mode="null",
        disaggregation_decode_enable_radix_cache=False,
        enable_lora=None,
        enable_two_batch_overlap=False,
        enable_mixed_chunk=False,
        cuda_graph_config=CudaGraphConfig(
            decode=PhaseConfig(backend=Backend.DISABLED),
            prefill=PhaseConfig(backend=Backend.DISABLED),
        ),
        disable_overlap_schedule=True,
        disable_hybrid_swa_memory=False,
        max_running_requests=None,
        speculative_algorithm="dfly",
        speculative_draft_model_path=None,
        speculative_draft_model_revision=None,
        speculative_draft_model_quantization=None,
        speculative_draft_kv_cache_dtype=None,
        speculative_draft_attention_backend=None,
        speculative_num_draft_tokens=None,
        speculative_dfly_block_size=None,
        speculative_num_steps=None,
        speculative_eagle_topk=None,
        speculative_moe_runner_backend=None,
        moe_runner_backend="auto",
        speculative_use_block_verification=False,
        decrypted_draft_config_file=None,
        trust_remote_code=False,
        speculative_draft_window_size=None,
        speculative_draft_sink_size=None,
        speculative_adaptive=False,
    )
    server_args = SimpleNamespace(**fields | overrides)
    server_args._model_config = ModelConfig(path)
    return server_args


def test_dfly_server_args_resolve_the_bundled_draft(tmp_path):
    """--speculative-algorithm DFLY resolves the draft from the target
    checkpoint with its block width (anchor plus drafts) and a bounded request
    count, dispatches to the DFLY worker outside the DFLASH family, admits a
    full decode graph, and fails loud on a prefill graph or a checkpoint format
    outside the DFly contract before any weight loads."""
    path = _dfly_model_path(tmp_path)
    server_args = _dfly_server_args(path)
    handle_speculative_decoding(server_args)
    fields = (
        "speculative_algorithm",
        "speculative_draft_model_path",
        "speculative_num_draft_tokens",
        "speculative_num_steps",
        "speculative_eagle_topk",
    )
    assert {f: resolution_result(server_args, f) for f in fields} == {
        "speculative_algorithm": "DFLY",
        "speculative_draft_model_path": path,
        "speculative_num_draft_tokens": DFLY_KEYS["dfly_block_size"],
        "speculative_num_steps": 1,
        "speculative_eagle_topk": 1,
    }
    # The request state and verify buffers are charged to the KV budget from it.
    assert resolution_result(server_args, "max_running_requests") == 48
    capped = _dfly_server_args(path, max_running_requests=16)
    handle_speculative_decoding(capped)
    assert resolution_result(capped, "max_running_requests") == 16
    overridden = _dfly_server_args(path, speculative_num_draft_tokens=4)
    handle_speculative_decoding(overridden)
    assert resolution_result(overridden, "speculative_num_draft_tokens") == 4
    algo = SpeculativeAlgorithm.from_string("dfly")
    assert algo.is_dfly() and not algo.is_dflash_family()
    assert algo.create_worker(server_args) is DFlyWorkerV2
    decode_graphed = CudaGraphConfig(
        decode=PhaseConfig(backend=Backend.FULL),
        prefill=PhaseConfig(backend=Backend.DISABLED),
    )
    handle_speculative_decoding(
        _dfly_server_args(path, cuda_graph_config=decode_graphed)
    )
    both_graphed = CudaGraphConfig(
        decode=PhaseConfig(backend=Backend.FULL),
        prefill=PhaseConfig(backend=Backend.BREAKABLE),
    )
    handle_speculative_decoding(_dfly_server_args(path, cuda_graph_config=both_graphed))
    full_prefill_graphed = CudaGraphConfig(
        decode=PhaseConfig(backend=Backend.FULL),
        prefill=PhaseConfig(backend=Backend.FULL),
    )
    with pytest.raises(ValueError, match="'full' prefill CUDA graph backend"):
        handle_speculative_decoding(
            _dfly_server_args(path, cuda_graph_config=full_prefill_graphed)
        )
    (tmp_path / "fp4").mkdir()
    fp4_path = _dfly_model_path(tmp_path / "fp4", FP4_QUANT_CONFIG)
    with pytest.raises(ValueError, match="compressed-tensors W8A8 FP8"):
        handle_speculative_decoding(_dfly_server_args(fp4_path))


def test_dfly_on_a_flash_target_resolves_the_flash_hook_worker_and_request_contract(
    tmp_path,
):
    """A Hy4-Flash DFLY target must reach its own hook, worker and request
    validator, never the eager Hy4-VL ones that a later definition could shadow."""
    flash_path = tmp_path / "flash"
    flash_path.mkdir()
    (flash_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["HYV4FlashForConditionalGeneration"],
                "model_type": "hy_v4_flash",
                "text_config": {"num_hidden_layers": 2},
                "dfly_block_size": 5,
            }
        )
    )
    graphed = CudaGraphConfig(
        decode=PhaseConfig(backend=Backend.FULL),
        prefill=PhaseConfig(backend=Backend.DISABLED),
    )
    flash_args = _dfly_server_args(
        str(flash_path),
        cuda_graph_config=graphed,
        max_running_requests=None,
        ep_join_mode=None,
        disable_overlap_schedule=False,
    )
    handle_speculative_decoding(flash_args)
    assert resolution_result(flash_args, "speculative_num_draft_tokens") == 5
    algo = SpeculativeAlgorithm.from_string("dfly")
    assert algo.create_worker(flash_args) is DFlyFlashWorkerV2
    assert resolution_result(flash_args, "disable_overlap_schedule") is False

    vl_args = _dfly_server_args(_dfly_model_path(tmp_path))
    sampled = SimpleNamespace(
        sampling_params=SimpleNamespace(
            top_k=40, json_schema=None, regex=None, ebnf=None, structural_tag=None
        ),
        return_hidden_states=False,
    )
    constrained = SimpleNamespace(
        sampling_params=SimpleNamespace(
            top_k=1, json_schema="{}", regex=None, ebnf=None, structural_tag=None
        ),
        return_hidden_states=False,
    )

    def validate(req, server_args):
        architectures = server_args._model_config.hf_config.architectures
        return validate_dfly_request(
            req,
            enable_overlap=False,
            targets_hy4_flash=dfly_targets_hy4_flash(architectures),
        )

    assert validate(sampled, flash_args) is None
    assert "verifies greedily" in validate(sampled, vl_args)
    assert "constrained decoding" in validate(constrained, flash_args)


@pytest.mark.parametrize(
    ("graph_flags", "refusal"),
    [
        ([], None),
        (["--disable-decode-cuda-graph", "--disable-prefill-cuda-graph"], None),
        (["--disable-prefill-cuda-graph"], None),
        (["--cuda-graph-backend-prefill=breakable"], None),
        (
            ["--cuda-graph-backend-prefill=full"],
            "'full' prefill CUDA graph backend",
        ),
        (
            ["--cuda-graph-backend-decode=breakable", "--disable-prefill-cuda-graph"],
            "'breakable' decode CUDA graph backend",
        ),
    ],
    ids=[
        "default_graphs_and_overlap",
        "both_phases_disabled",
        "decode_graph",
        "breakable_prefill_graph",
        "full_prefill_graph",
        "breakable_decode",
    ],
)
def test_dfly_cli_launch_admits_full_decode_and_breakable_prefill_graphs(
    tmp_path, graph_flags, refusal
):
    """A DFLY command line passes startup validation with an eager or full
    decode graph and an eager or breakable prefill graph; a full prefill graph
    or a breakable decode graph fails."""
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    argv = [
        "--model-path",
        _dfly_model_path(tmp_path),
        "--device",
        "cuda",
        "--speculative-algorithm",
        "DFLY",
        *graph_flags,
    ]
    server_args = ServerArgs.from_cli_args(parser.parse_args(argv))
    handle_cuda_graph_config(server_args)
    if refusal is None:
        handle_speculative_decoding(server_args)
        assert resolution_result(server_args, "speculative_algorithm") == "DFLY"
    else:
        with pytest.raises(ValueError, match=refusal):
            handle_speculative_decoding(server_args)


def _resolve_minimal_cli(monkeypatch, path, *flags, gpu_mem_mib=140 * 1024, tp=8):
    monkeypatch.setattr("sglang.srt.utils.is_cuda", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (9, 0))
    monkeypatch.setattr("sglang.srt.arg_groups.serving_hook.get_device", lambda: "cuda")
    monkeypatch.setattr(
        "sglang.srt.arg_groups.memory_hook.get_device_memory_capacity",
        lambda _: gpu_mem_mib,
    )
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    argv = [
        "--model-path",
        path,
        "--tp",
        str(tp),
        "--reasoning-parser",
        "auto",
        "--tool-call-parser",
        "auto",
        "--port",
        "8000",
        "--served-model-name",
        "hy4",
        *flags,
    ]
    with override_platform(is_cuda=True, is_sm90=True):
        args = ServerArgs.from_cli_args(parser.parse_args(argv))
        args.resolve_once()
        args.check_server_args()
    return args


@pytest.mark.parametrize("mode", ["dfly", "no_spec", "llama"])
def test_minimal_cli_preserves_graph_overlap_and_memory_defaults(
    tmp_path, monkeypatch, mode
):
    path = _dfly_model_path(tmp_path, FP8_QUANT_CONFIG, num_attention_heads=8)
    if mode == "llama":
        LlamaConfig(
            architectures=["LlamaForCausalLM"],
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=8,
            num_key_value_heads=8,
            num_hidden_layers=2,
            vocab_size=128,
        ).save_pretrained(path)
    flags = ["--speculative-algorithm", "DFLY"] if mode == "dfly" else []
    args = _resolve_minimal_cli(monkeypatch, path, *flags)
    explicit = _resolve_minimal_cli(
        monkeypatch,
        path,
        *flags,
        "--disable-overlap-schedule",
        "--mem-fraction-static",
        "0.83",
    )
    cfg = args.resolved_dict()
    assert args.mem_fraction_static is None
    assert cfg["disable_overlap_schedule"] is False
    assert cfg["cuda_graph_config"]["decode"]["backend"] == Backend.FULL
    assert cfg["cuda_graph_config"]["prefill"]["backend"] == Backend.BREAKABLE
    assert 0 < cfg["mem_fraction_static"] < 1
    memory_declarations = [
        (source, values["mem_fraction_static"])
        for source, values in args._resolved_overrides
        if "mem_fraction_static" in values
    ]
    assert memory_declarations == [
        ("_handle_gpu_memory_settings", cfg["mem_fraction_static"])
    ]
    assert resolution_result(explicit, "mem_fraction_static") == 0.83
    assert resolution_result(explicit, "disable_overlap_schedule") is True
    with use_context(create_context(args, role="scheduler")):
        assert get_schedule().disable_overlap_schedule is False
        assert get_exec().graph.cuda_graph_config.decode.backend == Backend.FULL
        assert get_exec().graph.cuda_graph_config.prefill.backend == Backend.BREAKABLE
        assert get_schedule().mem_fraction_static == cfg["mem_fraction_static"]
    if mode == "dfly":
        assert cfg["max_running_requests"] == 48
        assert cfg["speculative_algorithm"] == "DFLY"
    else:
        assert cfg["speculative_algorithm"] is None


HY4_VISION_CONFIG = VISION_OVERRIDES | dict(
    hidden_size=1536,
    intermediate_size=6144,
    num_attention_heads=16,
    num_hidden_layers=40,
    max_image_size=2048,
    max_vit_seq_len=16384,
)
# One maximal ViT forward holds its residual stream and MLP intermediate in bf16.
HY4_VIT_ACTIVATION_MIB = (
    HY4_VISION_CONFIG["max_vit_seq_len"]
    * (HY4_VISION_CONFIG["hidden_size"] + HY4_VISION_CONFIG["intermediate_size"])
    * 2
    / (1 << 20)
)
# Per rank on H200 TP8 in that checkpoint's failed minimal DFLY launch: free memory
# before loading, the KV sizer's minimum fraction, and its DFly plus mm charges.
H200_CAPACITY_MIB = 143771
H200_PRE_LOAD_GIB = 138.01
H200_MIN_VIABLE_MEM_FRACTION = 0.6939
H200_DFLY_AND_MM_CHARGES_GIB = 0.13
# The explicit fraction that loaded that checkpoint and passed serving on H200.
H200_SERVED_MEM_FRACTION = 0.80
GB300_CAPACITY_MIB = 284208
# The failed minimal no-spec H200 TP8 launch: KV sizer minimum and multimodal charge.
H200_NO_SPEC_MIN_VIABLE_MEM_FRACTION = 0.6894
MM_KV_CHARGE_GIB = 0.10
# Served minimal DFLY launches: resolved fractions and GB300 TP4 per-rank memory.
H200_TP8_DFLY_AUTO_MEM_FRACTION = 0.829020497875093
GB300_TP4_DFLY_AUTO_MEM_FRACTION = 0.8721760400833193
GB300_PRE_LOAD_GIB = 274.51
GB300_POST_TARGET_LOAD_GIB = 88.14
GB300_DFLY_KV_GIB = 51.93
# The checkpoint's context length and the served no-spec H200 TP8 KV pool.
HY4_CONTEXT_LEN = 1 << 20
H200_NO_SPEC_KV_TOKENS = 222144
HY4_ENCODER_RESERVE_MIB = 8 * 1024


def _req_table_mib(rows, context_len):
    return (rows + 1) * context_len * 4 / (1 << 20)


def _h200_dfly_kv_budget_gib(mem_fraction_static):
    return (
        mem_fraction_static - H200_MIN_VIABLE_MEM_FRACTION
    ) * H200_PRE_LOAD_GIB - H200_DFLY_AND_MM_CHARGES_GIB


def test_minimal_dfly_cli_auto_memory_fits_target_and_draft_weights(
    tmp_path, monkeypatch
):
    """On H200 TP8 the minimal DFLY command resolved a static budget below its loaded
    weights; automatic memory must keep a served-size KV pool and an encoder reserve."""
    (tmp_path / "vl").mkdir()
    (tmp_path / "flash").mkdir()
    vl_path = _dfly_model_path(
        tmp_path / "vl",
        FP8_QUANT_CONFIG,
        vision_config=HY4_VISION_CONFIG,
        num_attention_heads=8,
    )
    (tmp_path / "flash" / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["HYV4FlashForConditionalGeneration"],
                "model_type": "hy_v4_flash",
                "text_config": {"num_hidden_layers": 2},
                "vision_config": HY4_VISION_CONFIG | {"out_hidden_size": None},
                "dfly_block_size": 5,
            }
        )
    )
    flash_path = tmp_path / "flash"
    dfly = ["--speculative-algorithm", "DFLY"]
    text_only = [*dfly, "--language-model-only"]

    def mem_fraction(path, *flags, capacity=H200_CAPACITY_MIB):
        args = _resolve_minimal_cli(
            monkeypatch, str(path), *flags, gpu_mem_mib=capacity
        )
        return resolution_result(args, "mem_fraction_static")

    def encoder_reserve_mib(capacity):
        auto = mem_fraction(vl_path, *dfly, capacity=capacity)
        return (mem_fraction(vl_path, *text_only, capacity=capacity) - auto) * capacity

    auto = mem_fraction(vl_path, *dfly)
    assert _h200_dfly_kv_budget_gib(auto) >= _h200_dfly_kv_budget_gib(
        H200_SERVED_MEM_FRACTION
    ), auto
    h200_reserve_mib = encoder_reserve_mib(H200_CAPACITY_MIB)
    assert h200_reserve_mib >= HY4_VIT_ACTIVATION_MIB
    assert encoder_reserve_mib(GB300_CAPACITY_MIB) == pytest.approx(h200_reserve_mib)
    unknown = dict(capacity=None)
    assert (
        mem_fraction(vl_path, *dfly, **unknown)
        == mem_fraction(vl_path, **unknown)
        < mem_fraction(vl_path, *text_only, **unknown)
    )
    assert mem_fraction(vl_path, *dfly, "--mem-fraction-static", "0.6") == 0.6
    assert mem_fraction(flash_path, *dfly) == mem_fraction(flash_path)


def test_minimal_no_spec_cli_auto_memory_keeps_kv_after_target_weights(
    tmp_path, monkeypatch
):
    """The minimal no-spec command resolved a static budget below the loaded weights
    on H200 TP8; without the draft, its KV pool and request table together must be
    no smaller than DFLY's KV pool."""
    (tmp_path / "vl").mkdir()
    (tmp_path / "flash").mkdir()
    vl_path = _dfly_model_path(
        tmp_path / "vl",
        FP8_QUANT_CONFIG,
        vision_config=HY4_VISION_CONFIG,
        num_attention_heads=8,
        max_position_embeddings=HY4_CONTEXT_LEN,
    )
    (tmp_path / "flash" / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["HYV4FlashForConditionalGeneration"],
                "model_type": "hy_v4_flash",
                "text_config": {"num_hidden_layers": 2},
                "vision_config": HY4_VISION_CONFIG | {"out_hidden_size": None},
            }
        )
    )
    flash_path = tmp_path / "flash"
    dfly = ["--speculative-algorithm", "DFLY"]

    def mem_fraction(path, *flags, capacity=H200_CAPACITY_MIB, tp=8):
        args = _resolve_minimal_cli(
            monkeypatch, str(path), *flags, gpu_mem_mib=capacity, tp=tp
        )
        return resolution_result(args, "mem_fraction_static")

    def vision_reserve_mib(path, capacity):
        text_only = mem_fraction(path, "--language-model-only", capacity=capacity)
        return (text_only - mem_fraction(path, capacity=capacity)) * capacity

    table_mib = _req_table_mib(MIN_DEFAULT_MAX_RUNNING_REQUESTS, HY4_CONTEXT_LEN)
    h200 = mem_fraction(vl_path)
    h200_kv_gib = (
        h200 - H200_NO_SPEC_MIN_VIABLE_MEM_FRACTION
    ) * H200_PRE_LOAD_GIB - MM_KV_CHARGE_GIB
    # The KV sizer turns a capacity share into pre-load memory.
    h200_table_gib = table_mib / H200_CAPACITY_MIB * H200_PRE_LOAD_GIB
    assert h200_kv_gib > 0, h200
    assert h200_kv_gib + h200_table_gib >= _h200_dfly_kv_budget_gib(
        H200_TP8_DFLY_AUTO_MEM_FRACTION
    ), h200
    gb300 = mem_fraction(vl_path, capacity=GB300_CAPACITY_MIB, tp=4)
    gb300_min_viable = 1 - GB300_POST_TARGET_LOAD_GIB / GB300_PRE_LOAD_GIB
    gb300_kv_gib = (gb300 - gb300_min_viable) * GB300_PRE_LOAD_GIB - MM_KV_CHARGE_GIB
    gb300_table_gib = table_mib / GB300_CAPACITY_MIB * GB300_PRE_LOAD_GIB
    assert gb300_kv_gib + gb300_table_gib >= GB300_DFLY_KV_GIB, gb300

    h200_reserve_mib = vision_reserve_mib(vl_path, H200_CAPACITY_MIB) - table_mib
    assert h200_reserve_mib >= HY4_VIT_ACTIVATION_MIB
    assert vision_reserve_mib(vl_path, GB300_CAPACITY_MIB) - table_mib == pytest.approx(
        h200_reserve_mib
    )
    # Other VLM families keep the capacity-proportional vision cut.
    flash_share = [
        vision_reserve_mib(flash_path, capacity) / capacity
        for capacity in (H200_CAPACITY_MIB, GB300_CAPACITY_MIB)
    ]
    assert flash_share[1] == pytest.approx(flash_share[0], rel=0.1)

    assert mem_fraction(vl_path, *dfly) == pytest.approx(
        H200_TP8_DFLY_AUTO_MEM_FRACTION
    )
    assert mem_fraction(
        vl_path, *dfly, capacity=GB300_CAPACITY_MIB, tp=4
    ) == pytest.approx(GB300_TP4_DFLY_AUTO_MEM_FRACTION)
    assert mem_fraction(vl_path, "--mem-fraction-static", "0.6") == 0.6


def test_minimal_no_spec_cli_auto_memory_reserves_the_request_table(
    tmp_path, monkeypatch
):
    """The served no-spec launch allocated its 2048-row request table out of the
    non-static memory and left the encoder too little; automatic memory must hold
    the table besides the encoder reserve, sized by the context and request rows."""
    vl_path = _dfly_model_path(
        tmp_path,
        FP8_QUANT_CONFIG,
        vision_config=HY4_VISION_CONFIG,
        num_attention_heads=8,
        max_position_embeddings=HY4_CONTEXT_LEN,
    )

    def mem_fraction(*flags, capacity=H200_CAPACITY_MIB, tp=8):
        args = _resolve_minimal_cli(
            monkeypatch, vl_path, *flags, gpu_mem_mib=capacity, tp=tp
        )
        return resolution_result(args, "mem_fraction_static")

    def reserve_mib(*flags, capacity=H200_CAPACITY_MIB, tp=8):
        text_only = mem_fraction(
            *flags, "--language-model-only", capacity=capacity, tp=tp
        )
        return (text_only - mem_fraction(*flags, capacity=capacity, tp=tp)) * capacity

    rows = MIN_DEFAULT_MAX_RUNNING_REQUESTS
    for capacity, tp in ((H200_CAPACITY_MIB, 8), (GB300_CAPACITY_MIB, 4)):
        assert reserve_mib(capacity=capacity, tp=tp) == pytest.approx(
            HY4_ENCODER_RESERVE_MIB + _req_table_mib(rows, HY4_CONTEXT_LEN), rel=1e-9
        )
    assert reserve_mib("--context-length", "131072") == pytest.approx(
        HY4_ENCODER_RESERVE_MIB + _req_table_mib(rows, 131072), rel=1e-9
    )
    assert reserve_mib("--max-running-requests", "256") == pytest.approx(
        HY4_ENCODER_RESERVE_MIB + _req_table_mib(256, HY4_CONTEXT_LEN), rel=1e-9
    )
    # Each attention-DP worker holds a table for its share of the requests.
    assert reserve_mib(
        "--attn-dp-size", "2", "--max-running-requests", "256"
    ) == pytest.approx(
        HY4_ENCODER_RESERVE_MIB + _req_table_mib(128, HY4_CONTEXT_LEN), rel=1e-9
    )

    for algorithm in ("DFLY", "NEXTN"):
        spec = ["--speculative-algorithm", algorithm]
        assert reserve_mib(*spec) == pytest.approx(HY4_ENCODER_RESERVE_MIB, rel=1e-9)
        assert mem_fraction(*spec) == pytest.approx(H200_TP8_DFLY_AUTO_MEM_FRACTION)
    assert mem_fraction(
        "--speculative-algorithm", "DFLY", capacity=GB300_CAPACITY_MIB, tp=4
    ) == pytest.approx(GB300_TP4_DFLY_AUTO_MEM_FRACTION)
    assert mem_fraction("--mem-fraction-static", "0.6") == 0.6
    unknown = dict(capacity=None)
    assert mem_fraction(**unknown) == mem_fraction(
        "--speculative-algorithm", "DFLY", **unknown
    )


def test_hy4_request_table_reserve_matches_the_runtime_table():
    """The reserve must equal what ReqToTokenPool allocates: its dtype and padding
    row, and the KV sizer's default row count at Hy4 capacities. Above four
    contexts of KV the sizer adds rows past that default."""
    cfg = SimpleNamespace(max_running_requests=3, attn_dp_size=1)
    pool = ReqToTokenPool(
        size=3, max_context_len=40, device="cpu", enable_memory_saver=False
    )
    assert _req_to_token_table_mb(cfg, 40) * (1 << 20) == pool.req_to_token.nbytes

    sizer = SimpleNamespace(
        model_config=SimpleNamespace(context_len=HY4_CONTEXT_LEN),
        attn_dp_size=1,
        mambaish_config=None,
        _unified_sized=None,
    )
    with get_context().override_server_args(max_running_requests=None):
        for token_capacity in (H200_NO_SPEC_KV_TOKENS, 4 * HY4_CONTEXT_LEN):
            assert (
                KVCacheConfigurator.resolve_max_num_reqs(sizer, token_capacity)
                == MIN_DEFAULT_MAX_RUNNING_REQUESTS
            )
        assert (
            KVCacheConfigurator.resolve_max_num_reqs(sizer, 5 * HY4_CONTEXT_LEN)
            > MIN_DEFAULT_MAX_RUNNING_REQUESTS
        )


def test_dfly_request_rows_hold_the_double_buffered_reserve():
    """A DFLY request reserves two verify blocks past its committed length at
    every page size; near the context its req_to_token row must hold them, or
    the reservation spills into the next request's row."""
    with get_context().override_server_args(
        speculative_algorithm="DFLY",
        speculative_num_draft_tokens=7,
        speculative_num_steps=1,
        speculative_eagle_topk=1,
        page_size=1,
    ):
        assert get_req_to_token_extra_context_len() >= get_alloc_reserve_per_decode()


def test_dfly_greedy_head_never_picks_padded_vocab_rows(single_rank):
    head = ParallelLMHead(250, 8)
    head.weight.data.zero_()
    head.weight.data[:250] = -1.0
    head.weight.data[123] = -0.5
    draft = SimpleNamespace(lm_head=head)
    tokens = HYV4ForCausalLMDFly._greedy_tokens(draft, torch.ones(3, 8))
    assert tokens.tolist() == [123] * 3


@pytest.fixture
def interpreted_dfly(monkeypatch):
    import importlib

    import triton.language as tl
    from triton.runtime.interpreter import InterpretedFunction
    from triton.runtime.jit import JITFunction

    from sglang.kernels.ops.attention import dfly as sparse_ops
    from sglang.kernels.ops.speculative import dfly as plan_ops

    # Gluon imports the original standard-library JIT functions on first use.
    importlib.import_module("triton.experimental.gluon")
    for module in (tl, tl.standard):
        for name, fn in vars(module).copy().items():
            if isinstance(fn, JITFunction):
                monkeypatch.setattr(module, name, InterpretedFunction(fn.fn))

    for module, name in (
        (plan_ops, "_plan_dfly_inputs_kernel"),
        (sparse_ops, "_build_dfly_sparse_indices_stacked_kernel"),
    ):
        kernel = getattr(module, name)
        monkeypatch.setattr(module, name, InterpretedFunction(kernel.fn))
    monkeypatch.setattr(dfly_worker_v2, "can_fuse_dfly_inputs", lambda *a, **kw: True)


@pytest.mark.parametrize(
    "capacity,row,topk_capacity",
    [(8192, 0, 8192), (32768, 32767, 32768), (8192, 0, 131072)],
)
def test_dfly_fused_sparse_offsets_beyond_int32(
    interpreted_dfly, monkeypatch, capacity, row, topk_capacity
):
    """Large fixed-buffer strides must not wrap sparse reads or writes below base."""
    import triton.language as tl
    from triton.runtime.interpreter import TensorHandle, interpreter_builder

    from sglang.kernels.ops.attention import dfly as sparse_ops

    width, topk, out_width, producers = 16, 4096, 4224, 5
    metadata = torch.zeros(1, dtype=torch.int64)
    active = torch.ones(1, dtype=torch.bool)
    history = torch.zeros(1, dtype=torch.int32)
    out = torch.zeros(1, dtype=torch.int32)
    reads, writes = {}, {}
    load = interpreter_builder.create_masked_load
    program_id = interpreter_builder.create_get_program_id

    def sampled_program_id(axis):
        value = program_id(axis)
        if axis == 0:
            value.data[:] = row
        elif axis == 1:
            value.data[:] *= width - 1
        return value

    def record_offsets(records, ptrs, mask, tensor):
        _, query, plane = interpreter_builder.grid_idx
        offsets = (
            ptrs.data.astype("int64") - tensor.data_ptr()
        ) // tensor.element_size()
        records.setdefault((plane, query * (width - 1)), []).extend(
            offsets[mask.data.astype(bool)].tolist()
        )

    def record_load(ptrs, mask, other, cache_modifier, eviction_policy, is_volatile):
        dtype = ptrs.get_element_ty()
        if dtype == tl.int32:
            record_offsets(reads, ptrs, mask, history)
            tensor = history
        else:
            tensor = metadata if dtype == tl.int64 else active
        # Redirect all reads before dereferencing the virtual multi-GiB addresses.
        safe_ptrs = TensorHandle(ptrs.data * 0 + tensor.data_ptr(), ptrs.dtype)
        return load(
            safe_ptrs, mask, other, cache_modifier, eviction_policy, is_volatile
        )

    def record_store(ptrs, value, mask, cache_modifier, eviction_policy):
        record_offsets(writes, ptrs, mask, out)

    monkeypatch.setattr(
        interpreter_builder, "create_get_program_id", sampled_program_id
    )
    monkeypatch.setattr(interpreter_builder, "create_masked_load", record_load)
    monkeypatch.setattr(interpreter_builder, "create_masked_store", record_store)
    sparse_ops._build_dfly_sparse_indices_stacked_kernel[(1, 2, producers)](
        history,
        metadata,
        metadata,
        active,
        metadata,
        metadata,
        out,
        REQ_STRIDE=1,
        TOPK_PLANE_STRIDE=topk_capacity * topk,
        CONTEXT_STRIDE=1,
        OUT_PLANE_STRIDE=capacity * width * out_width,
        TOPK=topk,
        WINDOW=0,
        WIDTH=width,
        OUT_WIDTH=out_width,
        PHYSICAL=False,
        BLOCK=256,
        SEARCH_STEPS=1,
    )
    for plane in range(producers):
        for query in (0, width - 1):
            expected_reads = (
                torch.arange(topk, dtype=torch.int64) + plane * topk_capacity * topk
            )
            expected_writes = (
                torch.arange(out_width, dtype=torch.int64)
                + ((plane * capacity + row) * width + query) * out_width
            )
            assert reads[plane, query] == expected_reads.tolist()
            assert writes[plane, query] == expected_writes.tolist()


@pytest.mark.parametrize("physical", [False, True])
@pytest.mark.parametrize(
    "window,width,producers,mrope",
    [(0, 1, 1, False), (4, 5, 2, True), (9, 128, 2, True), (1024, 1, 2, True)],
)
def test_dfly_fused_plan_preserves_id_spaces_and_fixed_buffers(
    interpreted_dfly, physical, window, width, producers, mrope
):
    """Physical window exclusion, causal boundaries and padding must agree with
    the eager oracle even when a shared producer and stale output buffers recur."""
    from sglang.test.kernels.dfly import assert_dfly_plan_equal, dfly_plan_fixture

    execution, inputs = dfly_plan_fixture(
        device="cpu",
        physical=physical,
        window=window,
        width=width,
        producers=producers,
        mrope=mrope,
    )
    pointers = {
        name: value.data_ptr()
        for name, value in vars(execution).items()
        if isinstance(value, torch.Tensor)
    }
    schedule = (
        ([0, 1, 2, 3, 4],)
        if width == 128 or window == 1024
        else ([0, 1, 2, 3, 4], [4], [1, 0, 4], [3, 3], [4, 0, 1, 2, 3])
    )
    for selected in schedule:
        batch = {name: value[selected] for name, value in inputs.items()}
        batch["bonus_tokens"] = torch.stack(
            [batch["bonus_tokens"], batch["bonus_tokens"]], dim=1
        )[:, 0]
        execution._fuse_inputs = False
        expected = execution.plan(**batch)
        # The eager oracle also writes the shared sparse buffer.
        expected = dfly_worker_v2.VLDFlyDraftInputs(
            block=expected.block,
            target_hidden=expected.target_hidden,
            topk_indices=tuple(t.clone() for t in expected.topk_indices),
            mrope_positions=expected.mrope_positions,
        )
        execution._draft_rows.fill_(12345)
        execution._fuse_inputs = True
        actual = execution.plan(**batch)
        assert_dfly_plan_equal(actual, expected)
        assert actual.block.input_ids.data_ptr() == execution._plan_ids.data_ptr()
        assert actual.topk_indices[0].data_ptr() == actual.topk_indices[-1].data_ptr()
        if mrope:
            assert actual.mrope_positions.stride(0) == 5 * width
        assert pointers == {
            name: value.data_ptr()
            for name, value in vars(execution).items()
            if isinstance(value, torch.Tensor)
        }


@pytest.mark.parametrize("physical", [False, True])
def test_dfly_fused_plan_preserves_verify_candidates(
    single_rank, monkeypatch, interpreted_dfly, physical
):
    """Fused planning must preserve candidate IDs, accepted tokens, metadata
    restoration and target KV bytes through the complete CPU decode harness."""
    target, draft = _dfly_parity_models()
    common = dict(target=target, draft=draft, physical=physical)
    monkeypatch.setattr(dfly_worker_v2, "can_fuse_dfly_inputs", lambda *a, **kw: False)
    eager = _dfly_decode_steps(monkeypatch, **common, draft_in_target=False)
    monkeypatch.setattr(dfly_worker_v2, "can_fuse_dfly_inputs", lambda *a, **kw: True)
    fused = _dfly_decode_steps(monkeypatch, **common, draft_in_target=True)
    _assert_same_steps(eager, fused)


def test_dfly_fusion_capability_and_invalid_contracts(monkeypatch):
    from sglang.kernels.ops.speculative.dfly import can_fuse_dfly_inputs
    from sglang.test.kernels.dfly import dfly_plan_fixture

    monkeypatch.setattr(torch.version, "hip", None)
    assert can_fuse_dfly_inputs("cuda", query_width=128, window=1024)
    for device, width, window in (("cpu", 5, 4), ("cuda", 129, 4), ("cuda", 5, 1025)):
        assert not can_fuse_dfly_inputs(device, query_width=width, window=window)
    monkeypatch.setattr(torch.version, "hip", "6.0")
    assert not can_fuse_dfly_inputs("cuda", query_width=5, window=4)

    execution, inputs = dfly_plan_fixture(
        device="cpu", physical=True, window=4, width=5
    )
    execution._fuse_inputs = True
    with pytest.raises(ValueError, match="verify slots must have shape"):
        execution.plan(**(inputs | {"verify_slots": inputs["verify_slots"][:, :4]}))
    with pytest.raises(ValueError, match="DFly verify_slots"):
        execution.plan(**(inputs | {"verify_slots": inputs["verify_slots"].float()}))
    with pytest.raises(ValueError, match="DFly verify_slots"):
        execution.plan(**(inputs | {"verify_slots": inputs["verify_slots"].T}))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
