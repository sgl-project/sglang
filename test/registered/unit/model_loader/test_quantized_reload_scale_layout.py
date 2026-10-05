"""Quantized reload scales follow their destination projection's layout."""

import math
import socket
import unittest
from contextlib import nullcontext

import torch
from transformers import Qwen2Config, Qwen3Config

from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.model_loader.loader import DefaultModelLoader, QuantizedRLModelLoader
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from sglang.srt.runtime_context import SpawnRanks, get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


def values(shape, version=0):
    base = (
        torch.arange(math.prod(shape), device="cuda").reshape(shape) % 31 - 15
    ).float() / 64
    if len(shape) == 2:
        base *= ((torch.arange(shape[0], device="cuda") + version) % 17 + 1)[:, None]
        base *= (torch.arange(shape[1], device="cuda") // max(shape[1] // 4, 1) + 1)[
            None, :
        ]
    return (base + version / 128).to(torch.bfloat16)


def checkpoint(version=0, kind="qwen2", kv_heads=8):
    weights = []
    for name, shape in {
        "model.embed_tokens.weight": (512, 512),
        "lm_head.weight": (512, 512),
        "model.norm.weight": (512,),
        "model.layers.0.input_layernorm.weight": (512,),
        "model.layers.0.post_attention_layernorm.weight": (512,),
        **{
            f"model.layers.0.self_attn.{p}.weight": (
                kv_heads * 64 if p in ("k_proj", "v_proj") else 512,
                512,
            )
            for p in ("q_proj", "k_proj", "v_proj", "o_proj")
        },
        **{
            f"model.layers.0.self_attn.{p}.bias": (
                kv_heads * 64 if p in ("k_proj", "v_proj") else 512,
            )
            for p in ("q_proj", "k_proj", "v_proj")
            if kind == "qwen2"
        },
        **{
            f"model.layers.0.self_attn.{p}.weight": (64,)
            for p in ("q_norm", "k_norm")
            if kind == "qwen3"
        },
        **{
            f"model.layers.0.mlp.{p}.weight": (1024, 512)
            for p in ("gate_proj", "up_proj")
        },
        "model.layers.0.mlp.down_proj.weight": (512, 1024),
    }.items():
        weights.append((name, values(shape, version)))
    return weights


def build_model(kind="qwen2", *, kv_heads=8, version=0, weights=None, per_tensor=False):
    config_class, model_class = {
        "qwen2": (Qwen2Config, Qwen2ForCausalLM),
        "qwen3": (Qwen3Config, Qwen3ForCausalLM),
    }[kind]
    config = config_class(
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=kv_heads,
        head_dim=64,
        vocab_size=512,
        max_position_embeddings=32,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    with torch.device("cuda"):
        model = model_class(config, Fp8Config(is_checkpoint_fp8_serialized=False))
    if per_tensor:
        for module in model.modules():
            method = getattr(module, "quant_method", None)
            if hasattr(method, "cutlass_fp8_supported"):
                method.cutlass_fp8_supported = False
                method.use_marlin = False
    loader = QuantizedRLModelLoader(LoadConfig(load_format=LoadFormat.FLASH_RL))
    loader.load_weights_and_postprocess(
        model,
        checkpoint(version, kind, kv_heads) if weights is None else weights,
        torch.device("cuda"),
    )
    return model


def loading_scope(changed):
    if not changed:
        return nullcontext()
    return get_parallel().override(
        tp_size=4,
        tp_rank=3,
        tp_group=None,
        attn_tp_size=4,
        attn_tp_rank=3,
        attn_tp_group=None,
        attn_dp_size=1,
        attn_dp_rank=0,
        attn_cp_size=1,
        attn_cp_rank=0,
        moe_tp_size=4,
        moe_tp_rank=3,
        moe_ep_size=1,
        moe_ep_rank=0,
        moe_ep_group=None,
        moe_dp_size=1,
        moe_dp_rank=0,
    )


def assert_same_parameters(actual, expected):
    expected_params = dict(expected.named_parameters())
    for name, param in actual.named_parameters():
        reference = expected_params[name]
        assert param.shape == reference.shape, name
        assert param.stride() == reference.stride(), name
        assert param.dtype == reference.dtype, name
        torch.testing.assert_close(
            param.contiguous().reshape(-1).view(torch.uint8),
            reference.contiguous().reshape(-1).view(torch.uint8),
            rtol=0,
            atol=0,
            msg=name,
        )


def check_reload(model, version=13, changed=False, per_tensor=False):
    kind, kv_heads = model.config.model_type, model.config.num_key_value_heads
    source = checkpoint(version, kind, kv_heads)
    pointers = {n: p.data_ptr() for n, p in model.named_parameters()}
    DefaultModelLoader.restore_weights_before_loading(model, torch.device("cuda"))
    with loading_scope(changed):
        model.load_weights(source)
    DefaultModelLoader.postprocess_weights(model, torch.device("cuda"))
    for name, param in model.named_parameters():
        assert param.data_ptr() == pointers[name], name
    reference = build_model(
        kind, kv_heads=kv_heads, version=version, per_tensor=per_tensor
    )
    assert_same_parameters(model, reference)
    return model


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestQuantizedReloadScaleLayout(CustomTestCase):
    def setUp(self):
        if torch.distributed.is_initialized():
            self.skipTest("requires an isolated distributed test process")
        reset_context()
        self.addCleanup(reset_context)
        original = torch.get_default_dtype()
        self.addCleanup(torch.set_default_dtype, original)
        torch.set_default_dtype(torch.bfloat16)
        server = ServerArgs(model_path="dummy", device="cuda", tp_size=1)
        publish(server, role="test", ranks=SpawnRanks(world_rank=0, gpu_id=0))
        torch.cuda.set_device(0)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
        )
        self.addCleanup(destroy_distributed_environment)
        initialize_model_parallel()
        self.addCleanup(destroy_model_parallel)
        initialize_dp_attention(server)

    def test_native_model_reload_in_construction_scope(self):
        check_reload(build_model())

    def test_native_model_reload_after_scope_exit(self):
        model = build_model()
        check_reload(model, changed=True)
        check_reload(model, version=29, changed=True)

    def test_native_attention_group_projection_reload_after_scope_exit(self):
        model = build_model("qwen3")
        check_reload(model, changed=True)
        check_reload(model, version=29, changed=True)

    def test_unequal_qkv_and_replicated_kv_match_cold_load(self):
        for kv_heads in (1, 2, 4):
            with self.subTest(kv_heads=kv_heads):
                check_reload(build_model("qwen3", kv_heads=kv_heads), changed=True)

    def test_partial_qkv_and_mlp_updates_preserve_other_rows(self):
        for kind in ("qwen2", "qwen3"):
            for shard in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"):
                with self.subTest(kind=kind, shard=shard):
                    model = build_model(kind, kv_heads=1)
                    before = checkpoint(kind=kind, kv_heads=1)
                    update = {
                        n: t for n, t in checkpoint(13, kind, 1) if f".{shard}." in n
                    }
                    pointers = {n: p.data_ptr() for n, p in model.named_parameters()}
                    with loading_scope(True):
                        model.load_weights(list(update.items()))
                    expected = [(n, update.get(n, t)) for n, t in before]
                    reference = build_model(kind, kv_heads=1, weights=expected)
                    assert_same_parameters(model, reference)
                    for name, param in model.named_parameters():
                        self.assertEqual(param.data_ptr(), pointers[name], name)

    def test_per_tensor_quantization_matches_native_cold_load(self):
        model = build_model("qwen3", kv_heads=1, per_tensor=True)
        check_reload(model, version=0, changed=True, per_tensor=True)
        check_reload(model, version=13, changed=True, per_tensor=True)

    def test_empty_session_keeps_native_layout_and_memory(self):
        model = build_model("qwen3", kv_heads=1)
        pointers = {n: p.data_ptr() for n, p in model.named_parameters()}
        DefaultModelLoader.restore_weights_before_loading(model, torch.device("cuda"))
        model.load_weights([])
        DefaultModelLoader.postprocess_weights(model, torch.device("cuda"))
        assert_same_parameters(model, build_model("qwen3", kv_heads=1))
        for name, param in model.named_parameters():
            self.assertEqual(param.data_ptr(), pointers[name], name)


if __name__ == "__main__":
    unittest.main()
