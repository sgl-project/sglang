"""Quantized reload scales follow their destination projection's layout."""

import math
import socket
import unittest
from contextlib import nullcontext

import torch
from transformers import Qwen2Config, Qwen3Config

from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.srt.layers.linear import MergedColumnParallelLinear, ReplicatedLinear
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.model_loader.loader import QuantizedRLModelLoader
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
    return (base + version / 128).to(torch.bfloat16)


def checkpoint(version=0, kind="qwen2"):
    weights = []
    for name, shape in {
        "model.embed_tokens.weight": (512, 512),
        "lm_head.weight": (512, 512),
        "model.norm.weight": (512,),
        "model.layers.0.input_layernorm.weight": (512,),
        "model.layers.0.post_attention_layernorm.weight": (512,),
        **{
            f"model.layers.0.self_attn.{p}.weight": (512, 512)
            for p in ("q_proj", "k_proj", "v_proj", "o_proj")
        },
        **{
            f"model.layers.0.self_attn.{p}.bias": (512,)
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


def build_model(kind="qwen2"):
    config_class, model_class = {
        "qwen2": (Qwen2Config, Qwen2ForCausalLM),
        "qwen3": (Qwen3Config, Qwen3ForCausalLM),
    }[kind]
    config = config_class(
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=8,
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
    loader = QuantizedRLModelLoader(LoadConfig(load_format=LoadFormat.FLASH_RL))
    loader.load_weights_and_postprocess(
        model, checkpoint(kind=kind), torch.device("cuda")
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


def check_reload(model, version=13, changed=False):
    source = checkpoint(version, model.config.model_type)
    if model.config.model_type == "qwen3":
        # Q/K norms are outside the current quantized reload exclude list.
        # Keep their initial values while updating the native projections.
        source = [(n, t) for n, t in source if "_norm.weight" not in n]
    params = dict(model.named_parameters())
    pointers = {n: t.data_ptr() for n, t in params.items()}
    with loading_scope(changed):
        model.load_weights(source)
    for name, ptr in pointers.items():
        assert dict(model.named_parameters())[name].data_ptr() == ptr, name
    for prefix, shards in (
        ("model.layers.0.self_attn.qkv_proj", ("q_proj", "k_proj", "v_proj")),
        ("model.layers.0.mlp.gate_up_proj", ("gate_proj", "up_proj")),
    ):
        layer = model.get_submodule(prefix)
        parent_prefix = prefix.rpartition(".")[0]
        expected = []
        for shard in shards:
            full = dict(source)[f"{parent_prefix}.{shard}.weight"]
            _, scale = per_token_group_quant_fp8(full, full.shape[-1])
            rows = full.shape[0] // layer.tp_size
            start = layer.tp_rank * rows
            expected.append(scale[start : start + rows].t().contiguous())
        torch.testing.assert_close(
            layer.weight_scale, torch.cat(expected, dim=-1), rtol=0, atol=0
        )
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

    def test_replicated_stacked_scale_keeps_full_rows(self):
        for cls in (MergedColumnParallelLinear, ReplicatedLinear):
            with torch.device("cuda"):
                config = Fp8Config(is_checkpoint_fp8_serialized=False)
                if cls is MergedColumnParallelLinear:
                    layer = cls(
                        512,
                        [512, 512],
                        bias=False,
                        quant_config=config,
                        parallel_group="replicated",
                    )
                else:
                    layer = cls(512, 1024, bias=False, quant_config=config)
                layer.weight.copy_(values(layer.weight.shape))
                layer.quant_method.process_weights_after_loading(layer)
            scale_info = {}
            for index in (0, 1):
                _, scale_info[index] = per_token_group_quant_fp8(
                    values((512, 512), 13 + index), 512
                )
            before_ptr = layer.weight_scale.data_ptr()
            with loading_scope(True):
                QuantizedRLModelLoader._apply_scale_update(
                    {"gate_up_proj.weight_scale": layer.weight_scale},
                    "gate_up_proj.weight",
                    scale_info,
                    layer=layer,
                )
            expected = torch.cat([scale_info[i].t() for i in (0, 1)], dim=-1)
            torch.testing.assert_close(layer.weight_scale, expected, rtol=0, atol=0)
            self.assertEqual(layer.weight_scale.data_ptr(), before_ptr)

    def test_missing_scale_and_none_updates_keep_existing_values(self):
        model = build_model()
        layer = model.model.layers[0].self_attn.qkv_proj
        before = layer.weight_scale.detach().clone()
        with loading_scope(True):
            QuantizedRLModelLoader._apply_scale_update(
                {}, "missing.weight", torch.ones(1, device="cuda"), layer=layer
            )
            QuantizedRLModelLoader._apply_scale_update(
                dict(model.named_parameters()),
                "model.layers.0.self_attn.qkv_proj.weight",
                None,
                layer=layer,
            )
        torch.testing.assert_close(layer.weight_scale, before, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
