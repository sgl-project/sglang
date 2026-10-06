"""MiMo fused QKV checkpoint preprocessing follows the destination projection."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.layers.linear import QKVParallelLinear
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.srt.lora.layers import BaseLayerWithLoRA, unwrap_lora_layer
from sglang.srt.models.mimo_v2 import (
    MiMoV2ForCausalLM,
    _resolve_deferred_qkv_scale_inv,
    load_mimo_v2_qkv_proj_weight,
)
from sglang.srt.models.mimo_v2_nextn import MiMoV2MTP
from sglang.srt.runtime_context import SpawnRanks, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.parallel_groups import parallel_scope, publish, rank_size
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

WEIGHT_NAME = "model.layers.0.self_attn.qkv_proj.weight"
SCALE_NAME = WEIGHT_NAME.replace(".weight", ".weight_scale_inv")


def loading_scope(changed):
    if not changed:
        return nullcontext()
    return parallel_scope(
        tp_size=1,
        tp_rank=0,
        tp_group=None,
        attn_tp_size=1,
        attn_tp_rank=0,
        attn_tp_group=None,
        attn_dp_size=1,
        attn_dp_rank=0,
        attn_cp_size=1,
        attn_cp_rank=0,
        moe_tp_size=1,
        moe_tp_rank=0,
        moe_ep_size=1,
        moe_ep_rank=0,
        moe_ep_group=None,
        moe_dp_size=1,
        moe_dp_rank=0,
    )


def build_projection(dtype=torch.bfloat16, fp8=False):
    config = (
        Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
        if fp8
        else None
    )
    with torch.device("cuda"):
        projection = QKVParallelLinear(
            128,
            32,
            64,
            32,
            v_head_size=16,
            bias=False,
            params_dtype=dtype,
            quant_config=config,
            parallel_group="attn_tp",
        )
    if fp8:
        assert isinstance(projection.quant_method, Fp8LinearMethod)
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.model.layers[0].self_attn = torch.nn.Module()
    model.model.layers[0].self_attn.qkv_proj = projection
    return model, projection


def values(shape, dtype, offset=0):
    count = 1
    for size in shape:
        count *= size
    return (
        ((torch.arange(count, device="cuda") + offset) % 31 - 15)
        .reshape(shape)
        .to(dtype)
    )


def load_fused(projection, *, changed=False, ckpt_tp=None, sharded=False, offset=0):
    destination = projection
    projection = unwrap_lora_layer(projection)
    weight = projection.weight
    full = values(
        (weight.shape[0] * rank_size(projection)[1], weight.shape[1]),
        weight.dtype,
        offset,
    )
    shards = full.chunk(rank_size(projection)[1], dim=0)
    data = shards[rank_size(projection)[0]] if sharded else full
    with loading_scope(changed):
        load_mimo_v2_qkv_proj_weight(
            WEIGHT_NAME, weight, data, ckpt_tp, qkv_proj=destination
        )
    torch.testing.assert_close(weight, shards[rank_size(projection)[0]], rtol=0, atol=0)
    return weight


def load_deferred(model, projection, *, changed=False, offset=0):
    destination = projection
    projection = unwrap_lora_layer(projection)
    ckpt_tp = 8
    rows = projection.weight.shape[0] * rank_size(projection)[1]
    full = values((rows, 128), torch.float8_e4m3fn, offset)
    # Each checkpoint shard has 448 rows, requiring four 128-row scales.
    scale = ((torch.arange(32, device="cuda") + offset) % 3 + 1).float().reshape(
        32, 1
    ) / 128
    deferred = {}
    with loading_scope(changed):
        load_mimo_v2_qkv_proj_weight(
            WEIGHT_NAME, projection.weight, full, ckpt_tp, qkv_proj=destination
        )
        load_mimo_v2_qkv_proj_weight(
            SCALE_NAME,
            projection.weight_scale_inv,
            scale,
            ckpt_tp,
            deferred_scale_inv=deferred,
            qkv_proj=destination,
        )
    assert SCALE_NAME in deferred
    assert deferred[SCALE_NAME].data_ptr() != scale.data_ptr()
    torch.testing.assert_close(deferred[SCALE_NAME], scale, rtol=0, atol=0)
    params = {WEIGHT_NAME: projection.weight, SCALE_NAME: projection.weight_scale_inv}
    config = SimpleNamespace(
        num_attention_heads=64, num_key_value_heads=32, head_dim=32, v_head_dim=16
    )
    with loading_scope(changed):
        _resolve_deferred_qkv_scale_inv(
            params, deferred, ckpt_tp, config=config, model=model
        )
    first = rank_size(projection)[0] * (ckpt_tp // rank_size(projection)[1])
    last = first + ckpt_tp // rank_size(projection)[1]
    dequant = []
    for index in range(first, last):
        shard = full.chunk(ckpt_tp, dim=0)[index].float()
        shard_scale = scale.chunk(ckpt_tp, dim=0)[index]
        dequant.append(
            (shard * shard_scale.repeat_interleave(128, dim=0)[:448]).to(torch.bfloat16)
        )
    # Independently assemble the checkpoint Q, K and V windows.
    merged = torch.cat(
        [
            part
            for start, stop in ((0, 256), (256, 384), (384, 448))
            for shard in dequant
            for part in (shard[start:stop],)
        ],
        dim=0,
    )
    expected_weight = torch.empty_like(projection.weight)
    expected_scale = torch.empty_like(projection.weight_scale_inv)
    for start in range(0, merged.shape[0], 128):
        block = merged[start : start + 128].float()
        divisor = block.abs().max().clamp_min(1e-12) / 448
        expected_weight[start : start + 128] = (
            (block / divisor).clamp(-448, 448).to(torch.float8_e4m3fn)
        )
        expected_scale[start // 128, 0] = divisor
    torch.testing.assert_close(
        projection.weight.view(torch.uint8),
        expected_weight.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        projection.weight_scale_inv, expected_scale, rtol=0, atol=0
    )
    return projection.weight, projection.weight_scale_inv


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestMiMoFusedQkvLoaderLayout(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def check_loads(self, changed, wrapped=False):
        for dp in (1, 2):
            for rank in range(4):
                reset_context()
                publish(
                    ServerArgs(
                        model_path="dummy",
                        device="cuda",
                        tp_size=4,
                        attn_dp_size=dp,
                        fp8_gemm_runner_backend="triton",
                    ),
                    role="test",
                    ranks=SpawnRanks(world_rank=rank),
                )
                for dtype in (torch.float32, torch.bfloat16):
                    _, projection = build_projection(dtype)
                    if wrapped:
                        projection = BaseLayerWithLoRA(projection, Mock())
                    for ckpt_tp, sharded in (
                        (None, True),
                        (None, False),
                        (rank_size(unwrap_lora_layer(projection))[1], True),
                        (rank_size(unwrap_lora_layer(projection))[1], False),
                        (8, False),
                    ):
                        for offset in (0, 11):
                            load_fused(
                                projection,
                                changed=changed,
                                ckpt_tp=ckpt_tp,
                                sharded=sharded,
                                offset=offset,
                            )
                model, projection = build_projection(fp8=True)
                if wrapped:
                    projection = BaseLayerWithLoRA(projection, Mock())
                    model.model.layers[0].self_attn.qkv_proj = projection
                for offset in (0, 11):
                    load_deferred(model, projection, changed=changed, offset=offset)

    def test_native_projection_in_the_construction_scope(self):
        self.check_loads(False)

    def test_native_projection_after_scope_exit(self):
        self.check_loads(True)

    def test_wrapped_projection_and_deferred_scales_after_scope_exit(self):
        self.check_loads(True, wrapped=True)

    def check_main_and_mtp_callers(self, wrapped=False):
        for dp in (1, 2):
            for rank in range(4):
                reset_context()
                publish(
                    ServerArgs(
                        model_path="dummy", device="cuda", tp_size=4, attn_dp_size=dp
                    ),
                    role="test",
                    ranks=SpawnRanks(world_rank=rank),
                )
                for model_type in (MiMoV2ForCausalLM, MiMoV2MTP):
                    root, projection = build_projection()
                    model = model_type.__new__(model_type)
                    torch.nn.Module.__init__(model)
                    model.config = SimpleNamespace(
                        n_routed_experts=0,
                        encoder_only=False,
                        tie_word_embeddings=False,
                    )
                    model.quant_config = None
                    model._is_multimodal = False
                    model.model = root.model
                    if wrapped:
                        model.model.layers[0].self_attn.qkv_proj = BaseLayerWithLoRA(
                            projection, Mock()
                        )
                    name = WEIGHT_NAME
                    if model_type is MiMoV2MTP:
                        model.model.mtp_block = model.model.layers[0]
                        del model.model.layers
                        name = "model.mtp.layers.0.self_attn.qkv_proj.weight"
                    full = values((3584, 128), projection.weight.dtype)
                    expected = full.chunk(rank_size(projection)[1], dim=0)[
                        rank_size(projection)[0]
                    ]
                    for changed in (False, True):
                        with loading_scope(changed):
                            model.load_weights([(name, full)])
                        torch.testing.assert_close(
                            projection.weight, expected, rtol=0, atol=0
                        )

    def test_main_and_mtp_callers_pass_the_native_projection(self):
        self.check_main_and_mtp_callers()

    def test_main_and_mtp_callers_unwrap_the_projection(self):
        self.check_main_and_mtp_callers(wrapped=True)

    def test_rank_change_without_changing_partition_width(self):
        publish(
            ServerArgs(model_path="dummy", device="cuda", tp_size=4, attn_dp_size=2),
            role="test",
            ranks=SpawnRanks(world_rank=3),
        )
        _, projection = build_projection()
        model, fp8_projection = build_projection(fp8=True)
        with parallel_scope(tp_rank=0, attn_tp_rank=0, attn_dp_rank=0, moe_tp_rank=0):
            load_fused(projection, ckpt_tp=8)
            load_deferred(model, fp8_projection)

    def test_existing_layout_errors_use_the_projection_width(self):
        publish(
            ServerArgs(model_path="dummy", device="cuda", tp_size=4),
            role="test",
            ranks=SpawnRanks(world_rank=3),
        )
        _, projection = build_projection()
        full = values((3584, 128), projection.weight.dtype)
        for changed in (False, True):
            with loading_scope(changed):
                with self.assertRaisesRegex(ValueError, "TP=3-interleaved"):
                    load_mimo_v2_qkv_proj_weight(
                        WEIGHT_NAME, projection.weight, full, 3, qkv_proj=projection
                    )
                with self.assertRaisesRegex(ValueError, "unexpected shape"):
                    load_mimo_v2_qkv_proj_weight(
                        WEIGHT_NAME, projection.weight, full[:100], qkv_proj=projection
                    )
                with self.assertRaisesRegex(ValueError, "pass deferred_scale_inv"):
                    load_mimo_v2_qkv_proj_weight(
                        SCALE_NAME,
                        torch.nn.Parameter(torch.empty(7, 1)),
                        torch.ones(32, 1),
                        8,
                        qkv_proj=projection,
                    )


if __name__ == "__main__":
    unittest.main()
