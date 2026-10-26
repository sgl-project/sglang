"""Below the fused kv-head shard count, each MiMo-V2 rank holds its q, k and v heads in
that order, for BF16 weights and for block-FP8 codes regrouped with their scales."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.models.mimo_v2 as mimo_v2
from sglang.srt.models.mimo_v2_nextn import MiMoV2MTP
from sglang.test.test_utils import CustomTestCase

SHARDS = 4
HIDDEN = 8
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
# layer 0: global attention, one kv head per shard; layer 1: SWA, two
CONFIG = SimpleNamespace(
    hybrid_layer_pattern=[0, 1],
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=3,
    v_head_dim=2,
    swa_num_attention_heads=8,
    swa_num_key_value_heads=8,
    swa_head_dim=3,
    swa_v_head_dim=2,
)


def _qkv(layer):
    prefix = "swa_" if CONFIG.hybrid_layer_pattern[layer] else ""
    heads, kv_heads, head_dim, v_head_dim = (
        getattr(CONFIG, prefix + key)
        for key in (
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "v_head_dim",
        )
    )
    return (
        torch.randn(heads * head_dim, HIDDEN),
        torch.randn(kv_heads * head_dim, HIDDEN),
        torch.randn(kv_heads * v_head_dim, HIDDEN),
    )


def _interleave(q, k, v):
    return torch.cat(
        [
            torch.cat(parts)
            for parts in zip(
                q.chunk(SHARDS), k.chunk(SHARDS), v.chunk(SHARDS), strict=True
            )
        ]
    )


def _rank_rows(q, k, v, tp_size, tp_rank):
    return torch.cat([t.chunk(tp_size)[tp_rank] for t in (q, k, v)])


# The loader reads rank and size from the destination projection's TP group.
_QKV_PROJ = SimpleNamespace(tp_group=None)
_MODEL = SimpleNamespace(get_submodule=lambda name: _QKV_PROJ)


def _parallel(tp_size, tp_rank):
    return patch.object(mimo_v2, "get_group_rank_size", return_value=(tp_rank, tp_size))


def _param(rows, dtype):
    return torch.nn.Parameter(
        torch.empty(rows, HIDDEN, dtype=dtype), requires_grad=False
    )


def _fp8_checkpoint(layer):
    """(codes, scales, q, k, v): fused block-FP8 qkv and the q, k, v rows it decodes to."""
    # every shard fits one 128x128 block, so each carries its own scale row
    q, k, v = _qkv(layer)
    shards = _interleave(q, k, v).chunk(SHARDS)
    scales = torch.stack([s.abs().amax() / FP8_MAX for s in shards]).view(-1, 1)
    codes = torch.cat(
        [(s / scale).to(torch.float8_e4m3fn) for s, scale in zip(shards, scales)]
    )
    decoded = torch.cat(
        [c.float() * scale for c, scale in zip(codes.chunk(SHARDS), scales)]
    )
    sizes = [t.shape[0] // SHARDS for t in (q, k, v)]
    parts = [torch.split(shard, sizes) for shard in decoded.chunk(SHARDS)]
    return (codes, scales, *(torch.cat(group) for group in zip(*parts)))


def _assert_fp8_rows(test, weight, scale_inv, want, msg):
    got = weight.data.float() * scale_inv.data
    # the resolver re-quantizes the rank's rows as one block
    tolerance = 0.07 * want.abs().max()
    test.assertLessEqual((got - want).abs().max().item(), tolerance.item(), msg)


def _mtp_model(rows):
    """A MiMoV2MTP holding only its MTP layer's qkv projection, as block-FP8."""
    model = MiMoV2MTP.__new__(MiMoV2MTP)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        **vars(CONFIG),
        attention_projection_layout="fused_qkv",
        tie_word_embeddings=False,
    )
    projection = torch.nn.Module()
    projection.tp_group = None
    projection.weight = _param(rows, torch.float8_e4m3fn)
    projection.weight_scale_inv = torch.nn.Parameter(
        torch.empty(1, 1), requires_grad=False
    )
    model.model = torch.nn.Module()
    model.model.mtp_block = torch.nn.Module()
    model.model.mtp_block.self_attn = torch.nn.Module()
    model.model.mtp_block.self_attn.qkv_proj = projection
    return model, projection


class TestMiMoV2FusedQKVLoader(CustomTestCase):
    def test_bf16_rank_holds_its_q_k_v_heads_at_every_tp(self):
        for layer in (0, 1):
            q, k, v = _qkv(layer)
            fused = _interleave(q, k, v).to(torch.bfloat16)
            for tp_size in (1, 2, 4):
                for tp_rank in range(tp_size):
                    param = _param(fused.shape[0] // tp_size, torch.bfloat16)
                    with _parallel(tp_size, tp_rank):
                        mimo_v2.load_mimo_v2_qkv_proj_weight(
                            f"model.layers.{layer}.self_attn.qkv_proj.weight",
                            param,
                            fused,
                            expected_fused_tp_size=SHARDS,
                            deferred_scale_inv={},
                            qkv_proj=_QKV_PROJ,
                            config=CONFIG,
                        )
                    want = _rank_rows(q, k, v, tp_size, tp_rank).to(torch.bfloat16)
                    self.assertTrue(
                        torch.equal(param.data, want),
                        f"layer {layer} tp {tp_size} rank {tp_rank}",
                    )

    def test_block_fp8_regroups_with_its_deferred_scales(self):
        for layer in (0, 1):
            codes, scales, dq, dk, dv = _fp8_checkpoint(layer)
            for tp_size in (1, 2):
                for tp_rank in range(tp_size):
                    name = f"model.layers.{layer}.self_attn.qkv_proj.weight"
                    params = {
                        name: _param(codes.shape[0] // tp_size, torch.float8_e4m3fn),
                        name + "_scale_inv": torch.nn.Parameter(
                            torch.empty(1, 1), requires_grad=False
                        ),
                    }
                    deferred = {}
                    with _parallel(tp_size, tp_rank):
                        for tensor_name, tensor in (
                            (name, codes),
                            (name + "_scale_inv", scales),
                        ):
                            mimo_v2.load_mimo_v2_qkv_proj_weight(
                                tensor_name,
                                params[tensor_name],
                                tensor,
                                expected_fused_tp_size=SHARDS,
                                deferred_scale_inv=deferred,
                                qkv_proj=_QKV_PROJ,
                                config=CONFIG,
                            )
                        mimo_v2._resolve_deferred_qkv_scale_inv(
                            params,
                            deferred,
                            expected_fused_tp_size=SHARDS,
                            config=CONFIG,
                            model=_MODEL,
                        )
                    _assert_fp8_rows(
                        self,
                        params[name],
                        params[name + "_scale_inv"],
                        _rank_rows(dq, dk, dv, tp_size, tp_rank),
                        f"layer {layer} tp {tp_size} rank {tp_rank}",
                    )

    def test_mtp_layer_block_fp8_loads_below_fused_tp(self):
        """The MTP draft's block-FP8 qkv once failed to load below the fused shard count
        (scale shape mismatch), so EAGLE drafting from the MTP layers could not start at
        attention TP 1 or 2. The MTP layer has sliding-window attention dimensions."""
        codes, scales, dq, dk, dv = _fp8_checkpoint(1)
        prefix = "model.mtp.layers.0.self_attn.qkv_proj."
        for tp_size in (1, 2):
            for tp_rank in range(tp_size):
                model, projection = _mtp_model(codes.shape[0] // tp_size)
                with _parallel(tp_size, tp_rank):
                    model.load_weights(
                        [
                            (prefix + "weight", codes),
                            (prefix + "weight_scale_inv", scales),
                        ]
                    )
                _assert_fp8_rows(
                    self,
                    projection.weight,
                    projection.weight_scale_inv,
                    _rank_rows(dq, dk, dv, tp_size, tp_rank),
                    f"tp {tp_size} rank {tp_rank}",
                )

    def test_bf16_without_layer_sizes_raises(self):
        q, k, v = _qkv(1)
        fused = _interleave(q, k, v).to(torch.bfloat16)
        with _parallel(2, 0), self.assertRaises(ValueError):
            mimo_v2.load_mimo_v2_qkv_proj_weight(
                "model.layers.1.self_attn.qkv_proj.weight",
                _param(fused.shape[0] // 2, torch.bfloat16),
                fused,
                expected_fused_tp_size=SHARDS,
                deferred_scale_inv={},
                qkv_proj=_QKV_PROJ,
                config=None,
            )


if __name__ == "__main__":
    unittest.main()
