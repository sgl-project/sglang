"""Block-FP8 MoE alignment padding and tail-masked group quant for unaligned TP shards."""

import socket
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe as fused_moe_module
import sglang.srt.layers.quantization.fp8 as fp8_module
from sglang.kernels.ops.quantization.fp8_kernel import (
    _per_token_group_quant_8bit_raw,
    fp8_dtype,
    fp8_max,
    sglang_per_token_group_quant_fp8,
)
from sglang.srt import runtime_context
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe as fm
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
    try_get_optimal_moe_config,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.fp8 import (
    Fp8Config,
    Fp8MoEMethod,
    _compute_block_quant_alignment,
)
from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant
from sglang.srt.model_executor.weight_checker_comparator import (
    select_comparable_weight,
)
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.layer_ut_utils import assert_output_close, init_single_process_dist
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-a", runner_config="1-gpu-small")


def _quantize_block_fp8(
    weight: torch.Tensor, block_size: tuple[int, int] = (128, 128)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 3D weight [E, M, K] into FP8 E4M3 with [block_m, block_k] scales."""
    e, m, k = weight.shape
    bm, bk = block_size
    assert m % bm == 0 and k % bk == 0, f"Shape {(m, k)} must divide {(bm, bk)}"
    w_blocks = weight.float().view(e, m // bm, bm, k // bk, bk)
    amax = w_blocks.abs().amax(dim=(2, 4))
    scale = torch.where(amax > 0, amax / float(fp8_max), torch.ones_like(amax))
    q = (
        (w_blocks / scale[:, :, None, :, None])
        .clamp(-float(fp8_max), float(fp8_max))
        .to(fp8_dtype)
        .reshape(e, m, k)
    )
    return q, scale


def _reference_moe_bf16(
    x: torch.Tensor,
    w1_deq: torch.Tensor,
    w3_deq: torch.Tensor,
    w2_deq: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    b1: torch.Tensor | None = None,
    b3: torch.Tensor | None = None,
    b2: torch.Tensor | None = None,
) -> torch.Tensor:
    out = torch.zeros_like(x, dtype=torch.float32)
    m, topk = topk_ids.shape
    for t in range(m):
        xt = x[t].float()
        acc = torch.zeros_like(xt)
        for k_idx in range(topk):
            e = int(topk_ids[t, k_idx])
            w = float(topk_weights[t, k_idx])
            g = F.linear(
                xt, w1_deq[e].float(), b1[e].float() if b1 is not None else None
            )
            u = F.linear(
                xt, w3_deq[e].float(), b3[e].float() if b3 is not None else None
            )
            d = F.linear(
                F.silu(g) * u,
                w2_deq[e].float(),
                b2[e].float() if b2 is not None else None,
            )
            acc += w * d
        out[t] = acc
    return out


class TestMoeFp8UnalignedTp(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        init_single_process_dist(master_port=port)

    @contextmanager
    def _runtime(self, backend, tp_size, tp_rank):
        with (
            get_context().override_server_args(model_path="dummy"),
            get_flags().moe.override(runner_backend=backend),
            patch.object(runtime_context, "_validate_parallel"),
            get_parallel().override(
                tp_size=tp_size,
                tp_rank=tp_rank,
                attn_tp_size=tp_size,
                attn_tp_rank=tp_rank,
                moe_ep_size=1,
                moe_ep_rank=0,
                moe_tp_size=tp_size,
                moe_tp_rank=tp_rank,
                tp_group=None,
                attn_tp_group=None,
            ),
        ):
            yield

    @staticmethod
    def _load_fused_moe_checkpoint(layer, gate_q, gate_s, up_q, up_s, down_q, down_s):
        for e in range(layer.num_experts):
            layer.weight_loader(
                layer.w13_weight, gate_q[e], f"experts.{e}.gate_proj.weight", "w1", e
            )
            layer.weight_loader(
                layer.w13_weight_scale_inv,
                gate_s[e],
                f"experts.{e}.gate_proj.weight_scale_inv",
                "w1",
                e,
            )
            layer.weight_loader(
                layer.w13_weight, up_q[e], f"experts.{e}.up_proj.weight", "w3", e
            )
            layer.weight_loader(
                layer.w13_weight_scale_inv,
                up_s[e],
                f"experts.{e}.up_proj.weight_scale_inv",
                "w3",
                e,
            )
            layer.weight_loader(
                layer.w2_weight, down_q[e], f"experts.{e}.down_proj.weight", "w2", e
            )
            layer.weight_loader(
                layer.w2_weight_scale_inv,
                down_s[e],
                f"experts.{e}.down_proj.weight_scale_inv",
                "w2",
                e,
            )
        layer.quant_method.process_weights_after_loading(layer)

    def _verify_multi_rank_equivalence(
        self,
        *,
        backend,
        e_cnt: int,
        topk: int,
        h: int,
        global_inter: int,
        tp_sizes: tuple[int, ...],
        seed: int,
        max_rel_err_by_tp: dict[int, float],
    ):
        torch.manual_seed(seed)
        gate_bf16 = (
            torch.randn((e_cnt, global_inter, h), dtype=torch.bfloat16, device="cuda")
            * 0.02
        )
        up_bf16 = (
            torch.randn((e_cnt, global_inter, h), dtype=torch.bfloat16, device="cuda")
            * 0.02
        )
        down_bf16 = (
            torch.randn((e_cnt, h, global_inter), dtype=torch.bfloat16, device="cuda")
            * 0.02
        )
        gate_q, gate_s = _quantize_block_fp8(gate_bf16, (128, 128))
        up_q, up_s = _quantize_block_fp8(up_bf16, (128, 128))
        down_q, down_s = _quantize_block_fp8(down_bf16, (128, 128))

        quant_cfg = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )

        x = torch.randn((32, h), dtype=torch.bfloat16, device="cuda") * 0.5
        logits = torch.randn((32, e_cnt), dtype=torch.float32, device="cuda")
        topk_weights, topk_ids = torch.topk(logits, topk, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1).to(torch.float32)
        topk_ids = topk_ids.to(torch.int32)
        topk_out = StandardTopKOutput(topk_weights, topk_ids, None)

        with self._runtime(backend, tp_size=1, tp_rank=0):
            layer_tp1 = FusedMoE(
                num_experts=e_cnt,
                hidden_size=h,
                intermediate_size=global_inter,
                layer_id=0,
                top_k=topk,
                params_dtype=torch.bfloat16,
                reduce_results=False,
                quant_config=quant_cfg,
                inplace=False,
            ).cuda()
            self._load_fused_moe_checkpoint(
                layer_tp1, gate_q, gate_s, up_q, up_s, down_q, down_s
            )
            with torch.no_grad():
                y_tp1 = layer_tp1(x.clone(), topk_out).float()

        tp1_w13_deq = block_quant_dequant(
            layer_tp1.w13_weight.data,
            layer_tp1.w13_weight_scale_inv.data,
            [128, 128],
            torch.float32,
        )
        tp1_w2_deq = block_quant_dequant(
            layer_tp1.w2_weight.data,
            layer_tp1.w2_weight_scale_inv.data,
            [128, 128],
            torch.float32,
        )

        for tp_size in tp_sizes:
            unpadded_n = global_inter // tp_size
            y_tp_sum = torch.zeros_like(y_tp1)
            for r in range(tp_size):
                with self._runtime(backend, tp_size=tp_size, tp_rank=r):
                    layer_r = FusedMoE(
                        num_experts=e_cnt,
                        hidden_size=h,
                        intermediate_size=global_inter,
                        layer_id=0,
                        top_k=topk,
                        params_dtype=torch.bfloat16,
                        reduce_results=False,
                        quant_config=quant_cfg,
                        inplace=False,
                    ).cuda()
                    self._load_fused_moe_checkpoint(
                        layer_r, gate_q, gate_s, up_q, up_s, down_q, down_s
                    )
                    self.assertIsNone(layer_r.quant_method.w2_weight_block_size)
                    self.assertTrue(layer_r.quant_method.weights_padded_for_block_quant)
                    with torch.no_grad():
                        y_tp_sum += layer_r(x.clone(), topk_out).float()

                n_aligned, _, lead_pad, can_shift = _compute_block_quant_alignment(
                    unpadded_n, [128, 128], tp_rank=r
                )
                r_w13_deq = block_quant_dequant(
                    layer_r.w13_weight.data,
                    layer_r.w13_weight_scale_inv.data,
                    [128, 128],
                    torch.float32,
                )
                r_w2_deq = block_quant_dequant(
                    layer_r.w2_weight.data,
                    layer_r.w2_weight_scale_inv.data,
                    [128, 128],
                    torch.float32,
                )
                g0, g1 = r * unpadded_n, (r + 1) * unpadded_n
                w1_slice = r_w13_deq[:, lead_pad : lead_pad + unpadded_n, :]
                w3_slice = r_w13_deq[
                    :,
                    n_aligned + lead_pad : n_aligned + lead_pad + unpadded_n,
                    :,
                ]
                w2_slice = r_w2_deq[:, :, lead_pad : lead_pad + unpadded_n]
                w1_ref = tp1_w13_deq[:, g0:g1, :]
                w3_ref = tp1_w13_deq[:, global_inter + g0 : global_inter + g1, :]
                w2_ref = tp1_w2_deq[:, :, g0:g1]
                if can_shift:
                    self.assertEqual(float((w1_slice - w1_ref).abs().max()), 0.0)
                    self.assertEqual(float((w3_slice - w3_ref).abs().max()), 0.0)
                    self.assertEqual(float((w2_slice - w2_ref).abs().max()), 0.0)
                else:
                    for got_w, ref_w in (
                        (w1_slice, w1_ref),
                        (w3_slice, w3_ref),
                        (w2_slice, w2_ref),
                    ):
                        w_rel = float((got_w - ref_w).norm() / ref_w.norm())
                        self.assertLess(w_rel, 0.04)

            assert_output_close(self, y_tp_sum, y_tp1, cos_threshold=0.998)
            rel_err = float((y_tp_sum - y_tp1).norm() / y_tp1.norm())
            self.assertLess(rel_err, max_rel_err_by_tp[tp_size])

    def test_per_token_group_quant_fp8_unaligned_k(self):
        torch.manual_seed(0)
        group_size = 128
        for k in (192, 96, 160):
            k_pad = ((k + group_size - 1) // group_size) * group_size
            num_groups = (k + group_size - 1) // group_size
            for col_major in (False, True):
                # M=0 empty batch must succeed cleanly without launching a (0,) grid.
                x_empty = torch.empty((0, k), dtype=torch.bfloat16, device="cuda")
                q_empty, s_empty = sglang_per_token_group_quant_fp8(
                    x_empty, group_size, column_major_scales=col_major
                )
                self.assertEqual(tuple(q_empty.shape), (0, k))
                self.assertEqual(tuple(s_empty.shape), (0, num_groups))

                for m in (1, 16, 256):
                    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
                    x_pad = torch.zeros((m, k_pad), dtype=torch.bfloat16, device="cuda")
                    x_pad[:, :k] = x

                    q_out, s_out = sglang_per_token_group_quant_fp8(
                        x, group_size, column_major_scales=col_major
                    )
                    q_ref_raw, s_ref_raw = _per_token_group_quant_8bit_raw(
                        x_pad, group_size, column_major_scales=col_major
                    )
                    self.assertEqual(tuple(q_out.shape), (m, k))
                    self.assertEqual(tuple(s_out.shape), (m, num_groups))
                    self.assertEqual(
                        float((q_out.float() - q_ref_raw[:, :k].float()).abs().max()),
                        0.0,
                    )
                    self.assertLess(
                        float((s_out.float() - s_ref_raw.float()).abs().max()),
                        1e-7,
                    )

                    q_cuda_pad, s_cuda_pad = sglang_per_token_group_quant_fp8(
                        x_pad, group_size, column_major_scales=col_major
                    )
                    deq_out = (
                        q_out.float()
                        * s_out.repeat_interleave(group_size, dim=1)[:, :k]
                    )
                    deq_cuda = (
                        q_cuda_pad[:, :k].float()
                        * s_cuda_pad.repeat_interleave(group_size, dim=1)[:, :k]
                    )
                    rel_err = float((deq_out - deq_cuda).norm() / deq_cuda.norm())
                    self.assertLess(rel_err, 1e-2)

    def test_triton_fused_experts_unaligned_direct(self):
        torch.manual_seed(1)
        e_cnt, topk, h = 16, 4, 512
        block_shape = [128, 128]

        def with_down_tma(*args, **kwargs):
            config, (down_config, extra) = try_get_optimal_moe_config(*args, **kwargs)
            return config, ({**(down_config or config), "USE_TMA": True}, extra)

        for n in (192, 96):
            n_aligned = ((n + 127) // 128) * 128
            w1_raw = (
                torch.randn((e_cnt, 2 * n, h), dtype=torch.bfloat16, device="cuda")
                * 0.02
            )
            w2_raw = (
                torch.randn((e_cnt, h, n), dtype=torch.bfloat16, device="cuda") * 0.02
            )

            # Zero-pad BF16 weights per gate/up shard for bit-exact reference.
            w1_bf16_pad = torch.zeros(
                (e_cnt, 2 * n_aligned, h), dtype=torch.bfloat16, device="cuda"
            )
            w1_bf16_pad[:, :n, :] = w1_raw[:, :n, :]
            w1_bf16_pad[:, n_aligned : n_aligned + n, :] = w1_raw[:, n:, :]
            w2_bf16_pad = torch.zeros(
                (e_cnt, h, n_aligned), dtype=torch.bfloat16, device="cuda"
            )
            w2_bf16_pad[:, :, :n] = w2_raw

            cfg_unpad = MoeRunnerConfig(
                num_experts=e_cnt,
                num_local_experts=e_cnt,
                hidden_size=h,
                intermediate_size_per_partition=n,
                top_k=topk,
                params_dtype=torch.bfloat16,
                inplace=False,
            )
            cfg_pad = MoeRunnerConfig(
                num_experts=e_cnt,
                num_local_experts=e_cnt,
                hidden_size=h,
                intermediate_size_per_partition=n_aligned,
                top_k=topk,
                params_dtype=torch.bfloat16,
                inplace=False,
            )

            x = torch.randn((16, h), dtype=torch.bfloat16, device="cuda")
            logits = torch.randn((16, e_cnt), dtype=torch.float32, device="cuda")
            topk_weights, topk_ids = torch.topk(logits, topk, dim=-1)
            topk_weights = F.softmax(topk_weights, dim=-1).to(torch.float32)
            topk_ids = topk_ids.to(torch.int32)
            topk_out = StandardTopKOutput(topk_weights, topk_ids, None)

            out_bf16_unpad = fm.fused_experts(x, w1_raw, w2_raw, topk_out, cfg_unpad)
            out_bf16_pad = fm.fused_experts(
                x, w1_bf16_pad, w2_bf16_pad, topk_out, cfg_pad
            )
            self.assertEqual(
                float((out_bf16_unpad.float() - out_bf16_pad.float()).abs().max()),
                0.0,
            )

            # TMA down-projection must stay gated off when w2 K-dim is unaligned.
            with (
                patch.object(fused_moe_module, "_moe_support_tma", return_value=True),
                patch.object(
                    fused_moe_module, "try_get_optimal_moe_config", with_down_tma
                ),
            ):
                _, _, down_tma_unaligned, _, _, _, _ = fm._prepare_fused_moe_run(
                    x,
                    w1_raw.to(fp8_dtype),
                    w2_raw.to(fp8_dtype),
                    topk_ids,
                    use_fp8_w8a8=True,
                    use_int8_w8a8=False,
                    use_int8_w8a16=False,
                    use_int4_w4a16=False,
                    per_channel_quant=False,
                    block_shape=block_shape,
                )
            self.assertFalse(down_tma_unaligned)

            # Compare padded (N_aligned) vs unpadded (N with tail-masked group quant).
            w1_fp8_pad, s1_fp8_pad = _quantize_block_fp8(w1_bf16_pad, (128, 128))
            w2_fp8_pad, s2_fp8_pad = _quantize_block_fp8(w2_bf16_pad, (128, 128))
            const_s1 = float(s1_fp8_pad.mean())
            s1_const_pad = torch.full_like(s1_fp8_pad, const_s1)
            s1_const_raw = torch.full(
                (e_cnt, (2 * n + 127) // 128, h // 128), const_s1, device="cuda"
            )
            w1_fp8_raw = torch.empty((e_cnt, 2 * n, h), dtype=fp8_dtype, device="cuda")
            w1_fp8_raw[:, :n, :] = w1_fp8_pad[:, :n, :]
            w1_fp8_raw[:, n:, :] = w1_fp8_pad[:, n_aligned : n_aligned + n, :]
            w2_fp8_unpad = w2_fp8_pad[:, :, :n].contiguous()

            out_padded = fm.fused_experts(
                x,
                w1_fp8_pad,
                w2_fp8_pad,
                topk_out,
                cfg_pad,
                use_fp8_w8a8=True,
                w1_scale=s1_const_pad,
                w2_scale=s2_fp8_pad,
                block_shape=block_shape,
            )
            out_unpadded = fm.fused_experts(
                x,
                w1_fp8_raw,
                w2_fp8_unpad,
                topk_out,
                cfg_unpad,
                use_fp8_w8a8=True,
                w1_scale=s1_const_raw,
                w2_scale=s2_fp8_pad,
                block_shape=block_shape,
            )
            rel_err = float(
                (out_padded.float() - out_unpadded.float()).norm()
                / out_padded.float().norm()
            )
            self.assertLess(rel_err, 1e-2)

    def test_fp8_moe_method_align_with_bias(self):
        torch.manual_seed(2)
        e_cnt, topk, h = 8, 2, 256
        quant_cfg = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )

        # (tp=8, n=80) has gcd=16 < 32 (never refined); (tp=4, n=192) exercises
        # the unrefined/Hopper path with lead_pad=64 when block refinement is off.
        for tp_size, n, ranks in ((8, 80, (0, 1, 2)), (4, 192, (0, 1))):
            global_inter = tp_size * n
            # Quantize padded global tensor cleanly along 128-blocks when needed.
            pad_global = ((global_inter + 127) // 128) * 128
            w1_global = (
                torch.randn((e_cnt, pad_global, h), dtype=torch.bfloat16, device="cuda")
                * 0.02
            )
            w3_global = (
                torch.randn((e_cnt, pad_global, h), dtype=torch.bfloat16, device="cuda")
                * 0.02
            )
            w2_global = (
                torch.randn((e_cnt, h, pad_global), dtype=torch.bfloat16, device="cuda")
                * 0.02
            )
            w1_q, w1_s = _quantize_block_fp8(w1_global, (128, 128))
            w3_q, w3_s = _quantize_block_fp8(w3_global, (128, 128))
            w2_q, w2_s = _quantize_block_fp8(w2_global, (128, 128))

            for tp_rank in ranks:
                g0 = tp_rank * n
                g1 = g0 + n
                b0 = g0 // 128
                b1 = (g1 + 127) // 128

                with (
                    self._runtime(MoeRunnerBackend.TRITON, tp_size, tp_rank),
                    patch.object(
                        fp8_module, "_tp_refined_moe_block_sizes", return_value=None
                    ),
                ):
                    layer = torch.nn.Module()
                    layer.moe_ep_size = 1
                    layer.moe_tp_size = tp_size
                    layer.moe_tp_rank = tp_rank
                    layer.use_presharded_weights = False
                    layer.allow_fp8_blockwise_tail = False
                    layer.num_experts = e_cnt
                    layer.num_local_experts = e_cnt
                    layer.hidden_size = h
                    layer.intermediate_size_per_partition = n
                    layer.moe_runner_config = MoeRunnerConfig(
                        num_experts=e_cnt,
                        num_local_experts=e_cnt,
                        hidden_size=h,
                        intermediate_size_per_partition=n,
                        top_k=topk,
                        params_dtype=torch.bfloat16,
                        inplace=False,
                        is_gated=True,
                    )
                    layer.dispatcher = SimpleNamespace(
                        expert_mask_gpu=None,
                        set_quant_config=lambda _cfg: None,
                    )

                    method = Fp8MoEMethod(quant_cfg)
                    method.create_weights(
                        layer=layer,
                        num_experts=e_cnt,
                        hidden_size=h,
                        intermediate_size_per_partition=n,
                        params_dtype=torch.bfloat16,
                        with_bias=True,
                    )
                    method.create_moe_runner(layer, layer.moe_runner_config)
                    layer.to("cuda")

                    layer.w13_weight.data[:, :n, :].copy_(w1_q[:, g0:g1, :])
                    layer.w13_weight.data[:, n:, :].copy_(w3_q[:, g0:g1, :])
                    layer.w13_weight_scale_inv.data[:, : (b1 - b0), :].copy_(
                        w1_s[:, b0:b1, :]
                    )
                    layer.w13_weight_scale_inv.data[:, (b1 - b0) :, :].copy_(
                        w3_s[:, b0:b1, :]
                    )
                    layer.w2_weight.data.copy_(w2_q[:, :, g0:g1])
                    layer.w2_weight_scale_inv.data.copy_(w2_s[:, :, b0:b1])

                    b1_shard = (
                        torch.randn((e_cnt, n), dtype=torch.float32, device="cuda")
                        * 0.05
                    )
                    b3_shard = (
                        torch.randn((e_cnt, n), dtype=torch.float32, device="cuda")
                        * 0.05
                    )
                    b2_shard = (
                        torch.randn((e_cnt, h), dtype=torch.float32, device="cuda")
                        * 0.05
                    )
                    layer.w13_weight_bias.data[:, :n].copy_(b1_shard)
                    layer.w13_weight_bias.data[:, n:].copy_(b3_shard)
                    layer.w2_weight_bias.data.copy_(b2_shard)

                    w1_ref_deq = block_quant_dequant(
                        w1_q, w1_s, [128, 128], torch.float32
                    )[:, g0:g1, :]
                    w3_ref_deq = block_quant_dequant(
                        w3_q, w3_s, [128, 128], torch.float32
                    )[:, g0:g1, :]
                    w2_ref_deq = block_quant_dequant(
                        w2_q, w2_s, [128, 128], torch.float32
                    )[:, :, g0:g1]

                    method.process_weights_after_loading(layer)

                    self.assertTrue(method.weights_padded_for_block_quant)
                    with self.assertRaises(NotImplementedError):
                        select_comparable_weight(method)

                    n_aligned, _, lead_pad, _ = _compute_block_quant_alignment(
                        n, [128, 128], tp_rank=tp_rank
                    )
                    self.assertEqual(layer.intermediate_size_per_partition, n_aligned)
                    self.assertEqual(layer.w13_weight.shape, (e_cnt, 2 * n_aligned, h))
                    self.assertEqual(layer.w2_weight.shape, (e_cnt, h, n_aligned))
                    self.assertEqual(
                        layer.w13_weight_bias.shape, (e_cnt, 2 * n_aligned)
                    )
                    if lead_pad > 0:
                        self.assertEqual(
                            float(layer.w13_weight_bias[:, :lead_pad].abs().max()), 0.0
                        )
                        self.assertEqual(
                            float(
                                layer.w13_weight_bias[
                                    :, n_aligned : n_aligned + lead_pad
                                ]
                                .abs()
                                .max()
                            ),
                            0.0,
                        )

                    x = torch.randn((8, h), dtype=torch.bfloat16, device="cuda") * 0.5
                    logits = torch.randn((8, e_cnt), dtype=torch.float32, device="cuda")
                    topk_weights, topk_ids = torch.topk(logits, topk, dim=-1)
                    topk_weights = F.softmax(topk_weights, dim=-1).to(torch.float32)
                    topk_ids = topk_ids.to(torch.int32)

                    out = fm.fused_experts(
                        x,
                        layer.w13_weight,
                        layer.w2_weight,
                        StandardTopKOutput(topk_weights, topk_ids, None),
                        layer.moe_runner_config,
                        use_fp8_w8a8=True,
                        w1_scale=layer.w13_weight_scale_inv,
                        w2_scale=layer.w2_weight_scale_inv,
                        b1=layer.w13_weight_bias,
                        b2=layer.w2_weight_bias,
                        block_shape=[128, 128],
                    )
                    ref = _reference_moe_bf16(
                        x,
                        w1_ref_deq,
                        w3_ref_deq,
                        w2_ref_deq,
                        topk_weights,
                        topk_ids,
                        b1=b1_shard,
                        b3=b3_shard,
                        b2=b2_shard,
                    )
                    assert_output_close(self, out, ref, cos_threshold=0.995)
                    rel_err = float((out.float() - ref).norm() / ref.norm())
                    self.assertLess(rel_err, 0.04)

    def test_triton_unrefined_gcd16_tp8_equivalence(self):
        # I=640 at TP=8 gives shard=80 where gcd(128, 80)=16 < 32, so block-scale
        # refinement returns None even on Triton and relies on alignment padding.
        self._verify_multi_rank_equivalence(
            backend=MoeRunnerBackend.TRITON,
            e_cnt=8,
            topk=2,
            h=256,
            global_inter=640,
            tp_sizes=(8,),
            seed=3,
            max_rel_err_by_tp={8: 0.05},
        )

    @unittest.skipUnless(
        deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
        and not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        "needs Hopper SM90 DeepGEMM without UE8M0 scales",
    )
    def test_hopper_deep_gemm_tp4_tp8_equivalence(self):
        # Hopper SM90 DeepGEMM requires 128-element weight alignment because
        # DEEPGEMM_SCALE_UE8M0 is False and finer K=32/64 block refinement is N/A.
        self._verify_multi_rank_equivalence(
            backend=MoeRunnerBackend.DEEP_GEMM,
            e_cnt=16,
            topk=4,
            h=512,
            global_inter=768,
            tp_sizes=(4, 8),
            seed=4,
            max_rel_err_by_tp={4: 0.03, 8: 0.05},
        )


if __name__ == "__main__":
    unittest.main()
