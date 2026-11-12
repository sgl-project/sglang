"""LSE-only DCP: dense attention reference and projection-after-correction."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.dcp import cp_lse_ag_out
from sglang.srt.layers.dcp.projection import (
    gather_block_fp8_linear,
    prepare_dcp_output_projections,
)
from sglang.srt.layers.linear import RowParallelLinear
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
    DeepseekMLAForwardMixin,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")


def _prepare(module, group, *, is_mla=True, num_kv_heads=1):
    return prepare_dcp_output_projections(
        model=module,
        group=group,
        is_mla=is_mla,
        decode_attention_backend="triton",
        num_kv_heads=num_kv_heads,
    )


class GatherOnlyGroup:
    def __init__(self, world_size, rank, gathered):
        self.world_size = world_size
        self.rank_in_group = rank
        self.gathered = iter(gathered)
        self.calls = []

    def all_gather(self, tensor, dim):
        self.calls.append((tensor.shape, dim))
        return next(self.gathered)

    def all_reduce(self, *args, **kwargs):
        raise AssertionError("Attention must not reduce outputs in ag_delay_reduce")

    reduce_scatter_along_dim = all_reduce
    all_to_all_single = all_reduce


class AttentionProjection(torch.nn.Module, DeepseekMLAForwardMixin):
    def __init__(self, local_heads, dim, hidden, values=None, kv_heads=1):
        super().__init__()
        self.attn = RadixAttention(
            num_heads=local_heads,
            head_dim=dim,
            scaling=dim**-0.5,
            num_kv_heads=kv_heads,
            layer_id=0,
        )
        self.num_local_heads = local_heads
        self.v_head_dim = dim
        self.use_deep_gemm_bmm = False
        self.o_proj = RowParallelLinear(
            local_heads * dim,
            hidden,
            bias=False,
            params_dtype=torch.float32,
            parallel_group="replicated",
        ).cuda()
        if values is not None:
            self.w_vc = values


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestDcpLseOnly(unittest.TestCase):
    def setUp(self):
        self.enterContext(get_parallel().override(tp_group=None))
        tf32 = torch.backends.cuda.matmul.allow_tf32
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", tf32)
        torch.backends.cuda.matmul.allow_tf32 = False

    def _dense_reference(self, world, mla, dtype):
        torch.manual_seed(42)
        batch, heads, q_dim, out_dim, hidden = 3, 2 * world, 32, 80, 96
        tokens, kv_heads = 17, 2
        latent = 64 if mla else out_dim
        q = torch.randn(batch, heads, q_dim, device="cuda")
        k = torch.randn(tokens, kv_heads, q_dim, device="cuda")
        v = torch.randn(tokens, kv_heads, latent, device="cuda")
        head_to_kv = torch.arange(heads, device="cuda") // (heads // kv_heads)
        scores = torch.einsum("bhd,thd->bht", q, k[:, head_to_kv]) * q_dim**-0.5
        values = v[:, head_to_kv]
        dense = torch.einsum("bht,thd->bhd", scores.softmax(-1), values)
        v_weights = (
            torch.randn(heads, latent, out_dim, device="cuda", dtype=dtype) * 0.1
            if mla
            else None
        )
        if mla:
            dense = torch.bmm(dense.transpose(0, 1), v_weights.float()).transpose(0, 1)
        o_weights = (
            torch.randn(hidden, heads * out_dim, device="cuda", dtype=dtype) * 0.1
        )
        expected = torch.nn.functional.linear(dense.flatten(1), o_weights.float())

        partials, lses = [], []
        for rank in range(world):
            s = scores[..., rank::world]
            partials.append(
                torch.einsum("bht,thd->bhd", s.softmax(-1), values[rank::world]).to(
                    dtype
                )
            )
            lses.append(s.logsumexp(-1))
        gathered_lses = torch.cat(lses, dim=0)
        projected = []
        for rank in range(world):
            start, end = rank * 2, (rank + 1) * 2
            module = AttentionProjection(
                2,
                out_dim,
                hidden,
                v_weights[start:end] if mla else None,
            )
            module.o_proj.weight.data = o_weights[
                :, start * out_dim : end * out_dim
            ].contiguous()
            gathers = [o_weights]
            if mla:
                gathers.append(v_weights)
            gathers.append(gathered_lses)
            group = GatherOnlyGroup(world, rank, gathers)
            self.assertEqual(_prepare(module, group, is_mla=mla), 1)
            partial = cp_lse_ag_out(
                partials[rank], lses[rank], group, is_lse_base_on_e=True
            )
            self.assertEqual(partial.shape, partials[rank].shape)
            self.assertTrue(partial.is_contiguous())
            if mla:
                partial = module.project_mla_values(
                    partial, zero_allocator=None, dcp_group_heads=True
                )
            else:
                partial = partial.flatten(1)
            projected.append(module.o_proj(partial)[0].float())
            # The only runtime collective is a [B,H] FP32 LSE gather.
            self.assertEqual(group.calls[-1], (lses[rank].shape, 0))
            self.assertEqual(len(group.calls), 3 if mla else 2)
            # Prefill still uses the original head shard and its weight.
            local_input = torch.randn(batch, 2 * out_dim, device="cuda", dtype=dtype)
            torch.testing.assert_close(
                module.o_proj(local_input)[0],
                torch.nn.functional.linear(local_input, module.o_proj.weight),
            )
        torch.testing.assert_close(
            torch.stack(projected).sum(0),
            expected,
            atol=3e-2 if dtype == torch.bfloat16 else 2e-5,
            rtol=3e-2 if dtype == torch.bfloat16 else 2e-5,
        )

    def test_gqa_against_full_context_attention(self):
        for world in (2, 3, 4, 8):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(world=world, dtype=dtype):
                    self._dense_reference(world, mla=False, dtype=dtype)

    def test_mla_against_full_context_attention(self):
        for world in (2, 3, 4, 8):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(world=world, dtype=dtype):
                    self._dense_reference(world, mla=True, dtype=dtype)

    def test_empty_shards_padding_and_lse_bases(self):
        for base_e in (False, True):
            # Includes an entirely empty/padded row and a NaN local partial.
            outputs = torch.randn(3, 2, 6, 80, device="cuda")
            outputs[0, 0] = torch.nan
            outputs[:, 1] = torch.nan
            lses = torch.randn(3, 2, 6, device="cuda")
            lses[0, 0] = -torch.inf
            lses[:, 1] = -torch.inf
            scaled = []
            for rank in range(3):
                group = GatherOnlyGroup(3, rank, [lses.flatten(0, 1)])
                scaled.append(cp_lse_ag_out(outputs[rank], lses[rank], group, base_e))
            combined = torch.stack(scaled).sum(0)
            log_lses = (
                lses if base_e else lses * torch.log(torch.tensor(2.0, device="cuda"))
            )
            weights = (log_lses - log_lses.logsumexp(0)).exp().nan_to_num()
            expected = (outputs.nan_to_num() * weights[..., None]).sum(0)
            torch.testing.assert_close(combined, expected, atol=2e-6, rtol=2e-6)
            self.assertTrue((combined[1] == 0).all())

    def test_fp8_gather_preserves_weight_bytes_and_scales(self):
        for dim in (0, 1):
            for packed in (False, True):
                for width in (128, 256, 384, 1024):
                    self._check_fp8_gather(dim, packed, width)
            self._check_fp8_gather(dim, False, 1024, format_ue8m0=True)

    def _check_fp8_gather(self, dim, packed, width, format_ue8m0=False):
        from sglang.srt.layers.quantization.fp8_utils import (
            inverse_transform_scale_ue8m0,
            transform_scale_ue8m0,
        )

        with self.subTest(dim=dim, packed=packed, width=width):
            weight = torch.randn(256, width, device="cuda").to(torch.float8_e4m3fn)
            scale = torch.full((2, width // 128), 0.125, device="cuda")
            # Different peer scales catch an accidental local-scale reuse.
            full_weight = torch.cat([weight, (-weight.float()).to(weight.dtype)], dim)
            full_scale = torch.cat([scale, scale * 2], dim)
            layer = torch.nn.Module()
            layer.weight = torch.nn.Parameter(weight, requires_grad=False)
            layer.weight_scale_inv = torch.nn.Parameter(
                transform_scale_ue8m0(scale, mn=weight.shape[0]) if packed else scale,
                requires_grad=False,
            )
            layer.weight_scale_inv.format_ue8m0 = packed or format_ue8m0
            processed = []
            layer.quant_method = SimpleNamespace(
                process_weights_after_loading=processed.append
            )
            group = GatherOnlyGroup(2, 0, [full_weight.view(torch.uint8), full_scale])
            replica = gather_block_fp8_linear(layer, group, dim)
            torch.testing.assert_close(
                replica.weight.view(torch.uint8), full_weight.view(torch.uint8)
            )
            replica_scale = replica.weight_scale_inv
            if packed:
                replica_scale = inverse_transform_scale_ue8m0(
                    replica_scale, mn=full_weight.shape[0]
                )
                replica_scale = replica_scale[:, : full_scale.shape[1]]
            torch.testing.assert_close(replica_scale, full_scale)
            self.assertEqual(
                replica.weight_scale_inv.format_ue8m0, packed or format_ue8m0
            )
            self.assertEqual(processed, [replica])
            self.assertEqual(group.calls[0], (weight.shape, dim))
            self.assertEqual(group.calls[1][1], dim)
            self.assertEqual(group.calls[1][0][0], scale.shape[0])

    def test_model_specific_mla_gate_is_rejected(self):
        module = AttentionProjection(2, 80, 96)
        module.prepare_attention_output_gate = lambda hidden: hidden
        group = GatherOnlyGroup(2, 0, [])
        with self.assertRaisesRegex(ValueError, "model-specific MLA output gates"):
            _prepare(module, group)
        self.assertEqual(group.calls, [])

    def test_gqa_kv_cache_layout_mismatch_is_rejected(self):
        module = AttentionProjection(2, 80, 96, kv_heads=2)
        group = GatherOnlyGroup(2, 0, [])
        with self.assertRaisesRegex(ValueError, "layer has 2 heads.*expects 4"):
            _prepare(module, group, is_mla=False, num_kv_heads=4)
        self.assertEqual(group.calls, [])

    def test_block_fp8_replica_runs_full_width_projection(self):
        from sglang.srt.layers.quantization.fp8 import Fp8Config
        from sglang.srt.layers.quantization.fp8_utils import Fp8GemmRunnerBackend

        # Test the actual FP8 post-load/apply methods, not just byte copying.
        with patch(
            "sglang.srt.layers.quantization.fp8_utils.FP8_GEMM_RUNNER_BACKEND",
            Fp8GemmRunnerBackend("triton"),
        ):
            module = torch.nn.Module()
            module.attn = RadixAttention(
                num_heads=2, head_dim=128, scaling=1.0, num_kv_heads=1, layer_id=0
            )
            module.o_proj = RowParallelLinear(
                256,
                256,
                bias=False,
                params_dtype=torch.bfloat16,
                parallel_group="replicated",
                quant_config=Fp8Config(
                    is_checkpoint_fp8_serialized=True,
                    weight_block_size=[128, 128],
                ),
            ).cuda()
            weight = torch.randn(256, 256, device="cuda").to(torch.float8_e4m3fn)
            scales = torch.tensor([[0.0625, 0.125], [0.25, 0.5]], device="cuda")
            module.o_proj.weight.data = weight
            module.o_proj.weight_scale_inv.data = scales
            module.o_proj.quant_method.process_weights_after_loading(module.o_proj)
            full_weight = torch.cat([weight, weight], dim=1)
            full_scales = torch.cat([scales, scales * 2], dim=1)
            group = GatherOnlyGroup(2, 0, [full_weight.view(torch.uint8), full_scales])
            _prepare(module, group, is_mla=False)
            inputs = torch.randn(3, 512, device="cuda", dtype=torch.bfloat16)
            actual = module.o_proj(inputs)[0]
            dequant = full_weight.float() * full_scales.repeat_interleave(
                128, 0
            ).repeat_interleave(128, 1)
            expected = inputs.float() @ dequant.T
            # Dynamic activation quantization contributes FP8 rounding error.
            relative_error = (actual.float() - expected).norm() / expected.norm()
            self.assertLess(relative_error.item(), 0.05)
            torch.testing.assert_close(
                module.o_proj.dcp_replica.weight.view(torch.uint8),
                full_weight.view(torch.uint8),
            )

    def test_incorrect_projection_width_is_rejected(self):
        module = AttentionProjection(2, 80, 96)
        weights = torch.randn(96, 320, device="cuda")
        group = GatherOnlyGroup(2, 0, [weights])
        _prepare(module, group, is_mla=False)
        with self.assertRaisesRegex(ValueError, r"width 240 matches neither"):
            module.o_proj(torch.randn(3, 240, device="cuda"))


if __name__ == "__main__":
    unittest.main()
