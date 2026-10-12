"""GLM colocated CP/TP and DFlash placement contracts (no checkpoint)."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.arg_groups.cp_tp_group_sharing_hook import resolve_cp_tp_group_sharing
from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.layers.cp.bcg import execute_prefill_cp_bcg
from sglang.srt.layers.cp.cp_decode_attn_tp import CpDecodeAttnTpContext
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.glm5_next import Glm5NextDecoderLayer
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Boundary:
    input_on_attn_tp_slices = False

    def prepare(self, hidden, batch, **kwargs):
        return hidden

    def finish(self, hidden, batch):
        return hidden


class _Attention(nn.Module):
    """Tiny real GEMMs, with the production cached-slice context."""

    def __init__(self, context):
        super().__init__()
        self.context = context
        self.q = ColumnParallelLinear.__new__(ColumnParallelLinear)
        nn.Module.__init__(self.q)
        self.q.weight = nn.Parameter(torch.eye(4), requires_grad=False)
        self.q.output_size_per_partition = 4
        self.o = RowParallelLinear.__new__(RowParallelLinear)
        nn.Module.__init__(self.o)
        self.o.weight = nn.Parameter(torch.eye(4), requires_grad=False)
        self.o.input_size_per_partition = 4
        self.o.use_decode_attn_tp = False

    def maybe_use_decode_attn_tp(self, batch):
        return self.context.maybe_use_decode_attn_tp(batch, [self.q, self.o])

    def forward(self, *, hidden_states, **kwargs):
        return hidden_states @ self.q.weight.T @ self.o.weight.T


class _MLP(nn.Module):
    def forward(self, hidden, *args):
        return hidden


class TestGlm53DecodeAttnTP(CustomTestCase):
    def test_packed_fp8_scale_shards_keep_mn_major_layout(self):
        with published_topology(
            tp_size=2, attn_cp_size=2, enable_cp_decode_attn_tp=True
        ):
            for dim, shape in ((0, (12, 4)), (1, (12, 2)), (1, (12, 4))):
                with self.subTest(dim=dim, shape=shape):
                    context = CpDecodeAttnTpContext()
                    scale = torch.empty_strided(shape, (1, 12), dtype=torch.int32)
                    scale.copy_(torch.arange(scale.numel()).reshape(shape))
                    module = nn.Module()
                    module.weight_scale_inv = nn.Parameter(scale, requires_grad=False)
                    context._activate(module, "weight_scale_inv", dim)
                    shard = module.weight_scale_inv
                    expected = scale.narrow(dim, 0, shape[dim] // 2)
                    torch.testing.assert_close(shard, expected)
                    aligned_mn = (shard.shape[0] + 3) // 4 * 4
                    self.assertEqual(shard.stride(), (1, aligned_mn))
                    pointer = shard.data_ptr()
                    context._restore(module, "weight_scale_inv")
                    torch.testing.assert_close(module.weight_scale_inv, scale)
                    context._activate(module, "weight_scale_inv", dim)
                    self.assertEqual(module.weight_scale_inv.data_ptr(), pointer)
                    context._restore(module, "weight_scale_inv")

    def test_bcg_gathers_draft_features_after_removing_capture_padding(self):
        hidden = torch.tensor([[1.0], [3.0], [-99.0]])
        aux = hidden + 10
        batch = SimpleNamespace(input_ids=torch.arange(4))

        # Simulate the other CP rank's interleaved rows at the collective edge.
        def gather(local, batch, stream):
            return torch.stack((local, local + 1), dim=1).flatten(0, 1)

        for features in (aux, [aux, aux + 10]):
            with self.subTest(as_list=isinstance(features, list)):
                model = SimpleNamespace(
                    capture_aux_hidden_states=True,
                    pp_group=SimpleNamespace(is_last_rank=True),
                    lm_head=None,
                    logits_processor=lambda ids, hidden, head, batch, aux: (
                        hidden,
                        aux,
                    ),
                )
                runner = SimpleNamespace(
                    prefill_cp_bcg_input=SimpleNamespace(live_local_tokens=2),
                    model_runner=SimpleNamespace(model=model),
                    _prefill_forward_context=lambda *args, **kwargs: nullcontext(),
                    backend=SimpleNamespace(
                        replay=lambda *args, **kwargs: (hidden, features)
                    ),
                )
                with (
                    patch(
                        "sglang.srt.layers.cp.bcg.cp_gather_after_forward",
                        side_effect=gather,
                    ),
                    patch("torch.cuda.current_stream", return_value=None),
                ):
                    result, result_aux = execute_prefill_cp_bcg(
                        runner, batch, batch, 6, 4
                    )
                expected = torch.arange(1.0, 5.0).unsqueeze(1)
                torch.testing.assert_close(result, expected)
                if isinstance(features, list):
                    for index, feature in enumerate(result_aux):
                        torch.testing.assert_close(feature, expected + 10 * (index + 1))
                else:
                    torch.testing.assert_close(result_aux, expected + 10)

    def test_dflash_external_draft_is_allowed_with_shared_cp_tp(self):
        args = ServerArgs(
            model_path="target",
            tp_size=8,
            attn_cp_size=8,
            enable_prefill_cp=True,
            cp_strategy="interleave",
            enable_cp_decode_attn_tp=True,
            speculative_algorithm="DFLASH",
            speculative_draft_model_path="incoai/GLM-5.3-Flash-DFlash2",
            speculative_draft_attention_backend="fa4",
            cuda_graph_config=CudaGraphConfig(prefill=PhaseConfig()),
        )
        model = SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["Glm5NextForConditionalGeneration"]
            )
        )
        with patch(
            "sglang.srt.arg_groups.cp_tp_group_sharing_hook.mambaish_config",
            return_value=SimpleNamespace(linear_attn_config={"num_heads": 64}),
        ):
            resolve_cp_tp_group_sharing(args, model)
        self.assertTrue(resolving_view(args).enable_cp_tp_group_sharing)
        self.assertEqual(args.speculative_draft_attention_backend, "fa4")

    def test_glm_dsa_slices_verify_but_never_kda_or_active_prefill(self):
        with published_topology(
            tp_size=2, attn_cp_size=2, enable_cp_decode_attn_tp=True
        ):
            with get_parallel().override(attn_cp_rank=1, tp_rank=1, moe_tp_rank=1):
                context = CpDecodeAttnTpContext()
            layer = Glm5NextDecoderLayer.__new__(Glm5NextDecoderLayer)
            nn.Module.__init__(layer)
            layer.self_attn = _Attention(context)
            layer.mlp = _MLP()
            layer.attn_boundary = layer.ffn_boundary = _Boundary()
            values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
            for mode, cp_active, is_kda, expected in (
                (ForwardMode.TARGET_VERIFY, False, False, [[0.0, 0.0, 3.0, 4.0]]),
                (ForwardMode.DECODE, False, False, [[0.0, 0.0, 3.0, 4.0]]),
                (ForwardMode.EXTEND, True, False, [[1.0, 2.0, 3.0, 4.0]]),
                (ForwardMode.EXTEND, False, False, [[0.0, 0.0, 3.0, 4.0]]),
                (ForwardMode.DECODE, False, True, [[1.0, 2.0, 3.0, 4.0]]),
            ):
                with self.subTest(mode=mode, cp_active=cp_active, is_kda=is_kda):
                    layer.is_linear_attn = is_kda
                    batch = SimpleNamespace(forward_mode=mode)
                    with (
                        patch(
                            "sglang.srt.layers.cp.cp_decode_attn_tp.is_cp_active",
                            return_value=cp_active,
                        ),
                        patch(
                            "sglang.srt.layers.cp.cp_decode_attn_tp.dsa_use_prefill_cp",
                            return_value=cp_active,
                        ),
                    ):
                        output, _ = layer(torch.zeros(1), values, batch)
                    torch.testing.assert_close(output, torch.tensor(expected))
                    torch.testing.assert_close(layer.self_attn.q.weight, torch.eye(4))
                    torch.testing.assert_close(layer.self_attn.o.weight, torch.eye(4))
                    self.assertFalse(layer.self_attn.o.use_decode_attn_tp)

    def test_decode_shards_restore_after_exception(self):
        with published_topology(
            tp_size=2, attn_cp_size=2, enable_cp_decode_attn_tp=True
        ):
            attention = _Attention(CpDecodeAttnTpContext())
            batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
            with (
                patch(
                    "sglang.srt.layers.cp.cp_decode_attn_tp.is_cp_active",
                    return_value=False,
                ),
                patch(
                    "sglang.srt.layers.cp.cp_decode_attn_tp.dsa_use_prefill_cp",
                    return_value=False,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "attention failed"):
                    with attention.maybe_use_decode_attn_tp(batch):
                        self.assertEqual(attention.q.weight.shape, (2, 4))
                        self.assertEqual(attention.o.weight.shape, (4, 2))
                        self.assertTrue(attention.o.use_decode_attn_tp)
                        raise RuntimeError("attention failed")
            torch.testing.assert_close(attention.q.weight, torch.eye(4))
            torch.testing.assert_close(attention.o.weight, torch.eye(4))
            self.assertEqual(attention.q.output_size_per_partition, 4)
            self.assertEqual(attention.o.input_size_per_partition, 4)
            self.assertFalse(attention.o.use_decode_attn_tp)


if __name__ == "__main__":
    unittest.main()
