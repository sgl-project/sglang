import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers import communicator as comm
from sglang.srt.layers.communicator import LayerCommunicator, ScatterMode
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _fake_communicator():
    return types.SimpleNamespace(
        _speculative_algo=None,
        layer_scatter_modes=types.SimpleNamespace(mlp_mode=ScatterMode.TP_ATTN_FULL),
        is_last_layer=False,
        _context=types.SimpleNamespace(tp_size=4),
    )


class TestFuseMlpAllReduceGate(CustomTestCase):
    """Hybrid EP+TP must not fuse the post-experts all-reduce away.

    The fused residual+LN reduces over a single group, but with moe_ep_size > 1
    and moe_tp_size > 1 the post-experts reduction spans two disjoint groups
    (_MOE_EP then _MOE_TP) and should_skip_post_experts_all_reduce() drops both
    once fusion is published. The result is activations reduced over only half
    the peers -- wrong output, no crash. Observed as garbage completions on
    Qwen3-30B-A3B with --tp-size 4 --ep-size 2.
    """

    def _should_fuse(self, *, moe_ep_size, moe_tp_size):
        forward_batch = types.SimpleNamespace(
            input_ids=types.SimpleNamespace(shape=(8,))
        )
        with (
            patch.object(comm, "is_enable_moe_cp_allgather", return_value=False),
            patch.object(comm, "apply_flashinfer_allreduce_fusion", return_value=True),
            patch.object(
                comm,
                "get_attn_tp_context",
                return_value=types.SimpleNamespace(input_scattered=False),
            ),
            get_parallel().override(
                moe_ep_size=moe_ep_size, moe_tp_size=moe_tp_size, tp_size=4
            ),
        ):
            return LayerCommunicator.should_fuse_mlp_allreduce_with_next_layer(
                _fake_communicator(), forward_batch
            )

    def test_hybrid_ep_tp_does_not_fuse(self):
        self.assertFalse(self._should_fuse(moe_ep_size=2, moe_tp_size=2))

    def test_pure_tp_still_fuses(self):
        self.assertTrue(self._should_fuse(moe_ep_size=1, moe_tp_size=4))

    def test_pure_ep_still_fuses(self):
        self.assertTrue(self._should_fuse(moe_ep_size=4, moe_tp_size=1))


class TestStaticFp8ScaleHandoff(CustomTestCase):
    def test_scattered_gather_keeps_static_scale_scalar(self):
        scale = torch.ones(1, dtype=torch.float32)
        quant = torch.zeros(2, 8, dtype=torch.float8_e4m3fn)
        bf16 = torch.zeros(2, 8, dtype=torch.bfloat16)
        residual_out = torch.zeros(2, 8, dtype=torch.bfloat16)

        class _Norm:
            def forward_with_allreduce_fusion(self, *args, **kwargs):
                raise AssertionError("static-FP8 path should be selected")

            def forward_with_allreduce_fusion_static_fp8_quant(
                self, *args, keep_bf16, **kwargs
            ):
                hidden_states = (bf16, quant, scale) if keep_bf16 else (quant, scale)
                return hidden_states, residual_out

        def _gather_every_tuple_element(hidden_states, **kwargs):
            if isinstance(hidden_states, tuple):
                return tuple(torch.cat((item, item), dim=0) for item in hidden_states)
            return torch.cat((hidden_states, hidden_states), dim=0)

        for keep_bf16 in (False, True):
            with self.subTest(keep_bf16=keep_bf16):
                layer = types.SimpleNamespace(
                    enable_fused_ar_quant=True,
                    fused_ar_quant_linear=object(),
                    fused_ar_quant_keep_bf16=keep_bf16,
                    input_layernorm=_Norm(),
                    _communicate_simple_fn=_gather_every_tuple_element,
                    _context=object(),
                    qkv_latent_func=None,
                )
                hidden_states = torch.zeros(2, 8, dtype=torch.bfloat16)
                hidden_states._sglang_needs_allreduce_fusion = True
                with (
                    patch.object(comm, "_use_aiter", False),
                    patch.object(
                        comm, "apply_flashinfer_allreduce_fusion", return_value=True
                    ),
                    patch.object(
                        comm,
                        "get_attn_tp_context",
                        return_value=types.SimpleNamespace(input_scattered=False),
                    ),
                ):
                    result, _ = LayerCommunicator.prepare_attn(
                        layer,
                        hidden_states,
                        torch.zeros_like(hidden_states),
                        types.SimpleNamespace(),
                    )

                self.assertIs(result[-1], scale)
                self.assertEqual(result[-1].numel(), 1)
                for token_tensor in result[:-1]:
                    self.assertEqual(token_tensor.shape[0], 4)


if __name__ == "__main__":
    unittest.main()
