"""CPU regression coverage for padded linear-attention inputs and outputs."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.layers.radix_linear_attention as radix_linear_attention
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeAttentionBackend:
    def forward(
        self,
        *,
        layer,
        forward_batch,
        mixed_qkv,
        a,
        b,
        linear_attn_output=None,
    ):
        del layer
        torch.testing.assert_close(forward_batch.out_cache_loc, torch.arange(3))
        assert mixed_qkv.shape[0] == 3
        assert a.shape[0] == 3
        assert b.shape[0] == 3
        if linear_attn_output is None:
            return torch.full((1, 3, 2, 4), 5.0)
        linear_attn_output.fill_(5.0)
        return linear_attn_output


class _FailingAttentionBackend:
    def forward(self, **kwargs):
        del kwargs
        raise RuntimeError("backend failure")


class _ExtendMode:
    def is_extend(self):
        return True

    def is_target_verify(self):
        return False


class _TargetVerifyMode:
    def is_extend(self):
        return True

    def is_target_verify(self):
        return True


class _PhysicalAttentionBackend:
    def forward(self, *, layer, forward_batch, mixed_qkv, a, b):
        del layer, forward_batch
        assert mixed_qkv.shape[0] == 5
        assert a.shape[0] == 5
        assert b.shape[0] == 5
        return torch.full((1, 5, 2, 4), 9.0)


class TestRadixLinearAttentionPadding(CustomTestCase):
    def test_eager_padded_input_is_sliced_and_output_shape_is_restored(self):
        layer = radix_linear_attention.RadixLinearAttention(
            layer_id=0,
            num_q_heads=1,
            num_k_heads=1,
            num_v_heads=2,
            head_q_dim=4,
            head_k_dim=4,
            head_v_dim=4,
        )
        original_out_cache_loc = torch.arange(5)
        forward_batch = SimpleNamespace(
            forward_mode=_ExtendMode(),
            global_num_token_non_padded_cpu=3,
            out_cache_loc=original_out_cache_loc,
        )

        with (
            patch.object(
                radix_linear_attention,
                "get_tc_piecewise_forward_context",
                return_value=None,
            ),
            patch.object(
                radix_linear_attention,
                "get_attn_backend",
                return_value=_FakeAttentionBackend(),
            ),
        ):
            output = layer.forward(
                forward_batch=forward_batch,
                mixed_qkv=torch.zeros((5, 8)),
                a=torch.zeros((5, 2)),
                b=torch.zeros((5, 2)),
            )

        torch.testing.assert_close(output[:, :3], torch.full((1, 3, 2, 4), 5.0))
        torch.testing.assert_close(output[:, 3:], torch.zeros((1, 2, 2, 4)))
        self.assertIs(forward_batch.out_cache_loc, original_out_cache_loc)

    def test_target_verify_keeps_physical_rows_matching_its_metadata(self):
        layer = radix_linear_attention.RadixLinearAttention(
            layer_id=0,
            num_q_heads=1,
            num_k_heads=1,
            num_v_heads=2,
            head_q_dim=4,
            head_k_dim=4,
            head_v_dim=4,
        )
        original_out_cache_loc = torch.arange(5)
        forward_batch = SimpleNamespace(
            forward_mode=_TargetVerifyMode(),
            global_num_token_non_padded_cpu=3,
            out_cache_loc=original_out_cache_loc,
        )

        with (
            patch.object(
                radix_linear_attention,
                "get_tc_piecewise_forward_context",
                return_value=None,
            ),
            patch.object(
                radix_linear_attention,
                "get_attn_backend",
                return_value=_PhysicalAttentionBackend(),
            ),
        ):
            output = layer.forward(
                forward_batch=forward_batch,
                mixed_qkv=torch.zeros((5, 8)),
                a=torch.zeros((5, 2)),
                b=torch.zeros((5, 2)),
            )

        torch.testing.assert_close(output, torch.full((1, 5, 2, 4), 9.0))
        self.assertIs(forward_batch.out_cache_loc, original_out_cache_loc)

    def test_eager_backend_failure_restores_out_cache_loc(self):
        layer = radix_linear_attention.RadixLinearAttention(
            layer_id=0,
            num_q_heads=1,
            num_k_heads=1,
            num_v_heads=2,
            head_q_dim=4,
            head_k_dim=4,
            head_v_dim=4,
        )
        original_out_cache_loc = torch.arange(5)
        forward_batch = SimpleNamespace(
            forward_mode=_ExtendMode(),
            global_num_token_non_padded_cpu=3,
            out_cache_loc=original_out_cache_loc,
        )

        with (
            patch.object(
                radix_linear_attention,
                "get_tc_piecewise_forward_context",
                return_value=None,
            ),
            patch.object(
                radix_linear_attention,
                "get_attn_backend",
                return_value=_FailingAttentionBackend(),
            ),
            self.assertRaisesRegex(RuntimeError, "backend failure"),
        ):
            layer.forward(
                forward_batch=forward_batch,
                mixed_qkv=torch.zeros((5, 8)),
                a=torch.zeros((5, 2)),
                b=torch.zeros((5, 2)),
            )

        self.assertIs(forward_batch.out_cache_loc, original_out_cache_loc)

    def test_padded_output_tail_is_initialized(self):
        for padded_num_tokens in (3, 5):
            with self.subTest(padded_num_tokens=padded_num_tokens):
                original_out_cache_loc = torch.arange(padded_num_tokens)
                forward_batch = SimpleNamespace(
                    global_num_token_non_padded_cpu=3,
                    out_cache_loc=original_out_cache_loc,
                )
                context = SimpleNamespace(
                    forward_batch=forward_batch,
                    attention_layers=[object()],
                )
                output = torch.full((1, padded_num_tokens, 2, 4), float("nan"))

                with (
                    patch.object(
                        radix_linear_attention,
                        "get_tc_piecewise_forward_context",
                        return_value=context,
                    ),
                    patch.object(
                        radix_linear_attention,
                        "get_attn_backend",
                        return_value=_FakeAttentionBackend(),
                    ),
                ):
                    radix_linear_attention._unified_linear_attention_with_output_impl(
                        mixed_qkv=torch.zeros((padded_num_tokens, 8)),
                        a=torch.zeros((padded_num_tokens, 2)),
                        b=torch.zeros((padded_num_tokens, 2)),
                        output=output,
                        layer_id=0,
                    )

                torch.testing.assert_close(output[:, :3], torch.full((1, 3, 2, 4), 5.0))
                torch.testing.assert_close(
                    output[:, 3:],
                    torch.zeros((1, padded_num_tokens - 3, 2, 4)),
                )
                self.assertIs(forward_batch.out_cache_loc, original_out_cache_loc)


if __name__ == "__main__":
    import unittest

    unittest.main()
