import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.models.deepseek_common.attention_forward_methods import (
    forward_mha,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDeepseekMHABreakablePrefix(CustomTestCase):
    def test_eager_region_uses_live_batch_and_fixed_shape_bridge(self):
        capture_batch = SimpleNamespace(global_num_token_non_padded_cpu=8)
        live_batch = SimpleNamespace(global_num_token_non_padded_cpu=5)
        force_depth = 0
        calls = []

        @contextmanager
        def fake_force_eager_attention():
            nonlocal force_depth
            force_depth += 1
            try:
                yield
            finally:
                force_depth -= 1

        class FakeAttention:
            def _forward_normal_chunked_kv_attention(self, q, k, v, forward_batch):
                calls.append(
                    (
                        q.shape[0],
                        k.shape[0],
                        v.shape[0],
                        forward_batch,
                        force_depth,
                    )
                )
                return q + 10

        q = torch.arange(8 * 2 * 3, dtype=torch.float32).reshape(8, 2, 3)
        k = q + 100
        v = q + 200
        with (
            mock.patch.object(
                forward_mha,
                "get_tc_piecewise_forward_context",
                return_value=SimpleNamespace(forward_batch=live_batch),
            ),
            mock.patch.object(
                forward_mha,
                "force_eager_attention",
                fake_force_eager_attention,
            ),
        ):
            output = forward_mha._bcg_mha_chunked_kv_attention(
                FakeAttention(), q, k, v, capture_batch
            )

        self.assertEqual(calls, [(5, 5, 5, live_batch, 1)])
        self.assertEqual(output.shape, q.shape)
        torch.testing.assert_close(output[:5], q[:5] + 10)
        torch.testing.assert_close(output[5:], torch.zeros_like(output[5:]))
        self.assertEqual(force_depth, 0)

    def test_eager_region_skips_attention_for_idle_replay(self):
        class FakeAttention:
            num_local_heads = 2
            v_head_dim = 5

            def _forward_normal_chunked_kv_attention(self, *args):
                raise AssertionError("an idle replay must not launch attention")

        q = torch.ones((4, 2, 3))
        live_batch = SimpleNamespace(global_num_token_non_padded_cpu=0)
        with mock.patch.object(
            forward_mha,
            "get_tc_piecewise_forward_context",
            return_value=SimpleNamespace(forward_batch=live_batch),
        ):
            output = forward_mha._bcg_mha_chunked_kv_attention(
                FakeAttention(), q, q, q, live_batch
            )

        torch.testing.assert_close(output, torch.zeros((4, 2, 5)))

    def test_core_keeps_output_projection_after_single_eager_region(self):
        events = []
        eager_output = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
        projected = torch.full((4, 7), 9.0)

        class FakeAttention(forward_mha.DeepseekMHAForwardMixin):
            num_local_heads = 2
            v_head_dim = 3

            def _forward_normal_chunked_kv_attention(self, *args):
                raise AssertionError("BCG must use the eager attention region")

            def o_proj(self, value):
                events.append(("o_proj", value.shape))
                return projected, None

        def fake_eager_region(attn, q, k, v, forward_batch):
            events.append(("attention", forward_batch))
            return eager_output

        q = k = v = torch.empty((4, 2, 3))
        forward_batch = object()
        with (
            mock.patch.object(
                forward_mha,
                "is_in_breakable_cuda_graph",
                return_value=True,
            ),
            mock.patch.object(
                forward_mha,
                "bcg_mha_chunked_kv_attention",
                side_effect=fake_eager_region,
            ) as eager_region,
        ):
            output = FakeAttention().forward_normal_chunked_kv_core(
                q, k, v, forward_batch
            )

        self.assertIs(output, projected)
        self.assertEqual(
            events,
            [
                ("attention", forward_batch),
                ("o_proj", torch.Size([4, 6])),
            ],
        )
        eager_region.assert_called_once_with(mock.ANY, q, k, v, forward_batch)

    def test_fused_prefix_projects_output_exactly_once(self):
        calls = []

        class FakeAttention(forward_mha.DeepseekMHAForwardMixin):
            num_local_heads = 2
            v_head_dim = 3

            def _fused_prefix_extend_attn_mha(self, q, k, v, batch):
                return q

            def o_proj(self, value):
                calls.append(value.shape)
                return value + 10, None

        q = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
        batch = SimpleNamespace(extend_prefix_lens_cpu=[8], num_prefix_chunks=1)
        with (
            mock.patch.object(
                forward_mha, "is_in_breakable_cuda_graph", return_value=False
            ),
            mock.patch.object(
                forward_mha, "resolve_attn_backend", return_value=object()
            ),
            mock.patch.object(
                forward_mha, "use_fused_prefix_extend", return_value=True
            ),
        ):
            output = FakeAttention().forward_normal_chunked_kv_core(q, q, q, batch)
        self.assertEqual(calls, [torch.Size([4, 6])])
        torch.testing.assert_close(output, q.reshape(4, 6) + 10)


if __name__ == "__main__":
    unittest.main()
