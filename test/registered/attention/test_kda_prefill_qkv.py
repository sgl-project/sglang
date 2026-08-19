import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_qkv_split_gdn_prefill,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn
from sglang.srt.layers.attention.linear import kda_backend
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=12, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=12, stage="stage-b", runner_config="1-gpu-large-amd")


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA or ROCm")
class TestKDAPrefillQKV(unittest.TestCase):
    def setUp(self):
        self._configure_case(
            dtype=torch.bfloat16,
            conv_width=4,
            head_counts=(4, 4, 4),
            head_dims=(128, 128, 128),
            initial_state_mask=(True, False, True),
            use_bias=True,
        )

    def _configure_case(
        self,
        *,
        dtype,
        conv_width,
        head_counts,
        head_dims,
        initial_state_mask,
        use_bias,
    ):
        torch.manual_seed(42)
        self.batch_size = 3
        self.seq_lens_cpu = [67, 128, 35]
        self.total_tokens = sum(self.seq_lens_cpu)
        self.head_counts = head_counts
        self.head_dims = head_dims
        self.channel_dims = tuple(
            head_count * head_dim
            for head_count, head_dim in zip(head_counts, head_dims)
        )
        self.qkv_dim = sum(self.channel_dims)
        self.conv_width = conv_width
        self.pool_size = 8
        self.device = "cuda"
        self.dtype = dtype

        self.mixed_qkv = torch.randn(
            self.total_tokens,
            self.qkv_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self.conv_weights = torch.randn(
            self.qkv_dim,
            self.conv_width,
            device=self.device,
            dtype=self.dtype,
        )
        self.conv_bias = (
            torch.randn(self.qkv_dim, device=self.device, dtype=self.dtype)
            if use_bias
            else None
        )
        self.initial_states = torch.randn(
            self.pool_size,
            self.qkv_dim,
            self.conv_width - 1,
            device=self.device,
            dtype=self.dtype,
        )
        self.cache_indices = torch.tensor(
            [1, 6, 3], device=self.device, dtype=torch.int32
        )
        self.has_initial_state = torch.tensor(
            initial_state_mask, device=self.device, dtype=torch.bool
        )
        self.query_start_loc = torch.tensor(
            [0, 67, 195, 230], device=self.device, dtype=torch.int32
        )

    def _run_baseline(self):
        conv_states = self.initial_states.clone()
        outputs = []
        biases = (
            self.conv_bias.split(self.channel_dims, dim=0)
            if self.conv_bias is not None
            else (None, None, None)
        )
        for qkv, weight, bias, state, head_count, head_dim in zip(
            self.mixed_qkv.transpose(0, 1).split(self.channel_dims, dim=0),
            self.conv_weights.split(self.channel_dims, dim=0),
            biases,
            conv_states.split(self.channel_dims, dim=1),
            self.head_counts,
            self.head_dims,
        ):
            output = causal_conv1d_fn(
                qkv,
                weight,
                bias,
                activation="silu",
                conv_states=state,
                has_initial_state=self.has_initial_state,
                cache_indices=self.cache_indices,
                query_start_loc=self.query_start_loc,
                seq_lens_cpu=self.seq_lens_cpu,
            ).transpose(0, 1)
            outputs.append(
                output.view(1, self.total_tokens, head_count, head_dim).contiguous()
            )
        return outputs, conv_states

    def _run_fused(self):
        conv_states = self.initial_states.clone()
        mixed_qkv = causal_conv1d_fn(
            self.mixed_qkv.transpose(0, 1),
            self.conv_weights,
            self.conv_bias,
            activation="silu",
            conv_states=conv_states,
            has_initial_state=self.has_initial_state,
            cache_indices=self.cache_indices,
            query_start_loc=self.query_start_loc,
            seq_lens_cpu=self.seq_lens_cpu,
        ).transpose(0, 1)
        outputs = fused_qkv_split_gdn_prefill(
            mixed_qkv,
            *self.head_counts,
            *self.head_dims,
        )
        return outputs, conv_states

    def test_fused_matches_three_convolutions(self):
        expected_outputs, expected_states = self._run_baseline()
        actual_outputs, actual_states = self._run_fused()

        for actual, expected in zip(actual_outputs, expected_outputs):
            self.assertTrue(actual.is_contiguous())
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_states, expected_states, rtol=0, atol=0)

    def test_dtype_width_layout_and_state_variants(self):
        cases = (
            {
                "dtype": torch.float16,
                "conv_width": 3,
                "head_counts": (4, 2, 3),
                "head_dims": (64, 64, 32),
                "initial_state_mask": (False, False, False),
                "use_bias": False,
            },
            {
                "dtype": torch.bfloat16,
                "conv_width": 4,
                "head_counts": (2, 2, 4),
                "head_dims": (64, 64, 32),
                "initial_state_mask": (True, True, True),
                "use_bias": True,
            },
        )
        for case in cases:
            with self.subTest(case=case):
                self._configure_case(**case)
                expected_outputs, expected_states = self._run_baseline()
                actual_outputs, actual_states = self._run_fused()
                for actual, expected in zip(actual_outputs, expected_outputs):
                    self.assertTrue(actual.is_contiguous())
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    actual_states, expected_states, rtol=0, atol=0
                )

    def test_fused_dispatch_limits(self):
        element_size = torch.empty((), dtype=torch.bfloat16).element_size()
        self.assertTrue(
            kda_backend._can_fuse_kda_prefill_qkv(
                kda_backend.MAX_FUSED_QKV_SPLIT_DIM,
                kda_backend.MAX_FUSED_QKV_BYTES // element_size,
                element_size,
            )
        )
        self.assertFalse(
            kda_backend._can_fuse_kda_prefill_qkv(
                kda_backend.MAX_FUSED_QKV_SPLIT_DIM + 1,
                1,
                element_size,
            )
        )
        self.assertFalse(
            kda_backend._can_fuse_kda_prefill_qkv(
                kda_backend.MAX_FUSED_QKV_SPLIT_DIM,
                kda_backend.MAX_FUSED_QKV_BYTES // element_size + 1,
                element_size,
            )
        )

    def test_forward_extend_dispatches_one_or_three_convolutions(self):
        qkv_dim = 3 * self.channel_dims[0]
        total_tokens = 8
        mixed_qkv = torch.randn(
            total_tokens, qkv_dim, device=self.device, dtype=self.dtype
        )
        cache = SimpleNamespace(
            conv=[
                torch.zeros(
                    2,
                    self.conv_width - 1,
                    qkv_dim,
                    device=self.device,
                    dtype=self.dtype,
                )
            ],
            temporal=torch.empty(0, device=self.device),
        )
        dispatcher = SimpleNamespace(extend=Mock(return_value="core-attn-output"))
        backend = object.__new__(kda_backend.KDAAttnBackend)
        backend.req_to_token_pool = SimpleNamespace(
            mamba2_layer_cache=lambda layer_id: cache
        )
        backend.forward_metadata = SimpleNamespace(
            query_start_loc=torch.tensor(
                [0, total_tokens], device=self.device, dtype=torch.int32
            ),
            mamba_cache_indices=torch.tensor(
                [0], device=self.device, dtype=torch.int32
            ),
            has_mamba_track_mask=False,
        )
        backend.kernel_dispatcher = dispatcher
        layer = SimpleNamespace(
            layer_id=0,
            q_dim=self.channel_dims[0],
            k_dim=self.channel_dims[0],
            v_dim=self.channel_dims[0],
            num_q_heads=self.head_counts[0],
            num_k_heads=self.head_counts[0],
            num_v_heads=self.head_counts[0],
            head_q_dim=self.head_dims[0],
            head_k_dim=self.head_dims[0],
            head_v_dim=self.head_dims[0],
            conv_weights=torch.empty(
                qkv_dim,
                self.conv_width,
                device=self.device,
                dtype=self.dtype,
            ),
            bias=None,
            A_log=None,
            dt_bias=None,
        )
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_target_verify=lambda: False,
                is_draft_extend_v2=lambda: False,
            ),
            extend_prefix_lens=torch.tensor([0], device=self.device),
            extend_seq_lens_cpu=[total_tokens],
        )
        fused_outputs = tuple(
            torch.empty(
                1,
                total_tokens,
                self.head_counts[0],
                self.head_dims[0],
                device=self.device,
                dtype=self.dtype,
            )
            for _ in range(3)
        )

        for use_fused, expected_conv_calls in ((True, 1), (False, 3)):
            with self.subTest(use_fused=use_fused):
                dispatcher.extend.reset_mock()
                with (
                    patch.object(
                        kda_backend,
                        "_can_fuse_kda_prefill_qkv",
                        return_value=use_fused,
                    ),
                    patch.object(
                        kda_backend,
                        "causal_conv1d_fn",
                        side_effect=lambda tensor, *args, **kwargs: tensor,
                    ) as conv_mock,
                    patch.object(
                        kda_backend,
                        "fused_qkv_split_gdn_prefill",
                        return_value=fused_outputs,
                    ) as split_mock,
                ):
                    output = backend.forward_extend(
                        layer,
                        forward_batch,
                        mixed_qkv,
                        torch.empty(0, device=self.device),
                        torch.empty(0, device=self.device),
                    )

                self.assertEqual(output, "core-attn-output")
                self.assertEqual(conv_mock.call_count, expected_conv_calls)
                self.assertEqual(split_mock.call_count, int(use_fused))
                dispatcher.extend.assert_called_once()


if __name__ == "__main__":
    unittest.main()
