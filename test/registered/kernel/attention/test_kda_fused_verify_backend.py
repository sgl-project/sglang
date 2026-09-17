"""KDA backend dispatch and ReplaySSM verify -> commit -> verify parity."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.kernels.ops.attention.fla.fused_kda_conv_recurrent_verify import (
    fused_kda_conv_gating_verify,
)
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.linear.kda_backend import (
    KDAAttnBackend,
    KDAKernelDispatcher,
)
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")

# One bf16 ulp: the fused and unfused kernels reduce K in different orders.
_OUTPUT_TOL = dict(rtol=2**-7, atol=1e-7)
# Fold and snapshot recurrence differ by ~1 fp32 ulp; a wrong-step commit
# moves elements by >= 1e-3, so 1e-5 still catches it.
_SNAPSHOT_ORACLE_TOL = dict(rtol=0, atol=1e-5)


class TestKDAFusedVerifyBackend(CustomTestCase):
    def _make_case(self, batch_size=1, heads=2, v_heads=4, lower_bound=-5.0):
        torch.manual_seed(36821)
        steps, head_dim, num_layers = 4, 128, 2
        num_slots = batch_size + 3
        dim = (2 * heads + v_heads) * head_dim

        def randn(*shape, dtype=torch.bfloat16):
            return torch.randn(*shape, device="cuda", dtype=dtype) * 0.2

        layers = [
            SimpleNamespace(
                layer_id=i,
                num_q_heads=heads,
                num_k_heads=heads,
                num_v_heads=v_heads,
                head_q_dim=head_dim,
                head_k_dim=head_dim,
                head_v_dim=head_dim,
                q_dim=heads * head_dim,
                k_dim=heads * head_dim,
                v_dim=v_heads * head_dim,
                conv_weights=randn(dim, 4),
                bias=randn(dim),
                A_log=randn(v_heads, dtype=torch.float32),
                dt_bias=randn(v_heads * head_dim, dtype=torch.float32),
                lower_bound=lower_bound,
            )
            for i in range(num_layers)
        ]
        state = MambaPool.SpeculativeState(
            conv=[randn(num_layers, num_slots, 3, dim)],
            temporal=randn(
                num_layers,
                num_slots,
                v_heads,
                head_dim,
                head_dim,
                dtype=torch.float32,
            ),
            intermediate_ssm=None,
            intermediate_conv_window=[randn(num_layers, batch_size, steps, 3, dim)],
            replayssm_rawv=randn(num_layers, num_slots, v_heads, 16, head_dim),
            replayssm_rawk=randn(num_layers, num_slots, heads, 16, head_dim),
            replayssm_g=randn(
                num_layers, num_slots, v_heads, 16, head_dim, dtype=torch.float32
            ),
            replayssm_beta=randn(
                num_layers, num_slots, v_heads, 16, dtype=torch.float32
            ),
        )
        # Physical slots differ from scratch rows; reverse them to catch callers
        # accidentally committing by request index instead of mamba slot.
        slots = torch.arange(batch_size + 1, 1, -1, device="cuda", dtype=torch.int32)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=SimpleNamespace(draft_token_num=steps, ragged_verify_layout=None),
        )
        rounds = [
            [
                (
                    randn(batch_size * steps, dim),
                    randn(1, batch_size * steps, v_heads * head_dim),
                    randn(1, batch_size * steps, v_heads),
                )
                for _ in layers
            ]
            for _ in range(2)
        ]
        return layers, state, slots, batch, rounds

    def _make_backend(self, template, slots, steps, *, fused, ring=True):
        state = MambaPool.SpeculativeState(
            conv=[template.conv[0].clone()],
            temporal=template.temporal.clone(),
            intermediate_conv_window=[template.intermediate_conv_window[0].clone()],
            intermediate_ssm=(
                None
                if ring
                else template.temporal.new_zeros(
                    template.temporal.shape[0],
                    slots.numel(),
                    steps,
                    *template.temporal.shape[2:],
                )
            ),
            **{
                name: getattr(template, name).clone() if ring else None
                for name in (
                    "replayssm_rawv",
                    "replayssm_rawk",
                    "replayssm_g",
                    "replayssm_beta",
                )
            },
        )
        # Only the model/pool setup is a fixture. Verify, dispatch, ring fold and
        # conv rollback below all use the production backend and GPU kernels.
        backend = KDAAttnBackend.__new__(KDAAttnBackend)
        backend.req_to_token_pool = SimpleNamespace(
            mamba2_layer_cache=state.at_layer_idx,
            get_speculative_mamba2_params_all_layers=lambda: state,
            mamba_pool=SimpleNamespace(replayssm_is_kda=ring),
        )
        backend.forward_metadata = SimpleNamespace(
            query_start_loc=torch.arange(
                slots.numel() + 1, device="cuda", dtype=torch.int32
            )
            * steps,
            mamba_cache_indices=slots,
            retrieve_next_token=None,
            retrieve_next_sibling=None,
            retrieve_parent_token=None,
        )
        backend.verify_intermediate_state_indices = torch.arange(
            slots.numel(), device="cuda", dtype=torch.int32
        )
        backend.accept_lens_pool = None
        backend.kernel_dispatcher = KDAKernelDispatcher(
            LinearAttnKernelBackend.TRITON,
            LinearAttnKernelBackend.TRITON,
            LinearAttnKernelBackend.TRITON,
        )
        backend._fused_chain_verify_fn = (
            Mock(wraps=fused_kda_conv_gating_verify) if fused else None
        )
        hybrid = HybridLinearAttnBackend.__new__(HybridLinearAttnBackend)
        hybrid.linear_attn_backend = backend
        return backend, hybrid, state

    @staticmethod
    def _verify(backend, layers, batch, inputs):
        return [
            backend.forward_extend(layer, batch, mixed, a, b)
            for layer, (mixed, a, b) in zip(layers, inputs)
        ]

    def test_verify_commit_verify(self):
        # B=1 exercises the enabled path. Platform override makes the dispatch
        # testable on any CUDA CI runner; it does not replace a GPU kernel.
        for platform, (heads, v_heads, lower_bound), num_accept_tokens in (
            ({"is_sm90": True}, (2, 2, None), 1),
            ({"is_sm90": True}, (2, 2, None), 2),
            ({"is_sm90": True}, (2, 2, None), 4),
            ({"is_sm90": True}, (2, 4, -5.0), 1),
            ({"is_sm90": True}, (2, 4, -5.0), 2),
            ({"is_sm90": True}, (2, 4, -5.0), 4),
            ({"is_sm90": False, "is_sm100": True}, (2, 4, -5.0), 2),
        ):
            with (
                self.subTest(
                    platform=platform,
                    heads=heads,
                    v_heads=v_heads,
                    lower_bound=lower_bound,
                    num_accept_tokens=num_accept_tokens,
                ),
                override_platform(**platform),
            ):
                layers, initial, slots, batch, rounds = self._make_case(
                    heads=heads, v_heads=v_heads, lower_bound=lower_bound
                )
                fused, fused_hybrid, fused_state = self._make_backend(
                    initial, slots, 4, fused=True
                )
                reference, ref_hybrid, ref_state = self._make_backend(
                    initial, slots, 4, fused=False
                )
                snapshots, _, snapshot_state = self._make_backend(
                    initial, slots, 4, fused=False, ring=False
                )
                out_fused = self._verify(fused, layers, batch, rounds[0])
                out_ref = self._verify(reference, layers, batch, rounds[0])
                self._verify(snapshots, layers, batch, rounds[0])
                for actual, expected in zip(out_fused, out_ref):
                    torch.testing.assert_close(actual, expected, **_OUTPUT_TOL)
                for state in (fused_state, ref_state):
                    torch.testing.assert_close(
                        state.temporal, initial.temporal, rtol=0, atol=0
                    )

                last_steps = torch.full_like(slots, num_accept_tokens - 1)
                for hybrid in (fused_hybrid, ref_hybrid):
                    hybrid.update_mamba_state_after_mtp_verify(
                        last_correct_step_indices=last_steps,
                        mamba_track_indices=None,
                        mamba_steps_to_track=None,
                        model=None,
                    )
                torch.testing.assert_close(
                    fused_state.temporal, ref_state.temporal, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    fused_state.conv[0], ref_state.conv[0], rtol=0, atol=0
                )
                # Independent snapshot oracle: equality between two ring
                # arms alone would miss a shared no-op / wrong-step commit.
                expected_ssm = initial.temporal.clone()
                expected_ssm[:, slots.long()] = snapshot_state.intermediate_ssm[
                    :, :, num_accept_tokens - 1
                ]
                torch.testing.assert_close(
                    fused_state.temporal, expected_ssm, **_SNAPSHOT_ORACLE_TOL
                )
                expected_conv = initial.conv[0].clone()
                for i, (mixed, _, _) in enumerate(rounds[0]):
                    history = torch.cat(
                        (
                            initial.conv[0][i, slots.long()],
                            mixed.view(1, 4, -1)[:, :num_accept_tokens],
                        ),
                        dim=1,
                    )
                    expected_conv[i, slots.long()] = history[:, -3:]
                torch.testing.assert_close(
                    fused_state.conv[0], expected_conv, rtol=0, atol=0
                )

                out_fused = self._verify(fused, layers, batch, rounds[1])
                out_ref = self._verify(reference, layers, batch, rounds[1])
                for actual, expected in zip(out_fused, out_ref):
                    torch.testing.assert_close(actual, expected, **_OUTPUT_TOL)
                self.assertEqual(fused._fused_chain_verify_fn.call_count, 4)

    def test_ring_dispatch_falls_back(self):
        # B=2 and the measured regression sizes on the enabled architectures,
        # plus B=1 on an architecture without ring measurements.
        sm90 = {"is_sm90": True, "is_sm100": False}
        sm100 = {"is_sm90": False, "is_sm100": True}
        other = {"is_sm90": False, "is_sm100": False}
        for platform, batch_size in (
            (sm90, 2),
            (sm90, 4),
            (sm90, 16),
            (sm90, 64),
            (sm100, 2),
            (sm100, 4),
            (sm100, 16),
            (other, 1),
            (other, 4),
        ):
            with (
                self.subTest(platform=platform, batch_size=batch_size),
                override_platform(**platform),
            ):
                layers, initial, slots, batch, rounds = self._make_case(batch_size)
                backend, _, state = self._make_backend(initial, slots, 4, fused=True)
                reference, _, ref_state = self._make_backend(
                    initial, slots, 4, fused=False
                )
                out = self._verify(backend, layers, batch, rounds[0])
                ref = self._verify(reference, layers, batch, rounds[0])
                backend._fused_chain_verify_fn.assert_not_called()
                for actual, expected in zip(out, ref):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for name in (
                    "replayssm_rawv",
                    "replayssm_rawk",
                    "replayssm_g",
                    "replayssm_beta",
                ):
                    torch.testing.assert_close(
                        getattr(state, name), getattr(ref_state, name), rtol=0, atol=0
                    )

    def test_snapshot_dispatch_is_unchanged(self):
        for platform in (
            {"is_sm90": True, "is_sm100": False},
            {"is_sm90": False, "is_sm100": True},
            {"is_sm90": False, "is_sm100": False},
        ):
            with self.subTest(platform=platform), override_platform(**platform):
                layers, initial, slots, batch, rounds = self._make_case(batch_size=4)
                backend, _, _ = self._make_backend(
                    initial, slots, 4, fused=True, ring=False
                )
                reference, _, _ = self._make_backend(
                    initial, slots, 4, fused=False, ring=False
                )
                out = self._verify(backend, layers, batch, rounds[0])
                ref = self._verify(reference, layers, batch, rounds[0])
                self.assertEqual(backend._fused_chain_verify_fn.call_count, 2)
                for actual, expected in zip(out, ref):
                    torch.testing.assert_close(actual, expected, **_OUTPUT_TOL)


if __name__ == "__main__":
    unittest.main()
