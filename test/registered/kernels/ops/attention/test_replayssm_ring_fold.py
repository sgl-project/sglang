"""Representative parity coverage for the lightweight Kimi-K3 prerequisites."""

import unittest

import torch

from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.kernels.ops.attention.fla.kda_replayssm_spec_decode import (
    commit_kda_replayssm_spec,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

NUM_EXPERTS = 896

TOPK = 16

NOPE_DIM = 512

ROPE_DIM = 64

MLA_DIM = NOPE_DIM + ROPE_DIM

MLA_PAGES = 256


class TestKimiK3PrerequisiteOps(CustomTestCase):
    def test_replayssm_ring_fold(self):
        batch_size, num_steps = 8, 4
        num_value_heads, num_key_heads = 8, 2
        key_dim = value_dim = 128
        ring_size = 16
        torch.manual_seed(6)

        q = torch.randn(
            batch_size,
            num_steps,
            num_key_heads,
            key_dim,
            device="cuda",
        )
        k = torch.randn_like(q)
        v = torch.randn(
            batch_size,
            num_steps,
            num_value_heads,
            value_dim,
            device="cuda",
        )
        a = torch.randn(
            batch_size,
            num_steps,
            num_value_heads,
            key_dim,
            device="cuda",
        )
        b = torch.randn(batch_size, num_steps, num_value_heads, device="cuda")
        a_log = torch.randn(num_value_heads, device="cuda")
        dt_bias = torch.randn(num_value_heads, key_dim, device="cuda")
        slots = torch.arange(1, batch_size + 1, device="cuda", dtype=torch.int32)
        slots[-1] = -1
        num_slots = batch_size + 1
        state = torch.randn(
            num_slots,
            num_value_heads,
            value_dim,
            key_dim,
            device="cuda",
        )
        intermediate = torch.zeros(
            num_slots,
            num_steps,
            num_value_heads,
            value_dim,
            key_dim,
            device="cuda",
        )
        raw_v = torch.zeros(
            num_slots,
            num_value_heads,
            ring_size,
            value_dim,
            device="cuda",
        )
        raw_k = torch.zeros(
            num_slots,
            num_key_heads,
            ring_size,
            key_dim,
            device="cuda",
        )
        gate = torch.zeros_like(raw_v)
        beta = torch.zeros(
            num_slots,
            num_value_heads,
            ring_size,
            device="cuda",
        )

        fused_sigmoid_gating_delta_rule_update(
            A_log=a_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=state,
            initial_state_indices=slots,
            scale=key_dim**-0.5,
            use_qk_l2norm_in_kernel=True,
            is_kda=True,
            lower_bound=-5.0,
            disable_state_update=True,
            intermediate_states_buffer=intermediate,
            intermediate_state_indices=slots,
            cache_steps=num_steps,
            cache_ring=True,
            replayssm_rawv=raw_v,
            replayssm_rawk=raw_k,
            replayssm_g=gate,
            replayssm_beta=beta,
        )
        checkpoint = state.clone()
        commit_kda_replayssm_spec(
            checkpoint,
            raw_v,
            raw_k,
            gate,
            beta,
            slots,
            torch.full((batch_size,), num_steps, device="cuda", dtype=torch.int32),
            max_cache_len=ring_size,
            num_k_heads=num_key_heads,
            use_qk_l2norm_in_kernel=True,
            null_block_id=-1,
        )
        for slot in slots[:-1].tolist():
            expected = intermediate[slot, num_steps - 1]
            actual = checkpoint[slot]
            relative_error = (
                actual - expected
            ).abs().max() / expected.abs().max().clamp_min(1e-6)
            self.assertLess(relative_error.item(), 1e-3)


if __name__ == "__main__":
    unittest.main()
