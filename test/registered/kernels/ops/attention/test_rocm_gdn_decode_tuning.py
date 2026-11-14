"""The ROCm GDN decode tuning must be bit-identical to the stock launches it replaces:
the packed recurrent decode with state cache hints, and the fused QKVZ/BA Conv1D
update with the state-reusing kernel at 4 warps. Outputs and whole state pools
(untouched slots included) are compared over consecutive steps at the Qwen3.8-27B
TP1/TP2 head shards, decode batch widths, and pad / out-of-range slots.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")

# (num_k_heads, num_v_heads) per rank of Qwen3.8-27B at TP1 and TP2.
HEADS = ((16, 48), (8, 24))
HEAD_DIM = 128
WIDTHS = (1, 9, 32, 64, 100, 128, 256)
SLOTS = 300
STEPS = 3


def _hip() -> bool:
    return torch.cuda.is_available() and torch.version.hip is not None


def _indices(batch: int, gen: torch.Generator) -> torch.Tensor:
    idx = torch.randperm(SLOTS, generator=gen)[:batch].to(torch.int32) + 1
    if batch > 2:
        idx[batch // 3] = -1
    return idx.cuda()


@unittest.skipUnless(_hip(), "ROCm only")
class RecurrentPackedDecodeTest(CustomTestCase):
    def _run(self, fr, tuned, inputs, state):
        fr._stream_packed_decode_state = tuned
        outs = []
        for mixed_qkv, a, b, A_log, dt_bias, idx in inputs:
            out = mixed_qkv.new_empty(mixed_qkv.shape[0], 1, *state.shape[-3:-1])
            fr.fused_recurrent_gated_delta_rule_packed_decode(
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=HEAD_DIM**-0.5,
                initial_state=state,
                out=out,
                ssm_state_indices=idx,
                use_qk_l2norm_in_kernel=True,
            )
            outs.append(out)
        return outs

    def test_bit_exact(self):
        from sglang.kernels.ops.attention.fla import fused_recurrent as fr

        saved = fr._stream_packed_decode_state
        gen = torch.Generator().manual_seed(0)
        try:
            for h, hv in HEADS:
                state0 = torch.randn(SLOTS + 1, hv, HEAD_DIM, HEAD_DIM, generator=gen)
                state0 = (state0 * 0.1).cuda()
                A_log = torch.randn(hv, generator=gen).cuda()
                dt_bias = torch.randn(hv, generator=gen).cuda()
                for batch in WIDTHS:
                    idx = _indices(batch, gen)
                    inputs = [
                        (
                            torch.randn(batch, (2 * h + hv) * HEAD_DIM, generator=gen)
                            .to(torch.bfloat16)
                            .cuda(),
                            torch.randn(batch, hv, generator=gen)
                            .to(torch.bfloat16)
                            .cuda(),
                            torch.randn(batch, hv, generator=gen)
                            .to(torch.bfloat16)
                            .cuda(),
                            A_log,
                            dt_bias,
                            idx,
                        )
                        for _ in range(STEPS)
                    ]
                    ref_state, new_state = state0.clone(), state0.clone()
                    ref = self._run(fr, False, inputs, ref_state)
                    new = self._run(fr, True, inputs, new_state)
                    with self.subTest(tp_heads=(h, hv), batch=batch):
                        for r, n in zip(ref, new):
                            self.assertTrue(
                                torch.equal(r.view(torch.int16), n.view(torch.int16))
                            )
                        self.assertTrue(
                            torch.equal(
                                ref_state.view(torch.int32), new_state.view(torch.int32)
                            )
                        )
        finally:
            fr._stream_packed_decode_state = saved


@unittest.skipUnless(_hip(), "ROCm only")
class ConvUpdateTest(CustomTestCase):
    def _run(self, m, tuned, inputs, state, weight, bias, activation, dims):
        m._tune_gdn_decode = tuned
        qkv_dim, hv = dims
        outs = []
        for qkvz, ba, idx in inputs:
            outs.extend(
                m.fused_qkvzba_causal_conv1d_update_contiguous(
                    qkvz,
                    ba,
                    state,
                    weight,
                    bias,
                    idx,
                    qkv_dim=qkv_dim,
                    v_dim=hv * HEAD_DIM,
                    num_v_heads=hv,
                    head_v_dim=HEAD_DIM,
                    activation=activation,
                )
            )
        return outs

    def test_bit_exact(self):
        import sglang.kernels.ops.attention.triton_gdn_fused_proj as m

        saved = m._tune_gdn_decode
        gen = torch.Generator().manual_seed(1)
        # (kernel width, physical state length, bias, activation)
        variants = (
            (4, 3, False, "silu"),
            (4, 3, True, None),
            (3, 3, False, "silu"),
            (2, 1, True, "silu"),
        )
        try:
            for h, hv in HEADS:
                qkv_dim = (2 * h + hv) * HEAD_DIM
                for width, state_len, has_bias, activation in variants:
                    state0 = torch.randn(SLOTS + 1, qkv_dim, state_len, generator=gen)
                    state0 = state0.to(torch.bfloat16).cuda()
                    weight = (torch.randn(qkv_dim, width, generator=gen) * 0.3).to(
                        torch.bfloat16
                    )
                    weight = weight.cuda()
                    bias = (
                        torch.randn(qkv_dim, generator=gen).to(torch.bfloat16).cuda()
                        if has_bias
                        else None
                    )
                    for batch in WIDTHS:
                        idx = _indices(batch, gen)
                        if batch > 4:
                            idx[batch // 2] = SLOTS + 7  # out of range: treated as pad
                        inputs = [
                            (
                                torch.randn(
                                    batch, qkv_dim + hv * HEAD_DIM, generator=gen
                                )
                                .to(torch.bfloat16)
                                .cuda(),
                                torch.randn(batch, 2 * hv, generator=gen)
                                .to(torch.bfloat16)
                                .cuda(),
                                idx,
                            )
                            for _ in range(STEPS)
                        ]
                        ref_state, new_state = state0.clone(), state0.clone()
                        args = (weight, bias, activation, (qkv_dim, hv))
                        ref = self._run(m, False, inputs, ref_state, *args)
                        new = self._run(m, True, inputs, new_state, *args)
                        with self.subTest(
                            tp_heads=(h, hv), width=width, bias=has_bias, batch=batch
                        ):
                            for r, n in zip(ref, new):
                                self.assertTrue(
                                    torch.equal(
                                        r.view(torch.int16), n.view(torch.int16)
                                    )
                                )
                            self.assertTrue(
                                torch.equal(
                                    ref_state.view(torch.int16),
                                    new_state.view(torch.int16),
                                )
                            )
        finally:
            m._tune_gdn_decode = saved


if __name__ == "__main__":
    unittest.main()
