from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=6, stage="base-b", runner_config="1-gpu-small")

import importlib.util
import unittest

import torch

TILE_K = 128


def _sm100():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


def _strided_replica(shape, slot_stride, dtype, device):
    """A tensor whose slot dim (dim 0) has an ARTIFICIALLY large stride, with
    zeroed gap bytes — the unified pool's envelope-strided state layout, scaled
    so `slot * stride` exceeds int32 at small slot ids."""
    inner = 1
    for s in shape[1:]:
        inner *= s
    reach = (shape[0] - 1) * slot_stride + inner
    base = torch.zeros(reach, dtype=dtype, device=device)
    strides = [slot_stride]
    acc = inner
    for s in shape[1:]:
        acc //= s
        strides.append(acc)
    return base.as_strided(tuple(shape), tuple(strides))


@unittest.skipUnless(_sm100(), "SM100-only CuTe kernel")
@unittest.skipUnless(
    importlib.util.find_spec("cutlass") is not None, "nvidia-cutlass-dsl required"
)
class TestKdaDecodeMtpSlotStride(unittest.TestCase):
    """Root-cause guard: `slot * stride` must be computed in int64.

    The DSPARK KDA verify kernel compiles with STATIC CuTe layouts, so a
    state-pool slot stride that individually fits int32 folds into 32-bit
    arithmetic and `slot * stride` wraps mod 2^32 once the product exceeds
    int32 — reads land inside other slots (silent corruption) or off the
    allocation (illegal access). The unified pool's envelope-strided KDA
    views reach that regime at slot ids ~153 (conv) / ~306 (ssm). This test
    reproduces the regime with an artificially large ssm slot stride at a
    small slot id and asserts bitwise parity against a contiguous pool."""

    def test_wrap_regime_matches_contiguous(self):
        from sglang.kernels.ops.kimi_k3.kda_decode_mtp import (
            fused_kda_decode_mtp_dspark,
        )

        device = "cuda"
        torch.manual_seed(3)
        H, num_spec = 2, 7
        T, N = 1 + num_spec, 1
        dim = H * TILE_K

        # slot * stride crosses 2^31 elements at slot 8. The stride must NOT
        # be a power of two: pow2 constants lower to shifts, which dodge the
        # 32-bit imul this test pins (the real pool strides — e.g. K3's
        # 14,042,880 ssm / 28,085,760 conv — are not pow2). Multiple of 4
        # (wrapper's cp.async alignment contract). Both state families get
        # the huge stride: in the production repro the conv direct-index path
        # (cs_q[slot, ch, w]) wrapped at lower slot ids than the ssm tiled
        # copy, so pinning only one path can silently pass.
        slot_id, slots = 8, 9
        ssm_slot_stride = (1 << 28) + 12_344  # fp32 base ~8.6 GB
        conv_slot_stride = (1 << 28) + 23_448  # bf16 base ~4.3 GB x3
        free = torch.cuda.mem_get_info()[0]
        if free < 26 << 30:
            self.skipTest(f"needs ~26GB free GPU memory, have {free >> 30}GB")

        def acts(shape, dtype=torch.bfloat16):
            return (torch.randn(shape, device=device, dtype=torch.float32) * 0.1).to(
                dtype
            )

        x_q, x_k, x_v, g = (acts((1, T, H, TILE_K)) for _ in range(4))
        beta = acts((1, T, H))
        w = torch.randn(3 * dim, 4, device=device, dtype=torch.float32) * 0.1
        w_q, w_k, w_v = w.split([dim, dim, dim], dim=0)
        A_log = torch.randn(H, device=device, dtype=torch.float32) * 0.1
        dt_bias = torch.randn(dim, device=device, dtype=torch.float32) * 0.1

        state_c = torch.randn(
            slots, H, TILE_K, TILE_K, device=device, dtype=torch.float32
        )
        # conv pool in the backend's post-split/transpose shape [slots, dim, 3]
        # with the production stride pattern (slot_stride, 1, dim): the
        # underlying envelope is [slots, 3, dim] and the backend transposes.
        conv_c = [
            (torch.randn(slots, 3, dim, device=device, dtype=torch.float32) * 0.1)
            .to(torch.bfloat16)
            .transpose(-1, -2)
            for _ in range(3)
        ]
        inter_ssm = torch.zeros(
            2, T, H, TILE_K, TILE_K, device=device, dtype=torch.float32
        )
        inter_conv = [
            torch.zeros(2, T, dim, 3, device=device, dtype=torch.bfloat16)
            for _ in range(3)
        ]
        common = dict(
            x_q=x_q,
            x_k=x_k,
            x_v=x_v,
            w_q=w_q,
            w_k=w_k,
            w_v=w_v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            intermediate_state_indices=torch.zeros(N, dtype=torch.int32, device=device),
            ssm_state_indices=torch.full(
                (N,), slot_id, dtype=torch.int32, device=device
            ),
            cu_seqlens=torch.tensor([0, T], dtype=torch.int32, device=device),
            lower_bound=-5.0,
        )

        def run(state, conv, issm, iconv):
            out = fused_kda_decode_mtp_dspark(
                recurrent_state=state,
                cs_q=conv[0],
                cs_k=conv[1],
                cs_v=conv[2],
                intermediate_ssm=issm,
                intermediate_conv_q=iconv[0],
                intermediate_conv_k=iconv[1],
                intermediate_conv_v=iconv[2],
                **common,
            )
            torch.cuda.synchronize()
            return out

        ref = run(state_c, conv_c, inter_ssm.clone(), [c.clone() for c in inter_conv])

        state_s = _strided_replica(
            (slots, H, TILE_K, TILE_K), ssm_slot_stride, torch.float32, device
        )
        state_s.copy_(state_c)
        conv_s = []
        for c in conv_c:
            v = _strided_replica(
                (slots, 3, dim), conv_slot_stride, torch.bfloat16, device
            ).transpose(-1, -2)
            v.copy_(c)
            conv_s.append(v)
        issm_s = inter_ssm.clone()
        iconv_s = [c.clone() for c in inter_conv]
        got = run(state_s, conv_s, issm_s, iconv_s)

        # Pre-fix: 32-bit `slot * stride` wraps (8 * 2^28 = 2^31) and the read
        # lands at offset 0 of the pool — silently returning slot 0's state —
        # or off the allocation. Post-fix: bit-exact.
        torch.testing.assert_close(got, ref, rtol=0, atol=0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
