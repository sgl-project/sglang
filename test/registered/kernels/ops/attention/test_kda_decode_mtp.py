import unittest
from unittest import mock

import torch

from sglang.kernels.ops.attention import kda_decode_mtp
from sglang.kernels.ops.attention.kda_decode_mtp import fused_kda_decode_mtp_dspark
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def _make_inputs(num_requests, num_spec, key_dim=128, num_heads=2):
    num_tokens = num_requests * (1 + num_spec)
    num_slots, conv_width = num_requests + 2, 4
    torch.manual_seed(5)
    x_q = torch.randn(
        1,
        num_tokens,
        num_heads,
        key_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    x_k = torch.randn_like(x_q)
    x_v = torch.randn_like(x_q)
    gate = torch.randn_like(x_q)
    beta = torch.randn(
        1,
        num_tokens,
        num_heads,
        device="cuda",
        dtype=torch.bfloat16,
    )
    conv_weight = [
        torch.randn(
            num_heads * key_dim,
            conv_width,
            device="cuda",
        )
        * 0.1
        for _ in range(3)
    ]
    conv_state = [
        torch.randn(
            num_slots,
            num_heads * key_dim,
            conv_width - 1,
            device="cuda",
            dtype=torch.bfloat16,
        )
        for _ in range(3)
    ]
    slots = torch.arange(1, num_requests + 1, device="cuda", dtype=torch.int32)
    scratch = torch.arange(num_requests, device="cuda", dtype=torch.int32)
    state = torch.randn(
        num_slots,
        num_heads,
        key_dim,
        key_dim,
        device="cuda",
    )
    intermediate_conv = torch.zeros(
        num_requests,
        1 + num_spec,
        num_heads * key_dim,
        conv_width - 1,
        device="cuda",
        dtype=torch.bfloat16,
    )
    return dict(
        x_q=x_q,
        x_k=x_k,
        x_v=x_v,
        w_q=conv_weight[0],
        w_k=conv_weight[1],
        w_v=conv_weight[2],
        cs_q=conv_state[0],
        cs_k=conv_state[1],
        cs_v=conv_state[2],
        g=gate,
        beta=beta,
        A_log=torch.randn(num_heads, device="cuda"),
        dt_bias=torch.randn(num_heads * key_dim, device="cuda"),
        recurrent_state=state,
        intermediate_state_indices=scratch,
        intermediate_conv_q=intermediate_conv.clone(),
        intermediate_conv_k=intermediate_conv.clone(),
        intermediate_conv_v=intermediate_conv.clone(),
        ssm_state_indices=slots,
        cu_seqlens=torch.arange(
            0,
            num_tokens + 1,
            1 + num_spec,
            device="cuda",
            dtype=torch.int32,
        ),
        lower_bound=-5.0,
    )


def _stride2_view(tensor):
    backing = torch.zeros(
        *tensor.shape[:-1], 2 * tensor.shape[-1], device="cuda", dtype=tensor.dtype
    )
    backing[..., ::2] = tensor
    return backing[..., ::2]


class TestKdaDecodeMtp(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if get_device_sm() < 100:
            raise unittest.SkipTest("Kimi K3 compute kernels require SM100a+")

    def _run(self, num_requests, num_spec, *, onorm, block_threads=None, strided=False):
        kwargs = _make_inputs(num_requests, num_spec)
        _, num_tokens, num_heads, key_dim = kwargs["x_v"].shape
        intermediate = torch.zeros(
            num_requests,
            1 + num_spec,
            num_heads,
            key_dim,
            key_dim,
            device="cuda",
        )
        gate = torch.randn(
            1, num_tokens, num_heads, key_dim, device="cuda", dtype=torch.bfloat16
        )
        weight = torch.rand(key_dim, device="cuda") + 0.5
        if strided:
            # Same values through views whose last mode has stride 2.
            kwargs["beta"] = _stride2_view(kwargs["beta"])
            gate = _stride2_view(gate)
        if onorm:
            kwargs.update(onorm_gate=gate, onorm_weight=weight, onorm_eps=1e-6)
        with mock.patch.object(
            kda_decode_mtp,
            "_block_threads",
            (lambda **_: block_threads)
            if block_threads
            else kda_decode_mtp._block_threads,
        ):
            output = fused_kda_decode_mtp_dspark(
                intermediate_ssm=intermediate, **kwargs
            )
        return output, intermediate, gate, weight

    def test_output_norm(self):
        for num_spec in (0, 2):
            with self.subTest(num_spec=num_spec):
                raw, raw_states, gate, weight = self._run(4, num_spec, onorm=False)
                fused, fused_states, _, _ = self._run(4, num_spec, onorm=True)
                raw = raw.float()
                expected = (
                    raw
                    * torch.rsqrt(raw.pow(2).mean(-1, keepdim=True) + 1e-6)
                    * weight
                    * torch.sigmoid(gate.float())
                )
                torch.testing.assert_close(
                    fused.float(), expected, atol=3e-2, rtol=3e-2
                )
                torch.testing.assert_close(fused_states, raw_states, atol=0, rtol=0)

    def test_strided_inputs_match_contiguous(self):
        for block_threads in (
            kda_decode_mtp.BLOCK_THREADS_NARROW,
            kda_decode_mtp.BLOCK_THREADS_WIDE,
        ):
            with self.subTest(block_threads=block_threads):
                dense = self._run(4, 5, onorm=True, block_threads=block_threads)
                strided = self._run(
                    4, 5, onorm=True, block_threads=block_threads, strided=True
                )
                torch.testing.assert_close(strided[0], dense[0], atol=0, rtol=0)
                torch.testing.assert_close(strided[1], dense[1], atol=0, rtol=0)

    def test_streaming_state_matches_resident(self):
        for onorm in (False, True):
            with self.subTest(onorm=onorm):
                resident = self._run(
                    4, 0, onorm=onorm, block_threads=kda_decode_mtp.BLOCK_THREADS_WIDE
                )
                streaming = self._run(
                    4, 0, onorm=onorm, block_threads=kda_decode_mtp.BLOCK_THREADS_NARROW
                )
                torch.testing.assert_close(
                    streaming[0], resident[0], atol=2e-2, rtol=2e-2
                )
                torch.testing.assert_close(
                    streaming[1], resident[1], atol=1e-4, rtol=1e-4
                )


_K = 128
_W = 4  # conv kernel width
_RING_LEN = 16


def _make_bf16_state_inputs(num_requests, num_spec, num_heads, seed):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)

    def randn(*shape, dtype=torch.float32, scale=1.0):
        return (
            torch.randn(*shape, device="cuda", generator=gen, dtype=torch.float32)
            * scale
        ).to(dtype)

    T = num_requests * (1 + num_spec)
    num_slots = num_requests + 3
    dim = num_heads * _K
    # Non-trivial slot ids (slot 0 left unused, as in the pool).
    slots = (
        torch.randperm(num_slots - 1, device="cuda", generator=gen)[:num_requests].to(
            torch.int32
        )
        + 1
    )
    return dict(
        x_q=randn(1, T, num_heads, _K, dtype=torch.bfloat16),
        x_k=randn(1, T, num_heads, _K, dtype=torch.bfloat16),
        x_v=randn(1, T, num_heads, _K, dtype=torch.bfloat16),
        w_q=randn(dim, _W, scale=0.1),
        w_k=randn(dim, _W, scale=0.1),
        w_v=randn(dim, _W, scale=0.1),
        cs_q=randn(num_slots, dim, _W - 1, dtype=torch.bfloat16),
        cs_k=randn(num_slots, dim, _W - 1, dtype=torch.bfloat16),
        cs_v=randn(num_slots, dim, _W - 1, dtype=torch.bfloat16),
        g=randn(1, T, num_heads, _K, dtype=torch.bfloat16),
        beta=randn(1, T, num_heads, dtype=torch.bfloat16),
        A_log=randn(num_heads),
        dt_bias=randn(dim),
        intermediate_state_indices=torch.arange(
            num_requests, device="cuda", dtype=torch.int32
        ),
        ssm_state_indices=slots,
        cu_seqlens=torch.arange(
            0, T + 1, 1 + num_spec, device="cuda", dtype=torch.int32
        ),
        lower_bound=-5.0,
        # Recurrent state with a realistic magnitude, rounded to bf16 so the
        # fp32 and bf16 arms start from the same values.
        h0=randn(num_slots, num_heads, _K, _K, scale=0.05).bfloat16().float(),
        onorm_gate=randn(1, T, num_heads, _K, dtype=torch.bfloat16),
        onorm_weight=torch.rand(_K, device="cuda", generator=gen) + 0.5,
    )


def _run_state_dtype(inp, state_dtype, *, ring, onorm, block_threads=None):
    """One verify call; returns (output, recurrent_state, side buffers)."""
    num_requests = inp["intermediate_state_indices"].numel()
    T = inp["x_v"].shape[1]
    t_req = T // num_requests
    num_heads = inp["x_v"].shape[2]
    num_slots = inp["h0"].shape[0]
    state = inp["h0"].to(state_dtype).clone()
    kwargs = {
        k: (v.clone() if torch.is_tensor(v) else v)
        for k, v in inp.items()
        if k not in ("h0", "onorm_gate", "onorm_weight")
    }
    for name in ("q", "k", "v"):
        kwargs[f"intermediate_conv_{name}"] = torch.zeros(
            num_requests,
            t_req,
            num_heads * _K,
            _W - 1,
            device="cuda",
            dtype=torch.bfloat16,
        )
    if onorm:
        kwargs.update(
            onorm_gate=inp["onorm_gate"],
            onorm_weight=inp["onorm_weight"],
            onorm_eps=1e-6,
        )
    side = {}
    if ring:
        side = dict(
            replayssm_rawv=torch.zeros(
                num_slots, num_heads, _RING_LEN, _K, device="cuda", dtype=torch.bfloat16
            ),
            replayssm_rawk=torch.zeros(
                num_slots, num_heads, _RING_LEN, _K, device="cuda", dtype=torch.bfloat16
            ),
            replayssm_g=torch.zeros(num_slots, num_heads, _RING_LEN, _K, device="cuda"),
            replayssm_beta=torch.zeros(num_slots, num_heads, _RING_LEN, device="cuda"),
        )
        kwargs.update(side)
        kwargs["intermediate_ssm"] = None
    else:
        side["intermediate_ssm"] = torch.zeros(
            num_requests, t_req, num_heads, _K, _K, device="cuda", dtype=state_dtype
        )
        kwargs["intermediate_ssm"] = side["intermediate_ssm"]
    with mock.patch.object(
        kda_decode_mtp,
        "_block_threads",
        (lambda **_: block_threads) if block_threads else kda_decode_mtp._block_threads,
    ):
        out = fused_kda_decode_mtp_dspark(recurrent_state=state, **kwargs)
    torch.cuda.synchronize()
    side["intermediate_conv"] = [kwargs[f"intermediate_conv_{n}"] for n in "qkv"]
    return out, state, side


class TestKdaDecodeMtpBf16State(CustomTestCase):
    """bf16 recurrent state (--mamba-ssm-dtype bfloat16).

    The kernel stages a bf16 state in smem, widens it to fp32 on load, runs the
    recurrence in fp32 and narrows only the intermediate_ssm store. So on a
    bf16-representable state it must reproduce the fp32 kernel bit for bit:
    outputs, fused output norm, conv windows and ReplaySSM rings equal, and
    snapshots equal to the fp32 snapshots rounded to bf16.
    """

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if get_device_sm() < 100:
            raise unittest.SkipTest("Kimi K3 compute kernels require SM100a+")

    def _assert_bf16_matches_fp32(self, ref, got, *, ring):
        out32, state32, side32 = ref
        out16, state16, side16 = got
        # Same fp32 math on the same (bf16-representable) state values.
        torch.testing.assert_close(out16, out32, rtol=0, atol=0)
        # Verify never writes the persistent state.
        torch.testing.assert_close(state16.float(), state32, rtol=0, atol=0)
        for c16, c32 in zip(side16["intermediate_conv"], side32["intermediate_conv"]):
            torch.testing.assert_close(c16, c32, rtol=0, atol=0)
        if ring:
            for name in (
                "replayssm_rawv",
                "replayssm_rawk",
                "replayssm_g",
                "replayssm_beta",
            ):
                torch.testing.assert_close(side16[name], side32[name], rtol=0, atol=0)
        else:
            snap16 = side16["intermediate_ssm"]
            self.assertEqual(snap16.dtype, torch.bfloat16)
            # Only the snapshot store narrows (round-to-nearest-even).
            torch.testing.assert_close(
                snap16, side32["intermediate_ssm"].to(torch.bfloat16), rtol=0, atol=0
            )

    def test_bf16_state_matches_fp32_kernel(self):
        cases = [
            # (num_requests, num_spec, num_heads)
            (4, 4, 2),
            (32, 4, 12),  # Kimi-K3.1 at TP8: 12 KDA heads per rank, 5 tokens
            (16, 6, 12),  # 7 tokens
            (3, 7, 2),  # 8 tokens, the gate's upper bound
        ]
        for n, num_spec, h in cases:
            for ring in (False, True):
                for onorm in (False, True):
                    with self.subTest(
                        n=n, num_spec=num_spec, h=h, ring=ring, onorm=onorm
                    ):
                        inp = _make_bf16_state_inputs(
                            n, num_spec, h, seed=n * 100 + num_spec
                        )
                        ref = _run_state_dtype(
                            inp, torch.float32, ring=ring, onorm=onorm
                        )
                        got = _run_state_dtype(
                            inp, torch.bfloat16, ring=ring, onorm=onorm
                        )
                        self._assert_bf16_matches_fp32(ref, got, ring=ring)

    def test_bf16_state_streaming_path(self):
        # Single-token step with the narrow block streams state tiles one at a time.
        inp = _make_bf16_state_inputs(4, 0, 2, seed=7)
        for onorm in (False, True):
            with self.subTest(onorm=onorm):
                ref = _run_state_dtype(
                    inp,
                    torch.float32,
                    ring=False,
                    onorm=onorm,
                    block_threads=kda_decode_mtp.BLOCK_THREADS_NARROW,
                )
                got = _run_state_dtype(
                    inp,
                    torch.bfloat16,
                    ring=False,
                    onorm=onorm,
                    block_threads=kda_decode_mtp.BLOCK_THREADS_NARROW,
                )
                self._assert_bf16_matches_fp32(ref, got, ring=False)

    def test_wrapper_rejects_bad_bf16_layouts(self):
        inp = _make_bf16_state_inputs(2, 2, 2, seed=1)
        # 16-byte cp.async: a bf16 slot must start on an 8-element boundary.
        flat = torch.zeros(inp["h0"].numel() + 4, device="cuda", dtype=torch.bfloat16)
        misaligned = flat[4:].view(inp["h0"].shape)
        with self.assertRaisesRegex(ValueError, "16-byte-aligned"):
            self._call_with_state(inp, misaligned, torch.bfloat16)
        # fp16 state is not supported.
        with self.assertRaisesRegex(ValueError, "fp32/bf16"):
            self._call_with_state(inp, inp["h0"].half(), torch.float16)
        # Snapshot buffer must use the state dtype.
        with self.assertRaisesRegex(ValueError, "intermediate_ssm dtype"):
            self._call_with_state(inp, inp["h0"].bfloat16(), torch.float32)

    @staticmethod
    def _call_with_state(inp, state, snap_dtype):
        n = inp["intermediate_state_indices"].numel()
        t_req = inp["x_v"].shape[1] // n
        h = inp["x_v"].shape[2]
        kwargs = {
            k: v
            for k, v in inp.items()
            if k not in ("h0", "onorm_gate", "onorm_weight")
        }
        for name in ("q", "k", "v"):
            kwargs[f"intermediate_conv_{name}"] = torch.zeros(
                n, t_req, h * _K, _W - 1, device="cuda", dtype=torch.bfloat16
            )
        fused_kda_decode_mtp_dspark(
            recurrent_state=state,
            intermediate_ssm=torch.zeros(
                n, t_req, h, _K, _K, device="cuda", dtype=snap_dtype
            ),
            **kwargs,
        )


if __name__ == "__main__":
    unittest.main()
