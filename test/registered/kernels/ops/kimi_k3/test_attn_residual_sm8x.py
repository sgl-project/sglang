"""Correctness coverage for the Kimi-K3 PPU SM8x attention-residual path."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.attn_residual import (
    AttnResidual,
    _aggregate_fused,
    _aggregate_sm8x,
    _use_sm8x_fused,
    aggregate_stream_torch,
    get_cw,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_H = 7168
_EPS = 1e-6


def _make_modules(seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    norm = RMSNorm(_H, eps=_EPS).to(device="cuda", dtype=torch.bfloat16)
    proj = ReplicatedLinear(
        _H, 1, bias=False, params_dtype=torch.bfloat16, quant_config=None
    ).to(device="cuda")
    with torch.no_grad():
        norm.weight.copy_(1 + 0.1 * torch.randn(_H, generator=gen, device="cuda"))
        proj.weight.copy_(torch.randn(1, _H, generator=gen, device="cuda") * _H**-0.5)
    return proj, norm


def _make_inputs(T: int, seed: int, num_bank_slots: int = 9):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    prefix = torch.randn(T, _H, generator=gen, device="cuda").to(torch.bfloat16)
    bank = torch.randn(T, num_bank_slots, _H, generator=gen, device="cuda").to(
        torch.bfloat16
    )
    return prefix, bank


def _sm8x_reference(prefix, delta, bank, nvb, proj, score_norm, out_norm):
    if delta is not None:
        prefix = (prefix.float() + delta.float()).to(prefix.dtype)
    rows = torch.cat([bank[:, :nvb], prefix.unsqueeze(1)], dim=1).float()
    cw = get_cw(proj, score_norm)
    scores = (rows * cw.float()).sum(-1)
    scores *= torch.rsqrt(rows.square().mean(-1) + _EPS)
    probs = torch.softmax(scores, dim=-1)
    mixed = (probs.unsqueeze(-1) * rows).sum(dim=1)
    normed = mixed * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + _EPS)
    return normed * out_norm.weight.float(), prefix


class TestKimiK3AttentionResidualSM8x(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        cls.proj, cls.norm = _make_modules(seed=0)

    def test_sm8x_fused_matches_reference_and_writes_bank(self):
        out_norm = RMSNorm(_H, eps=_EPS).to(device="cuda", dtype=torch.bfloat16)
        gen = torch.Generator(device="cuda").manual_seed(5)
        with torch.no_grad():
            out_norm.weight.copy_(
                1 + 0.1 * torch.randn(_H, generator=gen, device="cuda")
            )

        for nvb in range(1, 9):
            for T, use_delta, write_bank in (
                (1, False, False),
                (17, False, True),
                (17, True, True),
                (257, True, False),
            ):
                with self.subTest(nvb=nvb, T=T, delta=use_delta, write=write_bank):
                    prefix, bank = _make_inputs(T, 1000 * nvb + 10 * T + use_delta)
                    delta = torch.randn_like(prefix) if use_delta else None
                    prefix_ref, bank_ref = prefix.clone(), bank.clone()

                    out, updated_prefix = _aggregate_sm8x(
                        prefix,
                        bank,
                        nvb,
                        self.proj,
                        self.norm,
                        out_norm,
                        delta=delta,
                        write_bank_row=write_bank,
                    )
                    ref, expected_prefix = _sm8x_reference(
                        prefix_ref,
                        delta,
                        bank_ref,
                        nvb,
                        self.proj,
                        self.norm,
                        out_norm,
                    )
                    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=4e-2)
                    torch.testing.assert_close(updated_prefix, expected_prefix)

                    eager = out_norm(
                        aggregate_stream_torch(
                            expected_prefix,
                            bank_ref,
                            nvb,
                            self.proj,
                            self.norm,
                        )
                    )
                    fallback = _aggregate_fused(
                        expected_prefix,
                        bank_ref,
                        nvb,
                        self.proj,
                        self.norm,
                        out_norm,
                    )
                    torch.testing.assert_close(
                        out.float(), eager.float(), rtol=2e-2, atol=4e-2
                    )
                    torch.testing.assert_close(
                        out.float(), fallback.float(), rtol=2e-2, atol=4e-2
                    )
                    self.assertTrue(torch.equal(bank[:, :nvb], bank_ref[:, :nvb]))
                    if write_bank:
                        self.assertTrue(torch.equal(bank[:, nvb], expected_prefix))
                    else:
                        self.assertTrue(torch.equal(bank[:, nvb], bank_ref[:, nvb]))
                    self.assertTrue(
                        torch.equal(bank[:, nvb + 1 :], bank_ref[:, nvb + 1 :])
                    )
                    if use_delta:
                        self.assertTrue(torch.equal(prefix, prefix_ref))
                    else:
                        self.assertIs(updated_prefix, prefix)

    def test_sm8x_fuses_initial_snapshot(self):
        out_norm = RMSNorm(_H, eps=_EPS).to(device="cuda", dtype=torch.bfloat16)
        gen = torch.Generator(device="cuda").manual_seed(6)
        with torch.no_grad():
            out_norm.weight.copy_(
                1 + 0.1 * torch.randn(_H, generator=gen, device="cuda")
            )

        hidden_states, _ = _make_inputs(17, seed=7)
        residual = AttnResidual(hidden_states, block_num=2)
        with (
            patch("sglang.srt.layers.attn_residual._use_sm8x_fused", return_value=True),
            patch.object(
                residual,
                "write",
                side_effect=AssertionError("initial snapshot must be fused"),
            ),
        ):
            out, prefix = residual.forward(
                hidden_states,
                None,
                self.proj,
                self.norm,
                out_norm,
                write=True,
            )

        torch.testing.assert_close(
            out.float(), out_norm(hidden_states).float(), rtol=2e-2, atol=4e-2
        )
        self.assertIs(prefix, hidden_states)
        self.assertEqual(residual.num_valid_blocks, 1)
        self.assertTrue(torch.equal(residual.block_residual[:, 0], hidden_states))

    def test_sm8x_dispatch_is_ppu_only(self):
        with (
            patch("sglang.srt.layers.attn_residual.is_ppu", return_value=False),
            patch(
                "sglang.srt.layers.attn_residual.torch.cuda.get_device_capability",
                return_value=(8, 9),
            ),
            patch.object(torch.version, "hip", None),
            patch("sglang.srt.layers.attn_residual._SM8X_FUSED_SUPPORTED", None),
        ):
            self.assertFalse(_use_sm8x_fused(_H))

        with (
            patch("sglang.srt.layers.attn_residual.is_ppu", return_value=True),
            patch(
                "sglang.srt.layers.attn_residual.torch.cuda.get_device_capability",
                return_value=(8, 9),
            ),
            patch.object(torch.version, "hip", None),
            patch("sglang.srt.layers.attn_residual._SM8X_FUSED_SUPPORTED", None),
        ):
            self.assertTrue(_use_sm8x_fused(_H))
            self.assertFalse(_use_sm8x_fused(4096))


if __name__ == "__main__":
    unittest.main()
