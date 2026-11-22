"""The AITER fused KDA verify's coverage predicate runs without the kernel."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.attention import kda_fused_verify_aiter_hip as adapter
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_HEADS = 12
_DIM = 128
_CHANNELS = 3 * _HEADS * _DIM
_CONV_WIDTH = 4
_BATCH, _STEPS, _SLOTS = 3, 5, 7


def _empty(*shape, dtype=torch.bfloat16):
    return torch.empty(*shape, dtype=dtype, device="meta")


def _ring(ring_len, g_dtype=torch.float32):
    return (
        _empty(_SLOTS, _HEADS, ring_len, _DIM),
        _empty(_SLOTS, _HEADS, ring_len, _DIM),
        _empty(_SLOTS, _HEADS, ring_len, _DIM, dtype=g_dtype),
        _empty(_SLOTS, _HEADS, ring_len, dtype=torch.float32),
    )


def _kwargs(ring):
    tokens = _BATCH * _STEPS
    return dict(
        f_a=_empty(tokens, _DIM),
        mixed_qkv=_empty(tokens, _CHANNELS),
        raw_beta=_empty(1, tokens, _HEADS),
        output_gate=_empty(tokens, _HEADS, _DIM),
        conv_state=_empty(_SLOTS, _CONV_WIDTH - 1, _CHANNELS).transpose(-1, -2),
        state=_empty(_SLOTS, _HEADS, _DIM, _DIM),
        cache_indices=_empty(_BATCH, dtype=torch.int32),
        intermediate_indices=_empty(_BATCH, dtype=torch.int32),
        intermediate_conv_window=_empty(
            _SLOTS, _STEPS, _CONV_WIDTH - 1, _CHANNELS
        ).transpose(-1, -2),
        intermediate_states=(
            None if ring else _empty(_SLOTS, _STEPS, _HEADS, _DIM, _DIM)
        ),
        replay_ring=_ring(2 * _STEPS) if ring else None,
        num_draft_tokens=_STEPS,
        armed_draft_tokens=_STEPS,
    )


class TestFusedVerifyCovered(CustomTestCase):
    """A draft-token count other than the warmed one would JIT-build inside graph
    capture, and a cache dtype or ring shape AITER rejects would raise; both fall
    back. covered() reads only max_tokens and state_dtypes from the op."""

    def test_covered_falls_back_instead_of_aiter_raising(self):
        op = SimpleNamespace(max_tokens=8, state_dtypes=(torch.bfloat16, torch.float32))
        for ring in (False, True):
            kwargs = _kwargs(ring)
            rejected = [
                {"armed_draft_tokens": _STEPS + 1},
                {"state": _empty(_SLOTS, _HEADS, _DIM, _DIM, dtype=torch.float16)},
                {
                    "conv_state": _empty(
                        _SLOTS, _CONV_WIDTH - 1, _CHANNELS, dtype=torch.float16
                    ).transpose(-1, -2)
                },
            ]
            if ring:
                rejected += [
                    {"replay_ring": _ring(_STEPS - 1)},
                    {"replay_ring": _ring(2 * _STEPS, g_dtype=torch.bfloat16)},
                ]
            else:
                rejected.append(
                    {
                        "intermediate_states": _empty(
                            _SLOTS, _STEPS, _HEADS, _DIM, _DIM + 8
                        )
                    }
                )
            with (
                self.subTest(ring=ring),
                mock.patch.object(adapter, "_op", return_value=op),
            ):
                self.assertTrue(adapter.covered(**kwargs))
                for change in rejected:
                    with self.subTest(change=list(change)):
                        self.assertFalse(adapter.covered(**{**kwargs, **change}))


if __name__ == "__main__":
    unittest.main()
