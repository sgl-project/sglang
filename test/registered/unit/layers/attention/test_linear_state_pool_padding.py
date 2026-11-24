import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
    FlashInferGDNKernel,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _kernel():
    kernel = FlashInferGDNKernel.__new__(FlashInferGDNKernel)
    kernel.use_state_pool = False
    kernel._prefill_needs_fp32_state = False
    kernel._aligned_input_buffers = {}
    kernel._aligned_parameter_cache = {}
    kernel._alignment_fallback_warned = False
    return kernel


def _states():
    envelope = torch.full((8, 48), 19.0)
    states = torch.as_strided(envelope, (8, 1, 4, 4), (48, 16, 4, 1))
    states[0].fill_(float("nan"))
    states[1].fill_(3)
    return states


class TestLinearStatePoolPadding(CustomTestCase):
    def test_sm90_decode_keeps_padding_out_of_reserved_tail(self):
        """Padded decode reads the backed sink and never overwrites a live or tail row."""
        kernel = _kernel()
        states = _states()
        captured = {}

        def decode(**kw):
            captured["state"] = kw["state"].clone()
            changes = torch.arange(1, 5, dtype=states.dtype).view(4, 1, 1, 1)
            return torch.full_like(kw["v"], 7), kw["state"] + changes

        kernel._decode_fn = decode
        qkv = torch.ones((1, 4, 1, 4))
        gate = torch.zeros((1, 4, 1))
        result = kernel.decode(
            qkv,
            qkv,
            qkv,
            gate,
            gate,
            A_log=torch.zeros(1),
            dt_bias=torch.zeros(1),
            ssm_states=states,
            cache_indices=torch.tensor([1, -1, -1, 0], dtype=torch.int32),
            query_start_loc=torch.arange(5, dtype=torch.int32),
        )
        torch.testing.assert_close(captured["state"][0], torch.full((1, 4, 4), 3.0))
        self.assertEqual(torch.count_nonzero(captured["state"][1:]), 0)
        torch.testing.assert_close(states[1], torch.full((1, 4, 4), 4.0))
        self.assertEqual(torch.count_nonzero(states[0]), 0)
        torch.testing.assert_close(states[2:], torch.full((6, 1, 4, 4), 19.0))
        torch.testing.assert_close(result[:, :1], torch.full((1, 1, 1, 4), 7.0))
        self.assertEqual(torch.count_nonzero(result[:, 1:]), 0)

    def test_flashinfer_prefill_uses_the_same_sink_for_every_architecture(self):
        """Empty prefill lanes neither gather reserved-tail state nor scatter into it."""
        for state_pool in (False, True):
            with self.subTest(state_pool=state_pool):
                kernel = _kernel()
                kernel.use_state_pool = state_pool
                states = _states()
                captured = {}

                def prefill(**kw):
                    captured["initial"] = kw["initial_state"].clone()
                    kw["output_state"].copy_(kw["initial_state"] + 2)
                    return torch.zeros_like(kw["v"]), kw["output_state"]

                kernel._prefill_fn = prefill
                qkv = torch.ones((1, 1, 1, 4))
                gate = torch.zeros((1, 1, 1))
                with patch(
                    "sglang.kernels.ops.attention.fla.l2norm.gdn_prefill_qkv_prepare_fwd",
                    side_effect=lambda q, k, v: (q, k, v),
                ):
                    kernel.extend(
                        qkv,
                        qkv,
                        qkv,
                        gate,
                        gate,
                        ssm_states=states,
                        cache_indices=torch.tensor([1, -1, -1, 0], dtype=torch.int32),
                        query_start_loc=torch.tensor(
                            [0, 1, 1, 1, 1], dtype=torch.int32
                        ),
                    )
                self.assertEqual(torch.count_nonzero(captured["initial"][1:]), 0)
                torch.testing.assert_close(states[1], torch.full((1, 4, 4), 5.0))
                self.assertEqual(torch.count_nonzero(states[0]), 0)
                torch.testing.assert_close(states[2:], torch.full((6, 1, 4, 4), 19.0))

    def test_cutedsl_prefill_passes_a_backed_slot_for_empty_sequences(self):
        """CuTe TMA can load H0 for an empty sequence, so its slot must stay backed."""
        from sglang.srt.layers.attention.linear.kernels.gdn_cutedsl import (
            CuteDSLGDNKernel,
        )
        from sglang.srt.layers.attention.linear.kernels.kda_cutedsl import (
            CuteDSLKDAKernel,
        )

        for cls in (CuteDSLGDNKernel, CuteDSLKDAKernel):
            with self.subTest(kernel=cls.__name__):
                kernel = cls.__new__(cls)
                kernel._l2norm_fn = lambda t: t
                kernel._prepare_meta_fn = lambda *args, **kw: (None, None)
                captured = {}

                def extend(*args, **kw):
                    captured["indices"] = kw.get(
                        "initial_state_indices", kw.get("h0_indices")
                    )
                    return torch.zeros((1, 1, 1, 4)), None

                kernel._extend_fn = extend
                qkv = torch.ones((1, 1, 1, 4))
                gate = torch.zeros((1, 1, 1))
                kernel.extend(
                    qkv,
                    qkv,
                    qkv,
                    gate,
                    gate,
                    ssm_states=_states(),
                    cache_indices=torch.tensor([1, -1, -1], dtype=torch.int32),
                    query_start_loc=torch.tensor([0, 1, 1, 1], dtype=torch.int32),
                )
                torch.testing.assert_close(
                    captured["indices"], torch.tensor([1, 0, 0], dtype=torch.int32)
                )


if __name__ == "__main__":
    unittest.main()
