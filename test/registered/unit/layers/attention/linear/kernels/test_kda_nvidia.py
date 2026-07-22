"""Unit tests for the NVIDIA KDA prefill routing/repacking wrapper."""

import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.linear.kernels.kda_nvidia import (
    NvidiaKDAKernel,
    _from_nvidia_kda_state_layout,
    _to_nvidia_kda_state_layout,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RejectTriton:
    def extend(self, *args, **kwargs):
        raise AssertionError("ordinary prefill unexpectedly fell back to Triton")


class TestNvidiaKDAAllPrefillWrapper(CustomTestCase):
    def test_state_layout_round_trip(self):
        state = torch.arange(2 * 3 * 5 * 7, dtype=torch.float32).view(2, 3, 5, 7)

        nvidia_kda_state = _to_nvidia_kda_state_layout(
            state, head_k_dim=7, head_v_dim=5
        )

        self.assertEqual(tuple(nvidia_kda_state.shape), (2, 3, 7, 5))
        self.assertTrue(nvidia_kda_state.is_contiguous())
        self.assertEqual(nvidia_kda_state[1, 2, 6, 4], state[1, 2, 4, 6])

        restored = _from_nvidia_kda_state_layout(
            nvidia_kda_state,
            head_k_dim=7,
            head_v_dim=5,
            dtype=torch.bfloat16,
        )

        self.assertEqual(tuple(restored.shape), (2, 3, 5, 7))
        self.assertTrue(restored.is_contiguous())
        self.assertTrue(torch.equal(restored, state.to(torch.bfloat16)))

    def test_state_layout_rejects_swapped_contract(self):
        with self.assertRaisesRegex(ValueError, "SGLang KDA state"):
            _to_nvidia_kda_state_layout(
                torch.zeros(1, 2, 7, 5),
                head_k_dim=7,
                head_v_dim=5,
            )

    def _make_kernel(self):
        calls = []
        kernel = NvidiaKDAKernel()
        kernel._l2norm = lambda x: x
        kernel._triton = _RejectTriton()

        def fake_fwd(q, k, v, g, beta, **kwargs):
            calls.append(
                {
                    "q": q.clone(),
                    "k": k.clone(),
                    "v": v.clone(),
                    "g": g.clone(),
                    "beta": beta.clone(),
                    "initial_state": kwargs["initial_state"].clone(),
                    "cu_seqlens": kwargs["cu_seqlens"],
                }
            )
            return v.clone(), kwargs["initial_state"] + 1.0

        kernel._fwd = fake_fwd
        return kernel, calls

    @staticmethod
    def _inputs(seq_lens):
        total = sum(seq_lens)
        token_values = torch.arange(total * 128, dtype=torch.bfloat16).view(
            1, total, 1, 128
        )
        query_start_loc = torch.tensor(
            [0] + list(torch.tensor(seq_lens).cumsum(0).tolist()), dtype=torch.int32
        )
        return {
            "q": token_values + 100,
            "k": token_values + 200,
            "v": token_values + 300,
            "g": (token_values + 400).view(1, total, 128),
            "beta": torch.arange(total, dtype=torch.float32).view(1, total, 1),
            "query_start_loc": query_start_loc,
        }

    def test_short_single_sequence_uses_triton(self):
        kernel, calls = self._make_kernel()
        kernel._triton.extend = Mock(return_value="triton")
        x = self._inputs([5])
        states = torch.zeros(3, 1, 128, 128, dtype=torch.bfloat16)

        output = kernel.extend(
            x["q"],
            x["k"],
            x["v"],
            x["g"],
            x["beta"],
            ssm_states=states,
            cache_indices=torch.tensor([1], dtype=torch.int32),
            query_start_loc=x["query_start_loc"],
            extend_seq_lens_cpu=[5],
            A_log=torch.zeros(128, dtype=torch.float32),
        )

        self.assertEqual(output, "triton")
        self.assertEqual(calls, [])
        kernel._triton.extend.assert_called_once()
        self.assertTrue(torch.count_nonzero(states).item() == 0)

    def test_packed_multi_sequence_repacking_preserves_order_and_slots(self):
        kernel, calls = self._make_kernel()
        seq_lens = [2, 3, 1]
        x = self._inputs(seq_lens)
        states = torch.arange(5 * 128 * 128, dtype=torch.bfloat16).view(5, 1, 128, 128)
        states_before = states.clone()
        slots = torch.tensor([2, 0, 4], dtype=torch.int32)

        output = kernel.extend(
            x["q"],
            x["k"],
            x["v"],
            x["g"],
            x["beta"],
            ssm_states=states,
            cache_indices=slots,
            query_start_loc=x["query_start_loc"],
            extend_seq_lens_cpu=seq_lens,
            A_log=torch.zeros(128, dtype=torch.float32),
        )

        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(tuple(call["q"].shape), (3, 2048, 1, 128))
        self.assertIsNone(call["cu_seqlens"])
        self.assertTrue(torch.equal(output, x["v"]))

        start = 0
        for row, length in enumerate(seq_lens):
            end = start + length
            self.assertTrue(torch.equal(call["v"][row, :length], x["v"][0, start:end]))
            self.assertTrue(torch.count_nonzero(call["q"][row, length:]).item() == 0)
            self.assertTrue(torch.count_nonzero(call["k"][row, length:]).item() == 0)
            self.assertTrue(torch.count_nonzero(call["v"][row, length:]).item() == 0)
            self.assertTrue(torch.count_nonzero(call["beta"][row, length:]).item() == 0)
            self.assertTrue(torch.all(call["g"][row, length:] == -1000))
            start = end

        for slot in slots.tolist():
            self.assertTrue(torch.equal(states[slot], states_before[slot] + 1))
        untouched = {0, 1, 2, 3, 4} - set(slots.tolist())
        for slot in untouched:
            self.assertTrue(torch.equal(states[slot], states_before[slot]))

    def test_non_fp32_beta_falls_back_to_triton(self):
        kernel, calls = self._make_kernel()
        x = self._inputs([2, 3])
        x["beta"] = x["beta"].bfloat16()
        states = torch.zeros(3, 1, 128, 128, dtype=torch.bfloat16)

        with self.assertRaisesRegex(
            AssertionError, "ordinary prefill unexpectedly fell back to Triton"
        ):
            kernel.extend(
                x["q"],
                x["k"],
                x["v"],
                x["g"],
                x["beta"],
                ssm_states=states,
                cache_indices=torch.tensor([1, 2], dtype=torch.int32),
                query_start_loc=x["query_start_loc"],
                extend_seq_lens_cpu=[2, 3],
                A_log=torch.zeros(128, dtype=torch.float32),
            )
        self.assertEqual(calls, [])

    def test_supports_only_datacenter_blackwell(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(10, 0)),
        ):
            self.assertTrue(NvidiaKDAKernel().supports_prefill)

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(12, 0)),
        ):
            self.assertFalse(NvidiaKDAKernel().supports_prefill)


if __name__ == "__main__":
    unittest.main()
