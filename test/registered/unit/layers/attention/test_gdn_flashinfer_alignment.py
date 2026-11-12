import unittest
from unittest import mock

import torch

from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
    FlashInferGDNKernel,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _view_with_pointer_mod(
    shape: tuple[int, ...], dtype: torch.dtype, pointer_mod: int
) -> torch.Tensor:
    numel = 1
    for dim in shape:
        numel *= dim
    element_size = dtype.itemsize
    base = torch.empty(numel + 32 // element_size, dtype=dtype)
    for offset in range(32 // element_size):
        view = base[offset : offset + numel]
        if view.data_ptr() % 32 == pointer_mod:
            return view.view(shape)
    raise AssertionError(f"Could not construct a pointer with mod32={pointer_mod}")


def _make_kernel_without_flashinfer() -> FlashInferGDNKernel:
    kernel = object.__new__(FlashInferGDNKernel)
    # Match the SM100 path used by the CPU-only fake prefill tests. Real
    # instances initialize this from the detected SM architecture in __init__.
    kernel._prefill_needs_fp32_state = False
    kernel._aligned_input_buffers = {}
    kernel._aligned_parameter_cache = {}
    kernel._verify_intermediate_buffers = {}
    kernel._alignment_fallback_warned = False
    return kernel


class TestFlashInferGDNAlignment(unittest.TestCase):
    def test_extend_writes_directly_to_preallocated_output(self):
        kernel = _make_kernel_without_flashinfer()
        kernel.use_state_pool = True
        captured = {}

        def fake_prefill(**kwargs):
            captured.update(kwargs)
            kwargs["output"].fill_(7.0)
            kwargs["output_state"].copy_(kwargs["initial_state"])
            return kwargs["output"], kwargs["output_state"]

        kernel._prefill_fn = fake_prefill
        q = torch.ones((1, 3, 1, 4), dtype=torch.bfloat16)
        k = torch.ones_like(q)
        v = torch.ones((1, 3, 2, 4), dtype=torch.bfloat16)
        g = torch.zeros((1, 3, 2), dtype=torch.bfloat16)
        beta = torch.ones_like(g)
        ssm_states = torch.zeros((3, 2, 4, 4), dtype=torch.bfloat16)
        physical_output = torch.empty((1, 5, 2, 4), dtype=v.dtype)
        preallocated_output = physical_output[:, :3]

        with mock.patch(
            "sglang.kernels.ops.attention.fla.l2norm.l2norm_fwd",
            side_effect=lambda tensor, eps: tensor,
        ):
            result, _, checkpoints = kernel.extend(
                q,
                k,
                v,
                g,
                beta,
                ssm_states=ssm_states,
                cache_indices=torch.tensor([1], dtype=torch.int32),
                query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
                output=preallocated_output,
            )

        self.assertEqual(captured["output"].data_ptr(), preallocated_output.data_ptr())
        self.assertEqual(result.data_ptr(), preallocated_output.data_ptr())
        torch.testing.assert_close(result, torch.full_like(result, 7.0))
        self.assertIsNone(checkpoints)

    def test_ratio8_bs1_split_view_reproduces_under_alignment(self):
        # In a BF16 [b_local(8)|a_local(8)] projection, a begins 16 bytes in;
        # contiguous() is a no-op at BS=1 and cannot meet FlashInfer's 32-byte ABI.
        projected_ba = torch.empty((2, 16), dtype=torch.bfloat16)
        _, a = projected_ba.split((8, 8), dim=-1)

        a_bs1 = a[:1]
        self.assertEqual(a_bs1.stride(), (16, 1))
        self.assertTrue(a_bs1.is_contiguous())
        self.assertEqual(a_bs1.data_ptr() % 32, 16)
        self.assertEqual(a_bs1.contiguous().data_ptr(), a_bs1.data_ptr())

        # BS>1 exposes the row gap, so contiguous() does allocate a rebased,
        # allocator-aligned tensor. This explains why only BS=1 failed.
        a_bs2 = a[:2]
        self.assertFalse(a_bs2.is_contiguous())
        repaired = a_bs2.contiguous()
        self.assertNotEqual(repaired.data_ptr(), a_bs2.data_ptr())
        self.assertEqual(repaired.data_ptr() % 32, 0)

    def test_dynamic_repair_buffer_is_reused_without_allocator_churn(self):
        kernel = _make_kernel_without_flashinfer()
        source = _view_with_pointer_mod((1, 1, 8), torch.bfloat16, 16)
        source.fill_(1)

        first = kernel._prepare_dynamic_input("decode_a", source)
        first_ptr = first.data_ptr()
        self.assertEqual(first_ptr % 32, 0)
        torch.testing.assert_close(first, source)
        self.assertEqual(len(kernel._aligned_input_buffers), 1)

        source.fill_(2)
        second = kernel._prepare_dynamic_input("decode_a", source)
        self.assertIs(second, first)
        self.assertEqual(second.data_ptr(), first_ptr)
        torch.testing.assert_close(second, source)
        self.assertEqual(len(kernel._aligned_input_buffers), 1)

        # Distinct kernel arguments cannot alias because both are live at the
        # FlashInfer call boundary.
        other = kernel._prepare_dynamic_input("decode_b", source)
        self.assertNotEqual(other.data_ptr(), first_ptr)
        self.assertEqual(len(kernel._aligned_input_buffers), 2)

    def test_decode_repairs_read_only_arguments_before_flashinfer(self):
        kernel = _make_kernel_without_flashinfer()
        kernel.use_state_pool = True
        captured = {}

        def fake_decode(**kwargs):
            captured.update(kwargs)
            v = kwargs["v"]
            return (
                torch.zeros(
                    v.shape[0],
                    1,
                    v.shape[2],
                    v.shape[3],
                    dtype=v.dtype,
                ),
                None,
            )

        kernel._decode_fn = fake_decode

        q = torch.empty(1, 1, 1, 128, dtype=torch.bfloat16)
        k = torch.empty_like(q)
        v = torch.empty(1, 1, 8, 128, dtype=torch.bfloat16)
        a = _view_with_pointer_mod((1, 1, 8), torch.bfloat16, 16)
        b = _view_with_pointer_mod((1, 1, 8), torch.bfloat16, 16)
        a.copy_(torch.arange(a.numel(), dtype=a.dtype).view_as(a))
        b.copy_(torch.arange(b.numel(), dtype=b.dtype).add_(8).view_as(b))
        A_log = _view_with_pointer_mod((8,), torch.float32, 4)
        dt_bias = _view_with_pointer_mod((8,), torch.bfloat16, 2)
        state = torch.zeros(2, 8, 128, 128, dtype=torch.bfloat16)
        cache_indices = _view_with_pointer_mod((1,), torch.int32, 4)

        result = kernel.decode(
            q,
            k,
            v,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=state,
            cache_indices=cache_indices,
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        )

        self.assertEqual(result.shape, (1, 1, 8, 128))
        for name in (
            "q",
            "k",
            "v",
            "A_log",
            "a",
            "dt_bias",
            "b",
            "initial_state",
            "initial_state_indices",
        ):
            with self.subTest(name=name):
                self.assertEqual(captured[name].data_ptr() % 32, 0)
        torch.testing.assert_close(captured["a"], a)
        torch.testing.assert_close(captured["b"], b)

    def test_gate_parameter_cache_preserves_backend_dtype_contract(self):
        kernel = _make_kernel_without_flashinfer()
        A_log = torch.empty(8, dtype=torch.bfloat16)
        dt_bias = torch.empty(8, dtype=torch.bfloat16)

        A_log_sm90, _ = kernel._prepare_gate_parameters(A_log, dt_bias)
        A_log_sm100, _ = kernel._prepare_gate_parameters(
            A_log, dt_bias, A_log_dtype=torch.float32
        )

        self.assertEqual(A_log_sm90.dtype, torch.bfloat16)
        self.assertEqual(A_log_sm100.dtype, torch.float32)
        self.assertEqual(A_log_sm90.data_ptr() % 32, 0)
        self.assertEqual(A_log_sm100.data_ptr() % 32, 0)
        self.assertIs(
            kernel._prepare_gate_parameters(A_log, dt_bias)[0],
            A_log_sm90,
        )

    def test_mutable_state_falls_back_without_losing_writeback(self):
        kernel = _make_kernel_without_flashinfer()
        captured = {}
        expected = torch.empty(1)

        class FakeFallback:
            def decode(self, *args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs
                return expected

        kernel._alignment_fallback_kernel = FakeFallback()
        state = _view_with_pointer_mod((2, 8, 4, 4), torch.bfloat16, 16)
        q = torch.empty(1, 1, 1, 4, dtype=torch.bfloat16)
        k = torch.empty_like(q)
        v = torch.empty(1, 1, 8, 4, dtype=torch.bfloat16)
        a = torch.empty(1, 1, 8, dtype=torch.bfloat16)
        b = torch.empty_like(a)
        cache_indices = torch.zeros(1, dtype=torch.int32)
        query_start_loc = torch.tensor([0, 1], dtype=torch.int32)

        result = kernel.decode(
            q,
            k,
            v,
            a,
            b,
            A_log=torch.zeros(8),
            dt_bias=torch.zeros(8, dtype=torch.bfloat16),
            ssm_states=state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )

        self.assertIs(result, expected)
        self.assertIs(captured["kwargs"]["ssm_states"], state)
        self.assertEqual(len(kernel._aligned_input_buffers), 0)

    def test_mutable_mtp_workspace_falls_back_without_copying(self):
        kernel = _make_kernel_without_flashinfer()
        kernel.use_state_pool = True
        captured = {}
        expected = torch.empty(1)

        class FakeFallback:
            def target_verify(self, **kwargs):
                captured.update(kwargs)
                return expected

        kernel._alignment_fallback_kernel = FakeFallback()
        q = torch.empty(1, 2, 1, 4, dtype=torch.bfloat16)
        k = torch.empty_like(q)
        v = torch.empty(1, 2, 8, 4, dtype=torch.bfloat16)
        a = torch.empty(1, 2, 8, dtype=torch.bfloat16)
        b = torch.empty_like(a)
        state = torch.empty(2, 8, 4, 4, dtype=torch.bfloat16)
        workspace = _view_with_pointer_mod((2, 2, 8, 4, 4), torch.bfloat16, 16)

        result = kernel.target_verify(
            torch.zeros(8),
            torch.zeros(8, dtype=torch.bfloat16),
            q,
            k,
            v,
            a,
            b,
            ssm_states=state,
            cache_indices=torch.zeros(1, dtype=torch.int32),
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            intermediate_states_buffer=workspace,
            intermediate_state_indices=torch.zeros(1, 2, dtype=torch.int32),
            cache_steps=2,
            retrieve_parent_token=None,
        )

        self.assertIs(result, expected)
        self.assertIs(captured["intermediate_states_buffer"], workspace)
        self.assertEqual(len(kernel._aligned_input_buffers), 0)

    def test_mtp_padded_capture_uses_stable_exact_batch_workspace_and_copies_back(self):
        kernel = _make_kernel_without_flashinfer()
        kernel.use_state_pool = True
        captured_ptrs = []

        def fake_mtp(**kwargs):
            workspace = kwargs["intermediate_states_buffer"]
            captured_ptrs.append(workspace.data_ptr())
            self.assertEqual(workspace.shape[0], 8)
            for row in range(workspace.shape[0]):
                workspace[row].fill_(row + 1)
            return torch.zeros_like(kwargs["v"]), None

        kernel._mtp_fn = fake_mtp
        workspace = torch.zeros((7, 2, 8, 4, 4), dtype=torch.bfloat16)

        def run_once():
            return kernel.target_verify(
                torch.zeros(8),
                torch.zeros(8, dtype=torch.bfloat16),
                torch.empty(1, 16, 1, 4, dtype=torch.bfloat16),
                torch.empty(1, 16, 1, 4, dtype=torch.bfloat16),
                torch.empty(1, 16, 8, 4, dtype=torch.bfloat16),
                torch.empty(1, 16, 8, dtype=torch.bfloat16),
                torch.empty(1, 16, 8, dtype=torch.bfloat16),
                ssm_states=torch.zeros(8, 8, 4, 4, dtype=torch.bfloat16),
                cache_indices=torch.zeros(8, dtype=torch.int32),
                query_start_loc=torch.arange(0, 18, 2, dtype=torch.int32),
                intermediate_states_buffer=workspace,
                intermediate_state_indices=torch.arange(8, dtype=torch.int32),
                cache_steps=2,
                retrieve_parent_token=None,
            )

        self.assertEqual(run_once().shape, (1, 16, 8, 4))
        for row in range(workspace.shape[0]):
            torch.testing.assert_close(
                workspace[row], torch.full_like(workspace[row], row + 1)
            )
        self.assertEqual(len(kernel._verify_intermediate_buffers), 1)

        workspace.zero_()
        run_once()
        self.assertEqual(captured_ptrs[0], captured_ptrs[1])
        self.assertEqual(len(kernel._verify_intermediate_buffers), 1)

    def test_mtp_pool_sized_batch_keeps_zero_copy_fast_path(self):
        kernel = _make_kernel_without_flashinfer()
        kernel.use_state_pool = True
        workspace = torch.zeros((7, 2, 8, 4, 4), dtype=torch.bfloat16)
        captured = {}

        def fake_mtp(**kwargs):
            captured.update(kwargs)
            return torch.zeros_like(kwargs["v"]), None

        kernel._mtp_fn = fake_mtp
        result = kernel.target_verify(
            torch.zeros(8),
            torch.zeros(8, dtype=torch.bfloat16),
            torch.empty(1, 4, 1, 4, dtype=torch.bfloat16),
            torch.empty(1, 4, 1, 4, dtype=torch.bfloat16),
            torch.empty(1, 4, 8, 4, dtype=torch.bfloat16),
            torch.empty(1, 4, 8, dtype=torch.bfloat16),
            torch.empty(1, 4, 8, dtype=torch.bfloat16),
            ssm_states=torch.zeros(7, 8, 4, 4, dtype=torch.bfloat16),
            cache_indices=torch.zeros(2, dtype=torch.int32),
            query_start_loc=torch.arange(0, 6, 2, dtype=torch.int32),
            intermediate_states_buffer=workspace,
            intermediate_state_indices=torch.arange(7, dtype=torch.int32),
            cache_steps=2,
            retrieve_parent_token=None,
        )

        self.assertEqual(result.shape, (1, 4, 8, 4))
        self.assertEqual(
            captured["intermediate_states_buffer"].data_ptr(), workspace.data_ptr()
        )
        self.assertEqual(len(kernel._verify_intermediate_buffers), 0)

    def test_mtp_padded_workspace_is_reused_across_sequential_layer_pools(self):
        kernel = _make_kernel_without_flashinfer()
        kernel.use_state_pool = True
        captured_ptrs = []
        call_value = 0

        def fake_mtp(**kwargs):
            nonlocal call_value
            call_value += 1
            scratch = kwargs["intermediate_states_buffer"]
            captured_ptrs.append(scratch.data_ptr())
            scratch.fill_(call_value)
            return torch.zeros_like(kwargs["v"]), None

        kernel._mtp_fn = fake_mtp

        def run(pool):
            kernel.target_verify(
                torch.zeros(8),
                torch.zeros(8, dtype=torch.bfloat16),
                torch.empty(1, 16, 1, 4, dtype=torch.bfloat16),
                torch.empty(1, 16, 1, 4, dtype=torch.bfloat16),
                torch.empty(1, 16, 8, 4, dtype=torch.bfloat16),
                torch.empty(1, 16, 8, dtype=torch.bfloat16),
                torch.empty(1, 16, 8, dtype=torch.bfloat16),
                ssm_states=torch.zeros(8, 8, 4, 4, dtype=torch.bfloat16),
                cache_indices=torch.zeros(8, dtype=torch.int32),
                query_start_loc=torch.arange(0, 18, 2, dtype=torch.int32),
                intermediate_states_buffer=pool,
                intermediate_state_indices=torch.arange(8, dtype=torch.int32),
                cache_steps=2,
                retrieve_parent_token=None,
            )

        first_pool = torch.zeros((7, 2, 8, 4, 4), dtype=torch.bfloat16)
        second_pool = torch.zeros_like(first_pool)
        run(first_pool)
        run(second_pool)

        self.assertEqual(captured_ptrs[0], captured_ptrs[1])
        torch.testing.assert_close(first_pool, torch.ones_like(first_pool))
        torch.testing.assert_close(second_pool, torch.full_like(second_pool, 2))


if __name__ == "__main__":
    unittest.main()
