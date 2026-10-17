import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.layers.attention.linear.kda_backend import KDAKernelDispatcher
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestKDAPrefillDispatcher(CustomTestCase):
    def _dispatcher(self, backend=LinearAttnKernelBackend.FLASHINFER):
        # Avoid GPU adapter construction: these tests only check host dispatch.
        dispatcher = KDAKernelDispatcher.__new__(KDAKernelDispatcher)
        dispatcher.prefill_backend = backend
        dispatcher.extend_kernel = MagicMock(supports_safe_gate=True)
        dispatcher.triton_kernel = MagicMock()
        return dispatcher

    def test_flashinfer_packed_token_threshold(self):
        for lengths, fallback in (
            ((1,), True),
            ((1, 1), True),
            ((1, 1, 1, 1), True),
            ((2, 0), True),
            ((0, 2), True),
            ((2,), False),
            ((1, 2), False),
            ((2, 2), False),
            ((130, 128), False),
        ):
            for lower_bound in (None, -5.0):
                with self.subTest(lengths=lengths, lower_bound=lower_bound):
                    dispatcher = self._dispatcher()
                    selected = dispatcher.effective_extend_kernel(
                        lower_bound, sum(lengths), len(lengths)
                    )
                    self.assertIs(
                        selected,
                        dispatcher.triton_kernel
                        if fallback
                        else dispatcher.extend_kernel,
                    )

    def test_extend_uses_sequence_count_without_reading_offsets(self):
        dispatcher = self._dispatcher()
        q = torch.empty(1, 2, 2, 128)
        offsets = torch.tensor([0, 1, 2], dtype=torch.int32)
        state, indices = torch.empty(2, 2, 128, 128), torch.arange(2)
        result = dispatcher.extend(
            q,
            q,
            q,
            q,
            q[..., 0],
            ssm_states=state,
            cache_indices=indices,
            query_start_loc=offsets,
            lower_bound=-5.0,
            return_intermediate_states=True,
        )
        self.assertIs(result, dispatcher.triton_kernel.extend.return_value)
        dispatcher.extend_kernel.extend.assert_not_called()
        args = dispatcher.triton_kernel.extend.call_args
        self.assertIs(args.kwargs["query_start_loc"], offsets)
        self.assertIs(args.kwargs["ssm_states"], state)
        self.assertTrue(args.kwargs["return_intermediate_states"])

    def test_other_prefill_backends_keep_the_configured_kernel(self):
        dispatcher = self._dispatcher(LinearAttnKernelBackend.HELION)
        self.assertIs(
            dispatcher.effective_extend_kernel(-5.0, 2, 2), dispatcher.extend_kernel
        )

    def test_safe_gate_fallback_is_preserved(self):
        dispatcher = self._dispatcher(LinearAttnKernelBackend.HELION)
        dispatcher.extend_kernel.supports_safe_gate = False
        self.assertIs(
            dispatcher.effective_extend_kernel(-5.0, 3, 2), dispatcher.triton_kernel
        )
        self.assertIs(
            dispatcher.effective_extend_kernel(None, 3, 2), dispatcher.extend_kernel
        )


if __name__ == "__main__":
    unittest.main()
