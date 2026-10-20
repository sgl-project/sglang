import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.layer_boundary.residual.add_norm import (
    NORM_READOUT,
    Fp8Input,
    NormQuantReadout,
    NormReadout,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _PlainAdd:
    """The update object the boundary passes when the producer adds plainly."""

    is_plain_add = True


class TestEmptyBatchResidual(CustomTestCase):
    """NormQuantReadout is not a drop-in replacement for NormReadout.

    They diverge on the empty-batch path: NormReadout keeps the residual it was
    given, NormQuantReadout returns hidden_states as the residual. Installing
    the quantized readout unconditionally therefore changes behaviour for every
    backend, not only the one the fused kernel runs on.
    """

    def _read(self, readout, hidden_states, residual):
        return readout.update_and_read(
            _PlainAdd(), hidden_states, residual, norm=lambda *a: None
        )

    def test_norm_readout_keeps_the_residual_on_an_empty_batch(self):
        hidden_states = torch.empty(0, 8)
        residual = torch.empty(0, 8)
        _, out_residual = self._read(NORM_READOUT, hidden_states, residual)
        self.assertIs(out_residual, residual)

    def test_quant_readout_returns_hidden_states_on_an_empty_batch(self):
        hidden_states = torch.empty(0, 8)
        residual = torch.empty(0, 8)
        _, out_residual = self._read(NormQuantReadout(), hidden_states, residual)
        # Not a bug in NormQuantReadout; it is simply a different contract.
        # This asserts the divergence the model must not expose by accident.
        self.assertIs(out_residual, hidden_states)
        self.assertIsNot(out_residual, residual)


class TestQwen3NextFfnRead(CustomTestCase):
    """_ffn_read only installs the quantized readout when it is consumed."""

    def _ffn_read(self):
        from sglang.srt.models.qwen3_next import _ffn_read

        return _ffn_read

    def test_dense_layer_keeps_the_plain_readout(self):
        # A non-sparse layer has no MoE block to consume an fp8 tuple.
        self.assertIs(self._ffn_read()(object(), False), NORM_READOUT)

    def test_unconsumable_mlp_keeps_the_plain_readout(self):
        # Sparse, but the MLP is not the subclass that unpacks the tuple.
        read = self._ffn_read()(object(), True)
        self.assertIs(read, NORM_READOUT)
        self.assertIsInstance(read, NormReadout)

    def test_declared_read_carries_both_bf16_and_fp8(self):
        """When it is installed, the GDN consumer needs the bf16 side-output."""
        read = NormQuantReadout(fp8_input=Fp8Input.TUPLE_AND_BF16)
        self.assertIsInstance(read, NormQuantReadout)
        self.assertIs(read.fp8_input, Fp8Input.TUPLE_AND_BF16)


class TestNonAiterBackendsKeepThePlainPath(CustomTestCase):
    """No backend but aiter may see a changed declaration.

    The read declarations key off the projection's quant method, which is true
    on any backend with a block-quant FP8 projection, CUDA included. They must
    additionally be gated on _use_aiter, or a CUDA run of Qwen3-Next would get
    a tuple where it used to get a tensor.
    """

    def test_ffn_read_is_plain_without_aiter(self):
        import sglang.srt.models.qwen3_next as qn

        with patch.object(qn, "_use_aiter", False):
            self.assertIs(qn._ffn_read(object(), True), NORM_READOUT)

    def test_attention_declaration_is_gated_on_use_aiter(self):
        """The gate is `_use_aiter and _linear_accepts_fp8_tuple(...)`."""
        import inspect

        import sglang.srt.models.qwen3_next as qn

        src = inspect.getsource(qn)
        for line in src.splitlines():
            if "accepts_fp8_input = " in line and "_linear_accepts_fp8_tuple" in line:
                self.assertIn(
                    "_use_aiter and",
                    line,
                    "declare_attn's fp8_input must be gated on _use_aiter, "
                    "otherwise CUDA installs a quantized read",
                )

    def test_select_passes_tensors_through_untouched(self):
        """A non-tuple input is returned as-is, whatever the linear is."""
        import sglang.srt.models.qwen3_next as qn

        x = torch.zeros(4, 8)
        self.assertIs(qn._select_fused_ar_input_for_linear(x, object()), x)


if __name__ == "__main__":
    unittest.main()
