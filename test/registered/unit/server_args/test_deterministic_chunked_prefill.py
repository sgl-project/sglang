"""Unit tests for deterministic inference chunked-prefill alignment validation (#36344).

Validates that ``check_server_args`` rejects configs where
``chunked_prefill_size`` is smaller than the prefill truncation alignment
when ``--enable-deterministic-inference`` is set, which causes indefinite
request retries and server hang.
"""
import unittest

from sglang.srt.server_args import ServerArgs


class TestDeterministicChunkedPrefillValidation(unittest.TestCase):
    """Verify the chunked_prefill_size >= alignment invariant."""

    BASE_KWARGS = dict(
        model_path="dummy",
        served_model_name="dummy",
        page_size=1,
    )

    def test_chunked_lt_align_rejected_triton(self):
        """chunked_prefill_size < alignment + deterministic + triton -> reject."""
        args = ServerArgs(
            enable_deterministic_inference=True,
            chunked_prefill_size=128,
            attention_backend="triton",
            **self.BASE_KWARGS,
        )
        with self.assertRaises(AssertionError) as ctx:
            args.check_server_args()
        self.assertIn("alignment", str(ctx.exception))

    def test_chunked_lt_align_rejected_flashinfer(self):
        """chunked_prefill_size < alignment + deterministic + flashinfer -> reject."""
        args = ServerArgs(
            enable_deterministic_inference=True,
            chunked_prefill_size=128,
            attention_backend="flashinfer",
            **self.BASE_KWARGS,
        )
        with self.assertRaises(AssertionError) as ctx:
            args.check_server_args()
        self.assertIn("alignment", str(ctx.exception))

    def test_chunked_eq_align_accepted(self):
        """chunked_prefill_size == alignment must pass."""
        args = ServerArgs(
            enable_deterministic_inference=True,
            chunked_prefill_size=4096,
            attention_backend="triton",
            **self.BASE_KWARGS,
        )
        args.check_server_args()

    def test_chunked_gt_align_accepted(self):
        """chunked_prefill_size > alignment must pass."""
        args = ServerArgs(
            enable_deterministic_inference=True,
            chunked_prefill_size=8192,
            attention_backend="triton",
            **self.BASE_KWARGS,
        )
        args.check_server_args()

    def test_no_deterministic_unaffected(self):
        """Without deterministic inference, small chunked_prefill_size is fine."""
        args = ServerArgs(
            chunked_prefill_size=128,
            attention_backend="triton",
            **self.BASE_KWARGS,
        )
        args.check_server_args()


if __name__ == "__main__":
    unittest.main()
