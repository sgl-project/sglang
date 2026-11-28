"""CPU unit tests for MIS padded-tail safety under breakable prefill graphs."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.layers.logits_processor import (  # noqa: E402
    LogitsMetadata,
    LogitsProcessor,
)
from sglang.srt.layers.pooler import pool_at_delimiter_positions  # noqa: E402
from sglang.srt.model_executor.forward_mode import (  # noqa: E402
    CaptureHiddenMode,
    ForwardMode,
)

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class TestMISPaddedHiddenStatesSafety(CustomTestCase):
    """Empty-query MIS puts first_delim == 0; index -1 must never read padded rows."""

    def test_pool_at_delimiter_positions_clamps_empty_query_index(self):
        hidden = _make_hidden_with_nan_tail()
        fb = SimpleNamespace(
            extend_seq_lens_cpu=[6],
            multi_item_delimiter_indices=[torch.tensor([0, 3], dtype=torch.int64)],
        )

        gathered = pool_at_delimiter_positions(
            data=hidden, forward_batch=fb, device=torch.device("cpu")
        )

        self.assertEqual(len(gathered), 1)
        self.assertEqual(tuple(gathered[0].shape), (2, 4))
        self.assertFalse(torch.isnan(gathered[0]).any())
        torch.testing.assert_close(gathered[0][0], hidden[0], rtol=0, atol=0)
        torch.testing.assert_close(gathered[0][1], hidden[2], rtol=0, atol=0)

    def test_logits_processor_mis_clamps_empty_query_index(self):
        hidden = _make_hidden_with_nan_tail()
        vocab_size = 5
        weight = torch.linspace(
            -0.5, 0.5, steps=vocab_size * 4, dtype=torch.float32
        ).reshape(vocab_size, 4)
        delimiters = [torch.tensor([0, 3], dtype=torch.int64)]
        label_ids = [1, 3]
        input_ids = torch.zeros(6, dtype=torch.int64)

        for extend_seq_lens_cpu in ([6], None):
            with (
                self.subTest(extend_seq_lens_cpu=extend_seq_lens_cpu),
                published_topology(),
            ):
                processor = LogitsProcessor(SimpleNamespace(vocab_size=vocab_size))
                metadata = LogitsMetadata(
                    forward_mode=ForwardMode.EXTEND,
                    capture_hidden_mode=CaptureHiddenMode.NULL,
                    extend_seq_lens=(
                        torch.tensor([6]) if extend_seq_lens_cpu is not None else None
                    ),
                    extend_seq_lens_cpu=extend_seq_lens_cpu,
                    extend_return_logprob=True,
                    extend_token_ids_logprob=True,
                    token_ids_logprobs=[label_ids],
                    is_prefill_only=True,
                )
                with patch.object(
                    processor,
                    "_get_logits",
                    side_effect=lambda states, *args, **kwargs: states @ weight.T,
                ):
                    out = processor.compute_logprobs_for_multi_item_scoring(
                        input_ids=input_ids,
                        hidden_states=hidden,
                        lm_head=None,
                        logits_metadata=metadata,
                        multi_item_delimiter_indices=delimiters,
                    )

                self.assertEqual(len(out.input_token_ids_logprobs_val), 1)
                vals = out.input_token_ids_logprobs_val[0]
                self.assertEqual(tuple(vals.shape), (2, 2))
                self.assertTrue(torch.isfinite(vals).all())
                expected_item_logits = hidden[[0, 2]] @ weight.T
                expected_logprobs = torch.nn.functional.log_softmax(
                    expected_item_logits, dim=-1
                )[:, label_ids]
                torch.testing.assert_close(
                    vals, expected_logprobs, rtol=1e-5, atol=1e-5
                )


def _make_hidden_with_nan_tail():
    raw_num_tokens = 6
    static_num_tokens = 8
    hidden_dim = 4
    hidden = torch.arange(static_num_tokens * hidden_dim, dtype=torch.float32).reshape(
        static_num_tokens, hidden_dim
    )
    # Simulate uninitialized CUDA graph mempool tail past raw_num_tokens.
    hidden[raw_num_tokens:] = float("nan")
    return hidden


if __name__ == "__main__":
    unittest.main()
