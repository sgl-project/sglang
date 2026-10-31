"""fa3 on a model whose widest head exceeds FA3's 256 limit (Gemma-4: 256-wide
sliding-window layers, 512-wide global layers).

The backend initializes as FA3 and loads FA4 for the wide layers. Every case
here runs through that initialized backend, not a hand-wired kernel table, so
dropping the FA4 load, the per-layer routing, or the image mask plumbing fails
against the HF-style torch reference.
"""

import unittest

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.dense_attention import (
    DENSE_ATOL,
    DENSE_RTOL,
    DenseAttentionCase,
    build_dense_attention_fixture,
    dense_attention_reference_with_custom_mask,
    expected_dense_fixture_output,
    make_dense_cases,
    run_dense_attention_case,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

# Gemma-4 head widths: global layers run FA4, sliding-window layers stay on FA3.
WIDE_HEAD_DIM = 512
NARROW_HEAD_DIM = 256


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
@unittest.skipIf(
    get_device_sm() // 10 != 9,
    "fa3 promotes wide heads to FA4 only on SM90; SM8x is rejected at init",
)
class TestFa3Fa4HybridDenseAttention(CustomTestCase):
    HIDDEN_SIZE = 512
    # The hybrid rejects non-BF16 KV caches (FA4 has no FP16 path on SM90).
    DTYPE = torch.bfloat16
    CASES = make_dense_cases("fa3")

    def _fixture(self, case, head_dim):
        return build_dense_attention_fixture(
            self,
            case,
            head_dim=head_dim,
            model_head_dim=WIDE_HEAD_DIM,
            hidden_size=self.HIDDEN_SIZE,
            dtype=self.DTYPE,
        )

    def test_initialized_backend_routes_by_layer_head_dim(self):
        case = self.CASES[0]
        wide = self._fixture(case, WIDE_HEAD_DIM)
        backend = wide.backend
        self.assertTrue(backend.fa3_upgrades_to_fa4)
        self.assertEqual(set(backend._fa_kernels), {3, 4})
        self.assertIsNot(backend._fa_kernels[4][0], backend._fa_kernels[3][0])
        self.assertEqual(backend._ver_for(wide.actual_module.attn), 4)

        narrow = self._fixture(case, NARROW_HEAD_DIM)
        layer = narrow.actual_module.attn
        self.assertEqual(narrow.backend._ver_for(layer), 3)
        # Image masks exist only in FA4: an image batch sends every layer there.
        self.assertEqual(narrow.backend._ver_for(layer, needs_fa4=True), 4)

    def test_wide_layers_match_reference_on_fa4(self):
        for case in self.CASES:
            with self.subTest(case=case.name):
                run_dense_attention_case(
                    self,
                    case,
                    head_dim=WIDE_HEAD_DIM,
                    model_head_dim=WIDE_HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                    dtype=self.DTYPE,
                )

    def test_image_token_ranges_make_image_spans_bidirectional(self):
        """What gemma4_mm.prepare_attn_masks installs on an image prefill:
        per-query absolute image spans, -1 on text rows. Image rows see their
        whole span, text rows stay causal, for wide and narrow layers alike."""
        case = DenseAttentionCase(
            name="image_extend",
            backend="fa3",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            num_kv_heads=2,
            page_size=16,
            prefix_lens=(0, 4),
            extend_lens=(10, 6),
        )
        # Absolute (begin, end) per request, inclusive, inside its extend window.
        spans = [(2, 6), (5, 8)]

        ranges = torch.full((case.num_input_tokens, 2), -1, dtype=torch.int32)
        masks = []
        offset = 0
        for req, (prefix, length, (begin, end)) in enumerate(
            zip(case.prefix_lens, case.extend_lens, spans)
        ):
            ranges[offset + begin - prefix : offset + end + 1 - prefix] = torch.tensor(
                [begin, end], dtype=torch.int32
            )
            offset += length
            mask = torch.ones(length, prefix + length, dtype=torch.bool).tril(
                diagonal=prefix
            )
            mask[begin - prefix : end + 1 - prefix, begin : end + 1] = True
            masks.append(mask)

        for head_dim in (WIDE_HEAD_DIM, NARROW_HEAD_DIM):
            with self.subTest(head_dim=head_dim):
                fixture = self._fixture(case, head_dim)
                backend = fixture.backend
                with (
                    torch.no_grad(),
                    forward_context(ForwardContext(attn_backend=backend)),
                ):
                    backend.init_forward_metadata(fixture.forward_batch)
                    backend.forward_metadata.image_token_ranges = ranges.to(
                        fixture.input_hidden.device
                    )
                    actual = fixture.actual_module(
                        fixture.input_hidden, fixture.forward_batch
                    )

                expected = dense_attention_reference_with_custom_mask(
                    fixture.reference_module,
                    case,
                    fixture.prefix_hidden,
                    fixture.input_hidden,
                    masks,
                )
                torch.testing.assert_close(
                    actual, expected, atol=DENSE_ATOL, rtol=DENSE_RTOL
                )
                # The spans change the answer: a causal result would be wrong.
                causal = expected_dense_fixture_output(fixture)
                self.assertFalse(
                    torch.allclose(actual, causal, atol=DENSE_ATOL, rtol=DENSE_RTOL)
                )


if __name__ == "__main__":
    unittest.main()
