import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import graph_utils as radix_attention
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RecordingBackend:
    """Stand-in that checks the rows it is given against the batch's cache
    locations and returns one output row per query row."""

    def __init__(self):
        self.calls = []

    def forward(self, q, k, v, layer, forward_batch, save_kv_cache, **kwargs):
        assert forward_batch.out_cache_loc.shape[0] == q.shape[0]
        self.calls.append(
            dict(
                q=q.shape[0],
                k=None if k is None else k.shape[0],
                positions=forward_batch.positions.shape[0],
                **{name: _rows(value) for name, value in kwargs.items()},
            )
        )
        return torch.ones(q.shape[0], 6), torch.ones(q.shape[0], 2)


class _OutputBackend:
    """Stand-in that returns one output, written into the batch's
    _attn_output when in_place is set."""

    def __init__(self, in_place):
        self.in_place = in_place
        self.given = None

    def forward(self, q, k, v, layer, forward_batch, save_kv_cache, **kwargs):
        self.given = forward_batch._attn_output
        if self.in_place:
            return self.given.fill_(3.0)
        return torch.full((q.shape[0], q.shape[1]), 3.0)


class _IndexerBackend:
    """Stand-in with a sparse indexer's contract: one index output row per
    idx_q row, shaped by the query rows, as with index values enabled."""

    def forward(
        self, q, k, v, layer, forward_batch, save_kv_cache, idx_q, idx_k, idx_v
    ):
        assert idx_q.shape[0] == q.shape[0]
        assert idx_k.shape[0] == idx_v.shape[0] == k.shape[0]
        idx_out = torch.ones(idx_q.shape[0] * idx_q.shape[1] * idx_q.shape[2])
        return idx_out.reshape(q.shape[0], -1), torch.ones(q.shape[0], 2)


def _rows(value):
    if isinstance(value, list):
        return [t.shape[0] for t in value]
    return value.shape[0]


def _batch(mode, real, rows=8):
    return SimpleNamespace(
        forward_mode=mode,
        global_num_token_non_padded_cpu=real,
        out_cache_loc=torch.arange(rows),
        positions=torch.arange(rows),
        mha_return_lse=False,
        _attn_output=None,
    )


_LAYER = SimpleNamespace(
    qk_head_dim=2, v_head_dim=2, tp_q_head_num=1, is_cross_attention=False
)


class TestPaddedExtendAttention(CustomTestCase):
    """MLP sync pads an extend batch's rows to a multiple of attention TP while
    its attention metadata covers only the real rows."""

    def test_only_padded_extend_batches_are_narrowed(self):
        q = torch.zeros(8, 2)
        cases = [
            (ForwardMode.EXTEND, 5, 5),
            (ForwardMode.EXTEND, 8, None),  # nothing padded
            (ForwardMode.EXTEND, 0, None),  # an idle rank's batch keeps its path
            (ForwardMode.TARGET_VERIFY, 5, None),
            (ForwardMode.DECODE, 5, None),
        ]
        for mode, real, expected in cases:
            with self.subTest(mode=mode, real=real):
                self.assertEqual(
                    radix_attention.padded_extend_real_tokens(q, _batch(mode, real)),
                    expected,
                )

    def test_the_backend_sees_the_real_rows_and_the_tail_is_zero(self):
        backend = _RecordingBackend()
        batch = _batch(ForwardMode.EXTEND, 5)
        out_cache_loc, positions = batch.out_cache_loc, batch.positions
        with patch.object(radix_attention, "get_attn_backend", return_value=backend):
            output, lse = radix_attention._attention_on_real_rows(
                _LAYER,
                5,
                torch.zeros(8, 2),
                torch.zeros(8, 2),
                torch.zeros(8, 2),
                batch,
                True,
                q_rope=torch.zeros(8, 1),
                k_rope=torch.zeros(8, 1),
            )
        self.assertEqual(
            backend.calls, [dict(q=5, k=5, positions=5, q_rope=5, k_rope=5)]
        )
        # The batch's own tensors come back for the rest of the layer.
        self.assertIs(batch.out_cache_loc, out_cache_loc)
        self.assertIs(batch.positions, positions)
        self.assertEqual(tuple(output.shape), (8, 6))
        torch.testing.assert_close(output[5:], torch.zeros(3, 6))
        torch.testing.assert_close(output[:5], torch.ones(5, 6))
        self.assertEqual(tuple(lse.shape), (8, 2))

    def test_keys_with_their_own_extent_or_row_count_are_left_whole(self):
        # K and V holding a cached prefix, or a cross-attention's encoder rows,
        # that happen to fill the padding (8 rows) stay whole by their declared
        # extent or by the layer.
        cross = SimpleNamespace(**{**vars(_LAYER), "is_cross_attention": True})
        for layer, key_value_num_tokens, key_rows in (
            (_LAYER, 12, 12),
            (_LAYER, None, 12),
            (_LAYER, 8, 8),
            (cross, None, 8),
        ):
            with self.subTest(
                cross_attention=layer.is_cross_attention,
                key_value_num_tokens=key_value_num_tokens,
            ):
                backend = _RecordingBackend()
                with patch.object(
                    radix_attention, "get_attn_backend", return_value=backend
                ):
                    radix_attention._attention_on_real_rows(
                        layer,
                        5,
                        torch.zeros(8, 2),
                        torch.zeros(key_rows, 2),
                        torch.zeros(key_rows, 2),
                        _batch(ForwardMode.EXTEND, 5),
                        False,
                        key_value_num_tokens=key_value_num_tokens,
                    )
                self.assertEqual(backend.calls[0]["k"], key_rows)

    def test_per_row_inputs_follow_the_rows_they_belong_to(self):
        per_row = dict(
            rel_bias=torch.zeros(8, 2, 4),
            aux_tensors=[torch.zeros(8, 3)],
            q_descale=torch.zeros(8, 1),
            mxfp8_norm_rope_positions=torch.zeros(8),
        )
        for key_value_num_tokens, key_rows in ((None, 5), (12, 12)):
            with self.subTest(key_value_num_tokens=key_value_num_tokens):
                backend = _RecordingBackend()
                with patch.object(
                    radix_attention, "get_attn_backend", return_value=backend
                ):
                    radix_attention._attention_on_real_rows(
                        _LAYER,
                        5,
                        torch.zeros(8, 2),
                        torch.zeros(8 if key_rows == 5 else key_rows, 2),
                        torch.zeros(8 if key_rows == 5 else key_rows, 2),
                        _batch(ForwardMode.EXTEND, 5),
                        False,
                        key_value_num_tokens=key_value_num_tokens,
                        k_descale=torch.zeros(8 if key_rows == 5 else key_rows, 1),
                        v_descale=torch.zeros(8 if key_rows == 5 else key_rows, 1),
                        **per_row,
                    )
                call = backend.calls[0]
                self.assertEqual(
                    {name: call[name] for name in per_row},
                    dict(
                        rel_bias=5,
                        aux_tensors=[5],
                        q_descale=5,
                        mxfp8_norm_rope_positions=5,
                    ),
                )
                self.assertEqual(
                    (call["k_descale"], call["v_descale"]), (key_rows,) * 2
                )

    def test_a_sparse_indexer_takes_the_rows_it_indexes(self):
        with patch.object(
            radix_attention, "get_attn_backend", return_value=_IndexerBackend()
        ):
            idx_out, out = radix_attention._attention_on_real_rows(
                _LAYER,
                5,
                torch.zeros(6, 2),
                torch.zeros(6, 2),
                torch.zeros(6, 2),
                _batch(ForwardMode.EXTEND, 5, rows=6),
                False,
                idx_q=torch.zeros(6, 1, 128),
                idx_k=torch.zeros(6, 128),
                idx_v=torch.zeros(6, 128),
            )
        self.assertEqual(idx_out.shape, (6, 128))
        self.assertEqual(out.shape, (6, 2))
        self.assertEqual(idx_out[5:].count_nonzero().item(), 0)
        self.assertEqual(idx_out[:5].count_nonzero().item(), 5 * 128)

    def test_the_output_is_written_in_place_or_copied(self):
        for in_place in (True, False):
            with self.subTest(in_place=in_place):
                backend = _OutputBackend(in_place)
                batch = _batch(ForwardMode.EXTEND, 5)
                with patch.object(
                    radix_attention, "get_attn_backend", return_value=backend
                ):
                    output = radix_attention._attention_on_real_rows(
                        _LAYER,
                        5,
                        torch.zeros(8, 2),
                        torch.zeros(8, 2),
                        torch.zeros(8, 2),
                        batch,
                        False,
                    )
                # The backend was handed the real rows of the returned buffer.
                self.assertEqual(backend.given.data_ptr(), output.data_ptr())
                self.assertEqual(tuple(backend.given.shape), (5, 2))
                self.assertIsNone(batch._attn_output)
                torch.testing.assert_close(output[:5], torch.full((5, 2), 3.0))
                torch.testing.assert_close(output[5:], torch.zeros(3, 2))


if __name__ == "__main__":
    unittest.main()
