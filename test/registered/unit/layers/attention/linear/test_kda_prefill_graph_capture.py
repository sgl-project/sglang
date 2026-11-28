"""KDA varlen prefill under torch.compile with a data-dependent token count.

Graph capture cannot read host token counts (``extend_seq_lens_cpu``,
``logical_num_tokens``) without specializing on them, and cannot trace host
syncs with a data-dependent length. Each case compiles one helper with
``fullgraph=True`` and checks it against the eager result.
"""

import unittest

import torch

from sglang.kernels.ops.attention.fla.index import prepare_chunk_indices
from sglang.kernels.ops.attention.fla.kda import _compiled_chunk_index_capacity
from sglang.srt.layers.attention.linear.kda_backend import (
    _compiled_conv_block_table,
    _mask_kda_padding_rows,
    _restore_kda_token_padding,
    _trim_kda_token_padding,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _output_nodes(gm: torch.fx.GraphModule) -> list:
    (output,) = [node for node in gm.graph.nodes if node.op == "output"]
    return list(output.args[0])


class TestKDAChunkIndicesUnderCompile(CustomTestCase):
    def setUp(self):
        torch._dynamo.reset()

    def test_chunk_indices_accept_data_dependent_sequence_lengths(self):
        """The table was built with a loop over ``.tolist()`` of the chunk
        counts, a host sync with a data-dependent length. Under capture it is
        padded to a fixed capacity; padded rows point past the last sequence."""

        def prepare_bounded(
            cu_seqlens: torch.Tensor, token_storage: torch.Tensor
        ) -> torch.Tensor:
            return prepare_chunk_indices.__wrapped__(
                cu_seqlens,
                64,
                _compiled_chunk_index_capacity(token_storage.shape[0], cu_seqlens, 64),
            )

        compiled_prepare = torch.compile(
            prepare_bounded, backend="eager", dynamic=True, fullgraph=True
        )
        for dtype in (torch.int32, torch.int64):
            for cu_seqlens, expected in (
                ([0, 130, 194], [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]),
                (
                    [0, 64, 65, 193],
                    [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4]],
                ),
            ):
                with self.subTest(dtype=dtype, cu_seqlens=cu_seqlens):
                    with torch._dynamo.config.patch(capture_scalar_outputs=True):
                        actual = compiled_prepare(
                            torch.tensor(cu_seqlens, dtype=dtype),
                            torch.empty(cu_seqlens[-1]),
                        )
                    torch.testing.assert_close(
                        actual, torch.tensor(expected, dtype=dtype)
                    )

    def test_chunk_indices_reject_a_capacity_below_the_chunk_count(self):
        with self.assertRaises(RuntimeError):
            prepare_chunk_indices.__wrapped__(
                torch.tensor([0, 130], dtype=torch.int32), 64, 2
            )

    def test_eager_chunk_indices_are_unchanged(self):
        torch.testing.assert_close(
            prepare_chunk_indices.__wrapped__(
                torch.tensor([0, 130, 194, 195], dtype=torch.int64), 64
            ),
            torch.tensor([[0, 0], [0, 1], [0, 2], [1, 0], [2, 0]], dtype=torch.int64),
        )

    def test_tensor_cache_adds_no_graph_outputs(self):
        """``tensor_cache`` appends its arguments to a Python list. Traced, the
        symbolic capacity became an extra graph output."""
        graphs = []

        def record_graph(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        def prepare_cached(
            cu_seqlens: torch.Tensor, token_storage: torch.Tensor
        ) -> torch.Tensor:
            return prepare_chunk_indices(
                cu_seqlens,
                64,
                _compiled_chunk_index_capacity(token_storage.shape[0], cu_seqlens, 64),
            )

        compiled = torch.compile(
            prepare_cached, backend=record_graph, dynamic=True, fullgraph=True
        )
        torch.testing.assert_close(
            compiled(torch.tensor([0, 130, 194], dtype=torch.int32), torch.empty(194)),
            torch.tensor(
                [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]], dtype=torch.int32
            ),
        )
        self.assertTrue(graphs)
        for gm in graphs:
            self.assertEqual(len(_output_nodes(gm)), 1)


class TestKDABackendPrefillUnderCompile(CustomTestCase):
    def setUp(self):
        torch._dynamo.reset()

    def test_conv_block_table_is_a_tensor_built_on_device(self):
        """The conv launch grid was sized from ``extend_seq_lens_cpu``, so a
        graph warmed up on a one-token batch guarded on that length."""
        self.assertIsNone(
            _compiled_conv_block_table(torch.tensor([0, 1], dtype=torch.int32), 1)
        )
        graphs = []

        def record_graph(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        def build_table(
            cu_seqlens: torch.Tensor, token_storage: torch.Tensor
        ) -> torch.Tensor:
            return _compiled_conv_block_table(cu_seqlens, token_storage.shape[0])

        compiled = torch.compile(
            build_table, backend=record_graph, dynamic=True, fullgraph=True
        )
        for cu_seqlens, expected in (
            ([0, 1], [[0, 0], [0, 1]]),
            ([0, 10, 26], [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]]),
        ):
            with self.subTest(cu_seqlens=cu_seqlens):
                torch.testing.assert_close(
                    compiled(
                        torch.tensor(cu_seqlens, dtype=torch.int32),
                        torch.empty(cu_seqlens[-1]),
                    ),
                    torch.tensor(expected, dtype=torch.int32),
                )
        for gm in graphs:
            for value in _output_nodes(gm):
                self.assertIsInstance(value.meta["example_value"], torch.Tensor)

    def test_token_padding_accepts_a_data_dependent_logical_size(self):
        def round_trip_padding(
            mixed_qkv: torch.Tensor,
            a: torch.Tensor,
            b: torch.Tensor,
            logical_size: torch.Tensor,
        ) -> torch.Tensor:
            logical_num_tokens = int(logical_size[0])
            physical_num_tokens = mixed_qkv.shape[0]
            mixed_qkv, a, b = _trim_kda_token_padding(
                mixed_qkv, a, b, logical_num_tokens
            )
            core_attn_out = mixed_qkv.unsqueeze(0) + a + b
            return _restore_kda_token_padding(
                core_attn_out, logical_num_tokens, physical_num_tokens
            )

        mixed_qkv = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        a = torch.ones(1, 5, 4)
        b = torch.full((1, 5, 4), 2.0)
        with torch._dynamo.config.patch(capture_scalar_outputs=True):
            compiled = torch.compile(
                round_trip_padding, backend="eager", dynamic=True, fullgraph=True
            )
            for logical_num_tokens in (3, 5):
                with self.subTest(logical_num_tokens=logical_num_tokens):
                    actual = compiled(
                        mixed_qkv, a, b, torch.tensor([logical_num_tokens])
                    )
                    expected = torch.zeros(1, 5, 4)
                    expected[:, :logical_num_tokens] = (
                        mixed_qkv[:logical_num_tokens].unsqueeze(0)
                        + a[:, :logical_num_tokens]
                        + b[:, :logical_num_tokens]
                    )
                    torch.testing.assert_close(actual, expected)

    def test_padding_rows_are_zeroed_without_a_host_token_count(self):
        compiled = torch.compile(
            _mask_kda_padding_rows, backend="eager", dynamic=True, fullgraph=True
        )
        core_attn_out = torch.arange(30, dtype=torch.float32).reshape(1, 5, 2, 3)
        core_attn_out[:, 3:] = float("nan")
        expected = core_attn_out.clone()
        expected[:, 3:] = 0
        torch.testing.assert_close(
            compiled(core_attn_out, torch.tensor([0, 1, 3], dtype=torch.int32)),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
