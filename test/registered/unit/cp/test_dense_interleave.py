"""Noncontiguous CP queries must keep their original causal and SWA positions."""

from contextlib import nullcontext
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.cp import base, interleave
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDenseInterleave(CustomTestCase):
    def test_odd_requests_with_prefix_and_padding(self):
        # A suffix-shaped causal mask is wrong for every-fourth-token Q shards.
        # The independent reference attends to each original request prefix.
        torch.manual_seed(7)
        extend, prefix = [5, 3, 7], [3, 0, 2]
        for kv_heads in (1, 2, 4):
            for window in (-1, 2):
                with self.subTest(kv_heads=kv_heads, window=window):
                    keys = [
                        torch.randn(p + n, kv_heads, 8) for p, n in zip(prefix, extend)
                    ]
                    values = [
                        torch.randn(p + n, kv_heads, 6) for p, n in zip(prefix, extend)
                    ]
                    queries = torch.randn(sum(extend), 4, 8)

                    def attend(q, request, length):
                        start = 0 if window < 0 else max(0, length - 1 - window)
                        k = keys[request][start:length].repeat_interleave(
                            4 // kv_heads, 1
                        )
                        v = values[request][start:length].repeat_interleave(
                            4 // kv_heads, 1
                        )
                        scores = torch.einsum("hd,shd->hs", q, k) / 8**0.5
                        return torch.einsum("hs,shd->hd", scores.softmax(-1), v)

                    expected, offset = [], 0
                    for request, (p, n) in enumerate(zip(prefix, extend)):
                        for pos in range(n):
                            expected.append(
                                attend(queries[offset + pos], request, p + pos + 1)
                            )
                        offset += n
                    expected = torch.stack(expected)
                    for rank in range(4):
                        parallel = SimpleNamespace(attn_cp_rank=rank)
                        with patch.object(base, "get_parallel", return_value=parallel):
                            strategy = interleave.InterleaveCPStrategy(4)
                            batch = SimpleNamespace(
                                input_ids=torch.arange(sum(extend)),
                                extend_seq_lens_cpu=extend,
                                extend_seq_lens=torch.tensor(extend, dtype=torch.int32),
                                extend_prefix_lens=torch.tensor(
                                    prefix, dtype=torch.int32
                                ),
                                attn_cp_metadata=strategy.build_metadata(
                                    sum(extend), None, extend
                                ),
                            )
                            indices = strategy.local_q_indices(sum(extend), batch)
                            local_q = torch.cat(
                                [queries[indices], torch.zeros(2, 4, 8)]
                            )

                            def paged_attention(
                                q, cu_q, lengths, max_q, *, request_indices
                            ):
                                self.assertEqual(max_q, 1)
                                self.assertEqual(cu_q.tolist(), list(range(len(q) + 1)))
                                return torch.stack(
                                    [
                                        attend(row, int(req), int(length))
                                        for row, req, length in zip(
                                            q, request_indices, lengths
                                        )
                                    ]
                                )

                            actual = strategy.run_attention(
                                local_q, batch, "cpu", paged_attention
                            )
                            self.assertIsNotNone(
                                actual, "dense interleave attention was not dispatched"
                            )
                            torch.testing.assert_close(
                                actual[: len(indices)], expected[indices]
                            )
                            torch.testing.assert_close(
                                actual[len(indices) :], torch.zeros(2, 4, 6)
                            )

    def test_gathered_cache_rows_and_mla_return_contract(self):
        from sglang.srt.model_executor import forward_context

        # The collective boundary supplies rank-packed data. Real gather code
        # must strip sentinel padding and restore slots before the pool write.
        k = torch.arange(15 * 3).reshape(15, 1, 3).float()
        v = -torch.arange(15 * 2).reshape(15, 1, 2).float()
        layer = SimpleNamespace(is_cross_attention=False, k_scale=None, v_scale=None)
        for rank in range(4):
            parallel = SimpleNamespace(attn_cp_rank=rank, attn_cp_group=None)
            with patch.object(base, "get_parallel", return_value=parallel):
                strategy = interleave.InterleaveCPStrategy(4)
                meta = strategy.build_metadata(15, None, [5, 3, 7])
                meta.per_rank_logical_token = list(meta.per_rank_actual_token)
                meta.per_rank_actual_token = [8] * 4
                batch = SimpleNamespace(
                    attn_cp_metadata=meta,
                    out_cache_loc=torch.arange(15).flip(0),
                    out_cache_loc_is_physical=True,
                )
                stored_k, stored_v = torch.zeros_like(k), torch.zeros_like(v)

                def write(layer, loc, keys, values, *scales):
                    self.assertTrue(loc.physical)
                    torch.testing.assert_close(loc.swa_loc, torch.arange(15))
                    stored_k[loc.loc] = keys
                    stored_v[loc.loc] = values

                def gather(output, local):
                    full = torch.cat([k, v], -1).reshape(15, *local.shape[1:])
                    for source in range(4):
                        shard = full[source::4]
                        output[source * 8 : (source + 1) * 8].fill_(12345)
                        output[source * 8 : source * 8 + len(shard)] = shard
                    torch.testing.assert_close(local[: len(k[rank::4])], full[rank::4])

                pool = SimpleNamespace(set_kv_buffer=write)
                with (
                    patch.object(interleave, "get_parallel", return_value=parallel),
                    patch.object(
                        interleave, "use_symmetric_memory", return_value=nullcontext()
                    ),
                    patch.object(
                        interleave, "is_allocation_symmetric", return_value=False
                    ),
                    patch.object(
                        interleave, "attn_cp_all_gather_into_tensor", side_effect=gather
                    ),
                    patch.object(
                        forward_context, "get_token_to_kv_pool", return_value=pool
                    ),
                    patch.object(torch.cuda, "current_stream", return_value=None),
                ):
                    strategy.materialize_full_kv(
                        batch, layer, k[rank::4], v[rank::4], swa_loc=torch.arange(17)
                    )
                    torch.testing.assert_close(stored_k.flip(0), k)
                    torch.testing.assert_close(stored_v.flip(0), v)
                    # DSA expects a tuple, never a cache-write side effect.
                    full_k, full_rope = strategy.materialize_full_mla_kv(
                        batch, layer, k[rank::4], v[rank::4]
                    )
                    torch.testing.assert_close(full_k, k)
                    torch.testing.assert_close(full_rope, v)


if __name__ == "__main__":
    unittest.main()
