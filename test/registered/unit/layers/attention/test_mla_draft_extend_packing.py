import unittest
from itertools import product
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMLADraftExtendPacking(unittest.TestCase):
    def test_query_rows_stay_with_their_request(self):
        for lengths, buffer_width in product(
            ([1, 3], [3, 1], [2, 4], [1, 4], [2, 2], [4]), (None, 4)
        ):
            with self.subTest(lengths=lengths, buffer_width=buffer_width):
                query = torch.arange(1, sum(lengths) + 1, dtype=torch.float32)
                query = query.view(-1, 1, 1)
                seen_queries = []

                def pad(q, padded, seq_lens, indptr):
                    self.assertTrue(padded.is_contiguous())
                    for i, n in enumerate(seq_lens.tolist()):
                        padded[i, :n] = q[indptr[i] : indptr[i + 1]]
                    return padded

                def unpad(out, indptr, seq_lens, total, **kwargs):
                    return torch.cat(
                        [out[i, :n] for i, n in enumerate(seq_lens.tolist())]
                    )

                def kernel(*, query, **kwargs):
                    seen_queries.append(query.clone())
                    # Distinct KV caches give each request a distinct result.
                    request = torch.arange(len(lengths)).view(-1, 1, 1, 1)
                    return query + 10 * request

                metadata = SimpleNamespace(
                    batch_size=len(lengths),
                    max_seq_len_k=16,
                    max_seq_len_q=max(lengths),
                    sum_seq_lens_q=sum(lengths),
                    seq_lens_q=torch.tensor(lengths, dtype=torch.int32),
                    cu_seqlens_q=torch.tensor([0] + lengths, dtype=torch.int32).cumsum(
                        0, dtype=torch.int32
                    ),
                    block_kv_indices=None,
                    seq_lens_k=None,
                )
                backend = SimpleNamespace(
                    data_type=torch.float32,
                    page_size=1,
                    token_to_kv_pool=SimpleNamespace(
                        get_key_buffer=lambda _: torch.zeros(4, 1, 1)
                    ),
                    forward_decode_metadata=metadata,
                    padded_q_buffer=(
                        torch.empty(4, buffer_width, 1, 1)
                        if buffer_width is not None
                        else None
                    ),
                    pad_draft_extend_query=pad,
                    unpad_draft_extend_output=unpad,
                    _run_decode_kernel=kernel,
                )
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.DRAFT_EXTEND_V2,
                    batch_size=len(lengths),
                )
                layer = SimpleNamespace(
                    tp_q_head_num=1, head_dim=1, v_head_dim=1, layer_id=0
                )
                actual = TRTLLMMLABackend.forward_extend(
                    backend, query, None, None, layer, batch, save_kv_cache=False
                )
                request_ids = torch.repeat_interleave(
                    torch.arange(len(lengths)), torch.tensor(lengths)
                )
                torch.testing.assert_close(
                    actual[:, 0], query[:, 0, 0] + 10 * request_ids
                )
                self.assertEqual(
                    seen_queries[0].shape, (len(lengths), max(lengths), 1, 1)
                )


if __name__ == "__main__":
    unittest.main()
