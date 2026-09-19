"""DSA attention metadata for a DP-attention rank that idles through a
speculative round.

Such a rank is still padded to the sync group's token count, and
``ForwardBatch.prepare_mlp_sync_batch`` turns that into
``batch_size = num_tokens // spec_info.num_tokens_per_req`` fabricated
sequences — so the forward runs ``num_tokens_per_req`` query rows per sequence
while ``seq_lens`` / ``req_pool_indices`` hold one entry each. The eager
``init_forward_metadata`` used to build one metadata row per sequence, which
left the indexer's paged top-k with a ``lengths`` vector shorter than its score
rows ("Expected lengths.size(0) == B to be true"); on ROCm it is fatal because
``aiter_paged_mqa_logits`` scores every query row instead of slicing to
``q_offset`` first.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
from sglang.srt.layers.attention.dsa.utils import cal_padded_tokens
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

NUM_REQ_SLOTS = 8
MAX_CONTEXT = 32
NUM_DRAFT_TOKENS = 4
DP_SIZE = 2


def _make_backend() -> DeepseekSparseAttnBackend:
    """Backend carrying only the attributes init_forward_metadata reads.

    Building the real one needs a ModelRunner with loaded weights, while the
    metadata builder itself is pure tensor bookkeeping.
    """
    backend = object.__new__(DeepseekSparseAttnBackend)
    backend.device = torch.device("cpu")
    backend._arange_buf = torch.arange(8, dtype=torch.int32)
    backend.speculative_num_draft_tokens = NUM_DRAFT_TOKENS
    backend.dsa_index_topk = 2048
    backend.real_page_size = 1
    backend.dsa_decode_impl = "aiter"
    backend.dsa_prefill_impl = "aiter"
    backend.dsa_topk_backend = DSATopKBackend.SGL_KERNEL
    backend.dsa_kv_cache_store_fp8 = False
    backend.enable_auto_select_prefill_impl = False
    backend.use_mha = False
    backend.hisparse_coordinator = None
    backend.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(NUM_REQ_SLOTS * MAX_CONTEXT, dtype=torch.int32).view(
            NUM_REQ_SLOTS, MAX_CONTEXT
        )
    )
    backend.token_to_kv_pool = SimpleNamespace(
        dtype=torch.bfloat16, size=1024, page_size=1
    )
    return backend


def _make_batch(
    *, forward_mode: ForwardMode, batch_size: int, num_tokens_per_req
) -> SimpleNamespace:
    """A DP-padded batch shaped the way prepare_mlp_sync_batch leaves one.

    ``num_tokens_per_req=None`` stands for a round with no speculative input at
    all (``spec_info is None``).
    """
    # Padded rows carry the backend's cuda-graph seq-len fill value.
    seq_lens = torch.ones(batch_size, dtype=torch.int64)
    width = 1 if num_tokens_per_req is None else max(1, num_tokens_per_req)
    return SimpleNamespace(
        batch_size=batch_size,
        forward_mode=forward_mode,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.clone(),
        seq_lens_sum=int(seq_lens.sum()),
        req_pool_indices=torch.zeros(batch_size, dtype=torch.int64),
        spec_info=(
            None
            if num_tokens_per_req is None
            else SimpleNamespace(num_tokens_per_req=num_tokens_per_req)
        ),
        global_num_tokens_cpu=[batch_size * width] * DP_SIZE,
        dp_padding_mode=DpPaddingMode.MAX_LEN,
        is_extend_in_batch=False,
        attn_cp_metadata=None,
    )


class TestDSAIdleSpecMetadata(CustomTestCase):
    def _build_metadata(self, batch):
        """Return (metadata, query rows this rank actually forwards)."""
        backend = _make_backend()
        # The folded top-k v2 plan JIT-compiles a CUDA kernel; the row counts
        # under test are the same either way.
        with envs.SGLANG_OPT_USE_TOPK_V2.override(
            False
        ), get_context().override_server_args(), get_parallel().override(
            attn_cp_size=1, attn_dp_rank=0
        ):
            backend.init_forward_metadata(batch)
            # cal_padded_tokens is the padding calculation prepare_mlp_sync_batch
            # itself follows, so it is an independent statement of the row count.
            query_rows = cal_padded_tokens(batch)
        return backend.forward_metadata, query_rows

    def _assert_row_per_query_row(self, batch, expected_query_rows):
        metadata, query_rows = self._build_metadata(batch)
        self.assertEqual(query_rows, expected_query_rows)

        # `lengths` for the indexer's paged top-k, which requires one entry per
        # score row — the vector whose length used to trip the kernel check.
        self.assertEqual(metadata.dsa_seqlens_expanded.shape[0], query_rows)
        self.assertEqual(metadata.dsa_cache_seqlens_int32.shape[0], query_rows)
        # Per-row KV lengths and page tables the indexer reads alongside it.
        self.assertEqual(metadata.cache_seqlens_int32.shape[0], query_rows)
        self.assertEqual(metadata.page_table_1.shape[0], query_rows)
        self.assertEqual(metadata.real_page_table.shape[0], query_rows)
        self.assertEqual(metadata.cu_seqlens_q.shape[0], query_rows + 1)
        self.assertEqual(metadata.cu_seqlens_k.shape[0], query_rows + 1)
        # The indexer takes sum(dsa_extend_seq_lens_list) as the real (unpadded)
        # query length; a short value makes it score rows the metadata does not
        # describe.
        self.assertEqual(sum(metadata.dsa_extend_seq_lens_list), query_rows)

    def test_idle_speculative_round_covers_every_padded_query_row(self):
        batch = _make_batch(
            forward_mode=ForwardMode.IDLE,
            batch_size=3,
            num_tokens_per_req=NUM_DRAFT_TOKENS,
        )
        self._assert_row_per_query_row(batch, 3 * NUM_DRAFT_TOKENS)

    def test_idle_round_without_speculation_keeps_one_row_per_sequence(self):
        batch = _make_batch(
            forward_mode=ForwardMode.IDLE, batch_size=3, num_tokens_per_req=None
        )
        self._assert_row_per_query_row(batch, 3)

    def test_idle_round_with_unset_spec_width_keeps_one_row_per_sequence(self):
        # SpecInput.num_tokens_per_req defaults to -1 ("this flow never set a
        # width"), which must not scale the batch.
        batch = _make_batch(
            forward_mode=ForwardMode.IDLE, batch_size=3, num_tokens_per_req=-1
        )
        self._assert_row_per_query_row(batch, 3)

    def test_decode_round_is_unchanged(self):
        # A live decode rank describes its own sequences; only idle padding is
        # fabricated, so the expansion must not reach this path.
        batch = _make_batch(
            forward_mode=ForwardMode.DECODE, batch_size=3, num_tokens_per_req=1
        )
        self._assert_row_per_query_row(batch, 3)


if __name__ == "__main__":
    unittest.main()
