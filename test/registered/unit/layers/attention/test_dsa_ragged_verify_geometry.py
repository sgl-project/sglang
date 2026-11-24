"""CPU contracts for packed DSA target verify: which backends may run it,
which request slot owns each packed row, and how the indexer groups those rows."""

import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.speculative.ragged_verify_kernels import PadVerifyLayout
from sglang.srt.layers.attention.dsa import dsa_indexer
from sglang.srt.layers.attention.dsa.paged_mqa_logits_backend import (
    DSAPagedMQALogitsBackend,
)
from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    _num_ragged_verify_slots,
)
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_IS_CUDA = "sglang.srt.layers.attention.dsa_backend.is_cuda"

# What the graph runner leaves in the batch slots past the live requests.
_PAD_SEQ_LEN, _PAD_POOL_ROW = 1, 0

# Everything supports_ragged_verify_graph reads, in a configuration it admits.
_ADMITTED_CONFIG = dict(
    dsa_decode_impl="flashmla_sparse",
    dsa_kv_cache_store_fp8=False,
    kv_cache_dtype=torch.bfloat16,
    attn_dcp_size=1,
    dsa_index_kpool=1,
    hisparse_coordinator=None,
    _kv_shard_pool=None,
)

# Kimi-K3.1's deployment, fp8 flashmla_kv: the other configuration the gate admits.
_K31_CONFIG = dict(
    dsa_decode_impl="flashmla_kv",
    dsa_kv_cache_store_fp8=True,
    kv_cache_dtype=torch.float8_e4m3fn,
)


def _backend(**overrides):
    backend = object.__new__(DeepseekSparseAttnBackend)
    for name, value in {**_ADMITTED_CONFIG, **overrides}.items():
        setattr(backend, name, value)
    return backend


class TestDSARaggedVerifyCapability(CustomTestCase):
    def _assert_refused(self, backend):
        self.assertFalse(backend.supports_ragged_verify_graph)
        # The refusal comes before any argument is read.
        with self.assertRaisesRegex(NotImplementedError, "flashmla_sparse"):
            backend._ragged_verify_slots(
                layout=None, seq_lens=None, req_pool_indices=None
            )

    def test_unsupported_configs_cannot_enter_ragged_metadata(self):
        """Only CUDA flashmla_kv and BF16 flashmla_sparse pass; one bad setting fails;
        a packed batch reaching such a backend is refused rather than given metadata."""
        # Settings that keep metadata request-major, refused under either kernel.
        request_major = (
            ("attn_dcp_size", 2),
            ("dsa_index_kpool", 4),
            ("hisparse_coordinator", object()),
            ("_kv_shard_pool", object()),
        )
        with patch(_IS_CUDA, return_value=True):
            self.assertTrue(_backend().supports_ragged_verify_graph)
            self.assertTrue(_backend(**_K31_CONFIG).supports_ragged_verify_graph)
            for name, value in (
                ("dsa_decode_impl", "trtllm"),
                ("dsa_kv_cache_store_fp8", True),
                ("kv_cache_dtype", torch.float8_e4m3fn),
                *request_major,
            ):
                with self.subTest(name=name):
                    self._assert_refused(_backend(**{name: value}))
            for name, value in request_major:
                with self.subTest(base="flashmla_kv", name=name):
                    self._assert_refused(_backend(**_K31_CONFIG, **{name: value}))
        with patch(_IS_CUDA, return_value=False):
            self._assert_refused(_backend())
            self._assert_refused(_backend(**_K31_CONFIG))

    def test_hybrid_backend_needs_both_halves(self):
        """Kimi-K3.1 pairs DSA with a linear-attention backend;
        the wrapper admits packed verify only when both of them do."""
        hybrid = object.__new__(HybridLinearAttnBackend)
        for full, linear in product((True, False), repeat=2):
            with self.subTest(full=full, linear=linear):
                hybrid.full_attn_backend = SimpleNamespace(
                    supports_ragged_verify_graph=full
                )
                hybrid.linear_attn_backend = SimpleNamespace(
                    supports_ragged_verify_graph=linear
                )
                self.assertEqual(hybrid.supports_ragged_verify_graph, full and linear)


def _fewest_slots(*, num_rows, batch_size, num_draft_tokens):
    """Slots a tier needs at worst, when every batch slot is live with one row;
    the remaining rows go to padding slots of num_draft_tokens rows each."""
    num_slots = batch_size
    while (num_slots - batch_size) * num_draft_tokens < num_rows - batch_size:
        num_slots += 1
    return num_slots


def _live_layouts(*, num_live, num_rows, num_draft_tokens):
    """Layouts the planner may emit for num_live requests: fewest rows, most, a mix;
    each request has 1 to num_draft_tokens rows, num_rows at most in total."""
    wishes = (
        [1] * num_live,
        [num_draft_tokens] * num_live,
        [1 + i % num_draft_tokens for i in range(num_live)],
    )
    layouts = []
    for wish in wishes:
        spare_rows = num_rows - num_live
        lens = []
        for want in wish:
            extra = min(want - 1, spare_rows)
            lens.append(1 + extra)
            spare_rows -= extra
        if lens not in layouts:
            layouts.append(lens)
    return layouts


def _addresses(slots):
    return [
        tensor.data_ptr()
        for tensor in (
            slots.seq_lens,
            slots.req_pool_indices,
            slots.padded.verify_lens,
            slots.kv_lens,
            slots.cache_seqlens,
        )
    ]


class TestDSARaggedVerifySlots(CustomTestCase):
    def _slots(self, backend, *, num_rows, batch_size, live_lens, out=None):
        """Pad a live layout into request slots, from the graph runner's batch:
        the live requests, then its own padding slots."""
        num_live = len(live_lens)
        num_pad = batch_size - num_live
        prefix_lens = [60 + 7 * i for i in range(num_live)]
        pool_rows = [2 * (num_live - i) + 1 for i in range(num_live)]
        slots = backend._ragged_verify_slots(
            # Device-only, as at replay: the layout has no host copy of its lengths.
            layout=RaggedVerifyLayout.from_verify_lens_device(
                verify_lens=torch.tensor(live_lens, dtype=torch.int32),
                graph_num_tokens=num_rows,
            ),
            seq_lens=torch.tensor(prefix_lens + [_PAD_SEQ_LEN] * num_pad),
            req_pool_indices=torch.tensor(pool_rows + [_PAD_POOL_ROW] * num_pad),
            out=out,
        )
        return slots, prefix_lens, pool_rows

    def _check_tier(self, *, num_draft_tokens, num_rows, batch_size):
        backend = _backend(speculative_num_draft_tokens=num_draft_tokens)
        num_slots = _fewest_slots(
            num_rows=num_rows, batch_size=batch_size, num_draft_tokens=num_draft_tokens
        )
        slots = addresses = None
        # A tier keeps one set of slots and reloads it in place for each live layout,
        # as replay does; the batch fills up and then drains again.
        for num_live in [*range(1, batch_size + 1), *range(batch_size - 1, 0, -1)]:
            for live_lens in _live_layouts(
                num_live=num_live, num_rows=num_rows, num_draft_tokens=num_draft_tokens
            ):
                case = f"{num_draft_tokens=} {num_rows=} {batch_size=} {live_lens=}"
                loaded, prefix_lens, pool_rows = self._slots(
                    backend,
                    num_rows=num_rows,
                    batch_size=batch_size,
                    live_lens=live_lens,
                    out=slots,
                )
                if slots is None:
                    slots, addresses = loaded, _addresses(loaded)
                    # A pool row times the req_to_token row stride overflows int32;
                    # the fill kernel computes it in this dtype.
                    self.assertEqual(slots.req_pool_indices.dtype, torch.int64, case)
                self.assertIs(loaded, slots, case)
                self.assertEqual(_addresses(slots), addresses, case)

                # Together these give num_rows <= num_slots * num_draft_tokens,
                # which the fill kernel asserts before it launches.
                lens = slots.padded.verify_lens.tolist()
                self.assertEqual(len(lens), num_slots, case)
                self.assertEqual(lens[:num_live], live_lens, case)
                self.assertEqual(sum(lens), num_rows, case)
                self.assertTrue(all(0 <= n <= num_draft_tokens for n in lens), case)

                num_pad = num_slots - num_live
                seq_lens = prefix_lens + [_PAD_SEQ_LEN] * num_pad
                self.assertEqual(slots.seq_lens.tolist(), seq_lens, case)
                self.assertEqual(
                    slots.req_pool_indices.tolist(),
                    pool_rows + [_PAD_POOL_ROW] * num_pad,
                    case,
                )
                self.assertEqual(
                    slots.kv_lens.tolist(),
                    [seq_len + n for seq_len, n in zip(seq_lens, lens)],
                    case,
                )

    def test_every_packed_row_gets_an_owner_within_the_block(self):
        """Slots own every row, at most a block each, and live requests keep their rows;
        padding slots are seq_len 1 on pool row 0, and fewer slots would not do."""
        with patch(_IS_CUDA, return_value=True):
            for num_draft_tokens, num_rows, batch_size in product(
                (1, 2, 6, 8), range(1, 25), (1, 2, 3, 8)
            ):
                # The backend is never handed more request slots than packed rows.
                if batch_size <= num_rows:
                    self._check_tier(
                        num_draft_tokens=num_draft_tokens,
                        num_rows=num_rows,
                        batch_size=batch_size,
                    )

    def test_batch_slots_alone_cannot_own_a_tier_with_slack(self):
        """Eight one-row requests in a 16-row tier cannot be padded in their own slots;
        capping leaves rows unowned, not capping overfills one; extra slots fix both."""
        live = torch.ones(8, dtype=torch.int32)
        capped = PadVerifyLayout.torch(
            verify_lens=live, graph_num_tokens=16, padded_bs=8, cap=6
        )
        self.assertEqual(sum(capped.verify_lens.tolist()), 8)
        uncapped = PadVerifyLayout.torch(
            verify_lens=live, graph_num_tokens=16, padded_bs=8
        )
        self.assertEqual(max(uncapped.verify_lens.tolist()), 9)

        num_slots = _num_ragged_verify_slots(
            num_rows=16, batch_size=8, num_draft_tokens=6
        )
        self.assertEqual(num_slots, 10)
        owned = PadVerifyLayout.torch(
            verify_lens=live, graph_num_tokens=16, padded_bs=num_slots, cap=6
        )
        owned_lens = owned.verify_lens.tolist()
        self.assertEqual(owned_lens[:8], [1] * 8)
        self.assertEqual(sum(owned_lens), 16)
        self.assertLessEqual(max(owned_lens), 6)

    def test_a_request_without_rows_cannot_push_a_slot_past_the_block(self):
        """Even a live request without rows keeps every padding slot within one block;
        replay cannot test for one, and the kernels size a slot's rows by the block."""
        num_draft_tokens, num_rows, live_lens = 6, 14, [1, 0]
        with patch(_IS_CUDA, return_value=True):
            slots, _, _ = self._slots(
                _backend(speculative_num_draft_tokens=num_draft_tokens),
                num_rows=num_rows,
                batch_size=len(live_lens),
                live_lens=live_lens,
            )
        lens = slots.padded.verify_lens.tolist()
        # More leftover rows than the padding slots hold within the block.
        self.assertGreater(
            num_rows - sum(live_lens), (len(lens) - len(live_lens)) * num_draft_tokens
        )
        self.assertEqual(lens[: len(live_lens)], live_lens)
        self.assertLessEqual(max(lens), num_draft_tokens)


class TestDSARaggedVerifyIndexerRoute(CustomTestCase):
    def test_one_cache_seqlens_entry_per_row_keeps_logits_per_row(self):
        """The indexer takes its [B, next_n] grouping from cache_seqlens' row count,
        so one entry per packed row gives every logits route a length per row."""
        # 16 rows owned by 6 slots: three live requests verifying 1, 3 and 6 rows,
        # then three padding slots of two rows each over the fill prefix.
        prefix_lens = [62, 127, 250, 1, 1, 1]
        verify_lens = [1, 3, 6, 2, 2, 2]
        row_lens = [
            prefix + j + 1
            for prefix, num_verify in zip(prefix_lens, verify_lens)
            for j in range(num_verify)
        ]
        num_rows = len(row_lens)
        lengths = torch.tensor(row_lens, dtype=torch.int32)
        page_table = torch.zeros((num_rows, 8), dtype=torch.int32)
        q = torch.arange(num_rows * 2 * 128, dtype=torch.float32).reshape(
            num_rows, 2, 128
        )
        weights = torch.arange(num_rows * 2, dtype=torch.float32).reshape(
            num_rows, 2, 1
        )
        kv = torch.zeros((1, 64 * 132), dtype=torch.uint8)
        pool = SimpleNamespace(
            index_page_size=64,
            get_index_k_with_scale_buffer=lambda layer_id: kv,
        )

        def topk_transform(logits, topk):
            return logits.topk(topk, dim=-1).indices.to(torch.int32)

        # A packed DSAMetadata entry as DSAIndexerMetadata exposes it;
        # cache_seqlens_int32 is the per-row dsa_seqlens_expanded itself.
        metadata = SimpleNamespace(
            get_page_table_1=lambda: page_table,
            get_page_table_64=lambda: page_table,
            get_seqlens_int32=lambda: lengths,
            get_seqlens_expanded=lambda: lengths,
            get_dsa_extend_len_cpu=lambda: [1] * num_rows,
            paged_mqa_ctx_lens_2d=lengths.view(-1, 1),
            paged_mqa_schedule_metadata=torch.zeros(1, dtype=torch.int32),
            topk_transform=topk_transform,
        )

        def causal_logits(*, context_lens, max_len):
            # Stands in for the logits kernel:
            # a row scores the keys below its context length, later keys higher.
            logits = torch.arange(max_len, dtype=torch.float32).repeat(
                context_lens.shape[0], 1
            )
            logits.masked_fill_(
                torch.arange(max_len)[None, :] >= context_lens[:, None],
                float("-inf"),
            )
            return logits

        def split_logits(
            fn, query, cache, w, ctx_lens_2d, blocks, schedule, max_len, *, q_offset
        ):
            self.assertEqual(tuple(ctx_lens_2d.shape), (num_rows, 1))
            self.assertEqual((query.shape[0], w.shape[0], q_offset), (num_rows,) * 3)
            self.assertIs(blocks, page_table)
            return causal_logits(context_lens=ctx_lens_2d.reshape(-1), max_len=max_len)

        def cutedsl_logits(
            query, cache, w, ctx_lens_1d, blocks, schedule, max_len, **grouping
        ):
            self.assertEqual(tuple(ctx_lens_1d.shape), (num_rows,))
            self.assertEqual(
                (grouping["B"], grouping["next_n"], grouping["q_offset"]),
                (num_rows, 1, num_rows),
            )
            self.assertEqual(
                (grouping["dsl_expand_factor"], grouping["dsl_atom"]), (1, 1)
            )
            self.assertIs(blocks, page_table)
            return causal_logits(context_lens=ctx_lens_1d, max_len=max_len)

        for logits_backend in (
            DSAPagedMQALogitsBackend.DEEPGEMM,
            DSAPagedMQALogitsBackend.CUTEDSL,
        ):
            with self.subTest(logits_backend=logits_backend.value):
                indexer = dsa_indexer.Indexer.__new__(dsa_indexer.Indexer)
                torch.nn.Module.__init__(indexer)
                indexer.paged_mqa_logits_backend = logits_backend
                indexer.index_topk = 2
                indexer.num_init_tokens = 0
                indexer.num_local_tokens = 0
                # Arbitrary; only passed on to the CuTe DSL wrapper.
                indexer.sm_count = 148
                with (
                    patch.object(dsa_indexer, "_is_hip", False),
                    patch.object(dsa_indexer, "_is_cuda", True),
                    patch.object(dsa_indexer, "_is_xpu", False),
                    patch.object(dsa_indexer, "_use_aiter_gfx95", False),
                    patch.object(
                        dsa_indexer, "get_token_to_kv_pool", return_value=pool
                    ),
                    patch.object(dsa_indexer, "resolve_paged_mqa_logits_metadata_fn"),
                    patch.object(
                        dsa_indexer,
                        "deepgemm_paged_mqa_logits_split",
                        side_effect=split_logits,
                    ),
                    patch.object(
                        dsa_indexer,
                        "deepgemm_paged_mqa_logits_native",
                        side_effect=AssertionError("rows grouped as [B, next_n]"),
                    ),
                    patch.object(
                        dsa_indexer,
                        "cutedsl_paged_mqa_logits",
                        side_effect=cutedsl_logits,
                    ),
                ):
                    output = indexer._get_topk_paged(
                        SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY),
                        0,
                        q,
                        weights,
                        metadata,
                    )
                # Top-2 of each row are the last two keys below its own length.
                self.assertEqual(output.tolist(), [[n - 1, n - 2] for n in row_lens])


class TestCappedVerifyLayout(CustomTestCase):
    def test_graph_slack_does_not_extend_the_last_real_request(self):
        """A full request tier can still have unused token rows; these must
        never become extra recurrent steps for its final real request."""
        out = PadVerifyLayout.torch(
            verify_lens=torch.ones(8, dtype=torch.int32),
            graph_num_tokens=16,
            padded_bs=8,
            cap=6,
        )
        pointers = tuple(
            x.data_ptr() for x in (out.verify_lens, out.qo_indptr, out.extend_start_loc)
        )
        for widths in ([1] * 8, [1, 3, 1, 1, 1, 1, 1, 1]):
            with self.subTest(widths=widths):
                PadVerifyLayout.torch(
                    verify_lens=torch.tensor(widths, dtype=torch.int32),
                    graph_num_tokens=16,
                    padded_bs=8,
                    cap=6,
                    out=out,
                )
                self.assertEqual(out.verify_lens.tolist(), widths)
                self.assertEqual(out.qo_indptr[-1].item(), sum(widths))
                self.assertEqual(
                    tuple(
                        x.data_ptr()
                        for x in (out.verify_lens, out.qo_indptr, out.extend_start_loc)
                    ),
                    pointers,
                )


if __name__ == "__main__":
    unittest.main()
