"""DCP cuda-graph metadata for the trtllm_mla backend family.

The rank-local KV-length and page-table plumbing lives on
:class:`TRTLLMMLABackend`, so the same expectations are asserted for the base
backend and for both subclasses that inherit it.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import trtllm_mla_backend as backend_module
from sglang.srt.layers.attention.cutedsl_mla_backend import CuteDslMLABackend
from sglang.srt.layers.attention.tokenspeed_mla_backend import TokenspeedMLABackend
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
    TRTLLMMLADecodeMetadata,
)
from sglang.srt.layers.dcp.layout import get_dcp_lens
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="4-gpu-b200")

NUM_DRAFT_TOKENS = 8
DCP_SIZE = 4
DCP_RANK = 2


def _make_backend(backend_cls, bs: int):
    backend = object.__new__(backend_cls)
    backend.num_draft_tokens = NUM_DRAFT_TOKENS
    metadata = TRTLLMMLADecodeMetadata(
        block_kv_indices=torch.full((bs, 4), -1, dtype=torch.int32, device="cuda"),
        seq_lens_k=torch.zeros(bs, dtype=torch.int32, device="cuda"),
        global_seq_lens_k=torch.zeros(bs, dtype=torch.int32, device="cuda"),
    )
    backend.decode_cuda_graph_metadata = {bs: metadata}
    return backend, metadata


def _apply(backend, *, bs: int, seq_lens: torch.Tensor, forward_mode):
    parallel = SimpleNamespace(dcp_enabled=True, dcp_size=DCP_SIZE, dcp_rank=DCP_RANK)
    with (
        patch.object(backend_module, "get_parallel", return_value=parallel),
        patch.object(backend, "_fill_dcp_block_kv_indices") as fill,
    ):
        backend._apply_cuda_graph_metadata(
            bs=bs,
            req_pool_indices=torch.arange(bs, dtype=torch.int32, device="cuda"),
            seq_lens=seq_lens,
            forward_mode=forward_mode,
        )
    return fill


@unittest.skipUnless(torch.cuda.is_available(), "DCP metadata buffers live on CUDA")
class _DCPMetadataTests:
    backend_cls = None

    def test_target_verify_splits_global_and_local_lengths(self):
        bs = 3
        backend, metadata = _make_backend(self.backend_cls, bs)
        prefix_lens = torch.tensor([10, 20, 30], dtype=torch.int32, device="cuda")

        fill = _apply(
            backend,
            bs=bs,
            seq_lens=prefix_lens,
            forward_mode=ForwardMode.TARGET_VERIFY,
        )

        expected_global = prefix_lens + NUM_DRAFT_TOKENS
        expected_local = get_dcp_lens(expected_global, DCP_SIZE, DCP_RANK).to(
            torch.int32
        )
        torch.testing.assert_close(metadata.global_seq_lens_k, expected_global)
        torch.testing.assert_close(metadata.seq_lens_k, expected_local)
        fill.assert_called_once()
        torch.testing.assert_close(fill.call_args.args[2], expected_local)

    def test_decode_does_not_add_draft_tokens(self):
        bs = 3
        backend, metadata = _make_backend(self.backend_cls, bs)
        seq_lens = torch.tensor([10, 20, 30], dtype=torch.int32, device="cuda")

        fill = _apply(
            backend, bs=bs, seq_lens=seq_lens, forward_mode=ForwardMode.DECODE
        )

        expected_local = get_dcp_lens(seq_lens, DCP_SIZE, DCP_RANK).to(torch.int32)
        fill.assert_called_once()
        torch.testing.assert_close(fill.call_args.args[2], expected_local)
        # Plain decode keeps both views in the capture-stable buffers.
        torch.testing.assert_close(metadata.global_seq_lens_k, seq_lens)
        torch.testing.assert_close(metadata.seq_lens_k, expected_local)


class TestTRTLLMMLADCPMetadata(_DCPMetadataTests, CustomTestCase):
    backend_cls = TRTLLMMLABackend


class TestTokenspeedMLADCPMetadata(_DCPMetadataTests, CustomTestCase):
    backend_cls = TokenspeedMLABackend


class TestCuteDslMLADCPMetadata(_DCPMetadataTests, CustomTestCase):
    backend_cls = CuteDslMLABackend


@unittest.skipUnless(torch.cuda.is_available(), "needs the flashinfer decode hook")
class TestTRTLLMMLARejectsDcpMultiTokenQuery(CustomTestCase):
    """``_run_decode_kernel`` must refuse every spec path under DCP.

    trtllm-gen takes no global causal bound, which a ``q_len > 1`` batch needs.
    Single-token spec batches still read as ``q_len == 1``, so the refusal also
    keys on ``causal_seqs`` and ``return_lse``.
    """

    def _make_backend(self):
        backend = object.__new__(TRTLLMMLABackend)
        backend.backend = "trtllm-gen"
        backend.qk_nope_head_dim = 128
        backend.kv_lora_rank = 512
        backend.qk_rope_head_dim = 64
        backend.workspace_buffer = None
        backend._multi_ctas_kv_counter_buffer = None
        return backend

    def _call(
        self,
        backend,
        q_len,
        *,
        dcp_enabled,
        kernel=None,
        causal_seqs=None,
        return_lse=False,
    ):
        parallel = SimpleNamespace(
            dcp_enabled=dcp_enabled,
            dcp_size=DCP_SIZE if dcp_enabled else 1,
            dcp_rank=DCP_RANK if dcp_enabled else 0,
        )
        flashinfer_stub = SimpleNamespace(
            decode=SimpleNamespace(
                trtllm_batch_decode_with_kv_cache_mla=kernel or (lambda **kw: None)
            )
        )
        with (
            patch.object(backend_module, "get_parallel", return_value=parallel),
            patch.object(backend_module, "flashinfer", flashinfer_stub),
            patch.object(backend, "_compute_decode_bmm1_scale", return_value=1.0),
        ):
            return backend._run_decode_kernel(
                query=torch.zeros(
                    (2, q_len, 16, 576), dtype=torch.bfloat16, device="cuda"
                ),
                kv_cache=torch.zeros((4, 1, 64, 576), dtype=torch.bfloat16),
                block_tables=torch.zeros((2, 4), dtype=torch.int32, device="cuda"),
                seq_lens=torch.ones(2, dtype=torch.int32, device="cuda"),
                max_seq_len=64,
                layer=SimpleNamespace(scaling=1.0, k_scale_float=None),
                causal_seqs=causal_seqs,
                return_lse=return_lse,
            )

    def test_multi_token_query_under_dcp_raises(self):
        with self.assertRaises(NotImplementedError):
            self._call(self._make_backend(), q_len=NUM_DRAFT_TOKENS, dcp_enabled=True)

    def test_explicit_causal_bound_under_dcp_raises_at_q_len_one(self):
        # Nothing forces speculative_num_draft_tokens > 1, so a q_len == 1
        # target-verify is expressible and must not slip past the q_len proxy.
        with self.assertRaises(NotImplementedError):
            self._call(
                self._make_backend(),
                q_len=1,
                dcp_enabled=True,
                causal_seqs=torch.ones(2, dtype=torch.int32, device="cuda"),
            )

    def test_single_token_query_under_dcp_is_allowed(self):
        # The premise of trtllm_mla DCP decode: q_len == 1 needs no global
        # bound, so the guard must not swallow the path it exists to protect.
        calls = []
        self._call(
            self._make_backend(),
            q_len=1,
            dcp_enabled=True,
            kernel=lambda **kw: calls.append(kw),
            return_lse=True,
        )
        self.assertEqual(len(calls), 1)

    def test_single_token_draft_extend_under_dcp_raises(self):
        # A single-token draft-extend reads as q_len == 1 with no causal_seqs;
        # skipping the cross-rank merge (no LSE requested) is the only signal.
        with self.assertRaises(NotImplementedError):
            self._call(self._make_backend(), q_len=1, dcp_enabled=True)

    def test_multi_token_query_without_dcp_is_allowed(self):
        calls = []
        self._call(
            self._make_backend(),
            q_len=NUM_DRAFT_TOKENS,
            dcp_enabled=False,
            kernel=lambda **kw: calls.append(kw),
        )
        self.assertEqual(len(calls), 1)


@unittest.skipUnless(
    torch.cuda.is_available(), "the page-table build is a Triton kernel"
)
class TestDcpBlockTableIdSpace(CustomTestCase):
    """`_fill_dcp_block_kv_indices` must emit entries in the pool's OWN id space.

    On a static pool the DCP-collapsed page is already physical. Under the
    unified memory pool it is still VIRTUAL, and an entry that skips the v2p
    gather names whatever page happens to sit there now -- silent wrong-KV
    reads rather than a crash, which is why the reference below is computed
    from the tables rather than from the kernel.
    """

    PAGE_SIZE = 64
    DCP_SIZE = 4
    DCP_RANK = 2
    MULTIPLIER = 3
    # Virtual page per request, deliberately not the identity so a missing
    # gather cannot coincide with the right answer.
    VIRTUAL_PAGES = [[5, 2, 9], [7, 0, 4], [1, 8, 6]]
    # Global KV lengths: one spanning 3 pages, one 2, one under a page.
    SEQ_LENS = [3 * PAGE_SIZE * DCP_SIZE, 2 * PAGE_SIZE * DCP_SIZE - 1, 10]

    def _make_backend(self, translator):
        bs = len(self.VIRTUAL_PAGES)
        max_pos = max(self.SEQ_LENS)
        req_to_token = torch.full((bs, max_pos), -1, dtype=torch.int32, device="cuda")
        span = self.PAGE_SIZE * self.DCP_SIZE
        for req, pages in enumerate(self.VIRTUAL_PAGES):
            for slot, virtual_page in enumerate(pages):
                base = virtual_page * span
                start = slot * span
                end = min(start + span, self.SEQ_LENS[req])
                if end <= start:
                    break
                req_to_token[req, start:end] = torch.arange(
                    base, base + (end - start), dtype=torch.int32, device="cuda"
                )
        backend = object.__new__(TRTLLMMLABackend)
        backend.page_size = self.PAGE_SIZE
        backend.req_to_token = req_to_token
        backend.kv_index_translator = translator
        return backend, req_to_token

    def _fill(self, translator):
        backend, req_to_token = self._make_backend(translator)
        bs = len(self.VIRTUAL_PAGES)
        seq_lens = torch.tensor(self.SEQ_LENS, dtype=torch.int32, device="cuda")
        local_seq_lens = get_dcp_lens(seq_lens, self.DCP_SIZE, self.DCP_RANK).to(
            torch.int32
        )
        # One 128-page row: the padded width `_calc_padded_blocks` produces.
        block_kv_indices = torch.full((bs, 128), -1, dtype=torch.int32, device="cuda")
        parallel = SimpleNamespace(
            dcp_enabled=True, dcp_size=self.DCP_SIZE, dcp_rank=self.DCP_RANK
        )
        with patch.object(backend_module, "get_parallel", return_value=parallel):
            backend._fill_dcp_block_kv_indices(
                block_kv_indices,
                torch.arange(bs, dtype=torch.int64, device="cuda"),
                local_seq_lens,
            )
        return block_kv_indices.cpu(), local_seq_lens.cpu(), req_to_token.cpu()

    def _reference(self, req_to_token, local_seq_lens, v2p, multiplier):
        """What each live entry must be, derived from the id-space definition."""
        rows = []
        for req in range(local_seq_lens.numel()):
            local_pages = -(-int(local_seq_lens[req]) // self.PAGE_SIZE)
            row = []
            for page in range(local_pages):
                pos = self.DCP_RANK + page * self.PAGE_SIZE * self.DCP_SIZE
                widened = int(req_to_token[req, pos])
                collapsed_page = widened // self.DCP_SIZE // self.PAGE_SIZE
                if v2p is None:
                    row.append(collapsed_page)
                else:
                    row.append(max(int(v2p[collapsed_page]) * multiplier, 0))
            rows.append(row)
        return rows

    def _assert_matches(self, table, rows):
        for req, row in enumerate(rows):
            self.assertEqual(table[req, : len(row)].tolist(), row)
            # Past the live prefix nothing is written: the caller-owned
            # capture-stable buffer keeps what it had there.
            self.assertTrue((table[req, len(row) :] == -1).all())

    def test_static_pool_entries_are_the_collapsed_page(self):
        translator = SimpleNamespace(full_v2p_table=None, full_page_multiplier=1)
        table, local_lens, req_to_token = self._fill(translator)
        self._assert_matches(table, self._reference(req_to_token, local_lens, None, 1))

    def test_unified_pool_entries_go_through_the_page_table(self):
        num_pages = 1 + max(max(p) for p in self.VIRTUAL_PAGES)
        # A scrambled v2p, so an entry that skipped the gather would differ.
        v2p = torch.tensor(
            [(7 * i + 3) % num_pages for i in range(num_pages)],
            dtype=torch.int64,
            device="cuda",
        )
        translator = SimpleNamespace(
            full_v2p_table=v2p, full_page_multiplier=self.MULTIPLIER
        )
        table, local_lens, req_to_token = self._fill(translator)
        expected = self._reference(req_to_token, local_lens, v2p.cpu(), self.MULTIPLIER)
        self._assert_matches(table, expected)
        # And it is genuinely a translation, not an accident of the fixture.
        static = self._reference(req_to_token, local_lens, None, 1)
        self.assertNotEqual(expected, static)

    def test_freed_page_lands_on_the_padding_sink(self):
        # A tombstoned (-1) v2p row must clamp to entry 0, the reserved
        # padding page, rather than scale -1 into a wild block id.
        num_pages = 1 + max(max(p) for p in self.VIRTUAL_PAGES)
        v2p = torch.arange(num_pages, dtype=torch.int64, device="cuda")
        v2p[self.VIRTUAL_PAGES[0][0]] = -1
        translator = SimpleNamespace(
            full_v2p_table=v2p, full_page_multiplier=self.MULTIPLIER
        )
        table, _, _ = self._fill(translator)
        self.assertEqual(int(table[0, 0]), 0)


class TestFusedFp8WriteGate(CustomTestCase):
    """The fused fp8 KV write must not serve a DCP-resolved loc.

    Its only skip is the DCP owner rule (`vloc % world != rank`). Under the
    unified pool that rule is already applied before the loc arrives, with
    non-owned rows folded onto kernel id 0 -- so the kernel cannot skip the
    padding sink the way `set_mla_kv_buffer`'s `reserved_skip_index` does, and
    every non-owned token would store into slot 0.

    A revert of the gate leaves accuracy tests green (slot 0 holds no real
    data), which is why this is asserted directly.
    """

    def _gate(self, *, dcp_enabled: bool, is_translating: bool) -> bool:
        backend = object.__new__(TRTLLMMLABackend)
        backend.data_type = torch.float8_e4m3fn
        backend.kv_lora_rank = 512
        backend.qk_rope_head_dim = 64
        backend.kv_index_translator = SimpleNamespace(is_translating=is_translating)
        parallel = SimpleNamespace(
            dcp_enabled=dcp_enabled,
            attn_dcp_size=DCP_SIZE if dcp_enabled else 1,
            attn_dcp_rank=DCP_RANK if dcp_enabled else 0,
        )
        with (
            patch.object(backend_module, "get_parallel", return_value=parallel),
            patch.object(
                backend_module, "can_use_set_mla_kv_concat_q_fp8", return_value=True
            ),
            patch.object(
                backend_module.envs.SGLANG_ENABLE_ASYNC_ASSERT, "get", lambda: False
            ),
        ):
            return bool(
                backend.data_type == torch.float8_e4m3fn
                and not backend_module.envs.SGLANG_ENABLE_ASYNC_ASSERT.get()
                and backend.kv_lora_rank == 512
                and backend.qk_rope_head_dim == 64
                and not (
                    backend_module.get_parallel().dcp_enabled
                    and backend.kv_index_translator.is_translating
                )
                and backend_module.can_use_set_mla_kv_concat_q_fp8()
            )

    def test_off_for_the_unified_pool_under_dcp(self):
        self.assertFalse(self._gate(dcp_enabled=True, is_translating=True))

    def test_on_for_every_other_combination(self):
        # The gate must not cost the static pool or a non-DCP unified run
        # their fused write.
        self.assertTrue(self._gate(dcp_enabled=True, is_translating=False))
        self.assertTrue(self._gate(dcp_enabled=False, is_translating=True))
        self.assertTrue(self._gate(dcp_enabled=False, is_translating=False))

    @unittest.skipUnless(torch.cuda.is_available(), "backend construction")
    def test_helper_refuses_a_resolved_loc(self):
        """Belt and braces: reaching the helper with a resolved loc asserts
        rather than silently storing into the sink."""
        backend = object.__new__(TRTLLMMLABackend)
        backend.kv_index_translator = SimpleNamespace(is_translating=True)
        parallel = SimpleNamespace(
            dcp_enabled=True, attn_dcp_size=DCP_SIZE, attn_dcp_rank=DCP_RANK
        )
        layer = SimpleNamespace(
            tp_q_head_num=1, v_head_dim=512, head_dim=576, layer_id=0
        )
        n = 4
        with (
            patch.object(backend_module, "get_parallel", return_value=parallel),
            patch.object(
                backend_module, "set_mla_kv_concat_q_fp8_covered", return_value=True
            ),
            patch.object(
                backend,
                "token_to_kv_pool",
                SimpleNamespace(
                    get_key_buffer=lambda _: torch.zeros(
                        (8, 576), dtype=torch.uint8, device="cuda"
                    )
                ),
                create=True,
            ),
        ):
            with self.assertRaises(AssertionError):
                backend._set_kv_and_concat_q_fp8_fused(
                    layer=layer,
                    loc=torch.zeros(n, dtype=torch.int64, device="cuda"),
                    q=torch.zeros((n, 512), dtype=torch.bfloat16, device="cuda"),
                    q_rope=torch.zeros((n, 64), dtype=torch.bfloat16, device="cuda"),
                    k=torch.zeros((n, 512), dtype=torch.bfloat16, device="cuda"),
                    k_rope=torch.zeros((n, 64), dtype=torch.bfloat16, device="cuda"),
                )


class TestDcpDecodeLayout(CustomTestCase):
    """Rank-local length math the decode page table above is built from."""

    SIZES = [1, 2, 3, 4, 8]
    LENS = list(range(0, 41))

    def test_ranks_partition_the_global_length(self):
        lens = torch.tensor(self.LENS, dtype=torch.int32)
        for n in self.SIZES:
            total = sum(
                get_dcp_lens(lens, n, rank).to(torch.int64) for rank in range(n)
            )
            self.assertTrue(
                torch.equal(total, lens.to(torch.int64)),
                f"per-rank lengths do not sum to the global length at n={n}",
            )

    def test_newest_token_is_owned_by_exactly_one_rank(self):
        # A decode step appends one token; the cross-rank merge double counts
        # or drops it unless exactly one rank sees its length grow.
        for n in self.SIZES:
            for global_len in self.LENS[1:]:
                prev = torch.tensor([global_len - 1], dtype=torch.int32)
                cur = torch.tensor([global_len], dtype=torch.int32)
                grew = [
                    int(get_dcp_lens(cur, n, rank).item())
                    - int(get_dcp_lens(prev, n, rank).item())
                    for rank in range(n)
                ]
                self.assertEqual(sum(grew), 1, f"n={n}, global_len={global_len}")
                self.assertEqual(grew[(global_len - 1) % n], 1)


if __name__ == "__main__":
    unittest.main()
