"""BUG REGRESSION: step i of a multi-step MLA draft decodes over the committed prefix
plus i + 1 draft tokens, since `draft_forward` writes them but keeps them out of
`seq_lens`. Every case enters through the family's multi-step backend, so a step that
reads too few keys, two layers that both add the tokens, and a decode that reads
another step's metadata fail alike. Near the context limit, a draft step, a verify and
a draft extend all read past the context, so the bounds a graph fixes at capture must
reach their tokens.

Stood in for: the kernels, scheduler metadata, KV views, quantization, the parallel
context and the step backends' constructors. So these cases say nothing about physical
page ids, translated pools, quantization arithmetic or cross-rank numerics."""

import unittest
from contextlib import contextmanager
from itertools import product
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention import (
    cutedsl_mla_backend,
    flashinfer_mla_backend,
    flashmla_backend,
    trtllm_mla_backend,
)
from sglang.srt.layers.attention.cutedsl_mla_backend import (
    CuteDslMLABackend,
    CuteDslMLAMultiStepDraftBackend,
)
from sglang.srt.layers.attention.flashinfer_mla_backend import (
    FlashInferMLAAttnBackend,
    FlashInferMLAMultiStepDraftBackend,
)
from sglang.srt.layers.attention.flashmla_backend import (
    PAGE_SIZE,
    FlashMLABackend,
    FlashMLAMultiStepDraftBackend,
)
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
    TRTLLMMLADecodeMetadata,
    TRTLLMMLAMultiStepDraftBackend,
)
from sglang.srt.layers.dcp.layout import get_dcp_lens
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

NUM_STEPS = 4
BS = 3
MAX_PAGES = 4
DCP_SIZE = 4
DCP_RANK = 2
# Around a page boundary, so a dropped draft token can lose a page as well as keys.
COMMITTED = (61, 62, 64)
NO_DCP = SimpleNamespace(dcp_enabled=False)
DCP = SimpleNamespace(dcp_enabled=True, dcp_size=DCP_SIZE, dcp_rank=DCP_RANK)
TRTLLM_FAMILY = (TRTLLMMLAMultiStepDraftBackend, CuteDslMLAMultiStepDraftBackend)
TRTLLM_TARGETS = (TRTLLMMLABackend, CuteDslMLABackend)
LAYER = SimpleNamespace(layer_id=0, tp_q_head_num=1, head_dim=4, v_head_dim=4)
HOST_LENS = (True, False)
# A long chain near a page-aligned context. A request commits up to CONTEXT - 2.
CONTEXT = 64 * PAGE_SIZE
LONG_CHAIN = 8
COMMITTED_NEAR_LIMIT = (CONTEXT - 6, CONTEXT - 2)


def _step_lens(step):
    return torch.tensor(COMMITTED, dtype=torch.int32) + step + 1


def _pages(lens):
    return (lens + PAGE_SIZE - 1) // PAGE_SIZE


class _PageTableKernel:
    """create_flashmla_kv_indices_triton's contract: one entry per page of each length,
    read from that length's request row. A row the table cannot hold raises here; the
    real kernel's grid stops at the table's width and the decode then reads past it."""

    def __getitem__(self, _grid):
        return self

    def __call__(self, table, rows, seq_lens, _prefix, block_kv_indices, *_a, **_k):
        block_kv_indices.fill_(-1)
        for row, length in enumerate(seq_lens.tolist()):
            pages = (length + PAGE_SIZE - 1) // PAGE_SIZE
            assert pages <= block_kv_indices.shape[1], (pages, block_kv_indices.shape)
            block_kv_indices[row, :pages] = table[rows[row], 0]


def _on_cpu(factory):
    def call(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return factory(*args, **kwargs)

    return call


def _cuda_buffers_on_cpu():
    return patch.multiple(
        torch,
        zeros=_on_cpu(torch.zeros),
        full=_on_cpu(torch.full),
        empty=_on_cpu(torch.empty),
        ones=_on_cpu(torch.ones),
    )


def _step_init(self, _model_runner, *_args, **kwargs):
    """What the real constructors take that matters here: the id, when they are given one."""
    self.speculative_step_id = kwargs.get("speculative_step_id")
    self.max_context_len = MAX_PAGES * PAGE_SIZE
    self.page_size = PAGE_SIZE
    self.num_draft_tokens = NUM_STEPS
    # Row 0 is the padding sink, as in a real request table.
    self.req_to_token = torch.zeros((BS + 1, MAX_PAGES * PAGE_SIZE), dtype=torch.int32)
    self.runner_ids = SimpleNamespace(is_translating=False, reads_are_translated=False)


def _forward_batch(with_host_lens=True, committed=COMMITTED):
    """These backends opt out of the host mirror, so serving runs them without it."""
    committed = torch.tensor(committed, dtype=torch.int64)
    return SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        batch_size=BS,
        seq_lens=committed,
        seq_lens_cpu=committed.clone() if with_host_lens else None,
        seq_lens_sum=int(committed.sum()),
        req_pool_indices=torch.arange(1, BS + 1),
        positions=committed,
        iteration_ids=None,
        spec_info=SimpleNamespace(is_draft_input=lambda: True),
        ragged_verify_layout=None,
    )


def _spec_batch(committed, mode, with_host_lens=True):
    """A verify scores LONG_CHAIN drafts past its committed lengths; a draft extend
    arrives with them already in its lengths."""
    batch = _forward_batch(with_host_lens, committed=(committed,) * BS)
    batch.forward_mode = mode
    batch.positions = torch.arange(BS * LONG_CHAIN)
    batch.spec_info = SimpleNamespace(draft_token_num=LONG_CHAIN)
    if mode.is_draft_extend_v2():
        batch.seq_lens = batch.seq_lens + LONG_CHAIN
        if with_host_lens:
            batch.seq_lens_cpu = batch.seq_lens.clone()
        batch.extend_seq_lens = torch.full((BS,), LONG_CHAIN)
        batch.extend_seq_lens_cpu = [LONG_CHAIN] * BS
    return batch


SPEC_MODES = (ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND_V2)


class _MultiStepCase(CustomTestCase):
    def setUp(self):
        runner_ids = SimpleNamespace(
            read_source=lambda *_a, **_k: SimpleNamespace(
                row_ids=None,
                ids=None,
                row_stride=1,
                entry_page_size=PAGE_SIZE,
                v2p=None,
                is_translated=False,
            )
        )
        self.model_runner = SimpleNamespace(
            device="cpu",
            runner_ids=runner_ids,
            req_to_token_pool=SimpleNamespace(
                size=BS + 1, req_to_token=torch.zeros((BS + 1, MAX_PAGES * PAGE_SIZE))
            ),
        )
        for patcher in (
            patch.object(FlashInferMLAAttnBackend, "__init__", _step_init),
            patch.object(FlashMLABackend, "__init__", _step_init),
            patch.object(TRTLLMMLABackend, "__init__", _step_init),
            patch.object(CuteDslMLABackend, "__init__", _step_init),
            patch.object(
                flashinfer_mla_backend, "get_cuda_graph_max_batch_size", lambda _: BS
            ),
            patch.object(
                flashinfer_mla_backend,
                "get_schedule",
                lambda: SimpleNamespace(page_size=PAGE_SIZE),
            ),
            patch.object(flashinfer_mla_backend, "get_parallel", lambda: NO_DCP),
            patch.object(flashmla_backend, "get_parallel", lambda: NO_DCP),
            patch.object(
                flashmla_backend,
                "create_flashmla_kv_indices_triton",
                _PageTableKernel(),
            ),
            patch.object(flashmla_backend, "paged_view", lambda kv, _page: kv),
            patch.object(flashmla_backend, "scaled_fp8_quant", lambda q, _s: (q, None)),
            patch.object(
                trtllm_mla_backend,
                "create_flashmla_kv_indices_triton",
                _PageTableKernel(),
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)


class _BaseReached(Exception):
    pass


class TestStepIdReachesTheSharedBase(CustomTestCase):
    def test_each_step_backend_hands_its_id_to_the_shared_base(self):
        """The id is read on the shared base, so a constructor that drops it leaves
        that family's steps on the committed prefix. Each calls the base first."""
        for backend_cls in (FlashMLABackend, TRTLLMMLABackend, CuteDslMLABackend):
            seen = []

            def record(_self, *_args, speculative_step_id=None, **_kwargs):
                seen.append(speculative_step_id)
                raise _BaseReached

            with self.subTest(backend=backend_cls.__name__):
                with patch.object(FlashInferMLAAttnBackend, "__init__", record):
                    with self.assertRaises(_BaseReached):
                        backend_cls(None, speculative_step_id=2)
                self.assertEqual(seen, [2])


class TestTRTLLMFamilyDraftStepLengths(_MultiStepCase):
    """trtllm_mla and cutedsl_mla, each step read back from the metadata its kernel takes."""

    def _backend(self, backend_cls):
        backend = backend_cls(self.model_runner, 1, NUM_STEPS)
        graph_metadata = []
        for step in backend.attn_backends:
            step._kv_shard_pool = None
            metadata = TRTLLMMLADecodeMetadata(
                block_kv_indices=torch.full((BS, MAX_PAGES), -1, dtype=torch.int32),
                seq_lens_k=torch.zeros(BS, dtype=torch.int32),
                global_seq_lens_k=torch.zeros(BS, dtype=torch.int32),
            )
            step.decode_cuda_graph_metadata = {(BS, ForwardMode.DECODE): metadata}
            graph_metadata.append(metadata)
        return backend, graph_metadata

    def _decode_on(self, backend, batch, parallel):
        """Run each step's real decode on one batch and return its kernel arguments."""
        calls = {}
        bs = batch.batch_size
        for step, step_backend in enumerate(backend.attn_backends):
            step_backend.data_type = torch.float32
            step_backend.q_indptr_decode = torch.arange(bs + 1)

            def kernel(*, step=step, return_lse=False, **kwargs):
                calls[step] = kwargs
                out = torch.zeros((bs, 1, 4))
                return (out, torch.zeros((bs, 1))) if return_lse else out

            step_backend._run_decode_kernel = kernel
        with self._forward_patches(parallel):
            for step_backend in backend.attn_backends:
                step_backend.forward_decode(
                    torch.zeros((bs, 4)), None, None, LAYER, batch, save_kv_cache=False
                )
        return calls

    def _extend_on(self, backend, batch, parallel):
        """Run a verify's or a draft extend's real forward and return its kernel
        arguments."""
        calls = []

        def kernel(*, query, return_lse=False, **kwargs):
            calls.append(kwargs)
            out = torch.zeros((*query.shape[:2], 1, 4))
            return (out, torch.zeros((*query.shape[:2], 1))) if return_lse else out

        backend._run_decode_kernel = kernel
        backend.q_indptr_decode = torch.arange(BS + 1)
        backend.dense_q_indptr_verify = backend.q_indptr_decode * LONG_CHAIN
        with self._forward_patches(parallel):
            backend.forward_extend(
                torch.zeros((BS * LONG_CHAIN, 4)),
                None,
                None,
                LAYER,
                batch,
                save_kv_cache=False,
            )
        (call,) = calls
        return call

    @contextmanager
    def _forward_patches(self, parallel):
        kv_layers = SimpleNamespace(
            read=lambda _id: SimpleNamespace(key=lambda: torch.zeros(1))
        )
        with (
            patch.object(trtllm_mla_backend, "get_parallel", lambda: parallel),
            patch.object(cutedsl_mla_backend, "get_parallel", lambda: parallel),
            patch.object(trtllm_mla_backend, "paged_row_view", lambda kv, _page: kv),
            patch.object(cutedsl_mla_backend, "paged_row_view", lambda kv, _page: kv),
            patch.object(trtllm_mla_backend, "fixup_zero_kv_rows", lambda *_a: None),
            patch.object(cutedsl_mla_backend, "fixup_zero_kv_rows", lambda *_a: None),
            patch.object(TRTLLMMLABackend, "kv_layers", kv_layers),
            patch.object(
                TRTLLMMLABackend, "_fill_dcp_block_kv_indices", lambda *_a: None
            ),
        ):
            yield

    def _assert_lengths(self, metadata, step):
        lens = _step_lens(step)
        torch.testing.assert_close(metadata.seq_lens_k, lens)
        torch.testing.assert_close(
            (metadata.block_kv_indices >= 0).sum(dim=1).to(torch.int32), _pages(lens)
        )

    def test_eager_steps_read_their_tokens(self):
        for backend_cls, host_lens in product(TRTLLM_FAMILY, HOST_LENS):
            backend, _ = self._backend(backend_cls)
            with patch.object(trtllm_mla_backend, "get_parallel", lambda: NO_DCP):
                backend.init_forward_metadata(_forward_batch(host_lens))
            for step, step_backend in enumerate(backend.attn_backends):
                with self.subTest(
                    backend=backend_cls.__name__, step=step, host_lens=host_lens
                ):
                    metadata = step_backend.forward_decode_metadata
                    self._assert_lengths(metadata, step)
                    longest = (
                        int(_step_lens(step).max())
                        if host_lens
                        else MAX_PAGES * PAGE_SIZE + step + 1
                    )
                    self.assertEqual(metadata.max_seq_len_k, longest)

    def test_eager_decode_reads_its_own_step(self):
        """draft() plans every step on one batch before any step decodes, so a copy the
        batch holds is the last step's: earlier steps must not read it, or they attend
        over draft rows not written yet."""
        for backend_cls, host_lens in product(TRTLLM_FAMILY, HOST_LENS):
            backend, _ = self._backend(backend_cls)
            batch = _forward_batch(host_lens)
            with patch.object(trtllm_mla_backend, "get_parallel", lambda: NO_DCP):
                backend.init_forward_metadata(batch)
            calls = self._decode_on(backend, batch, NO_DCP)
            for step in range(NUM_STEPS - 1):
                with self.subTest(
                    backend=backend_cls.__name__, step=step, host_lens=host_lens
                ):
                    lens = _step_lens(step)
                    torch.testing.assert_close(calls[step]["seq_lens"], lens)
                    torch.testing.assert_close(
                        (calls[step]["block_tables"] >= 0).sum(dim=1).to(torch.int32),
                        _pages(lens),
                    )

    def test_eager_dcp_decode_reads_its_own_step(self):
        """The same under DCP, where cutedsl_mla also has its own decode path and its
        kernel takes the global lengths as well as this rank's."""
        for backend_cls in TRTLLM_FAMILY:
            backend, _ = self._backend(backend_cls)
            batch = _forward_batch()
            with (
                patch.object(trtllm_mla_backend, "get_parallel", lambda: DCP),
                patch.object(
                    TRTLLMMLABackend, "_fill_dcp_block_kv_indices", lambda *_a: None
                ),
            ):
                backend.init_forward_metadata(batch)
            calls = self._decode_on(backend, batch, DCP)
            for step in range(NUM_STEPS - 1):
                with self.subTest(backend=backend_cls.__name__, step=step):
                    lens = _step_lens(step)
                    local = get_dcp_lens(lens, DCP_SIZE, DCP_RANK).to(torch.int32)
                    torch.testing.assert_close(calls[step]["seq_lens"], local)
                    if backend_cls is CuteDslMLAMultiStepDraftBackend:
                        torch.testing.assert_close(calls[step]["causal_seqs"], lens)

    def test_a_batch_padded_after_planning_replans_each_step(self):
        """DP padding can grow a batch after the marked plan. Each step's decode then
        replans on the grown batch, and must still count its own tokens."""
        for backend_cls, parallel in product(TRTLLM_FAMILY, (NO_DCP, DCP)):
            backend, _ = self._backend(backend_cls)
            batch = _forward_batch()
            with (
                patch.object(trtllm_mla_backend, "get_parallel", lambda: parallel),
                patch.object(
                    TRTLLMMLABackend, "_fill_dcp_block_kv_indices", lambda *_a: None
                ),
            ):
                backend.init_forward_metadata(batch)
            # What ForwardBatch padding gives: the sink row at the fill length.
            grown = torch.tensor((*COMMITTED, 1), dtype=torch.int64)
            batch.batch_size = BS + 1
            batch.seq_lens = grown
            batch.seq_lens_cpu = grown.clone()
            batch.req_pool_indices = torch.tensor((1, 2, 3, 0))
            calls = self._decode_on(backend, batch, parallel)
            for step in range(NUM_STEPS - 1):
                with self.subTest(
                    backend=backend_cls.__name__, dcp=parallel.dcp_enabled, step=step
                ):
                    lens = grown.to(torch.int32) + step + 1
                    if parallel.dcp_enabled:
                        lens = get_dcp_lens(lens, DCP_SIZE, DCP_RANK).to(torch.int32)
                    call = calls[step]
                    torch.testing.assert_close(call["seq_lens"], lens)
                    self.assertEqual(call["query"].shape[0], BS + 1)
                    self.assertEqual(call["block_tables"].shape[0], BS + 1)

    def test_batch_held_metadata_still_serves_other_decodes(self):
        """A backend outside a draft chain keeps reading the batch's copy: one backend
        can plan several batches, so its own copy may belong to another."""
        for backend_cls in (TRTLLMMLABackend, CuteDslMLABackend):
            with self.subTest(backend=backend_cls.__name__):
                backend = backend_cls(self.model_runner)
                held_lens = torch.tensor(COMMITTED, dtype=torch.int32)
                backend.forward_decode_metadata = TRTLLMMLADecodeMetadata(
                    seq_lens_k=torch.zeros(BS, dtype=torch.int32)
                )
                batch = _forward_batch()
                batch.decode_trtllm_mla_metadata = TRTLLMMLADecodeMetadata(
                    seq_lens_k=held_lens
                )
                calls = self._decode_on(
                    SimpleNamespace(attn_backends=[backend]), batch, NO_DCP
                )
                torch.testing.assert_close(calls[0]["seq_lens"], held_lens)

    def test_capture_counts_each_step_once_and_replay_agrees(self):
        """Capture reaches the steps through the shared wrapper, which adds the tokens
        for a step that does not own them. Each must be counted once on both paths."""
        for backend_cls, parallel in product(TRTLLM_FAMILY, (NO_DCP, DCP)):
            backend, graph_metadata = self._backend(backend_cls)
            backend.cuda_graph_kv_indices = torch.zeros(
                (NUM_STEPS, 1024), dtype=torch.int32
            )
            backend.generate_draft_decode_kv_indices = MagicMock()
            for in_capture in (True, False):
                with (
                    patch.object(trtllm_mla_backend, "get_parallel", lambda: parallel),
                    patch.object(
                        TRTLLMMLABackend, "_fill_dcp_block_kv_indices", lambda *_a: None
                    ),
                ):
                    backend.init_forward_metadata_out_graph(
                        _forward_batch(), in_capture=in_capture
                    )
                for step, metadata in enumerate(graph_metadata):
                    with self.subTest(
                        backend=backend_cls.__name__,
                        dcp=parallel.dcp_enabled,
                        in_capture=in_capture,
                        step=step,
                    ):
                        lengths = (
                            metadata.global_seq_lens_k
                            if parallel.dcp_enabled
                            else metadata.seq_lens_k
                        )
                        torch.testing.assert_close(lengths, _step_lens(step))

    def _capture_state(self, step_backend):
        """What the real allocator reads, on the context these bound cases use."""
        step_backend.max_context_len = CONTEXT
        step_backend.num_draft_tokens = LONG_CHAIN
        step_backend.req_to_token = torch.zeros(
            (BS + 1, CONTEXT + LONG_CHAIN), dtype=torch.int32
        )
        step_backend.decode_cuda_graph_metadata = {}
        step_backend.device = "cpu"
        step_backend.data_type = torch.float32
        step_backend.num_q_heads = 1
        step_backend.kv_cache_dim = 4
        step_backend.is_draft_runner = True
        step_backend.skip_prefill = True
        step_backend._kv_shard_pool = None

    def _graph_patches(self, parallel):
        return (
            _cuda_buffers_on_cpu(),
            patch.object(trtllm_mla_backend, "get_parallel", lambda: parallel),
            patch.object(
                TRTLLMMLABackend, "_fill_dcp_block_kv_indices", lambda *_a: None
            ),
            # Its CSR buffers are not what this family's decode reads.
            patch.object(
                FlashInferMLAAttnBackend,
                "init_cuda_graph_state",
                lambda *_a, **_k: None,
            ),
        )

    def test_replay_near_the_context_limit_fits_the_captured_bounds(self):
        """A draft step reads past the context by the tokens drafted so far, so its
        graph's table and scheduling bound, both fixed at capture, must reach them."""
        for backend_cls, parallel, committed in product(
            TRTLLM_FAMILY, (NO_DCP, DCP), COMMITTED_NEAR_LIMIT
        ):
            with self.subTest(
                backend=backend_cls.__name__,
                dcp=parallel.dcp_enabled,
                committed=committed,
            ):
                backend = backend_cls(self.model_runner, 1, LONG_CHAIN)
                backend.max_context_len = CONTEXT
                backend.generate_draft_decode_kv_indices = MagicMock()
                for step_backend in backend.attn_backends:
                    self._capture_state(step_backend)
                batch = _forward_batch(committed=(committed,) * BS)
                graph, lens, fill, no_csr = self._graph_patches(parallel)
                with graph, lens, fill, no_csr:
                    backend.init_cuda_graph_state(BS, BS * LONG_CHAIN)
                    backend.init_forward_metadata_out_graph(
                        _forward_batch(), in_capture=True
                    )
                    backend.init_forward_metadata_out_graph(batch)
                calls = self._decode_on(backend, batch, parallel)
                for step, call in calls.items():
                    expected = torch.full(
                        (BS,), committed + step + 1, dtype=torch.int32
                    )
                    if parallel.dcp_enabled:
                        expected = get_dcp_lens(expected, DCP_SIZE, DCP_RANK).to(
                            torch.int32
                        )
                    torch.testing.assert_close(call["seq_lens"], expected)
                    self.assertLessEqual(expected.max().item(), call["max_seq_len"])
                    self.assertLessEqual(
                        _pages(expected).max().item(), call["block_tables"].shape[1]
                    )
                    self.assertEqual(
                        call["block_tables"].shape[1],
                        backend.attn_backends[step]._calc_padded_blocks(
                            call["block_tables"].shape[1] * PAGE_SIZE
                        ),
                    )

    def test_a_verify_and_a_draft_extend_near_the_context_limit_fit_the_bounds(self):
        """Their drafts sit past the committed length, so the table shared with the
        decode graph must reach them, as their scheduling bounds already do."""
        for backend_cls, parallel, committed, mode in product(
            TRTLLM_TARGETS, (NO_DCP, DCP), COMMITTED_NEAR_LIMIT, SPEC_MODES
        ):
            with self.subTest(
                backend=backend_cls.__name__,
                dcp=parallel.dcp_enabled,
                committed=committed,
                mode=mode.name,
            ):
                backend = backend_cls(self.model_runner)
                self._capture_state(backend)
                backend.is_draft_runner = mode.is_draft_extend_v2()
                batch = _spec_batch(committed, mode)
                graph, lens, fill, no_csr = self._graph_patches(parallel)
                with graph, lens, fill, no_csr:
                    backend.init_cuda_graph_state(BS, BS * LONG_CHAIN)
                    backend.init_forward_metadata_out_graph(
                        _spec_batch(1, mode), in_capture=True
                    )
                    backend.init_forward_metadata_out_graph(batch)
                call = self._extend_on(backend, batch, parallel)
                expected = torch.full((BS,), committed + LONG_CHAIN, dtype=torch.int32)
                if parallel.dcp_enabled:
                    expected = get_dcp_lens(expected, DCP_SIZE, DCP_RANK).to(
                        torch.int32
                    )
                torch.testing.assert_close(call["seq_lens"], expected)
                self.assertLessEqual(expected.max().item(), call["max_seq_len"])
                self.assertLessEqual(
                    _pages(expected).max().item(), call["block_tables"].shape[1]
                )

    def test_a_backend_without_a_step_keeps_its_captured_bounds(self):
        """Only a draft step reads past the context: an ordinary decode's graph keeps
        the context, and a verify's rank-local bound its own draft tail. Without
        speculation, the shared table keeps the context too."""
        for backend_cls in TRTLLM_TARGETS:
            with self.subTest(backend=backend_cls.__name__, speculative=False):
                backend = backend_cls(self.model_runner)
                self._capture_state(backend)
                backend.num_draft_tokens = None
                graph, lens, fill, no_csr = self._graph_patches(NO_DCP)
                with graph, lens, fill, no_csr:
                    backend.init_cuda_graph_state(BS, BS)
                self.assertEqual(
                    backend.decode_cuda_graph_kv_indices.shape[1],
                    backend._calc_padded_blocks(CONTEXT),
                )
        for backend_cls, parallel in product(TRTLLM_TARGETS, (NO_DCP, DCP)):
            with self.subTest(backend=backend_cls.__name__, dcp=parallel.dcp_enabled):
                backend = backend_cls(self.model_runner)
                self._capture_state(backend)
                graph, lens, fill, no_csr = self._graph_patches(parallel)
                with graph, lens, fill, no_csr:
                    backend.init_cuda_graph_state(BS, BS * LONG_CHAIN)
                    metadata = {}
                    for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
                        backend._init_cuda_graph_metadata(
                            BS, BS, mode, torch.zeros(BS, dtype=torch.int64), "cpu"
                        )
                        metadata[mode] = backend.forward_decode_metadata
                    local = backend._get_dcp_local_max_seq_len
                    decode = metadata[ForwardMode.DECODE]
                    self.assertEqual(decode.max_seq_len_k, local(CONTEXT))
                    self.assertEqual(
                        decode.block_kv_indices.shape[1],
                        backend._calc_padded_blocks(CONTEXT),
                    )
                    self.assertEqual(
                        metadata[ForwardMode.TARGET_VERIFY].max_seq_len_k,
                        local(CONTEXT + LONG_CHAIN)
                        if parallel.dcp_enabled
                        else CONTEXT,
                    )

    def test_graph_replay_steps_read_their_tokens(self):
        for backend_cls, host_lens in product(TRTLLM_FAMILY, HOST_LENS):
            backend, graph_metadata = self._backend(backend_cls)
            with patch.object(trtllm_mla_backend, "get_parallel", lambda: NO_DCP):
                backend.init_forward_metadata_out_graph(_forward_batch(host_lens))
            for step, metadata in enumerate(graph_metadata):
                with self.subTest(
                    backend=backend_cls.__name__, step=step, host_lens=host_lens
                ):
                    self._assert_lengths(metadata, step)

    def test_dcp_graph_replay_steps_read_their_tokens(self):
        for backend_cls in TRTLLM_FAMILY:
            backend, graph_metadata = self._backend(backend_cls)
            filled = []
            with (
                patch.object(trtllm_mla_backend, "get_parallel", lambda: DCP),
                patch.object(
                    TRTLLMMLABackend,
                    "_fill_dcp_block_kv_indices",
                    lambda _self, _table, _rows, lens: filled.append(lens.clone()),
                ),
            ):
                backend.init_forward_metadata_out_graph(_forward_batch())
            for step, metadata in enumerate(graph_metadata):
                with self.subTest(backend=backend_cls.__name__, step=step):
                    lens = _step_lens(step)
                    local = get_dcp_lens(lens, DCP_SIZE, DCP_RANK).to(torch.int32)
                    torch.testing.assert_close(metadata.global_seq_lens_k, lens)
                    torch.testing.assert_close(metadata.seq_lens_k, local)
                    torch.testing.assert_close(filled[step], local)


class TestFlashMLADraftStepLengths(_MultiStepCase):
    """flashmla, each step read back from the kernel call a decode makes."""

    def setUp(self):
        super().setUp()
        kv_layers = SimpleNamespace(
            read=lambda _id: SimpleNamespace(key=lambda: torch.zeros(1))
        )
        patcher = patch.object(FlashMLABackend, "kv_layers", kv_layers)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.scheduler_lens = {}
        self.kernel_calls = {}

    def _backend(self, num_steps=NUM_STEPS):
        backend = FlashMLAMultiStepDraftBackend(self.model_runner, 1, num_steps)
        for step, step_backend in enumerate(backend.attn_backends):
            self._stub_kernels(step_backend, step)
        return backend

    def _stub_kernels(self, step_backend, step):
        step_backend.num_q_heads = 1
        step_backend.kv_lora_rank = 4
        step_backend.is_fp8_kvcache = False
        step_backend.dcp_world_size = 1
        step_backend.device = "cpu"
        step_backend._eager_kv_indices_buf = None
        step_backend._call_get_mla_metadata = self._scheduler_metadata(step)
        step_backend._call_flash_mla_with_kvcache = self._decode_kernel(step)
        step_backend.cuda_graph_kv_indices = torch.full(
            (BS, MAX_PAGES), -1, dtype=torch.int32
        )
        step_backend.cuda_graph_mla_metadata = torch.zeros((1, 8), dtype=torch.int32)
        step_backend.cuda_graph_num_splits = torch.zeros(BS + 1, dtype=torch.int32)
        step_backend.cuda_graph_mla_metadata_view = None
        step_backend.cuda_graph_num_splits_view = None
        step_backend.cuda_graph_draft_extend_seq_lens_k = torch.ones(
            BS, dtype=torch.int32
        )

    def _scheduler_metadata(self, step):
        def call(seq_lens, *_a, **_k):
            self.scheduler_lens[step] = seq_lens.clone()
            return torch.zeros((1, 8), dtype=torch.int32), torch.zeros(
                BS + 1, dtype=torch.int32
            )

        return call

    def _decode_kernel(self, step):
        def call(**kwargs):
            self.kernel_calls[step] = kwargs
            return torch.zeros((BS, 1, 4)), None

        return call

    def _assert_decodes_read_their_tokens(self, backend):
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=1,
            head_dim=4,
            v_head_dim=4,
            scaling=1.0,
            k_scale=None,
        )
        for step, step_backend in enumerate(backend.attn_backends):
            for is_fp8 in (False, True):
                with self.subTest(step=step, is_fp8=is_fp8):
                    step_backend.is_fp8_kvcache = is_fp8
                    step_backend.forward_decode(
                        torch.zeros((BS, 4)), None, None, layer, _forward_batch()
                    )
                    call = self.kernel_calls[step]
                    lens = _step_lens(step)
                    torch.testing.assert_close(call["cache_seqlens"], lens)
                    torch.testing.assert_close(self.scheduler_lens[step], lens)
                    torch.testing.assert_close(
                        (call["block_table"] >= 0).sum(dim=1).to(torch.int32),
                        _pages(lens),
                    )

    def test_eager_steps_read_their_tokens(self):
        for host_lens in HOST_LENS:
            backend = self._backend()
            backend.init_forward_metadata(_forward_batch(host_lens))
            self._assert_decodes_read_their_tokens(backend)

    def test_graph_replay_steps_read_their_tokens(self):
        for host_lens in HOST_LENS:
            backend = self._backend()
            backend.init_forward_metadata_out_graph(_forward_batch(host_lens))
            self._assert_decodes_read_their_tokens(backend)

    def test_replay_near_the_context_limit_fits_the_captured_table(self):
        """The table reserves a page past the context, which a page-aligned context's
        tail fits in; a context ending just short of a page needs the step's tokens."""
        for context in (CONTEXT, CONTEXT - 1):
            for committed in (context - 6, context - 2):
                with self.subTest(context=context, committed=committed):
                    backend = self._backend(LONG_CHAIN)
                    for step_backend in backend.attn_backends:
                        step_backend.max_context_len = context
                        step_backend.num_draft_tokens = LONG_CHAIN
                        step_backend.req_to_token = torch.zeros(
                            (BS + 1, context + LONG_CHAIN), dtype=torch.int32
                        )
                        step_backend.is_draft_runner = True
                        step_backend.skip_prefill = True
                    batch = _forward_batch(committed=(committed,) * BS)
                    with (
                        _cuda_buffers_on_cpu(),
                        patch.object(
                            torch.cuda,
                            "get_device_properties",
                            lambda _device: SimpleNamespace(multi_processor_count=1),
                        ),
                    ):
                        backend.init_cuda_graph_state(BS, BS * LONG_CHAIN)
                        backend.init_forward_metadata_out_graph(batch)
                    for step, step_backend in enumerate(backend.attn_backends):
                        length = torch.tensor(committed + step + 1)
                        self.assertLessEqual(
                            _pages(length).item(),
                            step_backend.cuda_graph_kv_indices.shape[1],
                        )

    def test_a_verify_and_a_draft_extend_near_the_context_limit_fit_the_table(self):
        """Their drafts sit past the committed length; the page the table reserves past
        the context covers them only on a page-aligned context."""
        layer = SimpleNamespace(
            layer_id=0, tp_q_head_num=1, head_dim=4, v_head_dim=4, scaling=1.0
        )
        for context, mode, host_lens in product(
            (CONTEXT, CONTEXT - 1), SPEC_MODES, HOST_LENS
        ):
            for committed in (context - 6, context - 2):
                with self.subTest(
                    context=context,
                    committed=committed,
                    mode=mode.name,
                    host_lens=host_lens,
                ):
                    backend = FlashMLABackend(self.model_runner)
                    self._stub_kernels(backend, 0)
                    backend.max_context_len = context
                    backend.num_draft_tokens = LONG_CHAIN
                    backend.req_to_token = torch.zeros(
                        (BS + 1, context + LONG_CHAIN), dtype=torch.int32
                    )
                    backend.is_draft_runner = mode.is_draft_extend_v2()
                    backend.skip_prefill = True
                    batch = _spec_batch(committed, mode, host_lens)
                    with (
                        _cuda_buffers_on_cpu(),
                        patch.object(
                            torch.cuda,
                            "get_device_properties",
                            lambda _device: SimpleNamespace(multi_processor_count=1),
                        ),
                    ):
                        backend.init_cuda_graph_state(BS, BS * LONG_CHAIN)
                        backend.init_forward_metadata_out_graph(
                            _spec_batch(1, mode), in_capture=True
                        )
                        backend.init_forward_metadata_out_graph(batch)
                    backend.forward_extend(
                        torch.zeros((BS * LONG_CHAIN, 4)), None, None, layer, batch
                    )
                    call = self.kernel_calls[0]
                    lens = torch.full((BS,), committed + LONG_CHAIN, dtype=torch.int32)
                    torch.testing.assert_close(call["cache_seqlens"], lens)
                    torch.testing.assert_close(
                        (call["block_table"] >= 0).sum(dim=1).to(torch.int32),
                        _pages(lens),
                    )


class _DraftCsrKernel:
    """generate_draft_decode_kv_indices' contract: each step's CSR spans the committed
    prefix plus that step's draft tokens."""

    def __getitem__(self, _grid):
        return self

    def __call__(self, _rows, _ids, committed, _indices, indptr, *_a, **_k):
        for step in range(NUM_STEPS):
            lens = committed.to(torch.int32) + step + 1
            indptr[step, 0] = 0
            indptr[step, 1 : BS + 1] = torch.cumsum(lens, dim=0)


class TestFlashInferMLADraftStepLengths(_MultiStepCase):
    def test_eager_steps_plan_the_prepared_csr(self):
        """Eager plans the lengths of the CSR the wrapper prepared, not the committed
        `seq_lens` its steps are also given. The real planner runs; only its wrapper's
        plan call and the wrapper's CUDA allocation are stood in for."""
        backend = FlashInferMLAMultiStepDraftBackend(self.model_runner, 1, NUM_STEPS)
        backend.generate_draft_decode_kv_indices = _DraftCsrKernel()
        planned = {}
        for step, step_backend in enumerate(backend.attn_backends):
            step_backend.decode_wrapper = SimpleNamespace(
                plan=lambda _q, indptr, _indices, lens, *_a, step=step: (
                    planned.__setitem__(step, (indptr.clone(), lens.clone()))
                )
            )
            updater = object.__new__(
                flashinfer_mla_backend.FlashInferMLAIndicesUpdaterDecode
            )
            updater.attn_backend = step_backend
            updater.kv_indptr = torch.zeros(BS + 1, dtype=torch.int32)
            updater.q_indptr = torch.arange(BS + 1, dtype=torch.int32)
            updater.scaling = 1.0
            updater.num_local_heads = 1
            updater.kv_lora_rank = 4
            updater.qk_rope_head_dim = 4
            updater.data_type = torch.float32
            step_backend.indices_updater_decode = updater
        real_zeros = torch.zeros
        with patch.object(
            torch,
            "zeros",
            lambda *args, device=None, **kwargs: real_zeros(
                *args, device="cpu", **kwargs
            ),
        ):
            backend.init_forward_metadata(_forward_batch())
        for step in range(NUM_STEPS - 1):
            with self.subTest(step=step):
                indptr, lens = planned[step]
                torch.testing.assert_close(lens, _step_lens(step))
                torch.testing.assert_close(indptr[1:] - indptr[:-1], _step_lens(step))

    def test_graph_replay_plans_each_step_once(self):
        """Its multi-step backend passes each step lengths that hold the draft tokens,
        so a step adding them again would misplace later requests' packed segments."""
        backend = FlashInferMLAMultiStepDraftBackend(self.model_runner, 1, NUM_STEPS)
        backend.cuda_graph_kv_indices = torch.zeros((NUM_STEPS, 512), dtype=torch.int32)
        backend.generate_draft_decode_kv_indices = MagicMock()
        planned = {}
        for step, step_backend in enumerate(backend.attn_backends):
            step_backend.decode_cuda_graph_metadata = {BS: object()}
            step_backend.cuda_graph_kv_indptr_cpu = torch.zeros(
                BS + 1, dtype=torch.int32
            )
            step_backend.cuda_graph_qo_indptr_cpu = torch.arange(BS + 1)
            step_backend.fast_decode_kwargs = {}
            step_backend.indices_updater_decode = SimpleNamespace(
                update=lambda seq_lens, seq_lens_sum, step=step, **kwargs: (
                    planned.__setitem__(
                        step, (kwargs["kv_len_arr_cpu"].clone(), seq_lens, seq_lens_sum)
                    )
                )
            )
        with patch.object(
            flashinfer_mla_backend, "update_local_kv_lens_for_dcp", lambda _: None
        ):
            backend.init_forward_metadata_out_graph(_forward_batch())
        for step in range(NUM_STEPS - 1):
            with self.subTest(step=step):
                host_lens, seq_lens, seq_lens_sum = planned[step]
                torch.testing.assert_close(host_lens, _step_lens(step))
                torch.testing.assert_close(seq_lens.to(torch.int32), _step_lens(step))
                self.assertEqual(seq_lens_sum, int(_step_lens(step).sum()))


if __name__ == "__main__":
    unittest.main()
