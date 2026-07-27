"""Two-batch-overlap split math for the draft-extend forward mode (issue #7892).

Draft-extend (`DRAFT_EXTEND_V2`) fills a uniform `num_tokens_per_req` slots per
req, so it splits on a clean seq boundary at `bs // 2` exactly like
`TARGET_VERIFY` -- not like variable-length `EXTEND`. That uniform-width shape is
what lets the split stay well-defined at cuda-graph replay time (where
`extend_lens` is unavailable). CPU-only.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.batch_overlap import operations
from sglang.srt.batch_overlap.operations_strategy import (
    _compute_moe_deepseek_blog_decode,
    _compute_moe_deepseek_draft_extend,
)
from sglang.srt.batch_overlap.two_batch_overlap import (
    TboForwardBatchPreparer,
    compute_split_indices_for_cuda_graph_replay,
    compute_split_seq_index,
    compute_split_token_index,
    get_token_num_per_seq,
    op_capture_dsa_seed_topk,
    split_spec_info,
)
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _draft_extend_spec(num_tokens_per_req: int) -> SimpleNamespace:
    # get_token_num_per_seq only reads `.num_tokens_per_req` off the spec info.
    return SimpleNamespace(num_tokens_per_req=num_tokens_per_req)


# (batch_size, num_draft_tokens) pairs covering odd/even bs and topk-1/tree width.
_CASES = [(1, 1), (2, 4), (4, 8), (6, 8), (7, 2), (8, 3), (31, 4), (64, 8)]


class TestTboDraftExtendSplit(CustomTestCase):
    def test_get_token_num_per_seq_reads_num_tokens_per_req(self):
        for _, width in _CASES:
            spec = _draft_extend_spec(width)
            self.assertEqual(
                get_token_num_per_seq(ForwardMode.DRAFT_EXTEND_V2, spec), width
            )

    def test_split_seq_index_is_half_the_batch(self):
        # Uniform width => split on a seq boundary at bs // 2, independent of width.
        for bs, width in _CASES:
            num_tokens = bs * width
            got = compute_split_seq_index(
                ForwardMode.DRAFT_EXTEND_V2,
                num_tokens=num_tokens,
                extend_lens=None,
                token_num_per_seq=width,
            )
            self.assertEqual(got, bs // 2, msg=f"{bs=} {width=}")

    def test_split_token_index_matches_seq_boundary(self):
        # The token cut must land exactly on child_a's seq boundary so the two
        # micro-batches partition the flat token buffer with no overlap/gap.
        for bs, width in _CASES:
            num_tokens = bs * width
            ssi = compute_split_seq_index(
                ForwardMode.DRAFT_EXTEND_V2,
                num_tokens=num_tokens,
                extend_lens=None,
                token_num_per_seq=width,
            )
            sti = compute_split_token_index(
                ssi,
                ForwardMode.DRAFT_EXTEND_V2,
                extend_seq_lens=None,
                token_num_per_seq=width,
            )
            self.assertEqual(sti, ssi * width, msg=f"{bs=} {width=}")
            # child_a = [0, sti), child_b = [sti, num_tokens): a clean partition.
            self.assertGreaterEqual(sti, 0)
            self.assertLessEqual(sti, num_tokens)

    def test_parity_with_target_verify(self):
        # Design invariant: draft-extend and target-verify share the uniform-width
        # split. Same (bs, width) must yield identical seq/token cuts.
        for bs, width in _CASES:
            num_tokens = bs * width
            de_seq = compute_split_seq_index(
                ForwardMode.DRAFT_EXTEND_V2,
                num_tokens=num_tokens,
                extend_lens=None,
                token_num_per_seq=width,
            )
            tv_seq = compute_split_seq_index(
                ForwardMode.TARGET_VERIFY,
                num_tokens=num_tokens,
                extend_lens=None,
                token_num_per_seq=width,
            )
            self.assertEqual(de_seq, tv_seq, msg=f"{bs=} {width=}")

    def test_cuda_graph_replay_indices_without_extend_lens(self):
        # The replay path passes extend_lens=None and relies solely on
        # token_num_per_seq; draft-extend must resolve to concrete indices there.
        for bs, width in _CASES:
            spec = _draft_extend_spec(width)
            ssi, sti = compute_split_indices_for_cuda_graph_replay(
                forward_mode=ForwardMode.DRAFT_EXTEND_V2,
                cuda_graph_num_tokens=bs * width,
                spec_info=spec,
            )
            self.assertEqual(ssi, bs // 2, msg=f"{bs=} {width=}")
            self.assertEqual(sti, (bs // 2) * width, msg=f"{bs=} {width=}")


def _make_draft_extend_spec_info() -> EagleDraftExtendInput:
    # bs=5, width=4 -> 20 flat token rows. Values are all distinct so a
    # mis-slice cannot accidentally reproduce the parent.
    return EagleDraftExtendInput(
        hidden_states=torch.arange(60, dtype=torch.float32).reshape(20, 3),
        num_correct_drafts=torch.tensor([0, 1, 2, 3, 0]),
        num_accept_tokens=torch.tensor([1, 2, 3, 4, 1]),
        num_accept_tokens_cpu=[1, 2, 3, 4, 1],
        input_ids=torch.arange(20),
        seq_lens=torch.tensor([10, 20, 30, 40, 50]),
        seq_lens_cpu=torch.tensor([10, 20, 30, 40, 50]),
        req_pool_indices=torch.tensor([7, 8, 9, 10, 11]),
        positions=torch.arange(100, 120),
        bonus_tokens=torch.tensor([5, 6, 7, 8, 9]),
        capture_hidden_mode=CaptureHiddenMode.LAST,
        num_tokens_per_req=4,
        num_tokens_for_logprob_per_req=4,
        dsa_seed_topk_capture=torch.zeros(20, 2, dtype=torch.int64),
        dsa_seed_topk_select=torch.tensor([3, 7, 11, 15, 19]),
        select_index=torch.tensor([0, 5, 10, 15, 16]),
        kv_indptr=torch.tensor([0, 3, 7, 12, 18, 25]),
    )


def _split_in_two(spec: EagleDraftExtendInput, ssi: int, width: int):
    bs = spec.num_accept_tokens.shape[0]
    child_a = split_spec_info(spec, 0, ssi, 0, ssi * width)
    child_b = split_spec_info(spec, ssi, bs, ssi * width, bs * width)
    return child_a, child_b


class TestDraftExtendSpecInfoSplit(CustomTestCase):
    """Round-trip contract of the EagleDraftExtendInput splitter (P0-1).

    Derived property: the two children must partition the parent losslessly
    (concat(childA, childB) == parent), with flat-row pointers rebased into
    each child's token coordinates. A "looks equivalent" rewrite that slices
    one field on the wrong ruler silently corrupts requests; these cases are
    the only guard.
    """

    def test_roundtrip_partitions_every_field(self):
        parent = _make_draft_extend_spec_info()
        child_a, child_b = _split_in_two(parent, ssi=2, width=4)

        for name in ("input_ids", "positions", "hidden_states"):
            merged = torch.cat([getattr(child_a, name), getattr(child_b, name)])
            self.assertTrue(torch.equal(merged, getattr(parent, name)), name)
        for name in (
            "num_correct_drafts",
            "num_accept_tokens",
            "seq_lens",
            "seq_lens_cpu",
            "req_pool_indices",
            "bonus_tokens",
        ):
            merged = torch.cat([getattr(child_a, name), getattr(child_b, name)])
            self.assertTrue(torch.equal(merged, getattr(parent, name)), name)
        self.assertEqual(
            child_a.num_accept_tokens_cpu + child_b.num_accept_tokens_cpu,
            parent.num_accept_tokens_cpu,
        )
        # Scalars carry over; the child is a first-class spec input.
        for child in (child_a, child_b):
            self.assertEqual(child.num_tokens_per_req, 4)
            self.assertEqual(child.capture_hidden_mode, CaptureHiddenMode.LAST)
            self.assertEqual(child.spec_input_type, parent.spec_input_type)

    def test_flat_row_pointers_rebase_to_child_origin(self):
        parent = _make_draft_extend_spec_info()
        child_a, child_b = _split_in_two(parent, ssi=2, width=4)

        # kv_indptr: bs+1 cumulative entries, child origin rebased to zero.
        self.assertTrue(torch.equal(child_a.kv_indptr, torch.tensor([0, 3, 7])))
        self.assertTrue(torch.equal(child_b.kv_indptr, torch.tensor([0, 5, 11, 18])))
        # select_index / dsa_seed_topk_select: per-req flat-row pointers must
        # shift by the child's token offset (8), or child B reads child A rows.
        self.assertTrue(torch.equal(child_a.select_index, torch.tensor([0, 5])))
        self.assertTrue(torch.equal(child_b.select_index, torch.tensor([2, 7, 8])))
        self.assertTrue(torch.equal(child_a.dsa_seed_topk_select, torch.tensor([3, 7])))
        self.assertTrue(
            torch.equal(child_b.dsa_seed_topk_select, torch.tensor([3, 7, 11]))
        )

    def test_capture_buffer_stays_a_view_of_parent(self):
        # The DSA seed capture buffer is an OUTPUT: children must write into
        # the parent's rows. If a rewrite turns the slice into a copy, seeds
        # silently stop reaching the next draft loop (no crash, wrong decode).
        parent = _make_draft_extend_spec_info()
        child_a, child_b = _split_in_two(parent, ssi=2, width=4)
        child_b.dsa_seed_topk_capture.fill_(7)
        self.assertTrue(torch.all(parent.dsa_seed_topk_capture[8:] == 7))
        self.assertTrue(torch.all(parent.dsa_seed_topk_capture[:8] == 0))
        child_a.dsa_seed_topk_capture.fill_(3)
        self.assertTrue(torch.all(parent.dsa_seed_topk_capture[:8] == 3))

    def test_rejects_token_cut_off_the_seq_boundary(self):
        # Fixed width means the token knife and the request knife are the same
        # knife. A drifted caller (e.g. a ragged token cut) must fail loudly.
        parent = _make_draft_extend_spec_info()
        with self.assertRaises(AssertionError):
            split_spec_info(parent, 0, 2, 0, 7)

    def test_rejects_unset_width(self):
        # width=-1 (prefill-flavored or default-constructed spec) has no
        # defined split; slicing with it would corrupt instead of crash.
        with self.assertRaises(AssertionError):
            split_spec_info(EagleDraftExtendInput(), 0, 0, 0, 0)

    def test_rejects_widened_spec(self):
        # Widening (num_front_tokens > 0) changes the real per-req row count
        # away from num_tokens_per_req; fixed-width slicing would silently
        # mis-slice, so the splitter must refuse until it is widening-aware.
        parent = _make_draft_extend_spec_info()
        parent.num_front_tokens = 2
        with self.assertRaises(AssertionError):
            _split_in_two(parent, ssi=2, width=4)

    def test_unclassified_field_fails_loudly(self):
        # Bookkeeping guard: adding a field to EagleDraftExtendInput without
        # assigning it a split rule must crash the split, not silently drop
        # or mis-slice the new field.
        parent = _make_draft_extend_spec_info()
        parent.mystery_field = torch.zeros(2)
        with self.assertRaises(AssertionError) as ctx:
            _split_in_two(parent, ssi=2, width=4)
        self.assertIn("unclassified", str(ctx.exception))


class TestDraftExtendChildrenGate(CustomTestCase):
    """Draft-worker TBO gate (#7892).

    The gate decides which draft forwards split into two micro-batches. Its
    negative branches are load-bearing: a gate that degrades to always-true
    splits the draft decode loop (or the prefill-side draft-extend) and
    desynchronizes the per-rank EP collective schedule.
    """

    @staticmethod
    def _gate(forward_mode, spec_info, attn_backend, global_forward_mode):
        return TboForwardBatchPreparer._should_prepare_draft_extend_children(
            forward_mode=forward_mode,
            global_forward_mode=global_forward_mode,
            spec_info=spec_info,
            attn_backend=attn_backend,
        )

    def setUp(self):
        self.tbo_backend = MagicMock(spec=TboAttnBackend)
        self.extend_input = MagicMock(spec=EagleDraftExtendInput)

    def test_opens_for_decode_step_draft_extend(self):
        self.assertTrue(
            self._gate(
                ForwardMode.DRAFT_EXTEND_V2,
                self.extend_input,
                self.tbo_backend,
                ForwardMode.DECODE,
            )
        )

    def test_opens_for_decode_step_idle_rank(self):
        # P0-3: the idle rank must split ([0, 0] children) so every DP rank
        # runs the same two dispatch/combine collectives as active ranks.
        self.assertTrue(
            self._gate(
                ForwardMode.IDLE,
                self.extend_input,
                self.tbo_backend,
                ForwardMode.DECODE,
            )
        )

    def test_closed_without_tbo_backend(self):
        self.assertFalse(
            self._gate(
                ForwardMode.DRAFT_EXTEND_V2,
                self.extend_input,
                MagicMock(),
                ForwardMode.DECODE,
            )
        )

    def test_closed_for_draft_decode_loop(self):
        # The draft multi-step decode loop (mode DECODE) and non-draft-extend
        # spec inputs must stay monolithic even though the same TboAttnBackend
        # is installed for the whole draft runner.
        self.assertFalse(
            self._gate(
                ForwardMode.DECODE,
                self.extend_input,
                self.tbo_backend,
                ForwardMode.DECODE,
            )
        )
        self.assertFalse(
            self._gate(
                ForwardMode.DRAFT_EXTEND_V2,
                MagicMock(),
                self.tbo_backend,
                ForwardMode.DECODE,
            )
        )
        self.assertFalse(
            self._gate(
                ForwardMode.DRAFT_EXTEND_V2, None, self.tbo_backend, ForwardMode.DECODE
            )
        )

    def test_closed_for_prefill_step_idle_rank(self):
        """Bug regression (#7892 review finding F2).

        Prefill step with target TBO active: active ranks run the prefill-side
        draft-extend under mode EXTEND (gate closed, ONE collective schedule),
        while an idle DP rank's batch arrives as mode IDLE with the same
        EagleDraftExtendInput spec, the same wrapped backend, and a
        tbo_split_seq_index inherited from the target prefill vote. If the
        idle arm opens outside the decode phase, the idle rank executes TWO
        dispatch/combine rounds against the active ranks' one, and the EP
        all-to-all deadlocks.

        Contract pinned here: the idle arm opens only under a voted DECODE
        ticket — never for another phase's ticket, and never when the vote
        produced no ticket at all. Verified red against the pre-fix gate
        (which could not receive the phase signal).
        """
        self.assertFalse(
            self._gate(
                ForwardMode.IDLE,
                self.extend_input,
                self.tbo_backend,
                ForwardMode.EXTEND,
            )
        )
        self.assertFalse(
            self._gate(ForwardMode.IDLE, self.extend_input, self.tbo_backend, None)
        )


class TestDraftExtendOperationsStrategy(CustomTestCase):
    """Op-sequence invariants of the draft-extend strategy (P0-2).

    The capture op must sit in the same stage as the attn core (its state key
    is consumed before op_comm_postprocess_layer's state.clear), and building
    the draft strategy must not corrupt the shared decode strategy.
    """

    def test_swaps_core_and_inserts_capture_in_same_stage(self):
        layer = MagicMock()
        base = _compute_moe_deepseek_blog_decode(layer)
        draft = _compute_moe_deepseek_draft_extend(layer)

        idx = base.operations.index(layer.self_attn.op_core)
        self.assertIs(draft.operations[idx], layer.self_attn.op_core_with_topk)
        self.assertIs(draft.operations[idx + 1], op_capture_dsa_seed_topk)
        self.assertNotIn(layer.self_attn.op_core, draft.operations)

        # Everything around the swap point is untouched, so the capture op is
        # adjacent to the core (same stage) and stage boundaries are preserved.
        # YieldOperation instances are fresh per builder call, so compare them
        # by type rather than identity.
        def _normalize(ops):
            return [
                (
                    operations.YieldOperation
                    if isinstance(op, operations.YieldOperation)
                    else op
                )
                for op in ops
            ]

        self.assertEqual(
            _normalize(draft.operations[:idx]), _normalize(base.operations[:idx])
        )
        self.assertEqual(
            _normalize(draft.operations[idx + 2 :]),
            _normalize(base.operations[idx + 1 :]),
        )
        self.assertEqual(
            sum(isinstance(op, operations.YieldOperation) for op in draft.operations),
            sum(isinstance(op, operations.YieldOperation) for op in base.operations),
        )
        self.assertEqual(draft.tbo_delta_stages, base.tbo_delta_stages)
        self.assertEqual(draft.deep_gemm_num_sms, base.deep_gemm_num_sms)

    def test_does_not_mutate_the_decode_strategy(self):
        # The draft builder derives from the decode op list; an in-place
        # insert would poison every later decode/target-verify strategy.
        layer = MagicMock()
        base = _compute_moe_deepseek_blog_decode(layer)
        ops_before = list(base.operations)
        _compute_moe_deepseek_draft_extend(layer)
        self.assertEqual(base.operations, ops_before)


class _FakeOpState(dict):
    """Minimal operations-state stand-in: dict pop + forward_batch attr."""

    forward_batch = None


class TestCaptureDsaSeedTopkOp(CustomTestCase):
    """op_capture_dsa_seed_topk mirrors the monolithic NextN tail write."""

    @staticmethod
    def _state(topk_indices, spec_info, reuse=False):
        state = _FakeOpState(topk_indices_after_attn=topk_indices)
        state.forward_batch = SimpleNamespace(
            spec_info=spec_info, reuse_dsa_topk_indices=reuse
        )
        return state

    def test_writes_rows_into_the_per_token_buffer(self):
        # Decode-path contract: select is None, the buffer is per-token sized,
        # and the write is a plain row copy — which is exactly what makes a
        # per-child view write land like the monolithic write.
        topk = torch.arange(12).reshape(4, 3)
        buf = torch.zeros(8, 3, dtype=torch.int64)
        spec = SimpleNamespace(dsa_seed_topk_capture=buf, dsa_seed_topk_select=None)
        op_capture_dsa_seed_topk(self._state(topk, spec))
        self.assertTrue(torch.equal(buf[:4], topk))
        self.assertTrue(torch.all(buf[4:] == 0))

    def test_prefill_select_layout_is_refused(self):
        # A non-None select implies the prefill (bs, k) capture layout, whose
        # write targets a TBO split does not preserve — the op must fail
        # loudly instead of scattering seeds to wrong rows.
        topk = torch.arange(12).reshape(4, 3)
        spec = SimpleNamespace(
            dsa_seed_topk_capture=torch.zeros(8, 3, dtype=torch.int64),
            dsa_seed_topk_select=torch.tensor([1, 3]),
        )
        with self.assertRaises(AssertionError):
            op_capture_dsa_seed_topk(self._state(topk, spec))

    def test_noop_without_capture_request(self):
        # Non-DSA batches (no seed buffer) and topk-less cores must both be
        # silent no-ops; the state key is still consumed either way.
        topk = torch.arange(12).reshape(4, 3)
        state = self._state(topk, SimpleNamespace(dsa_seed_topk_capture=None))
        op_capture_dsa_seed_topk(state)
        self.assertNotIn("topk_indices_after_attn", state)
        op_capture_dsa_seed_topk(
            self._state(None, SimpleNamespace(dsa_seed_topk_capture=torch.zeros(2, 2)))
        )
        op_capture_dsa_seed_topk(self._state(topk, None))

    def test_reuse_channel_is_refused(self):
        # The op deliberately does not implement the dsa_topk_indices
        # write-back; a batch arriving with the reuse flag set means a caller
        # wired the wrong phase through it and must fail loudly.
        topk = torch.arange(12).reshape(4, 3)
        spec = SimpleNamespace(
            dsa_seed_topk_capture=torch.zeros(4, 3, dtype=torch.int64),
            dsa_seed_topk_select=None,
        )
        with self.assertRaises(AssertionError):
            op_capture_dsa_seed_topk(self._state(topk, spec, reuse=True))


if __name__ == "__main__":
    unittest.main()
