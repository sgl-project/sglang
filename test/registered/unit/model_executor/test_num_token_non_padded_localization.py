"""Per-rank-local non-padded token count across attn x MoE parallelism layouts.

``compute_local_num_token_non_padded`` (GPU tensor) and
``compute_local_num_token_non_padded_cpu`` (host int) convert a dp-group-global
real-token count into this attention-TP rank's local count. Each rank owns a
contiguous ``padded_bucket // attn_tp_size`` slice of the padded sequence, so the
localizer clamps ``real - chunk * attn_tp_rank`` into ``[0, chunk]``: a replicated
(non-sharded) rank keeps the full count and SP ranks split it. The value is
identical whether the MoE runs TP or EP -- it is an attention-side quantity both
backends consume. This table locks the exact per-rank counts and that the GPU
tensor and host-int twin agree, so a change to the sharding math fails loudly.

``ForwardBatch.moe_num_token_non_padded()`` decides whether that count may bound
a sparse MoE's input at all. Masking a gathered buffer with it truncates every
peer rank's rows, which collapsed gsm8k accuracy under DP attention with
``--moe-a2a-backend none``, where each rank sums its own partial output. The
second table locks that MoE input x layout x CP decision.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import sglang.srt.models.deepseek_v4_replay_tail as replay_tail
from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    compute_local_num_token_non_padded,
    compute_local_num_token_non_padded_cpu,
    enable_num_token_non_padded,
)
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    get_forward_context,
    has_forward_context,
)
from sglang.srt.model_executor.runner.row_bucket_graph_runner import (
    RowBucketGraphRunner,
)
from sglang.srt.runtime_context import get_flags, get_parallel
from sglang.test.communicator_patch import patch_communicator
from sglang.test.test_utils import CustomTestCase


@contextmanager
def sparse_moe_input(rows):
    """Where a sparse MoE's input is: on each rank's local rows ("local": a2a
    dispatch, FP4 all-gather or dwdp), on rows a GQA prefill CP gathers over the
    MoE-CP group ("moe_cp", on a CP extend only), or on the FFN's rows ("ffn")."""
    local = rows == "local"
    with (
        patch.object(
            moe_utils, "is_moe_input_scattered_across_dp_ranks", return_value=local
        ),
        patch_communicator(
            "is_moe_input_scattered_across_dp_ranks", return_value=local
        ),
        patch_communicator("is_enable_moe_cp_allgather", return_value=rows == "moe_cp"),
        patch_communicator("_cp_gathers_over_attn_cp", return_value=False),
    ):
        yield


class TestNumTokenNonPaddedLayoutTable(CustomTestCase):
    # (label, attn_tp_size, sharded, padded_bucket, real-per-dp-group,
    #  expected [per attn-tp rank] per dp group)
    _LAYOUTS = [
        # Each dp rank gets a 10-token request, cuda graph pads it to 16.
        ("TP4", 4, False, 16, [10], [[10, 10, 10, 10]]),
        ("TP4.SP4", 4, True, 16, [10], [[4, 4, 2, 0]]),
        ("DP4", 1, False, 16, [10, 10, 10, 10], [[10], [10], [10], [10]]),
        ("TP2.DP2.SP2", 2, True, 16, [10, 10], [[8, 2], [8, 2]]),
        # dp0/dp2 get 10 tokens, dp1/dp3 get 20; all padded to 32.
        ("DP4.EP4", 1, False, 32, [10, 20, 10, 20], [[10], [20], [10], [20]]),
        ("TP2.DP2.SP2.EP2", 2, True, 32, [10, 20], [[10, 0], [16, 4]]),
    ]

    def test_layouts_match_expected_per_rank(self):
        for label, attn_tp, sharded, bucket, dp_reals, expected in self._LAYOUTS:
            for dp_idx, real in enumerate(dp_reals):
                for rank in range(attn_tp):
                    want = expected[dp_idx][rank]
                    with (
                        self.subTest(layout=label, dp=dp_idx, rank=rank),
                        get_parallel().override(
                            attn_tp_size=attn_tp, attn_tp_rank=rank
                        ),
                    ):
                        got_cpu = compute_local_num_token_non_padded_cpu(
                            global_num_token_non_padded=real,
                            num_tokens_per_dp=bucket,
                            sharded=sharded,
                        )
                        got_gpu = compute_local_num_token_non_padded(
                            global_num_token_non_padded=torch.tensor(real),
                            num_tokens_per_dp=bucket,
                            sharded=sharded,
                        )
                        self.assertEqual(got_cpu, want)
                        self.assertEqual(int(got_gpu), want)


LOCAL, GLOBAL = 7, 30


def _forward_batch(
    *,
    sharded: bool,
    local=LOCAL,
    glob=GLOBAL,
    forward_mode=ForwardMode.DECODE,
    attn_cp_metadata=None,
) -> ForwardBatch:
    empty = torch.empty(0, dtype=torch.int32)
    batch = ForwardBatch(
        forward_mode=forward_mode,
        batch_size=1,
        input_ids=empty,
        req_pool_indices=empty,
        seq_lens=empty,
        out_cache_loc=empty,
        seq_lens_sum=0,
    )
    batch.num_token_non_padded = (
        None if local is None else torch.tensor(local, dtype=torch.int32)
    )
    batch.global_num_token_non_padded = (
        None if glob is None else torch.tensor(glob, dtype=torch.int32)
    )
    batch.attn_tp_sequence_sharded = sharded
    batch.attn_cp_metadata = attn_cp_metadata
    return batch


def _value(batch: ForwardBatch):
    got = batch.moe_num_token_non_padded()
    return None if got is None else int(got)


class TestMoeNumTokenNonPaddedTable(CustomTestCase):
    # (label, MoE input, sharded, attn_dp_size, expected). Decode forwards, so
    # the MoE-CP rows follow the FFN rows' layout: that config all-gathers over
    # the MoE-CP group on a context-parallel extend only, which is the case below.
    _TABLE = [
        ("scattered.replicated", "local", False, 1, LOCAL),
        ("scattered.sharded", "local", True, 1, LOCAL),
        ("scattered.dp", "local", True, 4, LOCAL),
        ("moe_full.decode.replicated", "moe_cp", False, 1, LOCAL),
        ("moe_full.decode.sharded", "moe_cp", True, 1, GLOBAL),
        ("moe_full.decode.dp", "moe_cp", True, 4, None),
        ("full.dp_gathered", "ffn", True, 4, None),
        ("full.dp_gathered.replicated", "ffn", False, 4, None),
        ("full.tp_sharded", "ffn", True, 1, GLOBAL),
        ("full.replicated", "ffn", False, 1, LOCAL),
    ]

    def test_table(self):
        for label, mode, sharded, attn_dp_size, expected in self._TABLE:
            with (
                self.subTest(case=label),
                get_parallel().override(attn_dp_size=attn_dp_size, attn_cp_size=1),
                sparse_moe_input(mode),
            ):
                self.assertEqual(_value(_forward_batch(sharded=sharded)), expected)

    def test_cp_gathered_full_is_unmasked(self):
        """A CP prefill must route every row: DSA / MLA CP put a MoE on the
        FFN rows but all-gather across CP, which zigzag-permutes the real rows."""
        for sharded in (False, True):
            for dsa_cp, mla_cp in ((True, False), (False, True)):
                with (
                    self.subTest(sharded=sharded, dsa_cp=dsa_cp),
                    get_parallel().override(attn_dp_size=1, attn_cp_size=2),
                    sparse_moe_input("ffn"),
                    patch(
                        "sglang.srt.layers.attention.dsa.utils.dsa_use_prefill_cp",
                        return_value=dsa_cp,
                    ),
                    patch(
                        "sglang.srt.layers.cp.utils.is_mla_cp_active",
                        return_value=mla_cp,
                    ),
                ):
                    self.assertIsNone(_value(_forward_batch(sharded=sharded)))

    def test_moe_full_gathers_only_on_a_context_parallel_extend(self):
        cp_metadata = SimpleNamespace(per_rank_actual_token=[2, 2])
        """A moe-cp config keeps its bound on every forward but the CP extend,
        which is the only one that all-gathers over the MoE-CP group."""
        # get_moe_cp_size reads a live process group, so it is stubbed rather
        # than reached through a topology published without distributed init.
        for label, forward_mode, metadata, expected in [
            ("cp_extend", ForwardMode.EXTEND, cp_metadata, None),
            ("extend_without_cp_metadata", ForwardMode.EXTEND, None, LOCAL),
            ("decode", ForwardMode.DECODE, cp_metadata, LOCAL),
        ]:
            with (
                self.subTest(case=label),
                get_parallel().override(attn_dp_size=1, attn_cp_size=2),
                sparse_moe_input("moe_cp"),
                patch_communicator("get_moe_cp_size", return_value=2),
                patch(
                    "sglang.srt.layers.attention.dsa.utils.dsa_use_prefill_cp",
                    return_value=False,
                ),
                patch(
                    "sglang.srt.layers.cp.utils.is_mla_cp_active", return_value=False
                ),
            ):
                got = _value(
                    _forward_batch(
                        sharded=False,
                        forward_mode=forward_mode,
                        attn_cp_metadata=metadata,
                    )
                )
                self.assertEqual(got, expected)

    def test_graph_replay_without_global_count_skips_masking(self):
        """A captured batch carries the LOCAL buffer alone, so the attn-TP
        sharded case on the FFN rows has no bound to mask with and must not use it."""
        with (
            get_parallel().override(attn_dp_size=1, attn_cp_size=1),
            sparse_moe_input("ffn"),
        ):
            self.assertIsNone(_value(_forward_batch(sharded=True, glob=None)))

    def test_absent_local_count_stays_absent(self):
        """Without expert parallelism the count is never filled, and routing
        must not mask every row against the unfilled buffer."""
        for mode in ("local", "ffn", "moe_cp"):
            with (
                self.subTest(mode=mode),
                get_parallel().override(attn_dp_size=1, attn_cp_size=1),
                sparse_moe_input(mode),
            ):
                self.assertIsNone(_value(_forward_batch(sharded=False, local=None)))


class TestEnableNumTokenNonPadded(CustomTestCase):
    def test_customized_a2a_backend_fills_the_count_at_ep1(self):
        """At EP 1 without a WORLD-group DP gather, the customized backend
        enables the count and `none` leaves it disabled."""
        for backend, expected in (
            (MoeA2ABackend.NONE, False),
            (MoeA2ABackend.CUSTOMIZED, True),
        ):
            with (
                self.subTest(backend=backend.value),
                get_parallel().override(moe_ep_size=1),
                get_flags().moe.override(a2a_backend=backend),
                get_flags().dp.override(use_world_group_for_gather=False),
            ):
                self.assertIs(enable_num_token_non_padded(), expected)


class TestRowBucketMoeBound(CustomTestCase):
    def test_capture_owns_the_moe_bound_and_restores_the_outer_context(self):
        """A bucket's recorded routing must retain its own mutable count buffer."""
        for active_context in (False, True):
            with (
                self.subTest(active_context=active_context),
                get_parallel().override(attn_dp_size=1, attn_cp_size=1),
                sparse_moe_input("local"),
            ):
                outer_batch = _forward_batch(sharded=False, local=3, glob=3)
                outer = ForwardContext(attn_backend=None)
                scope = (
                    forward_context(outer, forward_batch=outer_batch)
                    if active_context
                    else nullcontext()
                )
                observed = []

                def body(inputs, capture_batch):
                    if active_context:
                        bound = get_forward_context().moe_num_token_non_padded()
                        self.assertIs(bound, capture_batch.num_token_non_padded)
                        observed.append(bound)
                    else:
                        self.assertFalse(has_forward_context())
                    return inputs

                def record(body, rows, count, batch, start, free_before):
                    body()
                    return SimpleNamespace(
                        batch=(),
                        num_token_non_padded=count,
                    )

                runner = RowBucketGraphRunner(row_step=8, max_rows=8, label="test")
                with (
                    scope,
                    patch.object(torch.cuda, "mem_get_info", return_value=(1, 2)),
                    patch.object(runner, "_record", side_effect=record),
                ):
                    graph = runner._capture(
                        8,
                        3,
                        [torch.zeros(8, 1)],
                        outer_batch,
                        body,
                        lambda batch, inputs: nullcontext(),
                    )
                    self.assertIsNone(graph.global_num_token_non_padded)
                    graph.num_token_non_padded.fill_(7)
                    outer_batch.num_token_non_padded.fill_(1)
                    if active_context:
                        self.assertEqual(int(observed[0]), 7)
                        self.assertIs(
                            get_forward_context().moe_num_token_non_padded(),
                            outer_batch.num_token_non_padded,
                        )
                self.assertFalse(has_forward_context())

    def test_global_tail_count_replays_without_borrowing_the_parent(self):
        """Gathered MoE rows use the live GLOBAL prefix, including an idle replay;
        each attention-TP rank retains a separately localized count."""
        for rank in range(4):
            with (
                self.subTest(rank=rank),
                get_parallel().override(
                    attn_dp_size=1, attn_cp_size=1, attn_tp_size=4, attn_tp_rank=rank
                ),
                sparse_moe_input("ffn"),
            ):
                parent = _forward_batch(sharded=True, local=27, glob=97)
                if rank in (1, 3):
                    parent.num_token_non_padded = parent.num_token_non_padded.reshape(1)
                if rank in (2, 3):
                    parent.global_num_token_non_padded = (
                        parent.global_num_token_non_padded.reshape(1)
                    )
                original_local = parent.num_token_non_padded
                original_global = parent.global_num_token_non_padded
                runner = RowBucketGraphRunner(row_step=8, max_rows=16, label="test")
                observed = []

                def body(inputs, batch):
                    bound = get_forward_context().moe_num_token_non_padded()
                    if bound is None:
                        observed.append((None, None, None))
                        return [torch.zeros(inputs[0].shape[0], 1, dtype=torch.bool)]
                    expected = (
                        batch.global_num_token_non_padded
                        if batch.attn_tp_sequence_sharded
                        else batch.num_token_non_padded
                    )
                    self.assertIs(bound, expected)
                    self.assertIsNot(bound, original_global)
                    self.assertIsNot(batch.num_token_non_padded, original_local)
                    self.assertEqual(
                        batch.num_token_non_padded.shape, original_local.shape
                    )
                    if batch.attn_tp_sequence_sharded:
                        self.assertEqual(bound.shape, original_global.shape)
                        self.assertIsNot(bound, batch.num_token_non_padded)
                    observed.append(
                        (int(bound), int(batch.num_token_non_padded), bound)
                    )
                    return [(torch.arange(inputs[0].shape[0]) < bound).reshape(-1, 1)]

                def record(body, rows, count, batch, start, free_before):
                    outputs = body()

                    def replay():
                        for output, value in zip(outputs, body()):
                            output.copy_(value)

                    return SimpleNamespace(
                        graph=SimpleNamespace(replay=replay),
                        batch=(),
                        outputs=outputs,
                        num_token_non_padded=count,
                    )

                addresses = {}
                with (
                    forward_context(
                        ForwardContext(attn_backend=None), forward_batch=parent
                    ),
                    patch.object(torch.cuda, "mem_get_info", return_value=(1, 2)),
                    patch.object(runner, "_record", side_effect=record),
                    patch(
                        "sglang.srt.model_executor.runner.row_bucket_graph_runner.graph_pool_replay_scope",
                        side_effect=nullcontext,
                    ),
                ):
                    for live, bucket, local_counts in (
                        (10, 16, [4, 4, 2, 0]),
                        (3, 8, [2, 1, 0, 0]),
                        (7, 8, [2, 2, 2, 1]),
                        (0, 8, [0, 0, 0, 0]),
                        (10, 16, [4, 4, 2, 0]),
                    ):
                        outputs = runner.run(
                            key=(),
                            live=[torch.ones(live, 1)],
                            forward_batch=parent,
                            body=body,
                            context=lambda batch: nullcontext(),
                            bucket_from=bucket,
                            global_num_rows=live,
                        )
                        self.assertEqual(observed[-1][:2], (live, local_counts[rank]))
                        self.assertTrue(outputs[0].all())
                        pointer = observed[-1][2].data_ptr()
                        self.assertEqual(addresses.setdefault(bucket, pointer), pointer)
                        self.assertIs(parent.num_token_non_padded, original_local)
                        self.assertIs(
                            parent.global_num_token_non_padded, original_global
                        )
                        self.assertEqual(int(original_local), 27)
                        self.assertEqual(int(original_global), 97)
                        self.assertIs(
                            get_forward_context().moe_num_token_non_padded(),
                            original_global,
                        )
                    parent.attn_tp_sequence_sharded = False
                    runner.run(
                        key=(),
                        live=[torch.ones(3, 1)],
                        forward_batch=parent,
                        body=body,
                        context=lambda batch: nullcontext(),
                    )
                    self.assertEqual(observed[-1][:2], (3, 3))
                    self.assertNotEqual(observed[-1][2].data_ptr(), addresses[8])
                    parent.attn_tp_sequence_sharded = True
                    runner.run(
                        key=(),
                        live=[torch.ones(3, 1)],
                        forward_batch=parent,
                        body=body,
                        context=lambda batch: nullcontext(),
                        global_num_rows=3,
                    )
                    self.assertEqual(observed[-1][2].data_ptr(), addresses[8])
                    parent.attn_tp_sequence_sharded = False
                    parent.num_token_non_padded = None
                    runner.run(
                        key=(),
                        live=[torch.ones(3, 1)],
                        forward_batch=parent,
                        body=body,
                        context=lambda batch: nullcontext(),
                    )
                    self.assertEqual(observed[-1], (None, None, None))
                    parent.num_token_non_padded = original_local
                self.assertNotEqual(addresses[8], addresses[16])

    def test_global_tail_count_rejects_an_out_of_bucket_prefix(self):
        runner = RowBucketGraphRunner(row_step=8, max_rows=8, label="test")
        for count in (-1, 9):
            with self.subTest(count=count), self.assertRaises(AssertionError):
                runner.run(
                    key=(),
                    live=[torch.ones(3, 1)],
                    forward_batch=_forward_batch(sharded=True),
                    body=lambda inputs, batch: inputs,
                    context=lambda batch: nullcontext(),
                    global_num_rows=count,
                )
        for field in ("num_token_non_padded", "global_num_token_non_padded"):
            for invalid in (
                torch.ones((), dtype=torch.int64),
                torch.ones(2, dtype=torch.int32),
                torch.ones(1, 1, dtype=torch.int32),
            ):
                with (
                    self.subTest(field=field, dtype=invalid.dtype, shape=invalid.shape),
                    patch.object(torch.cuda, "mem_get_info", return_value=(1, 2)),
                    self.assertRaises(AssertionError),
                ):
                    batch = _forward_batch(sharded=True)
                    setattr(batch, field, invalid)
                    runner._capture(
                        8,
                        3,
                        [torch.zeros(8, 1)],
                        batch,
                        lambda inputs, batch: inputs,
                        lambda batch, inputs: nullcontext(),
                        global_num_rows=3,
                    )


class TestDecoderReplayTailMoeBound(CustomTestCase):
    # GLOBAL is opted in only for the gathered, attention-TP-sharded layout.
    _CASES = (
        ("local", False, 3, 7, False, True),
        ("global", True, 3, 7, False, True),
        ("equal_global", True, 3, 3, False, True),
        ("aliased_local", False, 3, 3, True, True),
        ("aliased_global", True, 3, 3, True, True),
        ("unbounded_global", True, 3, None, False, True),
        ("absent_local", True, None, 7, False, True),
    )

    @staticmethod
    def _batch_and_tail(sharded, local, glob, alias):
        batch = _forward_batch(
            sharded=sharded, local=local, glob=glob, forward_mode=ForwardMode.EXTEND
        )
        batch.input_ids = torch.arange(4)
        if alias:
            batch.global_num_token_non_padded = batch.num_token_non_padded
        tail = SimpleNamespace(
            positions=torch.arange(3), cp_metadata=None, rows=lambda t: t[:3]
        )
        return batch, tail

    def test_eager_prefill_admits_only_owned_moe_count_domains(self):
        for label, sharded, local, glob, alias, expected in self._CASES:
            with (
                self.subTest(bound=label),
                get_parallel().override(attn_dp_size=1, attn_cp_size=1),
                sparse_moe_input("ffn"),
            ):
                batch, tail = self._batch_and_tail(sharded, local, glob, alias)
                original_global = batch.global_num_token_non_padded
                tail_layers = replay_tail.DecoderReplayTail.__new__(
                    replay_tail.DecoderReplayTail
                )
                tail_layers.runner = SimpleNamespace(
                    ready=Mock(), warm=True, bucket_rows=lambda n: 8
                )
                with (
                    patch.object(
                        replay_tail, "get_is_capture_mode", return_value=False
                    ),
                    patch.object(
                        replay_tail,
                        "get_global_expert_distribution_recorder",
                        return_value=SimpleNamespace(recording=False),
                    ),
                    patch(
                        "sglang.srt.layers.attention.dsv4.unified_kv_gate.is_unified_kv_triton",
                        return_value=False,
                    ),
                ):
                    use_graph, dp_rows = tail_layers.plan(
                        tail,
                        forward_batch=batch,
                        num_tokens=4,
                        tail_in_graph=False,
                        late_dp_counts=None,
                    )
                self.assertEqual(use_graph, expected)
                self.assertIsNone(dp_rows)
                self.assertIs(batch.global_num_token_non_padded, original_global)

    def test_prefill_graph_break_admits_only_owned_moe_count_domains(self):
        for label, sharded, local, glob, alias, expected in self._CASES:
            with (
                self.subTest(bound=label),
                get_parallel().override(attn_dp_size=1, attn_cp_size=1),
                sparse_moe_input("ffn"),
            ):
                batch, tail = self._batch_and_tail(sharded, local, glob, alias)
                original_global = batch.global_num_token_non_padded
                output = torch.tensor([11])
                tail_layers = SimpleNamespace(
                    runner=SimpleNamespace(warm=True, bucket_rows=lambda n: 8),
                    _can_capture_moe_bound=(
                        replay_tail.DecoderReplayTail._can_capture_moe_bound
                    ),
                    run=Mock(return_value=(output, None, [])),
                    _run_layers_eagerly=Mock(return_value=(output, None, [])),
                    _bridge_outputs=lambda rows, outputs: tuple(outputs),
                )
                backend = SimpleNamespace(
                    forward_metadata=SimpleNamespace(late_layer_tail=tail),
                    exit_late_layer_tail=Mock(),
                )
                with (
                    patch.object(replay_tail, "get_attn_backend", return_value=backend),
                    patch.object(
                        replay_tail,
                        "get_tc_piecewise_forward_context",
                        return_value=SimpleNamespace(forward_batch=batch),
                    ),
                    patch.object(
                        replay_tail,
                        "_enter_eager_tail_in_prefill_graph",
                        return_value=(None, batch),
                    ),
                    get_flags().dp.override(capturing_prefill_graph=False),
                ):
                    outputs = replay_tail._tail_layers_in_prefill_graph(
                        tail_layers,
                        lambda tensors: tensors,
                        False,
                        4,
                        None,
                        False,
                        [torch.zeros(4, 1)],
                        torch.arange(4),
                        batch.input_ids,
                        batch.input_ids,
                    )
                self.assertIs(outputs[0], output)
                self.assertEqual(tail_layers.run.call_count, expected)
                self.assertEqual(
                    tail_layers._run_layers_eagerly.call_count, not expected
                )
                backend.exit_late_layer_tail.assert_called_once_with(None, batch)
                self.assertIs(batch.global_num_token_non_padded, original_global)

    def test_custom_moe_count_has_no_graph_owner(self):
        batch = _forward_batch(sharded=True)
        with patch.object(
            ForwardBatch, "moe_num_token_non_padded", return_value=torch.tensor(3)
        ):
            self.assertFalse(
                replay_tail.DecoderReplayTail._can_capture_moe_bound(batch)
            )


if __name__ == "__main__":
    unittest.main()
