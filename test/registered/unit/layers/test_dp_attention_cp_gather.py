"""The dense-FFN gather and take-back under attention DP x CP x TP, rank by rank.

Every rank builds its layer's communicator and runs its prepare_mlp and
postprocess_layer for real on CPU; the TP-group all-reduce is replaced by the
sum of the buffers the ranks hand to it.
"""

import unittest
from contextlib import ExitStack, contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import torch

from sglang.srt.layers import communicator as comm
from sglang.srt.layers import dp_attention, layernorm_sp
from sglang.srt.layers.cp import base as cp_base
from sglang.srt.layers.cp import padding as cp_padding
from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

HIDDEN = 4
DP_SIZE = 2
CP_SIZE = 2
# Rows past a shard's tokens hold whatever attention left there.
PAD_VALUE = 99.0


def zigzag_metadata(extend_seqs_len):
    """A DP group's zigzag metadata as a forward sees it: every shard padded to
    one length."""
    with patch.object(
        cp_base.ContextParallelStrategy,
        "cp_rank",
        new_callable=PropertyMock,
        return_value=0,
    ):
        metadata = ZigzagCPStrategy(cp_size=CP_SIZE).build_metadata(
            sum(extend_seqs_len), extend_seqs_len, extend_seqs_len
        )
    with (
        patch.object(
            cp_padding, "get_parallel", lambda: SimpleNamespace(attn_cp_size=CP_SIZE)
        ),
        patch.object(cp_base, "is_zigzag", lambda: True),
    ):
        cp_padding.pad_logical_token_to_physical(metadata)
    return metadata


def ceil_align(n, m):
    return (n + m - 1) // m * m


def layernorm(hidden_states, residual=None):
    """Norm as the identity, keeping the fused add of the two-argument form."""
    if residual is None:
        return hidden_states.clone()
    summed = hidden_states + residual
    return summed, summed.clone()


def rms_norm(hidden_states, residual=None):
    """A row-wise RMS norm, keeping the fused add of the two-argument form. It
    is not linear: norming a partial sum does not give a share of the normed
    rows."""
    if residual is None:
        return rms_rows(hidden_states)
    summed = hidden_states + residual
    return rms_rows(summed), summed.clone()


def rms_rows(rows):
    return rows / (rows.pow(2).mean(-1, keepdim=True) + 1).sqrt()


def random_rows(n, generator):
    return torch.randint(-8, 8, (n, HIDDEN), generator=generator).double()


class DpCpGroup:
    """What one DP group holds before the gather, per CP rank: the value and
    residual of the rows holding tokens, and how many rows the rank holds."""

    def __init__(self, seed, forward_mode, token_rows, held_rows, metadata=None):
        generator = torch.Generator().manual_seed(seed)
        self.forward_mode = forward_mode
        self.metadata = metadata
        self.held_rows = held_rows
        self.values = [random_rows(n, generator) for n in token_rows]
        self.residuals = [random_rows(n, generator) for n in token_rows]

    def held(self, rows, cp):
        return torch.cat(
            [
                rows,
                torch.full((self.held_rows[cp] - rows.shape[0], HIDDEN), PAD_VALUE),
            ]
        ).double()


def cp_active(extend_seqs_len, seed):
    metadata = zigzag_metadata(extend_seqs_len)
    return DpCpGroup(
        seed,
        ForwardMode.EXTEND,
        metadata.per_rank_logical_token,
        metadata.per_rank_actual_token,
        metadata,
    )


def cp_replicated(forward_mode, rows, seed):
    """Decode or idle: every CP rank holds all of the group's rows."""
    group = DpCpGroup(seed, forward_mode, [rows], [rows] * CP_SIZE)
    group.values *= CP_SIZE
    group.residuals *= CP_SIZE
    return group


class TestDpCpGather(CustomTestCase):
    def run_ranks(
        self,
        groups,
        global_num_tokens,
        attn_tp_size,
        padding=DpPaddingMode.SUM_LEN,
        norm=layernorm,
        norm_rows=lambda rows: rows,
        force_layernorm_before_dp_gather=False,
        ffn_input=None,
    ):
        """Gather on every rank, check the FFN input and residual, then take back
        an FFN output of three times the input and check each rank's rows.
        ``norm_rows`` is what ``norm`` does to complete rows; ``ffn_input``, when
        given, is the step every rank's batch must take."""
        # Binary fractions, so the partial sums add back to the value exactly.
        weights = {1: [1.0], 2: [0.25, 0.75]}[attn_tp_size]
        buffer_len = sum(global_num_tokens)
        ranks = [
            (dp, cp, tp)
            for dp in range(DP_SIZE)
            for cp in range(CP_SIZE)
            for tp in range(attn_tp_size)
        ]

        def rank_inputs(dp, cp, tp):
            # Attention TP ranks hold partial sums of the value in partial mode,
            # and the value itself in replicate mode.
            group = groups[dp]
            return (
                group.held(group.values[cp] * weights[tp], cp),
                group.held(group.residuals[cp], cp),
            )

        def attention_tp_sum(dp, cp, tp):
            def all_reduce(x):
                torch.testing.assert_close(x, rank_inputs(dp, cp, tp)[0])
                return sum(rank_inputs(dp, cp, t)[0] for t in range(attn_tp_size))

            return all_reduce

        @contextmanager
        def as_rank(dp, cp, tp, all_reduce):
            tp_size = DP_SIZE * CP_SIZE * attn_tp_size
            parallel = SimpleNamespace(
                attn_dp_rank=dp,
                attn_cp_rank=cp,
                attn_tp_rank=tp,
                attn_dp_size=DP_SIZE,
                enable_dp_attention=True,
                attn_cp_size=CP_SIZE,
                attn_tp_size=attn_tp_size,
                tp_size=tp_size,
                tp_rank=(dp * CP_SIZE + cp) * attn_tp_size + tp,
                enable_prefill_cp=True,
                moe_dp_size=1,
                moe_ep_size=1,
                moe_tp_size=tp_size,
                moe_dense_tp_size=None,
                dwdp_size=1,
                enable_attn_tp_input_scattered=False,
                tp_group=SimpleNamespace(unique_name="tp"),
                attn_tp_group=SimpleNamespace(),
            )
            # CP forward prep sizes the local buffer to every shard of the group.
            group = groups[dp]
            local_buffer_len = (
                sum(group.held_rows)
                if group.metadata is not None
                else group.held_rows[0]
            )
            with ExitStack() as stack:
                for target, value in [
                    ((comm, "get_parallel"), lambda: parallel),
                    ((comm, "get_lora"), lambda: SimpleNamespace(enable_lora=False)),
                    ((dp_attention, "get_parallel"), lambda: parallel),
                    ((dp_attention, "world_dp_gather_enabled"), lambda: False),
                    ((dp_attention, "_note_dp_gather_in_prefill_graph"), lambda: None),
                    ((dp_attention, "memcpy_func"), dp_attention.memcpy_cpu),
                    ((dp_attention, "tensor_model_parallel_all_reduce"), all_reduce),
                    (
                        (comm, "attention_tensor_model_parallel_all_reduce"),
                        attention_tp_sum(dp, cp, tp),
                    ),
                    ((comm, "get_moe_cp_size"), lambda: CP_SIZE),
                    ((comm, "is_enable_moe_cp_allgather"), lambda: True),
                    ((comm, "is_dsa_enable_prefill_cp"), lambda: False),
                    ((comm, "is_mla_cp_enabled"), lambda: False),
                    ((comm, "is_moe_input_scattered_across_dp_ranks"), lambda: False),
                    ((comm, "should_use_dp_reduce_scatterv"), lambda: False),
                    (
                        (comm, "get_spec"),
                        lambda: SimpleNamespace(speculative_algorithm=None),
                    ),
                    ((layernorm_sp, "layernorm_sp_enabled"), lambda: False),
                    (
                        (comm, "get_attn_tp_context"),
                        lambda: SimpleNamespace(input_scattered=False),
                    ),
                    ((comm, "use_symmetric_memory"), lambda *a, **k: nullcontext()),
                    ((comm, "is_allocation_symmetric"), lambda: False),
                    (
                        (comm, "get_global_dp_buffer"),
                        lambda group: torch.empty(buffer_len, HIDDEN).double(),
                    ),
                    (
                        (comm, "get_local_dp_buffer"),
                        lambda group, hidden_size=None: torch.empty(
                            local_buffer_len, hidden_size or HIDDEN
                        ).double(),
                    ),
                ]:
                    stack.enter_context(patch.object(*target, value))
                # A dense layer in the middle of the model: its steps come from
                # the declarations, the modes only give the layer facts.
                communicator = comm.LayerCommunicator(
                    layer_scatter_modes=SimpleNamespace(
                        is_first_layer=False,
                        is_last_layer=False,
                        is_layer_sparse=False,
                        is_previous_layer_sparse=False,
                        is_next_layer_sparse=False,
                    ),
                    input_layernorm=norm,
                    post_attention_layernorm=norm,
                    force_layernorm_before_dp_gather=force_layernorm_before_dp_gather,
                )
                yield SimpleNamespace(
                    communicator=communicator,
                    forward_batch=SimpleNamespace(
                        forward_mode=groups[dp].forward_mode,
                        attn_cp_metadata=groups[dp].metadata,
                        global_num_tokens_cpu=list(global_num_tokens),
                        global_num_tokens_gpu=torch.tensor(
                            global_num_tokens, dtype=torch.int64
                        ),
                        dp_padding_mode=padding,
                        dp_local_start_pos=None,
                        dp_local_num_tokens=None,
                    ),
                )

        def gather(rank, all_reduce):
            with as_rank(*rank, all_reduce) as r:
                self.assertIsNotNone(r.communicator._cp_steps, "declared under CP")
                if ffn_input is not None:
                    steps = r.communicator._batch_steps(r.forward_batch)
                    self.assertIs(steps.ffn_input.func, ffn_input)
                hidden_states, residual = rank_inputs(*rank)
                return r.communicator.prepare_mlp(
                    hidden_states, residual, r.forward_batch
                )

        # The all-reduce is a sum over every rank: record what each rank hands
        # to it, then run again with the sum as its result.
        handed = {}

        def record(rank):
            def all_reduce(x):
                self.assertNotIn(rank, handed, "one all-reduce per rank")
                handed[rank] = x.clone()
                return x

            return all_reduce

        for rank in ranks:
            gather(rank, record(rank))
        self.assertEqual(sorted(handed), ranks, "every rank joins the gather")
        summed = sum(handed.values())

        # Each DP slot holds its tokens: the shards of different CP ranks side by
        # side, rows that CP ranks share once.
        expected = torch.zeros(buffer_len, HIDDEN).double()
        for dp, group in enumerate(groups):
            shards = group.values if group.metadata is not None else group.values[:1]
            start = sum(global_num_tokens[:dp])
            for value, residual in zip(shards, group.residuals):
                expected[start : start + value.shape[0]] = norm_rows(value + residual)
                start += value.shape[0]
        torch.testing.assert_close(summed, expected, rtol=0, atol=0)

        for rank in ranks:
            dp, cp, tp = rank
            group = groups[dp]
            own = group.values[cp] + group.residuals[cp]
            tokens = own.shape[0]
            hidden_states, residual = gather(rank, lambda x: summed.clone())
            torch.testing.assert_close(hidden_states, expected, rtol=0, atol=0)
            torch.testing.assert_close(residual[:tokens], own, rtol=0, atol=0)
            with as_rank(*rank, None) as r:
                back, _ = r.communicator.postprocess_layer(
                    3 * hidden_states, residual, r.forward_batch
                )
            # The rank's rows at their held length, padding zeroed.
            expected_back = torch.zeros(group.held_rows[cp], HIDDEN).double()
            expected_back[:tokens] = 3 * norm_rows(own)
            torch.testing.assert_close(back, expected_back, rtol=0, atol=0)

    def test_prefill_shards_every_dp_group(self):
        groups = [cp_active([7, 5], seed=0), cp_active([9], seed=1)]
        # Attention TP pads each DP group's slot past its tokens.
        self.assertEqual(
            groups[1].metadata.per_rank_logical_token,
            [5, 4],
            "shards of unequal token rows",
        )
        self.run_ranks(groups, [ceil_align(12, 2), ceil_align(9, 2)], attn_tp_size=2)

    def test_prefill_beside_an_idle_dp_group(self):
        groups = [
            cp_active([7, 5], seed=0),
            cp_replicated(ForwardMode.IDLE, 0, seed=1),
        ]
        self.run_ranks(groups, [12, 0], attn_tp_size=2)

    def test_decode_rows_are_the_same_on_every_cp_rank(self):
        for padding, global_num_tokens in [
            (DpPaddingMode.SUM_LEN, [4, 2]),
            (DpPaddingMode.MAX_LEN, [4, 4]),
        ]:
            with self.subTest(padding=padding.name):
                groups = [
                    cp_replicated(ForwardMode.DECODE, n, seed=dp)
                    for dp, n in enumerate(global_num_tokens)
                ]
                self.run_ranks(
                    groups, global_num_tokens, attn_tp_size=2, padding=padding
                )

    def test_norm_before_gather_without_attention_tp(self):
        groups = [cp_active([7, 5], seed=0), cp_active([4, 6], seed=1)]
        self.run_ranks(groups, [12, 10], attn_tp_size=1)

    def test_prefill_beside_a_short_prefill(self):
        # One DP group shards a long prefill over CP; the other runs a prefill
        # too short to shard, so its CP ranks hold the same rows. Both go into
        # the one sum.
        groups = [
            cp_active([7, 5], seed=0),
            cp_replicated(ForwardMode.EXTEND, 6, seed=1),
        ]
        self.run_ranks(groups, [ceil_align(12, 2), 6], attn_tp_size=2)

    def test_forced_norm_before_gather_under_attention_tp(self):
        # The attention-TP sum completes before the norm, and one TP copy of the
        # normed rows goes into the gather.
        groups = [cp_active([7, 5], seed=0), cp_active([9], seed=1)]
        self.run_ranks(
            groups,
            [ceil_align(12, 2), ceil_align(9, 2)],
            attn_tp_size=2,
            norm=rms_norm,
            norm_rows=rms_rows,
            force_layernorm_before_dp_gather=True,
            ffn_input=comm._mlp_input_dp_replicate,
        )


if __name__ == "__main__":
    unittest.main()
