"""CPU checks for the Kimi-K3 sequence-parallel residual stream over the CP group.

The rank-major zigzag block layout that `_sp_cp_rank_major_blocks` produces for a
reduce-scatter must equal, per rank, what `ZigzagCPStrategy.shard_hidden_states`
gives that rank (plus its zero padding), so that RS(blocks)[rank] lands exactly
the rows the rank's residual stream holds."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.cp.base import init_cp_strategy
from sglang.srt.layers.layer_boundary import BatchVariant, TokenAxis
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layer_boundary.residual.attn_bank import AttnBank, AttnBankState
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import kimi_k3
from sglang.test.boundary_fixtures import build_stages, parallel_of, planning
from sglang.srt.layers.cp.padding import pad_local_rows, pad_logical_token_to_physical
from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy
from sglang.srt.models.kimi_k3 import _sp_cp_rank_major_blocks
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.communicator_patch import patch_communicator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSPResidualRankMajorBlocks(CustomTestCase):
    def _check(self, cp_size, extend_lens):
        num_tokens = sum(extend_lens)
        hidden = torch.arange(num_tokens, dtype=torch.float32).unsqueeze(
            1
        ) * torch.ones(1, 3)
        with get_parallel().override(attn_cp_rank=0, attn_cp_size=cp_size):
            strategy = ZigzagCPStrategy(cp_size=cp_size)
            metadata = strategy.build_metadata(num_tokens, extend_lens)
            pad_logical_token_to_physical(metadata)  # as prepare_cp_forward does
            blocks = _sp_cp_rank_major_blocks(hidden, metadata, cp_size)
        phys = metadata.per_rank_actual_token[0]
        self.assertEqual(tuple(blocks.shape), (phys * cp_size, 3))
        for rank in range(cp_size):
            with get_parallel().override(attn_cp_rank=rank, attn_cp_size=cp_size):
                strategy_r = ZigzagCPStrategy(cp_size=cp_size)
                meta_r = strategy_r.build_metadata(num_tokens, extend_lens)
                pad_logical_token_to_physical(meta_r)
                forward_batch = SimpleNamespace(attn_cp_metadata=meta_r)
                local = strategy_r.shard_hidden_states(hidden, forward_batch)
                local = pad_local_rows(local, meta_r, dim=0)
            with self.subTest(rank=rank):
                torch.testing.assert_close(
                    blocks[rank * phys : (rank + 1) * phys], local, rtol=0, atol=0
                )

    def test_single_balanced_request(self):
        self._check(8, [32768 // 64])

    def test_ragged_requests_with_padding(self):
        self._check(8, [100, 33])
        self._check(4, [257, 16, 4096 // 16])

    def test_cp2(self):
        self._check(2, [7, 9, 24])


class TestSPResidualBoundaries(CustomTestCase):
    @staticmethod
    def _config(*, kda=True, sparse=False):
        return SimpleNamespace(
            attn_res_block_size=4,
            routed_expert_hidden_size=4,
            is_kda_layer=lambda _: kda,
            is_moe=sparse,
            num_experts=8,
            first_k_dense_replace=0,
            moe_layer_freq=1,
        )

    @staticmethod
    def _norm(hidden, residual=None, **kwargs):
        if residual is None:
            return hidden
        return hidden + residual, hidden + residual

    def test_static_gate_requires_supported_head_and_bank_ownership(self):
        cases = [({}, {}, True)]
        cases += [({}, {name: value}, False) for name, value in (
            ("enable_linear_attn_cp", False), ("attn_cp_size", 1),
            ("tp_size", 4), ("attn_dp_size", 2), ("pp_size", 2),
        )]
        cases += [({name: None}, {}, False) for name in (
            "attn_res_block_size", "routed_expert_hidden_size"
        )]
        for config_fields, parallel_fields, expected in cases:
            config = self._config()
            config.__dict__.update(config_fields)
            parallel = SimpleNamespace(**{
                "enable_linear_attn_cp": True, "attn_cp_size": 2,
                "tp_size": 2, "attn_dp_size": 1, "pp_size": 1,
                **parallel_fields,
            })
            with (
                self.subTest(config=config_fields, parallel=parallel_fields),
                envs.SGLANG_K3_CP_SP_RESIDUAL.override(True),
                patch.object(kimi_k3, "get_parallel", return_value=parallel),
                patch.object(kimi_k3, "get_moe_a2a_backend", return_value=SimpleNamespace(is_none=lambda: True)),
            ):
                self.assertEqual(kimi_k3._sp_cp_static_enabled(config), expected)
        with envs.SGLANG_K3_CP_SP_RESIDUAL.override(False):
            self.assertFalse(kimi_k3._sp_cp_static_enabled(self._config()))

    def test_stage_plans_keep_local_bank_rows_only_when_sp_is_enabled(self):
        for sp in (False, True):
            for kda, sparse in ((True, False), (False, True)):
                parallel = parallel_of(
                    attn_dp=1, attn_tp=1, attn_cp=2,
                    enable_prefill_cp=True, enable_linear_attn_cp=True, enable_dense_mlp_attn_tp=False,
                )
                with (
                    self.subTest(sp=sp, kda=kda, sparse=sparse),
                    get_parallel().override(tp_size=2, tp_rank=0, attn_tp_size=1, attn_tp_rank=0, attn_cp_size=2, attn_cp_rank=0),
                    planning(parallel, dsa_cp=True),
                    patch.object(kimi_k3, "get_parallel", return_value=parallel),
                    patch.object(kimi_k3, "_sp_cp_static_enabled", return_value=sp),
                    patch.object(kimi_k3, "_carries_bank_slices", return_value=False),
                    patch.object(kimi_k3, "_shards_moe_rows", return_value=False),
                    patch.object(kimi_k3, "is_dp_attention_enabled", return_value=False),
                ):
                    declarations = kimi_k3._stage_facts(self._config(kda=kda, sparse=sparse), 0, sp_collective=False)
                    stages = build_stages(*[(declaration, self._norm) for declaration in declarations])
                expected = frozenset({TokenAxis.ATTN_CP}) if sp else frozenset()
                for stage in stages:
                    self.assertEqual(stage.plan.paths[BatchVariant.CONTEXT_PARALLEL].entry.input_rows.sharded, expected)
                    self.assertEqual(stage.plan.paths[BatchVariant.ORDINARY].entry.input_rows.sharded, frozenset())
                if sp:
                    rows = declarations[1].row_transport.rows[BatchVariant.CONTEXT_PARALLEL]
                    self.assertEqual(rows.returned.sharded, expected)
                    self.assertEqual(rows.compute.sharded, expected if sparse else frozenset())
                else:
                    self.assertIsNone(declarations[0].row_transport)
                    self.assertIsNone(declarations[1].row_transport)

    def _check_bank_round_trip(self, rank, *, fails=False):
        lengths = [7, 10]
        full = torch.arange(1, sum(lengths) * 3 + 1, dtype=torch.float32).reshape(-1, 3)
        strategy = ZigzagCPStrategy(cp_size=2)
        shards, batches = [], []
        with get_context().override_server_args(enable_prefill_cp=True, enable_linear_attn_cp=True, enable_dense_mlp_attn_tp=False, cp_strategy="zigzag"):
            for cp_rank in range(2):
                with get_parallel().override(tp_size=2, moe_tp_size=2, tp_rank=cp_rank, attn_tp_size=1, attn_tp_rank=0, attn_cp_size=2, attn_cp_rank=cp_rank):
                    metadata = strategy.build_metadata(len(full), lengths, lengths)
                    pad_logical_token_to_physical(metadata)
                    batch = SimpleNamespace(
                        attn_cp_metadata=metadata, input_ids=torch.arange(len(full)),
                        extend_seq_lens_cpu=lengths, forward_mode=ForwardMode.EXTEND,
                    )
                    batches.append(batch)
                    shards.append(strategy.shard_hidden_states(full, batch))
        local, batch = shards[rank], batches[rank]
        group = Mock(world_size=2, ranks=[0, 1])

        def gather(output, value):
            scale = value[0, 0] / local[0, 0]
            count = output.shape[0] // 2
            output.copy_(torch.cat([shard[:count] for shard in shards]) * scale)

        group.all_gather_into_tensor.side_effect = gather
        group.reduce_scatter_tensor.side_effect = lambda out, value: out.copy_(value.chunk(2)[rank] * 2)
        bank = AttnBank()
        bank.open(local, block_num=2)
        reads = []

        def aggregate(value, residual, *args, rows=None, write=False):
            self.assertIsNone(rows)
            self.assertEqual(value.shape[0], bank.current.block_residual.shape[0])
            reads.append(value.shape[0])
            summed = value if residual is None else value + residual
            return summed * 2, summed

        bank.current.forward = Mock(side_effect=aggregate)
        bank_ops = AttnBankState(bank, None, None, None, None).residual_ops()
        parallel = parallel_of(
            attn_dp=1, attn_tp=1, attn_cp=2, tp_rank=rank, attn_cp_rank=rank,
            enable_prefill_cp=True, enable_linear_attn_cp=True, enable_dense_mlp_attn_tp=False, tp_group=group, attn_cp_group=group,
        )
        init_cp_strategy(enable_prefill_cp=True, cp_size=2, cp_strategy="zigzag")
        self.addCleanup(init_cp_strategy, enable_prefill_cp=False, cp_size=1, cp_strategy="zigzag")
        with (
            get_context().override_server_args(enable_prefill_cp=True, enable_linear_attn_cp=True, enable_dense_mlp_attn_tp=False, cp_strategy="zigzag"),
            get_parallel().override(tp_size=2, moe_tp_size=2, tp_rank=rank, attn_tp_size=1, attn_tp_rank=0, attn_cp_size=2, attn_cp_rank=rank, attn_cp_group=group, tp_group=group),
            planning(parallel, dsa_cp=True),
            patch_communicator("dsa_use_prefill_cp", lambda batch: True),
            patch.object(kimi_k3, "get_parallel", return_value=parallel),
            patch.object(kimi_k3, "_sp_cp_static_enabled", return_value=True),
            patch.object(kimi_k3, "_carries_bank_slices", return_value=False),
            patch.object(kimi_k3, "use_symmetric_memory", side_effect=lambda *args, **kwargs: nullcontext()),
        ):
            attn, ffn = kimi_k3._stage_facts(self._config(), 0, sp_collective=False, bank=bank_ops)
            attention, feed_forward = build_stages((attn, self._norm), (ffn, self._norm))
            residual_batch.start(batch)
            operand = attention.prepare(local, batch)
            torch.testing.assert_close(operand, full * 2)
            self.assertIs(batch.residual_stream.residual, local)
            invocation = batch.residual_stream.transport[1]
            if fails:
                with self.assertRaisesRegex(RuntimeError, "compute failed"):
                    with attention.exit(batch):
                        raise RuntimeError("compute failed")
                self.assertIsNone(batch.residual_stream.transport)
                self.assertIsNone(invocation.input)
                self.assertEqual(bank.current.block_residual.shape[0], local.shape[0])
                residual_batch.clear(batch)
                return
            output = attention.finish(operand * 1.5, batch)
            torch.testing.assert_close(output, local * 6)
            self.assertIsNone(invocation.input)
            operand = feed_forward.prepare(output, batch)
            torch.testing.assert_close(operand, full * 14)
            torch.testing.assert_close(batch.residual_stream.residual, local * 7)
            output = feed_forward.finish(operand / 2, batch)
            output = residual_batch.final_norm(output, batch, self._norm)
        torch.testing.assert_close(output, local * 21)
        self.assertEqual(reads, [len(local), len(local)])
        self.assertEqual(group.reduce_scatter_tensor.call_count, 2)
        group.all_reduce.assert_not_called()
        self.assertIsNone(batch.residual_stream)

    def test_kda_and_dense_compute_return_to_local_bank_before_residual_add(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                self._check_bank_round_trip(rank)

    def test_compute_failure_releases_gathered_operand_without_moving_bank(self):
        self._check_bank_round_trip(0, fails=True)

    def test_auxiliary_capture_gathers_after_reading_the_local_bank(self):
        local = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        captured_local, captured_full = local + 5, torch.ones(8, 3)
        bank = AttnBank()
        bank.open(local, block_num=1)
        for sp, active in ((True, True), (True, False), (False, True)):
            owner = SimpleNamespace(
                end_layer=1, config=SimpleNamespace(attn_res_block_size=4),
                output_attn_res_proj=object(), output_attn_res_norm=object(),
                attn_bank=bank, _sp_cp=sp,
            )
            batch = SimpleNamespace(residual_stream=None)
            residual_batch.start(batch)
            with (
                self.subTest(sp=sp, active=active),
                patch.object(kimi_k3, "is_cp_active", return_value=active),
                patch.object(kimi_k3, "aggregate_stream", return_value=captured_local) as aggregate,
                patch.object(kimi_k3, "cp_gather_after_forward", return_value=captured_full) as gather,
            ):
                output = kimi_k3.KimiK3LinearModel._dspark_capture_stream(owner, 0, local, batch)
            torch.testing.assert_close(aggregate.call_args.args[0], local)
            self.assertIs(aggregate.call_args.args[1], bank.current.block_residual)
            if sp and active:
                gather.assert_called_once_with(captured_local, batch)
                self.assertIs(output, captured_full)
            else:
                gather.assert_not_called()
                self.assertIs(output, captured_local)
            self.assertEqual(bank.current.block_residual.shape[0], len(local))

    def test_ordinary_batch_keeps_full_rows_and_completes_the_tp_sum(self):
        values = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        transport = kimi_k3._sp_cp_transport(token_order=True)
        rows = transport.rows[BatchVariant.ORDINARY]
        with (
            patch.object(kimi_k3, "cp_gather_after_forward") as gather,
            patch.object(kimi_k3, "tensor_model_parallel_all_reduce", side_effect=lambda x: x * 2) as reduce,
        ):
            invocation = transport.bind(values, SimpleNamespace(), None, rows, None)
            self.assertIs(invocation.input, values)
            torch.testing.assert_close(invocation.return_output(values / 2), values)
        gather.assert_not_called()
        reduce.assert_called_once()


if __name__ == "__main__":
    unittest.main()
