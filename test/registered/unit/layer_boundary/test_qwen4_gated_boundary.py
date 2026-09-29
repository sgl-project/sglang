"""Qwen4's actual layer orchestration through production stage boundaries.

Only the hyper-connection kernels and distributed collectives are replaced:
the nonlinear reference catches mixing before reduction or combining before
the FFN output returns to the residual's rows.
"""

import __future__

import ast
import itertools
import unittest
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang
from sglang.srt.layers import layernorm_sp
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.layers.layer_boundary import (
    HandoffRows,
    declare_attn,
    declare_ffn,
    make_stages,
)
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layer_boundary.residual.gated import GatedResidualState
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_forward
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.communicator_patch import patch_communicator
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HIDDEN = 3


def mix_reference(value):
    first, second = value.chunk(2, dim=-1)
    return torch.tanh(first + second / 2)


def combine_reference(value, residual):
    return residual + value.repeat(1, 2) * torch.sigmoid(residual)


class Connection:
    hc_count = 2
    hidden_size = HIDDEN

    def __init__(self):
        self.mixed = []
        self.combined = []

    def mix(self, value):
        self.mixed.append(value.clone())
        return mix_reference(value), (value, torch.sigmoid(value))

    def combine(self, value, residuals):
        residual, normalized = residuals
        self.combined.append(value.clone())
        torch.testing.assert_close(normalized, torch.sigmoid(residual))
        return residual + value.repeat(1, 2) * normalized


def load_mixin(namespace):
    # Avoid importing GPU-only attention kernels just to execute the model's
    # orchestration. The methods compiled here are the production source.
    path = Path(sglang.__file__).parent / "srt/models/qwen4_exp.py"
    node = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen4ExpLayerExtensionMixin"
    )
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace[node.name]


class Harness:
    def __init__(
        self,
        *,
        dp=1,
        tp=1,
        a2a=False,
        rsv=False,
        rows=2,
        rank=0,
        cp_rows=None,
        cp_rank=0,
    ):
        self.dp, self.tp, self.a2a, self.rsv = dp, tp, a2a, rsv
        self.rows, self.rank = rows, rank
        self.cp_size = len(cp_rows) if cp_rows is not None else 1
        self.cp_rank = cp_rank
        self.events = []
        self.compute_inputs = []
        self.compute_flags = []
        self.expected_gathered = None
        self.expected_cp_inputs = None
        self.parallel = SimpleNamespace(
            tp_size=dp * tp * self.cp_size,
            tp_rank=rank,
            attn_dp_size=dp,
            attn_dp_rank=0,
            attn_tp_size=tp,
            attn_tp_rank=rank,
            attn_cp_size=self.cp_size,
            attn_cp_rank=cp_rank,
            moe_dp_size=1,
            moe_ep_size=1,
            moe_tp_size=dp * tp * self.cp_size,
            moe_dense_tp_size=None,
            dwdp_size=1,
            enable_dp_attention=dp > 1,
            enable_prefill_cp=self.cp_size > 1,
            enable_attn_tp_input_scattered=False,
            tp_group=SimpleNamespace(reduce_scatterv=self.reduce_scatterv),
            attn_tp_group=SimpleNamespace(),
        )
        self.batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE if rows else ForwardMode.IDLE,
            residual_stream=None,
            dp_padding_mode=DpPaddingMode.SUM_LEN,
            attn_cp_metadata=(
                SimpleNamespace(per_rank_actual_token=cp_rows)
                if cp_rows is not None
                else None
            ),
            global_num_tokens_cpu=[rows] * dp,
            input_ids=torch.zeros(rows, dtype=torch.int64),
        )
        if cp_rows is not None:
            self.batch.forward_mode = ForwardMode.EXTEND

    def attention_sum(self, value):
        self.events.append("attention_sum")
        return value * self.tp

    def attention_scatter(self, output, value):
        self.events.append("attention_reduce_scatter")
        output.copy_((value * self.tp).tensor_split(self.tp)[self.rank])

    def attention_gather(self, output, value):
        self.events.append("attention_gather")
        torch.testing.assert_close(
            value, self.expected_gathered.tensor_split(self.tp)[self.rank]
        )
        output.copy_(self.expected_gathered)

    def gather_dp(self, output, value, batch, cp_shard_counts=None):
        self.events.append("dp_gather")
        output.copy_(torch.cat([value + i * 10 for i in range(self.dp)]))

    def gather_cp(self, output, value):
        self.events.append("cp_gather")
        torch.testing.assert_close(
            value, self.expected_cp_inputs.tensor_split(self.cp_size)[self.cp_rank]
        )
        output.copy_(self.expected_cp_inputs)

    def scatter_dp(self, output, value, batch):
        self.events.append("dp_scatter")
        output.copy_(value[: self.rows])

    def reduce_scatterv(self, value, *, output, sizes):
        self.events.append("dp_reduce_scatterv")
        output.copy_(value[: self.rows] * self.parallel.tp_size)

    def mlp(self, value, *args):
        self.compute_inputs.append(value.clone())
        skipped = get_forward().mlp_reduce_scatter
        self.compute_flags.append(skipped)
        return value * (3 / self.parallel.tp_size if skipped else 3)

    @contextmanager
    def running(self):
        backend = SimpleNamespace(is_none=lambda: not self.a2a)
        with ExitStack() as stack:
            overrides = {
                "get_parallel": lambda: self.parallel,
                "get_moe_a2a_backend": lambda: backend,
                "is_moe_input_scattered_across_dp_ranks": lambda: self.a2a,
                "is_enable_moe_cp_allgather": lambda: self.cp_size > 1,
                "get_moe_cp_size": lambda: self.cp_size,
                "get_moe_cp_rank": lambda: self.cp_rank,
                "moe_cp_all_gather_into_tensor": self.gather_cp,
                "is_dsa_enable_prefill_cp": lambda: False,
                "is_mla_cp_enabled": lambda: False,
                "get_spec": lambda: SimpleNamespace(speculative_algorithm=None),
                "get_lora": lambda: SimpleNamespace(enable_lora=False),
                "get_exec": lambda: SimpleNamespace(
                    comm=SimpleNamespace(
                        boundary_reduction="rs+rsv", enable_quant_communications=False
                    ),
                    overlap=SimpleNamespace(enable_two_batch_overlap=False),
                ),
                "get_attn_tp_context": lambda: SimpleNamespace(input_scattered=False),
                "post_experts_reduction_group": lambda: self.parallel.tp_group,
                "should_use_dp_reduce_scatterv": lambda: self.rsv,
                "can_use_dp_reduce_scatter": lambda: False,
                "use_symmetric_memory": lambda *a, **k: nullcontext(),
                "is_allocation_symmetric": lambda: False,
                "attention_tensor_model_parallel_all_reduce": self.attention_sum,
                "attn_tp_reduce_scatter_tensor": self.attention_scatter,
                "attn_tp_all_gather_into_tensor": self.attention_gather,
                "get_global_dp_buffer": lambda group: torch.empty(
                    self.rows * self.dp, HIDDEN
                ),
                "get_local_dp_buffer": lambda group, hidden_size=None: torch.empty(
                    self.rows, hidden_size or HIDDEN
                ),
                "get_dp_global_num_tokens": lambda: [self.rows] * self.dp,
                "dp_gather_replicate": self.gather_dp,
                "dp_scatter": self.scatter_dp,
            }
            for name, replacement in overrides.items():
                stack.enter_context(patch_communicator(name, replacement))
            stack.enter_context(
                patch.object(layernorm_sp, "layernorm_sp_enabled", lambda: False)
            )
            stack.enter_context(
                get_forward().scoped(
                    mlp_reduce_scatter=False,
                    fuse_mlp_allreduce=False,
                    sp_active=False,
                    attn_input_scattered=False,
                )
            )
            self.namespace = dict(
                torch=torch,
                residual_batch=residual_batch,
                GatedResidualState=GatedResidualState,
                GatedResidual=lambda *a, **k: Connection(),
                HyperConnectionConfig=SimpleNamespace,
                HandoffRows=HandoffRows,
                declare_attn=declare_attn,
                declare_ffn=declare_ffn,
                make_stages=make_stages,
                get_parallel=lambda: self.parallel,
                get_moe_a2a_backend=lambda: backend,
                _get_ple_forward_mode=lambda batch: batch.forward_mode,
            )
            yield

    def model(self, *, sparse, layer_id=0, num_layers=1):
        model = load_mixin(self.namespace)()
        model.config = SimpleNamespace(
            num_experts=4 if sparse else 0,
            num_hidden_layers=num_layers,
            hc_count=2,
            hidden_size=HIDDEN,
            ple_layer_ids=[],
            hc_lowrank=2,
            rms_norm_eps=1e-5,
        )
        model._init_qwen4_exp_layer_extensions(model.config, layer_id)
        model.mlp = self.mlp
        return model

    def forward(self, model, value, *, ple_batch=None):
        hidden, residual = model._prepare_qwen4_exp_attn(
            value, None, self.batch, ple_batch=ple_batch
        )
        if not self.batch.forward_mode.is_idle():
            hidden = hidden * (2 / self.tp)
        hidden, residual = model._prepare_qwen4_exp_mlp(hidden, residual, self.batch)
        hidden = model._run_qwen4_exp_mlp(hidden, self.batch)
        return model._postprocess_qwen4_exp_layer(hidden, residual, self.batch)


def reference(value):
    if value.shape[-1] == HIDDEN:
        value = value.repeat(1, 2)
    after_attention = combine_reference(mix_reference(value) * 2, value)
    mlp_input = mix_reference(after_attention)
    return combine_reference(mlp_input * 3, after_attention), mlp_input


class TestQwen4Boundaries(CustomTestCase):
    def test_dense_and_moe_complete_attention_sum_before_nonlinear_mix(self):
        value = torch.arange(6, dtype=torch.float32).reshape(2, HIDDEN) / 10
        expected, mlp_input = reference(value)
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                harness = Harness(tp=2)
                with harness.running():
                    model = harness.model(sparse=sparse)
                    output, residual = harness.forward(model, value)
                torch.testing.assert_close(output, expected)
                torch.testing.assert_close(harness.compute_inputs[0], mlp_input)
                self.assertIsNone(residual)
                self.assertEqual(harness.events, ["attention_sum"])
                self.assertIsNone(harness.batch.residual_stream.pending)

    def test_dp_ffn_gathers_mixed_input_and_combines_after_return(self):
        value = torch.arange(6, dtype=torch.float32).reshape(2, HIDDEN) / 10
        expected, local_input = reference(value)
        for sparse, rsv in itertools.product((False, True), repeat=2):
            with self.subTest(sparse=sparse, reduce_scatterv=rsv):
                harness = Harness(dp=2, tp=2, rsv=rsv)
                with harness.running():
                    output, residual = harness.forward(
                        harness.model(sparse=sparse), value
                    )
                    self.assertFalse(get_forward().mlp_reduce_scatter)
                torch.testing.assert_close(output, expected)
                torch.testing.assert_close(
                    harness.compute_inputs[0],
                    torch.cat([local_input, local_input + 10]),
                )
                self.assertEqual(harness.compute_flags, [rsv])
                self.assertEqual(
                    harness.events,
                    [
                        "attention_sum",
                        "dp_gather",
                        "dp_reduce_scatterv" if rsv else "dp_scatter",
                    ],
                )
                self.assertIsNone(residual)

    def test_a2a_moves_gated_state_with_attention_tp_slice(self):
        value = torch.arange(12, dtype=torch.float32).reshape(4, HIDDEN) / 10
        expected, mlp_input = reference(value)
        for rank in (0, 1):
            with self.subTest(rank=rank):
                harness = Harness(tp=2, a2a=True, rows=4, rank=rank)
                harness.expected_gathered = expected
                with harness.running():
                    output, residual = harness.forward(
                        harness.model(sparse=True), value
                    )
                torch.testing.assert_close(output, expected)
                torch.testing.assert_close(
                    harness.compute_inputs[0], mlp_input.tensor_split(2)[rank]
                )
                self.assertEqual(
                    harness.events, ["attention_reduce_scatter", "attention_gather"]
                )
                self.assertIsNone(residual)

    def test_idle_a2a_still_participates_in_expert_dispatch(self):
        value = torch.empty(0, HIDDEN)
        for a2a in (False, True):
            with self.subTest(a2a=a2a):
                harness = Harness(a2a=a2a, rows=0)
                with harness.running():
                    output, residual = harness.forward(
                        harness.model(sparse=True), value
                    )
                self.assertEqual(len(harness.compute_inputs), int(a2a))
                self.assertEqual(output.shape, (0, 2 * HIDDEN))
                self.assertIsNone(residual)

    def test_moe_cp_takes_back_uneven_rows_before_gated_combine(self):
        values = [
            torch.arange(6, dtype=torch.float32).reshape(2, HIDDEN) / 10,
            torch.arange(3, dtype=torch.float32).reshape(1, HIDDEN) / 4,
        ]
        references = [reference(value) for value in values]
        gathered_inputs = torch.cat(
            [references[0][1], references[1][1], torch.zeros(1, HIDDEN)]
        )
        for rank, value in enumerate(values):
            with self.subTest(cp_rank=rank):
                harness = Harness(rows=len(value), cp_rows=[2, 1], cp_rank=rank)
                harness.expected_cp_inputs = gathered_inputs
                with harness.running():
                    output, residual = harness.forward(
                        harness.model(sparse=True), value
                    )
                torch.testing.assert_close(output, references[rank][0])
                torch.testing.assert_close(harness.compute_inputs[0], gathered_inputs)
                self.assertEqual(harness.events, ["cp_gather"])
                self.assertIsNone(residual)

    def test_idle_with_moe_cp_still_consumes_gated_residual_reads(self):
        harness = Harness(rows=0, cp_rows=[0, 0])
        harness.batch.forward_mode = ForwardMode.IDLE
        harness.batch.attn_cp_metadata = None
        with harness.running():
            output, residual = harness.forward(
                harness.model(sparse=True), torch.empty(0, HIDDEN)
            )
        self.assertEqual(output.shape, (0, 2 * HIDDEN))
        self.assertIsNone(residual)
        self.assertEqual(harness.compute_inputs, [])
        self.assertEqual(harness.events, [])

    def test_intermediate_layers_complete_stream_and_reuse_auxiliary_state(self):
        value = torch.arange(6, dtype=torch.float32).reshape(2, HIDDEN) / 10
        harness = Harness(tp=2)
        with harness.running():
            layers = [
                harness.model(sparse=True, layer_id=i, num_layers=2) for i in range(2)
            ]
            for repetition in range(2):
                hidden = value + repetition
                expected = hidden
                for model in layers:
                    hidden, residual = harness.forward(model, hidden)
                    expected = reference(expected)[0]
                    torch.testing.assert_close(hidden, expected)
                    self.assertIsNone(residual)
                    self.assertIsNone(harness.batch.residual_stream.pending)
                    self.assertIs(harness.batch.residual_stream.residual, hidden)

    def test_ple_reads_expanded_input_before_attention_mix(self):
        value = torch.ones(2, HIDDEN) / 4
        calls = []

        def ple(hidden, batch, ple_batch):
            calls.append(hidden.clone())
            return hidden / 2

        harness = Harness()
        with harness.running():
            model = harness.model(sparse=False)
            model.ple = ple
            output, _ = harness.forward(model, value, ple_batch=object())
        torch.testing.assert_close(calls[0], value.repeat(1, 2))
        torch.testing.assert_close(output, reference(value * 1.5)[0])

    def test_ple_idle_enters_collective_path_and_missing_active_batch_fails(self):
        from unittest.mock import Mock

        for rows in (0, 2):
            with self.subTest(rows=rows):
                harness = Harness(rows=rows)
                with harness.running():
                    model = harness.model(sparse=False)
                    model.ple = Mock()
                    value = torch.empty(rows, HIDDEN)
                    if rows:
                        with self.assertRaisesRegex(RuntimeError, "missing its batch"):
                            harness.forward(model, value)
                    else:
                        output, _ = harness.forward(model, value)
                        model.ple.forward_idle.assert_called_once_with(harness.batch)
                        self.assertEqual(output.shape, (0, 2 * HIDDEN))


if __name__ == "__main__":
    unittest.main()
