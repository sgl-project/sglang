"""Unit tests for srt/layers/layernorm_sp (Megatron LayerNorm sequence parallelism).

Covers the pure logic that gates SP -- the Qwen3 allowlist, the config guards, and
the prefill-only activation rule -- without launching a server. The collectives and
fused matmul fast-paths need a real TP group and are covered by the e2e test.
"""

import gc
import unittest
import weakref
from functools import partial
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.arg_groups.layernorm_sp_hook import (
    validate_layernorm_sp,
    validate_layernorm_sp_min_tokens,
)
from sglang.srt.layers import layer_boundary as comm
from sglang.srt.layers import layernorm_sp
from sglang.srt.layers.layer_boundary import (
    Layout,
    OutputContract,
    StageKind,
    SumGroup,
)
from sglang.srt.layers.layer_boundary import prepare as comm_ops
from sglang.srt.layers.layer_boundary.contracts import BatchVariant
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import (
    get_flags,
    get_forward,
    publish,
    reset_context,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.boundary_fixtures import (
    postprocess_output,
    prepare_input,
    sp_region_steps,
    stub_plan,
    stub_stage,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.communicator_patch import patch_communicator
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _initialize(*, enable=True, arch="Qwen3ForCausalLM", min_tokens=0):
    publish(
        ServerArgs(
            model_path="dummy",
            enable_layernorm_sp=enable,
            layernorm_sp_min_tokens=min_tokens,
        ),
        role="tokenizer",
    )
    layernorm_sp.initialize_layernorm_sp(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[arch] if arch else [])
        ),
    )


class TestLayerNormSPGating(CustomTestCase):
    def tearDown(self):
        reset_context()

    def test_initialize_enables_only_for_allowlisted_arch(self):
        _initialize(enable=True, arch="Qwen3ForCausalLM")
        self.assertTrue(layernorm_sp.layernorm_sp_enabled())
        # Flag on but unsupported architecture -> stays off.
        _initialize(enable=True, arch="LlamaForCausalLM")
        self.assertFalse(layernorm_sp.layernorm_sp_enabled())
        # Supported architecture but flag off -> stays off.
        _initialize(enable=False, arch="Qwen3ForCausalLM")
        self.assertFalse(layernorm_sp.layernorm_sp_enabled())

    def test_runs_sp_only_on_extend(self):
        _initialize()
        self.assertTrue(layernorm_sp.runs_sp(ForwardMode.EXTEND, 1))
        # SP is prefill-only; decode must never engage it.
        self.assertFalse(layernorm_sp.runs_sp(ForwardMode.DECODE, 1))
        # A disabled model never activates, even on EXTEND.
        _initialize(enable=False)
        self.assertFalse(layernorm_sp.runs_sp(ForwardMode.EXTEND, 1))

    def test_runs_sp_above_the_min_tokens_only(self):
        _initialize(min_tokens=1280)
        self.assertFalse(layernorm_sp.runs_sp(ForwardMode.EXTEND, 512))
        self.assertFalse(layernorm_sp.runs_sp(ForwardMode.EXTEND, 1280))
        self.assertTrue(layernorm_sp.runs_sp(ForwardMode.EXTEND, 1281))
        self.assertFalse(layernorm_sp.runs_sp(ForwardMode.DECODE, 8192))
        # the threshold only narrows an enabled model
        _initialize(enable=False, min_tokens=1280)
        self.assertFalse(layernorm_sp.runs_sp(ForwardMode.EXTEND, 8192))

    def test_runs_sp_needs_the_count_only_on_extend(self):
        """A caller asking about decode / verify passes may omit the count; an
        extend pass without one must raise instead of skipping the threshold."""
        for min_tokens in (0, 1280):
            _initialize(min_tokens=min_tokens)
            for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
                self.assertFalse(layernorm_sp.runs_sp(mode))
            with self.assertRaisesRegex(ValueError, "num_tokens"):
                layernorm_sp.runs_sp(ForwardMode.EXTEND)
        # SP off answers before reading the count
        _initialize(enable=False)
        self.assertFalse(layernorm_sp.runs_sp(ForwardMode.EXTEND))

    def test_exit_gather_takes_the_entry_decision(self):
        """Regression: with --layernorm-sp-min-tokens, a pass at or below the
        threshold enters replicated, so the exit gather must not gather it."""
        hidden = torch.ones(1280, 4)
        _initialize(min_tokens=1280)
        with patch.object(
            layernorm_sp, "sp_exit_gather", side_effect=lambda h, num_tokens: h
        ) as gather:
            for rows, gathered in ((1280, False), (1281, True)):
                gather.reset_mock()
                layernorm_sp.maybe_exit_gather(
                    hidden_states=hidden,
                    hidden_states_before_norm=None,
                    input_ids=torch.zeros(rows, dtype=torch.long),
                    forward_mode=ForwardMode.EXTEND,
                )
                self.assertEqual(gather.called, gathered)

    def test_runs_sp_ignores_the_active_flag(self):
        """Regression: the exit gather must not key off ``sp_active``.

        ``sp_active`` is written by Python inside the CUDA-graph-captured region,
        so it is stale on graph replay. Callers outside that region (the exit
        gather in LogitsProcessor) recompute with ``runs_sp`` instead; if that ever
        starts consulting the flag, a replayed prefill skips the gather and feeds
        sequence-sharded hidden states to the LM head.
        """
        _initialize()
        get_forward().set("sp_active", False)
        self.assertTrue(layernorm_sp.runs_sp(ForwardMode.EXTEND, 1))
        reset_context()


class TestSPCollectiveWorkspaceOwnership(CustomTestCase):
    def test_growth_retains_storage_used_by_older_graphs(self):
        from sglang.srt.layers.communication import sp_collective_gemm_workspace as sp

        class Workspace:
            def __init__(self, kind, group, rows, *args):
                self.capacity = 1 << (rows - 1).bit_length()
                self.handle = SimpleNamespace(barrier=MagicMock())

        x = SimpleNamespace(dtype=torch.bfloat16, device=torch.device("cuda:0"))
        group, stream = object(), SimpleNamespace(synchronize=MagicMock())
        with (
            patch.dict(sp._workspaces, clear=True),
            patch.object(sp, "_retired_workspaces", [], create=True),
            patch.object(sp, "SPCollectiveGemmWorkspace", side_effect=Workspace),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(torch.cuda, "current_stream", return_value=stream),
        ):
            old = sp.sp_collective_gemm_workspace("ag", group, 64, 32, x)
            previous = weakref.ref(old)
            del old
            current = sp.sp_collective_gemm_workspace("ag", group, 128, 32, x)
            gc.collect()
            self.assertIsNotNone(previous())
            self.assertIsNot(previous(), current)
            self.assertIs(
                sp.sp_collective_gemm_workspace("ag", group, 32, 32, x), current
            )


class TestLayerNormSPValidation(CustomTestCase):
    """``validate_layernorm_sp`` is pure; pass config in directly."""

    VALID: ClassVar[dict[str, object]] = {
        "architecture": "Qwen3ForCausalLM",
        "tp_size": 2,
        "ep_size": 2,
        "pp_size": 1,
        "attn_cp_size": 1,
        "attn_dp_enabled": False,
        "speculative_algorithm": None,
    }

    def test_valid_config_passes(self):
        validate_layernorm_sp(**self.VALID)  # must not raise

    def test_rejects_unsupported_arch(self):
        with self.assertRaisesRegex(ValueError, "only supported"):
            validate_layernorm_sp(**{**self.VALID, "architecture": "LlamaForCausalLM"})

    def test_rejects_tp_size_one(self):
        with self.assertRaisesRegex(ValueError, "tp_size"):
            validate_layernorm_sp(**{**self.VALID, "tp_size": 1})

    def test_rejects_dp_attention(self):
        with self.assertRaisesRegex(ValueError, "attention DP"):
            validate_layernorm_sp(**{**self.VALID, "attn_dp_enabled": True})

    def test_rejects_speculative(self):
        with self.assertRaisesRegex(ValueError, "speculative"):
            validate_layernorm_sp(**{**self.VALID, "speculative_algorithm": "EAGLE3"})

    def test_qwen4_exp_uses_the_standard_flag_with_general_tp_ep_constraint(self):
        validate_layernorm_sp(
            **{
                **self.VALID,
                "architecture": "Qwen4ExpForCausalLM",
            }
        )

    def test_qwen4_exp_requires_ep_equal_tp(self):
        with self.assertRaisesRegex(ValueError, "ep_size == tp_size"):
            validate_layernorm_sp(
                **{
                    **self.VALID,
                    "architecture": "Qwen4ExpForCausalLM",
                    "tp_size": 4,
                    "ep_size": 2,
                }
            )

    def test_qwen4_exp_rejects_attention_context_parallelism(self):
        with self.assertRaisesRegex(ValueError, "attn_cp_size == 1"):
            validate_layernorm_sp(
                **{
                    **self.VALID,
                    "architecture": "Qwen4ExpForConditionalGeneration",
                    "attn_cp_size": 2,
                }
            )

    def test_rejects_pipeline_parallelism(self):
        with self.assertRaisesRegex(ValueError, "pipeline"):
            validate_layernorm_sp(**{**self.VALID, "pp_size": 2})

    def test_min_tokens_must_be_a_captured_prefill_bucket(self):
        graphs = dict(prefill_backend=Backend.FULL, prefill_bs=[512, 1280, 2048])
        for ok in (0, 512, 1280, 2048, 4096):
            validate_layernorm_sp_min_tokens(min_tokens=ok, **graphs)  # must not raise
        with self.assertRaisesRegex(ValueError, "captured"):
            validate_layernorm_sp_min_tokens(min_tokens=1000, **graphs)
        with self.assertRaisesRegex(ValueError, ">= 0"):
            validate_layernorm_sp_min_tokens(min_tokens=-1, **graphs)

    def test_min_tokens_is_free_without_prefill_graphs(self):
        validate_layernorm_sp_min_tokens(
            min_tokens=1000, prefill_backend=Backend.DISABLED, prefill_bs=[512, 1280]
        )  # must not raise


class _Norm:
    def __call__(self, x, residual=None, post_residual_addition=None):
        if residual is None:
            return x * 2
        s = x + residual
        return s * 2, s


class TestReplicatedTokenRows(CustomTestCase):
    def test_partial_output_is_summed_while_scattering(self):
        def reduce_scatter(output, full_partial):
            output.copy_(full_partial.tensor_split(2, dim=0)[1] * 2)

        group = SimpleNamespace(
            world_size=2,
            rank_in_group=1,
            all_gather_into_tensor=lambda output, local: output.copy_(
                torch.cat([local, local])
            ),
            reduce_scatter_tensor=MagicMock(side_effect=reduce_scatter),
        )
        output = layernorm_sp.run_replicated_token_rows(
            torch.tensor([[1.0], [2.0]]),
            lambda full_rows: full_rows + 10,
            output_is_tp_partial=True,
            group=group,
        )
        torch.testing.assert_close(output, torch.tensor([[22.0], [24.0]]))
        group.reduce_scatter_tensor.assert_called_once()

    def test_complete_output_is_sharded_without_sum(self):
        group = SimpleNamespace(
            world_size=2,
            rank_in_group=1,
            all_gather_into_tensor=lambda output, local: output.copy_(
                torch.cat([local, local])
            ),
            reduce_scatter_tensor=MagicMock(
                side_effect=AssertionError("complete output must not be summed")
            ),
        )
        output = layernorm_sp.run_replicated_token_rows(
            torch.tensor([[1.0], [2.0]]),
            lambda full_rows: full_rows + 10,
            output_is_tp_partial=False,
            group=group,
        )
        torch.testing.assert_close(output, torch.tensor([[11.0], [12.0]]))
        group.reduce_scatter_tensor.assert_not_called()


class TestSpRegionSteps(CustomTestCase):
    """The communicator opens the SP region at the model's first layer on a
    prefill forward and, inside it, keeps every boundary on this rank's sequence
    shard: add + norm, no move, no collective."""

    def setUp(self):
        # runs_sp reads --layernorm-sp-min-tokens from the published config
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        self.addCleanup(reset_context)

    def test_sp_flag_remains_visible_to_traced_linears(self):
        from sglang.srt.runtime_context import ForwardFlags

        self.assertIn("sp_active", ForwardFlags._GRAPH_VISIBLE)
        before = get_forward().sp_active
        with get_forward().scoped(sp_active=not before):
            self.assertEqual(get_forward().sp_active, not before)
        self.assertEqual(get_forward().sp_active, before)

    def communicator(self, *, first_layer):
        c = stub_plan()
        c.paths[BatchVariant.SEQUENCE_PARALLEL] = sp_region_steps()
        c.paths[BatchVariant.INPUT_SCATTERED] = None
        c.paths[BatchVariant.CONTEXT_PARALLEL] = None
        c.enters_stack = first_layer
        c.is_sparse = False
        c._attn_input_fusions = ()
        c.norm = _Norm()
        c.qkv_latent_func = None
        c.paths[BatchVariant.ORDINARY] = self.ordinary_steps(
            MagicMock(side_effect=AssertionError("moved"))
        )
        return c

    def ordinary_steps(self, attention_input):
        """The layer's steps outside the region, which must not run inside it."""
        return comm.StagePath(
            entry=comm.EntryPath(
                prepare=partial(
                    comm_ops._run_entry,
                    step=partial(
                        comm_ops._update_read,
                        pre_move=None,
                        enters_stack=False,
                        read=comm.NORM_QUANT_READOUT,
                        update=comm.PLAIN_ADD,
                    ),
                    carried_fusions=(),
                    is_plain_add=True,
                ),
                input_rows=comm.Layout(frozenset()),
                input_move=attention_input,
                attn_input_adapter=comm_ops._attn_input_default,
            ),
            output=OutputContract(Layout(frozenset()), group=SumGroup.TP),
            output_move=MagicMock(side_effect=AssertionError("postprocess ran")),
        )

    def run_prepare_attn(self, communicator, mode, hidden, residual):
        batch = SimpleNamespace(forward_mode=mode)
        with (
            get_flags().sp.override(enabled=True),
            patch_communicator("_batch_shards_over_cp", return_value=False),
            get_forward().scoped(sp_active=False),
            patch_communicator(
                "get_attn_tp_context",
                return_value=SimpleNamespace(input_scattered=False),
            ),
            patch.object(
                layernorm_sp, "sp_entry_scatter", side_effect=lambda h: h[:1]
            ) as scatter,
        ):
            out = prepare_input(
                stub_stage(communicator, StageKind.ATTENTION), hidden, residual, batch
            )
            return out, get_forward().sp_active, scatter

    def test_the_first_layer_opens_the_region_on_prefill(self):
        hidden = torch.ones(2, 4)
        (h, r), active, scatter = self.run_prepare_attn(
            self.communicator(first_layer=True), ForwardMode.EXTEND, hidden, None
        )
        self.assertTrue(active)
        scatter.assert_called_once()
        torch.testing.assert_close(h, torch.full((1, 4), 2.0))
        torch.testing.assert_close(r.residual, torch.ones(1, 4))

    def test_decode_leaves_the_region_closed(self):
        communicator = self.communicator(first_layer=True)
        move = MagicMock(side_effect=lambda **k: k["hidden_states"])
        communicator.paths[BatchVariant.ORDINARY] = self.ordinary_steps(move)
        (_h, _), active, scatter = self.run_prepare_attn(
            communicator, ForwardMode.DECODE, torch.ones(2, 4), None
        )
        self.assertFalse(active)
        scatter.assert_not_called()
        move.assert_called_once()

    def test_a_later_layer_does_not_reopen_the_region(self):
        communicator = self.communicator(first_layer=False)
        communicator.paths[BatchVariant.ORDINARY] = self.ordinary_steps(
            MagicMock(side_effect=lambda **k: k["hidden_states"])
        )
        _, active, scatter = self.run_prepare_attn(
            communicator, ForwardMode.EXTEND, torch.ones(2, 4), torch.ones(2, 4)
        )
        self.assertFalse(active)
        scatter.assert_not_called()

    def test_inside_the_region_each_boundary_stays_on_the_shard(self):
        communicator = self.communicator(first_layer=False)
        hidden, residual = torch.ones(1, 4), torch.full((1, 4), 3.0)
        with (
            get_flags().sp.override(enabled=True),
            patch_communicator("_batch_shards_over_cp", return_value=False),
            get_forward().scoped(sp_active=True),
        ):
            with patch_communicator(
                "get_attn_tp_context",
                return_value=SimpleNamespace(input_scattered=False),
            ):
                h, r = prepare_input(
                    stub_stage(communicator, StageKind.ATTENTION),
                    hidden,
                    residual.clone(),
                    SimpleNamespace(forward_mode=ForwardMode.EXTEND),
                )
            torch.testing.assert_close(h, torch.full((1, 4), 8.0))
            h, r = prepare_input(
                stub_stage(communicator, StageKind.FFN),
                hidden,
                residual.clone(),
                object(),
            )
            torch.testing.assert_close(h, torch.full((1, 4), 8.0))
            torch.testing.assert_close(r.residual, torch.full((1, 4), 4.0))
            out = postprocess_output(communicator.output, hidden, residual, object())
            self.assertIs(out[0], hidden)
            self.assertIs(out[1], residual)


if __name__ == "__main__":
    unittest.main()
