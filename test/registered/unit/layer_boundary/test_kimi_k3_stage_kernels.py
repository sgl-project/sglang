"""Kimi K3's own kernels for a layer's sums and gathers serve its
attention-residual bank only: the standard residual path keeps the
boundary's collectives, even where those kernels run.
With SGLANG_K3_SP_ATTN_RES the bank and the stream stay on each rank's shard
across SP-MoE layers."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import test_attn_bank_residual_ops as bank_fixture
import test_declared_decoder_boundary as fixture
import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers.communication import k3_ar_fusion, k3_sp_collective
from sglang.srt.layers.layer_boundary import BatchVariant, layer_stack
from sglang.srt.layers.layer_boundary import prepare as comm_ops
from sglang.srt.layers.layer_boundary.ops import attn_tp_gather_input, keep_output
from sglang.srt.layers.layer_boundary.residual import attn_bank
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layer_boundary.residual.attn_bank import (
    AttnBank,
    AttnBankOutputRead,
)
from sglang.srt.models import kimi_k3
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

# A latent MoE on every layer after the first.
CONFIG = SimpleNamespace(
    is_moe=True,
    num_experts=8,
    first_k_dense_replace=1,
    moe_layer_freq=1,
    routed_expert_hidden_size=2,
    num_hidden_layers=4,
    is_kda_layer=lambda layer_idx: False,
)


@contextmanager
def k3_planning(parallel, **options):
    """fixture.planning, also for the parallel layout K3's own declaration
    code reads (prefill CP, pipeline width)."""
    if not hasattr(parallel, "pp_size"):
        parallel.pp_size = 1
    with (
        fixture.planning(parallel, **options),
        patch.object(kimi_k3, "get_parallel", lambda: parallel),
    ):
        yield


def moe_layer(*, bank, sp_moe, all_reduce_fusion):
    """A K3 latent MoE layer's attributes that its stage declarations read."""
    layer = kimi_k3.KimiK3DecoderLayer.__new__(kimi_k3.KimiK3DecoderLayer)
    nn.Module.__init__(layer)
    layer.use_attn_residuals = bank
    layer.is_block_write_layer = False
    layer._sp_moe = sp_moe
    layer.all_reduce_fusion = all_reduce_fusion
    layer.input_layernorm = fixture.Norm()
    layer.post_attention_layernorm = fixture.Norm()
    for name in (
        "self_attention_res_proj",
        "self_attention_res_norm",
        "mlp_res_proj",
        "mlp_res_norm",
    ):
        setattr(layer, name, None)
    return layer


def declare(layers, *, sp_moe, carries=False, sp_collective=True, config=CONFIG):
    """Declare ``layers`` (index -> layer) in one layer stack, as if the
    stack enabled bank carry (``carries``), using K3's eligibility check.
    The final bank read gathers an SP-MoE layer's rows through the selected
    collective backend."""
    bank = AttnBank()
    has_bank = any(layer.use_attn_residuals for layer in layers.values())
    # Every layer of a model with a bank block size carries the bank.
    config = SimpleNamespace(
        **{**vars(config), "attn_res_block_size": 4 if has_bank else None}
    )
    parallel = fixture.parallel_of(attn_dp=1, attn_tp=2, pp_size=1)
    with (
        patch.object(k3_sp_collective, "enabled", return_value=sp_collective),
        patch.object(k3_ar_fusion, "enabled", return_value=True),
        patch.object(kimi_k3, "_shards_moe_rows", return_value=sp_moe),
        patch.object(kimi_k3, "get_parallel", return_value=parallel),
        patch.object(
            kimi_k3,
            "get_spec",
            return_value=SimpleNamespace(speculative_algorithm=None),
        ),
        envs.SGLANG_K3_SP_ATTN_RES.override(carries),
        k3_planning(
            parallel,
            a2a=sp_moe,
            boundary_reduction="ar",
        ),
    ):
        final_read = (
            AttnBankOutputRead(
                bank,
                None,
                None,
                fixture.Norm(),
                attn_tp_gather=(
                    k3_sp_collective.all_gather if sp_moe and sp_collective else None
                ),
                reads_attn_tp_slices=kimi_k3._carries_bank_slices(config),
            )
            if has_bank
            else None
        )
        with layer_stack(final_read=final_read):
            for idx, layer in layers.items():
                layer._declare_stages(config, idx, bank)


class TestKimiK3StageKernels(CustomTestCase):
    def ffn_path(self, *, bank, sp_moe, all_reduce_fusion):
        layer = moe_layer(bank=bank, sp_moe=sp_moe, all_reduce_fusion=all_reduce_fusion)
        declare({1: layer}, sp_moe=sp_moe)
        return layer.ffn_boundary.plan.paths[BatchVariant.ORDINARY]

    @staticmethod
    def kernels(path):
        step = path.entry.prepare.keywords["step"]
        gather = getattr(path.output_move, "keywords", {}).get("gather")
        fusions = tuple(f.run for f in step.keywords["read_fusions"])
        return step.func, fusions, gather

    def test_the_standard_path_keeps_the_boundary_collectives(self):
        for sp_moe in (True, False):
            with self.subTest(sp_moe=sp_moe):
                entry, fusions, gather = self.kernels(
                    self.ffn_path(bank=False, sp_moe=sp_moe, all_reduce_fusion=True)
                )
                self.assertIs(
                    entry,
                    comm_ops._attn_tp_reduce_scatter_update_read
                    if sp_moe
                    else comm_ops._reduce_update_read,
                )
                self.assertEqual(fusions, ())
                self.assertIsNone(gather)

    def test_the_bank_takes_the_tuned_reduce_scatter_and_gather(self):
        entry, fusions, gather = self.kernels(
            self.ffn_path(bank=True, sp_moe=True, all_reduce_fusion=False)
        )
        self.assertIs(entry, comm_ops._attn_tp_reduce_scatter_update_read)
        self.assertEqual(fusions, (kimi_k3._k3_reduce_scatter_add,))
        self.assertIs(gather, k3_sp_collective.all_gather)

    def test_the_bank_takes_the_fused_all_reduce(self):
        entry, fusions, gather = self.kernels(
            self.ffn_path(bank=True, sp_moe=False, all_reduce_fusion=True)
        )
        self.assertIs(entry, comm_ops._reduce_update_read)
        self.assertEqual(fusions, (kimi_k3._k3_all_reduce_add,))
        self.assertIsNone(gather)


class TestKimiK3BankOnShards(CustomTestCase):
    def test_dspark_captures_only_owned_bank_rows(self):
        generator = torch.Generator().manual_seed(17)
        hidden = torch.randn(4, bank_fixture.HIDDEN, generator=generator)
        bank = AttnBank()
        values = bank.open(hidden, 3).block_residual
        values.normal_(generator=generator)
        reference_bank = values.clone()
        layers = [
            SimpleNamespace(
                self_attention_res_proj=layer.attn_proj,
                self_attention_res_norm=layer.attn_score_norm,
                prev_valid_blocks=(i + 1) // 2,
            )
            for i, layer in enumerate(bank_fixture.LAYER_LIST)
        ]
        model = SimpleNamespace(
            layers=layers,
            end_layer=5,
            config=SimpleNamespace(attn_res_block_size=2),
            attn_bank=bank,
            output_attn_res_proj=bank_fixture.OUT_PROJ,
            output_attn_res_norm=bank_fixture.OUT_SCORE_NORM,
        )
        # Before the first bank write, at a block boundary, and at the last
        # layer, which uses the output-side weights instead of a next layer.
        for tap in (-1, 0, 1, 3, 4):
            proj, norm = (
                (bank_fixture.OUT_PROJ, bank_fixture.OUT_SCORE_NORM)
                if tap == 4
                else (
                    layers[tap + 1].self_attention_res_proj,
                    layers[tap + 1].self_attention_res_norm,
                )
            )
            # Keep the reference's CPU GEMM shape fixed across ranks; this
            # test checks bank-row ownership, not batch-dependent BLAS rounding.
            want = torch.cat(
                [
                    bank_fixture._mix(h, b, (tap + 2) // 2, proj, norm)
                    for h, b in zip(hidden.chunk(2), reference_bank.chunk(2))
                ]
            )
            for rank in (0, 1):
                rows = slice(rank * 2, (rank + 1) * 2)
                values.fill_(float("nan"))
                values[rows].copy_(reference_bank[rows])
                batch = SimpleNamespace()
                local = residual_batch.set_written(hidden[rows], batch)

                def gather(mixed, dim):
                    self.assertEqual(dim, 0)
                    torch.testing.assert_close(mixed, want[rows], rtol=0, atol=0)
                    return want.clone()

                parallel = SimpleNamespace(
                    attn_tp_size=2,
                    attn_tp_rank=rank,
                    attn_tp_group=SimpleNamespace(all_gather=gather),
                )
                with (
                    self.subTest(tap=tap, rank=rank),
                    patch.object(attn_bank, "get_parallel", return_value=parallel),
                    patch.object(kimi_k3, "get_parallel", return_value=parallel),
                ):
                    got = kimi_k3.KimiK3LinearModel._dspark_capture_stream(
                        model, tap, local, batch
                    )
                    torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_nccl_keeps_bank_shards_without_custom_collectives(self):
        layers = {
            idx: moe_layer(bank=True, sp_moe=True, all_reduce_fusion=False)
            for idx in (1, 2, 3)
        }
        # Exercise the real eligibility check and stage planner. No custom
        # communicator or device tuning table is available on this path.
        declare(layers, sp_moe=True, carries=True, sp_collective=False)
        for idx, layer in layers.items():
            with self.subTest(layer=idx):
                attn = layer.attn_boundary.plan.paths[BatchVariant.ORDINARY]
                ffn = layer.ffn_boundary.plan.paths[BatchVariant.ORDINARY]
                self.assertIs(ffn.output_move, keep_output)
                step = ffn.entry.prepare.keywords["step"]
                self.assertIs(step.func, comm_ops._attn_tp_reduce_scatter_update_read)
                self.assertEqual(step.keywords["read_fusions"], ())
                self.assertEqual(step.keywords["scatters_residual"], idx == 1)
                if idx > 1:
                    self.assertIs(attn.entry.input_move, attn_tp_gather_input)
                    read = attn.entry.prepare.keywords["step"]
                    self.assertEqual(read.keywords["read_gathers"], ())

    def stack(self, *, carries):
        layers = {
            idx: moe_layer(bank=True, sp_moe=True, all_reduce_fusion=False)
            for idx in (1, 2, 3)
        }
        declare(layers, sp_moe=True, carries=carries)
        return [
            (
                layer.attn_boundary.plan.paths[BatchVariant.ORDINARY],
                layer.ffn_boundary.plan.paths[BatchVariant.ORDINARY],
            )
            for layer in layers.values()
        ]

    def test_consecutive_moe_layers_stay_on_their_shards(self):
        for idx, (attn, ffn) in enumerate(self.stack(carries=True)):
            with self.subTest(layer=idx + 1):
                # The stack's last FFN too: the final read takes its shard.
                self.assertIs(ffn.output_move, keep_output)
                step = ffn.entry.prepare.keywords["step"]
                self.assertIs(step.func, comm_ops._attn_tp_reduce_scatter_update_read)
                self.assertEqual(step.keywords["scatters_residual"], idx == 0)
                # The bank's fused reduce-scatter and read first, then K3's
                # reduce-scatter with the residual add.
                fusions = step.keywords["read_fusions"]
                self.assertEqual([f.reads for f in fusions], [True, False])
                self.assertIs(fusions[1].run, kimi_k3._k3_reduce_scatter_add)
                if idx == 0:
                    self.assertIsNone(attn.entry.input_move)
                    continue
                # Read on the shard, then gathered: by the bank's kernel that
                # does both, else K3's tuned all-gather, else the boundary's.
                self.assertIs(attn.entry.input_move.func, attn_tp_gather_input)
                self.assertIs(
                    attn.entry.input_move.keywords["gather"],
                    k3_sp_collective.all_gather,
                )
                read = attn.entry.prepare.keywords["step"]
                self.assertEqual(len(read.keywords["read_gathers"]), 1)

    def test_the_shard_mode_is_decided_at_construction(self):
        # A rank holds only its own rows of the bank: no pipeline rank after
        # it and no dense layer after an MoE layer. DSpark can capture by
        # gathering the mixture; other aux-capture paths need the whole bank.
        cases = {
            (None, 1, 1): True,
            ("EAGLE", 1, 1): True,
            (None, 2, 1): False,
            ("EAGLE3", 1, 1): False,
            ("DFLASH", 1, 1): False,
            ("DSPARK", 1, 1): True,
            ("DSPARK", 2, 1): False,
            ("DSPARK", 1, 2): False,
            (None, 1, 2): False,
        }
        for (algorithm, pp_size, moe_layer_freq), carries in cases.items():
            config = SimpleNamespace(
                **{
                    **vars(CONFIG),
                    "attn_res_block_size": 4,
                    "num_hidden_layers": 6,
                    "moe_layer_freq": moe_layer_freq,
                }
            )
            with (
                self.subTest(
                    algorithm=algorithm,
                    pp_size=pp_size,
                    moe_layer_freq=moe_layer_freq,
                ),
                patch.object(kimi_k3, "_shards_moe_rows", return_value=True),
                envs.SGLANG_K3_SP_ATTN_RES.override(True),
                patch.object(
                    kimi_k3, "get_parallel", lambda: SimpleNamespace(pp_size=pp_size)
                ),
                patch.object(
                    kimi_k3,
                    "get_spec",
                    lambda: SimpleNamespace(speculative_algorithm=algorithm),
                ),
            ):
                self.assertEqual(kimi_k3._carries_bank_slices(config), carries)


if __name__ == "__main__":
    unittest.main()
