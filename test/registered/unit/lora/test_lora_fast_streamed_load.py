"""SGLANG_OPT_LORA_FAST_STREAMED_LOAD: the streamed-adapter stash keeps only this
rank's TP shard of packed per-expert MoE LoRA factors, the MoE LoRA slicers
pass such shards through and slice packed 3D factors with one batched copy, and
the memory pool installs packed factors with one copy per module. Every path
must produce the same tensors as the per-expert path it replaces."""

import types
import unittest
from unittest import mock
from unittest.mock import MagicMock, Mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.lora import layers as lora_layers
from sglang.srt.lora import mem_pool as lora_mem_pool
from sglang.srt.lora.layers import FusedMoEWithLoRA
from sglang.srt.lora.mem_pool import LoRAMemoryPool
from sglang.srt.managers.io_struct import LoRAUpdateOutput
from sglang.srt.managers.scheduler_components import weight_updater
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

NUM_EXPERTS, INTER, RANK, HIDDEN = 32, 768, 16, 256
PREFIX = "model.layers.1.mlp.experts"


def _moe_lora_stub(*, tp_size, tp_rank=0, is_gated=True, interleaved=False):
    """The attributes of FusedMoEWithLoRA the slicers read, with its methods."""
    stub = types.SimpleNamespace(
        tp_size=tp_size,
        tp_rank=tp_rank,
        intermediate_size_per_partition=INTER // tp_size,
        _uses_interleaved_gate_up=interleaved,
        base_layer=types.SimpleNamespace(
            moe_runner_config=types.SimpleNamespace(is_gated=is_gated),
            is_shared_fused_moe=False,
        ),
    )
    for name in (
        "slice_moe_lora_a_weights",
        "slice_moe_lora_b_weights",
        "_slice_moe_a",
        "_slice_moe_b_2d",
        "_slice_moe_b_3d",
    ):
        setattr(stub, name, types.MethodType(getattr(FusedMoEWithLoRA, name), stub))
    return stub


class TestMoeLoraSlicers(CustomTestCase):
    def _cases(self):
        for is_gated in (True, False):
            for interleaved in (False, True):
                for tp_size in (1, 4, 16):
                    for tp_rank in sorted({0, tp_size // 2, tp_size - 1}):
                        yield is_gated, interleaved, tp_size, tp_rank

    def test_batched_3d_slice_matches_per_expert(self):
        torch.manual_seed(0)
        for is_gated, interleaved, tp_size, tp_rank in self._cases():
            with self.subTest(
                gated=is_gated, interleaved=interleaved, tp=tp_size, rank=tp_rank
            ):
                rows = 2 * INTER if is_gated else INTER
                full_b = torch.randn(NUM_EXPERTS, rows, RANK).bfloat16()
                stub = _moe_lora_stub(
                    tp_size=tp_size, is_gated=is_gated, interleaved=interleaved
                )
                with mock.patch.object(lora_layers, "_LORA_FAST_STREAMED_LOAD", False):
                    ref = stub.slice_moe_lora_b_weights(
                        full_b, tp_rank, "gate_up_proj_moe"
                    )
                with mock.patch.object(lora_layers, "_LORA_FAST_STREAMED_LOAD", True):
                    got = stub.slice_moe_lora_b_weights(
                        full_b, tp_rank, "gate_up_proj_moe"
                    )
                self.assertTrue(torch.equal(ref, got))
                self.assertTrue(got.is_contiguous())

    def test_pre_sliced_factors_pass_through(self):
        """A factor the stash already cut to this rank's shard gives the same
        per-rank tensors as slicing the full factor."""
        torch.manual_seed(1)
        for is_gated, interleaved, tp_size, tp_rank in self._cases():
            if tp_size == 1:
                continue
            with self.subTest(
                gated=is_gated, interleaved=interleaved, tp=tp_size, rank=tp_rank
            ):
                shard = INTER // tp_size
                rows = slice(tp_rank * shard, (tp_rank + 1) * shard)
                w1_b = torch.randn(NUM_EXPERTS, INTER, RANK).bfloat16()
                w3_b = torch.randn(NUM_EXPERTS, INTER, RANK).bfloat16()
                w2_a = torch.randn(NUM_EXPERTS, RANK, INTER).bfloat16()
                full_b = torch.cat([w1_b, w3_b], 1) if is_gated else w1_b
                pre_b = (
                    torch.cat([w1_b[:, rows], w3_b[:, rows]], 1)
                    if is_gated
                    else w1_b[:, rows].clone()
                )
                stub = _moe_lora_stub(
                    tp_size=tp_size, is_gated=is_gated, interleaved=interleaved
                )
                with mock.patch.object(lora_layers, "_LORA_FAST_STREAMED_LOAD", True):
                    ref_b = stub.slice_moe_lora_b_weights(
                        full_b, tp_rank, "gate_up_proj_moe"
                    )
                    got_b_3d = stub.slice_moe_lora_b_weights(
                        pre_b, tp_rank, "gate_up_proj_moe"
                    )
                    got_b_2d = torch.stack(
                        [
                            stub.slice_moe_lora_b_weights(
                                pre_b[e], tp_rank, "gate_up_proj_moe"
                            )
                            for e in range(NUM_EXPERTS)
                        ]
                    )
                    ref_a = stub.slice_moe_lora_a_weights(
                        w2_a, tp_rank, "down_proj_moe"
                    )
                    got_a = stub.slice_moe_lora_a_weights(
                        w2_a[:, :, rows], tp_rank, "down_proj_moe"
                    )
                self.assertTrue(torch.equal(ref_b, got_b_3d))
                self.assertTrue(torch.equal(ref_b, got_b_2d))
                self.assertTrue(torch.equal(ref_a, got_a))
                self.assertTrue(got_a.is_contiguous())


def _make_manager(lora_manager, tp_rank=0) -> SchedulerWeightUpdaterManager:
    tp_worker = MagicMock()
    tp_worker.model_runner.tp_rank = tp_rank
    tp_worker.model_runner.lora_manager = lora_manager
    return SchedulerWeightUpdaterManager(
        tp_worker=tp_worker,
        draft_worker=None,
        tp_cpu_group=MagicMock(),
        memory_saver_adapter=MagicMock(),
        flush_cache=Mock(return_value=True),
        is_fully_idle=Mock(return_value=True),
    )


class TestStashTpShard(CustomTestCase):
    TP_SIZE, TP_RANK = 16, 3

    def _manager(self):
        moe = _moe_lora_stub(tp_size=self.TP_SIZE, tp_rank=self.TP_RANK)
        lora_manager = MagicMock()
        lora_manager.apply_streamed_adapter.return_value = LoRAUpdateOutput(
            success=True
        )
        lora_manager.lora_modules = [
            {"mlp.down_proj": object()},
            {"mlp.experts": moe},
        ]
        return _make_manager(lora_manager, tp_rank=self.TP_RANK), lora_manager

    def _tensors(self):
        torch.manual_seed(2)
        return {
            f"{PREFIX}.w1.lora_B.weight": torch.randn(NUM_EXPERTS, INTER, RANK),
            f"{PREFIX}.w3.lora_B.weight": torch.randn(NUM_EXPERTS, INTER, RANK),
            f"{PREFIX}.w2.lora_A.weight": torch.randn(NUM_EXPERTS, RANK, INTER),
            # shared outer factors (one per layer) and dense factors pass through
            f"{PREFIX}.w1.lora_A.weight": torch.randn(1, RANK, HIDDEN),
            f"{PREFIX}.w2.lora_B.weight": torch.randn(1, HIDDEN, RANK),
            "model.layers.1.mlp.shared_experts.gate_proj.lora_A.weight": torch.randn(
                RANK, HIDDEN
            ),
            "model.layers.0.mlp.down_proj.lora_B.weight": torch.randn(HIDDEN, RANK),
        }

    def _stash(self, enabled):
        mgr, lora_manager = self._manager()
        tensors = self._tensors()
        with mock.patch.object(weight_updater, "_LORA_FAST_STREAMED_LOAD", enabled):
            mgr._stash_lora_tensors(
                [(f"miles_lora:{k}", v) for k, v in tensors.items()],
                copy_tensors=True,
            )
        return mgr, lora_manager, tensors, mgr._lora_stash["miles_lora"]

    def test_stash_keeps_only_this_ranks_shard(self):
        _, _, tensors, stash = self._stash(True)
        shard = INTER // self.TP_SIZE
        rows = slice(self.TP_RANK * shard, (self.TP_RANK + 1) * shard)
        for proj in ("w1", "w3"):
            key = f"{PREFIX}.{proj}.lora_B.weight"
            self.assertTrue(torch.equal(stash[key], tensors[key][:, rows]))
        key = f"{PREFIX}.w2.lora_A.weight"
        self.assertTrue(torch.equal(stash[key], tensors[key][:, :, rows]))
        for key in (f"{PREFIX}.w1.lora_B.weight", f"{PREFIX}.w2.lora_A.weight"):
            # the stash holds a compact copy, not a view of the full factor
            self.assertEqual(
                stash[key].untyped_storage().nbytes(),
                stash[key].numel() * stash[key].element_size(),
            )
        for key in (
            f"{PREFIX}.w1.lora_A.weight",
            f"{PREFIX}.w2.lora_B.weight",
            "model.layers.1.mlp.shared_experts.gate_proj.lora_A.weight",
            "model.layers.0.mlp.down_proj.lora_B.weight",
        ):
            self.assertTrue(torch.equal(stash[key], tensors[key]))

    def test_disabled_stash_keeps_full_factors(self):
        _, _, tensors, stash = self._stash(False)
        for key, tensor in tensors.items():
            self.assertTrue(torch.equal(stash[key], tensor))

    def test_sharded_stash_applies(self):
        mgr, lora_manager, _, _ = self._stash(True)
        with mock.patch.object(weight_updater, "_LORA_FAST_STREAMED_LOAD", True):
            success, message = mgr._apply_lora_stash(None)
        self.assertTrue(success, message)
        lora_manager.apply_streamed_adapter.assert_called_once()
        applied = lora_manager.apply_streamed_adapter.call_args.args[1]
        self.assertEqual(
            tuple(applied[f"{PREFIX}.w1.lora_B.weight"].shape),
            (NUM_EXPERTS, INTER // self.TP_SIZE, RANK),
        )

    def test_checksums_are_rejected(self):
        """The checksum manifest covers the full factors, not the shard."""
        mgr, lora_manager, _, stash = self._stash(True)
        manifest = {"miles_lora": {k: "0" for k in stash}}
        with mock.patch.object(weight_updater, "_LORA_FAST_STREAMED_LOAD", True):
            success, message = mgr._apply_lora_stash(manifest)
        self.assertFalse(success)
        self.assertIn("SGLANG_OPT_LORA_FAST_STREAMED_LOAD", message)
        lora_manager.apply_streamed_adapter.assert_not_called()


def _pool_stub(*, ep_size=1, ep_rank=0, overlap_loading=False, pinned=False):
    stub = types.SimpleNamespace(
        moe_use_local_expert_ids=ep_size > 1,
        moe_ep_rank=ep_rank,
        _num_experts_local=NUM_EXPERTS // ep_size,
        pin_memory_available=pinned,
        enable_lora_overlap_loading=overlap_loading,
        is_shared_moe_module=lambda name: "shared" in name,
    )
    for name in ("_local_expert_weights_packed", "_iter_local_expert_weights"):
        setattr(stub, name, types.MethodType(getattr(LoRAMemoryPool, name), stub))
    return stub


class TestPackedExpertInstall(CustomTestCase):
    def test_packed_matches_per_expert(self):
        torch.manual_seed(3)
        weights = torch.randn(NUM_EXPERTS, 2 * RANK, HIDDEN)
        for ep_size in (1, 4):
            for ep_rank in range(ep_size):
                for module in ("gate_up_proj_moe", "shared_gate_up_proj_moe"):
                    with self.subTest(ep=ep_size, rank=ep_rank, module=module):
                        pool = _pool_stub(ep_size=ep_size, ep_rank=ep_rank)
                        per_expert = list(
                            pool._iter_local_expert_weights(
                                weights,
                                "key",
                                localize=not pool.is_shared_moe_module(module),
                            )
                        )
                        with mock.patch.object(
                            lora_mem_pool, "_LORA_FAST_STREAMED_LOAD", True
                        ):
                            count, packed = pool._local_expert_weights_packed(
                                weights, module
                            )
                        self.assertEqual(count, len(per_expert))
                        for local_eid, w, _ in per_expert:
                            self.assertTrue(torch.equal(packed[local_eid], w))

    def test_per_expert_path_kept(self):
        weights = torch.randn(NUM_EXPERTS, RANK, HIDDEN)
        with mock.patch.object(lora_mem_pool, "_LORA_FAST_STREAMED_LOAD", False):
            self.assertIsNone(
                _pool_stub()._local_expert_weights_packed(weights, "gate_up_proj_moe")
            )
        with mock.patch.object(lora_mem_pool, "_LORA_FAST_STREAMED_LOAD", True):
            # dicts of experts, and CPU weights under the pinned per-expert cache
            self.assertIsNone(
                _pool_stub()._local_expert_weights_packed(
                    {0: weights[0]}, "gate_up_proj_moe"
                )
            )
            self.assertIsNone(
                _pool_stub(
                    overlap_loading=True, pinned=True
                )._local_expert_weights_packed(weights, "gate_up_proj_moe")
            )


if __name__ == "__main__":
    unittest.main()
