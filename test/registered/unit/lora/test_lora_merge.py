# SPDX-License-Identifier: Apache-2.0
"""Dense merge arithmetic, restoration, and transition serialization."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.arg_groups.lora_hook import check_lora_server_args
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.lora.layers import QKVParallelLinearWithLoRA, RowParallelLinearWithLoRA
from sglang.srt.lora.merge import (
    LoRAMergeUnrecoverableError,
    merge_weights,
    unmerge_weights,
)
from sglang.srt.managers.io_struct import LoRAMergeOutput, MergeLoRAWeightsReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.utils.aio_rwlock import RWLock
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def make_manager(fused=False, unmerge_mode="restore", device="cpu"):
    torch.manual_seed(123)
    rank, max_rank, in_dim = 3, 8, 7
    sizes = [6, 2, 2] if fused else [9]
    target = "qkv_proj" if fused else "down_proj"
    cls = QKVParallelLinearWithLoRA if fused else RowParallelLinearWithLoRA
    module = cls.__new__(cls)
    torch.nn.Module.__init__(module)
    module.base_layer = torch.nn.Module()
    module.base_layer.weight = torch.nn.Parameter(
        torch.randn(sum(sizes), in_dim, device=device).bfloat16()
    )
    module.base_layer.quant_method = UnquantizedLinearMethod()
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    module.output_offset_cpu = torch.tensor(offsets)
    a = torch.randn(1, max_rank * len(sizes), in_dim, device=device).bfloat16()
    b = torch.randn(1, sum(sizes), max_rank, device=device).bfloat16()
    pool = SimpleNamespace(
        target_modules={target},
        max_lora_rank=max_rank,
        get_buffer_id=lambda _: 0,
        A_buffer={target: [a]},
        B_buffer={target: [b]},
    )
    adapter = SimpleNamespace(
        config=SimpleNamespace(r=rank),
        scaling=0.7,
        embedding_layers={},
        added_tokens_embeddings={},
        layers=[SimpleNamespace(weights={f"model.layers.0.{target}.lora_A.weight": a})],
    )
    manager = SimpleNamespace(
        loras={"a": adapter},
        memory_pool=pool,
        lora_modules=[{f"model.layers.0.{target}": module}],
        fetch_new_loras=Mock(),
        reset_lora_batch=Mock(),
        pending_lora_load_events={},
        lora_unmerge_mode=unmerge_mode,
        merged_lora_id=None,
        merged_lora_weights=[],
    )
    return manager, module.base_layer.weight, a, b, offsets


class TestMergeArithmetic(unittest.TestCase):
    def test_fused_slices_and_padded_rank(self):
        for fused in (False, True):
            manager, weight, a, b, offsets = make_manager(fused)
            original = weight.detach().clone()
            address = weight.data_ptr()
            # Dense fused A is concatenated at the loaded rank; trailing
            # max-rank capacity is not a gap between its projections.
            block_b = torch.block_diag(
                *[
                    b[0, start:end, :3].float()
                    for start, end in zip(offsets, offsets[1:])
                ]
            )
            expected = (
                original.float()
                + 0.7 * block_b @ a[0, : 3 * (len(offsets) - 1)].float()
            )
            for _ in range(10):
                merge_weights(manager, "a")
                for source, backup in manager.merged_lora_weights:
                    self.assertEqual(backup.device.type, "cpu")
                    self.assertNotEqual(source.data_ptr(), backup.data_ptr())
                torch.testing.assert_close(weight, expected.bfloat16(), rtol=0, atol=0)
                self.assertEqual(weight.data_ptr(), address)
                unmerge_weights(manager)
                self.assertTrue(torch.equal(weight, original))
            self.assertEqual(manager.merged_lora_weights, [])

    def test_subtraction_keeps_no_backup_and_uses_scaled_fused_slices(self):
        for fused in (False, True):
            manager, weight, a, b, offsets = make_manager(fused, "subtract")
            original = weight.detach().clone()
            address = weight.data_ptr()
            merge_weights(manager, "a")
            self.assertEqual(manager.merged_lora_weights, [])
            expected = weight.detach().clone()
            for i, (start, end) in enumerate(zip(offsets, offsets[1:])):
                expected[start:end].addmm_(
                    b[0, start:end, :3], a[0, i * 3 : (i + 1) * 3], alpha=-0.7
                )
            unmerge_weights(manager)
            torch.testing.assert_close(weight, expected, rtol=0, atol=0)
            self.assertFalse(torch.equal(weight, original))
            self.assertEqual(weight.data_ptr(), address)
            self.assertIsNone(manager.merged_lora_id)
            unmerge_weights(manager)  # Must not subtract twice.
            torch.testing.assert_close(weight, expected, rtol=0, atol=0)

    def test_partial_merge_without_backup_requires_restart(self):
        manager, *_ = make_manager(True, "subtract")
        addmm = torch.addmm
        with patch(
            "sglang.srt.lora.merge.torch.addmm",
            side_effect=[
                addmm(torch.ones(6, 7), torch.ones(6, 3), torch.ones(3, 7)),
                RuntimeError("injected failure"),
            ],
        ):
            with self.assertRaisesRegex(
                LoRAMergeUnrecoverableError, "without a base-weight backup"
            ):
                merge_weights(manager, "a")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_backups_stay_on_cpu_and_restore_exactly(self):
        manager, weight, *_ = make_manager(True, device="cuda")
        original = weight.detach().clone()
        for _ in range(3):
            merge_weights(manager, "a")
            torch.cuda.synchronize()
            for source, backup in manager.merged_lora_weights:
                self.assertEqual(backup.device.type, "cpu")
                self.assertTrue(backup.is_pinned())
                self.assertNotEqual(source.data_ptr(), backup.data_ptr())
            unmerge_weights(manager)
            torch.cuda.synchronize()
            torch.testing.assert_close(weight, original, rtol=0, atol=0)

    def test_unrecoverable_failure_on_any_rank_stops_tensor_parallel_group(self):
        manager, *_ = make_manager(True, "subtract")
        scheduler = SimpleNamespace(
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(lora_manager=manager)
            ),
            is_fully_idle=lambda: True,
            weight_updater=SimpleNamespace(_session=None),
        )
        group = SimpleNamespace(
            all_gather_object=Mock(
                side_effect=[
                    [None, None],
                    [(None, False), ("partial merge failed", True)],
                ]
            )
        )
        with (
            patch(
                "sglang.srt.managers.scheduler.get_lora",
                return_value=SimpleNamespace(enable_lora_merge=True),
            ),
            patch(
                "sglang.srt.managers.scheduler.get_parallel",
                return_value=SimpleNamespace(tp_group=group),
            ),
            patch("sglang.srt.managers.scheduler.torch.cuda.synchronize"),
        ):
            with self.assertRaisesRegex(RuntimeError, "unsafe to serve"):
                Scheduler.merge_lora_weights(
                    scheduler, MergeLoRAWeightsReqInput(lora_name="a", lora_id="a")
                )

    def test_failure_restores_already_changed_slices(self):
        manager, weight, *_ = make_manager(True)
        original = weight.detach().clone()
        addmm = torch.addmm
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected allocation failure")
            return addmm(*args, **kwargs)

        with patch("sglang.srt.lora.merge.torch.addmm", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                merge_weights(manager, "a")
        self.assertTrue(torch.equal(weight, original))
        self.assertIsNone(manager.merged_lora_id)

    def test_other_tensor_parallel_rank_failure_rolls_back(self):
        manager, weight, *_ = make_manager(True)
        original = weight.detach().clone()
        scheduler = SimpleNamespace(
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(lora_manager=manager)
            ),
            is_fully_idle=lambda: True,
            weight_updater=SimpleNamespace(_session=None),
        )
        group = SimpleNamespace(
            all_gather_object=Mock(
                side_effect=[
                    [None, None],
                    [(None, False), ("rank 1 allocation failed", False)],
                ]
            )
        )
        with (
            patch(
                "sglang.srt.managers.scheduler.get_lora",
                return_value=SimpleNamespace(enable_lora_merge=True),
            ),
            patch(
                "sglang.srt.managers.scheduler.get_parallel",
                return_value=SimpleNamespace(tp_group=group),
            ),
            patch("sglang.srt.managers.scheduler.torch.cuda.synchronize"),
        ):
            result = Scheduler.merge_lora_weights(
                scheduler, MergeLoRAWeightsReqInput(lora_name="a", lora_id="a")
            )
        self.assertFalse(result.success)
        self.assertIn("rank 1 allocation", result.error_message)
        self.assertTrue(torch.equal(weight, original))
        self.assertIsNone(manager.merged_lora_id)

    def test_rejects_unsupported_weights_before_mutation(self):
        manager, weight, *_ = make_manager()
        original = weight.detach().clone()
        next(iter(manager.lora_modules[0].values())).base_layer.quant_method = object()
        with self.assertRaisesRegex(ValueError, "unquantized"):
            merge_weights(manager, "a")
        self.assertTrue(torch.equal(weight, original))


class TestMergeConfiguration(unittest.TestCase):
    def test_rejects_invalid_unmerge_mode_from_python_api(self):
        config = SimpleNamespace(
            max_loras_per_batch=8,
            lora_paths=None,
            enable_lora=True,
            enable_lora_merge=True,
            lora_unmerge_mode="invalid",
        )
        with patch(
            "sglang.srt.arg_groups.lora_hook.resolving_view", return_value=config
        ):
            with self.assertRaisesRegex(ValueError, "lora-unmerge-mode"):
                check_lora_server_args(config)

    def test_rejects_shared_weights_and_external_caches(self):
        base = dict(
            max_loras_per_batch=8,
            lora_paths=None,
            enable_lora=True,
            enable_lora_merge=True,
            lora_unmerge_mode="restore",
            dp_size=1,
            pp_size=1,
            attn_dp_size=1,
            tokenizer_worker_num=1,
            speculative_algorithm=None,
            disaggregation_mode="null",
            enable_memory_saver=False,
            cpu_offload_gb=0,
            offload_group_size=-1,
            weight_cache_mode="off",
            enable_hierarchical_cache=False,
            enable_lmcache=False,
        )
        for field, value in [
            ("weight_cache_mode", "client"),
            ("enable_hierarchical_cache", True),
            ("enable_lmcache", True),
            ("cpu_offload_gb", 1),
            ("offload_group_size", 1),
            ("dp_size", 2),
        ]:
            config = SimpleNamespace(**dict(base, **{field: value}))
            with patch(
                "sglang.srt.arg_groups.lora_hook.resolving_view", return_value=config
            ):
                with self.assertRaisesRegex(ValueError, "LoRA merging requires"):
                    check_lora_server_args(config)


class TestMergeDrain(unittest.IsolatedAsyncioTestCase):
    async def test_rejected_adapter_does_not_release_an_unacquired_reference(self):
        manager = SimpleNamespace(merged_lora_name="a", enable_lora=True)
        request = SimpleNamespace(
            is_single=True, rid="test", lora_path="b", lora_id=None
        )
        with self.assertRaisesRegex(ValueError, "merged adapter"):
            await TokenizerManager._validate_and_resolve_lora(manager, request)
        state = SimpleNamespace(lora_released=False, obj=request)
        self.assertIsNone(TokenizerManager._release_lora_once(manager, state))
        self.assertFalse(state.lora_released)

    async def test_drain_blocks_new_readers_and_survives_cancellation(self):
        manager = TokenizerControlMixin()
        manager.auto_create_handle_loop = Mock()
        manager.model_update_lock = RWLock()
        manager.lora_update_lock = asyncio.Lock()
        manager.merged_lora_name = None
        manager.lora_registry = SimpleNamespace(
            get_all_adapters=lambda: {"a": SimpleNamespace(lora_id="id-a")}
        )
        entered, finish = asyncio.Event(), asyncio.Event()

        async def communicate(obj):
            entered.set()
            await finish.wait()
            return [LoRAMergeOutput(success=True, merged_lora_name="a")]

        manager.merge_lora_weights_communicator = communicate
        with patch(
            "sglang.srt.managers.tokenizer_control_mixin.get_lora",
            return_value=SimpleNamespace(enable_lora_merge=True),
        ):
            async with manager.model_update_lock.reader_lock:
                task = asyncio.create_task(
                    manager.merge_lora_weights(MergeLoRAWeightsReqInput(lora_name="a"))
                )
                await asyncio.sleep(0.01)
                self.assertFalse(entered.is_set())
            await entered.wait()
            task.cancel()
            admitted = asyncio.Event()

            async def reader():
                async with manager.model_update_lock.reader_lock:
                    self.assertEqual(manager.merged_lora_name, "a")
                    admitted.set()

            next_request = asyncio.create_task(reader())
            await asyncio.sleep(0.01)
            self.assertFalse(admitted.is_set())
            finish.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            await next_request
            self.assertTrue(admitted.is_set())


if __name__ == "__main__":
    unittest.main()
