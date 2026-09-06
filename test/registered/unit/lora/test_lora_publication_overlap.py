"""New LoRA versions publish while existing inference leases remain live."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.lora.lora_registry import LoRARef, LoRARegistry
from sglang.srt.managers.io_struct import (
    BeginWeightUpdateReqInput,
    EndWeightUpdateReqInput,
    LoRAUpdateOutput,
    RegisterLoRAAdapterReqInput,
    UnloadLoRAAdapterReqInput,
    UpdateWeightsFromDistributedReqInput,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _reply(success=True):
    return SimpleNamespace(
        success=success, message="ok" if success else "worker failed"
    )


def _manager():
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.server_args = SimpleNamespace(
        enable_lora=True,
        pp_size=1,
        max_loaded_loras=None,
        checkpoint_engine_wait_weights_before_ready=False,
    )
    tm.elastic_worker_count = 1
    tm.init_weight_update()
    tm.auto_create_handle_loop = Mock()
    tm._validate_lora_upsert_supported = Mock()
    tm.record_config_updates = Mock()
    tm.abort_request = Mock()
    tm.mm_processor = None
    tm.lora_update_lock = asyncio.Lock()
    tm.lora_registry = LoRARegistry(
        [LoRARef(lora_name=name, reloadable=False) for name in ("A@1", "B", "C")]
    )
    tm.lora_ref_cache = {}
    tm.update_lora_adapter_communicator = AsyncMock(
        side_effect=lambda obj: [LoRAUpdateOutput(success=True)]
    )
    tm.begin_weight_update_communicator = AsyncMock(return_value=[_reply()])
    tm.end_weight_update_communicator = AsyncMock(return_value=[_reply()])
    tm.update_weights_from_distributed_communicator = AsyncMock(return_value=[_reply()])
    return tm


def _begin(**kwargs):
    return BeginWeightUpdateReqInput(
        sync_base=False, new_lora_names=["A@2"], session_id="publication-2", **kwargs
    )


def _bucket(**kwargs):
    values = dict(
        names=["A@2:model.layers.0.self_attn.q_proj.lora_A.weight"],
        dtypes=["float32"],
        shapes=[[1]],
        flush_cache=False,
        session_id="publication-2",
    )
    return UpdateWeightsFromDistributedReqInput(**(values | kwargs))


def _end(**kwargs):
    values = dict(
        session_id="publication-2", expected_lora_checksums={"A@2": {"q_proj": "hash"}}
    )
    return EndWeightUpdateReqInput(**(values | kwargs))


class TestPublicationOverlap(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        parallel = patch(
            "sglang.srt.managers.tokenizer_control_mixin.get_parallel",
            return_value=SimpleNamespace(dp_size=1, enable_dp_attention=False),
        )
        parallel.start()
        self.addCleanup(parallel.stop)
        self.tm = _manager()
        result = await self.tm.register_lora_adapter(
            RegisterLoRAAdapterReqInput(
                lora_name="A@2", config_dict={"r": 8}, defer_publish=True
            )
        )
        self.assertTrue(result.success)
        self.assertTrue(result.pending)

    async def test_pipeline_parallel_registration_fails_before_allocating(self):
        self.tm.server_args.pp_size = 2
        self.tm.update_lora_adapter_communicator.reset_mock()
        result = await self.tm.register_lora_adapter(
            RegisterLoRAAdapterReqInput(
                lora_name="A@3", config_dict={"r": 8}, defer_publish=True
            )
        )
        self.assertFalse(result.success)
        self.assertIn("pp_size=1", result.error_message)
        self.tm.update_lora_adapter_communicator.assert_not_awaited()
        self.assertIsNone(await self.tm.lora_registry.get_lora_id("A@3"))

    async def test_full_session_finishes_with_existing_readers_and_lora_leases(self):
        tm = self.tm
        old_ids = await tm.lora_registry.acquire(["A@1", "B", "C"])
        async with tm.model_update_lock.reader_lock:
            self.assertTrue(
                (await asyncio.wait_for(tm.begin_weight_update(_begin()), 1))[0]
            )
            with self.assertRaisesRegex(ValueError, "not ready"):
                await tm.lora_registry.acquire("A@2")
            self.assertTrue(
                (
                    await asyncio.wait_for(
                        tm.update_weights_from_distributed(_bucket()), 1
                    )
                )[0]
            )

            entered, finish = asyncio.Event(), asyncio.Event()

            async def finish_workers(obj):
                entered.set()
                await finish.wait()
                return [_reply(), _reply()]

            tm.end_weight_update_communicator.side_effect = finish_workers
            task = asyncio.create_task(tm.end_weight_update(_end()))
            try:
                await asyncio.wait_for(entered.wait(), 1)
                with self.assertRaisesRegex(ValueError, "not ready"):
                    await tm.lora_registry.acquire("A@2")
                finish.set()
                self.assertTrue((await asyncio.wait_for(task, 1))[0])
                new_id = await tm.lora_registry.acquire("A@2")
                self.assertNotIn(new_id, old_ids)
                # B/C/A@1 are still live when the publication has completed.
                self.assertEqual(await tm.lora_registry.get_lora_id("A@1"), old_ids[0])
                await tm.lora_registry.release(new_id)
            finally:
                finish.set()
                await task
        await tm.lora_registry.release(old_ids)

    async def test_base_and_in_place_sessions_still_wait_for_readers(self):
        for sync_base in (True, False):
            tm = _manager()
            async with tm.model_update_lock.reader_lock:
                task = asyncio.create_task(
                    tm.begin_weight_update(
                        BeginWeightUpdateReqInput(sync_base=sync_base)
                    )
                )
                await asyncio.sleep(0)
                self.assertFalse(task.done())
                tm.begin_weight_update_communicator.assert_not_awaited()
            self.assertTrue((await asyncio.wait_for(task, 1))[0])
            self.assertTrue((await tm.end_weight_update(EndWeightUpdateReqInput()))[0])

    async def test_legacy_base_version_is_recorded_only_after_successful_end(self):
        tm = self.tm
        self.assertTrue((await tm.begin_weight_update(BeginWeightUpdateReqInput()))[0])
        self.assertTrue(
            (
                await tm.update_weights_from_distributed(
                    _bucket(session_id=None, weight_version="2")
                )
            )[0]
        )
        tm.record_config_updates.assert_not_called()
        self.assertTrue((await tm.end_weight_update(EndWeightUpdateReqInput()))[0])
        tm.record_config_updates.assert_called_once_with(
            "tokenizer.weight_version", weight_version="2"
        )

    async def test_existing_version_cannot_be_registered_or_used_as_new(self):
        calls = self.tm.update_lora_adapter_communicator.await_count
        result = await self.tm.register_lora_adapter(
            RegisterLoRAAdapterReqInput(
                lora_name="A@1", config_dict={}, defer_publish=True
            )
        )
        self.assertFalse(result.success)
        self.assertEqual(self.tm.update_lora_adapter_communicator.await_count, calls)
        result = await self.tm.begin_weight_update(
            BeginWeightUpdateReqInput(
                sync_base=False, new_lora_names=["A@1"], session_id="other"
            )
        )
        self.assertFalse(result[0])
        self.tm.begin_weight_update_communicator.assert_not_awaited()

    async def test_pending_version_cannot_be_upserted_or_unloaded_during_session(self):
        await self.tm.begin_weight_update(_begin())
        result = await self.tm.register_lora_adapter(
            RegisterLoRAAdapterReqInput(lora_name="A@2", config_dict={})
        )
        self.assertFalse(result.success)
        result = await self.tm.unload_lora_adapter(
            UnloadLoRAAdapterReqInput(lora_name="A@2")
        )
        self.assertFalse(result.success)
        self.assertTrue((await self.tm.end_weight_update(_end(abort=True)))[0])
        self.assertTrue(
            (
                await self.tm.unload_lora_adapter(
                    UnloadLoRAAdapterReqInput(lora_name="A@2")
                )
            ).success
        )

    async def test_overlapping_session_and_foreign_bucket_or_end_are_rejected(self):
        await self.tm.begin_weight_update(_begin())
        self.assertFalse(
            (await self.tm.begin_weight_update(BeginWeightUpdateReqInput()))[0]
        )
        self.assertFalse(
            (
                await self.tm.update_weights_from_distributed(
                    _bucket(session_id="other")
                )
            )[0]
        )
        self.assertFalse((await self.tm.end_weight_update(_end(session_id="other")))[0])
        self.tm.update_weights_from_distributed_communicator.assert_not_awaited()
        self.tm.end_weight_update_communicator.assert_not_awaited()
        self.assertTrue((await self.tm.end_weight_update(_end()))[0])

    async def test_new_version_cannot_abort_flush_or_change_base_version(self):
        await self.tm.begin_weight_update(_begin())
        for kwargs in (
            {"flush_cache": True},
            {"abort_all_requests": True},
            {"weight_version": "2"},
        ):
            self.assertFalse(
                (await self.tm.update_weights_from_distributed(_bucket(**kwargs)))[0]
            )
        self.tm.abort_request.assert_not_called()
        self.tm.update_weights_from_distributed_communicator.assert_not_awaited()

    async def test_any_worker_failure_keeps_version_unservable(self):
        await self.tm.begin_weight_update(_begin())
        self.tm.end_weight_update_communicator.return_value = [_reply(), _reply(False)]
        self.assertFalse((await self.tm.end_weight_update(_end()))[0])
        with self.assertRaisesRegex(ValueError, "not ready"):
            await self.tm.lora_registry.acquire("A@2")

    async def test_failed_bucket_poisoning_discards_instead_of_publishing(self):
        await self.tm.begin_weight_update(_begin())
        self.tm.update_weights_from_distributed_communicator.return_value = [
            _reply(False)
        ]
        self.assertFalse((await self.tm.update_weights_from_distributed(_bucket()))[0])
        self.assertFalse((await self.tm.end_weight_update(_end()))[0])
        self.assertTrue(self.tm.end_weight_update_communicator.call_args.args[0].abort)
        with self.assertRaisesRegex(ValueError, "not ready"):
            await self.tm.lora_registry.acquire("A@2")

    async def test_manifest_is_required_and_abort_never_publishes(self):
        await self.tm.begin_weight_update(_begin())
        self.assertFalse(
            (await self.tm.end_weight_update(_end(expected_lora_checksums=None)))[0]
        )
        self.tm.end_weight_update_communicator.assert_not_awaited()
        self.assertTrue((await self.tm.end_weight_update(_end(abort=True)))[0])
        with self.assertRaisesRegex(ValueError, "not ready"):
            await self.tm.lora_registry.acquire("A@2")


if __name__ == "__main__":
    unittest.main()
