"""Staged LoRA publication: fresh versions stream and commit alongside
generation, while base and fixed-name sessions keep the writer lock."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.lora.lora_registry import LoRARef, LoRARegistry
from sglang.srt.managers.io_struct import (
    BeginWeightUpdateReqInput,
    EndWeightUpdateReqInput,
    GenerateReqInput,
    LoRAUpdateOutput,
    RegisterLoRAAdapterReqInput,
    UnloadLoRAAdapterReqInput,
    UpdateWeightsFromDistributedReqInput,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_context

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _reply(success=True):
    return SimpleNamespace(success=success, message="ok" if success else "worker failed")


def _manager():
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.server_args = SimpleNamespace(max_loaded_loras=None)
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
    tm._pending_lora_publications = {}
    tm.update_lora_adapter_communicator = AsyncMock(
        side_effect=lambda obj: [LoRAUpdateOutput(success=True)]
    )
    tm.begin_weight_update_communicator = AsyncMock(return_value=[_reply()])
    tm.end_weight_update_communicator = AsyncMock(return_value=[_reply()])
    tm.update_weights_from_distributed_communicator = AsyncMock(return_value=[_reply()])
    return tm


def _begin(**kwargs):
    return BeginWeightUpdateReqInput(**(dict(sync_base=False) | kwargs))


def _bucket(**kwargs):
    values = dict(
        names=["A@2:model.layers.0.self_attn.q_proj.lora_A.weight"],
        dtypes=["float32"],
        shapes=[[1]],
        flush_cache=False,
    )
    return UpdateWeightsFromDistributedReqInput(**(values | kwargs))


def _end(**kwargs):
    values = dict(expected_lora_checksums={"A@2": {"q_proj": "hash"}})
    return EndWeightUpdateReqInput(**(values | kwargs))


class TestStagedPublication(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        config = get_context().override_server_args(
            enable_lora=True,
            pp_size=1,
            dp_size=1,
            enable_dp_attention=False,
            checkpoint_engine_wait_weights_before_ready=False,
        )
        config.install()
        self.addCleanup(config.restore)
        self.tm = _manager()
        result = await self.tm.register_lora_adapter(
            RegisterLoRAAdapterReqInput(
                lora_name="A@2", config_dict={"r": 8}, defer_publish=True
            )
        )
        self.assertTrue(result.success)
        self.assertTrue(result.pending)
        self.assertIn("A@2", self.tm._pending_lora_publications)

    async def test_deferred_name_is_unservable_until_commit(self):
        tm = self.tm
        old_ids = await tm.lora_registry.acquire(["A@1", "B", "C"])
        self.assertIsNone(await tm.lora_registry.get_lora_id("A@2"))
        self.assertTrue((await tm.begin_weight_update(_begin()))[0])
        self.assertTrue((await tm.update_weights_from_distributed(_bucket()))[0])
        self.assertIsNone(await tm.lora_registry.get_lora_id("A@2"))
        self.assertTrue((await tm.end_weight_update(_end()))[0])
        new_id = await tm.lora_registry.acquire("A@2")
        self.assertNotIn(new_id, old_ids)
        # Untouched adapters keep their identities through the publication.
        self.assertEqual(await tm.lora_registry.get_lora_id("A@1"), old_ids[0])
        await tm.lora_registry.release(new_id)
        await tm.lora_registry.release(old_ids)

    async def test_staged_session_completes_under_live_readers(self):
        tm = self.tm
        async with tm.model_update_lock.reader_lock:
            self.assertTrue(
                (await asyncio.wait_for(tm.begin_weight_update(_begin()), 1))[0]
            )
            self.assertTrue(tm._weight_update_staged_session)
            self.assertTrue(
                (
                    await asyncio.wait_for(
                        tm.update_weights_from_distributed(_bucket()), 1
                    )
                )[0]
            )
            self.assertTrue((await asyncio.wait_for(tm.end_weight_update(_end()), 1))[0])
        self.assertIsNotNone(await tm.lora_registry.get_lora_id("A@2"))

    async def test_base_and_fixed_name_sessions_still_wait_for_readers(self):
        for sync_base in (True, False):
            tm = _manager()  # no pending publications: not a staged session
            async with tm.model_update_lock.reader_lock:
                task = asyncio.create_task(
                    tm.begin_weight_update(BeginWeightUpdateReqInput(sync_base=sync_base))
                )
                await asyncio.sleep(0)
                self.assertFalse(task.done())
                tm.begin_weight_update_communicator.assert_not_awaited()
            self.assertTrue((await asyncio.wait_for(task, 1))[0])

    async def test_base_sync_with_pending_registration_keeps_writer_lock(self):
        tm = self.tm
        async with tm.model_update_lock.reader_lock:
            task = asyncio.create_task(
                tm.begin_weight_update(BeginWeightUpdateReqInput(sync_base=True))
            )
            await asyncio.sleep(0)
            self.assertFalse(task.done())
        self.assertTrue((await asyncio.wait_for(task, 1))[0])
        self.assertFalse(tm._weight_update_staged_session)

    async def test_defer_publish_requires_a_fresh_name(self):
        for taken in ("A@1", "A@2"):  # published name; pending name
            result = await self.tm.register_lora_adapter(
                RegisterLoRAAdapterReqInput(
                    lora_name=taken, config_dict={}, defer_publish=True
                )
            )
            self.assertFalse(result.success)

    async def test_pending_name_cannot_be_reregistered_or_unloaded(self):
        result = await self.tm.register_lora_adapter(
            RegisterLoRAAdapterReqInput(lora_name="A@2", config_dict={})
        )
        self.assertFalse(result.success)
        result = await self.tm.unload_lora_adapter(
            UnloadLoRAAdapterReqInput(lora_name="A@2")
        )
        self.assertFalse(result.success)

    async def test_staged_bucket_cannot_change_serving_state(self):
        tm = self.tm
        await tm.begin_weight_update(_begin())
        for kwargs in (
            {"flush_cache": True},
            {"abort_all_requests": True},
            {"weight_version": "2"},
        ):
            self.assertFalse(
                (await tm.update_weights_from_distributed(_bucket(**kwargs)))[0]
            )
        tm.abort_request.assert_not_called()
        tm.update_weights_from_distributed_communicator.assert_not_awaited()

    async def test_any_worker_failure_discards_the_publication(self):
        tm = self.tm
        await tm.begin_weight_update(_begin())
        tm.end_weight_update_communicator.return_value = [_reply(), _reply(False)]
        tm.update_lora_adapter_communicator.reset_mock()
        self.assertFalse((await tm.end_weight_update(_end()))[0])
        self.assertIsNone(await tm.lora_registry.get_lora_id("A@2"))
        self.assertEqual(tm._pending_lora_publications, {})
        # The backends held a zeroed identity for the name; it must be dropped.
        discarded = tm.update_lora_adapter_communicator.await_args.args[0]
        self.assertIsInstance(discarded, UnloadLoRAAdapterReqInput)
        self.assertEqual(discarded.lora_name, "A@2")

    async def test_missing_manifest_fails_closed(self):
        tm = self.tm
        await tm.begin_weight_update(_begin())
        success, message = await tm.end_weight_update(
            _end(expected_lora_checksums=None)
        )
        self.assertFalse(success)
        self.assertIn("A@2", message)
        sent = tm.end_weight_update_communicator.await_args.args[0]
        self.assertTrue(sent.abort)
        self.assertIsNone(await tm.lora_registry.get_lora_id("A@2"))

    async def test_abort_never_publishes(self):
        tm = self.tm
        await tm.begin_weight_update(_begin())
        self.assertTrue((await tm.end_weight_update(_end(abort=True)))[0])
        self.assertIsNone(await tm.lora_registry.get_lora_id("A@2"))
        self.assertEqual(tm._pending_lora_publications, {})

    async def test_version_is_recorded_only_at_successful_end(self):
        tm = _manager()
        self.assertTrue(
            (await tm.begin_weight_update(BeginWeightUpdateReqInput(sync_base=True)))[0]
        )
        self.assertTrue(
            (await tm.update_weights_from_distributed(_bucket(weight_version="2")))[0]
        )
        tm.record_config_updates.assert_not_called()
        self.assertTrue((await tm.end_weight_update(EndWeightUpdateReqInput()))[0])
        tm.record_config_updates.assert_called_once_with(
            "tokenizer.weight_version", weight_version="2"
        )


class TestRequestCarriedBackfill(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        config = get_context().override_server_args(
            enable_lora=True,
            pp_size=1,
            dp_size=1,
            enable_dp_attention=False,
            checkpoint_engine_wait_weights_before_ready=False,
        )
        config.install()
        self.addCleanup(config.restore)
        self.tm = _manager()

        async def load(obj):
            await self.tm.lora_registry.register(
                LoRARef(lora_name=obj.lora_name, lora_path=obj.lora_path)
            )
            return SimpleNamespace(success=True, error_message="")

        self.tm.load_lora_adapter = AsyncMock(side_effect=load)

    async def test_backfill_path_admits_an_unknown_version(self):
        obj = GenerateReqInput(
            text="x",
            lora_path="A@7",
            lora_backfill_paths={"A@7": "/ckpt/run/sampler_weights/7"},
        )
        await self.tm._resolve_lora_path(obj)
        self.assertIsNotNone(obj.lora_id)
        loaded = self.tm.load_lora_adapter.await_args.args[0]
        self.assertEqual(
            (loaded.lora_name, loaded.lora_path),
            ("A@7", "/ckpt/run/sampler_weights/7"),
        )
        await self.tm.lora_registry.release(obj.lora_id)

    async def test_unknown_version_without_a_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "never been loaded"):
            await self.tm._resolve_lora_path(GenerateReqInput(text="x", lora_path="A@7"))

    async def test_pending_name_is_not_servable_or_backfillable(self):
        self.tm._pending_lora_publications["A@7"] = LoRARef(
            lora_name="A@7", lora_path="__stream__", reloadable=False
        )
        with self.assertRaisesRegex(ValueError, "awaiting publication"):
            await self.tm._resolve_lora_path(
                GenerateReqInput(
                    text="x", lora_path="A@7", lora_backfill_paths={"A@7": "/ckpt/7"}
                )
            )


class TestStashCopiesIPCBuckets(unittest.TestCase):
    def test_stash_survives_sender_reusing_the_bucket(self):
        manager = SchedulerWeightUpdaterManager.__new__(SchedulerWeightUpdaterManager)
        manager.tp_worker = SimpleNamespace(ps=SimpleNamespace(tp_rank=0))
        manager._lora_stash = {}
        bucket = torch.ones(4)
        manager._stash_lora_tensors([("A@2:q_proj", bucket)], copy_tensors=True)
        bucket.zero_()
        self.assertTrue(torch.equal(manager._lora_stash["A@2"]["q_proj"], torch.ones(4)))


if __name__ == "__main__":
    unittest.main()
