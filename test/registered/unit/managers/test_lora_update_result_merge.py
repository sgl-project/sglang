# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Unit tests for merging per-rank LoRA update replies from the control fan-out."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_registry import LoRARef, LoRARegistry
from sglang.srt.managers.io_struct import (
    LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterReqInput,
    LoRAUpdateOutput,
    UnloadLoRAAdapterReqInput,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_control_mixin import (
    TokenizerControlMixin,
    _merge_lora_update_results,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _ok(adapters=None) -> LoRAUpdateOutput:
    return LoRAUpdateOutput(success=True, loaded_adapters=adapters or {})


def _err(message, adapters=None) -> LoRAUpdateOutput:
    return LoRAUpdateOutput(
        success=False, error_message=message, loaded_adapters=adapters or {}
    )


class TestMergeLoRAUpdateResults(CustomTestCase):
    def test_all_success_returns_first_rank_result(self):
        """On success the merge must hand back a rank's own reply: callers
        mutate result.loaded_adapters in place during LRU eviction, which a
        synthesized empty result would silently break."""
        results = [_ok({"a": "path"}), _ok({"a": "path"})]
        merged = _merge_lora_update_results(results)
        self.assertIs(merged, results[0])
        self.assertTrue(merged.success)

    def test_any_rank_failure_wins(self):
        """Regression guard for the pre-merge behavior of returning
        results[0]: a failure on a non-zero rank was reported as success,
        letting the tokenizer-side registry drift from that rank's actual
        adapter state."""
        merged = _merge_lora_update_results(
            [_ok({"a": "path"}), _err("out of memory", {"stale": "path"})]
        )
        self.assertFalse(merged.success)
        self.assertEqual(merged.error_message, "out of memory")
        self.assertEqual(merged.loaded_adapters, {"stale": "path"})

    def test_failed_load_does_not_redirect_existing_adapter_unload(self):
        """Rejected duplicates must keep the original adapter's unload ID;
        fresh partial loads must remain cleanable via their new ID."""

        async def scenario(from_tensors, duplicate):
            original = LoRARef(lora_name="a", lora_path="path", pinned=False)
            manager = TokenizerControlMixin()
            manager.auto_create_handle_loop = Mock()
            manager.lora_update_lock = asyncio.Lock()
            manager.lora_registry = LoRARegistry([original] if duplicate else [])
            manager.pending_lora_unloads = {}
            manager.lora_ref_cache = {}
            replies = (
                [_err("already loaded", {"a": "path"})]
                if duplicate
                else [_ok({"a": "path"}), _err("rank failed")]
            )
            manager.update_lora_adapter_communicator = AsyncMock(
                side_effect=[replies, [_ok()]]
            )
            if from_tensors:
                request = LoadLoRAAdapterFromTensorsReqInput(
                    lora_name="a", config_dict={}, serialized_named_tensors=[b""]
                )
                result = await manager.load_lora_adapter_from_tensors(request)
            else:
                request = LoadLoRAAdapterReqInput(lora_name="a", lora_path="path")
                result = await manager.load_lora_adapter(request)

            self.assertFalse(result.success)
            self.assertEqual(result.loaded_adapters, {"a": "path"} if duplicate else {})
            self.assertEqual(
                manager.pending_lora_unloads,
                {} if duplicate else {"a": request.lora_id},
            )
            self.assertEqual(
                manager.lora_registry.get_all_adapters(),
                {"a": original} if duplicate else {},
            )
            unload = UnloadLoRAAdapterReqInput(lora_name="a")
            result = await manager.unload_lora_adapter(unload)
            self.assertTrue(result.success)
            self.assertEqual(
                unload.lora_id, original.lora_id if duplicate else request.lora_id
            )
            manager.update_lora_adapter_communicator.assert_awaited_with(unload)
            self.assertEqual(manager.update_lora_adapter_communicator.await_count, 2)
            self.assertEqual(manager.lora_registry.get_all_adapters(), {})
            self.assertEqual(manager.pending_lora_unloads, {})

        with get_context().override_server_args(enable_lora=True, dp_size=1):
            for from_tensors in (False, True):
                for duplicate in (True, False):
                    with self.subTest(from_tensors=from_tensors, duplicate=duplicate):
                        asyncio.run(scenario(from_tensors, duplicate))


class TestSchedulerLoRAAgreement(CustomTestCase):
    def setUp(self):
        self.enterContext(get_parallel().override(pp_size=1))

    def _scheduler(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tp_worker = Mock()
        scheduler.kv_event_lora_names = Mock()
        scheduler.enable_overlap = False
        return scheduler

    def test_partial_tp_load_is_rolled_back_before_frontend_reply(self):
        """A non-I/O rank's load failure was invisible to registry commitment."""

        async def scenario(from_tensors, failed_rank, unknown_state):
            scheduler = self._scheduler()
            manager = TokenizerControlMixin()
            manager.auto_create_handle_loop = Mock()
            manager.lora_update_lock = asyncio.Lock()
            manager.lora_registry = LoRARegistry([])
            manager.pending_lora_unloads = {}
            manager.lora_ref_cache = {}
            peer = None

            def gather(outputs, value, group):
                outputs[:] = [value, peer]

            def rollback(request):
                nonlocal peer
                peer = _ok()
                return _ok()

            scheduler.tp_worker.unload_lora_adapter.side_effect = rollback

            async def communicate(request):
                nonlocal peer
                if isinstance(request, UnloadLoRAAdapterReqInput):
                    peer = _ok()
                    scheduler.tp_worker.unload_lora_adapter.return_value = _ok()
                    return [scheduler.unload_lora_adapter(request)]
                local = (
                    _err("rank 0 load failed")
                    if failed_rank == 0
                    else _ok({"a": "path"})
                )
                peer = (
                    _err("rank 1 load failed")
                    if failed_rank == 1
                    else _ok({"a": "path"})
                )
                if unknown_state:
                    peer = LoRAUpdateOutput(
                        success=False, error_message="rank 1 load failed"
                    )
                if from_tensors:
                    scheduler.tp_worker.load_lora_adapter_from_tensors.return_value = (
                        local
                    )
                    return [scheduler.load_lora_adapter_from_tensors(request)]
                scheduler.tp_worker.load_lora_adapter.return_value = local
                return [scheduler.load_lora_adapter(request)]

            manager.update_lora_adapter_communicator = communicate
            with patch("torch.distributed.all_gather_object", side_effect=gather):
                if from_tensors:
                    request = LoadLoRAAdapterFromTensorsReqInput(
                        lora_name="a", config_dict={}, serialized_named_tensors=[b""]
                    )
                    result = await manager.load_lora_adapter_from_tensors(request)
                else:
                    request = LoadLoRAAdapterReqInput(lora_name="a", lora_path="path")
                    result = await manager.load_lora_adapter(request)
                self.assertFalse(result.success)
                self.assertIn(f"rank {failed_rank}", result.error_message)
                self.assertEqual(manager.lora_registry.get_all_adapters(), {})
                self.assertEqual(manager.pending_lora_unloads, {})
                self.assertEqual(result.loaded_adapters, {})
                self.assertEqual(
                    scheduler.tp_worker.unload_lora_adapter.call_args.args[0].lora_id,
                    request.lora_id,
                )
                scheduler.kv_event_lora_names.register.assert_not_called()

        with (
            get_context().override_server_args(enable_lora=True, dp_size=1),
            get_parallel().override(
                tp_rank=0,
                tp_size=2,
                attn_tp_size=2,
                moe_tp_size=2,
                pp_size=1,
                attn_dp_enabled=False,
                tp_group=SimpleNamespace(cpu_group=None),
            ),
        ):
            for from_tensors in (False, True):
                for failed_rank, unknown_state in ((0, False), (1, False), (1, True)):
                    with self.subTest(
                        from_tensors=from_tensors,
                        failed_rank=failed_rank,
                        unknown_state=unknown_state,
                    ):
                        asyncio.run(scenario(from_tensors, failed_rank, unknown_state))

    def test_attention_dp_keeps_existing_frontend_failure_merge(self):
        """Every supported attention-DP rank already sends its own reply."""
        scheduler = self._scheduler()
        local = _ok({"a": "path"})
        scheduler.tp_worker.load_lora_adapter.return_value = local
        with (
            get_context().override_server_args(enable_lora=True, dp_size=1),
            get_parallel().override(
                tp_rank=0,
                tp_size=2,
                attn_tp_size=1,
                attn_dp_size=2,
                moe_tp_size=2,
                pp_size=1,
                attn_dp_enabled=True,
            ),
            patch("torch.distributed.all_gather_object") as gather,
        ):
            result = scheduler.load_lora_adapter(
                LoadLoRAAdapterReqInput(lora_name="a", lora_path="path")
            )
        self.assertIs(result, local)
        gather.assert_not_called()
        merged = _merge_lora_update_results([result, _err("DP peer load failed")])
        self.assertFalse(merged.success)
        self.assertEqual(merged.error_message, "DP peer load failed")

    def test_failed_tp_unload_retains_cleanup_identity_until_retry(self):
        """A failed peer unload must not erase the identity needed to retry cleanup."""

        async def scenario():
            ref = LoRARef(lora_name="a", lora_path="path", pinned=False)
            manager = TokenizerControlMixin()
            manager.auto_create_handle_loop = Mock()
            manager.lora_update_lock = asyncio.Lock()
            manager.lora_registry = LoRARegistry([ref])
            manager.pending_lora_unloads = {}
            scheduler = self._scheduler()
            scheduler.tp_worker.unload_lora_adapter.return_value = _ok()
            peer = _err("peer unload failed", {"a": "path"})

            def gather(outputs, value, group):
                outputs[:] = [value, peer]

            async def communicate(request):
                self.assertEqual(request.lora_id, ref.lora_id)
                return [scheduler.unload_lora_adapter(request)]

            manager.update_lora_adapter_communicator = communicate
            with patch("torch.distributed.all_gather_object", side_effect=gather):
                result = await manager.unload_lora_adapter(
                    UnloadLoRAAdapterReqInput(lora_name="a")
                )
                self.assertFalse(result.success)
                self.assertEqual(manager.pending_lora_unloads, {"a": ref.lora_id})
                peer = _ok()
                result = await manager.unload_lora_adapter(
                    UnloadLoRAAdapterReqInput(lora_name="a")
                )
                self.assertTrue(result.success)
                self.assertEqual(manager.pending_lora_unloads, {})

        with (
            get_context().override_server_args(enable_lora=True, dp_size=1),
            get_parallel().override(
                tp_rank=0,
                tp_size=2,
                attn_tp_size=2,
                moe_tp_size=2,
                pp_size=1,
                attn_dp_enabled=False,
                tp_group=SimpleNamespace(cpu_group=None),
            ),
        ):
            asyncio.run(scenario())

    def test_rank_local_exception_still_produces_a_collective_failure(self):
        """A deserialize/checksum exception must not strand peers in the gather."""
        scheduler = self._scheduler()
        scheduler.tp_worker.load_lora_adapter_from_tensors.side_effect = ValueError(
            "bad tensor payload"
        )
        scheduler.tp_worker.unload_lora_adapter.return_value = _ok()
        with (
            get_parallel().override(
                tp_rank=0,
                tp_size=2,
                attn_tp_size=2,
                moe_tp_size=2,
                pp_size=1,
                attn_dp_enabled=False,
                tp_group=SimpleNamespace(cpu_group=None),
            ),
            patch(
                "torch.distributed.all_gather_object",
                side_effect=lambda outputs, value, group: outputs.__setitem__(
                    slice(None), [value, _ok()]
                ),
            ),
        ):
            result = scheduler.load_lora_adapter_from_tensors(
                LoadLoRAAdapterFromTensorsReqInput(
                    lora_name="a",
                    lora_id="new-id",
                    config_dict={},
                    serialized_named_tensors=[b""],
                )
            )
        self.assertFalse(result.success)
        self.assertIn("bad tensor payload", result.error_message)
        self.assertEqual(result.loaded_adapters, {})


class TestLoRAMutationSafety(CustomTestCase):
    def test_failed_rollback_terminates_instead_of_resuming_with_partial_slots(self):
        """A cleanup refusal must not let the scheduler serve inconsistent TP capacity."""
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_overlap = False
        scheduler.kv_event_lora_names = Mock()
        scheduler.tp_worker = Mock()
        scheduler.tp_worker.load_lora_adapter.return_value = _ok({"new": "path"})
        scheduler.tp_worker.unload_lora_adapter.return_value = _err(
            "cannot clear slot", {"new": "path"}
        )
        peers = iter([_err("peer load failed"), _ok()])

        def gather(outputs, local, group):
            outputs[:] = [local, next(peers)]

        with (
            get_parallel().override(
                tp_rank=0,
                tp_size=2,
                pp_size=1,
                attn_dp_enabled=False,
                tp_group=SimpleNamespace(cpu_group=None),
            ),
            patch("torch.distributed.all_gather_object", side_effect=gather),
            self.assertRaisesRegex(RuntimeError, "rollback failed.*cannot clear slot"),
        ):
            scheduler.load_lora_adapter(
                LoadLoRAAdapterReqInput(
                    lora_name="new", lora_id="new", lora_path="path"
                )
            )
        scheduler.kv_event_lora_names.register.assert_not_called()

    def test_partial_pinned_load_restores_batch_capacity_before_reply(self):
        """A failed pinned TP load must not split base/adapter batch admission
        while the caller has not yet sent a cleanup request."""
        config = {"r": 1, "lora_alpha": 1, "target_modules": ["q_proj"]}
        for from_tensors in (False, True):
            with (
                self.subTest(from_tensors=from_tensors),
                tempfile.TemporaryDirectory() as path,
            ):
                Path(path, "adapter_config.json").write_text(json.dumps(config))
                existing = LoRARef(
                    lora_id="existing",
                    lora_name="existing",
                    lora_path="old",
                    pinned=False,
                )
                manager = LoRAManager.__new__(LoRAManager)
                manager.max_loras_per_batch = 2
                manager.num_pinned_loras = 0
                manager.lora_refs = {existing.lora_id: existing}
                manager.loras = {existing.lora_id: object()}
                manager.configs = {}
                manager.lora_modules = []
                manager.attn_dp_enabled = False
                manager.base_hf_config = SimpleNamespace(vocab_size=32)
                resident = {existing.lora_id}
                manager.memory_pool = Mock(max_loras_per_batch=2)
                manager.memory_pool.can_support.return_value = True
                manager.memory_pool.remove_lora.side_effect = lambda uid: (
                    resident.discard(uid)
                )

                def upload(ref, *args):
                    manager.loras[ref.lora_id] = object()
                    resident.add(ref.lora_id)

                manager.load_lora_weights = upload
                manager.load_lora_weights_from_tensors = upload
                scheduler = Scheduler.__new__(Scheduler)
                scheduler.enable_overlap = False
                scheduler.kv_event_lora_names = Mock()
                scheduler.tp_worker = Mock()
                scheduler.tp_worker.load_lora_adapter.side_effect = lambda req: (
                    manager._load_lora_adapter(req.to_ref())
                )
                scheduler.tp_worker.load_lora_adapter_from_tensors.side_effect = (
                    lambda req: manager._load_lora_adapter_from_tensors(
                        req.to_ref(), {}, req.config_dict
                    )
                )
                scheduler.tp_worker.unload_lora_adapter.side_effect = lambda req: (
                    manager._unload_lora_adapter(req.to_ref())
                )
                peer_results = iter(
                    [
                        _err("peer load failed", {"existing": "old"}),
                        _ok({"existing": "old"}),
                    ]
                )

                def gather(outputs, local, group):
                    outputs[:] = [local, next(peer_results)]

                with (
                    get_parallel().override(
                        tp_rank=0,
                        tp_size=2,
                        pp_size=1,
                        attn_dp_enabled=False,
                        tp_group=SimpleNamespace(cpu_group=None),
                    ),
                    patch("torch.distributed.all_gather_object", side_effect=gather),
                ):
                    if from_tensors:
                        request = LoadLoRAAdapterFromTensorsReqInput(
                            lora_name="new",
                            lora_id="new",
                            config_dict=config,
                            serialized_named_tensors=[b""],
                            pinned=True,
                        )
                        result = scheduler.load_lora_adapter_from_tensors(request)
                    else:
                        request = LoadLoRAAdapterReqInput(
                            lora_name="new", lora_id="new", lora_path=path, pinned=True
                        )
                        result = scheduler.load_lora_adapter(request)
                self.assertFalse(result.success)
                self.assertIn("peer load failed", result.error_message)
                self.assertEqual(manager.num_pinned_loras, 0)
                self.assertTrue(manager.validate_lora_batch({None, existing.lora_id}))
                self.assertEqual(set(manager.lora_refs), {existing.lora_id})
                self.assertEqual(resident, {existing.lora_id})
                scheduler.kv_event_lora_names.register.assert_not_called()

    def test_pp_mutations_are_rejected_before_worker_state_changes(self):
        """A stage-zero success must not hide a later stage's missing adapter."""
        for route in ("load", "tensor", "unload"):
            with self.subTest(route=route):
                state = {"existing": "old"}
                scheduler = Scheduler.__new__(Scheduler)
                scheduler.enable_overlap = False
                scheduler.kv_event_lora_names = Mock()
                scheduler.schedule_stream = Mock()
                scheduler.forward_stream = object()
                scheduler.tp_worker = Mock()

                def mutate(request):
                    state.clear()
                    return _ok()

                scheduler.tp_worker.load_lora_adapter.side_effect = mutate
                scheduler.tp_worker.load_lora_adapter_from_tensors.side_effect = mutate
                scheduler.tp_worker.unload_lora_adapter.side_effect = mutate
                with (
                    get_parallel().override(
                        tp_rank=0,
                        tp_size=2,
                        pp_size=2,
                        attn_dp_enabled=False,
                        tp_group=SimpleNamespace(cpu_group=None),
                    ),
                    patch(
                        "torch.distributed.all_gather_object",
                        side_effect=lambda outputs, local, group: outputs.__setitem__(
                            slice(None), [local, local]
                        ),
                    ),
                ):
                    if route == "load":
                        result = scheduler.load_lora_adapter(
                            LoadLoRAAdapterReqInput(
                                lora_name="new", lora_id="new", lora_path="path"
                            )
                        )
                    elif route == "tensor":
                        result = scheduler.load_lora_adapter_from_tensors(
                            LoadLoRAAdapterFromTensorsReqInput(
                                lora_name="new",
                                lora_id="new",
                                config_dict={},
                                serialized_named_tensors=[b""],
                            )
                        )
                    else:
                        result = scheduler.unload_lora_adapter(
                            UnloadLoRAAdapterReqInput(
                                lora_name="existing", lora_id="existing"
                            )
                        )
                self.assertFalse(result.success)
                self.assertIn("pipeline parallelism", result.error_message)
                self.assertEqual(state, {"existing": "old"})

    def test_pp_rejection_preserves_frontend_registry_and_cleanup_state(self):
        """Reject PP before minting cleanup IDs or deregistering an existing adapter."""

        async def scenario(route):
            ref = LoRARef(lora_name="existing", lora_path="old", pinned=False)
            manager = TokenizerControlMixin()
            manager.auto_create_handle_loop = Mock()
            manager.lora_update_lock = asyncio.Lock()
            manager.lora_registry = LoRARegistry([ref])
            manager.pending_lora_unloads = {}
            manager.lora_ref_cache = {"existing": ref}
            manager.update_lora_adapter_communicator = AsyncMock(return_value=[_ok()])
            before = manager.lora_registry.get_all_adapters()
            if route == "load":
                result = await manager.load_lora_adapter(
                    LoadLoRAAdapterReqInput(lora_name="new", lora_path="path")
                )
            elif route == "tensor":
                result = await manager.load_lora_adapter_from_tensors(
                    LoadLoRAAdapterFromTensorsReqInput(
                        lora_name="new", config_dict={}, serialized_named_tensors=[b""]
                    )
                )
            else:
                result = await manager.unload_lora_adapter(
                    UnloadLoRAAdapterReqInput(lora_name="existing")
                )
            self.assertFalse(result.success)
            self.assertIn("pipeline parallelism", result.error_message)
            self.assertEqual(manager.lora_registry.get_all_adapters(), before)
            self.assertEqual(manager.pending_lora_unloads, {})
            manager.update_lora_adapter_communicator.assert_not_awaited()

        with (
            get_context().override_server_args(enable_lora=True, dp_size=1),
            get_parallel().override(
                pp_size=2, tp_size=2, dp_size=1, attn_tp_size=2, moe_tp_size=2
            ),
        ):
            for route in ("load", "tensor", "unload"):
                with self.subTest(route=route):
                    asyncio.run(scenario(route))


if __name__ == "__main__":
    unittest.main()
