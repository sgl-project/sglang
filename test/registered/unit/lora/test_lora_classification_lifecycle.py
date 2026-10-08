"""Exercise classifier publication and leases with the real LoRA control path."""

import asyncio
import json
import tempfile
import threading
import unittest
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from sglang.srt.lora.classification_export import write_classification_manifest
from sglang.srt.lora.classification_head import ClassificationLease
from sglang.srt.lora.lora_registry import LoRARef, LoRARegistry
from sglang.srt.managers.communicator import FanOutCommunicator
from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    LoadLoRAAdapterReqInput,
    LoRAUpdateOutput,
    UnloadLoRAAdapterReqInput,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_scope

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_HIDDEN = 4


class _ControlManager(TokenizerManager):
    def __init__(self):
        # The real load/unload methods only need this tokenizer-side state.
        # Model workers and IPC are represented by the backend communicator.
        self.model_config = SimpleNamespace(
            hidden_size=_HIDDEN,
            hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
        )
        self.is_generation = True
        self.lora_registry = LoRARegistry()
        self.lora_update_lock = asyncio.Lock()
        self.pending_lora_unloads = {}
        self.lora_ref_cache = {}
        self.classification_heads = {}
        self.classification_snapshots = {}
        self.classification_tokenizer_lock = threading.Lock()
        self.classification_tokenizer = None
        self.tokenizer = None

    def auto_create_handle_loop(self):
        # No socket worker is needed when the communicator is supplied directly.
        pass


class TestClassificationLifecycle(CustomTestCase, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        super().setUp()
        enter_scope(
            self,
            get_context().override_server_args(
                enable_lora=True,
                dp_size=1,
                nnodes=1,
                pp_size=1,
                tokenizer_worker_num=1,
                max_loaded_loras=None,
                load_format="safetensors",
                return_hidden_states_mode="last",
                speculative_algorithm=None,
                disaggregation_mode="null",
            ),
        )
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manager = _ControlManager()
        self.backend_adapters = {}
        self.backend = AsyncMock(side_effect=self._backend_update)
        self.manager.update_lora_adapter_communicator = self.backend
        self.addCleanup(self._cleanup_snapshots)

    def _cleanup_snapshots(self):
        for snapshot in self.manager.classification_snapshots.values():
            snapshot.cleanup()

    async def _backend_update(self, obj):
        if isinstance(obj, LoadLoRAAdapterReqInput):
            self.backend_adapters[obj.lora_name] = obj.lora_path
        else:
            self.backend_adapters.pop(obj.lora_name, None)
        return [
            LoRAUpdateOutput(success=True, loaded_adapters=dict(self.backend_adapters))
            for _ in range(2)
        ]

    def _bundle(self, name="classifier", labels=("negative", "positive"), offset=0):
        path = self.root / name
        path.mkdir(exist_ok=True)
        (path / "adapter_config.json").write_text(
            json.dumps(
                {
                    "peft_type": "LORA",
                    "task_type": "SEQ_CLS",
                    "r": 2,
                    "lora_alpha": 4,
                    "target_modules": ["q_proj"],
                    "modules_to_save": ["score"],
                }
            )
        )
        prefix = "base_model.model.model.layers.0.self_attn.q_proj"
        save_file(
            {
                f"{prefix}.lora_A.weight": torch.ones(2, _HIDDEN),
                f"{prefix}.lora_B.weight": torch.ones(_HIDDEN, 2),
                "base_model.model.score.weight": torch.arange(
                    len(labels) * _HIDDEN, dtype=torch.float32
                ).reshape(len(labels), _HIDDEN)
                + offset,
            },
            str(path / "adapter_model.safetensors"),
        )
        write_classification_manifest(
            path,
            id2label=dict(enumerate(labels)),
            hidden_size=_HIDDEN,
            max_length=3,
            add_special_tokens=False,
        )
        return path

    async def _load(self, path, name="classifier"):
        request = LoadLoRAAdapterReqInput(lora_name=name, lora_path=str(path))
        result = await self.manager.load_lora_adapter(request)
        self.assertTrue(result.success, result.error_message)
        return self.manager.lora_registry.get_all_adapters()[name]

    async def _unload(self, name="classifier"):
        return await self.manager.unload_lora_adapter(
            UnloadLoRAAdapterReqInput(lora_name=name)
        )

    async def _lease(self, *, text=None, input_ids=None):
        obj = GenerateReqInput(text=text, input_ids=input_ids, lora_path="classifier")
        obj.normalize_batch_and_arguments()
        obj.lora_id = await self.manager.lora_registry.acquire("classifier")
        lease = ClassificationLease(self.manager)
        try:
            await lease.acquire(obj)
        finally:
            # The scheduler's own reference ends before CPU classification.
            await self.manager.lora_registry.release(obj.lora_id)
        self.addAsyncCleanup(lease.close)
        return lease, obj

    async def _cancel_task(self, task):
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def test_two_heads_publish_distinct_ids_and_class_dimensions(self):
        first = await self._load(self._bundle("binary"), "binary")
        second = await self._load(
            self._bundle("ternary", labels=("one", "two", "three")), "ternary"
        )
        self.assertNotEqual(first.lora_id, second.lora_id)
        self.assertNotEqual(first.lora_path, second.lora_path)
        for ref, labels in ((first, 2), (second, 3)):
            head = self.manager.classification_heads[ref.lora_id]
            self.assertEqual(tuple(head.weight.shape), (labels, _HIDDEN))
            output = head.classify([{"meta_info": {"hidden_states": [1, 0, 0, 0]}}])
            self.assertEqual(output[0]["num_classes"], labels)
            self.assertEqual(len(output[0]["probs"]), labels)
            self.assertEqual(self.backend_adapters[ref.lora_name], ref.lora_path)
            self.assertTrue(Path(ref.lora_path).is_dir())
        self.assertFalse(self.manager.pending_lora_unloads)

    async def test_invalid_manifest_is_never_published_or_sent_to_backend(self):
        path = self._bundle()
        manifest = path / "classification_config.json"
        config = json.loads(manifest.read_text())
        config["id2label"].pop("1")
        manifest.write_text(json.dumps(config))
        result = await self.manager.load_lora_adapter(
            LoadLoRAAdapterReqInput(lora_name="classifier", lora_path=str(path))
        )
        self.assertFalse(result.success)
        self.backend.assert_not_awaited()
        self.assertFalse(self.manager.lora_registry.get_all_adapters())
        self.assertFalse(self.manager.classification_heads)
        self.assertFalse(self.manager.classification_snapshots)

    def test_static_classifiers_fail_closed_and_generation_still_registers(self):
        path = self._bundle()
        ref = LoRARef(lora_name="static", lora_path=str(path))
        with get_context().override_server_args(lora_paths=[ref]):
            with self.assertRaisesRegex(ValueError, "/load_lora_adapter"):
                self.manager.init_lora()
            (path / "classification_config.json").unlink()
            with self.assertRaisesRegex(ValueError, "SEQ_CLS"):
                self.manager.init_lora()
            adapter_config = path / "adapter_config.json"
            config = json.loads(adapter_config.read_text())
            config["task_type"] = "CAUSAL_LM"
            adapter_config.write_text(json.dumps(config))
            for marker in ("classification_head.pt", "label_mapping.json"):
                (path / marker).write_text("legacy")
                with (
                    self.subTest(marker=marker),
                    self.assertRaisesRegex(ValueError, "offline conversion"),
                ):
                    self.manager.init_lora()
                (path / marker).unlink()
            self.manager.init_lora()
            self.assertEqual(
                self.manager.lora_registry.get_all_adapters(), {"static": ref}
            )
            self.assertEqual(self.manager.lora_ref_cache, {"static": ref})
            self.assertFalse(self.manager.classification_heads)
            self.assertFalse(self.manager.classification_snapshots)

    async def test_unsupported_runtime_never_publishes_a_classifier(self):
        path = self._bundle()
        for arguments in (
            {"tokenizer_worker_num": 2},
            {"nnodes": 2},
            {"pp_size": 2},
            {"load_format": "dummy"},
            {"return_hidden_states_mode": "none"},
            {"speculative_algorithm": "EAGLE"},
            {"disaggregation_mode": "prefill"},
        ):
            with (
                self.subTest(arguments=arguments),
                get_context().override_server_args(**arguments),
            ):
                result = await self.manager.load_lora_adapter(
                    LoadLoRAAdapterReqInput(lora_name="classifier", lora_path=str(path))
                )
                self.assertFalse(result.success)
                self.backend.assert_not_awaited()
                self.assertFalse(self.manager.classification_snapshots)
        self.manager.model_config.hf_config.architectures = ["UnverifiedDecoder"]
        result = await self.manager.load_lora_adapter(
            LoadLoRAAdapterReqInput(lora_name="classifier", lora_path=str(path))
        )
        self.assertFalse(result.success)
        self.assertIn("decoder", result.error_message)
        self.backend.assert_not_awaited()
        self.assertFalse(self.manager.classification_snapshots)

    async def test_partial_backend_failure_stays_unavailable_until_cleanup_retry(self):
        path = self._bundle()
        self.backend.side_effect = None
        self.backend.return_value = [
            LoRAUpdateOutput(success=True, loaded_adapters={"classifier": str(path)}),
            LoRAUpdateOutput(success=False, error_message="rank failed"),
        ]
        request = LoadLoRAAdapterReqInput(lora_name="classifier", lora_path=str(path))
        result = await self.manager.load_lora_adapter(request)
        self.assertFalse(result.success)
        self.assertEqual(
            self.manager.pending_lora_unloads["classifier"], request.lora_id
        )
        snapshot = Path(self.manager.classification_snapshots[request.lora_id].name)
        self.assertTrue(snapshot.exists())
        self.assertFalse(self.manager.classification_heads)
        with self.assertRaises(ValueError):
            await self.manager.lora_registry.acquire("classifier")
        self.backend.return_value = [
            LoRAUpdateOutput(success=False, error_message="busy")
        ]
        self.assertFalse((await self._unload()).success)
        self.assertTrue(snapshot.exists())
        self.backend.return_value = [LoRAUpdateOutput(success=True)]
        self.assertTrue((await self._unload()).success)
        self.assertFalse(snapshot.exists())
        self.assertFalse(self.manager.pending_lora_unloads)
        self.assertFalse(self.manager.classification_snapshots)

    async def test_cancelled_load_finishes_publication_before_returning(self):
        path = self._bundle()
        reached, resume = asyncio.Event(), asyncio.Event()

        async def backend(obj):
            reached.set()
            await resume.wait()
            return await self._backend_update(obj)

        self.backend.side_effect = backend
        task = asyncio.create_task(
            self.manager.load_lora_adapter(
                LoadLoRAAdapterReqInput(lora_name="classifier", lora_path=str(path))
            )
        )
        self.addAsyncCleanup(self._cancel_task, task)
        self.addCleanup(resume.set)
        await asyncio.wait_for(reached.wait(), timeout=2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        self.assertFalse(self.manager.classification_heads)
        resume.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        ref = self.manager.lora_registry.get_all_adapters()["classifier"]
        self.assertIn(ref.lora_id, self.manager.classification_heads)
        self.assertEqual(self.backend_adapters["classifier"], ref.lora_path)
        self.assertTrue(Path(ref.lora_path).exists())
        self.assertFalse(self.manager.pending_lora_unloads)

    async def test_cancelled_unload_drains_replies_before_next_load(self):
        ref = await self._load(self._bundle())
        snapshot = Path(ref.lora_path)
        following_path = self._bundle("following")
        sent = asyncio.Queue()
        communicator = FanOutCommunicator(sent.put_nowait, fan_out=2)
        self.manager.update_lora_adapter_communicator = communicator
        tasks = []

        async def finish_pending():
            for task in tasks:
                if not task.done():
                    task.cancel()

            async def drain():
                # Keep failed assertions from stranding a shielded transaction.
                while any(not task.done() for task in tasks):
                    for _ in range(2):
                        communicator.handle_recv(
                            LoRAUpdateOutput(success=True, loaded_adapters={})
                        )
                    await asyncio.sleep(0.01)
                await asyncio.gather(*tasks, return_exceptions=True)

            await asyncio.wait_for(drain(), timeout=5)

        self.addAsyncCleanup(finish_pending)

        unload = asyncio.create_task(self._unload())
        tasks.append(unload)
        dispatched = await asyncio.wait_for(sent.get(), timeout=2)
        self.assertIsInstance(dispatched, UnloadLoRAAdapterReqInput)
        unload.cancel()
        await asyncio.sleep(0)
        unload.cancel()

        following = asyncio.create_task(
            self.manager.load_lora_adapter(
                LoadLoRAAdapterReqInput(
                    lora_name="following", lora_path=str(following_path)
                )
            )
        )
        tasks.append(following)
        await asyncio.sleep(0)
        self.assertTrue(self.manager.lora_update_lock.locked())
        self.assertTrue(sent.empty(), "Next load dispatched before unload completed")
        self.assertTrue(snapshot.exists())
        self.assertIn(ref.lora_id, self.manager.classification_heads)

        # The cancelled HTTP caller must still consume both backend responses.
        communicator.handle_recv(LoRAUpdateOutput(success=True, loaded_adapters={}))
        await asyncio.sleep(0)
        self.assertFalse(unload.done())
        self.assertTrue(sent.empty())
        communicator.handle_recv(LoRAUpdateOutput(success=True, loaded_adapters={}))
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(unload, timeout=2)
        self.assertFalse(snapshot.exists())
        self.assertNotIn(ref.lora_id, self.manager.classification_heads)
        self.assertNotIn("classifier", self.manager.pending_lora_unloads)

        dispatched = await asyncio.wait_for(sent.get(), timeout=2)
        self.assertIsInstance(dispatched, LoadLoRAAdapterReqInput)
        self.assertEqual(dispatched.lora_name, "following")
        self.assertFalse(following.done(), "Old unload replies completed the new load")
        for _ in range(2):
            communicator.handle_recv(
                LoRAUpdateOutput(
                    success=True, loaded_adapters={"following": dispatched.lora_path}
                )
            )
        result = await asyncio.wait_for(following, timeout=2)
        self.assertTrue(result.success, result.error_message)
        current = self.manager.lora_registry.get_all_adapters()["following"]
        self.assertIn(current.lora_id, self.manager.classification_heads)
        self.assertNotEqual(current.lora_id, ref.lora_id)

    async def test_failed_unload_preserves_snapshot_for_retry(self):
        path = self._bundle()
        ref = await self._load(path)
        snapshot = Path(ref.lora_path)
        self.backend.side_effect = None
        self.backend.return_value = [
            LoRAUpdateOutput(success=False, error_message="busy")
        ]
        self.assertFalse((await self._unload()).success)
        self.assertTrue(snapshot.exists())
        self.assertIn(ref.lora_id, self.manager.classification_heads)
        self.assertEqual(self.manager.pending_lora_unloads["classifier"], ref.lora_id)
        self.assertFalse(self.manager.lora_registry.get_all_adapters())
        blocked = await self.manager.load_lora_adapter(
            LoadLoRAAdapterReqInput(lora_name="classifier", lora_path=str(path))
        )
        self.assertFalse(blocked.success)
        self.backend.return_value = [LoRAUpdateOutput(success=True)]
        self.assertTrue((await self._unload()).success)
        self.assertFalse(snapshot.exists())
        self.assertNotIn(ref.lora_id, self.manager.classification_heads)
        self.assertFalse(self.manager.pending_lora_unloads)

    async def test_unload_waits_for_cpu_job_despite_repeated_cancellation(self):
        ref = await self._load(self._bundle())
        lease, _ = await self._lease(input_ids=[1, 2])
        started, finish = asyncio.Event(), threading.Event()
        loop = asyncio.get_running_loop()

        def job():
            loop.call_soon_threadsafe(started.set)
            if not finish.wait(timeout=5):
                raise TimeoutError("test CPU job was not released")
            return lease.head.classify([{"meta_info": {"hidden_states": [1, 0, 0, 0]}}])

        async def classify():
            try:
                return await lease.run_cpu(job)
            finally:
                await lease.close()

        task = asyncio.create_task(classify())
        self.addAsyncCleanup(self._cancel_task, task)
        self.addCleanup(finish.set)
        await asyncio.wait_for(started.wait(), timeout=2)
        unload = asyncio.create_task(self._unload())
        self.addAsyncCleanup(self._cancel_task, unload)
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        self.assertFalse(unload.done())
        self.assertEqual(self.manager.lora_registry._counters[ref.lora_id].value(), 1)
        self.assertTrue(Path(ref.lora_path).exists())
        finish.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue((await asyncio.wait_for(unload, timeout=2)).success)
        self.assertFalse(Path(ref.lora_path).exists())
        self.assertNotIn(ref.lora_id, self.manager.classification_heads)

    async def test_source_changes_do_not_change_snapshot_and_reload_gets_new_id(self):
        path = self._bundle()
        old = await self._load(path)
        old_head = self.manager.classification_heads[old.lora_id]
        old_bytes = (Path(old.lora_path) / "adapter_model.safetensors").read_bytes()
        (path / "classification_config.json").unlink()
        self._bundle(labels=("small", "medium", "large"), offset=100)
        self.assertEqual(old_head.labels, ("negative", "positive"))
        self.assertEqual(
            (Path(old.lora_path) / "adapter_model.safetensors").read_bytes(), old_bytes
        )
        self.assertTrue((await self._unload()).success)
        new = await self._load(path)
        self.assertNotEqual(new.lora_id, old.lora_id)
        self.assertNotEqual(new.lora_path, old.lora_path)
        self.assertFalse(Path(old.lora_path).exists())
        self.assertEqual(
            self.manager.classification_heads[new.lora_id].labels,
            ("small", "medium", "large"),
        )
        self.assertEqual(self.manager.lora_ref_cache["classifier"].lora_path, str(path))

    async def test_tokenizer_copy_is_lazy_and_reused_only_for_classification(self):
        tokenizer = Tokenizer(WordLevel({"red": 0, "green": 1, "blue": 2, "[UNK]": 3}))
        tokenizer.pre_tokenizer = Whitespace()
        self.manager.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, unk_token="[UNK]"
        )
        await self._load(self._bundle())
        self.assertIsNone(self.manager.classification_tokenizer)
        lease, request = await self._lease(input_ids=[0, 1, 2, 0, 1])
        self.assertEqual(request.input_ids, [0, 1, 2])
        self.assertIsNone(self.manager.classification_tokenizer)
        await lease.close()
        text = "red green blue red green"
        lease, request = await self._lease(text=text)
        self.assertEqual(request.input_ids, [0, 1, 2])
        clone = self.manager.classification_tokenizer
        self.assertIsNotNone(clone)
        self.assertIsNot(clone, self.manager.tokenizer)
        self.assertEqual(self.manager.tokenizer(text)["input_ids"], [0, 1, 2, 0, 1])
        await lease.close()
        lease, _ = await self._lease(text=text)
        self.assertIs(self.manager.classification_tokenizer, clone)
        await lease.close()


if __name__ == "__main__":
    unittest.main()
