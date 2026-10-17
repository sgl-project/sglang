"""Unit tests for srt/managers/output_store.py and the detokenizer side of it."""

import concurrent.futures
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import numpy as np
import pybase64
import torch

from radix_transfer_plugins import OutputStore, StorageBackend, StoreRef
from sglang.srt.sampling.sampling_mask import SamplingMaskChunk
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.detokenizer_manager import DetokenizerManager  # noqa: E402
from sglang.srt.managers.io_struct import BatchTokenIDOutput  # noqa: E402
from sglang.srt.managers.output_store import (  # noqa: E402
    OutputStoreWriter,
    TokenOutputCollector,
    maybe_create_output_store,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _chunk(lengths, token_ids, logprobs):
    return SamplingMaskChunk(
        lengths=np.array(lengths, np.int32),
        token_ids=np.array(token_ids, np.int32),
        logprobs=np.array(logprobs, np.float32),
    )


class _MemoryBackend(StorageBackend):
    name = "memory"
    config_type = msgspec.Struct

    def __init__(self):
        self.objects = {}
        self.delete_error = None

    def put_many(self, objects):
        keys = []
        for parts in objects:
            key = f"mem/{len(self.objects)}"
            self.objects[key] = {name: bytes(data) for name, data in parts.items()}
            keys.append(key)
        return keys

    def get_many(self, keys, *, parts):
        return [
            {name: memoryview(self.objects[key][name]) for name, _ in named}
            for key, named in zip(keys, parts)
        ]

    def delete_many(self, keys, *, parts):
        if self.delete_error is not None:
            raise self.delete_error
        for key in keys:
            del self.objects[key]

    def close(self):
        pass


class TestTokenOutputCollector(CustomTestCase):
    def test_empty_until_something_arrives(self):
        collector = TokenOutputCollector()
        self.assertTrue(collector.is_empty())
        self.assertEqual(collector.inline_meta_info(), {})
        collector.add_sampling_mask(_chunk([], [], []))
        self.assertFalse(collector.is_empty())
        self.assertEqual(
            collector.inline_meta_info(), {"output_token_sampling_mask_length": 0}
        )

    def test_stash_concatenates_chunks_without_copying_tensors(self):
        routed = torch.arange(24, dtype=torch.int32).reshape(3, 2, 4)
        collector = TokenOutputCollector(routed_experts=routed)
        collector.add_sampling_mask(_chunk([2], [5, 6], [-0.5, -1.0]))
        collector.add_sampling_mask(_chunk([1, 1], [7, 8], [-2.0, -3.0]))

        stash = collector.to_stash()
        self.assertTrue(np.shares_memory(stash.routed_experts, routed.numpy()))
        np.testing.assert_array_equal(
            stash.output_token_sampling_mask_lengths, [2, 1, 1]
        )
        np.testing.assert_array_equal(
            stash.output_token_sampling_mask_token_ids, [5, 6, 7, 8]
        )
        np.testing.assert_array_equal(
            stash.output_token_sampling_logprobs, [-0.5, -1.0, -2.0, -3.0]
        )
        self.assertIsNone(stash.indexer_topk)
        self.assertEqual(
            collector.inline_meta_info(), {"output_token_sampling_mask_length": 3}
        )


class TestOutputStoreWriter(CustomTestCase):
    def _writer(self):
        backend = _MemoryBackend()
        return OutputStoreWriter(OutputStore(backend=backend)), backend

    def test_put_resolves_to_a_ref_of_the_stored_stash(self):
        writer, backend = self._writer()
        routed = torch.arange(8, dtype=torch.int32).reshape(2, 2, 2)
        ref = writer.submit_put(TokenOutputCollector(routed_experts=routed)).result()
        self.assertEqual(ref.backend, "memory")
        self.assertEqual(
            backend.objects[ref.key], {"routed_experts": routed.numpy().tobytes()}
        )

    def test_undelivered_objects_are_deleted(self):
        writer, backend = self._writer()
        written = writer.submit_put(
            TokenOutputCollector(routed_experts=torch.zeros(1, 1, 1, dtype=torch.int32))
        )
        written.result()
        failed = concurrent.futures.Future()
        failed.set_exception(RuntimeError("put failed"))
        writer.cleanup_after(written)
        writer.cleanup_after(failed)

        other = writer.submit_put(
            TokenOutputCollector(routed_experts=torch.ones(1, 1, 1, dtype=torch.int32))
        ).result()
        writer.cleanup_refs([other.to_builtins()])
        writer._executor.shutdown(wait=True)
        self.assertEqual(backend.objects, {})

    def test_failed_delete_logs_the_ref(self):
        writer, backend = self._writer()
        backend.delete_error = RuntimeError("master down")
        ref = StoreRef(backend="memory", key="mem/9", fields=())
        with self.assertLogs("sglang.srt.managers.output_store", "ERROR") as logs:
            writer.cleanup_refs([ref.to_builtins()])
            writer._executor.shutdown(wait=True)
        self.assertIn('"key": "mem/9"', "\n".join(logs.output))


class TestMaybeCreateOutputStore(CustomTestCase):
    def test_none_creates_nothing_and_pd_is_rejected(self):
        self.assertIsNone(
            maybe_create_output_store(
                backend="none", extra_config=None, disaggregation_mode="prefill"
            )
        )
        with self.assertRaisesRegex(ValueError, "PD disaggregation"):
            maybe_create_output_store(
                backend="mooncake", extra_config="{}", disaggregation_mode="prefill"
            )

    def test_backend_comes_from_the_package_registry(self):
        backend = _MemoryBackend()
        with patch(
            "sglang.srt.managers.output_store.create_backend", return_value=backend
        ) as create:
            writer = maybe_create_output_store(
                backend="mooncake", extra_config='{"a": 1}', disaggregation_mode="null"
            )
        create.assert_called_once_with("mooncake", '{"a": 1}')
        self.assertIsInstance(writer, OutputStoreWriter)

    def test_unknown_backend_names_fail_at_startup(self):
        with self.assertRaisesRegex(ValueError, "available"):
            maybe_create_output_store(
                backend="mooncak", extra_config=None, disaggregation_mode="null"
            )


class TestDetokenizerReplayOutputs(CustomTestCase):
    def _handle(self, *, output_store_enabled):
        manager = DetokenizerManager.__new__(DetokenizerManager)
        manager.output_store_enabled = output_store_enabled
        recv_obj = SimpleNamespace(
            **{f.name: None for f in msgspec.structs.fields(BatchTokenIDOutput)}
        )
        recv_obj.rids = []
        recv_obj.routed_experts = [torch.tensor([[1, 2]], dtype=torch.int32)]
        return manager.handle_batch_token_id_out(recv_obj), recv_obj.routed_experts

    def test_tensors_skip_base64_only_when_store_enabled(self):
        out, tensors = self._handle(output_store_enabled=True)
        self.assertIsNone(out.routed_experts)
        self.assertIs(out.routed_experts_raw, tensors)

        out, tensors = self._handle(output_store_enabled=False)
        self.assertEqual(
            out.routed_experts,
            [pybase64.b64encode(tensors[0].numpy().tobytes()).decode("utf-8")],
        )
        self.assertIsNone(out.routed_experts_raw)


if __name__ == "__main__":
    unittest.main()
