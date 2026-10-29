"""Unit tests for srt/managers/output_store.py and the detokenizer side of it."""

import concurrent.futures
import subprocess
import sys
import threading
import time
import unittest
from types import SimpleNamespace

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
from sglang.srt.managers.tokenizer_manager import (  # noqa: E402
    ReqState,
    TokenizerManager,
)

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


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

    def get_ranges(self, reads, *, into):
        for read in reads:
            stored = self.objects[read.ref.key][read.field]
            data = stored[read.src_offset : read.src_offset + read.nbytes]
            into[read.dst_offset : read.dst_offset + read.nbytes] = data

    def delete_many(self, keys, *, parts):
        if self.delete_error is not None:
            raise self.delete_error
        for key in keys:
            del self.objects[key]

    def close(self):
        pass


class _BlockingPutBackend(_MemoryBackend):
    """Holds each put until released, and records put, delete, and close order."""

    def __init__(self):
        super().__init__()
        self.put_started = threading.Event()
        self.release_put = threading.Event()
        self.events = []

    def put_many(self, objects):
        self.put_started.set()
        self.release_put.wait(timeout=10)
        self.events.append("put")
        return super().put_many(objects)

    def delete_many(self, keys, *, parts):
        self.events.append("delete")
        super().delete_many(keys, parts=parts)

    def close(self):
        self.events.append("close")


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
        collector = TokenOutputCollector(
            routed_experts=routed, sampling_support_logprobs=True
        )
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


class TestTokenizerSupportLogprobs(CustomTestCase):
    """Support mode also returns each sampled token's logprob inline, picked from its
    support; with the store only the support-sized rows are stored."""

    def _add(self, *, via_store):
        state = ReqState(
            [], False, None, SimpleNamespace(sampling_logprobs_mode="support"), None
        )
        if via_store:
            state.token_output_collector = TokenOutputCollector(
                sampling_support_logprobs=True
            )
        meta_info = {}
        for chunk, output_ids in (
            (_chunk([2], [5, 6], [-0.5, -1.0]), [6]),
            (_chunk([1, 2], [7, 9, 8], [-0.25, -2.0, -3.0]), [7, 8]),
        ):
            TokenizerManager._add_sampling_mask_output(
                meta_info=meta_info, state=state, chunk=chunk, output_ids=output_ids
            )
        return meta_info, state.token_output_collector

    def test_inline_response(self):
        meta_info, _ = self._add(via_store=False)
        self.assertEqual(
            meta_info["output_token_sampling_selected_logprobs"], [-1.0, -0.25, -3.0]
        )
        self.assertEqual(
            meta_info["output_token_sampling_logprobs"],
            [[-0.5, -1.0], [-0.25], [-2.0, -3.0]],
        )

    def test_store_response(self):
        meta_info, collector = self._add(via_store=True)
        self.assertEqual(
            meta_info, {"output_token_sampling_selected_logprobs": [-1.0, -0.25, -3.0]}
        )
        np.testing.assert_array_equal(
            collector.to_stash().output_token_sampling_logprobs,
            [-0.5, -1.0, -0.25, -2.0, -3.0],
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

    def test_shutdown_still_deletes_a_running_put_handed_to_cleanup(self):
        """A request cancelled while its put runs hands the put to cleanup; when the
        writer shuts down before that put finishes, the object must still be deleted
        before the backend disconnects."""
        backend = _BlockingPutBackend()
        writer = OutputStoreWriter(OutputStore(backend=backend))
        put = writer.submit_put(
            TokenOutputCollector(routed_experts=torch.zeros(1, 1, 1, dtype=torch.int32))
        )
        self.assertTrue(backend.put_started.wait(timeout=10))
        writer.cleanup_after(put)

        stopping = threading.Thread(target=writer.shutdown)
        stopping.start()
        deadline = time.monotonic() + 10
        while not writer._executor._shutdown and time.monotonic() < deadline:
            time.sleep(0.001)
        backend.release_put.set()
        stopping.join(timeout=10)

        self.assertEqual(backend.events, ["put", "delete", "close"])
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

    def test_servers_without_a_backend_do_not_need_the_package(self):
        """Importing the server must not require radix_transfer_plugins."""
        code = (
            "import sys\n"
            "sys.modules['radix_transfer_plugins'] = None\n"
            "from sglang.test.test_utils import maybe_stub_sgl_kernel\n"
            "maybe_stub_sgl_kernel()\n"
            "import sglang.srt.managers.tokenizer_manager\n"
            "import sglang.srt.entrypoints.openai.serving_chat\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

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
