import errno
import gc
import mmap
import pickle
import unittest
import weakref
from array import array
from datetime import timedelta
from multiprocessing import shared_memory
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.distributed
import torch.multiprocessing

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (  # noqa: E402
    BatchTokenizedEmbeddingReqInput,
    MMInputsProcessError,
    TokenizedEmbeddingReqInput,
)
from sglang.srt.managers.mm_utils import ShmPointerMMData  # noqa: E402
from sglang.srt.managers.schedule_batch import (  # noqa: E402
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.managers.scheduler import (  # noqa: E402
    Scheduler,
    _MultimodalInputProcessingError,
)
from sglang.srt.managers.scheduler_components.request_receiver import (  # noqa: E402
    SchedulerRequestReceiver,
)

register_cpu_ci(est_time=40, suite="base-a-test-cpu")


class _CloneFailure:
    def clone(self):
        raise RuntimeError("clone failed")


class _Handle:
    def __init__(self, *, fail_unlink: bool = False):
        self.closed = False
        self.unlinked = False
        self.fail_unlink = fail_unlink

    def close(self):
        self.closed = True

    def unlink(self):
        if self.fail_unlink:
            raise PermissionError("unlink denied")
        self.unlinked = True


def _failed_pointer() -> ShmPointerMMData:
    pointer = object.__new__(ShmPointerMMData)
    pointer._uses_shm_view = False
    pointer.shm_name = "missing-vlm-feature"
    pointer.shape = torch.Size([1])
    pointer.dtype = torch.float32
    pointer.precomputed_hash = None
    pointer._shm_handle = None
    pointer.tensor = None
    pointer._materialization_error = "FileNotFoundError: missing feature"
    return pointer


def _successful_pointer() -> ShmPointerMMData:
    pointer = object.__new__(ShmPointerMMData)
    pointer._uses_shm_view = False
    pointer.shm_name = "unused"
    pointer.shape = torch.Size([1])
    pointer.dtype = torch.float32
    pointer.precomputed_hash = None
    pointer._shm_handle = _Handle()
    pointer.tensor = torch.ones(1)
    pointer._materialization_error = None
    return pointer


def _request(feature, rid: str = "vlm-request") -> TokenizedEmbeddingReqInput:
    return TokenizedEmbeddingReqInput(
        rid=rid,
        input_text="",
        input_ids=array("q", [1]),
        mm_inputs=MultimodalProcessorOutput(
            mm_items=[MultimodalDataItem(modality=Modality.IMAGE, feature=feature)]
        ),
        token_type_ids=None,
        sampling_params=MagicMock(),
    )


def _receiver(tp_size: int = 1) -> SchedulerRequestReceiver:
    group = SimpleNamespace(rank=0, ranks=[0], cpu_group=object())
    return SchedulerRequestReceiver(
        recv_from_tokenizer=None,
        recv_from_rpc=None,
        recv_skipper=None,
        input_blocker=None,
        mm_receiver=None,
        ps=SimpleNamespace(
            pp_rank=0,
            tp_size=tp_size,
            attn_tp_rank=0,
            attn_cp_rank=0,
            attn_tp_size=1,
            attn_cp_size=1,
        ),
        tp_group=group,
        tp_cpu_group=group,
        attn_tp_group=group,
        attn_tp_cpu_group=group,
        attn_cp_group=group,
        attn_cp_cpu_group=group,
        world_group=group,
        server_args=SimpleNamespace(),
        model_config=SimpleNamespace(is_multimodal=True),
        max_recv_per_poll=-1,
        stream_output=lambda *args, **kwargs: None,
        get_last_batch=lambda: None,
    )


def _run_consensus_rank(
    rank: int, world_size: int, init_file: str, state=None, malformed=False
) -> None:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=Path(init_file).as_uri(),
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        if state is None:
            pointer = _failed_pointer() if rank == 1 else _successful_pointer()
        else:
            pointer = object.__new__(ShmPointerMMData)
            if malformed and rank == 1:
                state = {**state, "shape": (999999,)}
            pointer.__setstate__(state)
        req = _request(pointer)
        parallel = SimpleNamespace(enable_dp_attention=False)
        receiver = _receiver(tp_size=world_size)
        object.__setattr__(receiver, "tp_cpu_group", torch.distributed.group.WORLD)
        with (
            patch(
                "sglang.srt.managers.mm_utils._get_is_default_transport",
                return_value=False,
            ),
            patch(
                "sglang.srt.managers.mm_utils.get_serving",
                return_value=SimpleNamespace(skip_tokenizer_init=False),
            ),
            patch(
                "sglang.srt.managers.scheduler_components.request_receiver.get_parallel",
                return_value=parallel,
            ),
        ):
            receiver._finalize_shm_features([req])
        if state is None or malformed:
            if not isinstance(req.mm_inputs, MMInputsProcessError):
                raise AssertionError(
                    f"rank {rank} did not receive the VLM request error"
                )
        else:
            feature = req.mm_inputs.mm_items[0].feature
            feature.add_(rank + 1)
            torch.distributed.barrier()
            if not torch.equal(feature, torch.arange(8) + rank + 1):
                raise AssertionError(f"rank {rank} observed another rank's writes")
    finally:
        torch.distributed.destroy_process_group()


class TestShmPointerFailureCleanup(unittest.TestCase):
    def test_clone_failure_still_unlinks_and_closes(self):
        pointer = object.__new__(ShmPointerMMData)
        pointer._uses_shm_view = False
        handle = _Handle()
        pointer.shm_name = "unused"
        pointer._shm_handle = handle
        pointer.tensor = _CloneFailure()
        pointer._materialization_error = None

        with self.assertRaisesRegex(RuntimeError, "clone failed"):
            pointer.materialize()

        self.assertTrue(handle.unlinked)
        self.assertTrue(handle.closed)
        self.assertIsNone(pointer._shm_handle)
        self.assertIsNone(pointer.tensor)

    def test_shm_open_failure_is_deferred_until_materialization(self):
        pointer = object.__new__(ShmPointerMMData)
        pointer._uses_shm_view = False
        state = {
            "shm_name": "missing",
            "shape": torch.Size([1]),
            "dtype": torch.float32,
            "precomputed_hash": None,
        }

        with patch(
            "sglang.srt.managers.mm_utils.shared_memory.SharedMemory",
            side_effect=FileNotFoundError("missing"),
        ):
            pointer.__setstate__(state)
            with self.assertRaisesRegex(RuntimeError, "FileNotFoundError"):
                pointer.materialize()

    def test_cleanup_error_does_not_escape_the_request_boundary(self):
        pointer = object.__new__(ShmPointerMMData)
        pointer._uses_shm_view = False
        handle = _Handle(fail_unlink=True)
        pointer.shm_name = "unused"
        pointer._shm_handle = handle
        pointer.tensor = torch.ones(1)
        pointer._materialization_error = None

        with self.assertLogs("sglang.utils", level="WARNING"):
            result = pointer.materialize()

        self.assertTrue(torch.equal(result, torch.ones(1)))
        self.assertTrue(handle.closed)


class TestShmReceiverViews(CustomTestCase):
    def setUp(self):
        override = envs.SGLANG_ENABLE_MM_SHM_ZERO_COPY.override(True)
        override.__enter__()
        self.addCleanup(override.__exit__, None, None, None)

    def _sender(self, tensor):
        pointer = ShmPointerMMData(tensor)
        self.addCleanup(pointer.close_and_unlink)
        return pointer

    def test_storage_aliases_keep_mapping_after_unlink_and_wrapper_cleanup(self):
        """Slices, detach and NumPy aliases may outlive the receiving request."""
        sender = self._sender(torch.arange(24, dtype=torch.float32).reshape(4, 6))
        mappings = []
        original_mmap = mmap.mmap

        def track_mapping(*args, **kwargs):
            mapping = original_mmap(*args, **kwargs)
            if kwargs.get("access") == mmap.ACCESS_COPY:
                mappings.append(weakref.ref(mapping))
            return mapping

        with patch("sglang.srt.managers.mm_utils.mmap.mmap", track_mapping):
            receiver = pickle.loads(pickle.dumps(sender))
        address = receiver.tensor.data_ptr()
        tensor = receiver.materialize()
        self.assertEqual(tensor.data_ptr(), address)
        self.assertEqual(len(mappings), 1)
        detached = tensor[1:3].detach()
        array_view = detached.numpy()
        receiver.close_and_unlink()
        receiver.close_and_unlink()
        with self.assertRaisesRegex(RuntimeError, "released"):
            receiver.materialize()
        del receiver, tensor, detached
        gc.collect()
        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=sender.shm_name)
        self.assertIsNotNone(mappings[0]())
        self.assertEqual(
            array_view.tolist(), [[6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17]]
        )
        del array_view
        gc.collect()
        self.assertIsNone(mappings[0]())

    def test_materialization_uses_the_mode_selected_when_attaching(self):
        """Changing an env override cannot switch an attached buffer's owner."""
        for enabled in (False, True):
            sender = self._sender(torch.arange(8))
            with envs.SGLANG_ENABLE_MM_SHM_ZERO_COPY.override(enabled):
                receiver = pickle.loads(pickle.dumps(sender))
            address = receiver.tensor.data_ptr()
            with envs.SGLANG_ENABLE_MM_SHM_ZERO_COPY.override(not enabled):
                result = receiver.materialize()
            self.assertEqual(result.data_ptr() == address, enabled)
            self.assertTrue(torch.equal(result, torch.arange(8)))

    def test_writes_are_isolated_between_receivers(self):
        """Eliminating the clone must not make rank-local mutations shared."""
        for dtype in (torch.float32, torch.bfloat16, torch.int64):
            with self.subTest(dtype=dtype):
                expected = torch.arange(12, dtype=dtype).reshape(3, 4).t()
                sender = self._sender(expected)
                payload = pickle.dumps(sender)
                first, second = pickle.loads(payload), pickle.loads(payload)
                # All ranks have opened their mappings before any materialization.
                a, b = first.materialize(), second.materialize()
                a.add_(100)
                self.assertTrue(torch.equal(b, expected))
                self.assertTrue(torch.equal(a, expected + 100))

    def test_router_forwarding_does_not_unlink_the_segment(self):
        from sglang.srt.managers.io_struct import msgpack_decode, msgpack_encode
        from sglang.srt.sampling.sampling_params import SamplingParams

        for encode, decode in (
            (pickle.dumps, pickle.loads),
            (msgpack_encode, msgpack_decode),
        ):
            sender = self._sender(torch.arange(8))
            request = _request(sender)
            request.sampling_params = SamplingParams()
            forwarded = decode(encode(request))
            payload = encode(forwarded)
            del forwarded
            gc.collect()
            received = decode(payload)
            tensor = received.mm_inputs.mm_items[0].feature.materialize()
            self.assertTrue(torch.equal(tensor, torch.arange(8)))

    def test_mapping_failure_is_deferred_and_segment_is_cleaned(self):
        sender = self._sender(torch.ones(8))
        original_mmap = mmap.mmap

        def fail_private_mapping(*args, **kwargs):
            if kwargs.get("access") == mmap.ACCESS_COPY:
                raise OSError("mapping unavailable")
            return original_mmap(*args, **kwargs)

        with patch("sglang.srt.managers.mm_utils.mmap.mmap", fail_private_mapping):
            receiver = pickle.loads(pickle.dumps(sender))
        with self.assertRaisesRegex(RuntimeError, "mapping unavailable"):
            receiver.materialize()
        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=sender.shm_name)

    def test_discard_does_not_invalidate_an_existing_reader(self):
        from sglang.srt.managers.mm_utils import discard_shm_features

        sender = self._sender(torch.arange(8))
        receiver = pickle.loads(pickle.dumps(sender))
        reader = receiver.tensor[2:6]
        request = _request(receiver)
        discard_shm_features(request)
        self.assertTrue(torch.equal(reader, torch.arange(2, 6)))
        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=sender.shm_name)

    def test_real_rank_views_and_partial_deserialization_failure(self):
        for malformed in (False, True):
            with self.subTest(malformed=malformed), TemporaryDirectory() as directory:
                sender = self._sender(torch.arange(8))
                torch.multiprocessing.spawn(
                    _run_consensus_rank,
                    args=(
                        2,
                        str(Path(directory) / "gloo-init"),
                        sender.__getstate__(),
                        malformed,
                    ),
                    nprocs=2,
                    join=True,
                )
                with self.assertRaises(FileNotFoundError):
                    shared_memory.SharedMemory(name=sender.shm_name)

    def test_allocation_exhaustion_falls_back_without_leaking_a_name(self):
        from sglang.srt.managers.mm_utils import _wrap_shm_or_inline

        names = []
        original_shm = shared_memory.SharedMemory

        def track_allocation(*args, **kwargs):
            handle = original_shm(*args, **kwargs)
            if kwargs.get("create"):
                names.append(handle.name)
            return handle

        feature = torch.ones(8)
        with (
            patch(
                "sglang.srt.managers.mm_utils.shared_memory.SharedMemory",
                track_allocation,
            ),
            patch(
                "sglang.srt.managers.mm_utils.os.posix_fallocate",
                side_effect=OSError(errno.ENOSPC, "full"),
            ),
        ):
            result = _wrap_shm_or_inline(feature)
        self.assertIs(result, feature)
        self.assertEqual(len(names), 1)
        with self.assertRaises(FileNotFoundError):
            original_shm(name=names[0])

    def test_abort_preserves_session_features_and_active_aliases(self):
        from sglang.srt.managers.schedule_batch import MultimodalInputs, Req

        for session_owned in (False, True):
            sender = self._sender(torch.arange(8))
            receiver = pickle.loads(pickle.dumps(sender))
            item = MultimodalDataItem(
                modality=Modality.IMAGE, feature=receiver.materialize()
            )
            inputs = MultimodalInputs(mm_items=[item])
            alias = item.feature[2:6]
            request = object.__new__(Req)
            request.rid = "abort-shm-view"
            request.session = (
                SimpleNamespace(mm_inputs=inputs) if session_owned else None
            )
            request.multimodal_inputs = inputs
            request.grammar = None
            request.return_logprob = False
            request.logprob_start_len = 0
            with patch(
                "sglang.srt.managers.schedule_batch.get_parallel",
                return_value=SimpleNamespace(tp_rank=1),
            ):
                request.set_finish_with_abort("cancelled request")
            self.assertIsNone(request.multimodal_inputs)
            if session_owned:
                self.assertTrue(torch.equal(item.feature, torch.arange(8)))
            else:
                self.assertIsNone(item.feature)
            self.assertTrue(torch.equal(alias, torch.arange(2, 6)))


class TestShmRequestFailureConsensus(unittest.TestCase):
    def test_real_gloo_group_propagates_one_rank_failure(self):
        with TemporaryDirectory() as directory:
            init_file = str(Path(directory) / "gloo-init")
            torch.multiprocessing.spawn(
                _run_consensus_rank,
                args=(2, init_file),
                nprocs=2,
                join=True,
            )

    def test_local_materialization_failure_becomes_request_error(self):
        req = _request(_failed_pointer())
        parallel = SimpleNamespace(enable_dp_attention=False)

        with (
            patch(
                "sglang.srt.managers.mm_utils._get_is_default_transport",
                return_value=False,
            ),
            patch(
                "sglang.srt.managers.mm_utils.get_serving",
                return_value=SimpleNamespace(skip_tokenizer_init=False),
            ),
            patch(
                "sglang.srt.managers.scheduler_components.request_receiver.get_parallel",
                return_value=parallel,
            ),
        ):
            _receiver()._finalize_shm_features([req])

        self.assertIsInstance(req.mm_inputs, MMInputsProcessError)
        with self.assertRaises(_MultimodalInputProcessingError):
            Scheduler._get_multimodal_inputs(object.__new__(Scheduler), req.mm_inputs)

    def test_peer_failure_rejects_the_local_request(self):
        req = _request(torch.zeros(1))
        parallel = SimpleNamespace(enable_dp_attention=False)

        def inject_peer_failure(mask, **kwargs):
            mask.fill_(1)

        with (
            patch(
                "sglang.srt.managers.scheduler_components.request_receiver.get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.managers.scheduler_components.request_receiver.has_shm_features",
                return_value=True,
            ),
            patch(
                "sglang.srt.managers.scheduler_components.request_receiver.unwrap_shm_features"
            ),
            patch("sglang.srt.managers.scheduler_components.request_receiver.barrier"),
            patch(
                "sglang.srt.managers.scheduler_components.request_receiver.all_reduce",
                side_effect=inject_peer_failure,
            ) as all_reduce,
        ):
            _receiver(tp_size=2)._finalize_shm_features([req])

        all_reduce.assert_called_once()
        self.assertIsInstance(req.mm_inputs, MMInputsProcessError)

    def test_batched_requests_only_reject_the_failed_item(self):
        failed_req = _request(torch.zeros(1), rid="failed")
        healthy_req = _request(torch.zeros(1), rid="healthy")
        batch = BatchTokenizedEmbeddingReqInput(batch=[failed_req, healthy_req])
        parallel = SimpleNamespace(enable_dp_attention=False)

        def materialize(req):
            if req.rid == "failed":
                raise RuntimeError("bad shared feature")

        with (
            patch(
                "sglang.srt.managers.scheduler_components.request_receiver.get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.managers.scheduler_components.request_receiver.has_shm_features",
                return_value=True,
            ),
            patch(
                "sglang.srt.managers.scheduler_components.request_receiver.unwrap_shm_features",
                side_effect=materialize,
            ),
        ):
            _receiver()._finalize_shm_features([batch])

        self.assertIsInstance(failed_req.mm_inputs, MMInputsProcessError)
        self.assertIsInstance(healthy_req.mm_inputs, MultimodalProcessorOutput)


if __name__ == "__main__":
    unittest.main()
