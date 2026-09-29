import gc
import os
import pickle
import sys
import unittest
from array import array
from multiprocessing import shared_memory
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed
import torch.multiprocessing

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (  # noqa: E402
    BatchTokenizedEmbeddingReqInput,
    MMInputsProcessError,
    TokenizedEmbeddingReqInput,
)
from sglang.srt.managers.mm_utils import (  # noqa: E402
    ShmPointerMMData,
    wrap_shm_features,
)
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
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402

register_cpu_ci(est_time=33, suite="base-a-test-cpu")


class _CloneFailure:
    def clone(self):
        raise RuntimeError("clone failed")


class _Handle:
    def __init__(self):
        self.closed = False
        self.unlinked = False

    def close(self):
        self.closed = True

    def unlink(self):
        self.unlinked = True


def _failed_pointer() -> ShmPointerMMData:
    pointer = object.__new__(ShmPointerMMData)
    pointer.shm_name = "missing-vlm-feature"
    pointer.shape = torch.Size([1])
    pointer.dtype = torch.float32
    pointer.precomputed_hash = None
    pointer._shm_handle = None
    pointer.tensor = None
    pointer._materialization_error = "FileNotFoundError: missing feature"
    return pointer


def _successful_pointer() -> ShmPointerMMData:
    return pickle.loads(pickle.dumps(ShmPointerMMData(torch.ones(1))))


def _request(feature, rid: str = "vlm-request") -> TokenizedEmbeddingReqInput:
    return TokenizedEmbeddingReqInput(
        rid=rid,
        input_text="",
        input_ids=array("q", [1]),
        mm_inputs=MultimodalProcessorOutput(
            mm_items=[MultimodalDataItem(modality=Modality.IMAGE, feature=feature)]
        ),
        token_type_ids=None,
        sampling_params=SamplingParams(max_new_tokens=1),
    )


def _receiver() -> SchedulerRequestReceiver:
    group = SimpleNamespace(rank=0, ranks=[0], cpu_group=object())
    return SchedulerRequestReceiver(
        recv_from_tokenizer=None,
        recv_from_rpc=None,
        recv_skipper=None,
        input_blocker=None,
        mm_receiver=None,
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


def _run_consensus_rank(rank: int, world_size: int, init_file: str) -> None:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=Path(init_file).as_uri(),
        rank=rank,
        world_size=world_size,
    )
    try:
        req = _request(_failed_pointer() if rank == 1 else _successful_pointer())
        parallel = SimpleNamespace(enable_dp_attention=False, tp_size=world_size)
        receiver = _receiver()
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
        if not isinstance(req.mm_inputs, MMInputsProcessError):
            raise AssertionError(f"rank {rank} did not receive the VLM request error")
    finally:
        torch.distributed.destroy_process_group()


def _run_image_receiver(rank, init_file, pipe):
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        backend="gloo", init_method=Path(init_file).as_uri(), rank=rank, world_size=2
    )
    try:
        receiver = _receiver()
        object.__setattr__(receiver, "tp_cpu_group", torch.distributed.group.WORLD)
        torch.distributed.barrier()
        torch.distributed.all_reduce(torch.zeros(1))
        initial_fds = len(os.listdir("/proc/self/fd"))
        held = None
        pipe.send("ready")
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
                return_value=SimpleNamespace(enable_dp_attention=False, tp_size=2),
            ),
        ):
            for base in [30, 90]:
                req = pickle.loads(pipe.recv_bytes())
                receiver._finalize_shm_features([req])
                features = [item.feature for item in req.mm_inputs.mm_items]
                assert len(features) == 7
                assert all(torch.all(t == base + i) for i, t in enumerate(features))
                assert [item.hash for item in req.mm_inputs.mm_items] == list(
                    range(100, 107)
                )
                if held is None:
                    held = features[1][::2, ::2, :]
                assert torch.all(held == 31)
                # Copy-on-write must preserve the old clone's rank isolation.
                if rank == 0:
                    features[0].zero_()
                torch.distributed.barrier()
                assert torch.all(features[0] == (0 if rank == 0 else base))
                del features, req
                pipe.send("exact")
        del held
        gc.collect()
        assert len(os.listdir("/proc/self/fd")) <= initial_fds
    finally:
        pipe.close()
        torch.distributed.destroy_process_group()


class TestShmStorageOwnership(unittest.TestCase):
    def test_private_mappings_preserve_dtype_pixels_and_views(self):
        for dtype in [torch.uint8, torch.float32, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                source = torch.arange(3 * 13 * 19).to(dtype).reshape(3, 13, 19)
                wire = pickle.dumps(ShmPointerMMData(source))
                first, second = pickle.loads(wire), pickle.loads(wire)
                left, right = first.materialize(), second.materialize()
                self.assertEqual(left.dtype, dtype)
                self.assertTrue(torch.equal(left, source))
                view = right[:, ::2, ::2]
                left.zero_()
                self.assertTrue(torch.equal(right, source))
                del left, right, first, second
                gc.collect()
                self.assertTrue(torch.equal(view, source[:, ::2, ::2]))

    @unittest.skipUnless(sys.platform == "linux", "requires Linux fd accounting")
    def test_seven_input_images_across_two_ranks_and_successive_requests(self):
        torch.set_num_threads(1)
        ctx = torch.multiprocessing.get_context("spawn")
        pipes = [ctx.Pipe() for _ in range(2)]
        processes = []
        allocated = []
        with TemporaryDirectory() as directory:
            try:
                for rank, (parent, child) in enumerate(pipes):
                    proc = ctx.Process(
                        target=_run_image_receiver,
                        args=(rank, str(Path(directory) / "gloo-init"), child),
                    )
                    proc.start()
                    child.close()
                    processes.append(proc)
                for parent, _ in pipes:
                    self.assertTrue(parent.poll(120))
                    self.assertEqual(parent.recv(), "ready")
                for base in [30, 90]:
                    pixels = [
                        torch.full((641 + i, 643, 3), base + i, dtype=torch.uint8)
                        for i in range(7)
                    ]
                    req = _request(None)
                    req.mm_inputs.mm_items = [
                        MultimodalDataItem(
                            modality=Modality.IMAGE, feature=pixel, hash=100 + i
                        )
                        for i, pixel in enumerate(pixels)
                    ]
                    with (
                        patch(
                            "sglang.srt.managers.mm_utils._get_is_default_transport",
                            return_value=False,
                        ),
                        patch(
                            "sglang.srt.managers.mm_utils.get_serving",
                            return_value=SimpleNamespace(skip_tokenizer_init=False),
                        ),
                    ):
                        wrap_shm_features(req)
                    names = [item.feature.shm_name for item in req.mm_inputs.mm_items]
                    allocated.extend(item.feature for item in req.mm_inputs.mm_items)
                    wire = pickle.dumps(req)
                    self.assertLess(len(wire), 8192)
                    for pixel in pixels:
                        pixel.zero_()
                    for parent, _ in pipes:
                        parent.send_bytes(wire)
                    for parent, _ in pipes:
                        self.assertTrue(parent.poll(120))
                        self.assertEqual(parent.recv(), "exact")
                    for name in names:
                        with self.assertRaises(FileNotFoundError):
                            shared_memory.SharedMemory(name=name)
                for proc in processes:
                    proc.join(30)
                    self.assertEqual(proc.exitcode, 0)
            finally:
                for proc in processes:
                    if proc.is_alive():
                        proc.terminate()
                        proc.join(30)
                for parent, _ in pipes:
                    parent.close()
                for pointer in allocated:
                    pointer.close_and_unlink()


class TestShmPointerFailureCleanup(unittest.TestCase):
    def test_nonlinux_clone_failure_still_unlinks_and_closes(self):
        pointer = object.__new__(ShmPointerMMData)
        handle = _Handle()
        pointer.shm_name = "unused"
        pointer._shm_handle = handle
        pointer.tensor = _CloneFailure()
        pointer._materialization_error = None

        with (
            patch("sglang.srt.managers.mm_utils.sys.platform", "darwin"),
            self.assertRaisesRegex(RuntimeError, "clone failed"),
        ):
            pointer.materialize()

        self.assertTrue(handle.unlinked)
        self.assertTrue(handle.closed)
        self.assertIsNone(pointer._shm_handle)
        self.assertIsNone(pointer.tensor)

    def test_shm_open_failure_is_deferred_until_materialization(self):
        pointer = object.__new__(ShmPointerMMData)
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
        pointer = _successful_pointer()
        name, handle = pointer.shm_name, pointer._shm_handle
        try:
            with (
                patch.object(
                    handle, "unlink", side_effect=PermissionError("unlink denied")
                ),
                self.assertLogs("sglang.utils", level="WARNING"),
            ):
                result = pointer.materialize()
            self.assertTrue(torch.equal(result, torch.ones(1)))
            self.assertEqual(handle._fd, -1)
        finally:
            segment = shared_memory.SharedMemory(name=name)
            segment.unlink()
            segment.close()

    @unittest.skipUnless(sys.platform == "linux", "requires Linux private mappings")
    def test_mapping_failure_still_releases_the_segment(self):
        pointer = _successful_pointer()
        name, handle = pointer.shm_name, pointer._shm_handle
        with (
            patch(
                "sglang.srt.managers.mm_utils.mmap.mmap",
                side_effect=OSError("map failed"),
            ),
            self.assertRaisesRegex(OSError, "map failed"),
        ):
            pointer.materialize()
        self.assertEqual(handle._fd, -1)
        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=name)


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
        parallel = SimpleNamespace(enable_dp_attention=False, tp_size=1)

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
        parallel = SimpleNamespace(enable_dp_attention=False, tp_size=2)

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
            _receiver()._finalize_shm_features([req])

        all_reduce.assert_called_once()
        self.assertIsInstance(req.mm_inputs, MMInputsProcessError)

    def test_batched_requests_only_reject_the_failed_item(self):
        failed_req = _request(torch.zeros(1), rid="failed")
        healthy_req = _request(torch.zeros(1), rid="healthy")
        batch = BatchTokenizedEmbeddingReqInput(batch=[failed_req, healthy_req])
        parallel = SimpleNamespace(enable_dp_attention=False, tp_size=1)

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
