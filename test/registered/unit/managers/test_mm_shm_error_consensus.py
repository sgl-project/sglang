import unittest
from array import array
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


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


def _run_consensus_rank(rank: int, world_size: int, init_file: str) -> None:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=Path(init_file).as_uri(),
        rank=rank,
        world_size=world_size,
    )
    try:
        req = _request(_failed_pointer() if rank == 1 else _successful_pointer())
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
        if not isinstance(req.mm_inputs, MMInputsProcessError):
            raise AssertionError(f"rank {rank} did not receive the VLM request error")
    finally:
        torch.distributed.destroy_process_group()


class TestShmPointerFailureCleanup(unittest.TestCase):
    def test_clone_failure_still_unlinks_and_closes(self):
        pointer = object.__new__(ShmPointerMMData)
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
        handle = _Handle(fail_unlink=True)
        pointer.shm_name = "unused"
        pointer._shm_handle = handle
        pointer.tensor = torch.ones(1)
        pointer._materialization_error = None

        with self.assertLogs("sglang.utils", level="WARNING"):
            result = pointer.materialize()

        self.assertTrue(torch.equal(result, torch.ones(1)))
        self.assertTrue(handle.closed)


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
