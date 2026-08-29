import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

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


class TestShmRequestFailureConsensus(unittest.TestCase):
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
