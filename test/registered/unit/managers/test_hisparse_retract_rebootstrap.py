"""A retracted HiSparse PD decode request resumes by rebootstrap, not from a host copy.

`RequestRetirer.retract` lets HiSparse release the request before it takes the
retraction backup, so by then the request's device buffer and host slots are
already back in the allocator, and the HiSparse allocator has no `get_cpu_copy`
to take a backup with. On a DSA-style HiSparse pool the `cpu_tensor` backend
used to reach that `get_cpu_copy`; the `NotImplementedError` escaped the
scheduler loop and killed the decode worker on the first retraction.

The backup (`backup_kv_cache`) and the requeue (`DecodePreallocQueue.add`) must
agree on which pools recompute: a request retracted without a host copy that
still joins `retracted_queue` waits for a restore that has nothing to restore.
"""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components import request_retirer
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    ListKind,
)
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scheduler_test_utils import release_only_retirer
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_CPU_COPY = object()


def _hisparse_allocator():
    # The real class: the pool behind it (DSA or MiniMax) does not decide the
    # path, and its inherited `get_cpu_copy` is the production one that raises.
    allocator = HiSparseTokenToKVPoolAllocator.__new__(HiSparseTokenToKVPoolAllocator)
    allocator._kvcache = SimpleNamespace(device="cpu", cpu_copy_carries_mamba=True)
    return allocator


def _dsv4_hisparse_allocator():
    allocator = DeepSeekV4HiSparseTokenToKVPoolAllocator.__new__(
        DeepSeekV4HiSparseTokenToKVPoolAllocator
    )
    # Buffers are never touched: the pool class alone decides the path.
    allocator._kvcache = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
    allocator._kvcache.device = "cpu"
    return allocator


def _offloading_allocator():
    """A pool that keeps a host copy, as plain PD decode does."""
    return SimpleNamespace(
        cpu_copy_carries_mamba=True,
        get_kvcache=lambda: SimpleNamespace(device="cpu"),
        get_cpu_copy=lambda indices, mamba_indices=None, req_pool_index=None: _CPU_COPY,
    )


def _make_req(multimodal: bool = False) -> Req:
    req = Req(
        rid="retracted",
        origin_input_text="",
        origin_input_ids=array("q", range(8)),
        sampling_params=SamplingParams(max_new_tokens=8),
    )
    req.time_stats.trace_ctx = MagicMock()
    req.output_ids = [1, 2, 3]
    req.kv.req_pool_idx = 0
    req.kv.kv_allocated_len = req.seqlen
    if multimodal:
        req.multimodal_inputs = object()
    return req


def _retract(req: Req, allocator, backend: str = "cpu_tensor") -> bool:
    reset_context()
    publish(
        ServerArgs(
            model_path="dummy",
            disaggregation_mode="decode",
            disaggregation_decode_retraction_backup=backend,
        ),
        role="test",
    )
    tree_cache = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.zeros((1, req.seqlen), dtype=torch.int64)
        ),
        token_to_kv_pool_allocator=allocator,
    )
    # HiSparse releases the request's KV before the backup would run.
    retirer = release_only_retirer(
        tree_cache,
        hisparse_coordinator=SimpleNamespace(release_request=lambda req: None),
    )
    with patch.object(request_retirer, "release_kv_cache"):
        return retirer.retract(req, offload_kv=True)


def _requeue(req: Req, allocator):
    queue = SimpleNamespace(
        scheduler=SimpleNamespace(request_retirer=MagicMock()),
        token_to_kv_pool_allocator=allocator,
        retracted_queue=IndexedReqList(ListKind.PD_RETRACTED),
        pending_reqs=[],
        _check_if_req_exceed_kv_capacity=lambda req: False,
        _create_receiver_and_enqueue=MagicMock(),
        _resolve_prefill_dp_rank=lambda req: None,
    )
    with patch(
        "sglang.srt.disaggregation.decode._is_fake_transfer", return_value=False
    ):
        DecodePreallocQueue.add(queue, req, is_retracted=True)
    return queue


class TestHiSparseRetractRebootstrap(CustomTestCase):
    def setUp(self):
        self.addCleanup(reset_context)

    def test_backup_and_requeue_agree_on_which_pools_recompute(self):
        for name, make_allocator, recomputes in (
            ("hisparse", _hisparse_allocator, True),
            ("dsv4_hisparse", _dsv4_hisparse_allocator, True),
            ("plain", _offloading_allocator, False),
        ):
            with self.subTest(pool=name):
                allocator = make_allocator()
                req = _make_req()

                self.assertTrue(_retract(req, allocator))
                if recomputes:
                    self.assertIsNone(req.kv.retraction_backup)
                else:
                    self.assertIs(req.kv.retraction_backup.cpu_tensors, _CPU_COPY)

                queue = _requeue(req, allocator)
                if recomputes:
                    self.assertEqual(list(queue.retracted_queue), [])
                    (call,) = queue._create_receiver_and_enqueue.call_args_list
                    self.assertTrue(call.kwargs["is_rebootstrap"])
                else:
                    self.assertEqual(list(queue.retracted_queue), [req])
                    queue._create_receiver_and_enqueue.assert_not_called()

    def test_hisparse_request_that_cannot_recompute_is_not_resumable(self):
        """The retraction rules a DeepSeek-V4 recompute follows hold for HiSparse:
        the `none` backend aborts, and so does a request whose prompt the
        prefill cannot rebuild from token ids."""
        for name, backend, multimodal in (
            ("none_backend", "none", False),
            ("multimodal", "cpu_tensor", True),
        ):
            with self.subTest(case=name):
                req = _make_req(multimodal=multimodal)
                self.assertFalse(_retract(req, _hisparse_allocator(), backend))
                self.assertIsNone(req.kv.retraction_backup)


if __name__ == "__main__":
    unittest.main()
