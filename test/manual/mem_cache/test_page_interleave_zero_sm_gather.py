# Copyright 2023-2026 SGLang Team
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
"""Copy-engine (zero-SM) page-interleave KV gathers over a real process group.

Covers symmetric-window ownership and release, zero NCCL kernel launches,
the cross-node fallback, and comparisons against ordinary NCCL on 2 and 4
ranks, including CUDA graph replay. Needs NCCL >= 2.31.2 (CI pins an older
NCCL) and at least 2 GPUs. Run with:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 \
        test/manual/mem_cache/test_page_interleave_zero_sm_gather.py
"""

import gc
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import torch
import torch.multiprocessing as mp

from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    allocate_symmetric_buffer,
)
from sglang.srt.distributed.device_communicators.pynccl_wrapper import (
    NCCL_CTA_POLICY_ZERO,
    NCCLLibrary,
)
from sglang.srt.environ import envs
from sglang.srt.mem_cache.page_interleave_pool import (
    _NCCL_SYMMETRIC_SCRATCH_MIN_VERSION,
    _symmetric_scratch_layout,
)
from sglang.srt.utils.common import temp_set_env
from sglang.test.test_utils import CustomTestCase

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / "registered" / "unit" / "mem_cache"),
)
from test_page_interleave_gather_multi_gpu import (  # noqa: E402
    _GATHER_LAYER_NUM,
    _GATHER_WORLD,
    _run_mla,
)


def _assert_bytes(got, expected, label):
    got = got.contiguous().view(torch.uint8)
    expected = expected.contiguous().view(torch.uint8)
    assert got.shape == expected.shape, label
    assert torch.equal(got, expected), f"{label}: cache bytes differ"


def _gather_gpu_activity(pool, req_to_token, prefix_len, seq_len):
    """Names of the GPU kernels and copies one sharded batch's gathers issue."""
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        pool.begin_shard_extend(
            req_to_token, torch.tensor([0]), [prefix_len], [seq_len]
        )
        for layer_id in range(_GATHER_LAYER_NUM):
            pool.get_key_buffer(layer_id)
        torch.cuda.synchronize()
    return [
        event.name
        for event in prof.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]


def _run_zero_sm_mla(rank, world, port):
    pool, req_to_token, prefix_len, seq_len = _run_mla(rank, world, port)
    assert pool._use_symmetric_scratch and pool._scratch_window_registered

    # NCCL picks copy engines per collective, so a windowed gather can still
    # land on ring or symmetric SM kernels; only the launched work tells.
    activity = _gather_gpu_activity(pool, req_to_token, prefix_len, seq_len)
    nccl_kernels = sorted({name for name in activity if "nccl" in name.lower()})
    assert not nccl_kernels, f"gather ran on NCCL SM kernels: {nccl_kernels}"
    assert any("PtoP" in name for name in activity), "no peer copy-engine copy"

    scratch_bytes = sum(
        tensor.numel() * tensor.element_size()
        for slot in pool._slots
        for tensor in slot.tensors.values()
    )
    view = pool._slots[1].tensors["kv"]
    # Deregistration can allocate a CUDA stream whose driver storage persists.
    # Account for that separately from the backing allocation being released.
    comm = pool.kv_gather_comm
    deregister = comm.deregister_comm_window
    taken_by_deregister = []

    def measured_deregister(window):
        free_before = torch.cuda.mem_get_info()[0]
        deregister(window)
        taken = free_before - torch.cuda.mem_get_info()[0]
        taken_by_deregister.append(max(taken, 0))

    comm.deregister_comm_window = measured_deregister
    torch.cuda.synchronize()
    torch.distributed.barrier()
    free_with_pool = torch.cuda.mem_get_info()[0]
    del pool
    gc.collect()
    torch.distributed.barrier()
    free_with_view = torch.cuda.mem_get_info()[0]
    assert not taken_by_deregister, "window deregistered under a live view"
    # A peer's window maps this buffer until that peer deregisters too.
    del view
    gc.collect()
    torch.distributed.barrier()
    free_released = torch.cuda.mem_get_info()[0]
    assert len(taken_by_deregister) == 1, "window not deregistered exactly once"
    assert free_with_view - free_with_pool < scratch_bytes, "freed under a live view"
    freed = free_released - free_with_view + taken_by_deregister[0]
    assert freed >= scratch_bytes, "window buffer leaked"


def _run_cross_node_mla(rank, world, port):
    with unittest.mock.patch(
        "sglang.srt.mem_cache.page_interleave_pool.in_the_same_node_as",
        lambda pg, source_rank: [r == source_rank for r in range(world)],
    ):
        pool, req_to_token, prefix_len, seq_len = _run_mla(rank, world, port)
    assert not pool._use_symmetric_scratch and not pool._scratch_window_registered
    activity = _gather_gpu_activity(pool, req_to_token, prefix_len, seq_len)
    assert any("nccl" in name.lower() for name in activity)


def _run_zero_sm_allgather_comparison(rank, world, init_method):
    """Compare the pool's in-place window contract with ordinary NCCL."""
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        backend="gloo", rank=rank, world_size=world, init_method=init_method
    )
    device = torch.device(f"cuda:{rank}")
    group = torch.distributed.group.WORLD
    zero_sm = PyNcclCommunicator(group, device, cta_policy=NCCL_CTA_POLICY_ZERO)
    ordinary = PyNcclCommunicator(group, device)
    assert zero_sm.available and ordinary.available
    max_send_bytes = 1 << 20

    for dtype in (torch.uint8, torch.bfloat16, torch.float32):
        # Both alternating slots start inside the window, rather than at its
        # base. The input is each rank's slice of the registered receive view,
        # exactly as in _prefetch_layer; its offset also changes with payload.
        offsets, nbytes = _symmetric_scratch_layout(
            {
                "padding": ((4096,), torch.uint8),
                "output": ((world * max_send_bytes // dtype.itemsize,), dtype),
            },
            num_slots=2,
        )
        buffer, registered = allocate_symmetric_buffer(zero_sm, nbytes)
        assert registered, "comparison requires a real symmetric window"
        slots = [
            buffer[slot["output"] : slot["output"] + world * max_send_bytes].view(dtype)
            for slot in offsets
        ]
        assert all(slot["output"] > 0 for slot in offsets)

        def compare(slot_index, send_bytes, generation, graph=None):
            count = send_bytes // dtype.itemsize
            output = slots[slot_index][: world * count]
            send = output[rank * count : (rank + 1) * count]
            output.zero_()
            values = (
                (torch.arange(count, device=device, dtype=torch.int32) * 17)
                + rank * 29
                + generation * 31
            ) % 251
            send.copy_(values)
            reference_input = send.clone()
            reference = torch.empty_like(output)
            if graph is None:
                with zero_sm.change_state(enable=True):
                    zero_sm.all_gather(output, send)
            else:
                graph.replay()
            # Use the same loaded NCCL library for the ordinary baseline,
            # with torch allocations and the default communicator policy.
            with ordinary.change_state(enable=True):
                ordinary.all_gather(reference, reference_input)
            _assert_bytes(
                output,
                reference,
                f"rank {rank}, {dtype}, {send_bytes} B, generation {generation}",
            )

        for send_bytes in (256, 16 << 10, max_send_bytes):
            for generation in range(6):
                compare(generation % 2, send_bytes, generation)

        # Profile only the CE call: the ordinary baseline deliberately issues
        # an NCCL kernel and must not contaminate the zero-SM assertion.
        output = slots[1]
        count = max_send_bytes // dtype.itemsize
        send = output[rank * count : (rank + 1) * count]
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            with zero_sm.change_state(enable=True):
                zero_sm.all_gather(output, send)
            torch.cuda.synchronize()
        activity = [
            event.name
            for event in prof.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
        ]
        assert not any("nccl" in name.lower() for name in activity), activity
        assert any("PtoP" in name for name in activity), activity

        # Replay the same registered addresses with fresh contributions. A
        # comparison only at capture time would miss stale peer data on replay.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with zero_sm.change_state(enable=True):
                zero_sm.all_gather(output, send)
        for generation in (7, 8, 9):
            compare(1, max_send_bytes, generation, graph=graph)
        del graph, send, output
        slots.clear()
        del buffer
        gc.collect()
        torch.distributed.barrier()

    torch.cuda.synchronize()
    torch.distributed.destroy_process_group()


@unittest.skipIf(
    torch.cuda.device_count() < 2, "page-interleave gather needs 2 CUDA devices"
)
class TestPageInterleaveZeroSmGatherMultiGpu(CustomTestCase):
    def _require_patched_nccl(self):
        # Gate on the library used by PyNccl, which may differ from torch's.
        if NCCLLibrary().ncclGetRawVersion() < _NCCL_SYMMETRIC_SCRATCH_MIN_VERSION:
            self.skipTest("the symmetric gather scratch needs NCCL >= 2.31.2")

    def _compare_allgather(self, world):
        if torch.cuda.device_count() < world:
            self.skipTest(f"copy-engine allgather comparison needs {world} GPUs")
        self._require_patched_nccl()
        with (
            temp_set_env(NCCL_CUMEM_ENABLE="1", NCCL_CTA_POLICY=None),
            tempfile.TemporaryDirectory(prefix="sglang-zero-sm-gather-") as tmp,
        ):
            mp.spawn(
                _run_zero_sm_allgather_comparison,
                args=(world, f"file://{tmp}/rendezvous"),
                nprocs=world,
                join=True,
            )

    def test_copy_engine_matches_ordinary_allgather_two_gpu(self):
        self._compare_allgather(2)

    def test_copy_engine_matches_ordinary_allgather_four_gpu(self):
        self._compare_allgather(4)

    def test_copy_engine_window_gather_and_release(self):
        """The zero-SM pool registers its scratch as a symmetric window, still
        assembles the canonical prefix, and frees the window buffer only once
        the pool and every scratch view are gone."""
        self._require_patched_nccl()
        # Clear the global override before spawning: only the dedicated
        # gather communicator requests ZERO. cuMem enables window registration.
        with (
            temp_set_env(NCCL_CUMEM_ENABLE="1", NCCL_CTA_POLICY=None),
            envs.SGLANG_ENABLE_KV_SHARD_ZERO_SM_GATHER.override(True),
        ):
            mp.spawn(
                _run_zero_sm_mla,
                args=(_GATHER_WORLD, 29811),
                nprocs=_GATHER_WORLD,
                join=True,
            )

    def test_cross_node_group_keeps_sm_gather(self):
        """A shard group spanning nodes keeps the torch scratch and the SM
        allgather, even with cuMem on."""
        with (
            temp_set_env(NCCL_CUMEM_ENABLE="1", NCCL_CTA_POLICY=None),
            envs.SGLANG_ENABLE_KV_SHARD_ZERO_SM_GATHER.override(True),
        ):
            mp.spawn(
                _run_cross_node_mla,
                args=(_GATHER_WORLD, 29811),
                nprocs=_GATHER_WORLD,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
