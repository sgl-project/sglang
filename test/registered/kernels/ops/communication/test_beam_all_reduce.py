"""Beam all-reduce boundary numerics and shared-plane CUDA graph replay."""

import os
import unittest

import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.ops.communication.all_reduce import (
    AllReduceAlgo,
    Communicator,
    PushPlane,
)
from sglang.kernels.ops.communication.beam_all_reduce import (
    _cluster_size,
    _module,
    beam_all_reduce_norm,
)
from sglang.srt.distributed.comm_cleanup import register_comm_cleanup
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="extra-b", runner_config="8-gpu-h200")

ROWS = (1, 8, 16, 32, 64, 128)


def _precompile(num_gpus):
    for world in num_gpus:
        for cluster in {_cluster_size(world, rows) for rows in ROWS}:
            for topk in (1, 8):
                _module(world, 5120, cluster, topk)


class TestBeamAllReduce(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        override = get_context().override_server_args(
            cuda_graph_config=CudaGraphConfig()
        )
        override.install()
        cls.addClassCleanup(override.restore)
        # Global rank for the TP layout, local rank for the device (2-node TP8).
        rank = int(os.environ["RANK"])
        local = int(os.environ["LOCAL_RANK"])
        world = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local)
        dist.init_process_group(backend="gloo")
        ps._WORLD = coord = ps.init_world_group(list(range(world)), local, "nccl")
        get_parallel().override_permanently(
            world_group=coord,
            tp_group=coord,
            attn_tp_group=coord,
            tp_size=world,
            tp_rank=rank,
            attn_tp_size=world,
            attn_tp_rank=rank,
            moe_tp_size=world,
            moe_tp_rank=rank,
        )
        cls.group = coord.device_group
        cls.comm = CustomAllReduceV2(
            coord.cpu_group,
            torch.device("cuda", local),
            max_push_size=2 * 1024 * 1024,
            max_push_blocks=128,
        )
        if cls.comm.disabled:
            raise RuntimeError(
                "Beam collective test requires an NVLink V2 communicator"
            )
        register_comm_cleanup(cls.comm)
        # Exercise the shared phase counters, regardless of platform tuning.
        cls.comm.override_algo = AllReduceAlgo.ONE_SHOT_PUSH
        # The same workspaces and counters without a multicast address take the
        # per-peer store path; the multicast plane is the control where it exists.
        unicast = Communicator(
            push=PushPlane(
                cls.comm.rank,
                cls.comm.world_size,
                workspaces=cls.comm._push_workspaces,
                counter=cls.comm._push_counter.view(-1, 1).view(torch.uint8),
            ),
            pull=cls.comm.obj.pull,
        )
        cls.planes = [("unicast", unicast)]
        if cls.comm.has_multicast:
            cls.planes.append(("multicast", cls.comm.obj))

    @classmethod
    def tearDownClass(cls):
        cls.planes = []  # Drop the extra pointer holder before the workspaces.
        cls.comm.close()
        dist.destroy_process_group()

    @staticmethod
    def norm(x, w):
        x = x.float()
        return (
            x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * w.float()
        ).bfloat16()

    @staticmethod
    def local_sum(x):
        if x.ndim == 2:
            return x
        summed = torch.zeros_like(x[:, 0], dtype=torch.float32)
        for expert in range(8):
            summed += x[:, expert].float()
        return (summed * 2.5).bfloat16()

    def reference(self, x, residual, post, next_weight):
        x = self.local_sum(x)
        peers = [torch.empty_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(peers, x, group=self.group)
        reduced = torch.zeros_like(x, dtype=torch.float32)
        for peer in peers:
            reduced += peer.float()
        reduced = reduced.bfloat16()
        combined = self.norm(reduced, post) * 0.13 + residual * 0.97
        return (
            (None if next_weight is None else self.norm(combined, next_weight)),
            combined,
            reduced,
        )

    def test_rounding_graph_replay_and_interleaved_collectives(self):
        if dist.get_world_size() == 8:
            # Both TP8 launch shapes (5-CTA clusters and one CTA per row) must run.
            self.assertEqual({_cluster_size(8, rows) for rows in ROWS}, {1, 5})
        owned = self.comm.obj
        try:
            for name, plane in self.planes:
                # The interleaved generic all-reduce pushes through the same plane.
                self.comm.obj = plane
                try:
                    self._check_plane()
                except AssertionError as error:
                    raise AssertionError(f"{name} push plane: {error}") from error
        finally:
            self.comm.obj = owned

    def _check_plane(self):
        for topk in (1, 8):
            for rows in ROWS:
                torch.manual_seed(42 + rows)
                shape = (rows, 5120) if topk == 1 else (rows, 8, 5120)
                x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                x *= dist.get_rank() + 1
                # Consecutive zero pairs exercise the 32-bit Lamport marker.
                x[..., :16] = 0
                residual = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
                post = torch.randn(5120, device="cuda", dtype=torch.bfloat16)
                weight = torch.randn_like(post)
                for next_weight in (None, weight):

                    def fused():
                        return beam_all_reduce_norm(
                            self.comm.obj,
                            x,
                            residual,
                            post,
                            next_weight,
                            1e-6,
                            (0.13, 0.97),
                            routed_scale=2.5,
                        )

                    fused()  # Compile before capture.
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        normed, combined = fused()
                        reduced = self.comm.custom_all_reduce(self.local_sum(x))
                        normed2, combined2 = fused()
                    # Reuse graph storage with different payloads and several epochs.
                    for change in (1.0, -0.5, 2.0):
                        x.mul_(change)
                        for _ in range(5):
                            graph.replay()
                        expected_norm, expected, expected_reduced = self.reference(
                            x, residual, post, next_weight
                        )
                        torch.testing.assert_close(
                            combined, expected, atol=0.016, rtol=0.016
                        )
                        torch.testing.assert_close(combined2, combined, atol=0, rtol=0)
                        torch.testing.assert_close(
                            reduced, expected_reduced, atol=0, rtol=0
                        )
                        if next_weight is not None:
                            torch.testing.assert_close(
                                normed, expected_norm, atol=0.032, rtol=0.016
                            )
                            torch.testing.assert_close(normed2, normed, atol=0, rtol=0)


if __name__ == "__main__":
    if os.environ.get("_IS_TEST_MULTIGPU_SGLANG_JIT_KERNEL"):
        unittest.main()
    else:
        multigpu_pytest_main(
            __name__, __file__, num_gpus=(2, 4, 8), pre_launch_fn=_precompile
        )
