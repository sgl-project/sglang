"""IPC initialization must survive a lock left by an interrupted JIT build.

Unlike the parity test, this test needs isolated cold caches and a bounded
subprocess lifetime so a blocked compiler cannot consume the whole CI job.
"""

import os
import signal
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.file_baton import FileBaton

from sglang.multimodal_gen.runtime.distributed.device_communicators.ipc_a2a import (
    IPC_A2A,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_group,
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_input_all_to_all,
    _usp_output_all_to_all,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="diffusion-2-gpu-h100")


def _worker():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1, sp_size=2, ulysses_degree=2
    )
    IPC_A2A.init(get_sp_group().ulysses_group)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        value = torch.full((1, 32, 4, 64), rank, dtype=torch.bfloat16, device="cuda")
        result = _usp_input_all_to_all(value, head_dim=2)
        expected = torch.cat(
            [torch.zeros_like(result[:, :32]), torch.ones_like(result[:, :32])],
            dim=1,
        )
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        restored = _usp_output_all_to_all(result, head_dim=2)
        torch.testing.assert_close(restored, value, rtol=0, atol=0)
    stream.synchronize()
    # The replacement binding must preserve current-stream and capture semantics.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = _usp_input_all_to_all(value, head_dim=2)
        restored = _usp_output_all_to_all(captured, head_dim=2)
    with torch.cuda.stream(stream):
        for _ in range(3):
            graph.replay()
            torch.testing.assert_close(captured, expected, rtol=0, atol=0)
            torch.testing.assert_close(restored, value, rtol=0, atol=0)
    stream.synchronize()
    assert IPC_A2A.inited and IPC_A2A.calls > 0
    IPC_A2A.check_timeout()
    print(f"IPC_RECOVERY PASS rank={rank}", flush=True)
    dist.barrier()
    IPC_A2A.reset()
    dist.barrier()
    dist.destroy_process_group()


class TestIpcA2AJitRecovery(CustomTestCase):
    def test_initialization_with_orphaned_legacy_lock(self):
        # Every diffusion partition invokes registered files; run this once.
        if os.environ.get("DIFFUSION_PARTITION_ID", "0") != "0":
            self.skipTest("cold-cache recovery runs on diffusion partition 0")
        if not current_platform.is_cuda() or torch.cuda.device_count() < 2:
            self.skipTest("requires two CUDA GPUs")
        if not torch.cuda.can_device_access_peer(0, 1):
            self.skipTest("requires CUDA peer access")
        with tempfile.TemporaryDirectory(prefix="ipc-jit-recovery-") as cache_dir:
            for rank in range(2):
                build_dir = Path(cache_dir) / f"ipc_a2a_sync_r{rank}"
                build_dir.mkdir()
                baton = FileBaton(str(build_dir / "lock"))
                self.assertTrue(baton.try_acquire())
                # model a dead owner: its descriptor is closed, its marker remains
                os.close(baton.fd)
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            processes = []
            try:
                for rank in range(2):
                    env = os.environ.copy()
                    env.update(
                        RANK=str(rank),
                        LOCAL_RANK=str(rank),
                        WORLD_SIZE="2",
                        MASTER_ADDR="127.0.0.1",
                        MASTER_PORT=str(port),
                        SGLANG_DIFFUSION_CACHE_ROOT=cache_dir,
                        SGLANG_JIT_CACHE_DIR=str(Path(cache_dir) / "jit"),
                        SGLANG_DIFFUSION_IPC_A2A="1",
                    )
                    processes.append(
                        subprocess.Popen(
                            [sys.executable, __file__, "--worker"],
                            env=env,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            start_new_session=True,
                        )
                    )
                for rank, process in enumerate(processes):
                    output, _ = process.communicate(timeout=120)
                    self.assertEqual(process.returncode, 0, output)
                    self.assertIn(f"IPC_RECOVERY PASS rank={rank}", output)
            finally:
                for process in processes:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.communicate()


if __name__ == "__main__":
    if "--worker" in sys.argv:
        _worker()
    else:
        unittest.main()
