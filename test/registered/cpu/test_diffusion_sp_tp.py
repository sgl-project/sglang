import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import sgl_kernel  # noqa: F401
import torch.multiprocessing as mp

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    get_ring_parallel_world_size,
    get_sp_world_size,
    get_tp_world_size,
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    find_available_port,
    get_tp_cpu_bind,
)

register_cpu_ci(est_time=800, suite="extra-tp-test-cpu-intel")

DEFAULT_WAN_MODEL_NAME_FOR_TEST = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
GENERATE_TIMEOUT = int(os.environ.get("WAN_GENERATE_TIMEOUT", "1800"))


class TestDiffusionCPUParallel(CustomTestCase):
    model = os.environ.get("WAN_MODEL_PATH", DEFAULT_WAN_MODEL_NAME_FOR_TEST)

    def test_cpu_tp2_sp2(self):
        num_workers = 4

        with tempfile.TemporaryDirectory(prefix="sglang_cpu_diffusion_") as tmp:
            output_dir = Path(tmp) / "outputs"
            output_dir.mkdir()
            output_file = output_dir / "test.mp4"
            log_path = Path(tmp) / "generate.log"

            command = [
                "sglang",
                "generate",
                "--model-path",
                self.model,
                "--tp-size",
                "2",
                "--sp-degree",
                "2",
                "--ulysses-degree",
                "2",
                "--ring-degree",
                "1",
                "--num-gpus",
                str(num_workers),
                "--prompt",
                "A cat walking in a garden.",
                "--width",
                "128",
                "--height",
                "128",
                "--num-frames",
                "5",
                "--num-inference-steps",
                "1",
                "--seed",
                "42",
                "--save-output",
                "--output-path",
                str(output_dir),
                "--output-file-name",
                output_file.name,
            ]

            env = os.environ.copy()

            bind = get_tp_cpu_bind(num_workers)
            if bind is not None:
                bind_groups = bind.split("|")
                if len(bind_groups) != num_workers:
                    raise RuntimeError(
                        f"SGLANG_CPU_OMP_THREADS_BIND provides {len(bind_groups)} "
                        f"groups, but the diffusion run needs {num_workers} workers."
                    )
                env["SGLANG_CPU_OMP_THREADS_BIND"] = bind

            timed_out = False
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
                try:
                    returncode = process.wait(timeout=GENERATE_TIMEOUT)
                except subprocess.TimeoutExpired:
                    timed_out = True
                finally:
                    kill_process_tree(process.pid)

            log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-16000:]

            debug_msg = f"Command: {' '.join(command)}\nGeneration log:\n{log_tail}"

            if timed_out:
                self.fail(
                    f"Generation timed out after {GENERATE_TIMEOUT}s.\n{debug_msg}"
                )

            self.assertEqual(returncode, 0, msg=debug_msg)

            self.assertTrue(
                output_file.is_file(),
                msg=(
                    f"Expected output video not found: {output_file}\n"
                    f"Actual files: {list(output_dir.rglob('*'))}\n"
                    f"{debug_msg}"
                ),
            )

            self.assertGreater(
                output_file.stat().st_size,
                0,
                msg=f"Generated video is empty.\n{debug_msg}",
            )


def _parallel_worker(
    rank: int,
    world_size: int,
    tp_size: int,
    sp_degree: int,
    init_method: str,
    master_port: int,
):
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        LOCAL_SIZE=str(world_size),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(master_port),
    )

    try:
        maybe_init_distributed_environment_and_model_parallel(
            tp_size=tp_size,
            sp_size=sp_degree,
            ulysses_degree=sp_degree,
            ring_degree=1,
            distributed_init_method=init_method,
        )
        assert get_tp_world_size() == tp_size
        assert get_sp_world_size() == sp_degree
        assert get_ring_parallel_world_size() == 1
    finally:
        cleanup_dist_env_and_memory()


class TestDiffusionCPUParallelUnit(CustomTestCase):
    def _run_parallel_test(self, tp_size: int, sp_degree: int):
        world_size = tp_size * sp_degree
        master_port = find_available_port(23456)
        init_method = f"tcp://127.0.0.1:{master_port}"
        mp.spawn(
            _parallel_worker,
            args=(world_size, tp_size, sp_degree, init_method, master_port),
            nprocs=world_size,
            join=True,
            start_method="spawn",
        )

    def test_cpu_tp_only(self):
        self._run_parallel_test(tp_size=2, sp_degree=1)

    def test_cpu_sp_only(self):
        self._run_parallel_test(tp_size=1, sp_degree=2)


if __name__ == "__main__":
    unittest.main()
