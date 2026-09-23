"""DP startup must report a dead worker even while another group is loading."""

import multiprocessing as mp
import os
import queue
import threading
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

register_cpu_ci(est_time=90, suite="base-a-test-cpu")


def _scheduler_worker(*args):
    server_args, rank = args[0], args[8]
    if rank == 0:
        server_args.loading.set()
        server_args.release.wait(30)
        os._exit(24)
    if not server_args.loading.wait(30):
        os._exit(25)
    os._exit(23)


def _cleanup_startup(controller, launcher, threads, signals, pipes, release):
    release.set()
    for process in controller.scheduler_procs:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
    # Unblock both bare recv and the old Event implementation on regression runs.
    for _, writer in pipes:
        try:
            writer.send({"status": "test cleanup"})
        except (BrokenPipeError, OSError):
            pass
    for signal in signals:
        if isinstance(signal, queue.Queue):
            signal.put(RuntimeError("test cleanup"))
        else:
            signal.set()
    for thread in threads.values():
        thread.join(timeout=7)
    launcher.join(timeout=2)
    for reader, writer in pipes:
        reader.close()
        writer.close()


class TestDataParallelControllerStartup(CustomTestCase):
    def test_later_group_failure_interrupts_earlier_group_loading(self):
        """A child that exits without ready must reach the controller's caller."""
        from sglang.srt.managers import data_parallel_controller as dp

        context = mp.get_context("spawn")
        server_args = SimpleNamespace(loading=context.Event(), release=context.Event())
        outcomes = queue.Queue()
        group_threads, completion_signals, pipes = {}, [], []
        controller = dp.DataParallelController.__new__(dp.DataParallelController)
        controller.env_lock = threading.Lock()
        controller.scheduler_procs = []
        controller.run_scheduler_process_func = _scheduler_worker
        controller.context, controller.workers = object(), [None, None]
        create_pipe = context.Pipe

        def tracked_pipe(*args, **kwargs):
            pair = create_pipe(*args, **kwargs)
            pipes.append(pair)
            return pair

        def launch_thread(*args):
            group_threads[args[3]] = threading.current_thread()
            completion_signals.append(args[-1])
            dp.DataParallelController.launch_tensor_parallel_group_thread(
                controller, *args
            )

        controller.launch_tensor_parallel_group_thread = launch_thread
        ports = SimpleNamespace(
            tokenizer_ipc_name="unused", detokenizer_ipc_name="unused", instance_id=0
        )

        def launch():
            try:
                controller.launch_dp_schedulers(server_args, ports)
            except Exception as error:
                outcomes.put(error)
            else:
                outcomes.put(None)

        launcher = threading.Thread(target=launch, daemon=True)
        parallel = SimpleNamespace(
            dp_size=2,
            tp_size=1,
            pp_size=1,
            node_rank=0,
            nnodes=1,
            enable_dp_attention=False,
            attn_cp_size=1,
            moe_dp_size=1,
            ep_size=1,
            ep_join_rank_offset=0,
        )
        execution = SimpleNamespace(
            features=SimpleNamespace(enable_memory_saver=False),
            moe=SimpleNamespace(is_ep_scale_joiner=False),
        )
        with (
            patch.object(dp, "mp", context),
            patch.object(context, "Pipe", side_effect=tracked_pipe),
            patch.object(dp, "get_parallel", return_value=parallel),
            patch.object(dp, "get_exec", return_value=execution),
            patch.object(
                dp,
                "get_device",
                return_value=SimpleNamespace(base_gpu_id=0, gpu_id_step=1),
            ),
            patch.object(dp, "maybe_reindex_device_id", side_effect=nullcontext),
            patch.object(
                dp.TorchMemorySaverAdapter,
                "create",
                return_value=SimpleNamespace(configure_subprocess=nullcontext),
            ),
            patch.object(
                dp.numa_utils,
                "configure_subprocess",
                side_effect=lambda *_: nullcontext(),
            ),
            patch.object(
                dp.PortArgs,
                "init_new",
                side_effect=lambda _: SimpleNamespace(
                    nccl_port=0, scheduler_input_ipc_name="unused"
                ),
            ),
            patch.object(dp, "bind_port", return_value=MagicMock()),
            patch.object(dp, "get_zmq_socket", return_value=MagicMock()),
        ):
            try:
                launcher.start()
                self.assertTrue(
                    server_args.loading.wait(60), "first group did not start"
                )
                error = outcomes.get(timeout=15)
                self.assertIsInstance(error, RuntimeError)
                self.assertIn("exit code: 23", str(error))
                self.assertFalse(server_args.release.is_set())
                self.assertTrue(group_threads[0].is_alive())
                launcher.join(timeout=2)
                self.assertFalse(launcher.is_alive())
            finally:
                _cleanup_startup(
                    controller,
                    launcher,
                    group_threads,
                    completion_signals,
                    pipes,
                    server_args.release,
                )
        self.assertFalse(launcher.is_alive())
        self.assertTrue(all(not thread.is_alive() for thread in group_threads.values()))

    def test_ready_group_keeps_its_launcher_thread_alive(self):
        """Reporting ready must not terminate the scheduler's parent thread."""
        from sglang.srt.managers import data_parallel_controller as dp

        controller = dp.DataParallelController.__new__(dp.DataParallelController)
        controller.launch_tensor_parallel_group = lambda *args: None
        startup_results = queue.Queue()
        parked, stop = threading.Event(), threading.Event()

        def park(seconds):
            parked.set()
            stop.wait(15)
            raise SystemExit

        def launch():
            try:
                controller.launch_tensor_parallel_group_thread(
                    None, None, 0, 0, startup_results
                )
            except SystemExit:
                pass

        thread = threading.Thread(target=launch, daemon=True)
        with patch.object(dp, "time", SimpleNamespace(sleep=park)):
            try:
                thread.start()
                self.assertIsNone(startup_results.get(timeout=5))
                self.assertTrue(parked.wait(5))
                self.assertTrue(thread.is_alive())
            finally:
                stop.set()
                thread.join(timeout=5)
        self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
