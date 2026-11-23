"""run_scheduler_process's graceful exit frees captured graphs before the communicators."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.managers import scheduler as scheduler_module
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSchedulerGracefulExit(CustomTestCase):
    def _run_scheduler_process(self, events, *, loop_error=None):
        scheduler = MagicMock()
        scheduler.gracefully_exit = True
        scheduler.async_d2h_worker = None
        scheduler.run_event_loop.side_effect = loop_error
        scheduler.model_worker.release_cuda_graphs.side_effect = lambda: events.append(
            "release graphs"
        )
        scheduler.release_host_resources.side_effect = lambda: events.append(
            "release host resources"
        )
        with (
            patch.multiple(
                scheduler_module,
                load_plugins=MagicMock(),
                publish=MagicMock(),
                spawn_world_rank=MagicMock(),
                configure_scheduler_process=MagicMock(),
                get_observability=MagicMock(
                    return_value=SimpleNamespace(enable_trace=False)
                ),
                Scheduler=MagicMock(return_value=scheduler),
                close_custom_all_reduce=lambda: events.append(
                    "close custom all-reduce"
                ),
                abort_distributed_environment=lambda: events.append(
                    "abort distributed"
                ),
            ),
            patch.object(scheduler_module.psutil, "Process"),
            patch.object(scheduler_module.os, "killpg"),
        ):
            scheduler_module.run_scheduler_process(
                server_args=MagicMock(),
                port_args=MagicMock(),
                gpu_id=0,
                tp_rank=0,
                pp_rank=0,
                dp_rank=None,
                pipe_writer=MagicMock(),
            )

    def test_graphs_are_released_before_the_communicators(self):
        """Captured graphs used to outlive close_custom_all_reduce() and
        abort_distributed_environment(); a graph backend whose captured
        collectives keep communicator state could then only be torn down after
        its communicator was gone."""
        events = []
        self._run_scheduler_process(events)

        self.assertEqual(
            events,
            [
                "release graphs",
                "release host resources",
                "close custom all-reduce",
                "abort distributed",
            ],
        )

    def test_crash_after_a_shutdown_request_skips_the_release(self):
        """A shutdown request can arrive and the same iteration then hit a device
        error. Releasing would synchronize a possibly wedged device and hang
        before the crash exit; it now takes the crash path only."""
        events = []
        with self.assertRaises(SystemExit):
            self._run_scheduler_process(events, loop_error=RuntimeError("device"))

        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
