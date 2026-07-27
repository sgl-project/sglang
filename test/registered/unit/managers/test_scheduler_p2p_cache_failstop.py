import signal
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.p2p_kv_transfer import _P2PCacheIntegrityError
from sglang.srt.managers import scheduler as scheduler_module
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSchedulerP2PCacheFailStop(unittest.TestCase):
    def test_cache_integrity_error_signals_parent_and_shuts_down_metrics(self):
        scheduler = MagicMock()
        scheduler.get_init_info.return_value = {"ready": True}
        scheduler.run_event_loop.side_effect = _P2PCacheIntegrityError(
            "ownership settlement failed"
        )
        scheduler.gracefully_exit = False
        pipe_writer = MagicMock()
        parent_process = MagicMock()
        process = MagicMock()
        process.parent.return_value = parent_process
        server_args = SimpleNamespace(enable_trace=False)

        with (
            patch.object(scheduler_module, "load_plugins"),
            patch.object(
                scheduler_module,
                "configure_scheduler_process",
                return_value=0,
            ),
            patch.object(scheduler_module.psutil, "Process", return_value=process),
            patch.object(scheduler_module, "Scheduler", return_value=scheduler),
            patch.object(
                scheduler_module.envs.SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION,
                "get",
                return_value=False,
            ),
        ):
            scheduler_module.run_scheduler_process(
                server_args=server_args,
                port_args=SimpleNamespace(),
                gpu_id=0,
                tp_rank=0,
                attn_cp_rank=0,
                moe_dp_rank=0,
                moe_ep_rank=0,
                pp_rank=0,
                dp_rank=0,
                pipe_writer=pipe_writer,
            )

        scheduler.run_event_loop.assert_called_once_with()
        parent_process.send_signal.assert_called_once_with(signal.SIGQUIT)
        scheduler.metrics_reporter._shutdown_fpm.assert_called_once_with()
        scheduler.release_host_resources.assert_not_called()


if __name__ == "__main__":
    unittest.main()
