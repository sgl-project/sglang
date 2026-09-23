"""Keep child monitoring active throughout engine startup, including failures."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from sglang.srt.entrypoints import engine
from sglang.srt.runtime_context import reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.mark.parametrize("failure_stage", [None, "tokenizer", "scheduler"])
def test_watchdog_covers_startup_and_stops_on_failure(failure_stage):
    server_args = ServerArgs(model_path="dummy", tokenizer_worker_num=1)
    server_args.check_server_args = MagicMock()
    manager = object.__new__(engine.TokenizerManager)
    manager.cuda_vmm_feature_transport = MagicMock()
    monitor = MagicMock()

    def check_stage(stage):
        monitor.start.assert_called_once_with()
        monitor.stop.assert_not_called()
        monitor.mark_startup_complete.assert_not_called()
        if failure_stage == stage:
            raise RuntimeError(f"{stage} failed")

    def init_tokenizer(*args):
        check_stage("tokenizer")
        return manager, None

    scheduler = SimpleNamespace(
        all_child_pids=[],
        scheduler_infos=[{"max_req_input_len": 128}],
        wait_for_ready=lambda: check_stage("scheduler"),
    )
    try:
        with (
            patch.object(engine, "configure_logger"),
            patch.object(engine, "_set_envs_and_config"),
            patch.object(engine, "load_plugins"),
            patch.object(engine, "SubprocessWatchdog", return_value=monitor) as factory,
            patch.object(
                engine.Engine,
                "_launch_scheduler_processes",
                return_value=(scheduler, []),
            ),
            patch.object(
                engine.Engine, "_launch_detokenizer_subprocesses", return_value=([], [])
            ),
            patch.object(engine.Engine, "_set_startup_time"),
        ):
            kwargs = dict(
                server_args=server_args,
                init_tokenizer_manager_func=init_tokenizer,
                run_scheduler_process_func=MagicMock(),
                run_detokenizer_process_func=MagicMock(),
                port_args=SimpleNamespace(),
            )
            if failure_stage:
                with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
                    engine.Engine._launch_subprocesses(**kwargs)
                monitor.stop.assert_called_once_with()
                monitor.mark_startup_complete.assert_not_called()
            else:
                result = engine.Engine._launch_subprocesses(**kwargs)
                assert result[0] is manager and result[4] is monitor
                monitor.mark_startup_complete.assert_called_once_with()
                monitor.stop.assert_not_called()
            factory.assert_called_once_with(
                processes=[], process_names=[], startup=True
            )
            if failure_stage == "scheduler":
                manager.cuda_vmm_feature_transport.shutdown.assert_called_once_with()
            else:
                manager.cuda_vmm_feature_transport.shutdown.assert_not_called()
    finally:
        reset_context()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
