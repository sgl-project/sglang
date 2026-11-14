"""The `--crash-dump-folder` pickle: the Python TokenizerManager and the Rust
server's scheduler write the same file, which the dump readers load."""

import dataclasses
import functools
import glob
import importlib.util
import json
import os
import pickle
import signal
import sys
import tempfile
import threading
import time
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints import engine as engine_module  # noqa: E402
from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers import scheduler as scheduler_module  # noqa: E402
from sglang.srt.managers.io_struct import GenerateReqInput  # noqa: E402
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402
from sglang.srt.observability import request_dump as crash_dump_module  # noqa: E402
from sglang.srt.observability.request_dump import (  # noqa: E402
    CrashDumpGrace,
    dumping_marker,
    wait_for_crash_dumps,
    write_crash_dump,
)
from sglang.srt.runtime_context import create_context, use_context  # noqa: E402
from sglang.srt.rust_server import server as server_module  # noqa: E402
from sglang.srt.rust_server.server import RustServer  # noqa: E402
from sglang.srt.server_args import ServerArgs  # noqa: E402
from sglang.srt.utils.serialization import safe_pickle_load  # noqa: E402
from sglang.srt.utils.watchdog import WatchdogRaw  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_SGLANG_DIR = os.path.join(os.path.dirname(__file__), *[os.pardir] * 4)


def _load_script(*path: str):
    """A dump reader that lives outside the package, imported from its file."""
    spec = importlib.util.spec_from_file_location(
        path[-1].removesuffix(".py"), os.path.join(_SGLANG_DIR, *path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _request(rid: str) -> GenerateReqInput:
    obj = GenerateReqInput(
        rid=rid, text="Hello", sampling_params={"max_new_tokens": 4, "stop": ["\n"]}
    )
    obj.normalize_batch_and_arguments()
    return obj


def _out(rid: str) -> dict:
    return {
        "text": " world",
        "output_ids": [11, 12],
        "meta_info": {"id": rid, "prompt_tokens": 3, "completion_tokens": 2},
    }


def _load_only_dump(folder: str) -> tuple:
    (path,) = glob.glob(os.path.join(folder, "*", "crash_dump_*.pkl"))
    with open(path, "rb") as f:
        return path, pickle.load(f)


# Every secret a dump must not carry. The watermark keys have no flag; the
# command line spells the others as argparse accepts them.
_SECRETS = {
    "api_key": "api-secret",
    "admin_api_key": "admin-secret",
    "ssl_keyfile_password": "ssl-secret",
    "watermark_key": "abcdef0123456789",
    "watermark_key_b": "0123456789abcdef",
}
_SECRET_ARGV = (
    "launch_server",
    "--api-key",
    "api-secret",
    "--admin-api-ke=admin-secret",
    "--ssl-keyfile-password",
    "ssl-secret",
)


def _assert_secrets_redacted(test: CustomTestCase, path: str, dump: dict):
    with open(path, "rb") as f:
        dumped = f.read()
    for secret in _SECRETS.values():
        test.assertNotIn(secret.encode(), dumped)
    for name in _SECRETS:
        test.assertEqual(dump["server_args"][name], "<redacted>")
        test.assertEqual(dump["resolved_config"][name], "<redacted>")
    test.assertEqual(
        dump["launch_command"],
        "launch_server --api-key <redacted> --admin-api-ke=<redacted> "
        "--ssl-keyfile-password <redacted>",
    )


class TestTokenizerManagerCrashDump(CustomTestCase):
    def test_file_name_and_contents(self):
        """The TokenizerManager's dump: one file per crash under the host's
        folder, holding the finished and unfinished requests with the launch
        state."""
        with tempfile.TemporaryDirectory() as folder:
            finished = (_request("done"), _out("done"), 100.0, 101.5)
            unfinished_state = SimpleNamespace(
                finished=False,
                obj=_request("running"),
                out_list=[],
                get_crash_dump_output=lambda: {"text": " par", "output_ids": [7]},
                time_stats=SimpleNamespace(
                    created_time=1.0,
                    set_finished_time=lambda: None,
                    finished_time=2.0,
                ),
            )
            tm = SimpleNamespace(
                crash_dump_folder=folder,
                crash_dump_performed=False,
                # The finished history is kept pickled: (finish time, bytes).
                crash_dump_request_list=deque([(finished[3], pickle.dumps(finished))]),
                rid_to_state={"running": unfinished_state},
                server_args=SimpleNamespace(watermark_config=None),
                _server_args_for_dump=lambda: {"model_path": "m"},
                _dump_config_snapshot=lambda: {"resolved": True},
            )
            with (
                patch(
                    "sglang.srt.managers.tokenizer_manager.convert_time_to_realtime",
                    side_effect=lambda t: t + 1000.0,
                ),
                patch.object(sys, "argv", ["launch_server", "--model-path", "m"]),
                # Only the pickle: no py-spy, CUDA coredump or their sleep.
                envs.SGLANG_PYSPY_DUMP_BEFORE_CRASH.override(False),
                envs.SGLANG_CUDA_COREDUMP_BEFORE_CRASH.override(False),
            ):
                TokenizerManager.dump_requests_before_crash(tm, hostname="host-a")
                # Once per process: a second dump would add a host-b file.
                TokenizerManager.dump_requests_before_crash(tm, hostname="host-b")

            path, dump = _load_only_dump(folder)
            self.assertEqual(os.path.dirname(path), os.path.join(folder, "host-a"))
            self.assertRegex(
                os.path.basename(path),
                r"^crash_dump_\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d\.pkl$",
            )
            # Preserve the file mode produced by ordinary open() under the current umask.
            umask = os.umask(0)
            os.umask(umask)
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o666 & ~umask)
            self.assertEqual(
                list(dump),
                [
                    "server_args",
                    "config_updates",
                    "resolved_config",
                    "requests",
                    "launch_command",
                ],
            )
            self.assertEqual(dump["server_args"], {"model_path": "m"})
            self.assertEqual(dump["resolved_config"], {"resolved": True})
            self.assertEqual(dump["launch_command"], "launch_server --model-path m")
            self.assertIsInstance(dump["config_updates"], list)
            (done, running) = dump["requests"]
            self.assertEqual(done[0], finished[0])
            self.assertEqual(done[1:], finished[1:])
            self.assertEqual(running[0], unfinished_state.obj)
            self.assertEqual(
                running[1:], ({"text": " par", "output_ids": [7]}, 1001.0, 1002.0)
            )

    def test_server_args_snapshot_redacted_with_watermarking_off(self):
        # The request dump pickles this snapshot as is. The Python API accepts
        # a server key without --enable-watermark, and validation then skips it.
        server_args = ServerArgs(model_path="dummy", **_SECRETS)
        context = create_context(server_args, role="test")
        self.assertFalse(context.config_bag("exec").features.enable_watermark)
        tm = SimpleNamespace(server_args=server_args)
        with use_context(context):
            dumped = pickle.dumps(TokenizerManager._server_args_for_dump(tm))
        for secret in _SECRETS.values():
            self.assertNotIn(secret.encode(), dumped)
        self.assertIn(b"<redacted>", dumped)

    def test_secrets_redacted_whatever_the_writer_passes(self):
        """`write_crash_dump` redacts on its own, so the dump carries no
        secret even when the writer passes the server args unredacted, as a
        dict or as the record."""
        server_args = ServerArgs(model_path="dummy", **_SECRETS)
        finished = (_request("done"), _out("done"), 100.0, 101.5)
        for passed in (server_args.resolved_dict(), server_args):
            tm = SimpleNamespace(
                crash_dump_folder=None,
                crash_dump_performed=False,
                crash_dump_request_list=deque([(finished[3], pickle.dumps(finished))]),
                rid_to_state={},
                server_args=server_args,
                _server_args_for_dump=lambda: passed,
                _dump_config_snapshot=server_args.resolved_dict,
            )
            with (
                self.subTest(type(passed).__name__),
                tempfile.TemporaryDirectory() as folder,
                use_context(create_context(server_args, role="test")),
                patch.object(sys, "argv", list(_SECRET_ARGV)),
                envs.SGLANG_PYSPY_DUMP_BEFORE_CRASH.override(False),
                envs.SGLANG_CUDA_COREDUMP_BEFORE_CRASH.override(False),
            ):
                tm.crash_dump_folder = folder
                TokenizerManager.dump_requests_before_crash(tm, hostname="host")
                _assert_secrets_redacted(self, *_load_only_dump(folder))

    def test_server_args_snapshot_failure_keeps_the_dump(self):
        # Before the redaction, a watermarking-off dump pickled ServerArgs
        # without snapshotting it; a snapshot failure must not cost the dump.
        class Unresolvable:
            def resolved_dict(self):
                raise RuntimeError("boom")

        tm = SimpleNamespace(server_args=Unresolvable())
        self.assertIsNone(TokenizerManager._server_args_for_dump(tm))


# The `Server.crash_dump_snapshot()` JSON the Rust server returns
# (rust/sglang-server/src/api_server/core/request_log.rs).
_SINGLE_BODY = {
    "rid": "single",
    "text": "Hello",
    "sampling_params": {"max_new_tokens": 4},
    "stream": True,
}
_BATCH_BODY = {
    "rid": ["b0", "b1"],
    "input_ids": [[1, 2], [3, 4, 5]],
    "sampling_params": {"temperature": 0.0},
}
# A request the Rust server built itself (an OpenAI route), in Python's
# field names; its rid is a minted one.
_TYPED_FIELDS = {
    "rid": "typed",
    "text": "chat prompt",
    "input_ids": None,
    "sampling_params": {"max_new_tokens": 8, "temperature": 0.5},
    "return_logprob": False,
    "stream": False,
}
# Each body once, however many items of a batch name it.
_SNAPSHOT = {
    "bodies": {
        "0": _SINGLE_BODY,
        "1": _BATCH_BODY,
        "2": _TYPED_FIELDS,
        "3": {"input_ids": [7, 8]},
    },
    "finished": [
        {
            "body": 0,
            "batch_index": None,
            "rid": "single",
            "out": _out("single"),
            "created_time": 10.0,
            "finished_time": 11.0,
        },
        {
            "body": 1,
            "batch_index": 1,
            "rid": "b1",
            "out": _out("b1"),
            "created_time": 12.0,
            "finished_time": 13.0,
        },
        {
            "body": 2,
            "batch_index": None,
            "rid": "typed",
            "out": _out("typed"),
            "created_time": 14.0,
            "finished_time": 15.0,
        },
    ],
    "unfinished": [
        {
            "body": 1,
            "batch_index": 0,
            "rid": "b0",
            "out": {},
            "created_time": 12.0,
            "finished_time": 17.0,
        },
        {
            "body": 3,
            "batch_index": None,
            "rid": "minted",
            "out": {"output_ids": [9]},
            "created_time": 16.0,
            "finished_time": 17.0,
        },
    ],
}


class _Panic(BaseException):
    """What a Rust panic surfaces as in Python (pyo3's `PanicException`)."""


def _rust_server(snapshot: dict) -> RustServer:
    server = RustServer(
        server=SimpleNamespace(
            crash_dump_snapshot=lambda: json.dumps(snapshot),
            flush_request_log=lambda: None,
        ),
        http_port=0,
    )
    # Picklable, as a real `ServerArgs` is, and allowed by `safe_pickle_load`.
    server.server_args = SimpleNamespace(
        watermark_config=None, resolved_dict=functools.partial(dict, model_path="m")
    )
    return server


def _normalized(body: dict, index=None) -> GenerateReqInput:
    obj = GenerateReqInput(**body)
    obj.normalize_batch_and_arguments()
    return obj if index is None else obj[index]


class TestRustServerCrashDump(CustomTestCase):
    def _dump(self, server: RustServer, folder: str, argv=("launch_server",)):
        with (
            patch.object(
                server_module,
                "get_observability",
                return_value=SimpleNamespace(crash_dump_folder=folder),
            ),
            patch.object(
                server_module,
                "get_context",
                return_value=SimpleNamespace(
                    resolved_server_args_dict=lambda base: {**base, "resolved": True}
                ),
            ),
            patch.object(sys, "argv", list(argv)),
        ):
            server.dump_requests_before_crash()
            # Once per process: the watchdog may follow the exception path.
            server.dump_requests_before_crash()

    def test_dump_holds_the_objects_python_would(self):
        """Each record becomes the object Python's TokenizerManager holds:
        the body normalized by Python's code, the batch item it was, the rid
        the client sees and a finished request's ids as a tuple."""
        with tempfile.TemporaryDirectory() as folder:
            self._dump(_rust_server(_SNAPSHOT), folder)
            path, dump = _load_only_dump(folder)

        # One file per process that hosts a listener.
        self.assertRegex(
            os.path.basename(path),
            rf"^crash_dump_\d{{4}}-\d\d-\d\d_\d\d-\d\d-\d\d_{os.getpid()}\.pkl$",
        )
        self.assertEqual(dump["server_args"], {"model_path": "m"})
        self.assertEqual(dump["resolved_config"], {"model_path": "m", "resolved": True})
        self.assertEqual(dump["launch_command"], "launch_server")
        single, batch_item, typed, other_item, unfinished = dump["requests"]

        self.assertEqual(single, (_normalized(_SINGLE_BODY),) + single[1:])
        self.assertEqual(single[1:], (_out("single"), 10.0, 11.0))

        expected_item = _normalized(_BATCH_BODY, 1)
        expected_item.input_ids = (3, 4, 5)
        self.assertEqual(batch_item[0], expected_item)
        self.assertEqual(batch_item[0].sampling_params, {"temperature": 0.0})

        self.assertEqual(typed[0], _normalized(_TYPED_FIELDS))

        # The other item of the same body; an unfinished request keeps its ids.
        self.assertEqual(other_item[0], _normalized(_BATCH_BODY, 0))
        self.assertEqual(other_item[0].input_ids, [1, 2])

        expected_unfinished = _normalized({"input_ids": [7, 8]})
        expected_unfinished.rid = "minted"
        self.assertEqual(unfinished[0], expected_unfinished)
        self.assertEqual(unfinished[1:], ({"output_ids": [9]}, 16.0, 17.0))

    def test_secrets_redacted(self):
        """The dump carries no secret, as Python's does. Rust mode refuses
        `--enable-watermark`, but the Python API takes a server key without
        it and validation then skips the key."""
        server_args = ServerArgs(model_path="dummy", **_SECRETS)
        server = _rust_server(_SNAPSHOT)
        server.server_args = server_args
        with tempfile.TemporaryDirectory() as folder:
            with use_context(create_context(server_args, role="test")):
                self._dump(server, folder, argv=_SECRET_ARGV)
            _assert_secrets_redacted(self, *_load_only_dump(folder))

    def test_dump_loads_with_the_readers(self):
        """`safe_pickle_load`, the replay script and the incident triage tool
        all read the file."""
        replay = _load_script("scripts", "playground", "replay_request_dump.py")
        triage = _load_script(
            ".agents",
            "skills",
            "sglang-prod-incident-triage",
            "scripts",
            "incident_artifact_tool.py",
        )
        with tempfile.TemporaryDirectory() as folder:
            self._dump(_rust_server(_SNAPSHOT), folder)
            path, _ = _load_only_dump(folder)
            with open(path, "rb") as f:
                self.assertEqual(len(safe_pickle_load(f)["requests"]), 5)
            records = replay.read_records([path])
            summary = triage.summarize_dump_file(
                triage.Path(path), max_requests=10, preview_chars=40
            )
        self.assertEqual(
            [dataclasses.asdict(req)["rid"] for req, *_ in records],
            ["single", "b1", "typed", "b0", "minted"],
        )
        self.assertIn("Requests: 5", summary)
        self.assertIn("rid=single stream=True", summary)

    def test_a_failed_dump_does_not_raise(self):
        """The dump runs on a crash path: even a Rust panic must not stop the
        scheduler from reporting its failure."""

        def panic():
            raise _Panic("snapshot panicked")

        server = _rust_server(_SNAPSHOT)
        server.server.crash_dump_snapshot = panic
        with tempfile.TemporaryDirectory() as folder:
            self._dump(server, folder)
            self.assertEqual([f for _, _, files in os.walk(folder) for f in files], [])

    def test_a_stalled_dump_does_not_hold_back_sigquit(self):
        """A snapshot that never returns must not keep the scheduler from
        sending the SIGQUIT that starts the main process's grace period."""
        release = threading.Event()

        def stalled_snapshot():
            release.wait()
            return json.dumps({"bodies": {}, "finished": [], "unfinished": []})

        server = _rust_server(_SNAPSHOT)
        server.server.crash_dump_snapshot = stalled_snapshot
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(server_module, "CRASH_DUMP_GRACE_SECS", 0.05),
        ):
            caller = threading.Thread(target=self._dump, args=(server, folder))
            caller.start()
            caller.join(5)
            returned = not caller.is_alive()
            release.set()
            caller.join()
            for thread in threading.enumerate():
                if thread.name == "crash-dump":
                    thread.join()
        self.assertTrue(returned)

    def test_no_partial_file(self):
        """A dump that cannot be written leaves no file behind, partial or
        temporary, for the readers to trip on."""
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(Exception):
                write_crash_dump(
                    folder,
                    "host-a",
                    [(lambda: None, {}, 0.0, 0.0)],  # never pickles
                    server_args=None,
                    resolved_config=None,
                    watermark_config=None,
                )
            self.assertEqual(os.listdir(os.path.join(folder, "host-a")), [])

    def test_no_requests_no_file(self):
        with tempfile.TemporaryDirectory() as folder:
            self._dump(
                _rust_server({"bodies": {}, "finished": [], "unfinished": []}), folder
            )
            self.assertEqual([f for _, _, files in os.walk(folder) for f in files], [])

    def test_marker_held_while_writing(self):
        """The main process waits on the marker this process holds while it
        writes, and it is gone once the dump is in place."""
        seen = []
        real_write = server_module.write_crash_dump

        def write(folder, hostname, *args, **kwargs):
            seen.append(os.listdir(os.path.join(folder, hostname)))
            return real_write(folder, hostname, *args, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            with patch.object(server_module, "write_crash_dump", side_effect=write):
                self._dump(_rust_server(_SNAPSHOT), folder)
            path, _ = _load_only_dump(folder)
            (host,) = os.listdir(folder)
            self.assertEqual(
                os.listdir(os.path.join(folder, host)), [os.path.basename(path)]
            )
        self.assertEqual(seen, [[f".dumping.{os.getpid()}"]])


class TestCrashDumpTriggers(CustomTestCase):
    def test_scheduler_exception_dumps_before_sigquit(self):
        """A scheduler that raises writes the Rust server's dump before it
        asks the parent to kill the process tree."""
        calls = []
        scheduler = MagicMock()
        scheduler.run_event_loop.side_effect = RuntimeError("boom")
        scheduler.dump_requests_before_crash.side_effect = lambda: calls.append("dump")
        parent = MagicMock()
        parent.send_signal.side_effect = lambda sig: calls.append(sig)
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
            ),
            patch.object(scheduler_module.psutil, "Process") as process,
        ):
            process.return_value.parent.return_value = parent
            with self.assertRaises(SystemExit) as exited:
                scheduler_module.run_scheduler_process(
                    server_args=MagicMock(),
                    port_args=MagicMock(),
                    gpu_id=0,
                    tp_rank=0,
                    pp_rank=0,
                    dp_rank=None,
                    pipe_writer=MagicMock(),
                )
        self.assertEqual(calls, ["dump", signal.SIGQUIT])
        # A crash must not exit 0, which SubprocessWatchdog reads as a clean shutdown.
        self.assertEqual(exited.exception.code, 1)

    def test_watchdog_timeout_dumps_before_sigquit(self):
        """The watchdog's own thread writes the dump on a timeout, then still
        asks the parent to kill the tree when the dump fails."""
        real_sleep = time.sleep
        calls = []
        signalled = threading.Event()

        def before_crash():
            calls.append("dump")
            raise _Panic("dump failed")

        parent = MagicMock()
        parent.send_signal.side_effect = lambda sig: (
            calls.append(sig),
            signalled.set(),
        )
        active = [True]
        with (
            patch("sglang.srt.utils.watchdog.pyspy_dump_schedulers"),
            patch("sglang.srt.utils.watchdog.psutil.Process") as process,
            patch("sglang.srt.utils.watchdog.time.sleep", lambda s: real_sleep(0.001)),
        ):
            process.return_value.parent.return_value = parent
            WatchdogRaw(
                debug_name="test",
                get_counter=lambda: 0,
                is_active=lambda: active[0],
                watchdog_timeout=0.01,
                before_crash=before_crash,
            )
            self.assertTrue(signalled.wait(10))
            active[0] = False
        self.assertEqual(calls[:2], ["dump", signal.SIGQUIT])


class TestCrashDumpGrace(CustomTestCase):
    """Rust mode's main process waits for a dump under way (its writer holds a
    marker) before a child's SIGQUIT kills the tree, and does not wait when
    no dump is being written."""

    def _wait(self, folder, pids, **kwargs):
        start = time.monotonic()
        reason = wait_for_crash_dumps(
            folder, "host-a", pids, poll_interval=0.01, **kwargs
        )
        return reason, time.monotonic() - start

    def test_no_known_writer_no_wait(self):
        with tempfile.TemporaryDirectory() as folder:
            reason, elapsed = self._wait(folder, [])
        self.assertEqual(reason, "none")
        self.assertLess(elapsed, 0.1)

    def test_no_dump_started_ends_after_the_start_window(self):
        with tempfile.TemporaryDirectory() as folder:
            reason, elapsed = self._wait(folder, [123], start_timeout=0.3, timeout=10)
        self.assertEqual(reason, "none")
        self.assertGreaterEqual(elapsed, 0.3)
        self.assertLess(elapsed, 3)

    def test_ends_when_the_dump_is_written(self):
        with tempfile.TemporaryDirectory() as folder:
            marker = dumping_marker(folder, "host-a", 123)
            os.makedirs(os.path.dirname(marker))
            open(marker, "wb").close()
            threading.Timer(0.3, os.unlink, [marker]).start()
            # Another process's marker in a shared folder does not count.
            open(dumping_marker(folder, "host-a", 999), "wb").close()
            reason, elapsed = self._wait(folder, [123], timeout=10)
        self.assertEqual(reason, "written")
        self.assertLess(elapsed, 5)

    def test_bound_is_honored(self):
        with tempfile.TemporaryDirectory() as folder:
            marker = dumping_marker(folder, "host-a", 123)
            os.makedirs(os.path.dirname(marker))
            open(marker, "wb").close()
            reason, elapsed = self._wait(folder, [123], timeout=0.5)
        self.assertEqual(reason, "timeout")
        self.assertGreaterEqual(elapsed, 0.5)
        self.assertLess(elapsed, 3)

    def test_first_sigquit_waits_then_the_old_handler_kills_on_the_main_thread(self):
        """After the wait the grace signals the process again, and that
        SIGQUIT runs the handler it replaced; reports during the wait start
        nothing new."""
        previous = MagicMock()
        resignalled = threading.Event()
        release = threading.Event()
        waits = []

        def before_wait():
            waits.append(1)
            release.wait(10)

        with tempfile.TemporaryDirectory() as folder:
            grace = CrashDumpGrace(
                folder, [], previous=previous, before_wait=before_wait
            )
            with patch.object(
                crash_dump_module.os,
                "kill",
                side_effect=lambda pid, sig: resignalled.set(),
            ) as kill:
                grace.handle_sigquit(signal.SIGQUIT, None)
                grace.handle_sigquit(signal.SIGQUIT, None)  # while waiting
                release.set()
                self.assertTrue(resignalled.wait(10))
            kill.assert_called_once_with(os.getpid(), signal.SIGQUIT)
            previous.assert_not_called()
            grace.handle_sigquit(signal.SIGQUIT, None)
        previous.assert_called_once_with(signal.SIGQUIT, None)
        self.assertEqual(waits, [1])

    def test_rust_mode_installs_the_grace_for_the_hosting_schedulers(self):
        result = SimpleNamespace(
            scheduler_infos=[{"rust_server_pid": 11}, {}, {"rust_server_pid": 12}]
        )
        watchdog = MagicMock()
        launch_phase = MagicMock()
        with (
            patch.object(
                engine_module,
                "get_observability",
                return_value=SimpleNamespace(crash_dump_folder="/dumps"),
            ),
            patch.object(
                engine_module,
                "resolving_view",
                return_value=SimpleNamespace(custom_sigquit_handler=None),
            ),
            patch.object(engine_module.signal, "signal") as install,
            patch.object(engine_module.signal, "getsignal", return_value=launch_phase),
        ):
            engine_module.Engine._wait_for_rust_crash_dumps_on_sigquit(
                MagicMock(), result, watchdog
            )
        (signum, handler), _ = install.call_args
        self.assertEqual(signum, signal.SIGQUIT)
        grace = handler.__self__
        self.assertEqual(grace.writer_pids, [11, 12])
        self.assertEqual(grace.folder, "/dumps")
        self.assertIs(grace.before_wait, watchdog.stop)
        # The launch-phase handler still does the kill.
        self.assertIs(grace.previous, launch_phase)

        # Without a dump folder the launch-phase handler stays.
        with (
            patch.object(
                engine_module,
                "get_observability",
                return_value=SimpleNamespace(crash_dump_folder=None),
            ),
            patch.object(engine_module.signal, "signal") as install,
        ):
            engine_module.Engine._wait_for_rust_crash_dumps_on_sigquit(
                MagicMock(), result, watchdog
            )
        install.assert_not_called()


if __name__ == "__main__":
    unittest.main()
