import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.srt.mem_cache.storage.hf3fs.mini_3fs_metadata_server import (
    GlobalMetadataState,
    Hf3fsMetadataServer,
    RankMetadata,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHf3fsMetadataShutdown(unittest.TestCase):
    def test_run_stops_timer_and_saves_on_return_or_error(self):
        for error in (None, RuntimeError("serving failed"), KeyboardInterrupt()):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "metadata.json"
                server = Hf3fsMetadataServer(str(path), save_interval=60)

                def serve(*args, **kwargs):
                    metadata = RankMetadata(2)
                    metadata.reserve_and_allocate_page_indices([("a", "")])
                    metadata.confirm_write([("a", 1)], [])
                    server.state.ranks["0:kv"] = metadata
                    if error is not None:
                        raise error

                try:
                    with patch("uvicorn.run", side_effect=serve):
                        if error is None:
                            server.run()
                        else:
                            with self.assertRaises(type(error)) as raised:
                                server.run()
                            self.assertIs(raised.exception, error)
                    self.assertTrue(server.state.is_shutting_down)
                    self.assertFalse(server.state.save_timer.is_alive())
                    self.assertEqual(
                        json.loads(path.read_text())["0:kv"]["key_to_index"],
                        [["a", 1]],
                    )
                finally:
                    server.state.shutdown()

    def test_shutdown_waits_for_active_save_without_rearming(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            state = GlobalMetadataState(str(path), save_interval=60)
            state.ranks["0:kv"] = RankMetadata(2)
            saving = threading.Event()
            release_save = threading.Event()
            shutdown_progress = threading.Event()
            final_save_started = threading.Event()
            errors = []
            dump = json.dump
            timer = threading.Timer(0, state.schedule_save)
            state.save_timer = timer
            join_timer = timer.join

            def gated_dump(*args, **kwargs):
                if threading.current_thread() is timer:
                    saving.set()
                    if not release_save.wait(5):
                        raise TimeoutError("periodic save gate was not released")
                else:
                    final_save_started.set()
                    shutdown_progress.set()
                return dump(*args, **kwargs)

            def observed_join(*args, **kwargs):
                shutdown_progress.set()
                return join_timer(*args, **kwargs)

            def shutdown():
                try:
                    state.shutdown()
                except BaseException as error:
                    errors.append(error)

            closer = threading.Thread(target=shutdown, daemon=True)
            with (
                patch(
                    "sglang.srt.mem_cache.storage.hf3fs.mini_3fs_metadata_server.json.dump",
                    side_effect=gated_dump,
                ),
                patch.object(timer, "join", side_effect=observed_join),
            ):
                timer.start()
                try:
                    self.assertTrue(saving.wait(5), "periodic save did not start")
                    closer.start()
                    self.assertTrue(
                        shutdown_progress.wait(5), "shutdown did not reach the timer"
                    )
                    final_save_overlapped = final_save_started.is_set()
                finally:
                    release_save.set()
                    if closer.ident is not None:
                        closer.join(5)
                    join_timer(5)
                    # Clean up a wrongly rearmed timer even on a failed assertion.
                    state.save_timer.cancel()
                    state.save_timer.join(5)
            self.assertFalse(closer.is_alive(), "shutdown did not finish")
            self.assertFalse(timer.is_alive(), "periodic save did not finish")
            self.assertEqual(errors, [])
            self.assertFalse(final_save_overlapped)
            self.assertIs(state.save_timer, timer, "shutdown rearmed persistence")
            self.assertTrue(final_save_started.is_set())
            self.assertEqual(json.loads(path.read_text())["0:kv"]["free_pages"], [0, 1])

    def test_shutdown_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            state = GlobalMetadataState(str(Path(directory) / "metadata.json"), 60)
            with patch.object(state, "save_to_disk", wraps=state.save_to_disk) as save:
                state.shutdown()
                state.shutdown()
            save.assert_called_once_with()

    def test_run_without_persistence(self):
        server = Hf3fsMetadataServer()
        with patch("uvicorn.run") as serve:
            server.run(host="127.0.0.1", port=1234)
        serve.assert_called_once_with(server.app, host="127.0.0.1", port=1234)
        self.assertIsNone(server.state.save_timer)


if __name__ == "__main__":
    unittest.main()
