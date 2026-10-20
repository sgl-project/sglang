"""Thread scheduling is an explicit seed propagated to the scheduler child."""

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import unittest

from sglang.srt.managers.scheduler_components.thread_scheduling import (
    configure_python_thread_scheduling,
)
from sglang.srt.runtime_context import get_device, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _scheduler_child(seed, connection):
    try:
        publish(seed, role="scheduler")
        configure_python_thread_scheduling(get_device().python_thread_switch_interval)
        connection.send((os.getpid(), sys.getswitchinterval()))
    finally:
        connection.close()


class TestPythonThreadSwitchIntervalSeed(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def test_unset_remains_none(self):
        seed = ServerArgs(model_path="dummy")
        publish(seed, role="test")
        self.assertIsNone(seed.python_thread_switch_interval)
        self.assertIsNone(get_device().python_thread_switch_interval)

    def test_existing_positional_field_order_is_preserved(self):
        fields = ServerArgs.__struct_fields__
        index = fields.index("sleep_on_idle")
        self.assertEqual(fields[index + 1], "use_ray")
        self.assertEqual(fields[-1], "python_thread_switch_interval")
        legacy = json.dumps(list(fields[:-1]), separators=(",", ":")).encode()
        self.assertEqual(len(fields[:-1]), 527)
        self.assertEqual(
            hashlib.sha256(legacy).hexdigest(),
            "7ee1f836a5af82275fc8fb37e9dd74903e956d8aef6ac786a51053a2bc5cfae9",
        )

    def test_cli_and_published_namespace_preserve_explicit_value(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        seed = ServerArgs.from_cli_args(
            parser.parse_args(
                ["--model", "dummy", "--python-thread-switch-interval", "0.001"]
            )
        )
        publish(seed, role="test")
        self.assertEqual(seed.python_thread_switch_interval, 0.001)
        self.assertEqual(get_device().python_thread_switch_interval, 0.001)

    def test_argument_validation_rejects_invalid_intervals(self):
        for value in (0.0, -0.001, float("nan"), float("inf")):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "positive and finite"),
            ):
                publish(
                    ServerArgs(model_path="dummy", python_thread_switch_interval=value),
                    role="test",
                )
            reset_context()

    def test_spawned_child_uses_seed_not_parent_interpreter(self):
        old = sys.getswitchinterval()
        self.addCleanup(sys.setswitchinterval, old)
        sys.setswitchinterval(0.003)
        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_scheduler_child,
            args=(
                ServerArgs(model_path="dummy", python_thread_switch_interval=0.001),
                sender,
            ),
        )
        process.start()
        sender.close()
        try:
            self.assertTrue(receiver.poll(45), "scheduler child did not report")
            pid, interval = receiver.recv()
            process.join(10)
            self.assertEqual(process.exitcode, 0)
            self.assertNotEqual(pid, os.getpid())
            self.assertAlmostEqual(interval, 0.001, places=8)
            self.assertAlmostEqual(sys.getswitchinterval(), 0.003, places=8)
        finally:
            receiver.close()
            if process.is_alive():
                process.terminate()
                process.join(10)


if __name__ == "__main__":
    unittest.main()
