"""Unit tests for how sglang.test.test_utils derives per-device test server ports."""

import os
import subprocess
import sys

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    _device_index_from_visible_devices,
    _port_for_device_index,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MAX_TCP_PORT = 65535
# DEFAULT_URL_FOR_TEST is DEFAULT_PORT_FOR_SRT_TEST_RUNNER + 1000, so a derived
# base port must leave room for that offset and still be a valid TCP port.
URL_PORT_OFFSET = 1000
# (base, stride) for the non-CI and in-CI layouts respectively.
LAYOUTS = ((20000, 1000), (10000, 2000))


class TestDeviceIndexParsing(CustomTestCase):
    # (CUDA_VISIBLE_DEVICES, expected index, why)
    CASES = [
        (None, 0, "unset keeps the historical default"),
        ("", 0, "empty string hides every GPU; must not raise"),
        ("0", 0, "single digit"),
        ("7", 7, "single digit"),
        ("10", 10, "multi-digit must not be truncated to its first character"),
        ("2,3", 2, "comma separated list uses the first device"),
        ("11,12", 11, "multi-digit inside a list"),
        (" 1 ", 1, "surrounding whitespace is ignored"),
        ("2, 3", 2, "whitespace after the separator is ignored"),
        ("GPU-8f2e1c3a-0000-0000-0000-000000000000", 0, "UUID form degrades"),
        ("-1", 0, "negative is not a usable index"),
    ]

    def test_parsing_table(self):
        for raw, expected, why in self.CASES:
            with self.subTest(cuda_visible_devices=raw, why=why):
                self.assertEqual(_device_index_from_visible_devices(raw), expected)

    def test_ten_and_one_no_longer_share_a_port(self):
        # The previous implementation read only the first character, so device
        # 10 and device 1 derived the same port and raced for it.
        for base, stride in LAYOUTS:
            with self.subTest(base=base):
                self.assertNotEqual(
                    _port_for_device_index(
                        base, stride, _device_index_from_visible_devices("10")
                    ),
                    _port_for_device_index(
                        base, stride, _device_index_from_visible_devices("1")
                    ),
                )


class TestMalformedValuesDegradeGracefully(CustomTestCase):
    """CUDA_VISIBLE_DEVICES is user supplied. Values this helper cannot map to
    an index must degrade to the default port rather than raise: choosing a
    test port is a convenience, not a correctness requirement.

    Note this is a table of awkward inputs, not generative property testing --
    it exists so the fallback is exercised by values the cases above did not
    anticipate.
    """

    MALFORMED = [
        None,
        "",
        "  ",
        "\n",
        "abc",
        "none",
        "all",
        "GPU-1,GPU-2",
        "MIG-GPU-8f2e1c3a",
        "1.5",
        "1e3",
        "0x10",
        "-0",
        "--1",
        ",",
        ",,,",
        ",2",
        "1 2",
        "x" * 512,
    ]

    def test_malformed_values_fall_back_to_the_default_index(self):
        for raw in self.MALFORMED:
            with self.subTest(cuda_visible_devices=raw):
                self.assertEqual(_device_index_from_visible_devices(raw), 0)


class TestPortDerivation(CustomTestCase):
    def test_historical_ports_are_unchanged(self):
        # Indices that occur in practice must keep their existing ports,
        # otherwise running tests would silently move.
        self.assertEqual(_port_for_device_index(20000, 1000, 0), 20000)
        self.assertEqual(_port_for_device_index(20000, 1000, 1), 21000)
        self.assertEqual(_port_for_device_index(10000, 2000, 0), 10000)
        self.assertEqual(_port_for_device_index(10000, 2000, 3), 16000)

    def test_ports_are_distinct_across_plausible_device_counts(self):
        # Uniqueness is guaranteed below the wrap-around point, which is far
        # above any realistic single-host device count.
        for base, stride in LAYOUTS:
            with self.subTest(base=base):
                ports = [_port_for_device_index(base, stride, i) for i in range(16)]
                self.assertEqual(len(set(ports)), len(ports), msg=str(ports))

    def test_ports_wrap_instead_of_leaving_the_tcp_range(self):
        # Indices are no longer truncated, so they can be arbitrarily large.
        # Wrapping trades uniqueness for validity past the wrap point; a reused
        # port is recoverable ("address already in use"), an invalid one is not.
        for base, stride in LAYOUTS:
            slots = (MAX_TCP_PORT - URL_PORT_OFFSET - base) // stride + 1
            with self.subTest(base=base, slots=slots):
                for index in (0, 1, slots - 1, slots, 999, 10**6):
                    port = _port_for_device_index(base, stride, index)
                    self.assertGreater(port, 0)
                    self.assertLessEqual(port + URL_PORT_OFFSET, MAX_TCP_PORT)
                self.assertEqual(
                    _port_for_device_index(base, stride, slots),
                    _port_for_device_index(base, stride, 0),
                )


class TestModuleImportIsRobust(CustomTestCase):
    """The port is computed at import time, so a rejected value breaks the
    import itself rather than any single assertion."""

    def test_empty_cuda_visible_devices_does_not_break_import(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "from sglang.test.test_utils import DEFAULT_PORT_FOR_SRT_TEST_RUNNER as p; print(p)",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr[-2000:])
        self.assertGreater(int(proc.stdout.strip()), 0)
