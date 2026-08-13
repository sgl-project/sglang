"""Unit tests for load reporter fields in ServerArgs."""

from __future__ import annotations

import argparse
import sys

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _parse(args: list[str]):
    """Parse args through ServerArgs CLI into a ServerArgs instance."""
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    namespace = parser.parse_args(["--model-path", "dummy"] + args)
    return ServerArgs.from_cli_args(namespace)


class TestLoadReporterPortDefault:
    def test_default_is_none(self):
        sa = _parse([])
        assert sa.load_reporter_port is None

    def test_none_means_disabled(self):
        """None is the only disabled sentinel; no extra enable flag required."""
        sa = _parse([])
        assert sa.load_reporter_port is None


class TestLoadReporterPortParsing:
    def test_valid_low_boundary(self):
        sa = _parse(["--load-reporter-port", "1"])
        assert sa.load_reporter_port == 1

    def test_valid_typical(self):
        sa = _parse(["--load-reporter-port", "30100"])
        assert sa.load_reporter_port == 30100

    def test_valid_high_boundary(self):
        sa = _parse(["--load-reporter-port", "65535"])
        assert sa.load_reporter_port == 65535


class TestLoadReporterPortValidation:
    def test_zero_is_rejected(self):
        with pytest.raises((ValueError, SystemExit)):
            _parse(["--load-reporter-port", "0"])

    def test_above_max_is_rejected(self):
        with pytest.raises((ValueError, SystemExit)):
            _parse(["--load-reporter-port", "65536"])

    def test_negative_is_rejected(self):
        with pytest.raises((ValueError, SystemExit)):
            _parse(["--load-reporter-port", "-1"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
