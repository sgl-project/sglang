"""
Unit tests for when NemotronHMoE overlaps shared and routed experts.

The overlap runs the routed experts on a side stream. Dynamo cannot replay
those stream switches, so a compiled forward must take the serial path.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.models import nemotron_h
from sglang.test.test_utils import CustomTestCase


def _moe():
    return SimpleNamespace(
        _forward_core_shared_routed_overlap=lambda hidden: (hidden + 1, None),
        _forward_core_normal=lambda hidden: (hidden + 2, None),
    )


def _patched(*, compiling=None, capturing=False, flashinfer=False):
    backend = SimpleNamespace(is_flashinfer=lambda: flashinfer)
    patches = [
        mock.patch.object(nemotron_h, "_is_cuda", True),
        mock.patch.object(nemotron_h, "get_moe_a2a_backend", return_value=backend),
        mock.patch.object(nemotron_h, "get_is_capture_mode", return_value=capturing),
    ]
    if compiling is not None:
        patches.append(
            mock.patch.object(
                nemotron_h.torch.compiler, "is_compiling", return_value=compiling
            )
        )
    return patches


def _path(**kwargs) -> str:
    patches = _patched(**kwargs)
    for patch in patches:
        patch.start()
    try:
        output, _ = nemotron_h.NemotronHMoE._forward_core(_moe(), torch.zeros(1))
    finally:
        for patch in reversed(patches):
            patch.stop()
    return {1.0: "overlap", 2.0: "serial"}[output.item()]


class TestNemotronHMoEStreamOverlap(CustomTestCase):
    def test_eager_cuda_overlaps_the_routed_experts(self):
        self.assertEqual(_path(compiling=False), "overlap")

    def test_compiled_forward_runs_the_experts_serially(self):
        self.assertEqual(_path(compiling=True), "serial")
        self.assertEqual(_path(compiling=True, capturing=True), "serial")

    def test_flashinfer_a2a_overlaps_only_under_graph_capture(self):
        self.assertEqual(_path(compiling=False, flashinfer=True), "serial")
        self.assertEqual(
            _path(compiling=False, flashinfer=True, capturing=True), "overlap"
        )

    def test_dynamo_traces_the_serial_path(self):
        moe = _moe()
        patches = _patched()
        for patch in patches:
            patch.start()
        try:
            compiled = torch.compile(
                lambda hidden: nemotron_h.NemotronHMoE._forward_core(moe, hidden)[0],
                backend="eager",
            )
            output = compiled(torch.zeros(1))
        finally:
            for patch in reversed(patches):
                patch.stop()
        self.assertEqual(output.item(), 2.0)


if __name__ == "__main__":
    unittest.main()
