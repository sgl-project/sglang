# SPDX-License-Identifier: Apache-2.0
"""``--enable-speculative-step-cuda-graph`` turns on its six fused pieces.

CPU only. A launch's server args (parsed from the command line, through the
speculative resolution step, and published as a scheduler process publishes
them) decide what each piece's switch reads:
  * the flag alone: every piece is on;
  * the flag and one switch set to 0: only that piece is off;
  * no flag, a setup that turns the flag off at startup, or no published
    config: every piece is off;
  * a switch set to 1 without the flag: that piece is on (a test override).
"""

import argparse
import contextlib
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.environ import EnvField, envs
from sglang.srt.runtime_context import (
    ParallelContext,
    RuntimeContext,
    create_context,
    use_context,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative import eagle_step_cuda_graph_runner as runner_module
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SWITCHES = (
    envs.SGLANG_SPEC_PACKED_VERIFY_BROADCAST,
    envs.SGLANG_SPEC_LEAN_SAMPLED_VERIFY,
    envs.SGLANG_SPEC_CHUNKED_VERIFY_SAMPLING,
    envs.SGLANG_SPEC_FUSED_RELAY,
    envs.SGLANG_SPEC_FUSED_DRAFT_EXTEND_TAIL,
    envs.SGLANG_SPEC_STEP_GRAPH_FUSED_INPUT_COPY,
)


def _argv(flag, topk=1, draft_tokens=4):
    """An EAGLE launch command line, with or without the flag."""
    argv = ["--model-path", "dummy", "--device", "cuda"]
    argv += ["--speculative-algorithm", "EAGLE", "--speculative-num-steps", "3"]
    argv += ["--speculative-eagle-topk", str(topk)]
    argv += ["--speculative-num-draft-tokens", str(draft_tokens)]
    return argv + (["--enable-speculative-step-cuda-graph"] if flag else [])


@contextlib.contextmanager
def _cuda_platform(cuda=True):
    """Pretend to run on CUDA with cuda-python and the native allocator."""
    with (
        mock.patch.object(runner_module, "is_cuda", return_value=cuda),
        mock.patch.object(runner_module, "_cuda_python_available", return_value=True),
        mock.patch.object(torch.cuda, "get_allocator_backend", return_value="native"),
    ):
        yield


@contextlib.contextmanager
def _no_switch_set():
    """None of the six switches is set; the environment is restored after."""
    with mock.patch.dict(os.environ):
        for switch in SWITCHES:
            switch.clear()
        yield


def _launch(argv, cuda=True):
    """Server args parsed from ``argv``, through the launch's speculative
    resolution step (the step that keeps the flag or turns it off)."""
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = ServerArgs.from_cli_args(parser.parse_args(argv))
    args._model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"])
    )
    with _cuda_platform(cuda):
        handle_speculative_decoding(args)
    return args


@contextlib.contextmanager
def _scheduler_process(args):
    """The config a scheduler process publishes for ``args``."""
    with use_context(create_context(args, role="scheduler")):
        yield


def _values():
    return {switch.name: switch.get() for switch in SWITCHES}


class TestSwitchesFollowTheFlag(CustomTestCase):
    def test_the_flag_has_six_switches(self):
        # Every env var whose default follows the flag, and only those.
        default = envs.SGLANG_SPEC_FUSED_RELAY.default
        followers = {
            field.name
            for field in vars(type(envs)).values()
            if isinstance(field, EnvField) and field.default is default
        }
        self.assertEqual(followers, {switch.name for switch in SWITCHES})
        self.assertEqual(len(followers), 6)

    def test_flag_alone_turns_every_piece_on(self):
        args = _launch(_argv(flag=True))
        self.assertTrue(resolution_result(args, "enable_speculative_step_cuda_graph"))
        with _no_switch_set(), _scheduler_process(args):
            self.assertEqual(_values(), {switch.name: True for switch in SWITCHES})

    def test_each_override_turns_only_its_piece_off(self):
        args = _launch(_argv(flag=True))
        with _no_switch_set(), _scheduler_process(args):
            for off in SWITCHES:
                with self.subTest(switch=off.name):
                    with mock.patch.dict(os.environ, {off.name: "0"}):
                        self.assertEqual(
                            _values(),
                            {switch.name: switch is not off for switch in SWITCHES},
                        )

    def test_without_the_flag_every_piece_is_off(self):
        args = _launch(_argv(flag=False))
        self.assertFalse(resolution_result(args, "enable_speculative_step_cuda_graph"))
        all_off = {switch.name: False for switch in SWITCHES}
        with _no_switch_set():
            with _scheduler_process(args):
                self.assertEqual(_values(), all_off)
            # No server config published (tools, unit tests).
            with use_context(RuntimeContext(parallel=ParallelContext())):
                self.assertEqual(_values(), all_off)

    def test_flag_turned_off_at_startup_turns_every_piece_off(self):
        cases = {
            "not CUDA": dict(argv=_argv(flag=True), cuda=False),
            "topk 2": dict(
                argv=_argv(flag=True, topk=2, draft_tokens=8) + ["--page-size", "1"]
            ),
            "pipeline parallelism": dict(argv=_argv(flag=True) + ["--pp-size", "2"]),
        }
        for name, case in cases.items():
            with self.subTest(case=name):
                args = _launch(**case)
                self.assertFalse(
                    resolution_result(args, "enable_speculative_step_cuda_graph")
                )
                with _no_switch_set(), _scheduler_process(args):
                    self.assertEqual(
                        _values(), {switch.name: False for switch in SWITCHES}
                    )

    def test_override_on_without_the_flag(self):
        args = _launch(_argv(flag=False))
        with _no_switch_set(), _scheduler_process(args):
            for on in SWITCHES:
                with self.subTest(switch=on.name):
                    with mock.patch.dict(os.environ, {on.name: "1"}):
                        self.assertEqual(
                            _values(),
                            {switch.name: switch is on for switch in SWITCHES},
                        )


if __name__ == "__main__":
    unittest.main()
