# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the capture check of a draft spec, fired by the EAGLE draft graph
runners' capture hook call: after the draft forwards of each captured batch size,
with the runner's batch sizes, under the draft decode's first per-step backend."""

import ast
import inspect
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from msgspec.structs import replace

from sglang.srt.megakernels import runtime
from sglang.srt.megakernels import spec as registry
from sglang.srt.megakernels.binding import MegakernelBinding
from sglang.srt.megakernels.spec import Step
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fakes import ARCH, ISA, SPEC, Lm, batch, server_args  # noqa: E402

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class DraftBuilt:
    """Covers draft decode steps of up to max_bs rows: one row a request, so off the
    width grid."""

    draft_sampler = None

    def __init__(self, widths, max_bs):
        self.widths, self.max_bs = widths, max_bs

    def summary(self):
        return "fake draft"

    def plan(self, input_ids, forward_batch, *args, **kwargs):
        rows = int(input_ids.shape[0])
        return Step(mode="draft_decode", rows=rows) if rows <= self.max_bs else None

    def forward(self, step, original, input_ids, positions, forward_batch, **kw):
        return "megakernel"


def draft_spec(max_bs):
    return replace(
        SPEC, build=lambda model, hf_config, ctx: DraftBuilt(ctx.widths, max_bs)
    )


def draft_decode_runner(hooks, capture_bs=(), backends=None):
    """An EAGLE draft decode runner with only what its capture hook call reads."""
    pool = object()
    runner = EAGLEDraftCudaGraphRunner.__new__(EAGLEDraftCudaGraphRunner)
    runner.capture_bs = list(capture_bs)
    runner.model_runner = SimpleNamespace(capture_tail_hooks=hooks)
    runner.draft_attn_backend = SimpleNamespace(
        attn_backends=backends
        or [SimpleNamespace(token_to_kv_pool=pool, step=i) for i in range(3)]
    )
    return runner


def capture_draft_graphs(spec, capture_bs, steps=2):
    """Install spec for a draft model, then capture its graphs as the EAGLE draft
    decode runner does: the draft forwards of a batch size, then its capture hook
    call."""
    lm, hooks = Lm(), []
    gpu = SimpleNamespace(multi_processor_count=160)
    with (
        patch.object(runtime, "find_spec", return_value=(spec, "")),
        patch.object(runtime, "device_isa", return_value=ISA),
        patch.object(runtime, "deadline_s", return_value=30.0),
        patch.object(runtime, "_gather_tp", side_effect=lambda x, group=None: [x]),
        patch.object(runtime, "_tokens_per_request", return_value=4),
        patch.object(runtime, "_decode_graphs_enabled", return_value=True),
        patch.object(torch.cuda, "current_device", return_value=0),
        patch.object(torch.cuda, "get_device_properties", return_value=gpu),
        patch.object(torch.cuda, "mem_get_info", return_value=(0, 0)),
    ):
        runtime.install_megakernel(
            model=lm,
            hf_config=SimpleNamespace(architectures=[ARCH]),
            capture_tail_hooks=hooks,
            is_draft=True,
        )
    runner = draft_decode_runner(hooks, capture_bs)
    with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
        for bs in capture_bs:
            out = [lm(batch(bs).input_ids, None, batch(bs)) for _ in range(steps)]
            runner._run_capture_tail_hooks(out, batch(bs), bs)
    return hooks


class TestMegakernelDraftCapture(CustomTestCase):
    def setUp(self):
        registry_patch = patch.dict(registry._REGISTRY, clear=True)
        registry_patch.start()
        self.addCleanup(registry_patch.stop)
        registry.register(architecture=ARCH, path="_fakes:SPEC")
        publish(server_args(enable_nvidia_megakernel=True), role="test")
        self.addCleanup(reset_context)
        view_patch = patch.object(
            MegakernelBinding, "forward_view", return_value="view"
        )
        view_patch.start()
        self.addCleanup(view_patch.stop)

    def test_a_draft_runner_capture_is_checked(self):
        with (
            patch.object(MegakernelBinding, "assert_live") as assert_live,
            self.assertLogs(runtime.logger, "INFO") as logs,
        ):
            hooks = capture_draft_graphs(draft_spec(max_bs=2), capture_bs=[1, 2, 4])
        self.assertEqual(len(hooks), 1)
        assert_live.assert_called_once_with()
        self.assertIn(
            "runs the decode graphs of batch sizes [1, 2] (2 of 3)",
            "".join(logs.output),
        )

    def test_a_draft_runner_with_no_routed_graph_is_refused(self):
        with self.assertRaisesRegex(ValueError, "none of the 2 captured decode graphs"):
            capture_draft_graphs(draft_spec(max_bs=0), capture_bs=[1, 2])


class TestEagleDraftRunnersFireTheHooks(CustomTestCase):
    def test_the_draft_decode_runner_uses_its_first_step_backend(self):
        seen = []
        runner = draft_decode_runner(
            [lambda *args: seen.append((args, get_attn_backend()))]
        )
        runner._run_capture_tail_hooks("out", "forward batch", 8)
        first = runner.draft_attn_backend.attn_backends[0]
        self.assertEqual(seen, [((runner, "out", "forward batch", 8), first)])

    def test_step_backends_on_different_pools_are_refused(self):
        backends = [SimpleNamespace(token_to_kv_pool=object()) for _ in range(2)]
        runner = draft_decode_runner([lambda *args: None], backends=backends)
        with self.assertRaisesRegex(RuntimeError, "do not share one KV pool"):
            runner._run_capture_tail_hooks("out", "forward batch", 8)

    def test_without_hooks_nothing_is_read(self):
        runner = draft_decode_runner([], backends=[None])
        runner._run_capture_tail_hooks("out", "forward batch", 8)

    def test_the_draft_extend_runner_fires_the_hooks(self):
        seen = []
        runner = EAGLEDraftExtendCudaGraphRunner.__new__(
            EAGLEDraftExtendCudaGraphRunner
        )
        runner.model_runner = SimpleNamespace(
            capture_tail_hooks=[lambda *args: seen.append(args)]
        )
        runner._run_capture_tail_hooks("out", "forward batch", 16)
        self.assertEqual(seen, [(runner, "out", "forward batch", 16)])

    def test_the_captured_functions_call_the_hooks(self):
        """capture_one_shape's captured function ends with the hook call, with its
        output, batch and token count."""
        for cls in (EAGLEDraftCudaGraphRunner, EAGLEDraftExtendCudaGraphRunner):
            with self.subTest(runner=cls.__name__):
                tree = ast.parse(inspect.getsource(inspect.getmodule(cls)))
                owner = next(
                    n
                    for n in ast.walk(tree)
                    if isinstance(n, ast.ClassDef) and n.name == cls.__name__
                )
                capture = next(
                    n
                    for n in owner.body
                    if isinstance(n, ast.FunctionDef) and n.name == "capture_one_shape"
                )
                run_once = next(
                    n
                    for n in ast.walk(capture)
                    if isinstance(n, ast.FunctionDef) and n.name == "run_once"
                )
                calls = [
                    [ast.unparse(a) for a in n.args]
                    for n in ast.walk(run_once)
                    if isinstance(n, ast.Call)
                    and ast.unparse(n.func) == "self._run_capture_tail_hooks"
                ]
                self.assertEqual(calls, [["ret", "forward_batch", "num_tokens"]])


if __name__ == "__main__":
    unittest.main()
