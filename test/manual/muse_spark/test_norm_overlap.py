"""Weight-free Muse Spark host scheduling and CUDA graph regression tests.

Run with only PyTorch installed:
    python test/manual/muse_spark/test_norm_overlap.py -v

The production host methods are extracted without importing model/kernel
backends. Small torch operators replace the pooler, router and experts: this
checks scheduling, not FlashInfer kernels, distributed execution or accuracy
of a loaded model. CUDA tests skip on CPU-only machines.
"""

import ast
import contextlib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

SOURCE = Path(__file__).resolve().parents[3] / "python/sglang/srt/models/muse_spark.py"


def load_methods(scope, source=SOURCE):
    tree = ast.parse(Path(source).read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MuseSpark1xMoE"
    )
    methods = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("_forward_experts", "_forward_sp_experts")
    ]
    module = ast.Module(
        body=[*ast.parse("from __future__ import annotations").body, *methods],
        type_ignores=[],
    )
    exec(compile(module, str(source), "exec"), scope)
    return scope


class Fixture:
    def __init__(
        self,
        device="cpu",
        *,
        sp=False,
        nvfp4=False,
        projection=True,
        norm=True,
        split=False,
        hidden=32,
        expert_hidden=32,
        num_experts=8,
        top_k=2,
        source=SOURCE,
    ):
        self.events = []
        self.sp = sp
        self.device = device
        self.side = torch.cuda.Stream() if device == "cuda" else None
        self.weight = torch.randn(expert_hidden, hidden, device=device)
        self.router_weight = torch.randn(num_experts, hidden, device=device)
        self.forward = SimpleNamespace(sp_active=sp)
        self.scope = load_methods(
            dict(
                torch=torch,
                contextlib=contextlib,
                _is_cpu=device == "cpu",
                get_forward=lambda: self.forward,
                _use_split_bf16_router=lambda _: split,
                bf16_gemm_dispatch=lambda x, w, _: self.project(x),
                get_global_experts_capturer=lambda: None,
            ),
            source,
        )
        experts = self.Experts(self)
        self.model = SimpleNamespace(
            router_in_fp32=split,
            gate=self.Gate(self),
            nvfp4_checkpoint=nvfp4,
            experts=experts,
            tp_size=1,
            top_k=top_k,
            expert_hidden_size=expert_hidden,
            pre_expert_proj=SimpleNamespace(weight=self.weight) if projection else None,
            pre_expert_norm=self.norm if norm else None,
            _side_stream=lambda: self.side,
            _routing_op=lambda: self.route,
        )
        if device == "cpu" and projection:
            self.model.pre_expert_proj = lambda x: (self.project(x), None)
        self.model._forward_sp_experts = self.sp_experts

    def project(self, x):
        self.events.append("project")
        return F.linear(x, self.weight)

    def norm(self, x):
        self.events.append("norm")
        return F.rms_norm(x, (x.shape[-1],))

    class Gate:
        def __init__(self, fixture):
            self.fixture = fixture
            self.weight = fixture.router_weight

        def __call__(self, x):
            self.fixture.events.append("gate")
            return F.linear(x.float(), self.weight)

        def forward_split2(self, x):
            return self(x)

    def route(self, _, logits, k, renormalize, want_packed):
        self.events.append("route")
        scores, ids = torch.topk(torch.sigmoid(logits), k)
        scores = scores / scores.sum(dim=-1, keepdim=True)
        return scores, ids, ids.clone()

    class Experts:
        post_expert_norm = None
        use_marlin = False

        def __init__(self, fixture):
            self.fixture = fixture

        def prepare_input(self, x, norm):
            self.fixture.events.append("prepare")
            return norm(x) if norm else x

        def __call__(
            self,
            x,
            scores,
            ids,
            post_norm,
            *,
            pre_expert_norm=None,
            prepared_input=None,
            **kwargs,
        ):
            self.fixture.events.append("experts")
            if prepared_input is not None:
                x = prepared_input
            elif pre_expert_norm is not None:
                x = pre_expert_norm(x)
            # Depend on every routing result as well as the normalized input.
            return x * ((ids.float() + 1) * scores).sum(-1, keepdim=True)

    def sp_experts(self, x, scores, ids, logits):
        self.events.append("sp")
        if self.model.pre_expert_norm:
            x = self.model.pre_expert_norm(x)
        return self.model.experts(x, scores, ids, None)

    def run(self, x, router_input=None):
        return self.scope["_forward_experts"](self.model, x, router_input)

    def reference(self, x, router_input=None):
        # Serial mathematical oracle, independent of the host scheduling method.
        expert_input = (
            F.linear(x, self.weight) if self.model.pre_expert_proj is not None else x
        )
        if self.model.pre_expert_norm:
            expert_input = F.rms_norm(expert_input, (expert_input.shape[-1],))
        logits = F.linear(
            x.float() if router_input is None else router_input, self.router_weight
        )
        scores, ids = torch.topk(torch.sigmoid(logits), self.model.top_k)
        scores /= scores.sum(-1, keepdim=True)
        return expert_input * ((ids.float() + 1) * scores).sum(-1, keepdim=True)


class Stream:
    def __init__(self, name, events):
        self.name, self.events = name, events

    def wait_stream(self, other):
        self.events.append(f"{self.name} waits {other.name}")


class TestHostSchedule(unittest.TestCase):
    def test_fork_norm_join_and_lifetime(self):
        for sp in (False, True):
            for nvfp4 in (False, True):
                for capture in (False, True):
                    with self.subTest(sp=sp, nvfp4=nvfp4, capture=capture):
                        f = Fixture(sp=sp, nvfp4=nvfp4, projection=False)
                        f.scope["_is_cpu"] = False
                        main, side = Stream("main", f.events), Stream("side", f.events)
                        f.side = side
                        with (
                            patch.object(torch.cuda, "is_available", return_value=True),
                            patch.object(
                                torch.cuda, "current_stream", return_value=main
                            ),
                            patch.object(
                                torch.cuda,
                                "stream",
                                side_effect=lambda _: contextlib.nullcontext(),
                            ),
                            patch.object(
                                torch.cuda,
                                "is_current_stream_capturing",
                                return_value=capture,
                            ),
                            patch.object(
                                torch.Tensor,
                                "record_stream",
                                side_effect=lambda _: f.events.append("record"),
                            ),
                        ):
                            x = torch.randn(4, 32)
                            torch.testing.assert_close(f.run(x), f.reference(x))
                        self.assertEqual(f.events.count("norm"), 1)
                        self.assertLess(
                            f.events.index("side waits main"), f.events.index("gate")
                        )
                        self.assertLess(
                            f.events.index("route"), f.events.index("main waits side")
                        )
                        self.assertLess(
                            f.events.index("main waits side"), f.events.index("experts")
                        )
                        if not sp:
                            self.assertLess(
                                f.events.index("norm"),
                                f.events.index("main waits side"),
                            )
                        else:
                            self.assertLess(
                                f.events.index("main waits side"),
                                f.events.index("norm"),
                            )
                        self.assertEqual(f.events.count("record"), 0 if capture else 3)

    def test_int4_selection_is_cpu_only(self):
        # Load the production selection predicates without GPU dependencies.
        tree = ast.parse(SOURCE.read_text())
        functions = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name in ("_quantization_name", "_uses_gptq_int4_experts")
        ]
        scope = {}
        exec(
            compile(
                ast.Module(
                    body=[
                        *ast.parse("from __future__ import annotations").body,
                        *functions,
                    ],
                    type_ignores=[],
                ),
                str(SOURCE),
                "exec",
            ),
            scope,
        )
        configs = [
            SimpleNamespace(get_name=lambda: "gptq"),
            SimpleNamespace(get_name=lambda: "auto-round", packing_format="auto_gptq"),
            SimpleNamespace(get_name=lambda: "auto-round", backend="gptq"),
        ]
        for cpu in (False, True):
            scope["_is_cpu"] = cpu
            for config in configs:
                self.assertEqual(scope["_uses_gptq_int4_experts"](config), cpu)
            self.assertFalse(scope["_uses_gptq_int4_experts"](None))
        # The INT4 caller uses the non-NVFP4 CPU path. Its order stays serial:
        # projection -> gate -> routing -> norm -> experts, with no CUDA stream.
        f = Fixture(device="cpu", nvfp4=False, projection=True)
        f.run(torch.randn(3, 32))
        self.assertEqual(f.events, ["project", "gate", "route", "norm", "experts"])

    def test_cpu_fallback_and_empty(self):
        for projection in (False, True):
            for norm in (False, True):
                for tokens in (0, 1, 129):
                    with self.subTest(projection=projection, norm=norm, tokens=tokens):
                        f = Fixture(projection=projection, norm=norm)
                        x = torch.randn(tokens, 32)
                        torch.testing.assert_close(f.run(x), f.reference(x))
                        if tokens == 0:
                            self.assertEqual(f.events, [])


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestCudaSchedule(unittest.TestCase):
    def test_eager_and_graph_replay(self):
        for projection in (False, True):
            for split in (False, True):
                for tokens in (1, 16, 129):
                    with self.subTest(
                        projection=projection, split=split, tokens=tokens
                    ):
                        f = Fixture("cuda", projection=projection, split=split)
                        x = torch.randn(
                            tokens,
                            32,
                            device="cuda",
                            dtype=torch.bfloat16 if split else torch.float32,
                        )
                        # Projection uses float32 for this weight-free scheduling fixture.
                        if split:
                            f.weight = f.weight.bfloat16()
                            if projection:
                                f.model.pre_expert_proj.weight = f.weight
                        for _ in range(3):
                            torch.testing.assert_close(f.run(x), f.reference(x))
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            output = f.run(x)
                        for _ in range(5):
                            x.normal_()  # Replays must use new routing and norm inputs.
                            graph.replay()
                            torch.testing.assert_close(output, f.reference(x))

    def test_allocator_churn_on_two_call_streams(self):
        fixtures = [Fixture("cuda"), Fixture("cuda")]
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        results = []
        for _ in range(20):
            for f, stream in zip(fixtures, streams):
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    x = torch.randn(16, 32, device="cuda")
                    results.append((f.run(x), f.reference(x)))
                    # Drop router intermediates, then encourage allocator reuse.
                    torch.empty((16, 32), device="cuda").fill_(float("nan"))
        torch.cuda.synchronize()
        for output, expected in results:
            torch.testing.assert_close(output, expected)


if __name__ == "__main__":
    unittest.main()
