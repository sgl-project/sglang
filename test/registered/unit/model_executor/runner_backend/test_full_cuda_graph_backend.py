"""CPU tests for FULL profiling hooks and FULL/BCG grouped capture.

These cover the changes from the "cuda graph profile traces" PR that wire the
runner's torch profiler into the capture loop:

  * When profiling is disabled, ``capture_one`` runs exactly two warmups + one
    capture and never touches a profiler (behavior-identical to before the PR).
  * When the runner exposes an active ``_profiler`` (per-bs capture profiling,
    ``--enable-profile-cuda-graph`` + ``SGLANG_GRAPH_BATCH_CAPTURE``),
    ``capture_one`` calls ``profiler.step()`` past the two warmups and once after
    the capture (schedule ``wait=2, warmup=0, active=1``). The captured forward is
    NOT wrapped in a ``record_function``; per-bs trace naming is handled by the
    profiler's ``on_trace_ready`` callback instead.
  * The ``getattr`` guards mean a runner that sets the flag but has no
    ``_profiler`` attribute degrades gracefully (no stepping, no crash).

The real capture path needs CUDA (``torch.cuda.CUDAGraph`` + device graph
context), so those are mocked; the logic under test (call counts, ordering,
profiler stepping) is pure-Python and runs on CPU.
"""

import contextlib
import unittest
import weakref
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_backend import breakable_cuda_graph_backend as bcg
from sglang.srt.model_executor.runner_backend import full_cuda_graph_backend as full
from sglang.srt.model_executor.runner_backend import utils as backend_utils
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    FullCudaGraphBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# Sentinel: distinguishes "runner has no _profiler attribute" from
# "_profiler is None" in the test fixtures.
_UNSET = object()


class _FakeGraphCtx:
    """Stand-in for ``device_module.graph(...)`` — a no-op context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _make_backend(runner):
    """Build a ``FullCudaGraphBackend`` without running ``__init__`` (which would
    touch CUDA), wiring just the attributes ``capture_one`` reads."""
    backend = FullCudaGraphBackend.__new__(FullCudaGraphBackend)
    backend._group_preparation_enabled = False
    backend._prepared_groups = set()
    backend._graphs = {}
    backend._outputs = {}
    backend._capture_inputs = {}
    backend._pool = None
    backend._capture_stream = None
    backend._precarve = SimpleNamespace(
        measure=contextlib.nullcontext, mint=mock.Mock()
    )
    backend._reuse_output_buffer = False
    backend._output_buffer = None
    backend._memory_saver_adapter = None
    backend._cuda_graph_runner = runner
    backend._device_module = runner.device_module
    backend._tp_group = SimpleNamespace(barrier=mock.Mock(name="barrier"))
    return backend


def _make_runner(*, enable_profile, profiler, num_tokens_per_bs=1, mode_name="DECODE"):
    device_module = SimpleNamespace(
        synchronize=mock.Mock(name="synchronize"),
        graph=mock.Mock(name="graph", side_effect=lambda **kw: _FakeGraphCtx()),
    )
    runner = SimpleNamespace(
        device_module=device_module,
        num_tokens_per_bs=num_tokens_per_bs,
        capture_forward_mode=SimpleNamespace(name=mode_name),
        enable_profile_cuda_graph=enable_profile,
    )
    if profiler is not _UNSET:
        runner._profiler = profiler
    return runner


class TestCaptureOneNoProfiling(CustomTestCase):
    def test_capture_inputs_stay_alive_until_graph_reset(self):
        backend = _make_backend(_make_runner(enable_profile=False, profiler=None))
        counts = torch.tensor([64, 64])
        reference = weakref.ref(counts)
        graph = mock.Mock()
        with mock.patch("torch.cuda.CUDAGraph", return_value=graph):
            backend.capture_one(
                ShapeKey(size=64),
                lambda: None,
                capture_inputs=SimpleNamespace(global_num_tokens_gpu=counts),
            )
        del counts
        self.assertIsNotNone(reference())
        graph.reset.side_effect = lambda: self.assertIsNotNone(reference())
        backend.cleanup()
        self.assertIsNone(reference())

    def test_runs_two_warmups_and_capture_without_stepping(self):
        runner = _make_runner(enable_profile=False, profiler=None)
        backend = _make_backend(runner)

        sentinel_out = object()
        forward_fn = mock.Mock(return_value=sentinel_out)
        post_warmup_hook = mock.Mock()
        shape_key = ShapeKey(size=4)

        with mock.patch("torch.cuda.CUDAGraph", return_value="GRAPH"):
            backend.capture_one(
                shape_key, forward_fn, post_warmup_hook=post_warmup_hook
            )

        # 2 warmups + 1 capture.
        self.assertEqual(forward_fn.call_count, 3)
        # post_warmup_hook only runs in the two warmup iterations.
        self.assertEqual(post_warmup_hook.call_count, 2)
        # Graph + output are recorded against the shape key.
        self.assertEqual(backend._graphs[shape_key], "GRAPH")
        self.assertIs(backend._outputs[shape_key], sentinel_out)

    def test_prefill_shapes_share_one_output_storage(self):
        runner = _make_runner(enable_profile=False, profiler=None, mode_name="EXTEND")
        backend = _make_backend(runner)
        backend._reuse_output_buffer = True

        outputs = iter(
            [
                torch.ones((4, 2)),
                torch.ones((4, 2)),
                torch.ones((4, 2)),
                torch.ones((2, 2)),
                torch.ones((2, 2)),
                torch.ones((2, 2)),
            ]
        )
        with mock.patch("torch.cuda.CUDAGraph", side_effect=["GRAPH4", "GRAPH2"]):
            backend.capture_one(ShapeKey(size=4), lambda: next(outputs))
            backend.capture_one(ShapeKey(size=2), lambda: next(outputs))

        large = backend._outputs[ShapeKey(size=4)]
        small = backend._outputs[ShapeKey(size=2)]
        self.assertEqual(large.shape, (4, 2))
        self.assertEqual(small.shape, (2, 2))
        self.assertEqual(large.data_ptr(), small.data_ptr())
        self.assertEqual(backend._output_buffer.shape, (4, 2))

    def test_prefill_tuple_outputs_share_one_output_storage(self):
        # A model whose forward returns a tuple of row-major tensors shares one
        # buffer per leaf, and a later output that does not match the buffer's
        # structure keeps its own storage without disturbing the shared views.
        runner = _make_runner(enable_profile=False, profiler=None, mode_name="EXTEND")
        backend = _make_backend(runner)
        backend._reuse_output_buffer = True

        large_out = (torch.ones((4, 2)), torch.ones((4, 3)))
        small_out = (torch.full((2, 2), 2.0), torch.full((2, 3), 3.0))
        plain_out = torch.full((1, 2), 7.0)
        outputs = iter([large_out] * 3 + [small_out] * 3 + [plain_out] * 3)
        with mock.patch("torch.cuda.CUDAGraph", side_effect=["G4", "G2", "G1"]):
            backend.capture_one(ShapeKey(size=4), lambda: next(outputs))
            backend.capture_one(ShapeKey(size=2), lambda: next(outputs))
            backend.capture_one(ShapeKey(size=1), lambda: next(outputs))

        large = backend._outputs[ShapeKey(size=4)]
        small = backend._outputs[ShapeKey(size=2)]
        self.assertIsInstance(large, tuple)
        self.assertIsInstance(small, tuple)
        self.assertEqual([t.shape for t in large], [(4, 2), (4, 3)])
        self.assertEqual([t.shape for t in small], [(2, 2), (2, 3)])
        for large_leaf, small_leaf, buffer_leaf in zip(
            large, small, backend._output_buffer
        ):
            self.assertEqual(large_leaf.data_ptr(), small_leaf.data_ptr())
            self.assertEqual(large_leaf.data_ptr(), buffer_leaf.data_ptr())
        self.assertTrue(torch.equal(small[0], small_out[0]))
        self.assertTrue(torch.equal(small[1], small_out[1]))
        # The plain tensor does not fit a tuple buffer: it is kept as-is and
        # sharing is switched off for the rest of the capture.
        self.assertIs(backend._outputs[ShapeKey(size=1)], plain_out)
        self.assertFalse(backend._reuse_output_buffer)

    def test_prefill_tuple_with_scalar_leaf_keeps_independent_storage(self):
        # A zero-dim leaf has no leading row axis to share, so the whole tuple
        # keeps independent storage instead of allocating a partial buffer.
        runner = _make_runner(enable_profile=False, profiler=None, mode_name="EXTEND")
        backend = _make_backend(runner)
        backend._reuse_output_buffer = True

        first_out = (torch.ones((4, 2)), torch.tensor(0.0))
        second_out = (torch.ones((2, 2)), torch.tensor(1.0))
        outputs = iter([first_out] * 3 + [second_out] * 3)
        with mock.patch("torch.cuda.CUDAGraph", side_effect=["G4", "G2"]):
            backend.capture_one(ShapeKey(size=4), lambda: next(outputs))
            backend.capture_one(ShapeKey(size=2), lambda: next(outputs))

        self.assertIsNone(backend._output_buffer)
        self.assertFalse(backend._reuse_output_buffer)
        self.assertIs(backend._outputs[ShapeKey(size=4)], first_out)
        self.assertIs(backend._outputs[ShapeKey(size=2)], second_out)

    def test_enable_flag_set_but_no_profiler_attr_does_not_step(self):
        # The runner advertises the flag but never created a profiler; the
        # getattr guard must keep capture_one on the non-profiling path.
        runner = _make_runner(enable_profile=True, profiler=_UNSET)
        backend = _make_backend(runner)
        self.assertFalse(hasattr(runner, "_profiler"))

        forward_fn = mock.Mock(return_value=object())
        with mock.patch("torch.cuda.CUDAGraph", return_value="GRAPH"):
            backend.capture_one(ShapeKey(size=2), forward_fn)

        self.assertEqual(forward_fn.call_count, 3)


class TestCaptureOneWithProfiling(CustomTestCase):
    def _run(self, *, size, num_tokens_per_bs, mode_name):
        profiler = SimpleNamespace(step=mock.Mock(name="step"))
        runner = _make_runner(
            enable_profile=True,
            profiler=profiler,
            num_tokens_per_bs=num_tokens_per_bs,
            mode_name=mode_name,
        )
        backend = _make_backend(runner)

        forward_fn = mock.Mock(return_value=object())

        with mock.patch("torch.cuda.CUDAGraph", return_value="GRAPH"):
            backend.capture_one(ShapeKey(size=size), forward_fn)

        return profiler, forward_fn

    def test_steps_twice_in_warmup_and_once_after_capture(self):
        profiler, forward_fn = self._run(
            size=4, num_tokens_per_bs=1, mode_name="DECODE"
        )
        # Schedule wait=2 + active=1 => one step per warmup (x2) + one post-capture.
        self.assertEqual(profiler.step.call_count, 3)
        self.assertEqual(forward_fn.call_count, 3)


class TestCleanup(CustomTestCase):
    def test_resets_graphs_before_releasing_references(self):
        backend = _make_backend(_make_runner(enable_profile=False, profiler=None))
        graphs = [mock.Mock(), mock.Mock()]
        backend._graphs = {
            ShapeKey(size=i + 1): graph for i, graph in enumerate(graphs)
        }
        backend._outputs = {ShapeKey(size=1): object()}
        backend._pool = object()

        backend.cleanup()

        for graph in graphs:
            graph.reset.assert_called_once_with()
        self.assertEqual(backend._graphs, {})
        self.assertEqual(backend._outputs, {})
        self.assertIsNone(backend._pool)


class _FakeGraph:
    def __init__(self, *args, **kwargs):
        self.replay = mock.Mock()
        self.reset = mock.Mock()


@contextlib.contextmanager
def _backend_env(module, *, enabled, reuse=False):
    """Use upstream's CPU fake-device pattern with the real backend constructor."""
    state = SimpleNamespace(active=False, fail_exit=False, events=[])

    def record(event):
        if event in ("sync", "barrier") and state.active:
            raise AssertionError("synchronization during capture")
        state.events.append(event)

    @contextlib.contextmanager
    def capture(**kwargs):
        record("capture.enter")
        state.active = True
        try:
            yield
        finally:
            state.active = False
            record("capture.exit")
        if state.fail_exit:
            state.fail_exit = False
            raise RuntimeError("capture exit")

    @contextlib.contextmanager
    def measure():
        record("measure.enter")
        yield
        record("measure.exit")

    device = SimpleNamespace(
        synchronize=mock.Mock(side_effect=lambda: record("sync")),
        graph=capture,
        state=state,
    )
    runner = SimpleNamespace(
        device_module=device,
        enable_profile_cuda_graph=False,
        model_runner=SimpleNamespace(
            device="cuda",
            lora_manager=None,
            attn_backend=SimpleNamespace(requires_per_shape_decode_graph_warmup=False),
        ),
        cuda_graph_output_capacity_rows=lambda x: None,
        cuda_graph_output_rows=lambda x: None,
    )
    runner.attn_backend = runner.model_runner.attn_backend
    barrier = mock.Mock(side_effect=lambda: record("barrier"))
    carve = SimpleNamespace(
        measure=measure, mint=mock.Mock(side_effect=lambda: record("mint"))
    )
    patches = [
        mock.patch.object(
            module,
            "get_parallel",
            return_value=SimpleNamespace(tp_group=SimpleNamespace(barrier=barrier)),
        ),
        mock.patch.object(module, "GraphPoolPrecarve", return_value=carve),
        mock.patch.object(module.TorchMemorySaverAdapter, "create", return_value=None),
        mock.patch.object(
            module, "get_or_create_global_graph_memory_pool", return_value=object()
        ),
        mock.patch.object(module, "set_graph_pool_id"),
        mock.patch.object(module, "graph_pool_capture_scope", contextlib.nullcontext),
        mock.patch.object(module, "graph_pool_replay_scope", contextlib.nullcontext),
    ]
    if module is full:
        patches.append(mock.patch.object(torch.cuda, "CUDAGraph", _FakeGraph))
    else:
        patches.extend(
            [
                mock.patch.object(module, "BreakableCUDAGraph", _FakeGraph),
                mock.patch.object(module, "BreakableCUDAGraphCapture", capture),
                mock.patch.object(
                    module, "enable_breakable_cuda_graph", contextlib.nullcontext
                ),
                mock.patch.object(
                    module.BreakableCudaGraphBackend, "begin_cuda_graph_capture"
                ),
                mock.patch.object(
                    module.BreakableCudaGraphBackend, "end_cuda_graph_capture"
                ),
            ]
        )
    with contextlib.ExitStack() as stack:
        for patch in patches:
            stack.enter_context(patch)
        if module is full:
            backend = module.FullCudaGraphBackend(runner, reuse_output_buffer=reuse)
        else:
            backend = module.BreakableCudaGraphBackend(runner)
        backend._group_preparation_enabled = enabled
        yield backend, device, barrier


class TestGroupedCapture(CustomTestCase):
    def test_group_counts_sync_hook_and_formal_capture(self):
        keys = [
            ShapeKey(size=8),
            ShapeKey(size=4),
            ShapeKey(size=1),
            ShapeKey(size=2),
            ShapeKey(size=4, variant_label="lora"),
            ShapeKey(size=8, variant_label="lora"),
            ShapeKey(size=2, attention_variant="lora"),
        ]
        for module in (full, bcg):
            for enabled, eager in ((False, [2] * 7), (True, [1, 0, 1, 0, 1, 0, 1])):
                with self.subTest(backend=module.__name__, enabled=enabled):
                    with _backend_env(module, enabled=enabled) as (
                        backend,
                        device,
                        barrier,
                    ):
                        hook = mock.Mock(
                            side_effect=lambda: device.state.events.append("hook")
                        )
                        with backend.capture_session(object()):
                            for key, count in zip(keys, eager):

                                def forward():
                                    device.state.events.append(
                                        "capture" if device.state.active else "eager"
                                    )
                                    return torch.ones((key.size, 2))

                                device.state.events.clear()
                                before_sync = device.synchronize.call_count
                                before_barrier = barrier.call_count
                                before_hook = hook.call_count
                                backend.capture_one(key, forward, post_warmup_hook=hook)
                                expected = []
                                for step in range(2):
                                    expected.extend(
                                        ["sync", "barrier", "measure.enter"]
                                    )
                                    if step < count:
                                        expected.append("eager")
                                    expected.extend(["measure.exit", "hook"])
                                expected.extend(
                                    ["capture.enter", "mint", "capture", "capture.exit"]
                                )
                                self.assertEqual(device.state.events, expected)
                                self.assertEqual(
                                    device.synchronize.call_count - before_sync, 2
                                )
                                self.assertEqual(barrier.call_count - before_barrier, 2)
                                self.assertEqual(hook.call_count - before_hook, 2)
                                self.assertTrue(backend.can_run(None, key))
                                self.assertEqual(
                                    tuple(backend.replay(key, None).shape),
                                    (key.size, 2),
                                )

    def test_session_reset_and_failure_retry(self):
        for module in (full, bcg):
            for fail_call in (1, 2, "exit"):
                with self.subTest(backend=module.__name__, fail_call=fail_call):
                    with _backend_env(module, enabled=True) as (backend, device, _):
                        key = ShapeKey(size=4)
                        calls = 0

                        def fail():
                            nonlocal calls
                            calls += 1
                            if calls == fail_call:
                                raise RuntimeError("injected")
                            return torch.ones((4, 2))

                        with backend.capture_session(object()):
                            device.state.fail_exit = fail_call == "exit"
                            with self.assertRaisesRegex(
                                RuntimeError, "injected|capture exit"
                            ):
                                backend.capture_one(key, fail)
                            self.assertFalse(backend.can_run(None, key))
                            forward = mock.Mock(return_value=torch.ones((4, 2)))
                            backend.capture_one(key, forward)
                            self.assertEqual(forward.call_count, 2)
                            with self.assertRaisesRegex(RuntimeError, "ready"):
                                backend.capture_one(
                                    key, mock.Mock(side_effect=RuntimeError("ready"))
                                )
                            forward = mock.Mock(return_value=torch.ones((4, 2)))
                            backend.capture_one(key, forward)
                            self.assertEqual(forward.call_count, 1)
                        with backend.capture_session(object()):
                            forward = mock.Mock(return_value=torch.ones((4, 2)))
                            backend.capture_one(key, forward)
                            self.assertEqual(forward.call_count, 2)
                        # A separate backend must prepare the same group independently.
                        with _backend_env(module, enabled=True) as (other, _, _):
                            with other.capture_session(object()):
                                forward = mock.Mock(return_value=torch.ones((4, 2)))
                                other.capture_one(key, forward)
                                self.assertEqual(forward.call_count, 2)

    def test_warmup_outputs_released_before_capture(self):
        for enabled in (False, True):
            for reuse in (False, True):
                with self.subTest(enabled=enabled, reuse=reuse):
                    with _backend_env(full, enabled=enabled, reuse=reuse) as (
                        backend,
                        device,
                        _,
                    ):
                        warmups = []

                        def forward():
                            self.assertTrue(all(ref() is None for ref in warmups))
                            output = torch.ones((4, 2))
                            if not device.state.active:
                                warmups.append(weakref.ref(output))
                            return output

                        with backend.capture_session(object()):
                            backend.capture_one(ShapeKey(size=4), forward)
                        self.assertEqual(len(warmups), 1 if enabled else 2)
                        self.assertTrue(all(ref() is None for ref in warmups))

    def test_resolver_policy_controls_real_capture(self):
        for module, phase, kind in (
            (full, "decode", Backend.FULL),
            (bcg, "prefill", Backend.BREAKABLE),
        ):
            for compiled, precarve, attention_warmup in (
                (False, False, False),
                (True, False, False),
                (False, True, False),
                (False, False, True),
            ):
                with self.subTest(phase=phase, compiled=compiled, precarve=precarve):
                    with _backend_env(module, enabled=False) as (backend, device, _):
                        backend._cuda_graph_runner.model_runner.attn_backend.requires_per_shape_decode_graph_warmup = attention_warmup
                        fallback = (
                            compiled
                            or precarve
                            or (attention_warmup and phase == "decode")
                        )
                        config = SimpleNamespace(
                            graph=SimpleNamespace(
                                cuda_graph_config=SimpleNamespace(
                                    decode=SimpleNamespace(backend=kind),
                                    prefill=SimpleNamespace(backend=kind),
                                ),
                                debug_cuda_graph=False,
                                enable_profile_cuda_graph=False,
                            ),
                            features=SimpleNamespace(enable_memory_saver=False),
                        )
                        factory = mock.Mock(return_value=backend)
                        with contextlib.ExitStack() as stack:
                            stack.enter_context(
                                mock.patch.object(
                                    backend_utils,
                                    "get_memory",
                                    return_value=SimpleNamespace(
                                        enable_hierarchical_cache=False
                                    ),
                                )
                            )
                            stack.enter_context(
                                mock.patch.object(
                                    backend_utils, "get_exec", return_value=config
                                )
                            )
                            stack.enter_context(
                                mock.patch.object(
                                    backend_utils,
                                    "get_flags",
                                    return_value=SimpleNamespace(
                                        capture=SimpleNamespace(
                                            enable_torch_compile=compiled
                                        )
                                    ),
                                )
                            )
                            stack.enter_context(
                                mock.patch.object(
                                    backend_utils.envs.SGLANG_ENABLE_GRAPH_POOL_PRECARVE,
                                    "get",
                                    return_value=precarve,
                                )
                            )
                            stack.enter_context(
                                mock.patch.object(
                                    backend_utils.envs.SGLANG_ENABLE_GRAPH_POOL_BORROW,
                                    "get",
                                    return_value=False,
                                )
                            )
                            stack.enter_context(
                                mock.patch.object(
                                    backend_utils,
                                    "current_platform",
                                    SimpleNamespace(
                                        is_cuda=lambda: True,
                                        is_npu=lambda: False,
                                        is_xpu=lambda: False,
                                        is_out_of_tree=lambda: False,
                                        get_full_graph_backend_cls=lambda: None,
                                    ),
                                )
                            )
                            stack.enter_context(
                                mock.patch.object(
                                    backend_utils, "FullCudaGraphBackend", factory
                                )
                            )
                            stack.enter_context(
                                mock.patch.object(
                                    backend_utils, "BreakableCudaGraphBackend", factory
                                )
                            )
                            resolved = getattr(
                                backend_utils, f"resolve_{phase}_backend"
                            )(backend._cuda_graph_runner)
                        self.assertIs(resolved, backend)
                        with backend.capture_session(object()):
                            for size, expected_eager in (
                                (8, 2 if fallback else 1),
                                (4, 2 if fallback else 0),
                            ):
                                phases = []

                                def forward():
                                    phases.append(device.state.active)
                                    if (compiled and len(phases) == 2) or (
                                        attention_warmup
                                        and phase == "decode"
                                        and len(phases) == 1
                                    ):
                                        device.synchronize()
                                    return torch.ones((size, 2))

                                backend.capture_one(ShapeKey(size=size), forward)
                                self.assertEqual(
                                    phases, [False] * expected_eager + [True]
                                )

    def test_output_buffer_reuse_and_cleanup(self):
        for module in (full, bcg):
            with self.subTest(backend=module.__name__):
                with _backend_env(module, enabled=True, reuse=True) as (backend, _, _):
                    large = ShapeKey(size=8)
                    small = ShapeKey(size=4)
                    input_owner = torch.ones((2,))
                    retained = weakref.ref(input_owner)
                    with backend.capture_session(object()):
                        backend.capture_one(
                            large,
                            lambda: (torch.ones((8, 2)), torch.ones((8, 3))),
                            capture_inputs=input_owner,
                        )
                        backend.capture_one(
                            small,
                            lambda: (torch.full((4, 2), 2.0), torch.full((4, 3), 3.0)),
                        )
                    del input_owner
                    self.assertIsNotNone(retained())
                    first = backend.replay(large, None)
                    second = backend.replay(small, None)
                    for index, (large_leaf, small_leaf) in enumerate(
                        zip(first, second)
                    ):
                        self.assertEqual(large_leaf.data_ptr(), small_leaf.data_ptr())
                        self.assertEqual(tuple(small_leaf.shape), (4, index + 2))
                        self.assertTrue(
                            torch.equal(
                                small_leaf, torch.full((4, index + 2), index + 2.0)
                            )
                        )
                    backend.cleanup()
                    self.assertFalse(backend.can_run(None, large))
                    self.assertFalse(backend.can_run(None, small))
                    self.assertIsNone(retained())


if __name__ == "__main__":
    unittest.main()
