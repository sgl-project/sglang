"""CPU tests for symmetric-memory routing and post-registration rank joins."""

import contextlib
import unittest
from itertools import product
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_backend import base_cuda_graph_backend as base
from sglang.srt.model_executor.runner_backend import (
    breakable_cuda_graph_backend as breakable,
)
from sglang.srt.model_executor.runner_backend import full_cuda_graph_backend as full
from sglang.srt.model_executor.runner_backend import (
    tc_piecewise_cuda_graph_backend as piecewise,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestSymmetricMemoryGraphCapture(CustomTestCase):
    def test_routing_is_independent_of_speculative_priming(self):
        for cuda, enabled, world_size, speculative in product(
            (False, True), (False, True), (1, 2), (False, True)
        ):
            with self.subTest(
                cuda=cuda, enabled=enabled, tp=world_size, speculative=speculative
            ):
                runner = SimpleNamespace(
                    model_runner=SimpleNamespace(
                        tp_group=SimpleNamespace(world_size=world_size),
                        spec_algorithm=SimpleNamespace(
                            is_speculative=lambda: speculative
                        ),
                    )
                )
                with (
                    mock.patch.object(base, "is_cuda", return_value=cuda),
                    mock.patch.object(
                        base, "is_symmetric_memory_enabled", return_value=enabled
                    ),
                ):
                    routing = cuda and enabled and world_size > 1
                    self.assertEqual(
                        base.should_use_dedicated_symmetric_memory_graph_pool(runner),
                        routing,
                    )
                    self.assertEqual(
                        base.should_prime_symmetric_memory_graph(runner),
                        routing and speculative,
                    )

    def test_registration_joins_ranks_before_retained_capture(self):
        for module, backend_class in (
            (full, full.FullCudaGraphBackend),
            (breakable, breakable.BreakableCudaGraphBackend),
        ):
            for prime, with_hook in product((False, True), repeat=2):
                with self.subTest(
                    backend=backend_class.__name__, prime=prime, hook=with_hook
                ):
                    events = []

                    @contextlib.contextmanager
                    def defer(group, *, enabled):
                        self.assertEqual(enabled, prime)
                        yield enabled
                        if enabled:
                            events.append("register")

                    def forward():
                        events.append("forward")
                        return torch.ones((1, 2))

                    backend = SimpleNamespace(
                        _device_module=SimpleNamespace(
                            synchronize=lambda: events.append("sync"),
                            graph=lambda **kwargs: contextlib.nullcontext(),
                        ),
                        _tp_group=SimpleNamespace(
                            barrier=lambda: events.append("barrier")
                        ),
                        _cuda_graph_runner=SimpleNamespace(
                            enable_profile_cuda_graph=False
                        ),
                        _precarve=SimpleNamespace(
                            measure=contextlib.nullcontext, mint=lambda: None
                        ),
                        _prime_symmetric_memory_graph=prime,
                        _use_symmetric_memory_graph_pool=True,
                        _memory_saver_adapter=None,
                        _reuse_output_buffer=False,
                        _output_buffer=None,
                        _pool=None,
                        _capture_stream=None,
                        _graphs={},
                        _outputs={},
                        _capture_inputs={},
                        _debug_eager=False,
                        deduped_cuda_graph=None,
                        _shared_output_buffer=torch.empty((1, 2)),
                        _output_rows=lambda out, size: size,
                        _copy_output_to_buffer=lambda out, buf, rows: buf.copy_(out),
                        _slice_output=lambda buf, rows: buf[:rows],
                    )
                    with contextlib.ExitStack() as stack:
                        stack.enter_context(
                            mock.patch.object(
                                module,
                                "defer_symmetric_memory_graph_registration",
                                defer,
                            )
                        )
                        stack.enter_context(
                            mock.patch.object(
                                module,
                                "graph_pool_capture_scope",
                                contextlib.nullcontext,
                            )
                        )
                        stack.enter_context(
                            mock.patch("torch.cuda.CUDAGraph", return_value=object())
                        )
                        if module is breakable:
                            stack.enter_context(
                                mock.patch.object(
                                    module, "BreakableCUDAGraph", return_value=object()
                                )
                            )
                            stack.enter_context(
                                mock.patch.object(
                                    module,
                                    "BreakableCUDAGraphCapture",
                                    side_effect=lambda **kwargs: (
                                        contextlib.nullcontext()
                                    ),
                                )
                            )
                        backend_class.capture_one(
                            backend,
                            ShapeKey(size=1),
                            forward,
                            post_warmup_hook=(lambda: events.append("hook"))
                            if with_hook
                            else None,
                        )
                    self.assertEqual(events.count("forward"), 4 if prime else 3)
                    if prime:
                        expected = (["hook"] if with_hook else []) + [
                            "sync",
                            "barrier",
                            "forward",
                        ]
                        self.assertEqual(
                            events[events.index("register") + 1 :], expected
                        )
                    else:
                        self.assertNotIn("register", events)

    def test_capture_sessions_route_and_reset_on_error(self):
        for module, backend_class in (
            (full, full.FullCudaGraphBackend),
            (breakable, breakable.BreakableCudaGraphBackend),
            (piecewise, piecewise.TcPiecewiseCudaGraphBackend),
        ):
            with self.subTest(backend=backend_class.__name__):
                backend = SimpleNamespace(
                    _pool="POOL",
                    _capture_stream=None,
                    _use_symmetric_memory_graph_pool=True,
                    replay_session=contextlib.nullcontext,
                    begin_cuda_graph_capture=mock.Mock(),
                    end_cuda_graph_capture=mock.Mock(),
                )
                with contextlib.ExitStack() as stack:
                    routing = stack.enter_context(
                        mock.patch.object(
                            module, "set_use_dedicated_symmetric_memory_graph_pool"
                        )
                    )
                    pool = stack.enter_context(
                        mock.patch.object(module, "set_graph_pool_id")
                    )
                    if module is breakable:
                        stack.enter_context(
                            mock.patch.object(
                                module,
                                "enable_breakable_cuda_graph",
                                contextlib.nullcontext,
                            )
                        )
                    if module is piecewise:
                        stack.enter_context(
                            mock.patch.object(
                                module,
                                "set_pcg_capture_stream",
                                side_effect=lambda stream: contextlib.nullcontext(),
                            )
                        )
                    with self.assertRaisesRegex(RuntimeError, "capture failed"):
                        with backend_class.capture_session(backend, "STREAM"):
                            routing.assert_called_once_with(True)
                            pool.assert_called_once_with("POOL")
                            raise RuntimeError("capture failed")
                    self.assertEqual(
                        routing.call_args_list, [mock.call(True), mock.call(False)]
                    )
                    self.assertEqual(
                        pool.call_args_list, [mock.call("POOL"), mock.call(None)]
                    )
                    self.assertIsNone(backend._capture_stream)


if __name__ == "__main__":
    unittest.main()
