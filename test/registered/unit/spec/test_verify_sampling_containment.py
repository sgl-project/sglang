"""Containment behavior for speculative verify sampling-op failures (#43412).

A flashinfer sampling-op JIT failure used to escape the scheduler event loop
and SIGKILL the whole server on the first non-greedy request. These tests pin
the new contract:

- the renorm dispatch degrades (pytorch backend / fallback flag) instead of
  hardwiring flashinfer on CUDA;
- a renorm failure marks the backend failed (no per-request compile retry)
  and either degrades or raises SpecSamplingOpError;
- the scheduler converts SpecSamplingOpError into per-request aborts and
  keeps serving;
- the startup preflight fails fast without the fallback flag and degrades
  with it.
"""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative import eagle_utils as eu
from sglang.srt.speculative.eagle_utils import (
    SpecSamplingOpError,
    _renorm_verify_probs,
    preflight_verify_sampling_ops,
    resolve_verify_renorm_impls,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _fake_exec(backend="flashinfer"):
    return SimpleNamespace(
        kernel=SimpleNamespace(sampling_backend=backend),
    )


def _fake_spec(fallback=False):
    return SimpleNamespace(speculative_sampling_renorm_fallback=fallback)


def _fake_flashinfer_modules():
    """flashinfer.sampling with identifiable renorm functions."""
    sampling = types.ModuleType("flashinfer.sampling")

    def top_k_renorm_probs(probs, top_ks):
        return probs

    def top_p_renorm_probs(probs, top_ps):
        return probs

    sampling.top_k_renorm_probs = top_k_renorm_probs
    sampling.top_p_renorm_probs = top_p_renorm_probs
    package = types.ModuleType("flashinfer")
    package.__path__ = []
    return {"flashinfer": package, "flashinfer.sampling": sampling}


def _sampling_info(need_k=True, need_p=False):
    return SimpleNamespace(
        need_top_k_sampling=need_k,
        need_top_p_sampling=need_p,
        top_ks=torch.full((2, 1), 4, dtype=torch.int32),
        top_ps=torch.full((2, 1), 0.9, dtype=torch.float32),
    )


def _verify_input():
    return SimpleNamespace(draft_token_num=2)


class ResolveVerifyRenormImplsTest(CustomTestCase):
    def setUp(self):
        eu._verify_sampling_ops_failed = False

    def tearDown(self):
        eu._verify_sampling_ops_failed = False

    def test_hip_uses_triton_renorm(self):
        sentinel_k, sentinel_p = object(), object()
        with (
            patch.object(eu, "_is_hip", True),
            patch.object(eu, "_is_cuda", False),
            patch.object(
                eu, "_triton_renorm_impls", return_value=(sentinel_k, sentinel_p)
            ),
        ):
            self.assertIs(resolve_verify_renorm_impls()[0], sentinel_k)
            self.assertIs(resolve_verify_renorm_impls()[1], sentinel_p)

    def test_cuda_pytorch_backend_degrades_to_triton(self):
        sentinel_k, sentinel_p = object(), object()
        with (
            patch.object(eu, "_is_hip", False),
            patch.object(eu, "_is_cuda", True),
            patch.object(eu, "get_exec", return_value=_fake_exec("pytorch")),
            patch.object(
                eu, "_triton_renorm_impls", return_value=(sentinel_k, sentinel_p)
            ),
        ):
            self.assertEqual(resolve_verify_renorm_impls(), (sentinel_k, sentinel_p))

    def test_cuda_default_resolves_flashinfer(self):
        modules = _fake_flashinfer_modules()
        with (
            patch.object(eu, "_is_hip", False),
            patch.object(eu, "_is_cuda", True),
            patch.object(eu, "get_exec", return_value=_fake_exec("flashinfer")),
            patch.dict(sys.modules, modules),
        ):
            k_fn, p_fn = resolve_verify_renorm_impls()
        self.assertIs(k_fn, modules["flashinfer.sampling"].top_k_renorm_probs)
        self.assertIs(p_fn, modules["flashinfer.sampling"].top_p_renorm_probs)

    def test_cuda_failed_with_fallback_flag_degrades(self):
        sentinel_k, sentinel_p = object(), object()
        eu._verify_sampling_ops_failed = True
        with (
            patch.object(eu, "_is_hip", False),
            patch.object(eu, "_is_cuda", True),
            patch.object(eu, "get_exec", return_value=_fake_exec("flashinfer")),
            patch.object(eu, "get_spec", return_value=_fake_spec(fallback=True)),
            patch.object(
                eu, "_triton_renorm_impls", return_value=(sentinel_k, sentinel_p)
            ),
        ):
            self.assertEqual(resolve_verify_renorm_impls(), (sentinel_k, sentinel_p))

    def test_cuda_failed_without_fallback_flag_still_resolves_flashinfer(self):
        # The fail-fast lives in _renorm_verify_probs (gated on renorm being
        # needed); the resolver keeps resolving so renorm-free batches stay
        # servable after a failure.
        eu._verify_sampling_ops_failed = True
        modules = _fake_flashinfer_modules()
        with (
            patch.object(eu, "_is_hip", False),
            patch.object(eu, "_is_cuda", True),
            patch.object(eu, "get_exec", return_value=_fake_exec("flashinfer")),
            patch.object(eu, "get_spec", return_value=_fake_spec(fallback=False)),
            patch.dict(sys.modules, modules),
        ):
            k_fn, p_fn = resolve_verify_renorm_impls()
        self.assertIs(k_fn, modules["flashinfer.sampling"].top_k_renorm_probs)
        self.assertIs(p_fn, modules["flashinfer.sampling"].top_p_renorm_probs)

    def test_sgl_kernel_fallthrough_for_other_devices(self):
        # CPU/XPU/MUSA keep the sgl_kernel kernels (NPU imports its own
        # renorm in the caller and never reaches the resolver).
        sgl_kernel = types.ModuleType("sgl_kernel")
        sgl_kernel.top_k_renorm_prob = object()
        sgl_kernel.top_p_renorm_prob = object()
        with (
            patch.object(eu, "_is_hip", False),
            patch.object(eu, "_is_cuda", False),
            patch.dict(sys.modules, {"sgl_kernel": sgl_kernel}),
        ):
            k_fn, p_fn = resolve_verify_renorm_impls()
        self.assertIs(k_fn, sgl_kernel.top_k_renorm_prob)
        self.assertIs(p_fn, sgl_kernel.top_p_renorm_prob)


class RenormVerifyProbsTest(CustomTestCase):
    def setUp(self):
        eu._verify_sampling_ops_failed = False

    def tearDown(self):
        eu._verify_sampling_ops_failed = False

    def test_applies_requested_renorms(self):
        probs = torch.ones(4, 8)

        def k_fn(probs, top_ks):
            return probs * 2.0

        def p_fn(probs, top_ps):
            return probs * 3.0

        out = _renorm_verify_probs(
            probs,
            _sampling_info(need_k=True, need_p=True),
            _verify_input(),
            k_fn,
            p_fn,
        )
        torch.testing.assert_close(out, probs * 6.0)

    def test_skips_unrequested_renorms(self):
        probs = torch.ones(4, 8)
        calls = []

        def k_fn(probs, top_ks):
            calls.append("k")
            return probs

        def p_fn(probs, top_ps):
            calls.append("p")
            return probs

        _renorm_verify_probs(
            probs,
            _sampling_info(need_k=False, need_p=False),
            _verify_input(),
            k_fn,
            p_fn,
        )
        self.assertEqual(calls, [])

    def test_failure_without_fallback_raises_and_marks_failed(self):
        def broken_fn(probs, _):
            raise RuntimeError("Ninja build failed. Ninja output: ...")

        with patch.object(eu, "get_spec", return_value=_fake_spec(fallback=False)):
            with self.assertRaises(SpecSamplingOpError) as ctx:
                _renorm_verify_probs(
                    torch.ones(4, 8),
                    _sampling_info(need_k=True),
                    _verify_input(),
                    broken_fn,
                    broken_fn,
                )
        self.assertTrue(eu._verify_sampling_ops_failed)
        self.assertIn("--sampling-backend pytorch", str(ctx.exception))
        self.assertIn("Ninja build failed", str(ctx.exception.__cause__))

    def test_failure_with_fallback_retries_on_triton(self):
        def broken_fn(probs, _):
            raise RuntimeError("Ninja build failed.")

        def triton_k(probs, top_ks):
            return probs * 5.0

        def triton_p(probs, top_ps):
            return probs * 7.0

        # Renorm input must be retried from the pre-renorm probs, not the
        # partially-renormed intermediate.
        info = _sampling_info(need_k=True, need_p=True)
        probs = torch.ones(4, 8)
        with (
            patch.object(eu, "get_spec", return_value=_fake_spec(fallback=True)),
            patch.object(eu, "_triton_renorm_impls", return_value=(triton_k, triton_p)),
        ):
            out = _renorm_verify_probs(
                probs, info, _verify_input(), broken_fn, broken_fn
            )
        self.assertTrue(eu._verify_sampling_ops_failed)
        torch.testing.assert_close(out, probs * 35.0)

    def test_fail_fast_after_failure_only_hits_renorm_batches(self):
        # After one failure (no fallback) a batch that needs renorm fails
        # fast without touching flashinfer again; a temperature-only batch
        # that needs no renorm stays servable.
        def broken_fn(probs, _):
            raise RuntimeError("Ninja build failed.")

        calls = []

        def tracking_fn(probs, _):
            calls.append("called")
            return probs

        with patch.object(eu, "get_spec", return_value=_fake_spec(fallback=False)):
            with self.assertRaises(SpecSamplingOpError):
                _renorm_verify_probs(  # first failure marks the backend
                    torch.ones(4, 8),
                    _sampling_info(need_k=True),
                    _verify_input(),
                    broken_fn,
                    broken_fn,
                )
            with self.assertRaises(SpecSamplingOpError) as ctx:
                _renorm_verify_probs(  # renorm needed -> fail fast
                    torch.ones(4, 8),
                    _sampling_info(need_k=True),
                    _verify_input(),
                    tracking_fn,
                    tracking_fn,
                )
            self.assertIn("previously failed", str(ctx.exception))
            self.assertEqual(calls, [])  # flashinfer not retried
            probs = torch.ones(4, 8)  # no renorm needed -> served untouched
            out = _renorm_verify_probs(
                probs,
                _sampling_info(need_k=False, need_p=False),
                _verify_input(),
                tracking_fn,
                tracking_fn,
            )
            torch.testing.assert_close(out, probs)
            self.assertEqual(calls, [])

    def test_oom_is_not_swallowed_as_backend_failure(self):
        # torch.cuda.OutOfMemoryError is a RuntimeError subclass: it must
        # propagate (pre-existing crash semantics) without poisoning the
        # backend flag — a retry after the OOM passes would succeed.
        def oom_fn(probs, _):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

        with patch.object(eu, "get_spec", return_value=_fake_spec(fallback=False)):
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                _renorm_verify_probs(
                    torch.ones(4, 8),
                    _sampling_info(need_k=True),
                    _verify_input(),
                    oom_fn,
                    oom_fn,
                )
        self.assertFalse(eu._verify_sampling_ops_failed)

    def test_fallback_retry_failure_is_also_contained(self):
        # A triton renorm failure during the degraded retry must not escape
        # containment either.
        def broken_fn(probs, _):
            raise RuntimeError("Ninja build failed.")

        def also_broken(probs, _):
            raise RuntimeError("triton compile failed")

        with (
            patch.object(eu, "get_spec", return_value=_fake_spec(fallback=True)),
            patch.object(
                eu, "_triton_renorm_impls", return_value=(also_broken, also_broken)
            ),
        ):
            with self.assertRaises(SpecSamplingOpError) as ctx:
                _renorm_verify_probs(
                    torch.ones(4, 8),
                    _sampling_info(need_k=True),
                    _verify_input(),
                    broken_fn,
                    broken_fn,
                )
        self.assertIn("both", str(ctx.exception))

    def test_fallback_retry_with_only_top_p(self):
        def broken_fn(probs, _):
            raise RuntimeError("Ninja build failed.")

        k_calls = []

        def triton_k(probs, top_ks):
            k_calls.append("k")
            return probs

        def triton_p(probs, top_ps):
            return probs * 3.0

        with (
            patch.object(eu, "get_spec", return_value=_fake_spec(fallback=True)),
            patch.object(eu, "_triton_renorm_impls", return_value=(triton_k, triton_p)),
        ):
            out = _renorm_verify_probs(
                torch.ones(4, 8),
                _sampling_info(need_k=False, need_p=True),
                _verify_input(),
                broken_fn,
                broken_fn,
            )
        self.assertEqual(k_calls, [])
        torch.testing.assert_close(out, torch.ones(4, 8) * 3.0)


class PreflightVerifySamplingOpsTest(CustomTestCase):
    def setUp(self):
        eu._verify_sampling_ops_failed = False

    def tearDown(self):
        eu._verify_sampling_ops_failed = False

    def test_noop_off_cuda(self):
        with patch.object(eu, "_is_cuda", False):
            preflight_verify_sampling_ops()  # must not touch anything

    def test_pytorch_backend_warms_triton_only(self):
        warmed = []

        def fake_warm():
            warmed.append(True)

        with (
            patch.object(eu, "_is_cuda", True),
            patch.object(eu, "get_exec", return_value=_fake_exec("pytorch")),
            patch.object(eu, "warm_verify_renorm_ops", side_effect=fake_warm),
        ):
            preflight_verify_sampling_ops()
        self.assertEqual(warmed, [True])

    def test_flashinfer_failure_without_fallback_fails_startup(self):
        import sglang.srt.layers.flashinfer_jit_preflight as fjp

        with (
            patch.object(eu, "_is_cuda", True),
            patch.object(eu, "get_exec", return_value=_fake_exec("flashinfer")),
            patch.object(eu, "get_spec", return_value=_fake_spec(fallback=False)),
            patch.object(
                fjp,
                "preflight_flashinfer_sampling_ops",
                return_value=(False, "Ninja build failed. ..."),
            ),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                preflight_verify_sampling_ops()
        self.assertIn("refuses to start", str(ctx.exception))
        self.assertFalse(eu._verify_sampling_ops_failed)

    def test_flashinfer_failure_with_fallback_degrades(self):
        import sglang.srt.layers.flashinfer_jit_preflight as fjp

        warmed = []

        def fake_warm():
            warmed.append(True)

        with (
            patch.object(eu, "_is_cuda", True),
            patch.object(eu, "get_exec", return_value=_fake_exec("flashinfer")),
            patch.object(eu, "get_spec", return_value=_fake_spec(fallback=True)),
            patch.object(
                fjp,
                "preflight_flashinfer_sampling_ops",
                return_value=(False, "Ninja build failed. ..."),
            ),
            patch.object(eu, "warm_verify_renorm_ops", side_effect=fake_warm),
        ):
            preflight_verify_sampling_ops()
        self.assertTrue(eu._verify_sampling_ops_failed)
        self.assertEqual(warmed, [True])

    def test_flashinfer_success_is_silent(self):
        import sglang.srt.layers.flashinfer_jit_preflight as fjp

        with (
            patch.object(eu, "_is_cuda", True),
            patch.object(eu, "get_exec", return_value=_fake_exec("flashinfer")),
            patch.object(
                fjp,
                "preflight_flashinfer_sampling_ops",
                return_value=(True, "warmed"),
            ),
        ):
            preflight_verify_sampling_ops()
        self.assertFalse(eu._verify_sampling_ops_failed)


class _FakeReq:
    def __init__(self, rid):
        self.rid = rid
        self.to_finish = None
        self.weight_version_events = []
        self.output_ids = []


class _FakeBatch:
    def __init__(self, reqs):
        self.reqs = list(reqs)
        self.released = []
        self.filtered_with = None

    def release_req(self, idx, remaining, offload_kv=True):
        self.released.append((idx, remaining, offload_kv))

    def filter_batch(self, keep_indices=None, **kwargs):
        self.filtered_with = keep_indices


class _FakeIpc:
    def __init__(self):
        self.sent = []

    class _Sender:
        def __init__(self, outer):
            self._outer = outer

        def send_output(self, abort_req, req):
            self._outer.sent.append((abort_req, req))

    @property
    def send_to_tokenizer(self):
        return _FakeIpc._Sender(self)


class SchedulerAbortBatchTest(CustomTestCase):
    """_abort_batch_on_spec_sampling_op_failure keeps the server serving."""

    def _make_scheduler(self):
        from sglang.srt.managers.scheduler import Scheduler

        scheduler = object.__new__(Scheduler)
        scheduler.ipc_channels = _FakeIpc()
        # _make_abort_req reads the serving namespace for weight versions;
        # stub it — these tests exercise the abort plumbing, not versions.
        self._serving_patch = patch.object(
            __import__("sglang.srt.managers.scheduler", fromlist=["get_serving"]),
            "get_serving",
            return_value=SimpleNamespace(weight_version="test"),
        )
        self._serving_patch.start()
        self.addCleanup(self._serving_patch.stop)
        return scheduler

    def test_aborts_all_reqs_and_keeps_serving(self):
        scheduler = self._make_scheduler()
        reqs = [_FakeReq("r1"), _FakeReq("r2")]
        batch = _FakeBatch(reqs)

        scheduler._abort_batch_on_spec_sampling_op_failure(
            batch, "flashinfer JIT failed"
        )

        # Every req carries a 500 abort with the original message.
        for req in reqs:
            self.assertIsNotNone(req.to_finish)
            self.assertEqual(req.to_finish.status_code, 500)
            self.assertIn("flashinfer JIT failed", req.to_finish.message)
        # Resources released (no KV offload for aborts) and batch emptied.
        self.assertEqual(len(batch.released), 2)
        for _, _, offload_kv in batch.released:
            self.assertFalse(offload_kv)
        self.assertEqual(batch.filtered_with, [])
        # Every aborted req is reported to the tokenizer with its reason.
        self.assertEqual(len(scheduler.ipc_channels.sent), 2)
        for abort_req, req in scheduler.ipc_channels.sent:
            self.assertEqual(req.rid, abort_req.rid)
            self.assertEqual(abort_req.finished_reason["type"], "abort")
            self.assertEqual(abort_req.finished_reason["status_code"], 500)
            self.assertIn("flashinfer JIT failed", abort_req.finished_reason["message"])

    def test_existing_to_finish_is_preserved_not_overwritten(self):
        # A req already aborting (client abort, abort method 3) keeps its
        # own reason; it is still released and reported so the empty filter
        # cannot orphan it.
        from sglang.srt.managers.schedule_batch import FINISH_ABORT

        scheduler = self._make_scheduler()
        preset = FINISH_ABORT("client aborted")
        req = _FakeReq("r1")
        req.to_finish = preset
        batch = _FakeBatch([req])

        scheduler._abort_batch_on_spec_sampling_op_failure(
            batch, "flashinfer JIT failed"
        )

        self.assertIs(req.to_finish, preset)
        self.assertEqual(len(batch.released), 1)
        self.assertEqual(len(scheduler.ipc_channels.sent), 1)
        abort_req, _ = scheduler.ipc_channels.sent[0]
        self.assertEqual(abort_req.finished_reason["message"], "client aborted")

    def test_empty_batch_is_a_noop(self):
        scheduler = self._make_scheduler()
        batch = _FakeBatch([])
        scheduler._abort_batch_on_spec_sampling_op_failure(batch, "x")
        self.assertEqual(batch.filtered_with, [])
        self.assertEqual(scheduler.ipc_channels.sent, [])


class BaseSpecWorkerPrewarmTest(CustomTestCase):
    """The scheduler's prewarm hook triggers the verify-op preflight only for
    workers whose verify path runs eagle_sample (EAGLE/NGRAM)."""

    def _make_worker(self, uses_eagle_sample):
        from sglang.srt.speculative.base_spec_worker import BaseSpecWorker

        worker = object.__new__(BaseSpecWorker)
        worker.verify_uses_eagle_sample = uses_eagle_sample
        # target_worker is a read-only property over _target_worker.
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(prewarm_sampling=lambda: "stub-result")
        )
        return worker

    def test_eagle_sample_workers_preflight(self):
        import sglang.srt.speculative.eagle_utils as eu

        worker = self._make_worker(uses_eagle_sample=True)
        with patch.object(eu, "preflight_verify_sampling_ops") as preflight:
            result = worker.prewarm_sampling()
        preflight.assert_called_once()
        self.assertEqual(result, "stub-result")

    def test_other_workers_skip_preflight(self):
        import sglang.srt.speculative.eagle_utils as eu

        worker = self._make_worker(uses_eagle_sample=False)
        with patch.object(eu, "preflight_verify_sampling_ops") as preflight:
            result = worker.prewarm_sampling()
        preflight.assert_not_called()
        self.assertEqual(result, "stub-result")

    def test_preflight_runs_before_target_prewarm(self):
        import sglang.srt.speculative.eagle_utils as eu

        calls = []
        worker = self._make_worker(uses_eagle_sample=True)
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                prewarm_sampling=lambda: calls.append("target") or "stub-result"
            )
        )
        with patch.object(
            eu,
            "preflight_verify_sampling_ops",
            side_effect=lambda: calls.append("preflight"),
        ):
            worker.prewarm_sampling()
        self.assertEqual(calls, ["preflight", "target"])

    def test_warm_uses_the_resolved_impls_with_dummy_tensors(self):
        import sglang.srt.speculative.eagle_utils as eu

        seen = []

        def k_fn(probs, top_ks):
            seen.append(
                (
                    "k",
                    tuple(probs.shape),
                    probs.dtype,
                    tuple(top_ks.shape),
                    top_ks.dtype,
                )
            )
            return probs

        def p_fn(probs, top_ps):
            seen.append(
                (
                    "p",
                    tuple(probs.shape),
                    probs.dtype,
                    tuple(top_ps.shape),
                    top_ps.dtype,
                )
            )
            return probs

        with patch.object(eu, "resolve_verify_renorm_impls", return_value=(k_fn, p_fn)):
            eu.warm_verify_renorm_ops(device="cpu")
        self.assertEqual(len(seen), 2)
        for name, pshape, pdtype, tshape, tdtype in seen:
            self.assertEqual(pshape, (1, 8))
            self.assertEqual(pdtype, torch.float32)
            self.assertEqual(tshape, (1, 1))
            self.assertEqual(tdtype, torch.int32 if name == "k" else torch.float32)

    def test_all_eagle_sample_workers_are_marked(self):
        from unittest.mock import MagicMock

        # ngram_worker imports the sgl_kernel binary extension at module
        # level; there is no CPU wheel. The attribute under test is a plain
        # class flag, so mocks stand in for the missing extension.
        sgl_kernel = types.ModuleType("sgl_kernel")
        sgl_kernel.__path__ = []
        with patch.dict(
            sys.modules,
            {"sgl_kernel": sgl_kernel, "sgl_kernel.speculative": MagicMock()},
        ):
            from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
            from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
                MultiLayerEagleWorkerV2,
            )
            from sglang.srt.speculative.ngram_worker import NGRAMWorker
            from sglang.srt.speculative.uno_worker_v2 import UnoWorkerV2

        for worker_cls in (
            EAGLEWorkerV2,
            MultiLayerEagleWorkerV2,
            NGRAMWorker,
            UnoWorkerV2,
        ):
            with self.subTest(worker=worker_cls.__name__):
                self.assertTrue(worker_cls.verify_uses_eagle_sample)


if __name__ == "__main__":
    unittest.main()
