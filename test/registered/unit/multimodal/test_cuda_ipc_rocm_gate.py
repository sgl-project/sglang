"""Unit tests for CUDA_IPC_TRANSPORT_SUPPORTED (base_processor.py) -- the
producer-side platform gate that keeps ROCm out of the CUDA-IPC transport
path entirely.

Background (sgl-project/sglang#29687, a follow-up to #29227): SGLang's
multimodal CUDA-IPC transport (SGLANG_USE_CUDA_IPC_TRANSPORT=1, opt-in,
default off) is used to share large VLM encoder features between tokenizer
workers and the scheduler without a host round-trip. Every call site used to
gate purely on the env var (`SGL_USE_CUDA_IPC`) or on `tensor.is_cuda`,
neither of which distinguishes real NVIDIA CUDA from ROCm/HIP -- PyTorch has
no separate device type for HIP, so `tensor.is_cuda` is True and
`torch.device("cuda")` is the only valid device string on both backends (see
sglang.srt.platforms.rocm). The raw CUDA-IPC handle reconstruction on the
consumer side (cuda_ipc_transport_utils.py) crashes with
"HIP error: invalid device pointer" on ROCm because a device-redirected IPC
handle isn't reopenable there the same way it is on real CUDA.

The fix: base_processor.py now derives
`CUDA_IPC_TRANSPORT_SUPPORTED = _compute_cuda_ipc_transport_supported(SGL_USE_CUDA_IPC, is_hip())`
(a pure `use_ipc_env and not is_hip_platform` function of two booleans) and
uses the result (not the raw env flag) at every point that would construct a
CudaIpcTensorTransportProxy with a real IPC handle attached (pool
construction, the wrap helper, the post-process invocation, and the mirrored
logic in moss_vl.py / ernie45_vl.py). On ROCm this routes tensors through the
transport's existing tensor_data passthrough (an ordinary `.to(device)` copy,
tested separately in
test/registered/unit/utils/test_cuda_ipc_transport_rocm_gate.py) instead of
attempting raw IPC -- genuinely bypassing CUDA-IPC end-to-end, matching what
the pool's own "falling back to non-IPC transport" log message already (and,
before this fix, inaccurately) claimed.

On mocking: the derivation logic itself
(TestComputeCudaIpcTransportSupportedPure) is tested with real booleans and
zero mocking, because it was extracted into a pure function specifically so
it could be. Only one test class
(TestCudaIpcTransportSupportedModuleWiring) still patches is_hip() + reloads
the module, and only to prove the module-level constant is actually wired
to that pure function with real inputs -- is_hip() reads real hardware/build
state that cannot be fabricated without ROCm, so that one patch is an
environment-boundary fake, not a fake of the code under test.
_wrap_tensor_for_cuda_ipc's tests still need `_FakeCudaTensor` (see its
docstring below) because `torch.Tensor.is_cuda` is a read-only, hardware-
backed property with no CPU-side override -- a genuine, unavoidable
environment limit, not a convenience shortcut.

No server, no model loading, no GPU required.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import importlib
import os
import unittest
from unittest.mock import patch

import torch

from sglang.srt.multimodal.processors import base_processor as base_processor_mod
from sglang.test.test_utils import CustomTestCase

_ENV_VAR = "SGLANG_USE_CUDA_IPC_TRANSPORT"


class _FakeServerArgs:
    tokenizer_worker_num = 1
    base_gpu_id = 0
    keep_mm_feature_on_device = False
    disable_fast_image_processor = True
    rl_on_policy_target = None


class TestComputeCudaIpcTransportSupportedPure(CustomTestCase):
    """Zero-mock unit tests for _compute_cuda_ipc_transport_supported, the
    pure function CUDA_IPC_TRANSPORT_SUPPORTED's derivation was extracted
    into. Plain booleans in, plain boolean out -- no env vars, no module
    reload, no patching of is_hip anywhere. This is the real regression
    coverage for the "ROCm must never be supported, even if the user opts
    in" rule; the integration test below it additionally proves the
    module-level constant is actually wired to this function with real
    inputs, which this class alone cannot prove."""

    def test_rocm_never_supported_even_when_user_opts_in(self):
        self.assertFalse(
            base_processor_mod._compute_cuda_ipc_transport_supported(
                use_ipc_env=True, is_hip_platform=True
            ),
            "ROCm must never be routed through raw CUDA-IPC reconstruction, "
            "regardless of the env var (sgl-project/sglang#29687)",
        )

    def test_rocm_stays_unsupported_when_user_opts_out_too(self):
        self.assertFalse(
            base_processor_mod._compute_cuda_ipc_transport_supported(
                use_ipc_env=False, is_hip_platform=True
            )
        )

    def test_non_hip_supported_when_user_opts_in(self):
        self.assertTrue(
            base_processor_mod._compute_cuda_ipc_transport_supported(
                use_ipc_env=True, is_hip_platform=False
            )
        )

    def test_non_hip_not_supported_when_user_opts_out(self):
        self.assertFalse(
            base_processor_mod._compute_cuda_ipc_transport_supported(
                use_ipc_env=False, is_hip_platform=False
            )
        )


class TestCudaIpcTransportSupportedModuleWiring(CustomTestCase):
    """Integration check that the module-level CUDA_IPC_TRANSPORT_SUPPORTED
    constant is actually wired to _compute_cuda_ipc_transport_supported fed
    by the real env var and the real is_hip() -- not just that the pure
    function's logic is correct in isolation (covered above with zero
    mocking). This is the one place in this file that still needs to patch
    is_hip() and reload the module, because proving the WIRING is correct
    inherently requires observing the module-level assignment re-run against
    controlled inputs; is_hip() itself reads real hardware/build state
    (torch.version.hip) that cannot be fabricated in a sandbox without ROCm,
    so patching it here is an environment-boundary fake, not a fake of the
    logic under test (that logic is already fully covered, unmocked, above).
    """

    def setUp(self):
        self._prev_env = os.environ.get(_ENV_VAR)

    def tearDown(self):
        if self._prev_env is None:
            os.environ.pop(_ENV_VAR, None)
        else:
            os.environ[_ENV_VAR] = self._prev_env
        # Always leave the real module state reloaded back to reality so
        # later tests (and this module's own re-import) see a clean flag
        # derived from the real is_hip() with no test-time env override.
        importlib.reload(base_processor_mod)

    def _reload_with(self, *, env_enabled: bool, hip: bool):
        os.environ[_ENV_VAR] = "1" if env_enabled else "0"
        # base_processor.py does `from sglang.srt.utils import is_hip`, which
        # resolves through sglang.srt.utils.__init__'s `from
        # sglang.srt.utils.common import *` re-export -- a separate name
        # binding in the `sglang.srt.utils` package namespace from
        # `sglang.srt.utils.common.is_hip`. Patch both so the reload picks
        # up the mock regardless of which one base_processor's fresh `from
        # ... import is_hip` binds to.
        with patch("sglang.srt.utils.common.is_hip", return_value=hip), patch(
            "sglang.srt.utils.is_hip", return_value=hip
        ):
            return importlib.reload(base_processor_mod)

    def test_module_constant_reflects_real_hip_and_env_wiring(self):
        # One representative case (opted-in + HIP) is enough to prove the
        # WIRING (env -> SGL_USE_CUDA_IPC, is_hip() -> the pure function's
        # is_hip_platform arg, result -> CUDA_IPC_TRANSPORT_SUPPORTED) is
        # intact end-to-end; the full logic truth table is already covered
        # with zero mocking in TestComputeCudaIpcTransportSupportedPure.
        reloaded = self._reload_with(env_enabled=True, hip=True)
        self.assertTrue(reloaded.SGL_USE_CUDA_IPC, "user intent should still read True")
        self.assertFalse(
            reloaded.CUDA_IPC_TRANSPORT_SUPPORTED,
            "module-level constant did not pick up is_hip()=True through "
            "the real wiring",
        )


class _ConcreteStubProcessor(base_processor_mod.BaseMultimodalProcessor):
    """BaseMultimodalProcessor is an ABC with exactly one abstract method
    (process_mm_data_async). We never call it -- it exists only so this
    stub can be instantiated to exercise the one concrete method under test,
    _wrap_tensor_for_cuda_ipc, without needing a real HF processor/config."""

    async def process_mm_data_async(self, *args, **kwargs):
        raise NotImplementedError("not exercised by this test")


class TestWrapTensorForCudaIpcRocmFallback(CustomTestCase):
    """_wrap_tensor_for_cuda_ipc must never touch self.cudaipc_mmfeature_pool
    (which, per the pool-construction gate at __init__, does not even exist
    when CUDA_IPC_TRANSPORT_SUPPORTED is False) and must fall back to the
    existing tensor_data-style passthrough instead."""

    def _make_stub_processor(self, *, cuda_ipc_transport_supported: bool):
        # Bypass BaseMultimodalProcessor.__init__ (it needs a real HF
        # processor/config) -- we only need the one method under test plus
        # the attributes it reads.
        proc = object.__new__(_ConcreteStubProcessor)
        proc.server_args = _FakeServerArgs()
        # Deliberately do NOT set self.cudaipc_mmfeature_pool: on ROCm (per
        # the __init__ gate) it is never constructed, so accessing it must
        # not happen on this path -- if it did, this test would fail with
        # AttributeError instead of the assertions below.
        self._patched_flag = patch.object(
            base_processor_mod,
            "CUDA_IPC_TRANSPORT_SUPPORTED",
            cuda_ipc_transport_supported,
        )
        self._patched_flag.start()
        self.addCleanup(self._patched_flag.stop)
        return proc

    def test_cpu_tensor_always_returned_unchanged_regardless_of_platform(self):
        # not tensor.is_cuda short-circuit stays first regardless of the
        # ROCm gate -- CPU tensors were never a CUDA-IPC candidate.
        proc = self._make_stub_processor(cuda_ipc_transport_supported=True)
        t = torch.tensor([1, 2, 3])
        self.assertIs(proc._wrap_tensor_for_cuda_ipc(t), t)

    def test_cuda_tensor_falls_back_without_touching_pool_when_unsupported(self):
        proc = self._make_stub_processor(cuda_ipc_transport_supported=False)
        fake_cuda_tensor = _FakeCudaTensor([1.0, 2.0, 3.0])

        result = proc._wrap_tensor_for_cuda_ipc(fake_cuda_tensor)

        # keep_mm_feature_on_device=False in _FakeServerArgs -> should hit
        # the .cpu() fallback branch, exactly mirroring the pre-existing
        # "pool full" fallback tail of this same method.
        self.assertTrue(result.moved_to_cpu)
        self.assertFalse(
            hasattr(proc, "cudaipc_mmfeature_pool"),
            "must not construct/touch the IPC pool when transport is unsupported",
        )

    def test_cuda_tensor_kept_on_device_when_keep_mm_feature_on_device(self):
        proc = self._make_stub_processor(cuda_ipc_transport_supported=False)
        proc.server_args.keep_mm_feature_on_device = True
        fake_cuda_tensor = _FakeCudaTensor([1.0, 2.0, 3.0])

        result = proc._wrap_tensor_for_cuda_ipc(fake_cuda_tensor)

        self.assertIs(result, fake_cuda_tensor)
        self.assertFalse(hasattr(proc, "cudaipc_mmfeature_pool"))


class _FakeCudaTensor:
    """Minimal stand-in for a CUDA torch.Tensor.

    This one is a genuine, unavoidable environment limit, not a convenience
    shortcut: `torch.Tensor.is_cuda` is a read-only property backed directly
    by the C++ TensorImpl (pybind-exposed via `torch._C.TensorBase`) with no
    CPU-side override -- verified directly: `some_real_cpu_tensor.is_cuda =
    True` raises `AttributeError: attribute 'is_cuda' of 'torch._C.TensorBase'
    objects is not writable`. There is no way to make a real torch.Tensor
    report is_cuda=True without an actual CUDA/ROCm device backing it, which
    this sandbox does not have.

    This stub exercises ONLY the narrow interface contract that
    _wrap_tensor_for_cuda_ipc's fallback branch actually touches on a CUDA
    tensor -- the `.is_cuda` truthiness check and, on the CPU-fallback path,
    `.cpu()` -- not any real CUDA semantics (no actual device memory, no
    real dtype/shape/stride behavior). It proves the ROCm-gate *branching*
    is correct (falls back instead of touching the IPC pool); it does not
    and cannot prove anything about real CUDA-IPC transport behavior itself,
    which requires real hardware (see this file's module docstring for what
    is and isn't covered by these tests)."""

    is_cuda = True

    def __init__(self, values):
        self._values = values
        self.moved_to_cpu = False

    def cpu(self):
        self.moved_to_cpu = True
        return self


if __name__ == "__main__":
    unittest.main()
