import unittest

from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.storage.backend_factory import (
    StorageBackendFactory,
    StorageCapability,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=3, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=2, suite="stage-b-test-1-gpu-small-amd")

_V2_BACKENDS = ["file", "mooncake"]
_NON_V2_BACKENDS = ["nixl", "hf3fs", "aibrix", "eic", "simm"]


def _should_build_hybrid_stack(is_nsa, backend, extra_config=None):
    """Mirrors the guard logic in HiRadixCache.__init__."""
    return is_nsa and (
        backend is None
        or StorageBackendFactory.supports(
            StorageCapability.interface_v2, backend, extra_config
        )
    )


def _should_build_mamba_hybrid_stack(backend, extra_config=None):
    """Mirrors the startup guard in HiMambaRadixCache.__init__."""
    return backend is None or StorageBackendFactory.supports(
        StorageCapability.interface_v2, backend, extra_config
    )


class TestHybridControllerRequiresV2(unittest.TestCase):
    """HybridCacheController.attach_storage_backend must reject non-v2 backends."""

    def _make_controller(self):
        ctrl = HybridCacheController.__new__(HybridCacheController)
        ctrl.mem_pool_host = type("FakeMemPoolHost", (), {"entries": []})()
        return ctrl

    def test_v2_backends_pass_guard(self):
        ctrl = self._make_controller()
        for name in _V2_BACKENDS:
            with self.subTest(backend=name):
                try:
                    ctrl.attach_storage_backend(storage_backend=name)
                except RuntimeError as e:
                    if "interface_v2" in str(e):
                        self.fail(f"{name} should pass the v2 guard but got: {e}")
                except Exception:
                    pass  # other init errors are expected with a stub controller

    def test_non_v2_backends_rejected(self):
        ctrl = self._make_controller()
        for name in _NON_V2_BACKENDS:
            with self.subTest(backend=name):
                with self.assertRaises(RuntimeError):
                    ctrl.attach_storage_backend(storage_backend=name)

    def test_dynamic_v2_passes_guard(self):
        ctrl = self._make_controller()
        try:
            ctrl.attach_storage_backend(
                storage_backend="dynamic",
                storage_backend_extra_config={"interface_v2": 1},
            )
        except RuntimeError as e:
            if "interface_v2" in str(e):
                self.fail(f"dynamic+interface_v2 should pass the guard: {e}")
        except Exception:
            pass

    def test_dynamic_without_v2_rejected(self):
        ctrl = self._make_controller()
        with self.assertRaises(RuntimeError):
            ctrl.attach_storage_backend(
                storage_backend="dynamic",
                storage_backend_extra_config={},
            )


if __name__ == "__main__":
    unittest.main()
