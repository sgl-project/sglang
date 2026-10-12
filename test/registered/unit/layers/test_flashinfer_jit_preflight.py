import os
import tempfile
import unittest

from sglang.srt.layers import flashinfer_jit_preflight as fjp
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _make_build_dir(state: str) -> str:
    """Create a fake flashinfer sampling build dir in the requested state.

    Layout mirrors flashinfer 0.6.18: cached_ops/sampling/ holds build.ninja,
    csrc_*.cuda.o objects, and the linked sampling.so.
    """
    build_dir = tempfile.mkdtemp(prefix="fi_sampling_")
    if state == "empty":
        pass
    elif state == "poisoned":
        # Exactly what a failed `RuntimeError: Ninja build failed` leaves:
        # build.ninja written up front, one object compiled, link never ran.
        open(os.path.join(build_dir, "build.ninja"), "w").close()
        open(os.path.join(build_dir, "csrc_renorm.cuda.o"), "w").close()
        open(os.path.join(build_dir, ".ninja_log"), "w").close()
    elif state == "objects_only":
        open(os.path.join(build_dir, "csrc_sampling.cuda.o"), "w").close()
    elif state == "compiled":
        open(os.path.join(build_dir, "build.ninja"), "w").close()
        open(os.path.join(build_dir, "csrc_renorm.cuda.o"), "w").close()
        open(os.path.join(build_dir, "sampling.so"), "w").close()
    else:
        raise ValueError(state)
    return build_dir


class TestPoisonedCacheDetection(CustomTestCase):
    def test_missing_dir_is_not_poisoned(self):
        self.assertFalse(fjp.is_flashinfer_sampling_poisoned("/nonexistent/path"))

    def test_empty_dir_is_not_poisoned(self):
        # A dir with no artifacts carries no signal either way.
        self.assertFalse(fjp.is_flashinfer_sampling_poisoned(_make_build_dir("empty")))

    def test_partial_build_without_so_is_poisoned(self):
        build_dir = _make_build_dir("poisoned")
        self.assertTrue(fjp.is_flashinfer_sampling_poisoned(build_dir))

    def test_objects_without_ninja_file_are_poisoned(self):
        # The .o-only variant: ninja file rewritten/cleaned, partials remain.
        self.assertTrue(
            fjp.is_flashinfer_sampling_poisoned(_make_build_dir("objects_only"))
        )

    def test_complete_build_is_not_poisoned(self):
        self.assertFalse(
            fjp.is_flashinfer_sampling_poisoned(_make_build_dir("compiled"))
        )

    def test_no_flashinfer_means_no_build_dir(self):
        # CPU-only environments: flashinfer.jit is not importable, so the
        # build dir resolves to None and nothing is ever declared poisoned.
        original = fjp._get_flashinfer_jit_env
        fjp._get_flashinfer_jit_env = lambda: None
        try:
            self.assertIsNone(fjp.flashinfer_sampling_build_dir())
            self.assertFalse(fjp.is_flashinfer_sampling_poisoned())
            cleared, path = fjp.clear_poisoned_flashinfer_sampling_cache()
            self.assertFalse(cleared)
            self.assertIsNone(path)
        finally:
            fjp._get_flashinfer_jit_env = original


class TestPoisonedCacheClearing(CustomTestCase):
    def test_clear_removes_poisoned_dir(self):
        build_dir = _make_build_dir("poisoned")
        cleared, path = fjp.clear_poisoned_flashinfer_sampling_cache(build_dir)
        self.assertTrue(cleared)
        self.assertEqual(path, build_dir)
        self.assertFalse(os.path.exists(build_dir))

    def test_clear_leaves_complete_build_alone(self):
        build_dir = _make_build_dir("compiled")
        cleared, _ = fjp.clear_poisoned_flashinfer_sampling_cache(build_dir)
        self.assertFalse(cleared)
        self.assertTrue(os.path.isfile(os.path.join(build_dir, "sampling.so")))

    def test_clear_leaves_clean_dirs_alone(self):
        build_dir = _make_build_dir("empty")
        cleared, _ = fjp.clear_poisoned_flashinfer_sampling_cache(build_dir)
        self.assertFalse(cleared)
        self.assertTrue(os.path.isdir(build_dir))

    def test_sibling_ops_survive_targeted_clear(self):
        # The clear must be surgical: only cached_ops/sampling/ goes away,
        # unlike flashinfer's own clear_cache_dir() which wipes every op.
        ops_root = tempfile.mkdtemp(prefix="fi_cached_ops_")
        sampling_dir = os.path.join(ops_root, "sampling")
        os.makedirs(sampling_dir)
        open(os.path.join(sampling_dir, "build.ninja"), "w").close()
        open(os.path.join(sampling_dir, "csrc_renorm.cuda.o"), "w").close()
        norm_dir = os.path.join(ops_root, "norm")
        os.makedirs(norm_dir)
        open(os.path.join(norm_dir, "norm.so"), "w").close()

        cleared, _ = fjp.clear_poisoned_flashinfer_sampling_cache(sampling_dir)
        self.assertTrue(cleared)
        self.assertFalse(os.path.exists(sampling_dir))
        self.assertTrue(os.path.isfile(os.path.join(norm_dir, "norm.so")))


if __name__ == "__main__":
    unittest.main()
