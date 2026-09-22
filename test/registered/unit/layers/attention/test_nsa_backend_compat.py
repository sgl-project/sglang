import importlib
import sys
import unittest
import warnings

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_NSA_MODULE = "sglang.srt.layers.attention.nsa_backend"


class TestNsaBackendCompatShim(CustomTestCase):
    """`nsa_backend.py` is a deprecation shim that re-exports the renamed DSA
    classes under their old NSA (Native Sparse Attention) names. Regression guard
    for the shim: importing it must succeed and every aliased name must resolve to
    the current DSA class (previously it raised ImportError because the alias block
    self-assigned the DSA names instead of defining the NSA names)."""

    def test_shim_imports_and_all_nsa_aliases_resolve(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nsa = importlib.import_module(_NSA_MODULE)
        from sglang.srt.layers.attention.dsa_backend import (
            DeepseekSparseAttnBackend,
            DeepseekSparseAttnMultiStepBackend,
            DSAFlashMLAMetadata,
            DSAIndexerMetadata,
            DSAMetadata,
        )

        self.assertIs(nsa.NativeSparseAttnBackend, DeepseekSparseAttnBackend)
        self.assertIs(
            nsa.NativeSparseAttnMultiStepBackend, DeepseekSparseAttnMultiStepBackend
        )
        self.assertIs(nsa.NSAMetadata, DSAMetadata)
        self.assertIs(nsa.NSAFlashMLAMetadata, DSAFlashMLAMetadata)
        self.assertIs(nsa.NSAIndexerMetadata, DSAIndexerMetadata)

    def test_shim_emits_deprecation_warning(self):
        sys.modules.pop(_NSA_MODULE, None)
        with self.assertWarns(DeprecationWarning):
            importlib.import_module(_NSA_MODULE)


if __name__ == "__main__":
    unittest.main()
