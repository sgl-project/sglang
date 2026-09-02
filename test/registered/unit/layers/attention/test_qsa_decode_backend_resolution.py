import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.attention import qwen_sparse_attn_backend as qsa
from sglang.srt.layers.attention.qsa import preflight as qsa_preflight
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestQSASparseDecodeBackendResolution(CustomTestCase):
    """Regress late FA4 selection on unsupported NVIDIA architectures.

    On SM89, an importable FA4 was previously selected when classic FA2 was
    absent and failed only after model loading. The matrix also verifies that
    the existing SM90/SM100/SM120/SM121 policies remain unchanged.
    """

    def setUp(self):
        super().setUp()
        qsa._resolve_sm121_kda_sparse_decode.cache_clear()
        qsa.resolve_qsa_sparse_decode_backend.cache_clear()

    def tearDown(self):
        qsa._resolve_sm121_kda_sparse_decode.cache_clear()
        qsa.resolve_qsa_sparse_decode_backend.cache_clear()
        super().tearDown()

    def _resolve(self, capability, *, trtllm=None, sm121_kda=None, fa2=None, fa4=None):
        with (
            patch.object(qsa.torch.version, "hip", None),
            patch.object(qsa, "_nvidia_compute_capability", return_value=capability),
            patch.object(qsa, "_resolve_trtllm_sparse_decode", return_value=trtllm),
            patch.object(
                qsa,
                "_resolve_sm121_kda_sparse_decode",
                return_value=sm121_kda,
            ) as resolve_sm121_kda,
            patch.object(
                qsa, "_resolve_classic_fa2_sparse_decode", return_value=fa2
            ) as resolve_fa2,
            patch.object(
                qsa, "_resolve_fa4_cute_sparse_decode", return_value=fa4
            ) as resolve_fa4,
        ):
            result = qsa.resolve_qsa_sparse_decode_backend()
        return result, resolve_sm121_kda, resolve_fa2, resolve_fa4

    def test_sm89_uses_classic_fa2(self):
        fa2 = object()
        result, _, _, resolve_fa4 = self._resolve((8, 9), fa2=fa2, fa4=object())
        self.assertEqual(result.kind, qsa.QSASparseDecodeBackendKind.FA2)
        self.assertIs(result.kernel, fa2)
        resolve_fa4.assert_not_called()

    def test_sm89_does_not_fall_back_to_fa4_when_fa2_is_missing(self):
        with self.assertRaisesRegex(RuntimeError, "SM89 requires classic"):
            self._resolve((8, 9), fa4=object())

    def test_sm90_uses_classic_fa2(self):
        fa2 = object()
        result, _, _, resolve_fa4 = self._resolve((9, 0), fa2=fa2, fa4=object())
        self.assertEqual(result.kind, qsa.QSASparseDecodeBackendKind.FA2)
        self.assertIs(result.kernel, fa2)
        resolve_fa4.assert_not_called()

    def test_sm90_preserves_fa4_fallback_when_fa2_is_missing(self):
        fa4 = object()
        result, _, _, _ = self._resolve((9, 0), fa4=fa4)
        self.assertEqual(result.kind, qsa.QSASparseDecodeBackendKind.FA4_CUTE)
        self.assertIs(result.kernel, fa4)

    def test_sm100_prefers_trtllm(self):
        trtllm = object()
        result, resolve_sm121_kda, resolve_fa2, resolve_fa4 = self._resolve(
            (10, 0), trtllm=trtllm, fa2=object(), fa4=object()
        )
        self.assertEqual(result.kind, qsa.QSASparseDecodeBackendKind.TRTLLM)
        self.assertIs(result.kernel, trtllm)
        resolve_sm121_kda.assert_not_called()
        resolve_fa2.assert_not_called()
        resolve_fa4.assert_not_called()

    def test_sm100_uses_fa4_when_trtllm_is_missing(self):
        fa4 = object()
        result, _, _, _ = self._resolve((10, 0), fa4=fa4)
        self.assertEqual(result.kind, qsa.QSASparseDecodeBackendKind.FA4_CUTE)
        self.assertIs(result.kernel, fa4)

    def test_sm120_uses_trtllm(self):
        trtllm = object()
        result, resolve_sm121_kda, resolve_fa2, resolve_fa4 = self._resolve(
            (12, 0), trtllm=trtllm, fa2=object(), fa4=object()
        )
        self.assertEqual(result.kind, qsa.QSASparseDecodeBackendKind.TRTLLM)
        self.assertIs(result.kernel, trtllm)
        resolve_sm121_kda.assert_not_called()
        resolve_fa2.assert_not_called()
        resolve_fa4.assert_not_called()

    def test_sm121_preserves_existing_kda_backend(self):
        sm121_kda = object()
        result, _, resolve_fa2, resolve_fa4 = self._resolve(
            (12, 1), sm121_kda=sm121_kda, fa2=object(), fa4=object()
        )
        self.assertEqual(result.kind, qsa.QSASparseDecodeBackendKind.KDA_SM121)
        self.assertIs(result.kernel, sm121_kda)
        resolve_fa2.assert_not_called()
        resolve_fa4.assert_not_called()


class TestQSASparseDecodePreflight(CustomTestCase):
    def test_non_qsa_model_does_not_resolve_qsa_backend(self):
        model_config = SimpleNamespace(hf_text_config=object())
        with (
            patch.object(qsa_preflight, "is_qwen_qsa", return_value=False),
            patch.object(
                qsa_preflight, "_resolve_qsa_sparse_decode_backend"
            ) as resolve,
        ):
            qsa_preflight.preflight_qsa_sparse_decode_backend(
                model_config=model_config, tp_rank=0
            )
        resolve.assert_not_called()

    def test_qsa_model_logs_selected_backend_on_rank_zero(self):
        model_config = SimpleNamespace(hf_text_config=object())
        backend = SimpleNamespace(kind=SimpleNamespace(value="classic FA2"))
        with (
            patch.object(qsa_preflight, "is_qwen_qsa", return_value=True),
            patch.object(
                qsa_preflight,
                "_resolve_qsa_sparse_decode_backend",
                return_value=backend,
            ),
            patch.object(qsa_preflight.logger, "info") as log_info,
        ):
            qsa_preflight.preflight_qsa_sparse_decode_backend(
                model_config=model_config, tp_rank=0
            )
        log_info.assert_called_once_with("QSA sparse decode backend: %s", "classic FA2")


if __name__ == "__main__":
    unittest.main()
