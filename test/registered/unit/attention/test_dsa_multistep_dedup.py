import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    DeepseekSparseAttnMultiStepBackend,
    _is_kpool_metadata_fusion_supported,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestMultiStepDedupGate(CustomTestCase):
    """The multi-step sibling-copy dedup is internal behavior of the fused
    metadata path: it must follow the step backends' effective fusion enable
    (which folds in the geometry and platform gates), with no standalone
    knob."""

    def _wrapper_with_backend_flag(self, value):
        wrapper = MagicMock(spec=DeepseekSparseAttnMultiStepBackend)
        backend0 = MagicMock()
        backend0.experimental_kpool_metadata_fusion = value
        wrapper.attn_backends = [backend0, MagicMock(), MagicMock()]
        return wrapper

    def test_dedup_follows_fusion_enable(self):
        wrapper = self._wrapper_with_backend_flag(True)
        self.assertTrue(
            DeepseekSparseAttnMultiStepBackend._multistep_dedup_enabled(wrapper)
        )

    def test_dedup_off_when_fusion_off(self):
        wrapper = self._wrapper_with_backend_flag(False)
        self.assertFalse(
            DeepseekSparseAttnMultiStepBackend._multistep_dedup_enabled(wrapper)
        )

    def test_dedup_off_when_backend_predates_flag(self):
        wrapper = MagicMock(spec=DeepseekSparseAttnMultiStepBackend)
        backend0 = MagicMock(spec=[])  # no fusion attribute at all
        wrapper.attn_backends = [backend0]
        self.assertFalse(
            DeepseekSparseAttnMultiStepBackend._multistep_dedup_enabled(wrapper)
        )


class TestSiblingCopyFallback(CustomTestCase):
    """_copy_replay_metadata_from_sibling must take the FULL recompute path
    whenever the CUDA-only derived-copy body cannot run bit-exactly:
    non-CUDA platforms, missing metadata, or a stale/incompatible sibling."""

    def _mock_backend(self, bs=4, with_metadata=True):
        be = MagicMock(spec=DeepseekSparseAttnBackend)
        metadata = MagicMock()
        be.decode_cuda_graph_metadata = {bs: metadata} if with_metadata else {}
        be.forward_metadata = metadata
        return be, metadata

    def _run(self, is_cuda_val, is_hip_val=False, self_has_metadata=True):
        bs = 4
        dst, dst_meta = self._mock_backend(bs, with_metadata=self_has_metadata)
        src, src_meta = self._mock_backend(bs)
        src.forward_metadata = src_meta  # fresh sibling
        dst._sibling_replay_metadata_compatible = MagicMock(return_value=True)
        precomputed = MagicMock()

        with patch(
            "sglang.srt.layers.attention.dsa_backend.is_cuda",
            return_value=is_cuda_val,
        ), patch("sglang.srt.layers.attention.dsa_backend._is_hip", is_hip_val):
            DeepseekSparseAttnBackend._copy_replay_metadata_from_sibling(
                dst,
                src_backend=src,
                bs=bs,
                precomputed=precomputed,
                forward_mode=ForwardMode.DECODE,
            )
        return dst

    def test_non_cuda_platform_recomputes_in_full(self):
        dst = self._run(is_cuda_val=False)
        dst.init_forward_metadata_replay_cuda_graph_from_precomputed.assert_called_once()
        dst._copy_base_replay_buffers.assert_not_called()

    def test_hip_platform_recomputes_in_full(self):
        dst = self._run(is_cuda_val=True, is_hip_val=True)
        dst.init_forward_metadata_replay_cuda_graph_from_precomputed.assert_called_once()
        dst._copy_base_replay_buffers.assert_not_called()

    def test_missing_metadata_recomputes_in_full(self):
        dst = self._run(is_cuda_val=True, self_has_metadata=False)
        dst.init_forward_metadata_replay_cuda_graph_from_precomputed.assert_called_once()
        dst._copy_base_replay_buffers.assert_not_called()

    def test_cuda_compatible_sibling_takes_copy_path(self):
        dst = self._run(is_cuda_val=True)
        dst.init_forward_metadata_replay_cuda_graph_from_precomputed.assert_not_called()
        dst._copy_base_replay_buffers.assert_called_once()
        dst._copy_kpool_metadata_from_sibling.assert_called_once()


class TestInGraphVerifyEligibility(CustomTestCase):
    def _eligible(self, dcp_enabled):
        backend = MagicMock(spec=DeepseekSparseAttnBackend)
        backend.ingraph_verify_metadata_enabled = True
        backend.dsa_index_kpool = 16
        backend.experimental_kpool_metadata_fusion = True
        backend.dsa_decode_impl = "trtllm"
        parallel = MagicMock(dcp_enabled=dcp_enabled)

        with patch(
            "sglang.srt.layers.attention.dsa_backend.is_cuda", return_value=True
        ), patch("sglang.srt.layers.attention.dsa_backend._is_hip", False), patch(
            "sglang.srt.layers.attention.dsa_backend.get_parallel",
            return_value=parallel,
        ):
            return DeepseekSparseAttnBackend._ingraph_verify_metadata_eligible(backend)

    def test_uninitialized_dcp_group_is_eligible(self):
        self.assertTrue(self._eligible(dcp_enabled=False))

    def test_dcp_is_not_eligible(self):
        self.assertFalse(self._eligible(dcp_enabled=True))


class TestKPoolMetadataFusionGeometry(CustomTestCase):
    def test_final_checkpoint_geometry_is_supported(self):
        self.assertTrue(_is_kpool_metadata_fusion_supported(4, 64, 2048))

    def test_legacy_checkpoint_geometry_is_supported(self):
        self.assertTrue(_is_kpool_metadata_fusion_supported(16, 64, 2048))

    def test_non_page_aligned_pool_is_not_supported(self):
        self.assertFalse(_is_kpool_metadata_fusion_supported(6, 64, 2048))

    def test_non_pool_aligned_topk_is_not_supported(self):
        self.assertFalse(_is_kpool_metadata_fusion_supported(4, 64, 2049))


if __name__ == "__main__":
    unittest.main()
