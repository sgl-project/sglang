"""Unit tests for attention-backend SWA KV pool resolution."""

import unittest
from unittest.mock import MagicMock

from sglang.srt.environ import envs
from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=11, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=8, suite="stage-b-test-1-gpu-large-amd")

_RESOLVERS = (
    ("trtllm_mha", TRTLLMHAAttnBackend._resolve_swa_kv_pool, SWAKVPool),
    ("flashinfer", FlashInferAttnBackend._resolve_swa_kv_pool, BaseSWAKVPool),
)


def _mock_runner(
    *,
    active_pool=None,
    is_draft_worker=False,
    spec_algorithm=SpeculativeAlgorithm.NONE,
    allocator_kvcache=None,
):
    runner = MagicMock()
    runner.token_to_kv_pool = active_pool
    runner.is_draft_worker = is_draft_worker
    runner.spec_algorithm = spec_algorithm
    runner.token_to_kv_pool_allocator.get_kvcache.return_value = allocator_kvcache
    return runner


class TestResolveSwaKvPool(CustomTestCase):
    def test_active_pool_is_swa_returns_it(self):
        for name, resolve, pool_type in _RESOLVERS:
            with self.subTest(backend=name):
                swa = MagicMock(spec=pool_type)
                runner = _mock_runner(active_pool=swa)
                self.assertIs(resolve(runner), swa)

    def test_non_swa_active_pool_falls_through_to_allocator(self):
        for name, resolve, pool_type in _RESOLVERS:
            with self.subTest(backend=name):
                swa = MagicMock(spec=pool_type)
                runner = _mock_runner(active_pool=MagicMock(), allocator_kvcache=swa)
                self.assertIs(resolve(runner), swa)

    def test_allocator_kvcache_not_swa_returns_none(self):
        for name, resolve, _ in _RESOLVERS:
            with self.subTest(backend=name):
                runner = _mock_runner(
                    active_pool=MagicMock(), allocator_kvcache=MagicMock()
                )
                self.assertIsNone(resolve(runner))

    def test_draft_worker_non_frozen_kv_returns_none(self):
        for name, resolve, pool_type in _RESOLVERS:
            with self.subTest(backend=name):
                runner = _mock_runner(
                    active_pool=MagicMock(),
                    is_draft_worker=True,
                    spec_algorithm=SpeculativeAlgorithm.EAGLE,
                    allocator_kvcache=MagicMock(spec=pool_type),
                )
                self.assertIsNone(resolve(runner))

    def test_draft_worker_frozen_kv_mtp_returns_allocator_swa(self):
        for name, resolve, pool_type in _RESOLVERS:
            with self.subTest(backend=name):
                swa = MagicMock(spec=pool_type)
                runner = _mock_runner(
                    active_pool=MagicMock(),
                    is_draft_worker=True,
                    spec_algorithm=SpeculativeAlgorithm.FROZEN_KV_MTP,
                    allocator_kvcache=swa,
                )
                self.assertIs(resolve(runner), swa)

    def test_non_draft_worker_ignores_spec_algorithm(self):
        for name, resolve, pool_type in _RESOLVERS:
            with self.subTest(backend=name):
                swa = MagicMock(spec=pool_type)
                runner = _mock_runner(
                    active_pool=MagicMock(),
                    is_draft_worker=False,
                    spec_algorithm=SpeculativeAlgorithm.EAGLE,
                    allocator_kvcache=swa,
                )
                self.assertIs(resolve(runner), swa)


class TestEncoderOnlyVerifyPageTable(CustomTestCase):
    """SGLANG_OPT_DFLASH_DRAFT_KV_ON_SWA: the expanded ENCODER_ONLY verify rows
    gather the SWA page table on a draft whose layers sit in the SWA sub-pool,
    and only there; a target reading it would get a wrong row, with no crash."""

    def _reads_swa(self, *, env=True, is_draft_worker=True, swa_layer_ids=(0, 1)):
        backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
        backend.expand_encoder_only_verify = True
        backend._swa_kv_pool = MagicMock(
            layers_mapping={0: (0, 0 in swa_layer_ids), 1: (1, 1 in swa_layer_ids)}
        )
        layers = []
        for layer_id in (0, 1):
            layer = MagicMock(spec=RadixAttention)
            layer.layer_id, layer.attn_type = layer_id, AttentionType.ENCODER_ONLY
            layers.append(layer)
        runner = MagicMock(is_draft_worker=is_draft_worker)
        runner.model.modules.return_value = layers
        # override() restores the env only on a clean exit, so nothing raises inside.
        with envs.SGLANG_OPT_DFLASH_DRAFT_KV_ON_SWA.override(env):
            try:
                return backend._resolve_encoder_only_reads_swa(runner)
            except NotImplementedError as error:
                return error

    def test_only_a_draft_on_swa_slots_reads_the_swa_page_table(self):
        self.assertTrue(self._reads_swa())
        self.assertFalse(self._reads_swa(env=False))
        self.assertFalse(self._reads_swa(is_draft_worker=False))
        self.assertIsInstance(self._reads_swa(swa_layer_ids=(0,)), NotImplementedError)


if __name__ == "__main__":
    unittest.main()
