"""Ownership and fallback checks for borrowing Full KV during GDN replay."""

import unittest
from array import array
from contextlib import ExitStack
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache import gdn_kv_reuse as reuse
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestGdnKVReuse(unittest.TestCase):
    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        values = {
            "attention_backends": lambda: ("flashinfer", "flashinfer"),
            "get_disagg": lambda: NS(disaggregation_mode="null"),
            "get_exec": lambda: NS(graph=NS(cuda_graph_backend_prefill="disabled")),
            "get_lora": lambda: NS(enable_lora=False),
            "get_memory": lambda: NS(
                enable_hierarchical_cache=False, enable_session_radix_cache=False
            ),
            "get_parallel": lambda: NS(
                tp_size=1, pp_size=1, attn_cp_size=1, attn_dp_size=1
            ),
            "get_schedule": lambda: NS(enable_mixed_chunk=False),
            "mamba_checkpoint_grid": lambda page_size: 256,
        }
        self.context.enter_context(patch.multiple(reuse, **values))
        self.context.enter_context(
            patch.object(
                reuse.envs.SGLANG_ENABLE_GDN_KV_REUSE, "get", return_value=True
            )
        )
        self.cache = Mock(spec=UnifiedRadixCache)
        self.cache.disable = False
        self.cache.supports_mamba.return_value = True
        self.cache.components = {0: None, 1: None}
        self.cache.enable_mamba_extra_buffer = True
        self.cache.page_size = 256
        self.cache.tree_core = Mock()
        self.cache.tree_core.match_full_device_prefix.return_value = (8192, 17, 12288)
        self.indices = torch.arange(12288, dtype=torch.int64) + 256
        self.cache.path_device_indices.return_value = self.indices
        self.req = NS(
            prefix_len=4096,
            extend_end=8192,
            kv=ReqKvInfo(cache_protected_len=4096),
            full_kv_replay_node=None,
            skip_radix_cache_insert=False,
            multimodal_inputs=None,
            input_embeds=None,
            positional_embed_overrides=None,
            session_id=None,
            mamba_branching_seqlen=8192,
            full_untruncated_fill_ids=array("q", range(8193)),
            extra_key=None,
            cache_salt="test-namespace",
            rid="test",
        )
        self.batch = NS(
            tree_cache=self.cache,
            reqs=[self.req],
            enable_overlap=False,
            model_config=NS(
                hf_text_config=NS(model_type="qwen3_5_text", attn_output_gate=True),
                quantization=None,
                dtype=torch.bfloat16,
            ),
            spec_algorithm=NS(is_none=lambda: True),
            full_kv_replay=False,
        )

    def assert_fallback(self):
        self.assertIsNone(reuse.try_borrow_full_kv(self.batch))
        self.assertEqual(self.req.kv.cache_protected_len, 4096)
        self.assertIsNone(self.req.full_kv_replay_node)
        self.assertFalse(self.batch.full_kv_replay)
        self.cache.tree_core.inc_full_pin.assert_not_called()

    def test_borrows_only_current_chunk_and_preserves_compute_prefix(self):
        out = reuse.try_borrow_full_kv(self.batch)
        self.assertTrue(torch.equal(out, self.indices[4096:8192]))
        self.assertEqual(
            out.untyped_storage().data_ptr(), self.indices.untyped_storage().data_ptr()
        )
        self.assertEqual(self.req.prefix_len, 4096)
        self.assertEqual(self.req.kv.cache_protected_len, 8192)
        self.assertTrue(self.batch.full_kv_replay)
        self.cache.tree_core.inc_full_pin.assert_called_once_with(17)
        key = self.cache.tree_core.match_full_device_prefix.call_args.args[0]
        self.assertEqual(len(key), 8192)
        self.assertEqual(key.cache_salt, "test-namespace")
        reuse.release_full_kv_pin(self.cache, self.req)
        reuse.release_full_kv_pin(self.cache, self.req)
        self.cache.tree_core.dec_full_pin.assert_called_once_with(17)

    def test_disabled_flag_falls_back(self):
        with patch.object(
            reuse.envs.SGLANG_ENABLE_GDN_KV_REUSE, "get", return_value=False
        ):
            self.assert_fallback()

    def test_release_without_checkpoint_cleans_up_and_unpins(self):
        reuse.try_borrow_full_kv(self.batch)
        component = Mock()
        self.cache._components_tuple = (component,)
        UnifiedRadixCache.on_release(self.cache, self.req, checkpointed=False)
        self.cache.tree_core.dec_full_pin.assert_called_once_with(17)
        component.cleanup_after_caching_req.assert_called_once_with(
            self.req, is_finished=True
        )

    def test_release_after_checkpoint_still_unpins(self):
        reuse.try_borrow_full_kv(self.batch)
        UnifiedRadixCache.on_release(self.cache, self.req, checkpointed=True)
        self.cache.tree_core.dec_full_pin.assert_called_once_with(17)

    def test_experimental_olmo_adapter_can_borrow(self):
        config = self.batch.model_config.hf_text_config
        config.model_type = "olmo_hybrid"
        config.attn_output_gate = False
        config.sglang_olmo_hybrid_adapter = True
        self.assertIsNotNone(reuse.try_borrow_full_kv(self.batch))
        reuse.release_full_kv_pin(self.cache, self.req)

    def test_unregistered_olmo_falls_back(self):
        self.batch.model_config.hf_text_config.model_type = "olmo_hybrid"
        self.assert_fallback()

    def test_partial_device_hit_falls_back(self):
        self.cache.tree_core.match_full_device_prefix.return_value = (7936, 17, 8192)
        self.assert_fallback()

    def test_mixed_new_tokens_fall_back(self):
        self.req.extend_end = 8448
        self.req.mamba_branching_seqlen = 8448
        self.assert_fallback()

    def test_multi_request_batch_falls_back(self):
        self.batch.reqs.append(self.req)
        self.assert_fallback()

    def test_unaligned_chunk_falls_back(self):
        self.req.extend_end = 8191
        self.assert_fallback()

    def test_quantized_model_falls_back(self):
        self.batch.model_config.quantization = "modelopt_fp4"
        self.assert_fallback()

    def test_overlap_falls_back(self):
        self.batch.enable_overlap = True
        self.assert_fallback()

    def test_earlier_branch_checkpoint_falls_back(self):
        self.req.mamba_branching_seqlen = 6144
        self.assert_fallback()

    def test_failed_path_collection_releases_pin(self):
        self.cache.path_device_indices.side_effect = RuntimeError("failed collection")
        with self.assertRaisesRegex(RuntimeError, "failed collection"):
            reuse.try_borrow_full_kv(self.batch)
        self.cache.tree_core.dec_full_pin.assert_called_once_with(17)
        self.assertEqual(self.req.kv.cache_protected_len, 4096)
        self.assertIsNone(self.req.full_kv_replay_node)


if __name__ == "__main__":
    unittest.main()
