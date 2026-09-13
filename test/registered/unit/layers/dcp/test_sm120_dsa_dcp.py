import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.dcp.sm120_dsa import (
    SM120_DSA_LAYOUT,
    localize_sparse_indices,
    validate_sm120_dsa_dcp,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSM120DSADCP(CustomTestCase):
    def test_indexer_and_attention_agree_on_topk_address_domain(self):
        from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        backend = object.__new__(DeepseekSparseAttnBackend)
        backend.hisparse_coordinator = None
        backend.dsa_kv_cache_store_fp8 = True
        backend.dsa_prefill_impl = "flashinfer_sparse_mla"
        backend.dsa_topk_backend = None
        backend.forward_metadata = SimpleNamespace(
            paged_mqa_schedule_metadata=None, paged_mqa_ctx_lens_2d=None
        )
        for layout in (None, SM120_DSA_LAYOUT):
            backend.dcp_packed_kv_layout = layout
            for enabled in (False, True):
                backend.use_fused_topk = enabled
                for mode in (ForwardMode.EXTEND, ForwardMode.DECODE):
                    batch = SimpleNamespace(forward_mode=mode)
                    expected_fused = enabled and not (
                        layout is not None and mode == ForwardMode.EXTEND
                    )
                    self.assertEqual(
                        backend._use_fused_topk_for_batch(batch), expected_fused
                    )
                    self.assertEqual(
                        backend.get_indexer_metadata(0, batch).force_unfused_topk,
                        not expected_fused,
                    )

    def test_indices_use_translator_once_and_never_translate_padding(self):
        indices = torch.tensor(
            [[0, 1, 7, -1, 15, 12], [-1] * 6, [8, 6, 5, 3, 19, -1]], dtype=torch.int32
        )
        for size in (2, 4, 8):
            for rank in range(size):
                calls = []

                def translate(values):
                    self.assertTrue((values >= 0).all())
                    calls.append(values.clone())
                    return values // size

                result, lengths = localize_sparse_indices(
                    indices,
                    SimpleNamespace(translate_dcp_read_ids=translate),
                    size,
                    rank,
                )
                self.assertEqual(len(calls), 1)
                expected = []
                expected_lengths = []
                for row in indices.tolist():
                    values = [
                        value // size
                        for value in row
                        if value >= 0 and value % size == rank
                    ]
                    expected_lengths.append(len(values))
                    expected.append(values + [-1] * (indices.shape[1] - len(values)))
                self.assertEqual(result.tolist(), expected)
                self.assertEqual(lengths.tolist(), expected_lengths)

    def config(self):
        return SimpleNamespace(
            dcp_size=4,
            dsa_prefill_backend="flashinfer_sparse_mla",
            dsa_decode_backend="flashinfer_sparse_mla",
            kv_cache_dtype="fp8_e4m3",
            page_size=64,
            speculative_algorithm=None,
            speculative_eagle_topk=1,
            speculative_draft_model_path=None,
            speculative_draft_model_revision=None,
            revision=None,
            speculative_draft_attention_backend=None,
            speculative_draft_kv_cache_dtype=None,
            enable_multi_layer_eagle=False,
            speculative_adaptive=False,
            model_path="glm-native-mtp",
            enable_hisparse=False,
            enable_hierarchical_cache=False,
            enable_lmcache=False,
            enable_prefill_cp=False,
            enable_unified_memory=False,
            enable_two_batch_overlap=False,
            enable_mixed_chunk=False,
            enable_dp_attention=False,
            disaggregation_mode="null",
            dcp_comm_backend="ag_rs",
            dcp_replicate_q_proj=False,
            enable_page_major_kv_layout=False,
            enable_unified_cache_external_linker=False,
            pp_size=1,
        )

    def model(self):
        return SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM"],
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            index_topk=2048,
            index_kpool=1,
        )

    def test_supported_and_unsupported_combinations(self):
        validate_sm120_dsa_dcp(self.config(), self.model(), 12)
        for field, value in (
            ("speculative_algorithm", "EAGLE3"),
            ("enable_hisparse", True),
            ("enable_hierarchical_cache", True),
            ("enable_lmcache", True),
            ("enable_prefill_cp", True),
            ("enable_unified_memory", True),
            ("enable_two_batch_overlap", True),
            ("enable_mixed_chunk", True),
            ("enable_dp_attention", True),
            ("dsa_decode_backend", "trtllm"),
            ("kv_cache_dtype", "bf16"),
            ("page_size", 256),
            ("dcp_size", 16),
            ("disaggregation_mode", "decode"),
            ("dcp_comm_backend", "a2a"),
            ("dcp_replicate_q_proj", True),
            ("enable_page_major_kv_layout", True),
            ("enable_unified_cache_external_linker", True),
            ("pp_size", 2),
        ):
            with self.subTest(field=field):
                cfg = self.config()
                setattr(cfg, field, value)
                with self.assertRaises(ValueError):
                    validate_sm120_dsa_dcp(cfg, self.model(), 12)
        for field, value in (
            ("index_kpool", 2),
            ("index_topk", 128),
            ("kv_lora_rank", 448),
            ("architectures", ["DeepseekV3ForCausalLM"]),
        ):
            model = self.model()
            setattr(model, field, value)
            with self.assertRaises(ValueError):
                validate_sm120_dsa_dcp(self.config(), model, 12)
        with self.assertRaises(ValueError):
            validate_sm120_dsa_dcp(self.config(), self.model(), 10)

    def test_unrelated_paths_and_layout_are_unchanged(self):
        cfg = self.config()
        cfg.dcp_size = 1
        cfg.speculative_algorithm = "EAGLE"
        validate_sm120_dsa_dcp(cfg, self.model(), 12)
        cfg.dcp_size = 4
        cfg.dsa_prefill_backend = cfg.dsa_decode_backend = "trtllm"
        validate_sm120_dsa_dcp(cfg, self.model(), 10)
        with self.assertRaises(AttributeError):
            SM120_DSA_LAYOUT.bytes_per_token = 576

    def test_chain_mtp_scope(self):
        cfg = self.config()
        cfg.speculative_algorithm = "EAGLE"
        validate_sm120_dsa_dcp(cfg, self.model(), 12)
        for field, value in (
            ("speculative_eagle_topk", 2),
            ("speculative_draft_model_path", "another-model"),
            ("speculative_draft_model_revision", "another-revision"),
            ("speculative_draft_attention_backend", "trtllm_mla"),
            ("speculative_draft_kv_cache_dtype", "bf16"),
            ("enable_multi_layer_eagle", True),
            ("speculative_adaptive", True),
        ):
            with self.subTest(field=field):
                candidate = SimpleNamespace(**vars(cfg))
                setattr(candidate, field, value)
                with self.assertRaises(ValueError):
                    validate_sm120_dsa_dcp(candidate, self.model(), 12)


if __name__ == "__main__":
    unittest.main()
