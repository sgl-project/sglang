"""CPU unit tests for ROCm --enable-dsa-fp4-indexer on DCP1 (gfx950)."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsa import fp4_index_hip
from sglang.srt.layers.attention.dsa import dsa_indexer, dsa_indexer_hip
from sglang.srt.layers.attention.dsa.indexer_prologue_hip import (
    RocmNoPEPrologue,
    make_rocm_nope_prologue,
)
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.mem_cache.index_key_cache_hip import (
    Fp4IndexKeyCache,
    validate_fp4_index_cache_config,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_memory,
    get_parallel,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDsaFp4IndexerDcp1(CustomTestCase):
    def test_check_pool_dcp1_and_negative_gates(self):
        with (
            published_topology(),
            patch("sglang.srt.utils.is_gfx95_supported", return_value=True),
            patch(
                "sglang.srt.layers.attention.dsa.utils.aiter_can_use_preshuffle_paged_mqa",
                return_value=True,
            ),
            patch.object(fp4_index_hip, "aiter_has_fp4_mqa_logits", return_value=True),
            patch(
                "sglang.srt.layers.dcp.layout.dsa_shards_index_k",
                return_value=False,
            ),
            get_parallel().override(attn_dcp_size=1),
            get_memory().override(
                enable_unified_cache_external_linker=False,
                enable_hierarchical_cache=False,
                hicache_storage_backend=None,
                enable_flexkv=False,
                radix_cache_backend=None,
            ),
        ):
            fp4_index_hip.check_pool(dcp_replicated=False, index_kpool=1, page_size=64)
            with self.assertRaises(ValueError):
                fp4_index_hip.check_pool(
                    dcp_replicated=True, index_kpool=1, page_size=64
                )
            with self.assertRaises(ValueError):
                fp4_index_hip.check_pool(
                    dcp_replicated=False, index_kpool=1, page_size=16
                )
            with self.assertRaises(ValueError):
                fp4_index_hip.check_pool(
                    dcp_replicated=False, index_kpool=2, page_size=64
                )

            validate_fp4_index_cache_config()

            with get_memory().override(enable_unified_cache_external_linker=True):
                with self.assertRaisesRegex(ValueError, "external-linker"):
                    validate_fp4_index_cache_config()

            with get_memory().override(
                enable_hierarchical_cache=True,
                hicache_storage_backend="mooncake",
            ):
                with self.assertRaisesRegex(ValueError, "mooncake"):
                    validate_fp4_index_cache_config()

            with get_memory().override(
                enable_hierarchical_cache=True,
                hicache_storage_backend=None,
            ):
                with self.assertRaisesRegex(ValueError, "enable-hierarchical-cache"):
                    validate_fp4_index_cache_config()

            with get_memory().override(enable_flexkv=True):
                with self.assertRaisesRegex(ValueError, "FlexKV"):
                    validate_fp4_index_cache_config()

            with get_memory().override(radix_cache_backend="flexkv"):
                with self.assertRaisesRegex(ValueError, "FlexKV"):
                    validate_fp4_index_cache_config()

            # DCP>1 with dsa_shards_index_k preserves file-backed L3 HiCache.
            with (
                patch(
                    "sglang.srt.layers.dcp.layout.dsa_shards_index_k",
                    return_value=True,
                ),
                get_parallel().override(attn_dcp_size=8),
                get_memory().override(
                    enable_hierarchical_cache=True,
                    hicache_storage_backend="file",
                ),
            ):
                validate_fp4_index_cache_config()

            with patch.object(
                fp4_index_hip, "aiter_has_fp4_mqa_logits", return_value=False
            ):
                with self.assertRaises(ValueError):
                    fp4_index_hip.check_pool(
                        dcp_replicated=False, index_kpool=1, page_size=64
                    )

            with get_disagg().override(disaggregation_mode="prefill"):
                with self.assertRaisesRegex(ValueError, "PD disaggregation"):
                    validate_fp4_index_cache_config()

            with get_parallel().override(attn_dcp_size=8):
                with self.assertRaisesRegex(ValueError, "DCP-sharded"):
                    validate_fp4_index_cache_config()

    def test_init_indexer_hip_gates_incompatible_models_and_disables_fp8_paths(self):
        rope_indexer = SimpleNamespace(
            skip_rope=False,
            bf16_head_gate=False,
            head_dim=128,
            block_size=128,
            rocm_nope_prologue=None,
            use_prefill_lite_topk=True,
        )
        kimi_indexer = SimpleNamespace(
            skip_rope=True,
            bf16_head_gate=True,
            apply_hadamard=False,
            head_dim=128,
            block_size=128,
            n_heads=32,
            rocm_nope_prologue=None,
            use_prefill_lite_topk=True,
        )
        bad_heads_indexer = SimpleNamespace(
            skip_rope=True,
            bf16_head_gate=True,
            apply_hadamard=False,
            head_dim=128,
            block_size=128,
            n_heads=24,
            rocm_nope_prologue=None,
            use_prefill_lite_topk=True,
        )
        hadamard_indexer = SimpleNamespace(
            skip_rope=True,
            bf16_head_gate=True,
            apply_hadamard=True,
            head_dim=128,
            block_size=128,
            n_heads=32,
            rocm_nope_prologue=None,
            use_prefill_lite_topk=True,
        )
        with (
            published_topology(),
            patch.object(dsa_indexer_hip, "_is_gfx95", True),
            patch.object(dsa_indexer_hip, "_prefill_mqa_logits_fn", return_value=None),
            get_exec().kernel.override(enable_dsa_fp4_indexer=True),
        ):
            with self.assertRaisesRegex(ValueError, "NoPE DSA indexer"):
                dsa_indexer_hip.init_indexer_hip(rope_indexer, config=SimpleNamespace())
            with self.assertRaisesRegex(ValueError, "index_n_heads"):
                dsa_indexer_hip.init_indexer_hip(
                    bad_heads_indexer, config=SimpleNamespace()
                )
            with self.assertRaisesRegex(ValueError, "index_apply_hadamard"):
                dsa_indexer_hip.init_indexer_hip(
                    hadamard_indexer, config=SimpleNamespace()
                )

            with patch.object(
                dsa_indexer_hip,
                "hist_topk_workspace",
                side_effect=AssertionError("hist_topk_workspace must not run for FP4"),
            ):
                dsa_indexer_hip.init_indexer_hip(kimi_indexer, config=SimpleNamespace())
            self.assertIsInstance(kimi_indexer.rocm_nope_prologue, RocmNoPEPrologue)
            self.assertTrue(kimi_indexer.fp4_index_k)
            self.assertFalse(kimi_indexer.use_prefill_lite_topk)
            self.assertIsNone(kimi_indexer._hist_topk)

            self.assertTrue(kimi_indexer.rocm_nope_prologue.allow_draft_extend_v2)

            # DRAFT_EXTEND_V2 stays gated off for default FP8 and for DCP>1.
            fp8_kimi = SimpleNamespace(
                skip_rope=True,
                bf16_head_gate=True,
                head_dim=128,
                block_size=128,
                fp4_index_k=False,
            )
            dcp8_fp4_kimi = SimpleNamespace(
                skip_rope=True,
                bf16_head_gate=True,
                head_dim=128,
                block_size=128,
                fp4_index_k=True,
            )
            fb_de = SimpleNamespace(forward_mode=ForwardMode.DRAFT_EXTEND_V2)
            x_dummy = torch.zeros(2, 8, dtype=torch.bfloat16)
            with patch(
                "sglang.srt.layers.attention.dsa.indexer_prologue_hip.is_cp_active",
                return_value=False,
            ):
                with get_parallel().override(attn_dcp_size=1):
                    p_fp8 = make_rocm_nope_prologue(fp8_kimi)
                    self.assertFalse(p_fp8.applies(x_dummy, fb_de, False, False))
                with get_parallel().override(attn_dcp_size=8):
                    p_dcp8 = make_rocm_nope_prologue(dcp8_fp4_kimi)
                    self.assertFalse(p_dcp8.applies(x_dummy, fb_de, False, False))

    def test_fuse_index_k_norm_disabled_when_fp4_index_k_enabled(self):
        """Regression guard: FP8 index_k_norm_store must stay off when fp4_index_k is on."""
        indexer = SimpleNamespace(
            skip_rope=True,
            bf16_head_gate=True,
            fp4_index_k=True,
            use_dsa_indexer_fusion=False,
            use_aiter_fused_fp8_writer=False,
            head_dim=128,
            block_size=128,
            scale_fmt="ue8m0",
            k_norm=SimpleNamespace(
                use_bias=True,
                elementwise_affine=True,
                weight=torch.zeros(128, dtype=torch.float32),
                bias=torch.zeros(128, dtype=torch.float32),
            ),
        )
        fb = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_decode_or_idle=lambda: True,
                is_target_verify=lambda: False,
            )
        )
        self.assertFalse(dsa_indexer_hip.fuse_index_k_norm(indexer, fb))

        indexer.fp4_index_k = False
        pool = SimpleNamespace(index_page_size=64)
        attn_backend = SimpleNamespace(token_to_kv_pool=pool)
        with (
            published_topology(),
            forward_context(ForwardContext(attn_backend=attn_backend)),
            patch(
                "sglang.srt.layers.attention.dsa.dsa_indexer._index_k_dcp",
                return_value=(1, 0),
            ),
            patch(
                "sglang.srt.layers.cp.utils.is_cp_active",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.layernorm.LayerNorm",
                type(indexer.k_norm),
            ),
        ):
            self.assertTrue(dsa_indexer_hip.fuse_index_k_norm(indexer, fb))

    def test_forward_cuda_draft_extend_v2_routes_through_fp4_prologue(self):
        """End-to-end forward_cuda check: DRAFT_EXTEND_V2 uses FP4 prologue -> fp4_topk_paged."""
        indexer = object.__new__(dsa_indexer.Indexer)
        torch.nn.Module.__init__(indexer)
        indexer.skip_rope = True
        indexer.bf16_head_gate = True
        indexer.head_dim = 128
        indexer.block_size = 128
        indexer.n_heads = 32
        indexer.index_topk = 4
        indexer.softmax_scale = 128**-0.5
        indexer.scale_fmt = "ue8m0"
        indexer.fp4_index_k = True
        indexer.use_dsa_indexer_fusion = False
        indexer.use_aiter_fused_fp8_writer = False
        indexer.use_gfx950_fused_indexer = False
        indexer._wk_gate_weight = None
        indexer.dsa_enable_prefill_cp = False
        indexer.alt_stream = None
        indexer.num_init_tokens = 0
        indexer.num_local_tokens = 0

        indexer.wk = lambda x: (torch.ones(x.shape[0], 128, dtype=torch.bfloat16),)
        indexer.weights_proj = lambda x: (
            torch.ones(x.shape[0], 32, dtype=torch.bfloat16),
        )
        indexer.wq_b = lambda q_lora: (
            torch.ones(q_lora.shape[0], 32 * 128, dtype=torch.bfloat16),
        )
        indexer.k_norm = lambda k: k

        rows = 4
        x = torch.ones(rows, 16, dtype=torch.bfloat16)
        q_lora = torch.ones(rows, 8, dtype=torch.bfloat16)
        positions = torch.arange(rows, dtype=torch.int64)
        out_cache_loc = torch.arange(64, 64 + rows, dtype=torch.int64)
        seqlens_req = torch.tensor([256], dtype=torch.int32)
        seqlens_exp = torch.tensor([253, 254, 255, 256], dtype=torch.int32)
        block_tables = torch.zeros(rows, 4, dtype=torch.int32)

        fb = SimpleNamespace(
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            out_cache_loc=out_cache_loc,
            seq_lens=seqlens_req,
            seq_lens_cpu=seqlens_req,
            attn_cp_metadata=None,
        )
        observed = {}

        def fake_store_fp4(key, pay, sc, loc):
            observed["stored_key_shape"] = tuple(key.shape)
            observed["stored_loc"] = loc.clone()

        def fake_q_fp4(q, gate, scale):
            observed["q_fp4_input_shape"] = tuple(q.shape)
            return (
                fp4_index_hip.Fp4IndexQuery(
                    payload=torch.zeros(q.shape[0], 32, 64, dtype=torch.uint8),
                    scale=torch.zeros(q.shape[0], 4, 64, dtype=torch.uint8),
                ),
                torch.ones(q.shape[0], 32, 1, dtype=torch.float32),
            )

        def fake_logits(q, w, pay, sc, tables, starts, row_ends, max_len):
            self.assertIsInstance(q, fp4_index_hip.Fp4IndexQuery)
            observed["logits_query_type"] = type(q).__name__
            observed["logits_row_ends"] = row_ends.tolist()
            return torch.arange(q.shape[0] * max_len, dtype=torch.float32).view(
                q.shape[0], max_len
            )

        meta = SimpleNamespace(
            get_page_table_64=lambda: block_tables,
            get_seqlens_expanded=lambda: seqlens_exp,
            get_seqlens_int32=lambda: seqlens_req,
            get_dsa_extend_len_cpu=lambda: [rows],
            topk_transform=lambda logits, topk: torch.full(
                (logits.shape[0], topk), 11, dtype=torch.int32
            ),
        )
        pool = SimpleNamespace(
            index_page_size=64,
            get_index_k_fp4_buffers=lambda lid: (torch.zeros(1), torch.zeros(1)),
        )
        attn_backend = SimpleNamespace(
            token_to_kv_pool=pool,
            get_indexer_metadata=lambda lid, _fb: meta,
        )

        with (
            published_topology(),
            forward_context(ForwardContext(attn_backend=attn_backend)),
            patch.object(dsa_indexer, "_is_hip", True),
            patch.object(dsa_indexer, "_is_cuda", False),
            patch.object(dsa_indexer, "_use_aiter", True),
            patch.object(dsa_indexer, "dsa_indexer_hip", dsa_indexer_hip, create=True),
            patch(
                "sglang.srt.layers.cp.utils.is_cp_active",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.attention.dsa.indexer_prologue_hip.is_cp_active",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.attention.dsa.indexer_prologue_hip.indexer_q_fp4_head_gate",
                side_effect=fake_q_fp4,
            ),
            patch.object(
                fp4_index_hip, "store_fp4_index_k", side_effect=fake_store_fp4
            ),
            patch.object(
                fp4_index_hip, "fp4_paged_mqa_logits", side_effect=fake_logits
            ),
            get_parallel().override(attn_dcp_size=1, attn_dcp_rank=0),
        ):
            indexer.rocm_nope_prologue = make_rocm_nope_prologue(indexer)
            self.assertIsNotNone(indexer.rocm_nope_prologue)
            out = indexer.forward_cuda(x, q_lora, positions, fb, layer_id=2)

        self.assertEqual(tuple(out.shape), (rows, 4))
        self.assertTrue(torch.all(out == 11))
        self.assertEqual(observed["stored_key_shape"], (rows, 128))
        self.assertTrue(torch.equal(observed["stored_loc"], out_cache_loc))
        self.assertEqual(observed["q_fp4_input_shape"], (rows, 32, 128))
        self.assertEqual(observed["logits_query_type"], "Fp4IndexQuery")
        self.assertEqual(observed["logits_row_ends"], [253, 254, 255, 256])

        # LoRA weights_proj stays on the FP4 prologue and bypasses k_gate_weight.
        lora_calls = []

        class _LoraWeightsProj:
            set_lora = True

            def __call__(self, inp):
                lora_calls.append(tuple(inp.shape))
                return (torch.ones(inp.shape[0], 32, dtype=torch.bfloat16),)

        indexer.weights_proj = _LoraWeightsProj()
        indexer.rocm_nope_prologue.k_gate_weight = torch.zeros(
            160, 16, dtype=torch.bfloat16
        )
        fb_decode = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            out_cache_loc=out_cache_loc[:1],
            seq_lens=seqlens_req,
            seq_lens_cpu=seqlens_req,
            attn_cp_metadata=None,
        )
        meta_decode = SimpleNamespace(
            get_page_table_64=lambda: block_tables[:1],
            get_seqlens_expanded=lambda: seqlens_req,
            get_seqlens_int32=lambda: seqlens_req,
            get_dsa_extend_len_cpu=lambda: [1],
            topk_transform=lambda logits, topk: torch.full(
                (logits.shape[0], topk), 13, dtype=torch.int32
            ),
        )
        attn_backend_decode = SimpleNamespace(
            token_to_kv_pool=pool,
            get_indexer_metadata=lambda lid, _fb: meta_decode,
        )
        with (
            published_topology(),
            forward_context(ForwardContext(attn_backend=attn_backend_decode)),
            patch.object(dsa_indexer, "_is_hip", True),
            patch.object(dsa_indexer, "_is_cuda", False),
            patch.object(dsa_indexer, "_use_aiter", True),
            patch.object(dsa_indexer, "dsa_indexer_hip", dsa_indexer_hip, create=True),
            patch("sglang.srt.layers.cp.utils.is_cp_active", return_value=False),
            patch(
                "sglang.srt.layers.attention.dsa.indexer_prologue_hip.is_cp_active",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.attention.dsa.indexer_prologue_hip.indexer_q_fp4_head_gate",
                side_effect=fake_q_fp4,
            ),
            patch.object(
                fp4_index_hip, "store_fp4_index_k", side_effect=fake_store_fp4
            ),
            patch.object(
                fp4_index_hip, "fp4_paged_mqa_logits", side_effect=fake_logits
            ),
            get_parallel().override(attn_dcp_size=1, attn_dcp_rank=0),
        ):
            out_lora = indexer.forward_cuda(
                x[:1], q_lora[:1], positions[:1], fb_decode, layer_id=2
            )
        self.assertEqual(lora_calls, [(1, 16)])
        self.assertTrue(torch.all(out_lora == 13))

    def _run_fp4_topk_dcp1(
        self,
        *,
        total_rows: int,
        q_offset: int,
        num_reqs: int,
        mode_name: str,
        topk: int = 4,
        ext_lens=None,
        req_to_token_cols: int = 0,
    ):
        q = fp4_index_hip.Fp4IndexQuery(
            payload=torch.zeros(total_rows, 32, 64, dtype=torch.uint8),
            scale=torch.zeros(total_rows, 4, 64, dtype=torch.uint8),
        )
        weights = torch.ones(total_rows, 32, 1, dtype=torch.float32)
        block_tables = torch.zeros(max(q_offset, 1), 4, dtype=torch.int32)
        seqlens_exp = torch.full((max(q_offset, 1),), 128, dtype=torch.int32)
        seqlens_req = torch.full((num_reqs,), 128, dtype=torch.int32)
        observed = {}

        def fake_logits(q_in, w_in, pay, sc, tables, starts, row_ends, max_len):
            observed["q_rows"] = q_in.shape[0]
            observed["table_cols"] = tables.shape[1]
            observed["table_stride0"] = tables.stride(0)
            observed["max_len"] = max_len
            return torch.zeros(q_in.shape[0], max_len, dtype=torch.float32)

        def fake_transform(logits, k):
            observed["transform_shape"] = tuple(logits.shape)
            return torch.full((logits.shape[0], k), 7, dtype=torch.int32)

        indexer = SimpleNamespace(
            index_topk=topk,
            _mask_init_and_local_tokens=lambda logits, sl: observed.setdefault(
                "masked_len", sl.shape[0]
            ),
        )
        fb = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_target_verify=lambda: mode_name == "target_verify",
                is_draft_extend_v2=lambda: mode_name == "draft_extend_v2",
            )
        )
        if ext_lens is None:
            ext_lens = (
                [q_offset // num_reqs] * num_reqs if (q_offset and num_reqs) else []
            )
        meta = SimpleNamespace(
            get_page_table_64=lambda: block_tables,
            get_seqlens_expanded=lambda: seqlens_exp,
            get_seqlens_int32=lambda: seqlens_req,
            get_dsa_extend_len_cpu=lambda: ext_lens,
            topk_transform=fake_transform,
        )
        pool = SimpleNamespace(
            index_page_size=64,
            get_index_k_fp4_buffers=lambda lid: (torch.zeros(1), torch.zeros(1)),
        )
        if req_to_token_cols:
            dsa = object.__new__(DeepseekSparseAttnBackend)
            dsa.token_to_kv_pool = pool
            dsa.req_to_token = torch.zeros(1, req_to_token_cols, dtype=torch.int32)
            dsa.req_to_token_pool = SimpleNamespace(req_to_token=dsa.req_to_token)
            dsa.runner_ids = None
            dsa.needs_cpu_seq_lens = False
            dsa.real_page_span = 64
            dsa.decode_cuda_graph_metadata = {}
            linear = SimpleNamespace(
                needs_cpu_seq_lens=False,
                supports_prefill_graph_extend=lambda: False,
            )
            attn_backend = HybridLinearAttnBackend(dsa, linear, [0])
        else:
            attn_backend = SimpleNamespace(token_to_kv_pool=pool)
        with (
            published_topology(),
            forward_context(ForwardContext(attn_backend=attn_backend)),
            get_parallel().override(attn_dcp_size=1),
            patch.object(
                fp4_index_hip, "fp4_paged_mqa_logits", side_effect=fake_logits
            ),
        ):
            out = dsa_indexer_hip.fp4_topk_paged(indexer, fb, 3, q, weights, meta)
        return out, observed

    def test_fp4_topk_paged_dcp1_modes_and_padding(self):
        out, obs = self._run_fp4_topk_dcp1(
            total_rows=8, q_offset=8, num_reqs=1, mode_name="target_verify"
        )
        self.assertEqual(tuple(out.shape), (8, 4))
        self.assertEqual(obs["masked_len"], 8)
        self.assertEqual(obs["max_len"], 256)

        out_d, obs_d = self._run_fp4_topk_dcp1(
            total_rows=4,
            q_offset=4,
            num_reqs=4,
            mode_name="decode",
            req_to_token_cols=1024,
        )
        self.assertEqual(tuple(out_d.shape), (4, 4))
        self.assertEqual(obs_d["masked_len"], 4)
        self.assertEqual(obs_d["table_cols"], 16)
        self.assertEqual(obs_d["table_stride0"], 16)
        self.assertEqual(obs_d["max_len"], 256)

        # Single-request multi-row slice at full width must still normalize stride(0)==cols.
        out_s, obs_s = self._run_fp4_topk_dcp1(
            total_rows=8,
            q_offset=8,
            num_reqs=1,
            mode_name="target_verify",
            req_to_token_cols=256,
        )
        self.assertEqual(tuple(out_s.shape), (8, 4))
        self.assertEqual(obs_s["table_cols"], 4)
        self.assertEqual(obs_s["table_stride0"], 4)

        out_p, obs_p = self._run_fp4_topk_dcp1(
            total_rows=8, q_offset=6, num_reqs=1, mode_name="draft_extend_v2"
        )
        self.assertEqual(tuple(out_p.shape), (8, 4))
        self.assertEqual(obs_p["transform_shape"], (6, 256))
        self.assertTrue(torch.all(out_p[6:] == -1))

        out_e, obs_e = self._run_fp4_topk_dcp1(
            total_rows=4, q_offset=0, num_reqs=0, mode_name="decode"
        )
        self.assertEqual(tuple(out_e.shape), (4, 4))
        self.assertTrue(torch.all(out_e == -1))
        self.assertEqual(obs_e, {})

        with self.assertRaises(AssertionError):
            self._run_fp4_topk_dcp1(
                total_rows=5,
                q_offset=5,
                num_reqs=2,
                mode_name="target_verify",
                ext_lens=[3, 2],
            )

    def test_fp4_dcp_gather_index_k_dcp1_nonempty_and_empty(self):
        pool = SimpleNamespace(
            get_index_k_fp4_buffers=lambda lid: (torch.zeros(1), torch.zeros(1)),
        )
        attn_backend = SimpleNamespace(token_to_kv_pool=pool)
        for seq_list, want_total, want_max in (([128, 64], 192, 128), ([], 0, 1)):
            seq_lens_cpu = torch.tensor(seq_list, dtype=torch.int32)
            seq_lens_dev = seq_lens_cpu.clone()
            block_tables = torch.zeros(len(seq_list), 4, dtype=torch.int32)
            meta = SimpleNamespace(get_indexer_seq_len=lambda d=seq_lens_dev: d)

            def fake_gather(pay, sc, tbl, lens, total, max_len):
                self.assertTrue(torch.equal(lens, seq_lens_dev))
                self.assertEqual(total, want_total)
                self.assertEqual(max_len, want_max)
                return torch.full(
                    (total, fp4_index_hip.ROW_BYTES), 7, dtype=torch.uint8
                )

            def fake_stage(rows):
                pages = max((rows.shape[0] + 63) // 64, 1)
                return fp4_index_hip.Fp4IndexKeys(
                    payload=torch.zeros(
                        pages, fp4_index_hip.PAYLOAD_PAGE_BYTES, dtype=torch.uint8
                    ),
                    scale=torch.zeros(
                        pages, fp4_index_hip.SCALE_PAGE_BYTES, dtype=torch.uint8
                    ),
                    num_keys=rows.shape[0],
                )

            with (
                published_topology(),
                forward_context(ForwardContext(attn_backend=attn_backend)),
                get_parallel().override(attn_dcp_size=1, attn_dcp_rank=0),
                patch.object(
                    fp4_index_hip, "gather_fp4_index_k_rows", side_effect=fake_gather
                ),
                patch.object(
                    fp4_index_hip,
                    "stage_fp4_index_k_rows",
                    side_effect=(
                        fake_stage
                        if want_total
                        else fp4_index_hip.stage_fp4_index_k_rows
                    ),
                ),
                patch(
                    "sglang.srt.layers.attention.dsa.dsa_dcp_sharded_hip.dcp_gather_request_rows",
                    side_effect=AssertionError("must not call dcp_gather_request_rows"),
                ),
            ):
                res = dsa_indexer_hip.fp4_dcp_gather_index_k(
                    3, meta, block_tables, seq_lens_cpu
                )
            self.assertIsInstance(res, fp4_index_hip.Fp4IndexKeys)
            self.assertEqual(res.num_keys, want_total)

    def test_fp4_index_key_cache_move_preserves_payload_and_scale_with_overlap(self):
        cache = object.__new__(Fp4IndexKeyCache)
        cache.pool = SimpleNamespace(
            page_size=64,
            index_kpool=1,
            index_page_size=64,
            kv_cache_layout="layer_major",
        )
        payload = torch.zeros((2, 4096), dtype=torch.uint8)
        scale = torch.zeros((2, 256), dtype=torch.uint8)
        cache.buffer = [payload]
        cache.scale_buffer = [scale]

        # Write unique payload/scale patterns into tokens 0, 1, 63, 64, and neighbor 2.
        def write_tok(tok: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
            page, slot = tok // 64, tok % 64
            p_vals = (
                torch.arange(seed, seed + 64, dtype=torch.int64)
                .remainder(251)
                .to(torch.uint8)
            )
            s_vals = torch.tensor(
                [(seed + c * 17) % 251 for c in range(4)], dtype=torch.uint8
            )
            for b in range(64):
                payload[page, (b // 16) * 1024 + slot * 16 + (b % 16)] = p_vals[b]
            for c in range(4):
                scale[page, c * 64 + (slot % 16) * 4 + (slot // 16)] = s_vals[c]
            return p_vals, s_vals

        def read_tok(tok: int) -> tuple[torch.Tensor, torch.Tensor]:
            page, slot = tok // 64, tok % 64
            p_out = torch.tensor(
                [
                    int(payload[page, (b // 16) * 1024 + slot * 16 + (b % 16)])
                    for b in range(64)
                ],
                dtype=torch.uint8,
            )
            s_out = torch.tensor(
                [
                    int(scale[page, c * 64 + (slot % 16) * 4 + (slot // 16)])
                    for c in range(4)
                ],
                dtype=torch.uint8,
            )
            return p_out, s_out

        orig = {
            0: write_tok(0, 10),
            1: write_tok(1, 40),
            2: write_tok(2, 70),
            63: write_tok(63, 100),
            64: write_tok(64, 130),
        }
        src_loc = torch.tensor([0, 1, 63, 64], dtype=torch.int64)
        tgt_loc = torch.tensor([1, 63, 64, 65], dtype=torch.int64)

        with published_topology():
            with get_parallel().override(attn_dcp_size=1):
                cache.move(tgt_loc, src_loc)
            with get_parallel().override(attn_dcp_size=8):
                with self.assertRaises(AssertionError):
                    cache.move(tgt_loc, src_loc)

        for src_t, tgt_t in zip(src_loc.tolist(), tgt_loc.tolist()):
            got_p, got_s = read_tok(tgt_t)
            self.assertTrue(torch.equal(got_p, orig[src_t][0]))
            self.assertTrue(torch.equal(got_s, orig[src_t][1]))
        # Untouched neighbor slot 2 must remain intact.
        got_p2, got_s2 = read_tok(2)
        self.assertTrue(torch.equal(got_p2, orig[2][0]))
        self.assertTrue(torch.equal(got_s2, orig[2][1]))


if __name__ == "__main__":
    unittest.main()
