import gc
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.xpu.graph_runner.xpu_full_graph_backend import (
    FullXPUGraphBackend,
)
from sglang.srt.layers.attention.xpu_backend import XPUAttentionBackend
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=10, suite="stage-b-test-1-gpu-xpu")


class TestXPUGraphMetadata(CustomTestCase):
    def test_full_prefill_swa_page_table_with_multi_token_page_size_and_boundary_masking(
        self,
    ):
        """Verify that with page_size > 1 and variable request lengths:
        1. SWA page table indexes raw token slot IDs first before integer division.
        2. Boundary / inactive token slots (e.g. -1 or unallocated slots when
           seq_len < max_context_len) are safely masked before calling
           translate_loc_from_full_to_swa, preventing UB or out-of-bounds indexing.
        3. Inactive page table slots are cleanly zeroed out.
        """
        page_size = 16
        max_context_len = 64
        backend = XPUAttentionBackend.__new__(XPUAttentionBackend)
        backend.attention_chunk_size = None
        backend.use_sliding_window_kv_pool = True
        backend.page_size = page_size
        backend.max_context_len = max_context_len

        # Pool setup: 2 requests, stride indices at 0, 16, 32, 48.
        # Req 0 has length 32 (needs 2 pages: pos 0 and 16).
        # Req 1 has length 10 (needs 1 page: pos 0). Pos 16 has -1 (unallocated / dirty).
        req_to_token = torch.full((2, max_context_len), -1, dtype=torch.int32)
        req_to_token[0, 0] = 64  # Page 4
        req_to_token[0, 16] = 80  # Page 5
        req_to_token[1, 0] = 32  # Page 2
        # req_to_token[1, 16] remains -1 (dirty/unallocated)
        backend.req_to_token = req_to_token

        received_translation_inputs = []

        def mock_translate(token_slots):
            # Assert no -1 or negative slot IDs are passed to translation!
            self.assertTrue(
                torch.all(token_slots >= 0), f"Negative slot found: {token_slots}"
            )
            received_translation_inputs.append(token_slots.clone())
            return token_slots + 1000

        backend.token_to_kv_pool = SimpleNamespace(
            translate_loc_from_full_to_swa=mock_translate
        )

        forward_batch = SimpleNamespace(
            batch_size=2,
            seq_lens=torch.tensor([32, 10], dtype=torch.int32),
            extend_seq_lens=torch.tensor([32, 10], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([32, 10], dtype=torch.int32),
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            out_cache_loc=torch.arange(4, dtype=torch.int64),
            positions=torch.arange(4, dtype=torch.int64),
        )

        backend._init_full_cg_prefill_metadata(forward_batch, in_capture=True)

        # Check translation input: unallocated slot (req 1, page 1) was masked to 0
        token_slots_input = received_translation_inputs[0]
        expected_slots = torch.tensor([[64, 80], [32, 0]], dtype=torch.int32)
        self.assertTrue(torch.equal(token_slots_input, expected_slots))

        # Check full page_table: req 1 page 1 is zeroed
        expected_full_pages = torch.tensor([[4, 5], [2, 0]], dtype=torch.int32)
        self.assertTrue(
            torch.equal(
                backend.full_cg_prefill_metadata.page_table[:, :2],
                expected_full_pages,
            )
        )

        # Check SWA page_table: (expected_slots + 1000) // page_size with masked entry zeroed
        # req 0: 1064 // 16 = 66, 1080 // 16 = 67
        # req 1: 1032 // 16 = 64, page 1 masked to 0
        expected_swa_pages = torch.tensor([[66, 67], [64, 0]], dtype=torch.int32)
        self.assertTrue(
            torch.equal(
                backend.full_cg_prefill_metadata.swa_page_table[:, :2],
                expected_swa_pages,
            )
        )
        # During capture, swa_out_cache_loc is zeroed and max_seq_len_k is baked to max_context_len
        self.assertEqual(
            backend.full_cg_prefill_metadata.max_seq_len_k, max_context_len
        )
        self.assertTrue(
            torch.equal(
                backend.full_cg_prefill_metadata.swa_out_cache_loc,
                torch.zeros(4, dtype=torch.int64),
            )
        )

        # Replay (in_capture=False) with prefix hit (seq_lens != extend_seq_lens) and shorter out_cache_loc:
        # verifies separate cu_seqlens_q/k buffers, dynamic max_seq_len_k update, and swa_out_cache_loc translation + tail zeroing
        replay_batch = SimpleNamespace(
            batch_size=2,
            seq_lens=torch.tensor([32, 0], dtype=torch.int32),
            extend_seq_lens=torch.tensor([16, 0], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([32, 0], dtype=torch.int32),
            req_pool_indices=torch.tensor([0, 0], dtype=torch.int64),
            out_cache_loc=torch.tensor([10, 20], dtype=torch.int64),
            positions=torch.arange(2, dtype=torch.int64),
        )
        backend._init_full_cg_prefill_metadata(replay_batch, in_capture=False)
        m = backend.full_cg_prefill_metadata
        self.assertEqual(m.max_seq_len_k, 32)
        self.assertTrue(
            torch.equal(m.cu_seqlens_k, torch.tensor([0, 32, 32], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(m.cu_seqlens_q, torch.tensor([0, 16, 16], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(
                m.swa_out_cache_loc, torch.tensor([1010, 1020], dtype=torch.int64)
            )
        )
        self.assertTrue(
            torch.equal(
                backend.full_cg_prefill_swa_out_cache_loc[2:],
                torch.zeros(2, dtype=torch.int64),
            )
        )

    def test_forward_extend_hybrid_swa_and_irope_rejection(self):
        backend = XPUAttentionBackend.__new__(XPUAttentionBackend)
        backend.attention_chunk_size = 1024
        backend.dcp_size = 1
        backend.topk = 1

        # Case 1: layer with use_irope=True, but local_attn_metadata is None -> MUST raise
        metadata = SimpleNamespace(local_attn_metadata=None)
        backend.forward_metadata = metadata
        layer_irope = SimpleNamespace(
            use_irope=True,
            is_cross_attention=False,
            attn_type=None,
            sliding_window_size=-1,
        )
        forward_batch = SimpleNamespace(
            batch_size=1,
            out_cache_loc=None,
            encoder_out_cache_loc=None,
            forward_mode=SimpleNamespace(is_target_verify=lambda: False),
        )

        with self.assertRaisesRegex(
            ValueError,
            "Full prefill graph does not support models with attention_chunk_size and use_irope",
        ):
            backend.forward_extend(
                q=torch.zeros((1, 1, 16)),
                k=None,
                v=None,
                layer=layer_irope,
                forward_batch=forward_batch,
                save_kv_cache=False,
            )

    def test_full_prefill_rejects_attention_chunk_size(self):
        backend = XPUAttentionBackend.__new__(XPUAttentionBackend)
        backend.attention_chunk_size = 1024
        forward_batch = SimpleNamespace(
            batch_size=1,
            seq_lens=torch.tensor([16], dtype=torch.int32),
        )
        with self.assertRaisesRegex(
            ValueError,
            "Full prefill graph does not support models with attention_chunk_size",
        ):
            backend._init_full_cg_prefill_metadata(forward_batch, in_capture=True)

    def test_prefill_cuda_graph_runner_rejects_attention_chunk_size_under_full_graph(
        self,
    ):
        override = get_context().override_server_args(
            cuda_graph_config=CudaGraphConfig(
                prefill=PhaseConfig(
                    backend="full",
                    bs=[1],
                    max_context_size=64,
                    full_prefill_max_req=1,
                )
            ),
        )
        override.install()
        self.addCleanup(override.restore)

        for owner_name in ("model_runner", "model_config", "attn_backend"):
            with self.subTest(owner=owner_name):

                class _FakeModelRunner:
                    device = "cpu"
                    is_draft_worker = False
                    attention_chunk_size = 512 if owner_name == "model_runner" else None
                    attn_backend = SimpleNamespace(
                        attention_chunk_size=(
                            512 if owner_name == "attn_backend" else None
                        ),
                        dllm_attention=None,
                        prefill_cuda_graph_default_max_seq_len=None,
                    )
                    req_to_token_pool = SimpleNamespace(
                        size=16,
                        req_to_token=torch.zeros((16, 64), dtype=torch.int32),
                    )
                    model_config = SimpleNamespace(
                        is_multimodal=False,
                        hidden_size=64,
                        attention_chunk_size=(
                            512 if owner_name == "model_config" else None
                        ),
                        hf_config=SimpleNamespace(model_type=None),
                    )
                    model = SimpleNamespace(quant_config=None)
                    lora_manager = None
                    is_generation = True
                    spec_algorithm = SimpleNamespace(
                        is_eagle=lambda: False,
                        is_dflash_family=lambda: False,
                    )
                    dtype = torch.float16
                    pp_group = SimpleNamespace(is_first_rank=True)
                    get_pp_proxy_topk_size = lambda self: 0
                    get_pp_proxy_residual_num_blocks = lambda self: 0
                    get_pp_proxy_dspark_hidden_size = lambda self: 0
                    attn_tp_sequence_sharded = lambda self: False
                    attention_layers = []
                    mha_companion_layers = []
                    moe_layers = []
                    moe_fusions = []

                with (
                    patch(
                        "sglang.srt.model_executor.runner.prefill_cuda_graph_runner.resolve_prefill_backend",
                        return_value=FullXPUGraphBackend.__new__(FullXPUGraphBackend),
                    ),
                    self.assertRaisesRegex(
                        ValueError,
                        "Full prefill XPU graph does not support models with "
                        "attention_chunk_size or use_irope",
                    ),
                ):
                    PrefillCudaGraphRunner(_FakeModelRunner())

    def test_sgl_kernel_scratchpad_realloc_safety_under_graph(self):
        """Verify that sgl_kernel internal scratchpad buffers (MoE and GDN)
        are graph-safe, memory-efficient, and fully compatible:
        1. Capture state (stream capturing): all graphs and layers share a single
           dedicated graph workspace buffer (reused via narrow slice).
        2. Capture expansion (small -> large capture): retired buffers are retained
           so earlier captured graph pointers remain valid.
        3. GDN workspace coverage: small -> large capture retains retired GDN buffers.
        4. Real GPU replay test: verifies that graph execution accurately reads and writes
           tensors through the shared workspace and matches eager ground truth.
        5. Multi-instance reference counting: tracks active graph owners so that cleaning up
           one backend instance does not invalidate scratchpads still needed by other runners.
        6. Dual compatibility bypass: _XPU_GRAPH_SAFE on one module does not block patching
           on other modules.
        """
        import sglang.srt.hardware_backend.xpu.xpu_scratchpad_patch as xpu_sp_patch
        from sglang.srt.hardware_backend.xpu.xpu_scratchpad_patch import (
            _ACTIVE_GRAPH_FINALIZERS,
            _GRAPH_SCRATCHPAD_STATE,
            _HAS_GDN_WS_CUSTOM_OP,
            _HAS_MOE_WS_CUSTOM_OP,
            patch_sgl_kernel_scratchpad_buffers,
            register_graph_scratchpad_owner,
            reset_sgl_kernel_graph_scratchpads,
            safe_get_gdn_workspace,
            safe_get_moe_ws,
            unregister_graph_scratchpad_owner,
        )

        # Isolate active finalizer / scratchpad state from prior tests in this process.
        gc.collect()
        saved_finalizers = dict(_ACTIVE_GRAPH_FINALIZERS)
        _ACTIVE_GRAPH_FINALIZERS.clear()
        reset_sgl_kernel_graph_scratchpads()

        def _restore_scratchpad_test_state():
            reset_sgl_kernel_graph_scratchpads()
            _ACTIVE_GRAPH_FINALIZERS.clear()
            _ACTIVE_GRAPH_FINALIZERS.update(saved_finalizers)

        self.addCleanup(_restore_scratchpad_test_state)
        patch_sgl_kernel_scratchpad_buffers()

        # Verify custom ops are registered and infer symbolic schemas (SymInt[] / SymInt)
        self.assertTrue(_HAS_MOE_WS_CUSTOM_OP)
        self.assertTrue(_HAS_GDN_WS_CUSTOM_OP)
        self.assertIn("SymInt[]", str(torch.ops.sgl_xpu.get_moe_ws_op.default._schema))
        self.assertIn(
            "SymInt nbytes", str(torch.ops.sgl_xpu.get_gdn_ws_op.default._schema)
        )

        try:
            import sgl_kernel.gdn_attn as sk_gdn
            import sgl_kernel.moe as sk_moe
        except ImportError:
            self.skipTest("sgl_kernel not available")

        if not (torch.xpu.is_available() and hasattr(torch.xpu, "XPUGraph")):
            self.skipTest("torch.xpu.XPUGraph not available")

        device = torch.device("xpu:0")

        # 1. Under graph capture (large to small): all graphs share ONE workspace
        g4 = torch.xpu.XPUGraph()
        with torch.xpu.graph(g4):
            buf_g4 = sk_moe._get_moe_ws(
                "test_shared_ws", (4, 128), torch.bfloat16, device
            )

        g1 = torch.xpu.XPUGraph()
        with torch.xpu.graph(g1):
            buf_g1 = sk_moe._get_moe_ws(
                "test_shared_ws", (1, 128), torch.bfloat16, device
            )

        # Both graphs must share the exact same underlying buffer pointer
        self.assertEqual(buf_g4.data_ptr(), buf_g1.data_ptr())

        # 2. Under graph capture (small to large): expansion retains retired buffer
        g_expand_small = torch.xpu.XPUGraph()
        with torch.xpu.graph(g_expand_small):
            buf_small = sk_moe._get_moe_ws(
                "test_expand_ws", (1, 64), torch.bfloat16, device
            )
        ptr_small = buf_small.data_ptr()

        g_expand_large = torch.xpu.XPUGraph()
        with torch.xpu.graph(g_expand_large):
            buf_large = sk_moe._get_moe_ws(
                "test_expand_ws", (16, 64), torch.bfloat16, device
            )
        ptr_large = buf_large.data_ptr()
        if not getattr(sk_moe, "_XPU_GRAPH_SAFE", False):
            self.assertNotEqual(ptr_small, ptr_large)
            retired_moe = _GRAPH_SCRATCHPAD_STATE["moe_retired"]
            self.assertTrue(any(b.data_ptr() == ptr_small for b in retired_moe))

        # 3. GDN workspace expansion retains retired GDN buffers
        g_gdn1 = torch.xpu.XPUGraph()
        with torch.xpu.graph(g_gdn1):
            gdn_ws1 = sk_gdn._get_gdn_workspace(1024, device)
        ptr_gdn1 = gdn_ws1.data_ptr()

        g_gdn2 = torch.xpu.XPUGraph()
        with torch.xpu.graph(g_gdn2):
            gdn_ws2 = sk_gdn._get_gdn_workspace(8192, device)
        ptr_gdn2 = gdn_ws2.data_ptr()
        if not getattr(sk_gdn, "_XPU_GRAPH_SAFE", False):
            self.assertNotEqual(ptr_gdn1, ptr_gdn2)
            retired_gdn = _GRAPH_SCRATCHPAD_STATE["gdn_retired"]
            self.assertTrue(any(b.data_ptr() == ptr_gdn1 for b in retired_gdn))

        # 4. Real GPU execution & replay correctness under expansion + eager shock
        from sgl_kernel import fused_experts

        E, hidden, inter, topk = 8, 256, 512, 2
        torch.manual_seed(42)

        # Test M=1 input and ground truth
        a_in1 = torch.randn((1, hidden), dtype=torch.bfloat16, device=device)
        w1 = torch.randn((E, 2 * inter, hidden), dtype=torch.bfloat16, device=device)
        w2 = torch.randn((E, hidden, inter), dtype=torch.bfloat16, device=device)
        score1 = torch.softmax(
            torch.randn((1, E), dtype=torch.float32, device=device), dim=-1
        )
        topk_w1, topk_i1 = torch.topk(score1, topk)
        expected_out1 = fused_experts(a_in1, w1, w2, topk_w1, topk_i1.int()).clone()

        # Capture M=1 small graph
        replay_graph1 = torch.xpu.XPUGraph()
        with torch.xpu.graph(replay_graph1):
            graph_out1 = fused_experts(a_in1, w1, w2, topk_w1, topk_i1.int())

        # Test M=16 input and ground truth (triggers capture expansion)
        a_in16 = torch.randn((16, hidden), dtype=torch.bfloat16, device=device)
        score16 = torch.softmax(
            torch.randn((16, E), dtype=torch.float32, device=device), dim=-1
        )
        topk_w16, topk_i16 = torch.topk(score16, topk)
        expected_out16 = fused_experts(a_in16, w1, w2, topk_w16, topk_i16.int()).clone()

        # Capture M=16 expanded graph
        replay_graph16 = torch.xpu.XPUGraph()
        with torch.xpu.graph(replay_graph16):
            graph_out16 = fused_experts(a_in16, w1, w2, topk_w16, topk_i16.int())

        # Simulate large eager allocation (512 tokens)
        a_eager = torch.randn((512, hidden), dtype=torch.bfloat16, device=device)
        score_e = torch.softmax(
            torch.randn((512, E), dtype=torch.float32, device=device), dim=-1
        )
        topk_we, topk_ie = torch.topk(score_e, topk)
        fused_experts(a_eager, w1, w2, topk_we, topk_ie.int())

        # Replay both graphs and assert numeric correctness
        replay_graph1.replay()
        torch.xpu.synchronize()
        self.assertTrue(torch.allclose(graph_out1, expected_out1, atol=1e-2, rtol=1e-2))

        replay_graph16.replay()
        torch.xpu.synchronize()
        self.assertTrue(
            torch.allclose(graph_out16, expected_out16, atol=1e-2, rtol=1e-2)
        )

        # Verify dynamic torch.compile fullgraph execution with SymInt shapes
        def _compiled_ws_fn(x: torch.Tensor) -> int:
            m = x.shape[0]
            moe_ws = sk_moe._get_moe_ws("test_dyn_ws", (m * 2, 16), x.dtype, x.device)
            gdn_ws = sk_gdn._get_gdn_workspace(m * 64, x.device)
            return moe_ws.shape[0] + gdn_ws.shape[0]

        compiled_ws = torch.compile(_compiled_ws_fn, dynamic=True, fullgraph=True)
        self.assertEqual(
            compiled_ws(torch.empty((4, 8), dtype=torch.bfloat16, device=device)),
            4 * 2 + 4 * 64,
        )
        self.assertEqual(
            compiled_ws(torch.empty((7, 8), dtype=torch.bfloat16, device=device)),
            7 * 2 + 7 * 64,
        )

        # Verify torch.compile + XPUGraph joint capture & replay across 2 back-to-back layers
        def _two_layer_compiled_moe(x, w1_p, w2_p, tw, ti):
            h = fused_experts(x, w1_p, w2_p, tw, ti)
            return fused_experts(h, w1_p, w2_p, tw, ti)

        compiled_two_layer = torch.compile(_two_layer_compiled_moe, dynamic=False)
        expected_joint = compiled_two_layer(
            a_in16, w1, w2, topk_w16, topk_i16.int()
        ).clone()

        joint_graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(joint_graph):
            joint_out = compiled_two_layer(a_in16, w1, w2, topk_w16, topk_i16.int())

        # Trigger large eager compiled run to ensure graph workspace stays isolated
        compiled_two_layer(a_eager, w1, w2, topk_we, topk_ie.int())

        joint_graph.replay()
        torch.xpu.synchronize()
        self.assertTrue(torch.allclose(joint_out, expected_joint, atol=1e-2, rtol=1e-2))

        # 5. Multi-instance reference counting (explicit unregister + GC finalizer)
        class DummyOwner:
            pass

        owner1 = DummyOwner()
        owner2 = DummyOwner()
        register_graph_scratchpad_owner(owner1)
        register_graph_scratchpad_owner(owner2)
        self.assertIn(id(owner1), _ACTIVE_GRAPH_FINALIZERS)
        self.assertIn(id(owner2), _ACTIVE_GRAPH_FINALIZERS)

        # Discarding one owner does NOT clear containers
        unregister_graph_scratchpad_owner(owner1)
        self.assertNotIn(id(owner1), _ACTIVE_GRAPH_FINALIZERS)
        self.assertIn(id(owner2), _ACTIVE_GRAPH_FINALIZERS)
        if not getattr(sk_moe, "_XPU_GRAPH_SAFE", False):
            self.assertGreater(len(_GRAPH_SCRATCHPAD_STATE["moe_cache"]), 0)

        # Discarding all owners triggers reset
        unregister_graph_scratchpad_owner(owner2)
        self.assertEqual(len(_ACTIVE_GRAPH_FINALIZERS), 0)
        self.assertTrue(all(len(c) == 0 for c in _GRAPH_SCRATCHPAD_STATE.values()))

        # Implicit GC of last owner (without explicit unregister) also triggers reset
        owner3 = DummyOwner()
        register_graph_scratchpad_owner(owner3)
        g_gc = torch.xpu.XPUGraph()
        with torch.xpu.graph(g_gc):
            sk_moe._get_moe_ws("test_gc_ws", (2, 32), torch.bfloat16, device)
        if not getattr(sk_moe, "_XPU_GRAPH_SAFE", False):
            self.assertGreater(len(_GRAPH_SCRATCHPAD_STATE["moe_cache"]), 0)
        del owner3
        gc.collect()
        self.assertEqual(len(_ACTIVE_GRAPH_FINALIZERS), 0)
        self.assertTrue(all(len(c) == 0 for c in _GRAPH_SCRATCHPAD_STATE.values()))

        # 6. Dual compatibility bypass & self-patch recursion guard
        orig_moe_fn = lambda *a, **kw: None
        orig_gdn_fn = lambda *a, **kw: None
        mock_moe = SimpleNamespace(
            _XPU_GRAPH_SAFE=True,
            _get_moe_ws=orig_moe_fn,
        )
        mock_gdn = SimpleNamespace(
            _XPU_GRAPH_SAFE=False,
            _get_gdn_workspace=orig_gdn_fn,
        )
        saved_orig_moe = xpu_sp_patch._orig_moe_ws
        saved_orig_gdn = xpu_sp_patch._orig_gdn_ws
        import sgl_kernel

        try:
            with (
                patch.dict(
                    sys.modules,
                    {
                        "sgl_kernel.moe": mock_moe,
                        "sgl_kernel.gdn_attn": mock_gdn,
                    },
                ),
                patch.object(sgl_kernel, "moe", mock_moe),
                patch.object(sgl_kernel, "gdn_attn", mock_gdn),
            ):
                patch_sgl_kernel_scratchpad_buffers()
                # _XPU_GRAPH_SAFE=True on moe leaves it untouched without blocking gdn
                self.assertIs(mock_moe._get_moe_ws, orig_moe_fn)
                self.assertFalse(getattr(mock_moe, "_xpu_graph_safe_ws_patched", False))
                self.assertIs(mock_gdn._get_gdn_workspace, safe_get_gdn_workspace)
                self.assertTrue(getattr(mock_gdn, "_xpu_graph_safe_ws_patched", False))

                # Self-patch guard: resetting _xpu_graph_safe_ws_patched must not bind
                # _orig_gdn_ws to safe_get_gdn_workspace
                mock_gdn._xpu_graph_safe_ws_patched = False
                patch_sgl_kernel_scratchpad_buffers()
                self.assertIs(xpu_sp_patch._orig_gdn_ws, orig_gdn_fn)
                self.assertIsNot(xpu_sp_patch._orig_gdn_ws, safe_get_gdn_workspace)
                self.assertIsNot(xpu_sp_patch._orig_moe_ws, safe_get_moe_ws)
        finally:
            xpu_sp_patch._orig_moe_ws = saved_orig_moe
            xpu_sp_patch._orig_gdn_ws = saved_orig_gdn


if __name__ == "__main__":
    unittest.main()
