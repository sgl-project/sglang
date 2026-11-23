import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.xpu.graph_runner.xpu_full_graph_backend import (
    FullXPUGraphBackend,
)
from sglang.srt.layers.attention.xpu_backend import XPUAttentionBackend
from sglang.srt.layers.attention.xpu_tree_verify import (
    tree_expand_attention,
    tree_expand_scratch_page,
    tree_verify_expand_metadata,
)
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=20, suite="extra-test-1-gpu-xpu")


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

    def test_full_xpu_graph_backend_retains_capture_inputs_and_cleans_up(self):
        from contextlib import nullcontext

        backend = FullXPUGraphBackend.__new__(FullXPUGraphBackend)
        backend._graphs = {}
        backend._outputs = {}
        backend._capture_inputs = {}
        backend._output_buffer = None
        backend._pool = None
        backend._reuse_output_buffer = False
        backend._cuda_graph_runner = SimpleNamespace(enable_profile_cuda_graph=False)
        backend._device_module = SimpleNamespace(
            synchronize=lambda: None,
            graph=lambda **kw: nullcontext(),
        )
        backend._tp_group = SimpleNamespace(barrier=lambda: None)
        backend._capture_stream = None

        reset_called = []
        mock_graph = SimpleNamespace(reset=lambda: reset_called.append(True))

        shape_key = 128
        dummy_input = SimpleNamespace(global_num_tokens_gpu=torch.tensor([128]))

        with patch("torch.xpu.XPUGraph", return_value=mock_graph):
            backend.capture_one(
                shape_key=shape_key,
                forward_fn=lambda: torch.tensor([1.0]),
                capture_inputs=dummy_input,
            )

        self.assertIs(backend._capture_inputs[shape_key], dummy_input)
        self.assertIn(shape_key, backend._graphs)

        backend.cleanup()
        self.assertEqual(len(backend._capture_inputs), 0)
        self.assertEqual(len(backend._graphs), 0)
        self.assertEqual(len(backend._outputs), 0)
        self.assertEqual(reset_called, [True])


_TREE_HQ, _TREE_HK, _TREE_D = 32, 8, 128
_TREE_PAGE_SIZE = 64


def _tree_inputs(device, prefix_lens, branches):
    """Two requests sharing one tree shape; ``branches[i]`` lists the draft
    columns query row i attends. Slots are a fixed permutation so a wrong
    gather cannot accidentally read the right KV."""
    num_draft = len(branches)
    num_slots = 4 * _TREE_PAGE_SIZE
    perm = torch.randperm(num_slots, generator=torch.Generator().manual_seed(0))
    width = max(prefix_lens) + num_draft
    req_to_token = torch.stack(
        [perm[r * width : (r + 1) * width] for r in range(len(prefix_lens))]
    ).to(device=device, dtype=torch.int32)
    rows = []
    for prefix_len in prefix_lens:
        for branch in branches:
            row = [True] * prefix_len + [j in branch for j in range(num_draft)]
            rows.append(torch.tensor(row))
    custom_mask = torch.cat(rows).to(device)
    seq_lens = torch.tensor(prefix_lens, device=device, dtype=torch.int64)
    req_pool_indices = torch.arange(len(prefix_lens), device=device)
    return req_to_token, custom_mask, seq_lens, req_pool_indices


def _reference(q, k_flat, v_flat, page_table, cache_seqlens, scale):
    rows, width = page_table.shape
    keys = k_flat[page_table.long()].float()
    values = v_flat[page_table.long()].float()
    scores = torch.einsum(
        "rhgd,rkhd->rhgk",
        q.float().view(rows, _TREE_HK, _TREE_HQ // _TREE_HK, _TREE_D),
        keys,
    )
    scores = scores * scale
    valid = torch.arange(width, device=q.device) < cache_seqlens[:, None]
    scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
    out = torch.einsum("rhgk,rkhd->rhgd", torch.softmax(scores, dim=-1), values)
    return out.reshape(rows, _TREE_HQ, _TREE_D), torch.logsumexp(scores, -1).reshape(
        rows, _TREE_HQ
    )


class TestXPUTreeVerifyExpand(CustomTestCase):
    """topk > 1 target-verify expand: each draft query row attends exactly
    its tree branch, and the scratch-page pass replays under XPU graphs."""

    def test_metadata_selects_each_rows_tree_branch(self):
        """Rows list exactly their branch slots first, also for the second
        request's offset mask rows and for a padded row past the real mask."""
        device = "xpu"
        branches = [[0], [0, 1], [0, 2], [0, 1, 3]]
        prefix_lens = [5, 9]
        req_to_token, custom_mask, seq_lens, req_pool_indices = _tree_inputs(
            device, prefix_lens, branches
        )
        page_table, cache_seqlens = tree_verify_expand_metadata(
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            custom_mask=custom_mask,
            num_draft_tokens=len(branches),
        )
        for req, prefix_len in enumerate(prefix_lens):
            for i, branch in enumerate(branches):
                row = req * len(branches) + i
                want = [int(req_to_token[req, prefix_len + j]) for j in branch]
                self.assertEqual(int(cache_seqlens[row]), len(branch))
                self.assertEqual(page_table[row, : len(branch)].tolist(), want)

        # Graph replay pads the batch but keeps the real (smaller) custom_mask.
        padded_seq_lens = torch.cat([seq_lens, seq_lens.new_ones(1)])
        padded_indices = torch.cat([req_pool_indices, req_pool_indices.new_zeros(1)])
        page_table, cache_seqlens = tree_verify_expand_metadata(
            req_to_token=req_to_token,
            req_pool_indices=padded_indices,
            seq_lens=padded_seq_lens,
            custom_mask=custom_mask,
            num_draft_tokens=len(branches),
        )
        real_rows = len(prefix_lens) * len(branches)
        self.assertEqual(
            cache_seqlens[:real_rows].tolist(),
            [len(b) for b in branches] * len(prefix_lens),
        )
        self.assertTrue(bool((cache_seqlens[real_rows:] >= 1).all()))

    def test_expand_attention_matches_reference_in_graph(self):
        """Eager and XPU-graph replay both match an fp32 reference (output
        and natural-log LSE, which merge_state_v2 consumes)."""
        device = torch.device("xpu")
        gen = torch.Generator(device=device).manual_seed(0)
        num_draft = 8
        rows = 3 * num_draft
        num_pages = 32
        k_cache = torch.randn(
            num_pages,
            _TREE_PAGE_SIZE,
            _TREE_HK,
            _TREE_D,
            device=device,
            dtype=torch.bfloat16,
            generator=gen,
        )
        v_cache = torch.randn_like(k_cache)
        k_flat = k_cache.view(-1, _TREE_HK, _TREE_D)
        v_flat = v_cache.view(-1, _TREE_HK, _TREE_D)

        def random_tree():
            slots = torch.stack(
                [
                    torch.randperm(
                        num_pages * _TREE_PAGE_SIZE, device=device, generator=gen
                    )[:num_draft]
                    for _ in range(rows)
                ]
            ).to(torch.int32)
            lens = torch.randint(
                1,
                num_draft + 1,
                (rows,),
                device=device,
                dtype=torch.int32,
                generator=gen,
            )
            return slots, lens

        q = torch.randn(
            rows, _TREE_HQ, _TREE_D, device=device, dtype=torch.bfloat16, generator=gen
        )
        page_table, cache_seqlens = random_tree()
        cu_seqlens_q = torch.arange(rows + 1, device=device, dtype=torch.int32)
        page = tree_expand_scratch_page(num_draft)
        scratch_k = torch.zeros(
            rows, page, _TREE_HK, _TREE_D, device=device, dtype=torch.bfloat16
        )
        scratch_v = torch.zeros_like(scratch_k)
        scratch_page_table = torch.arange(rows, device=device, dtype=torch.int32)[
            :, None
        ]
        scale = _TREE_D**-0.5

        def run():
            return tree_expand_attention(
                q=q,
                key_cache=k_cache,
                value_cache=v_cache,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                scratch_k=scratch_k,
                scratch_v=scratch_v,
                scratch_page_table=scratch_page_table,
                scale=scale,
                softcap=0.0,
                window_size=(-1, -1),
            )

        def check(out, lse):
            ref_out, ref_lse = _reference(
                q, k_flat, v_flat, page_table, cache_seqlens, scale
            )
            torch.testing.assert_close(out.float(), ref_out, atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(lse.T.float(), ref_lse, atol=1e-3, rtol=1e-3)

        out, lse, *_ = run()
        check(out, lse)

        stream = torch.xpu.Stream()
        with torch.xpu.stream(stream):
            run()
        torch.xpu.synchronize()
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph, stream=stream):
            graph_out, graph_lse, *_ = run()

        new_page_table, new_cache_seqlens = random_tree()
        page_table.copy_(new_page_table)
        cache_seqlens.copy_(new_cache_seqlens)
        q.copy_(torch.randn(q.shape, device=device, dtype=q.dtype, generator=gen))
        graph.replay()
        torch.xpu.synchronize()
        check(graph_out, graph_lse)


if __name__ == "__main__":
    unittest.main()
