"""SM120/SM121 (consumer Blackwell) native sparse-MLA kernel validation.

`DeepseekSparseAttnBackend._forward_trtllm` routes the dsa "trtllm" backend to
flashinfer's native sparse-MLA kernel on SM12x (`backend="auto"`, packed
uint8 KV view, `kv_scale_format="arbitrary_fp32"`) instead of the
datacenter-only "trtllm-gen" kernel. This test builds a KV pool with
sglang's REAL production `quantize_k_cache` (the exact 656-byte packed
inline-scale layout the runtime writes: 512 fp8 nope + 4x fp32 tile scales +
64 bf16 rope) and calls flashinfer's dispatcher exactly as `_forward_trtllm`
does, comparing against a torch dequant (sglang's real `dequantize_k_cache_
paged`) + masked-softmax reference.

Requires SM120/SM121 hardware and flashinfer >= 0.6.x (ships
`flashinfer/mla/_sparse_mla_sm120.py`); skipped everywhere else. Not
exercised by CI (no SM120/121 runner today) -- validate manually:

    pytest test/registered/kernels/test_dsa_sparse_mla_sm12x.py -v

See UPSTREAM_SGLANG_DSA_SM12X_NATIVE_SPARSE_MLA.md for the live-deploy
numbers this test's tolerances are based on (decode max|diff| 0.008 @
topk=2048, prefill max|diff| 0.016 @ 2400 extend tokens).
"""

from __future__ import annotations

import unittest

import torch

from sglang.kernels.ops.attention.dsa.dequant_k_cache import dequantize_k_cache_paged
from sglang.kernels.ops.attention.dsa.quant_k_cache import quantize_k_cache
from sglang.kernels.ops.attention.dsa.transform_index import (
    transform_index_page_table_decode,
    transform_index_page_table_prefill,
)
from sglang.srt.utils import is_sm120_supported
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

DIM_NOPE = 512  # == kv_lora_rank
DIM_ROPE = 64  # == qk_rope_head_dim
DIM_TOTAL = DIM_NOPE + DIM_ROPE  # 576, per-token bf16 k layout before packing
TOPK = 2048  # transform_index_page_table_*'s kernels hardcode TOPK=2048
NUM_HEADS = 16  # matches the TP4/GB10 live-deploy head count


def _build_kv_pool(
    num_kv: int, *, device, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random bf16 K (nope+rope) -> sglang's real packed pool, plus the
    exact reference dequant (also sglang's real production kernel) split
    back into nope/rope for the torch attention reference."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    k_bf16 = (
        torch.randn((num_kv, 1, 1, DIM_TOTAL), generator=g, dtype=torch.float32)
        .to(device)
        .to(torch.bfloat16)
    )
    packed_pool = quantize_k_cache(k_bf16)  # (num_kv, 1, 1, 656) fp8 view

    page_table_1_flat = torch.arange(num_kv, dtype=torch.int32, device=device)
    dequant = dequantize_k_cache_paged(
        packed_pool, page_table_1_flat
    )  # (num_kv, 1, 576) bf16
    k_nope_ref = dequant[:, 0, :DIM_NOPE].float()
    k_rope_ref = dequant[:, 0, DIM_NOPE:].float()
    return packed_pool, k_nope_ref, k_rope_ref


def _build_topk_indices(
    qo_len: int, seq_lens: list[int], *, topk: int, device, seed: int
) -> torch.Tensor:
    """Per-query selected KV positions, -1 padded. Mirrors the shape the DSA
    indexer produces: at most `min(seq_len, topk)` valid entries per row."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    result = torch.full((qo_len, topk), -1, dtype=torch.int32)
    for i in range(qo_len):
        n_valid = min(seq_lens[i], topk)
        chosen = torch.randperm(seq_lens[i], generator=g)[:n_valid]
        result[i, :n_valid] = chosen.to(torch.int32)
    return result.to(device)


def _torch_reference(
    q_nope: torch.Tensor,  # (qo_len, num_heads, DIM_NOPE)
    q_rope: torch.Tensor,  # (qo_len, num_heads, DIM_ROPE)
    k_nope_ref: torch.Tensor,  # (num_kv, DIM_NOPE)
    k_rope_ref: torch.Tensor,  # (num_kv, DIM_ROPE)
    topk_indices: torch.Tensor,  # (qo_len, topk), -1 padded
    bmm1_scale: float,
) -> torch.Tensor:
    """Masked-softmax MLA attention over the topk-selected KV positions only
    (V = the nope/kv_lora_rank portion, matching MLA's absorbed-V design).
    -1 entries are excluded from softmax entirely (native padding skip)."""
    qo_len, topk = topk_indices.shape
    valid = topk_indices >= 0  # (qo_len, topk)
    gather_idx = topk_indices.clamp(min=0).long()  # (qo_len, topk)

    gathered_nope = k_nope_ref[gather_idx]  # (qo_len, topk, DIM_NOPE)
    gathered_rope = k_rope_ref[gather_idx]  # (qo_len, topk, DIM_ROPE)

    # (qo_len, num_heads, topk)
    scores = torch.einsum("qhd,qkd->qhk", q_nope.float(), gathered_nope)
    scores = scores + torch.einsum("qhd,qkd->qhk", q_rope.float(), gathered_rope)
    scores = scores * bmm1_scale
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))

    weights = torch.softmax(scores, dim=-1)  # (qo_len, num_heads, topk)
    out = torch.einsum(
        "qhk,qkd->qhd", weights, gathered_nope
    )  # (qo_len, num_heads, DIM_NOPE)
    return out


@unittest.skipUnless(
    is_sm120_supported(), "Test requires SM120/SM121 (consumer Blackwell)"
)
class TestDsaSparseMlaSM12x(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda")
        import flashinfer.decode

        cls.flashinfer_decode = flashinfer.decode
        cls.workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=cls.device
        )

    def _run_case(
        self,
        *,
        seq_lens: list[int],
        num_kv: int,
        is_prefill: bool,
        extend_lens: list[int] | None = None,
        seed: int = 0,
    ):
        qo_len = len(seq_lens)
        packed_pool, k_nope_ref, k_rope_ref = _build_kv_pool(
            num_kv, device=self.device, seed=seed
        )

        g = torch.Generator(device="cpu").manual_seed(seed + 1)
        q_nope = (
            torch.randn((qo_len, NUM_HEADS, DIM_NOPE), generator=g, dtype=torch.float32)
            .to(self.device)
            .to(torch.bfloat16)
        )
        q_rope = (
            torch.randn((qo_len, NUM_HEADS, DIM_ROPE), generator=g, dtype=torch.float32)
            .to(self.device)
            .to(torch.bfloat16)
        )

        topk_indices = _build_topk_indices(
            qo_len, seq_lens, topk=TOPK, device=self.device, seed=seed + 2
        )

        # page_table: identity mapping, logical position i == physical kv slot i.
        page_table = torch.arange(num_kv, dtype=torch.int32, device=self.device)
        if is_prefill:
            assert extend_lens is not None
            page_table_2d = page_table.unsqueeze(0).expand(len(extend_lens), -1)
            block_tables_1 = transform_index_page_table_prefill(
                page_table=page_table_2d,
                topk_indices=topk_indices,
                extend_lens_cpu=extend_lens,
                page_size=1,
                output_num_tokens=qo_len,
                page_table_is_expanded=False,
            )
        else:
            page_table_2d = page_table.unsqueeze(0).expand(qo_len, -1)
            block_tables_1 = transform_index_page_table_decode(
                page_table=page_table_2d, topk_indices=topk_indices, page_size=1
            )
        block_tables = block_tables_1.unsqueeze(1)  # (qo_len, 1, topk)

        bmm1_scale = 1.0 / (DIM_TOTAL**0.5)
        q = torch.cat([q_nope, q_rope], dim=-1).view(qo_len, 1, NUM_HEADS, DIM_TOTAL)
        # Real page geometry (num_pages, 1, 64, 656) like sglang's
        # _forward_trtllm: page_block_size==64 is REQUIRED for the dedicated
        # decode kernels to be dispatchable; a page-size-1 view mis-routes
        # decode batches into the prefill orchestrator (num_tokens>64 check).
        # Token-slot indices in block_tables stay valid: the binding derives
        # (page, offset) from the tensor geometry.
        kv = packed_pool.view(torch.uint8).reshape(-1, 64, 656).unsqueeze(1)
        seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=self.device)

        out = self.flashinfer_decode.trtllm_batch_decode_with_kv_cache_mla(
            query=q,
            kv_cache=kv,
            workspace_buffer=self.workspace_buffer,
            qk_nope_head_dim=DIM_NOPE,
            kv_lora_rank=DIM_NOPE,
            qk_rope_head_dim=DIM_ROPE,
            block_tables=block_tables,
            seq_lens=seq_lens_t,
            max_seq_len=num_kv,
            sparse_mla_top_k=TOPK,
            bmm1_scale=bmm1_scale,
            backend="auto",
            kv_scale_format="arbitrary_fp32",
            skip_softmax_threshold_scale_factor=None,
        )
        out = out.view(qo_len, NUM_HEADS, DIM_NOPE)

        ref = _torch_reference(
            q_nope, q_rope, k_nope_ref, k_rope_ref, topk_indices, bmm1_scale
        )
        torch.testing.assert_close(
            out.float(), ref.float(), atol=0.05, rtol=0.05, equal_nan=False
        )

    def test_decode_small_batch(self):
        # bs=4, seq_len well under topk -- exercises the -1-padding skip.
        # num_kv must be a multiple of 64 (real page geometry, see _run_case).
        self._run_case(seq_lens=[100, 100, 100, 100], num_kv=128, is_prefill=False)

    def test_decode_seq_len_exceeds_topk(self):
        # seq_len (4096) > topk (2048): sglang passes the UNCLAMPED
        # cache_seqlens for decode -- the kernel must clamp internally.
        self._run_case(seq_lens=[4096, 4096], num_kv=4096, is_prefill=False)

    def test_decode_variable_seq_lens(self):
        self._run_case(seq_lens=[50, 500, 2500, 4096], num_kv=4096, is_prefill=False)

    def test_prefill_large_extend(self):
        # >64 extend tokens: routes into flashinfer's prefill orchestrator
        # rather than the <=64-token warp-specialized decode kernels.
        extend_lens = [96]
        qo_len = sum(extend_lens)
        self._run_case(
            seq_lens=[3000] * qo_len,
            num_kv=4096,
            is_prefill=True,
            extend_lens=extend_lens,
        )

    def test_graph_capture_and_replay(self):
        seq_lens = [128, 128]
        num_kv = 4096
        qo_len = len(seq_lens)
        packed_pool, k_nope_ref, k_rope_ref = _build_kv_pool(
            num_kv, device=self.device, seed=42
        )
        g = torch.Generator(device="cpu").manual_seed(43)
        q_nope = (
            torch.randn((qo_len, NUM_HEADS, DIM_NOPE), generator=g, dtype=torch.float32)
            .to(self.device)
            .to(torch.bfloat16)
        )
        q_rope = (
            torch.randn((qo_len, NUM_HEADS, DIM_ROPE), generator=g, dtype=torch.float32)
            .to(self.device)
            .to(torch.bfloat16)
        )
        topk_indices = _build_topk_indices(
            qo_len, seq_lens, topk=TOPK, device=self.device, seed=44
        )
        page_table = torch.arange(num_kv, dtype=torch.int32, device=self.device)
        page_table_2d = page_table.unsqueeze(0).expand(qo_len, -1)
        block_tables = transform_index_page_table_decode(
            page_table=page_table_2d, topk_indices=topk_indices, page_size=1
        ).unsqueeze(1)

        bmm1_scale = 1.0 / (DIM_TOTAL**0.5)
        q = torch.cat([q_nope, q_rope], dim=-1).view(qo_len, 1, NUM_HEADS, DIM_TOTAL)
        # Real page geometry (num_pages, 1, 64, 656) like sglang's
        # _forward_trtllm: page_block_size==64 is REQUIRED for the dedicated
        # decode kernels to be dispatchable; a page-size-1 view mis-routes
        # decode batches into the prefill orchestrator (num_tokens>64 check).
        # Token-slot indices in block_tables stay valid: the binding derives
        # (page, offset) from the tensor geometry.
        kv = packed_pool.view(torch.uint8).reshape(-1, 64, 656).unsqueeze(1)
        seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=self.device)

        call_kwargs = dict(
            query=q,
            kv_cache=kv,
            workspace_buffer=self.workspace_buffer,
            qk_nope_head_dim=DIM_NOPE,
            kv_lora_rank=DIM_NOPE,
            qk_rope_head_dim=DIM_ROPE,
            block_tables=block_tables,
            seq_lens=seq_lens_t,
            max_seq_len=num_kv,
            sparse_mla_top_k=TOPK,
            bmm1_scale=bmm1_scale,
            backend="auto",
            kv_scale_format="arbitrary_fp32",
            skip_softmax_threshold_scale_factor=None,
        )

        # Warmup outside the graph (required before capture on most kernels).
        for _ in range(2):
            _ = self.flashinfer_decode.trtllm_batch_decode_with_kv_cache_mla(
                **call_kwargs
            )
        torch.cuda.synchronize()

        eager_out = self.flashinfer_decode.trtllm_batch_decode_with_kv_cache_mla(
            **call_kwargs
        )

        graph = torch.cuda.CUDAGraph()
        holder = {}
        with torch.cuda.graph(graph):
            holder["out"] = (
                self.flashinfer_decode.trtllm_batch_decode_with_kv_cache_mla(
                    **call_kwargs
                )
            )
        graph.replay()
        torch.cuda.synchronize()

        self.assertTrue(torch.isfinite(holder["out"]).all())
        torch.testing.assert_close(
            holder["out"].float(), eager_out.float(), atol=0.05, rtol=0.05
        )

        # Replay again unchanged (static inputs) -- must be deterministic.
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            holder["out"].float(), eager_out.float(), atol=0.05, rtol=0.05
        )


if __name__ == "__main__":
    unittest.main()
