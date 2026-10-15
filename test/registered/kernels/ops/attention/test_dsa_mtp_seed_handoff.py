"""The first draft decode must read the KV slots its draft prefill selected,
including when shared cached-prefix offsets exceed the KV pool (fused top-k, fp8 KV)."""

import contextlib
import math
import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.dsa.dequant_k_cache import dequantize_k_cache_paged
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState
from sglang.srt.layers.linear import LinearBase
from sglang.srt.mem_cache.kv_cache_configurator import calculate_mla_kv_cache_dim
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mha import (
    forward_dsa_indexer_for_mha,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

PAGE_SIZE = 64
INDEX_TOPK = 2048
NUM_HEADS = 128
KV_LORA_RANK = 512
ROPE_DIM = 64
POOL_SIZE = 8192  # slots; the pool allocates one extra page
MAX_BS = 8
HIDDEN_SIZE = 1024
Q_LORA_RANK = 1536
KV_DTYPE = torch.float8_e4m3fn


def _runner():
    """The model-runner surface DeepseekSparseAttnBackend reads at init."""
    model_config = SimpleNamespace(
        context_len=POOL_SIZE,
        num_attention_heads=NUM_HEADS,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=ROPE_DIM,
        qk_nope_head_dim=128,
        hf_config=SimpleNamespace(
            architectures=["DeepseekV3ForCausalLM"],
            index_topk=INDEX_TOPK,
            index_head_dim=128,
            index_n_heads=32,
        ),
    )
    return SimpleNamespace(
        device="cuda",
        kv_cache_dtype=KV_DTYPE,
        page_size=PAGE_SIZE,
        max_running_requests=MAX_BS,
        hisparse_coordinator=None,
        model_config=model_config,
        req_to_token_pool=SimpleNamespace(
            size=MAX_BS,
            req_to_token=torch.zeros(
                MAX_BS, POOL_SIZE, dtype=torch.int32, device="cuda"
            ),
        ),
        # Production sizing: the scaled fp8 layout makes dsa_kv_cache_store_fp8
        # true, which is what routes fp8 prefill top-k through RAGGED.
        token_to_kv_pool=DSATokenToKVPool(
            size=POOL_SIZE,
            page_size=PAGE_SIZE,
            dtype=KV_DTYPE,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=ROPE_DIM,
            layer_num=1,
            device="cuda",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=calculate_mla_kv_cache_dim(
                model_config=model_config, kv_cache_dtype=KV_DTYPE
            ),
        ),
    )


def _indexer() -> Indexer:
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            indexer = Indexer(
                hidden_size=HIDDEN_SIZE,
                index_n_heads=32,
                index_head_dim=128,
                rope_head_dim=ROPE_DIM,
                index_topk=INDEX_TOPK,
                q_lora_rank=Q_LORA_RANK,
                max_position_embeddings=163840,
                rope_theta=10000.0,
                layer_id=0,
                scale_fmt="ue8m0",
            )
    finally:
        torch.set_default_dtype(prev_dtype)
    # weights_proj keeps fp32 params as in production.
    for name, module in indexer.named_modules():
        if isinstance(module, LinearBase) and "weights_proj" not in name:
            module.to(dtype=torch.bfloat16)
    return indexer.cuda()


@unittest.skipIf(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    "flashmla_sparse_q8 prefill is SM90-only",
)
class TestDSAMTPSeedHandoff(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        server_args = ServerArgs(model_path="dummy")
        server_args.kv_cache_dtype = "fp8_e4m3"
        server_args.dsa_prefill_backend = "flashmla_sparse_q8"
        server_args.dsa_decode_backend = "flashmla_kv"
        set_global_server_args_for_scheduler(server_args)
        cls._stack = contextlib.ExitStack()
        cls._stack.enter_context(get_parallel().override(attn_tp_size=1))
        cls._stack.enter_context(envs.SGLANG_DSA_FUSE_TOPK.override(True))
        # The server sets this to index_topk; here the backend is built directly.
        cls._stack.enter_context(
            envs.SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD.override(INDEX_TOPK)
        )

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()

    def setUp(self):
        torch.manual_seed(20261011)
        runner = _runner()
        self.backend = DeepseekSparseAttnBackend(runner)
        self.assertTrue(self.backend.use_fused_topk)
        self.assertTrue(self.backend.dsa_kv_cache_store_fp8)
        self.pool = runner.token_to_kv_pool
        self.req_to_token = runner.req_to_token_pool.req_to_token
        self.indexer = _indexer()
        self.layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=NUM_HEADS,
            head_dim=KV_LORA_RANK + ROPE_DIM,
            v_head_dim=KV_LORA_RANK,
            scaling=1.0 / math.sqrt(KV_LORA_RANK + ROPE_DIM),
            is_cross_attention=False,
        )

    def _layout(self, prefix_len, suffix_lens):
        """Scatter pages; requests share the prefix pages like a radix hit.

        Row b of req_to_token covers prefix + suffix + one decode slot. Row
        len(suffix_lens) owns only the prefix and is used to warm it.
        """
        pages = (torch.randperm(POOL_SIZE // PAGE_SIZE) + 1).tolist()  # page 0 pads
        prefix_pages = pages[: prefix_len // PAGE_SIZE]
        pages = pages[len(prefix_pages) :]
        rows = []
        for suffix_len in suffix_lens:
            own = math.ceil((suffix_len + 1) / PAGE_SIZE)
            rows.append(prefix_pages + pages[:own])
            pages = pages[own:]
        rows.append(prefix_pages)
        for b, row_pages in enumerate(rows):
            slots = torch.tensor(row_pages)[:, None] * PAGE_SIZE + torch.arange(
                PAGE_SIZE
            )
            self.req_to_token[b, : slots.numel()] = slots.flatten().to(torch.int32)
        used = torch.unique(self.req_to_token[: len(rows)])
        used = used[used > 0].long()
        # Random latent KV for every slot a request can reach.
        self.pool.set_mla_kv_buffer(
            self.layer,
            used,
            torch.randn(
                used.numel(), 1, KV_LORA_RANK, device="cuda", dtype=torch.bfloat16
            ),
            torch.randn(used.numel(), 1, ROPE_DIM, device="cuda", dtype=torch.bfloat16),
        )

    def _batch(self, mode, rows, prefix_lens, extend_lens, spec_info=None):
        seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
        return ForwardBatch(
            forward_mode=mode,
            batch_size=len(rows),
            input_ids=torch.zeros(sum(extend_lens), dtype=torch.long, device="cuda"),
            req_pool_indices=torch.tensor(rows, device="cuda"),
            seq_lens=torch.tensor(seq_lens, device="cuda"),
            seq_lens_cpu=torch.tensor(seq_lens),
            seq_lens_sum=sum(seq_lens),
            out_cache_loc=torch.cat(
                [
                    self.req_to_token[r, p : p + e]
                    for r, p, e in zip(rows, prefix_lens, extend_lens)
                ]
            ).long(),
            positions=torch.cat(
                [
                    torch.arange(p, p + e, device="cuda")
                    for p, e in zip(prefix_lens, extend_lens)
                ]
            ),
            extend_num_tokens=sum(extend_lens),
            extend_prefix_lens=torch.tensor(prefix_lens, device="cuda"),
            extend_prefix_lens_cpu=prefix_lens,
            extend_seq_lens=torch.tensor(extend_lens, device="cuda"),
            extend_seq_lens_cpu=extend_lens,
            spec_info=spec_info,
        )

    def _indexer_inputs(self, batch):
        n = batch.extend_num_tokens
        return dict(
            x=torch.randn(n, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16),
            q_lora=torch.randn(n, Q_LORA_RANK, device="cuda", dtype=torch.bfloat16),
            positions=batch.positions,
            forward_batch=batch,
            layer_id=0,
        )

    def _export_seed(self, path, prefix_len, suffix_lens):
        """Warm the shared prefix, then run the draft prefill export.

        Returns the exported seed and the raw top-k the indexer produced.
        """
        bs = len(suffix_lens)
        spec_info = SimpleNamespace(
            dsa_topk_indices=None,
            dsa_seed_topk_capture=torch.full(
                (bs, INDEX_TOPK), -2, dtype=torch.int32, device="cuda"
            ),
            dsa_seed_topk_select=torch.tensor(suffix_lens, device="cuda").cumsum(0) - 1,
        )
        produced = []

        def indexer(**kwargs):
            produced.append(self.indexer(**kwargs))
            return produced[-1]

        with forward_context(ForwardContext(attn_backend=self.backend)):
            warm = self._batch(ForwardMode.EXTEND, [bs], [0], [prefix_len])
            self.backend.init_forward_metadata(warm)
            self.indexer(**self._indexer_inputs(warm))

            batch = self._batch(
                ForwardMode.EXTEND,
                list(range(bs)),
                [prefix_len] * bs,
                suffix_lens,
                spec_info,
            )
            self.backend.init_forward_metadata(batch)
            self.assertEqual(self.backend.use_mha, path == "mha")
            inputs = self._indexer_inputs(batch)
            if path == "mha":
                forward_dsa_indexer_for_mha(
                    indexer,
                    hidden_states=inputs["x"],
                    q_lora=inputs["q_lora"],
                    positions=batch.positions,
                    forward_batch=batch,
                    layer_id=0,
                )
            else:
                # The sequence deepseek_nextn runs on the sparse-MLA path.
                state = IndexTopKShareState.from_mtp_carry(batch)
                state.update(indexer(**inputs))
                state.publish()
        (logical,) = produced
        return spec_info.dsa_seed_topk_capture, logical[spec_info.dsa_seed_topk_select]

    def _expected_slots(self, logical, prefix_len, suffix_lens):
        """Oracle: RAGGED offset -> (request, local position) -> physical slot."""
        kv_lens = torch.tensor([prefix_len + s for s in suffix_lens], device="cuda")
        local = logical.long() - (kv_lens.cumsum(0) - kv_lens)[:, None]
        valid = logical != -1
        self.assertTrue(bool(((local >= 0) & (local < kv_lens[:, None]))[valid].all()))
        slots = self.req_to_token[: len(suffix_lens)].gather(1, local.clamp(min=0))
        return torch.where(valid, slots, -1)

    def _decode(self, seed, prefix_len, suffix_lens):
        """First draft decode carrying the seed through mtp_iteration."""
        bs = len(suffix_lens)
        decode = self._batch(
            ForwardMode.DECODE,
            list(range(bs)),
            [prefix_len + s for s in suffix_lens],
            [1] * bs,
            SimpleNamespace(
                dsa_topk_indices=seed,
                dsa_seed_topk_capture=None,
                dsa_seed_topk_select=None,
            ),
        )
        q = torch.randn(
            bs, NUM_HEADS, KV_LORA_RANK + ROPE_DIM, device="cuda", dtype=torch.bfloat16
        )
        with forward_context(ForwardContext(attn_backend=self.backend)):
            with IndexTopKShareState.mtp_iteration(
                decode, keep_carry_seed=True
            ) as state:
                carried = state.topk_indices
                self.backend.init_forward_metadata(decode)
                out = self.backend.forward_decode(
                    q,
                    None,
                    None,
                    self.layer,
                    decode,
                    save_kv_cache=False,
                    topk_indices=carried,
                )
        return q, out.view(bs, NUM_HEADS, KV_LORA_RANK)

    def _reference_attention(self, q, slots):
        kv_buffer = self.pool.get_key_buffer(0)
        outs = []
        for b in range(q.shape[0]):
            row = slots[b][slots[b] >= 0]
            kv = dequantize_k_cache_paged(kv_buffer, row.int()).view(row.numel(), -1)
            scores = (q[b].float() @ kv.float().T) * self.layer.scaling
            outs.append(torch.softmax(scores, dim=-1) @ kv[:, :KV_LORA_RANK].float())
        return torch.stack(outs)

    def _check_handoff(self, path, prefix_len, suffix_lens):
        self._layout(prefix_len, suffix_lens)
        seed, logical = self._export_seed(path, prefix_len, suffix_lens)
        pool_slots = self.pool.size + self.pool.page_size
        # The case must put RAGGED offsets past the pool, as in the prod crash,
        # so an unconverted seed fails here rather than as an illegal memory access.
        self.assertGreaterEqual(int(logical.max()), pool_slots)
        in_pool = (seed == -1) | ((seed >= 0) & (seed < pool_slots))
        self.assertTrue(
            bool(in_pool.all()),
            f"seed max {int(seed.max())} is outside the {pool_slots}-slot KV pool",
        )
        expected = self._expected_slots(logical, prefix_len, suffix_lens)
        torch.testing.assert_close(seed, expected, rtol=0, atol=0)

        q, out = self._decode(seed, prefix_len, suffix_lens)
        torch.testing.assert_close(
            out.float(), self._reference_attention(q, expected), rtol=2e-2, atol=2e-2
        )

    def test_mha_export(self):
        # max kv_len <= index_topk routes the draft prefill through dense MHA.
        self._check_handoff(
            "mha", prefix_len=1792, suffix_lens=[200, 150, 250, 180, 220]
        )

    def test_sparse_mla_export(self):
        # Longer prefills take sparse MLA and export through publish().
        self._check_handoff("sparse_mla", prefix_len=5120, suffix_lens=[900, 700, 800])


if __name__ == "__main__":
    unittest.main(verbosity=3)
