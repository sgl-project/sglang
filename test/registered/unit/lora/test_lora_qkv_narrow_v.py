"""qkv_proj LoRA with a V head narrower than Q/K (MiMo-V2), through the real wrapper.

Each case builds the attention-TP shard of a QKVParallelLinear, wraps it, loads an
adapter through the memory pool's member writes and runs the Triton LoRA kernels. The
reference slices the adapter's q/k/v B rows by hand.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

HIDDEN, HEAD, V_HEAD, RANK, SCALING = 256, 192, 128, 16, 2.0
ADAPTED_TOKENS, BASE_TOKENS = 24, 8
# (q heads, kv heads, the config's top-level kv heads) of MiMo-V2.6 attention layers.
GEOMETRIES = {
    "flash_full": (64, 4, 4),
    "flash_swa": (64, 8, 4),
    "pro": (128, 8, 8),
}


def _load(view, weight):
    """The pool's buffer write: zero a slot, or copy a weight of exactly its shape."""
    if weight is None:
        view.zero_()
    else:
        assert view.shape == weight.shape, (view.shape, weight.shape)
        view.copy_(weight)


@unittest.skipUnless(torch.cuda.is_available(), "Triton LoRA kernels need CUDA")
class TestQKVLoRANarrowV(CustomTestCase):
    def setUp(self):
        from sglang.srt.runtime_context import reset_context

        reset_context()
        self.addCleanup(reset_context)

    def _shard(self, tp_size, rank, num_heads, num_kv_heads, weights):
        from sglang.srt.layers.linear import QKVParallelLinear
        from sglang.srt.runtime_context import SpawnRanks, reset_context
        from sglang.srt.server_args import ServerArgs
        from sglang.test.parallel_groups import publish

        reset_context()
        publish(
            ServerArgs(model_path="dummy", device="cpu", tp_size=tp_size),
            role="test",
            ranks=SpawnRanks(world_rank=rank),
        )
        qkv = QKVParallelLinear(
            HIDDEN,
            HEAD,
            num_heads,
            num_kv_heads,
            bias=False,
            v_head_size=V_HEAD,
            params_dtype=torch.bfloat16,
            parallel_group="attn_tp",
        )
        for name, weight in zip("qkv", weights):
            qkv.weight.weight_loader(qkv.weight, weight, name)
        return qkv.cuda()

    def _check(self, name, tp_size, rank, form):
        from sglang.srt.lora.backend.triton_backend import TritonLoRABackend
        from sglang.srt.lora.layers import QKVParallelLinearWithLoRA
        from sglang.srt.lora.mem_pool import LoRAMemoryPool
        from sglang.srt.lora.utils import LoRABatchInfo

        num_heads, num_kv_heads, config_kv_heads = GEOMETRIES[name]
        gen = torch.Generator().manual_seed(tp_size * 100 + rank)
        rows = (num_heads * HEAD, num_kv_heads * HEAD, num_kv_heads * V_HEAD)
        weights = [
            (torch.randn(r, HIDDEN, generator=gen) * HIDDEN**-0.5).bfloat16()
            for r in rows
        ]
        lora_a = [
            (torch.randn(RANK, HIDDEN, generator=gen) * HIDDEN**-0.5).bfloat16()
            for _ in rows
        ]
        lora_b = [(torch.randn(r, RANK, generator=gen) * 0.1).bfloat16() for r in rows]
        if form == "partial":  # an adapter without k_proj
            lora_a[1], lora_b[1] = None, None

        qkv = self._shard(tp_size, rank, num_heads, num_kv_heads, weights)
        backend = TritonLoRABackend(max_loras_per_batch=2, device=torch.device("cuda"))
        layer = QKVParallelLinearWithLoRA(qkv, backend)

        # This rank's q heads and the kv head it holds (replicated when TP > kv heads).
        q_rows = qkv.num_heads * HEAD
        kv_index = rank // qkv.num_kv_head_replicas
        local = [
            slice(rank * q_rows, (rank + 1) * q_rows),
            slice(
                kv_index * qkv.num_kv_heads * HEAD,
                (kv_index + 1) * qkv.num_kv_heads * HEAD,
            ),
            slice(
                kv_index * qkv.num_kv_heads * V_HEAD,
                (kv_index + 1) * qkv.num_kv_heads * V_HEAD,
            ),
        ]
        widths = [s.stop - s.start for s in local]
        self.assertEqual(
            layer.output_offset.tolist(),
            [0, widths[0], widths[0] + widths[1], sum(widths)],
        )
        self.assertEqual(layer.max_qkv_out_dim, max(widths))

        # The pool sizes B from the config's kv heads, so it may hold spare rows.
        pool = LoRAMemoryPool.__new__(LoRAMemoryPool)
        pool.base_hf_config = SimpleNamespace(
            num_key_value_heads=config_kv_heads, head_dim=HEAD
        )
        pool.tp_rank = rank
        pool.pin_memory_available = False
        pool.enable_lora_overlap_loading = False
        b_rows = pool._column_parallel_lora_b_per_rank_dim(
            "qkv_proj", sum(rows), tp_size
        )
        self.assertGreaterEqual(b_rows, sum(widths))
        a_buffer = torch.zeros(2, 3 * RANK, HIDDEN, dtype=torch.bfloat16, device="cuda")
        b_buffer = torch.zeros(2, b_rows, RANK, dtype=torch.bfloat16, device="cuda")
        if form == "fused":  # one canonical qkv_proj adapter tensor per side
            members, fused = {}, (torch.cat(lora_a), torch.cat(lora_b))
        else:
            members = {
                (side, slot): (weight, f"member{slot}")
                for slot in range(3)
                for side, weight in (("A", lora_a[slot]), ("B", lora_b[slot]))
                if weight is not None
            }
            fused = (None, None)
        a_writes, b_writes = pool._fused_member_writes(
            layer,
            "qkv_proj",
            member_weights=members,
            fused_weights=fused,
            fused_cache_keys=("qkv.A", "qkv.B"),
            lora_rank=RANK,
        )
        pool._load_fused_member_writes(
            a_view=a_buffer[1],
            b_buffer=b_buffer[1],
            a_writes=[(s, w.cuda(), k) for s, w, k in a_writes],
            b_writes=[(o, w.cuda(), k) for o, w, k in b_writes],
            lora_rank=RANK,
            pinned_layer_weights={},
            load_lora_weight_tensor=_load,
        )
        layer.set_lora_info(a_buffer, b_buffer)

        # Adapted tokens first: a write past a row's end lands in the next token.
        tokens = ADAPTED_TOKENS + BASE_TOKENS
        backend.batch_info = LoRABatchInfo(
            use_cuda_graph=False,
            bs=2,
            num_segments=2,
            seg_indptr=torch.tensor(
                [0, ADAPTED_TOKENS, tokens], dtype=torch.int32
            ).cuda(),
            weight_indices=torch.tensor([1, 0], dtype=torch.int32).cuda(),
            lora_ranks=torch.tensor([0, RANK], dtype=torch.int32).cuda(),
            scalings=torch.tensor([0.0, SCALING]).cuda(),
            max_len=ADAPTED_TOKENS,
            seg_lens=torch.tensor(
                [ADAPTED_TOKENS, BASE_TOKENS], dtype=torch.int32
            ).cuda(),
            permutation=None,
            expected_tokens=tokens,
            has_active_lora=True,
        )
        x = torch.randn(tokens, HIDDEN, generator=gen).bfloat16().cuda()
        base, _ = qkv(x)
        out, _ = layer(x)

        torch.testing.assert_close(
            out[ADAPTED_TOKENS:], base[ADAPTED_TOKENS:], atol=0, rtol=0
        )
        h = x[:ADAPTED_TOKENS].float()
        delta = torch.cat(
            [
                (
                    torch.zeros(ADAPTED_TOKENS, w, device="cuda")
                    if a is None
                    else SCALING * (h @ a.float().cuda().T) @ b[s].float().cuda().T
                )
                for a, b, s, w in zip(lora_a, lora_b, local, widths)
            ],
            dim=1,
        )
        torch.testing.assert_close(
            (out[:ADAPTED_TOKENS] - base[:ADAPTED_TOKENS]).float(),
            delta,
            atol=3e-2,
            rtol=2e-2,
        )

    def test_member_partial_and_fused_adapters_across_geometries_and_shards(self):
        cases = [
            ("flash_full", 1, 0, "members"),
            ("flash_full", 4, 3, "partial"),
            ("flash_full", 8, 1, "members"),
            ("flash_full", 8, 7, "fused"),
            ("flash_swa", 4, 3, "members"),
            ("flash_swa", 8, 7, "partial"),
            ("pro", 4, 1, "members"),
            ("pro", 8, 7, "fused"),
        ]
        for name, tp_size, rank, form in cases:
            with self.subTest(name=name, tp=tp_size, rank=rank, form=form):
                self._check(name, tp_size, rank, form)


if __name__ == "__main__":
    unittest.main()
