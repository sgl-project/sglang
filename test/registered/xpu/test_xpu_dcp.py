import math
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import (
    DEFAULT_HYBRID_GDN_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    CustomTestCase,
)

register_xpu_ci(est_time=120, suite="extra-test-1-gpu-xpu")


class _ThreadGroup:
    def __init__(self, world_size):
        self.world_size = world_size
        self._barrier = threading.Barrier(world_size, timeout=300)
        self._slots = [None] * world_size
        self._rank = threading.local()

    @property
    def rank_in_group(self):
        return self._rank.value

    def _exchange(self, tensor):
        self._slots[self.rank_in_group] = tensor
        self._barrier.wait()
        tensors = list(self._slots)
        self._barrier.wait()
        return tensors

    def all_gather(self, tensor, dim):
        return torch.cat(self._exchange(tensor), dim=dim)

    def all_reduce(self, tensor):
        return torch.stack(self._exchange(tensor)).sum(0)

    def all_to_all_single(self, output, input):
        chunks = self._exchange(input)
        output.view(self.world_size, -1).copy_(
            torch.stack(
                [c.view(self.world_size, -1)[self.rank_in_group] for c in chunks]
            )
        )


@unittest.skipIf(not torch.xpu.is_available(), "Intel XPU not available")
class TestDCPMatchesFullAttention(CustomTestCase):
    HEAD_DIM = 128
    PAGE_SIZE = 64
    MAX_TOKENS = 256

    def _reference(self, q, k, v, prefix_len):
        group = q.shape[1] // k.shape[1]
        k = k.float().repeat_interleave(group, dim=1)
        v = v.float().repeat_interleave(group, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q.float(), k) * self.HEAD_DIM**-0.5
        q_pos = prefix_len + torch.arange(q.shape[0], device=q.device)
        future = torch.arange(k.shape[0], device=q.device)[None] > q_pos[:, None]
        probs = scores.masked_fill(future, -float("inf")).softmax(-1)
        return torch.einsum("hqk,khd->qhd", probs, v)

    def _rank_forward(
        self, rank, group, *, decode, comm_backend, q, k, v, prefix_lens, kv_heads
    ):
        from sglang.srt.layers.attention.xpu_backend import XPUAttentionBackend
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
        from sglang.srt.model_executor.forward_mode import ForwardMode

        group._rank.value = rank
        device, dcp_size, bs = "xpu", group.world_size, len(q)
        extend_lens = [qb.shape[0] for qb in q]
        seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
        local_heads = q[0].shape[1] // dcp_size
        heads = slice(rank * local_heads, (rank + 1) * local_heads)

        pool = MHATokenToKVPool(
            size=bs * self.MAX_TOKENS // dcp_size + self.PAGE_SIZE,
            page_size=self.PAGE_SIZE,
            dtype=torch.bfloat16,
            head_num=kv_heads,
            head_dim=self.HEAD_DIM,
            layer_num=1,
            device=device,
            enable_memory_saver=False,
        )
        req_to_token = torch.arange(
            bs * self.MAX_TOKENS, dtype=torch.int32, device=device
        ).view(bs, self.MAX_TOKENS)
        key_cache, value_cache = pool.get_kv_buffer(0)
        for b, p in enumerate(prefix_lens):
            rows = b * self.MAX_TOKENS // dcp_size
            rows += torch.arange(len(range(rank, p, dcp_size)), device=device)
            key_cache[rows] = k[b][rank:p:dcp_size]
            value_cache[rows] = v[b][rank:p:dcp_size]

        backend = XPUAttentionBackend.__new__(XPUAttentionBackend)
        backend.use_mla = False
        backend.dcp_size = dcp_size
        backend.dcp_rank = rank
        backend.dcp_group = group
        backend.dcp_comm_backend = comm_backend
        backend.page_size = self.PAGE_SIZE
        backend.num_splits = 0
        backend.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
        backend.token_to_kv_pool = pool
        spans = list(zip(prefix_lens, seq_lens))
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE if decode else ForwardMode.EXTEND,
            batch_size=bs,
            req_pool_indices=torch.arange(bs, device=device),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
            seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32),
            extend_prefix_lens=torch.tensor(
                prefix_lens, dtype=torch.int32, device=device
            ),
            extend_prefix_lens_cpu=prefix_lens,
            extend_seq_lens=torch.tensor(extend_lens, dtype=torch.int32, device=device),
            extend_seq_lens_cpu=extend_lens,
            positions=torch.cat([torch.arange(p, n, device=device) for p, n in spans]),
            out_cache_loc=torch.cat(
                [req_to_token[b, p:n].long() for b, (p, n) in enumerate(spans)]
            ),
            out_cache_loc_is_physical=False,
        )
        backend._init_forward_metadata_dcp(batch)
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=local_heads,
            tp_k_head_num=kv_heads,
            tp_v_head_num=kv_heads,
            head_dim=self.HEAD_DIM,
            v_head_dim=self.HEAD_DIM,
            scaling=self.HEAD_DIM**-0.5,
            logit_cap=0.0,
            k_scale=None,
            v_scale=None,
        )
        q_local = torch.cat([qb[:, heads] for qb in q]).flatten(1)
        k_new = torch.cat([kb[p:] for kb, p in zip(k, prefix_lens)])
        v_new = torch.cat([vb[p:] for vb, p in zip(v, prefix_lens)])
        backend._set_kv_buffer_mha(layer, batch, batch.out_cache_loc, k_new, v_new)
        if decode:
            return backend._forward_decode_dcp(q_local, layer, None)
        return backend._forward_extend_dcp(
            q_local, k_new, v_new, layer, batch, True, None
        )

    def _assert_matches(
        self, *, kv_heads, dcp_size, prefix_lens, extend_lens, comm_backend
    ):
        torch.manual_seed(0)
        device, dim, q_heads = "xpu", self.HEAD_DIM, 8 * dcp_size
        seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
        k = [
            torch.randn(n, kv_heads, dim, device=device, dtype=torch.bfloat16)
            for n in seq_lens
        ]
        v = [torch.randn_like(kb) for kb in k]
        q = [
            torch.randn(n, q_heads, dim, device=device, dtype=torch.bfloat16)
            for n in extend_lens
        ]
        expected = torch.cat(
            [self._reference(*args) for args in zip(q, k, v, prefix_lens)]
        )

        group = _ThreadGroup(dcp_size)
        with ThreadPoolExecutor(dcp_size) as pool:
            futures = [
                pool.submit(
                    self._rank_forward,
                    rank,
                    group,
                    decode=self._decode,
                    comm_backend=comm_backend,
                    q=q,
                    k=k,
                    v=v,
                    prefix_lens=prefix_lens,
                    kv_heads=kv_heads,
                )
                for rank in range(dcp_size)
            ]
            outs = [f.result() for f in futures]

        local_heads = q_heads // dcp_size
        for rank, out in enumerate(outs):
            want = expected[:, rank * local_heads : (rank + 1) * local_heads]
            torch.testing.assert_close(
                out.view_as(want).float(),
                want,
                atol=2e-2,
                rtol=2e-2,
                msg=lambda m: f"rank {rank}: {m}",
            )

    def _check_cases(self, cases, decode, comm_backends=("ag_rs",)):
        self._decode = decode
        for comm_backend in comm_backends:
            for kv_heads, dcp_size, prefix_lens, extend_lens in cases:
                with self.subTest(
                    comm_backend=comm_backend,
                    kv_heads=kv_heads,
                    dcp_size=dcp_size,
                    extend_lens=extend_lens,
                ):
                    self._assert_matches(
                        kv_heads=kv_heads,
                        dcp_size=dcp_size,
                        prefix_lens=prefix_lens,
                        extend_lens=extend_lens,
                        comm_backend=comm_backend,
                    )

    def test_decode(self):
        self._check_cases(
            [
                (4, 2, [36, 0, 89], [1, 1, 1]),
                (2, 4, [36, 1, 89], [1, 1, 1]),
                (2, 4, [1, 0], [1, 1]),
            ],
            decode=True,
            comm_backends=("ag_rs", "a2a"),
        )

    def test_extend(self):
        self._check_cases(
            [
                (4, 2, [29, 0, 64], [7, 5, 3]),
                (2, 4, [29, 3, 64], [7, 5, 3]),
                (1, 2, [29, 64], [7, 3]),
                (4, 2, [30, 3], [1, 1]),
            ],
            decode=False,
        )


@unittest.skipIf(not torch.xpu.is_available(), "Intel XPU not available")
class TestMlaDCPMatchesFullAttention(CustomTestCase):
    KV_LORA = 512
    ROPE = 64
    PAGE_SIZE = 64
    MAX_TOKENS = 256
    DIM = KV_LORA + ROPE

    def _reference(self, q, latent, seq_len):
        scores = (q.float() @ latent[:seq_len].float().T) * self.DIM**-0.5
        return scores.softmax(-1) @ latent[:seq_len, : self.KV_LORA].float()

    def _rank_forward(self, rank, group, *, q, latent, seq_lens, heads):
        from sglang.srt.layers.attention.xpu_backend import XPUAttentionBackend
        from sglang.srt.model_executor.forward_mode import ForwardMode

        group._rank.value = rank
        device, dcp_size, bs = "xpu", group.world_size, len(seq_lens)
        rows_per_req = self.MAX_TOKENS // dcp_size
        kv = torch.zeros(
            bs * rows_per_req + self.PAGE_SIZE,
            1,
            self.DIM,
            device=device,
            dtype=torch.bfloat16,
        )
        for b, n in enumerate(seq_lens):
            mine = latent[b][rank:n:dcp_size]
            kv[b * rows_per_req : b * rows_per_req + len(mine), 0] = mine
        req_to_token = torch.arange(
            bs * self.MAX_TOKENS, dtype=torch.int32, device=device
        ).view(bs, self.MAX_TOKENS)

        backend = XPUAttentionBackend.__new__(XPUAttentionBackend)
        backend.use_mla = True
        backend.dcp_size = dcp_size
        backend.dcp_rank = rank
        backend.dcp_group = group
        backend.page_size = self.PAGE_SIZE
        backend.num_local_heads = heads // dcp_size
        backend.max_kv_len = self.MAX_TOKENS
        backend.device = device
        backend.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
        # kv_layers is a property over token_to_kv_pool.layer_router.
        backend.token_to_kv_pool = SimpleNamespace(
            layer_router=SimpleNamespace(read=lambda _: SimpleNamespace(key=lambda: kv))
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            batch_size=bs,
            req_pool_indices=torch.arange(bs, device=device),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
            seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32),
            out_cache_loc=None,
        )
        backend._init_forward_metadata_dcp(batch)
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=heads,
            v_head_dim=self.KV_LORA,
            head_dim=self.DIM,
            scaling=self.DIM**-0.5,
        )
        return backend._forward_decode_dcp_mla(q.flatten(1), None, layer)

    def _assert_matches(self, *, dcp_size, seq_lens, heads=8):
        torch.manual_seed(0)
        device, bs = "xpu", len(seq_lens)
        latent = [
            torch.randn(n, self.DIM, device=device, dtype=torch.bfloat16) * 0.1
            for n in seq_lens
        ]
        q = torch.randn(bs, heads, self.DIM, device=device, dtype=torch.bfloat16) * 0.1

        group = _ThreadGroup(dcp_size)
        with ThreadPoolExecutor(dcp_size) as pool:
            futures = [
                pool.submit(
                    self._rank_forward,
                    rank,
                    group,
                    q=q,
                    latent=latent,
                    seq_lens=seq_lens,
                    heads=heads,
                )
                for rank in range(dcp_size)
            ]
            parts = [f.result() for f in futures]

        # The model owns this merge; flash_mla_decode returns a base-2 LSE.
        lses = torch.stack([lse.float() for _, lse in parts])
        global_lse = torch.logsumexp(lses * math.log(2), dim=0) / math.log(2)
        merged = torch.zeros(bs, heads, self.KV_LORA, device=device)
        for out, lse in parts:
            weight = torch.exp2(lse.float() - global_lse)
            weight = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)
            merged += out.float().view(bs, heads, self.KV_LORA) * weight.unsqueeze(-1)

        expected = torch.stack(
            [self._reference(q[b], latent[b], n) for b, n in enumerate(seq_lens)]
        )
        torch.testing.assert_close(merged, expected, atol=2e-2, rtol=2e-2)

    def test_decode(self):
        for dcp_size, seq_lens in (
            (2, [36, 1, 89]),
            (4, [36, 2, 89]),
            # Shorter than dcp_size: the top ranks own nothing and get no page table.
            (4, [2, 3]),
        ):
            with self.subTest(dcp_size=dcp_size, seq_lens=seq_lens):
                self._assert_matches(dcp_size=dcp_size, seq_lens=seq_lens)

    def test_zero_length_prefix_row_is_neutralized(self):
        from sgl_kernel.flash_attn import flash_attn_varlen_func

        from sglang.srt.layers.attention.xpu_backend import XPUAttentionBackend

        torch.manual_seed(0)
        device, heads, head_dim, v_dim = "xpu", 8, 192, 128
        q_lens, k_lens = [3, 5], [0, 7]
        q = torch.randn(8, heads, head_dim, device=device, dtype=torch.bfloat16)
        k = torch.randn(7, heads, head_dim, device=device, dtype=torch.bfloat16)
        v = torch.randn(7, heads, v_dim, device=device, dtype=torch.bfloat16)
        out, lse, *_ = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.tensor([0, 3, 8], dtype=torch.int32, device=device),
            cu_seqlens_k=torch.tensor([0, 0, 7], dtype=torch.int32, device=device),
            max_seqlen_q=max(q_lens),
            max_seqlen_k=max(k_lens),
            softmax_scale=head_dim**-0.5,
            causal=False,
            return_softmax_lse=True,
        )
        batch = SimpleNamespace(
            prefix_chunk_has_zero_kv=[True],
            prefix_chunk_seq_lens=torch.tensor(
                [k_lens], dtype=torch.int32, device=device
            ),
            extend_seq_lens=torch.tensor(q_lens, dtype=torch.int32, device=device),
        )
        out, lse = XPUAttentionBackend._neutralize_empty_prefix_rows(
            out, lse.transpose(0, 1).contiguous(), batch, 0
        )

        self.assertTrue(bool((out[:3] == 0).all()))
        self.assertTrue(bool(torch.isneginf(lse[:3]).all()))
        scores = torch.einsum("qhd,khd->hqk", q[3:].float(), k.float())
        probs = (scores * head_dim**-0.5).softmax(-1)
        torch.testing.assert_close(
            out[3:].float(),
            torch.einsum("hqk,khd->qhd", probs, v.float()),
            atol=2e-2,
            rtol=2e-2,
        )


class TestXPUDCPStartupChecks(CustomTestCase):
    @staticmethod
    def _build(model_path=DEFAULT_HYBRID_GDN_SMALL_MODEL_NAME_FOR_TEST, **kwargs):
        from sglang.srt.server_args import ServerArgs

        kwargs.setdefault("attention_backend", "intel_xpu")
        server_args = ServerArgs(
            model_path=model_path,
            device="xpu",
            tp_size=2,
            dcp_size=2,
            mem_fraction_static=0.6,
            trust_remote_code=True,
            **kwargs,
        )
        server_args.resolve_once()
        return server_args

    def _assert_rejected(self, needle, **kwargs):
        with self.assertRaises(ValueError) as cm:
            self._build(**kwargs)
        self.assertIn(needle, str(cm.exception))

    def test_supported_config_is_admitted(self):
        self.assertEqual(self._build().dcp_size, 2)

    def test_triton_backend_rejected(self):
        for backend in ("attention_backend", "decode_attention_backend"):
            with self.subTest(backend=backend):
                self._assert_rejected("intel_xpu", **{backend: "triton"})

    def test_dense_mla_model_admitted(self):
        self.assertEqual(
            self._build(model_path=DEFAULT_MLA_MODEL_NAME_FOR_TEST).dcp_size, 2
        )

    def test_mla_arch_without_dcp_path_rejected(self):
        self._assert_rejected("MiniCPM3ForCausalLM", model_path="openbmb/MiniCPM3-4B")

    def test_mla_disabled_chunked_prefix_cache_rejected(self):
        self._assert_rejected(
            "chunked prefix cache",
            model_path=DEFAULT_MLA_MODEL_NAME_FOR_TEST,
            disable_chunked_prefix_cache=True,
        )

    def test_mla_speculative_decoding_rejected(self):
        self._assert_rejected(
            "speculative decoding",
            model_path=DEFAULT_MLA_MODEL_NAME_FOR_TEST,
            speculative_algorithm="EAGLE",
        )

    def test_unshared_kv_heads_rejected(self):
        self._assert_rejected(
            "DCP group's KV heads", model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        )


if __name__ == "__main__":
    unittest.main()
