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

register_xpu_ci(est_time=120, suite="stage-b-test-1-gpu-xpu")


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
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

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

    def test_mla_model_rejected(self):
        self._assert_rejected("MLA models", model_path=DEFAULT_MLA_MODEL_NAME_FOR_TEST)

    def test_unshared_kv_heads_rejected(self):
        self._assert_rejected(
            "DCP group's KV heads", model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        )


if __name__ == "__main__":
    unittest.main()
