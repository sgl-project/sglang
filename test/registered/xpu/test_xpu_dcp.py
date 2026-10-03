import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.attention_hook import (
    _check_xpu_dcp,
    _xpu_fmha_emits_softmax_lse,
)
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import (
    DEFAULT_HYBRID_GDN_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    CustomTestCase,
)

register_xpu_ci(est_time=120, suite="stage-b-test-1-gpu-xpu")

_SKIP_REASON = None
if not torch.xpu.is_available():
    _SKIP_REASON = "Intel XPU not available"
elif not _xpu_fmha_emits_softmax_lse():
    _SKIP_REASON = "installed sgl-kernel-xpu does not emit softmax_lse"


def _kernel_lse(emits: bool):
    return patch(
        "sglang.srt.arg_groups.attention_hook._xpu_fmha_emits_softmax_lse",
        return_value=emits,
    )


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


@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON or "")
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

    def _rank_forward(self, rank, group, *, decode, q, k, v, prefix_lens, kv_heads):
        from sglang.srt.layers.attention.xpu_backend import XPUAttentionBackend
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        group._rank.value = rank
        device, dcp_size, bs = "xpu", group.world_size, len(q)
        extend_lens = [qb.shape[0] for qb in q]
        seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
        local_heads = q[0].shape[1] // dcp_size
        heads = slice(rank * local_heads, (rank + 1) * local_heads)

        key_cache = torch.zeros(
            bs * self.MAX_TOKENS // dcp_size,
            kv_heads,
            self.HEAD_DIM,
            device=device,
            dtype=torch.bfloat16,
        )
        value_cache = torch.zeros_like(key_cache)
        for b in range(bs):
            rows = (b * self.MAX_TOKENS + rank) // dcp_size + torch.arange(
                len(range(rank, seq_lens[b], dcp_size)), device=device
            )
            key_cache[rows] = k[b][rank::dcp_size]
            value_cache[rows] = v[b][rank::dcp_size]

        backend = XPUAttentionBackend.__new__(XPUAttentionBackend)
        backend.dcp_size = dcp_size
        backend.dcp_rank = rank
        backend.dcp_group = group
        backend.page_size = self.PAGE_SIZE
        backend.num_splits = 0
        backend.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(
                bs * self.MAX_TOKENS, dtype=torch.int32, device=device
            ).view(bs, self.MAX_TOKENS)
        )
        backend.token_to_kv_pool = SimpleNamespace(
            get_kv_buffer=lambda layer_id: (key_cache, value_cache)
        )
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
        )
        q_local = torch.cat([qb[:, heads] for qb in q]).flatten(1)
        if decode:
            return backend._forward_decode_dcp(q_local, layer, None)
        k_new = torch.cat([kb[p:] for kb, p in zip(k, prefix_lens)])
        v_new = torch.cat([vb[p:] for vb, p in zip(v, prefix_lens)])
        return backend._forward_extend_dcp(
            q_local, k_new, v_new, layer, batch, True, None
        )

    def _assert_matches(self, *, kv_heads, dcp_size, prefix_lens, extend_lens):
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

    def _check_cases(self, cases, decode):
        self._decode = decode
        for kv_heads, dcp_size, prefix_lens, extend_lens in cases:
            with self.subTest(
                kv_heads=kv_heads, dcp_size=dcp_size, extend_lens=extend_lens
            ):
                self._assert_matches(
                    kv_heads=kv_heads,
                    dcp_size=dcp_size,
                    prefix_lens=prefix_lens,
                    extend_lens=extend_lens,
                )

    def test_decode(self):
        self._check_cases(
            [(4, 2, [36, 0, 89], [1, 1, 1]), (2, 4, [36, 2, 89], [1, 1, 1])],
            decode=True,
        )

    def test_extend(self):
        self._check_cases(
            [
                (4, 2, [29, 0, 64], [7, 5, 3]),
                (2, 4, [29, 3, 64], [7, 5, 3]),
                (1, 2, [29, 64], [7, 3]),
                # seqlen_q == 1 takes the closed form instead of the kernel.
                (4, 2, [30, 3], [1, 1]),
            ],
            decode=False,
        )


class TestDCPShardPageTable(CustomTestCase):
    def test_pages_cover_only_this_ranks_tokens(self):
        from sglang.srt.layers.attention.xpu_backend import _dcp_shard_page_table

        req_to_token = torch.arange(2 * 32, dtype=torch.int32).view(2, 32)
        table = _dcp_shard_page_table(
            req_to_token,
            torch.tensor([1, 0]),
            page_size=4,
            dcp_size=2,
            dcp_rank=1,
            max_local_len=9,
        )
        self.assertEqual(table.tolist(), [[4, 5, 6], [0, 1, 2]])
        self.assertIsNone(
            _dcp_shard_page_table(req_to_token, torch.tensor([0]), 4, 2, 1, 0)
        )


class _StubModelConfig:
    def __init__(self, q_heads, kv_heads, **fields):
        self._q_heads = q_heads
        self._kv_heads = kv_heads
        self.hf_config = SimpleNamespace(architectures=["StubForCausalLM"])
        self.attention_chunk_size = None
        self.is_encoder_decoder = False
        self.__dict__.update(fields)

    def get_max_num_attention_heads(self):
        return self._q_heads

    def get_num_kv_heads(self, tensor_parallel_size, dcp_size=1):
        return max(1, self._kv_heads // max(1, tensor_parallel_size // dcp_size))


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

    def _assert_rejected(self, needle, lse=True, **kwargs):
        with _kernel_lse(lse), self.assertRaises(ValueError) as cm:
            self._build(**kwargs)
        self.assertIn(needle, str(cm.exception))

    def test_supported_config_is_admitted(self):
        with _kernel_lse(True):
            self.assertEqual(self._build().dcp_size, 2)

    def test_triton_backend_rejected(self):
        for backend in ("attention_backend", "decode_attention_backend"):
            with self.subTest(backend=backend):
                self._assert_rejected("intel_xpu", **{backend: "triton"})

    def test_old_kernel_rejected(self):
        self._assert_rejected("sgl-kernel-xpu >= 0.3.0", lse=False)

    def test_mla_model_rejected(self):
        self._assert_rejected("MLA models", model_path=DEFAULT_MLA_MODEL_NAME_FOR_TEST)

    def test_unshared_kv_heads_rejected(self):
        self._assert_rejected(
            "DCP group's KV heads", model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        )

    def test_model_features_rejected(self):
        with _kernel_lse(True):
            server_args = self._build()
            _check_xpu_dcp(server_args, _StubModelConfig(16, 1))
            for model_config, needle in (
                (_StubModelConfig(32, 1), "GQA group of 32"),
                (_StubModelConfig(16, 1, attention_chunk_size=8192), "chunked local"),
                (_StubModelConfig(16, 1, is_encoder_decoder=True), "encoder-decoder"),
            ):
                with self.subTest(needle=needle):
                    with self.assertRaises(ValueError) as cm:
                        _check_xpu_dcp(server_args, model_config)
                    self.assertIn(needle, str(cm.exception))


if __name__ == "__main__":
    unittest.main()
