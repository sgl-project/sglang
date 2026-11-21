# SPDX-License-Identifier: Apache-2.0

import importlib.util
import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.models.kimi_k3_amd.mla import (
    create_q_cache_writer,
    fused_q_cache_available,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=15, stage="base-b", runner_config="1-gpu-small-amd-mi35x")


def _fused_write_available():
    if importlib.util.find_spec("aiter") is None:
        return False
    with envs.SGLANG_USE_AITER.override(True):
        return fused_q_cache_available()


def _inputs(tokens, cache_dtype=torch.bfloat16, cache_rows=None):
    torch.manual_seed(23)
    heads = 12
    return {
        "q_nope": torch.randn(tokens, heads, 512, device="cuda", dtype=torch.bfloat16),
        "q_pe": torch.randn(tokens, heads, 64, device="cuda", dtype=torch.bfloat16),
        "k_nope": torch.randn(tokens, 1, 512, device="cuda", dtype=torch.bfloat16),
        "k_pe": torch.randn(tokens, 1, 64, device="cuda", dtype=torch.bfloat16),
        "kv_cache": torch.zeros(
            cache_rows or tokens + 4, 1, 576, device="cuda", dtype=cache_dtype
        ),
        "slot_mapping": torch.arange(tokens, device="cuda", dtype=torch.int64),
    }


def _write(writer, values):
    return writer.write(
        **values, k_scale_float=1.0, attention_backend="aiter", is_target_verify=False
    )


def _split_chain(values):
    """The split chain: Q is the BF16 concat, the cache a plain cast at scale 1."""
    q = torch.cat((values["q_nope"], values["q_pe"]), dim=-1)
    k = torch.cat((values["k_nope"], values["k_pe"]), dim=-1)
    return q, k.to(values["kv_cache"].dtype)


@unittest.skipUnless(
    _fused_write_available(), "Kimi-K3 fused MLA Q/cache write runs on gfx950 only"
)
class TestKimiK3RocmMlaQCacheWriter(CustomTestCase):
    def setUp(self):
        with envs.SGLANG_USE_AITER.override(True):
            self.writer = create_q_cache_writer(kv_lora_rank=512, qk_rope_head_dim=64)

    def test_writer_matches_split_chain_bytes(self):
        """K3's KV scale is 1, so the fused write must store the split chain's bytes,
        also for -0.0, inf and NaN lanes (inf and NaN in k: BF16 cache only).
        A slot < 0 token skips only its cache write; its Q row is still written."""
        for cache_dtype in (torch.bfloat16, torch.float8_e4m3fn):
            for tokens in (1, 33, 256):
                with self.subTest(cache_dtype=cache_dtype, tokens=tokens):
                    self._check_split_chain_bytes(cache_dtype, tokens)

    def _check_split_chain_bytes(self, cache_dtype, tokens):
        values = _inputs(tokens, cache_dtype=cache_dtype, cache_rows=tokens + 8)
        if tokens >= 4:
            # The PE halves are copied, not rotated (the NeoX partner of lane i is lane
            # i + 32): an inf lane must not turn its partner into NaN, and a -0.0 lane
            # must keep its sign whatever its partner's.
            values["q_pe"][0, 0, 0] = float("inf")
            values["q_pe"][1, 0, (0, 33)] = -0.0
            values["q_pe"][1, 0, (1, 32)] = -3.0
            values["k_pe"][2, 0, (0, 33)] = -0.0
            values["k_pe"][2, 0, (1, 32)] = -3.0
            values["q_nope"][3, 0, 0] = float("nan")
            if cache_dtype == torch.bfloat16:
                # An FP8 cache saturates inf and maps NaN differently from the split
                # chain's torch cast; that is not the launch's doing.
                values["k_pe"][0, 0, 0] = float("inf")
                values["k_nope"][3, 0, 0] = float("nan")
        slots = torch.randperm(tokens + 8, device="cuda")[:tokens]
        slots[-1] = -1
        values["slot_mapping"] = slots
        q, k = _write(self.writer, values)
        torch.cuda.synchronize()
        expected_q, expected_k = _split_chain(values)
        self.assertTrue(torch.equal(q.view(torch.int16), expected_q.view(torch.int16)))
        self.assertEqual(k.shape, expected_k.shape)
        cache = values["kv_cache"].view(torch.uint8)
        written = slots[:-1]
        self.assertTrue(torch.equal(cache[written], expected_k[:-1].view(torch.uint8)))
        untouched = torch.ones(cache.shape[0], dtype=torch.bool, device="cuda")
        untouched[written] = False
        self.assertEqual(torch.count_nonzero(cache[untouched]).item(), 0)

    def test_writer_fp8_q_out_matches_quant_then_pad(self):
        """With the backend's FP8 query buffer, the writer must store the bytes the asm
        verify built for itself before: the static FP8 quant at its unit q scale of
        cat(q_nope, q_pe), saturating at +-448, in a zeroed 16-head buffer. The cache
        keeps the split chain's bytes."""
        for cache_dtype in (torch.bfloat16, torch.float8_e4m3fn):
            for tokens in (1, 8, 33):
                with self.subTest(cache_dtype=cache_dtype, tokens=tokens):
                    self._check_fp8_q_out(cache_dtype, tokens)

    def _check_fp8_q_out(self, cache_dtype, tokens):
        from sglang.kernels.ops.quantization.fp8_kernel import scaled_fp8_quant

        values = _inputs(tokens, cache_dtype=cache_dtype)
        values["q_nope"][:, 0, :4] = 1.0e4
        values["q_nope"][:, 1, :4] = -1.0e4
        values["q_pe"][:, 2, :4] = 1.0e4
        values["q_pe"][:, 3, :4] = -600.0
        buf = torch.zeros(64, 16, 576, device="cuda", dtype=torch.float8_e4m3fn)[
            :tokens
        ]

        q, _ = self.writer.write(
            **values,
            k_scale_float=1.0,
            attention_backend="aiter",
            is_target_verify=True,
            get_fp8_q_out=lambda: buf,
        )
        torch.cuda.synchronize()

        bf16_q, expected_k = _split_chain(values)
        unit_scale = torch.ones(1, device="cuda", dtype=torch.float32)
        quant, _ = scaled_fp8_quant(bf16_q.reshape(tokens, -1), unit_scale)
        expected = torch.zeros_like(buf)
        expected[:, :12] = quant.view(tokens, 12, 576)
        self.assertIs(q, buf)
        self.assertTrue(torch.equal(q.view(torch.uint8), expected.view(torch.uint8)))
        cache = values["kv_cache"].view(torch.uint8)
        self.assertTrue(torch.equal(cache[:tokens], expected_k.view(torch.uint8)))

    def test_writer_captures_without_a_warmup_and_replays_on_new_inputs(self):
        """The launch allocates no persistent buffer, so a capture needs no eager warmup
        and a replay must read the live inputs, not the ones it was captured with."""
        small = _inputs(4)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            q, _ = _write(self.writer, small)
        for name in ("q_nope", "q_pe", "k_nope", "k_pe"):
            small[name].copy_(torch.randn_like(small[name]))
        small["kv_cache"].zero_()
        graph.replay()
        torch.cuda.synchronize()
        expected_q, expected_k = _split_chain(small)
        self.assertTrue(torch.equal(q, expected_q))
        self.assertTrue(torch.equal(small["kv_cache"][:4], expected_k))

    def test_writer_raises_on_a_strided_slot_mapping(self):
        """The kernel reads slot_mapping at stride 1, so a strided one must fail loudly
        rather than land tokens in the wrong slots; the caller makes it contiguous."""
        values = _inputs(16, cache_rows=32)
        values["slot_mapping"] = torch.arange(32, device="cuda", dtype=torch.int64)[::2]
        with self.assertRaises(ValueError):
            _write(self.writer, values)
        torch.cuda.synchronize()
        self.assertEqual(torch.count_nonzero(values["kv_cache"]).item(), 0)

    def test_writer_falls_back_on_caches_the_kernel_does_not_take(self):
        """Caches the kernel asserts on must reach the split chain unlaunched,
        not fail the decode step or graph capture."""
        for cache, kv_cache_shape, kv_cache_dtype in (
            ("fp8_e5m2", (8, 1, 576), torch.float8_e5m2),
            ("paged_4d", (8, 1, 1, 576), torch.bfloat16),
        ):
            with self.subTest(cache=cache):
                values = _inputs(4)
                values["kv_cache"] = torch.zeros(
                    kv_cache_shape, device="cuda", dtype=kv_cache_dtype
                )
                self.assertIsNone(_write(self.writer, values))
                torch.cuda.synchronize()
                self.assertEqual(
                    torch.count_nonzero(values["kv_cache"].view(torch.uint8)).item(), 0
                )


if __name__ == "__main__":
    unittest.main()
