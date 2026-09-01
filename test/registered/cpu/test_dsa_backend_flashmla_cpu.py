"""Unit tests for DeepseekSparseAttnBackend's CPU ``flashmla_cpu`` path.

Exercises the backend construction + KV-cache write path + ``_forward_flashmla_cpu``
dispatch end-to-end for both bf16 and fp8_e4m3 KV caches, comparing against a
pure-PyTorch dense-attention reference computed from the original (unquantized)
K. This covers the CPU fp8 KV-cache quantization write path added in
``memory_pool.py`` / ``quant_k_cache.py`` (Triton kernels are GPU-only, so the
CPU backend routes through a pure-PyTorch quantizer with the same formula).
"""

import unittest

import torch

from sglang.srt.utils import is_cpu
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-b-test-cpu")

try:
    from sgl_kernel.flash_mla import flash_mla_with_kvcache_cpu  # noqa: F401

    _IMPORT_ERROR = None
except Exception as _e:  # pragma: no cover - exercised only when kernel missing
    _IMPORT_ERROR = _e

from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

_KV_LORA_RANK = 512
_QK_ROPE_HEAD_DIM = 64
_QUANT_BLOCK_SIZE = 128  # matches DSATokenToKVPool.quant_block_size (V32FP8Sparse)
_NUM_HEADS = 4


class _FakeLayer:
    layer_id = 0


def _fp8_override_kv_cache_dim() -> int:
    # nope + per-tile fp32 scales + rope stored in bf16 (see calculate_mla_kv_cache_dim).
    num_tiles = _KV_LORA_RANK // _QUANT_BLOCK_SIZE
    return _KV_LORA_RANK + num_tiles * 4 + _QK_ROPE_HEAD_DIM * 2


def _build_backend(kv_cache_dtype: torch.dtype):
    override_dim = (
        _fp8_override_kv_cache_dim() if kv_cache_dtype == torch.float8_e4m3fn else None
    )
    pool = MLATokenToKVPool(
        size=64,
        page_size=1,
        dtype=kv_cache_dtype,
        kv_lora_rank=_KV_LORA_RANK,
        qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
        use_dsa=True,
        override_kv_cache_dim=override_dim,
    )

    hf_config = type(
        "HfConfig",
        (),
        {"architectures": ["DeepseekV3ForCausalLM"], "index_topk": 64},
    )()
    model_config = type(
        "ModelConfig",
        (),
        {
            "context_len": 256,
            "num_attention_heads": _NUM_HEADS,
            "kv_lora_rank": _KV_LORA_RANK,
            "qk_rope_head_dim": _QK_ROPE_HEAD_DIM,
            "qk_nope_head_dim": _KV_LORA_RANK,
            "hf_config": hf_config,
        },
    )()
    req_to_token_pool = type(
        "ReqPool",
        (),
        {"size": 8, "req_to_token": torch.zeros(8, 256, dtype=torch.int32)},
    )()
    server_args = type(
        "MockServerArgs",
        (),
        {
            "enable_deterministic_inference": False,
            "dsa_prefill_backend": "flashmla_cpu",
            "dsa_decode_backend": "flashmla_cpu",
            "dsa_topk_backend": "sgl-kernel",
            "speculative_eagle_topk": None,
            "speculative_algorithm": None,
            "enable_two_batch_overlap": False,
        },
    )()
    runner = type(
        "MockModelRunner",
        (),
        {
            "device": "cpu",
            "page_size": 1,
            "server_args": server_args,
            "model_config": model_config,
            "token_to_kv_pool": pool,
            "req_to_token_pool": req_to_token_pool,
            "hisparse_coordinator": None,
            "max_running_requests": 8,
            "kv_cache_dtype": kv_cache_dtype,
        },
    )()
    return DeepseekSparseAttnBackend(runner), pool


@unittest.skipUnless(
    is_cpu(), "requires SGLANG_USE_CPU_ENGINE=1 on a CPU-engine-supported host"
)
@unittest.skipIf(
    _IMPORT_ERROR is not None,
    f"flash_mla_with_kvcache_cpu unavailable: {_IMPORT_ERROR}",
)
class TestDSABackendFlashMLACPU(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls._parallel_override = get_parallel().override(attn_tp_size=1)
        cls._parallel_override.__enter__()
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    @classmethod
    def tearDownClass(cls):
        cls._parallel_override.__exit__(None, None, None)

    def setUp(self):
        torch.manual_seed(0)

    def _run_decode_case(self, kv_cache_dtype: torch.dtype, atol: float, rtol: float):
        backend, pool = self._build_backend_cached(kv_cache_dtype)
        bs, seq_len, topk = 2, 8, 16
        total_tokens = bs * seq_len

        k_nope = torch.randn(total_tokens, 1, _KV_LORA_RANK, dtype=torch.bfloat16) * 0.3
        k_rope = (
            torch.randn(total_tokens, 1, _QK_ROPE_HEAD_DIM, dtype=torch.bfloat16) * 0.3
        )
        loc = torch.arange(total_tokens, dtype=torch.int64)
        pool.set_mla_kv_buffer(_FakeLayer(), loc, k_nope, k_rope)

        q_nope = torch.randn(bs, _NUM_HEADS, _KV_LORA_RANK, dtype=torch.bfloat16) * 0.3
        q_rope = (
            torch.randn(bs, _NUM_HEADS, _QK_ROPE_HEAD_DIM, dtype=torch.bfloat16) * 0.3
        )

        page_table_1 = torch.full((bs, topk), -1, dtype=torch.int32)
        for b in range(bs):
            page_table_1[b, :seq_len] = torch.arange(
                b * seq_len, (b + 1) * seq_len, dtype=torch.int32
            )
        topk_length = torch.full((bs,), seq_len, dtype=torch.int32)

        kv_cache = pool.get_key_buffer(0)
        sm_scale = 1.0 / (_KV_LORA_RANK + _QK_ROPE_HEAD_DIM) ** 0.5

        out = backend._forward_flashmla_cpu(
            q_nope=q_nope,
            q_rope=q_rope,
            kv_cache=kv_cache,
            v_head_dim=_KV_LORA_RANK,
            page_table_1=page_table_1,
            topk_length=topk_length,
            sm_scale=sm_scale,
        )
        self.assertEqual(out.shape, (bs, _NUM_HEADS, _KV_LORA_RANK))

        # Dense-attention reference against the ORIGINAL (unquantized) K, since
        # every position is within topk_length (no masking) for this case.
        q_all = torch.cat([q_nope, q_rope], dim=-1).float()
        k_all = torch.cat([k_nope, k_rope], dim=-1).float().squeeze(1)
        ref_out = torch.empty(bs, _NUM_HEADS, _KV_LORA_RANK)
        for b in range(bs):
            k_b = k_all[b * seq_len : (b + 1) * seq_len]
            logits = torch.einsum("hd,sd->hs", q_all[b], k_b) * sm_scale
            probs = torch.softmax(logits, dim=-1)
            ref_out[b] = probs @ k_b[:, :_KV_LORA_RANK]

        torch.testing.assert_close(out.float(), ref_out, atol=atol, rtol=rtol)

    def _build_backend_cached(self, kv_cache_dtype: torch.dtype):
        # Fresh pool/backend per case: KV writes must not leak across dtypes.
        return _build_backend(kv_cache_dtype)

    def test_decode_bf16_kv_cache(self):
        self._run_decode_case(torch.bfloat16, atol=0.02, rtol=0.02)

    def test_decode_fp8_kv_cache(self):
        # Looser tolerance: fp8_e4m3 per-128-tile quantization of K introduces
        # attention-output error on top of the bf16 baseline.
        self._run_decode_case(torch.float8_e4m3fn, atol=0.05, rtol=0.05)

    def test_fp8_write_path_roundtrip(self):
        """Directly validates the CPU fp8 KV-cache quantization write path
        (memory_pool.py's dsa_kv_cache_store_fp8 branch + quant_k_cache.py's
        pure-PyTorch quantizer) by dequantizing the written bytes and
        comparing against the original bf16 K.
        """
        _backend, pool = _build_backend(torch.float8_e4m3fn)
        total_tokens = 8
        k_nope = torch.randn(total_tokens, 1, _KV_LORA_RANK, dtype=torch.bfloat16) * 0.3
        k_rope = (
            torch.randn(total_tokens, 1, _QK_ROPE_HEAD_DIM, dtype=torch.bfloat16) * 0.3
        )
        loc = torch.arange(total_tokens, dtype=torch.int64)
        pool.set_mla_kv_buffer(_FakeLayer(), loc, k_nope, k_rope)

        kv_cache = pool.get_key_buffer(0)[:total_tokens].squeeze(1)  # fp8, (T, 656)
        num_tiles = _KV_LORA_RANK // _QUANT_BLOCK_SIZE
        nope_fp8 = kv_cache[:, :_KV_LORA_RANK].float()
        scales = kv_cache[:, _KV_LORA_RANK : _KV_LORA_RANK + num_tiles * 4].view(
            torch.float32
        )
        rope_bytes = kv_cache[:, _KV_LORA_RANK + num_tiles * 4 :].view(torch.bfloat16)

        dequant_nope = torch.empty(total_tokens, _KV_LORA_RANK)
        for t in range(num_tiles):
            sl = slice(t * _QUANT_BLOCK_SIZE, (t + 1) * _QUANT_BLOCK_SIZE)
            dequant_nope[:, sl] = nope_fp8[:, sl] * scales[:, t : t + 1]

        # Rope is stored losslessly (raw bf16 bytes, no quantization).
        torch.testing.assert_close(rope_bytes, k_rope.squeeze(1))
        torch.testing.assert_close(
            dequant_nope, k_nope.squeeze(1).float(), atol=0.05, rtol=0.05
        )


if __name__ == "__main__":
    unittest.main()
