"""Real packed DSA kernels, including changing layouts under CUDA graph replay."""

import unittest
from dataclasses import fields
from itertools import accumulate
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsa.paged_mqa_logits import (
    cutedsl_paged_mqa_logits,
    deepgemm_paged_mqa_logits_split,
)
from sglang.kernels.ops.quantization.dsa_quant_k_cache import (
    fp8_ds_mla_row_width,
    quantize_k_cache,
)
from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_indexer_metadata import DSAIndexerMetadata
from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    DSATopKBackend,
    TopkTransformMethod,
)
from sglang.srt.layers.attention.dsa.ragged_verify import update_ragged_dsa_query_rows
from sglang.srt.layers.attention.dsa.utils import (
    resolve_fp8_paged_mqa_logits_fn,
    resolve_num_sms,
    resolve_paged_mqa_logits_metadata_fn,
)
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# The CuTe DSL paged-logits cases need SM100.
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

# TOPK: FlashMLA's SM100 128-head sparse kernel walks top-k in blocks of 128.
SLOTS, TIER, WIDTH, TOPK = 8, 16, 256, 128


def _paged_index_cache_view(packed):
    # DeepGEMM consumes the production pool's raw page bytes, including FP32
    # scales. Reinterpreting this tensor as FP8 violates its fused-cache ABI.
    if packed.dtype != DSATokenToKVPool.index_k_with_scale_buffer_dtype:
        raise ValueError("paged index cache must retain the production byte dtype")
    return packed.view(
        packed.shape[0], 64, 1, DSATokenToKVPool.index_bytes_per_token_for(128)
    )


def _k31_model_config():
    # Dimensions from moonshotai/Kimi-K3.1-Pro's pinned text config. NoPE
    # disables rotation; its 64-dimensional unrotated tail remains in KV.
    text = KimiLinearConfig(
        hidden_size=7168,
        num_attention_heads=96,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mla_use_nope=True,
        linear_attn_config={"kda_layers": [1], "full_attn_layers": [0]},
        dsa_config={
            "dsa_layers": [0],
            "index_topk": 2048,
            "index_head_dim": 128,
            "index_num_heads": 32,
            "index_rope_dim": 0,
        },
    )
    config = object.__new__(ModelConfig)
    config.hf_config = SimpleNamespace(
        architectures=["KimiK31ForConditionalGeneration"],
        model_type="kimi_k31",
        text_config=text,
    )
    config.hf_text_config = text
    config._derive_model_shapes()
    return config


def _backend(*, width=WIDTH, topk=TOPK, k31=False):
    backend = object.__new__(DeepseekSparseAttnBackend)
    backend.device = torch.device("cuda")
    backend.device_sm_major = torch.cuda.get_device_capability()[0]
    backend.attn_dcp_size, backend.attn_dcp_rank = 1, 0
    backend.dsa_index_kpool, backend.dsa_index_topk = 1, topk
    backend.hisparse_coordinator = backend._kv_shard_pool = None
    backend.dsa_decode_impl = "flashmla_kv" if k31 else "flashmla_sparse"
    backend.dsa_kv_cache_store_fp8 = k31
    backend.kv_cache_dtype = torch.float8_e4m3fn if k31 else torch.bfloat16
    backend.dsa_shards_index_k = False
    backend.kv_cache_dim = 656 if k31 else 576
    backend.kv_scale_ue8m0 = k31
    backend.flashmla_kv_format = "V32"
    backend.flashmla_kv_num_q_heads = backend.flashmla_kv_num_q_heads_prefill = 64
    backend.physical_page_size = backend.real_page_span = 64
    backend.dsa_topk_backend = DSATopKBackend.SGL_KERNEL
    backend.speculative_num_draft_tokens = 6
    backend.set_dsa_prefill_impl = lambda batch: None
    backend._arange_buf = torch.arange(TIER + 1, device="cuda", dtype=torch.int32)
    backend.req_to_token = torch.arange(
        SLOTS * width, device="cuda", dtype=torch.int32
    ).view(SLOTS, width)
    backend.decode_cuda_graph_metadata = {
        "page_table": torch.zeros(TIER, width, device="cuda", dtype=torch.int32)
    }
    backend.ragged_verify_cuda_graph_metadata = {}
    backend._sink_pad_cache = {}
    backend._init_kpool_metadata_fusion()
    return backend


def _batch(prefix, requests, verify):
    # Poison padded request slots: an accidental last-request continuation or
    # use of captured slots instead of live indptr would read out of bounds.
    seq = torch.full((SLOTS,), 1000000, device="cuda", dtype=torch.int64)
    req = torch.full((SLOTS,), 1000000, device="cuda", dtype=torch.int64)
    seq[: len(prefix)] = torch.tensor(prefix, device="cuda")
    req[: len(requests)] = torch.tensor(requests, device="cuda")
    layout = RaggedVerifyLayout.from_verify_lens(
        verify_lens_cpu=verify, device=torch.device("cuda"), grid=[TIER]
    )
    return SimpleNamespace(
        batch_size=SLOTS,
        seq_lens=seq,
        req_pool_indices=req,
        forward_mode=ForwardMode.TARGET_VERIFY,
        ragged_verify_layout=layout,
    )


def _addresses(metadata):
    def storage(tensor):
        return (
            tensor.data_ptr(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            str(tensor.dtype),
        )

    result = {
        f.name: storage(value)
        for f in fields(metadata)
        if isinstance(value := getattr(metadata, f.name), torch.Tensor)
    }
    if metadata.flashmla_metadata is not None:
        result.update(
            {
                "flashmla." + f.name: storage(
                    getattr(metadata.flashmla_metadata, f.name)
                )
                for f in fields(metadata.flashmla_metadata)
            }
        )
    return result


class TestDSARaggedVerifyCapability(CustomTestCase):
    def test_unsupported_configs_cannot_enter_ragged_metadata(self):
        backend = object.__new__(DeepseekSparseAttnBackend)
        supported = dict(
            dsa_decode_impl="flashmla_sparse",
            dsa_kv_cache_store_fp8=False,
            kv_cache_dtype=torch.bfloat16,
            attn_dcp_size=1,
            dsa_index_kpool=1,
            hisparse_coordinator=None,
            _kv_shard_pool=None,
        )
        for name, value in supported.items():
            setattr(backend, name, value)
        with patch(
            "sglang.srt.layers.attention.dsa_backend.is_cuda", return_value=True
        ):
            self.assertTrue(backend.supports_ragged_verify_graph)
            backend.dsa_decode_impl = "flashmla_kv"
            backend.dsa_kv_cache_store_fp8 = True
            backend.kv_cache_dtype = torch.float8_e4m3fn
            self.assertTrue(backend.supports_ragged_verify_graph)
            backend.dsa_decode_impl = "flashmla_sparse"
            backend.dsa_kv_cache_store_fp8 = False
            backend.kv_cache_dtype = torch.bfloat16
            for name, value in (
                ("dsa_decode_impl", "trtllm"),
                ("dsa_kv_cache_store_fp8", True),
                ("kv_cache_dtype", torch.float8_e4m3fn),
                ("attn_dcp_size", 2),
                ("dsa_index_kpool", 4),
                ("hisparse_coordinator", object()),
                ("_kv_shard_pool", object()),
            ):
                with self.subTest(name=name, value=value):
                    setattr(backend, name, value)
                    self.assertFalse(backend.supports_ragged_verify_graph)
                    with self.assertRaisesRegex(NotImplementedError, "flashmla_sparse"):
                        backend._init_ragged_verify_metadata(None, cuda_graph=False)
                    setattr(backend, name, supported[name])
        with patch(
            "sglang.srt.layers.attention.dsa_backend.is_cuda", return_value=False
        ):
            self.assertFalse(backend.supports_ragged_verify_graph)

    def test_k31_nope_keeps_the_unrotated_tail_and_attention_scale(self):
        config = _k31_model_config()
        self.assertTrue(config.hf_text_config.mla_use_nope)
        self.assertEqual(config.get_num_attention_heads(8), 12)
        self.assertEqual(config.qk_rope_head_dim, 64)
        self.assertEqual(config.kv_lora_rank + config.qk_rope_head_dim, 576)
        self.assertEqual(
            fp8_ds_mla_row_width(config.kv_lora_rank, config.qk_rope_head_dim), 656
        )
        self.assertAlmostEqual(config.scaling, 192**-0.5)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestDSARaggedVerifyCUDA(CustomTestCase):
    def test_empty_request_slots_skip_poisoned_owners(self):
        prefix = torch.tensor([62, 1000000, 127, 1000000], device="cuda")
        requests = torch.tensor([7, 1000000, 3, 1000000], device="cuda")
        layout = SimpleNamespace(
            graph_num_tokens=8,
            qo_indptr_device=torch.tensor(
                [0, 1, 1, 4, 4], device="cuda", dtype=torch.int32
            ),
        )
        lengths = torch.empty(8, device="cuda", dtype=torch.int32)
        owners = torch.empty(8, device="cuda", dtype=torch.int64)
        update_ragged_dsa_query_rows(prefix, requests, layout, lengths, owners)
        self.assertEqual(lengths.tolist(), [63, 128, 129, 130, 0, 0, 0, 0])
        self.assertEqual(owners.tolist(), [7, 3, 3, 3, 0, 0, 0, 0])

    def test_native_indexer_sparse_attention_and_graph_replay(self):
        if torch.cuda.get_device_capability()[0] < 9:
            self.skipTest("native FP8 DSA requires Hopper or newer")
        with (
            envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(False),
            envs.SGLANG_DSA_FUSE_TOPK.override(True),
            envs.SGLANG_OPT_USE_TOPK_V2.override(True),
        ):
            self._check_native_graph("deepgemm")

    def test_cutedsl_indexer_sparse_attention_and_graph_replay(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("CuTe DSL paged DSA requires Blackwell")
        with (
            envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(False),
            envs.SGLANG_DSA_FUSE_TOPK.override(True),
            envs.SGLANG_OPT_USE_TOPK_V2.override(True),
        ):
            self._check_native_graph("cutedsl")

    def test_k31_fp8_paged_attention_and_graph_replay(self):
        if torch.cuda.get_device_capability()[0] < 9:
            self.skipTest("native FP8 DSA requires Hopper or newer")
        with (
            envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(False),
            envs.SGLANG_DSA_FUSE_TOPK.override(True),
            envs.SGLANG_OPT_USE_TOPK_V2.override(True),
        ):
            self._check_native_graph("deepgemm", k31=True)

    def test_k31_fp8_cutedsl_paged_attention_and_graph_replay(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("CuTe DSL paged DSA requires Blackwell")
        with (
            envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(False),
            envs.SGLANG_DSA_FUSE_TOPK.override(True),
            envs.SGLANG_OPT_USE_TOPK_V2.override(True),
        ):
            self._check_native_graph("cutedsl", k31=True)

    def _check_native_graph(self, logits_backend, *, k31=False):
        torch.manual_seed(19)
        config = _k31_model_config() if k31 else None
        width, topk = (4096, config.hf_text_config.index_topk) if k31 else (WIDTH, TOPK)
        heads = config.get_num_attention_heads(8) if k31 else 16
        index_heads = config.hf_text_config.index_n_heads if k31 else 64
        q_dim = config.kv_lora_rank + config.qk_rope_head_dim if k31 else 576
        sm_scale = config.scaling if k31 else 576**-0.5
        long_prefix = 4090 if k31 else 127
        backend = _backend(width=width, topk=topk, k31=k31)
        first = _batch([0, 62, long_prefix], [5, 2, 7], [1, 3, 6])
        backend._init_ragged_verify_metadata(first, cuda_graph=True)
        metadata = backend.forward_metadata
        pointers = _addresses(metadata)
        num_pages = SLOTS * width // 64
        index_k = torch.randn(num_pages, 64, 128, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.ones(num_pages, 64, device="cuda", dtype=torch.float32)
        packed = torch.empty(num_pages, 64 * 132, device="cuda", dtype=torch.uint8)
        packed[:, : 64 * 128] = index_k.view(torch.uint8).reshape(num_pages, -1)
        packed[:, 64 * 128 :] = scales.view(torch.uint8).reshape(num_pages, -1)
        index_cache = _paged_index_cache_view(packed)
        index_q = torch.randn(TIER, index_heads, 128, device="cuda").to(
            torch.float8_e4m3fn
        )
        weights = torch.rand(TIER, index_heads, device="cuda")
        q = torch.randn(TIER, heads, q_dim, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(SLOTS * width, 1, q_dim, device="cuda", dtype=torch.bfloat16)
        sink = None if k31 else torch.randn(heads, device="cuda", dtype=torch.float32)
        if k31:
            kv = quantize_k_cache(
                kv.view(num_pages, 64, 1, q_dim),
                dv=config.kv_lora_rank,
                scale_ue8m0=True,
            ).view(SLOTS * width, 1, -1)
            self.assertEqual(kv.shape[-1], 656)
            raw = kv.view(torch.uint8).view(SLOTS * width, 656)
            values = raw[:, :512].contiguous().view(torch.float8_e4m3fn).float()
            scales = raw[:, 512:528].contiguous().view(torch.float32)
            tail = raw[:, 528:656].contiguous().view(torch.bfloat16).float()
            reference_kv = torch.cat(
                [values * scales.repeat_interleave(128, dim=1), tail], dim=1
            )
        else:
            reference_kv = kv[:, 0].float()
        layer = SimpleNamespace(tp_q_head_num=heads, head_dim=q_dim)

        def forward(meta=metadata):
            indexer_metadata = DSAIndexerMetadata(
                meta, TopkTransformMethod.PAGED, topk_backend=DSATopKBackend.SGL_KERNEL
            )
            if logits_backend == "cutedsl":
                logits = cutedsl_paged_mqa_logits(
                    index_q,
                    index_cache,
                    weights,
                    meta.cache_seqlens_int32,
                    meta.real_page_table,
                    meta.paged_mqa_schedule_metadata,
                    width,
                    q_offset=TIER,
                    B=TIER,
                    next_n=1,
                    is_target_verify=True,
                    dsl_expand_factor=1,
                    dsl_atom=1,
                    blocksize=64,
                    sm_count=resolve_num_sms(),
                    get_paged_mqa_logits_metadata_fn=resolve_paged_mqa_logits_metadata_fn(),
                )
            else:
                logits = deepgemm_paged_mqa_logits_split(
                    resolve_fp8_paged_mqa_logits_fn(),
                    index_q,
                    index_cache,
                    weights,
                    meta.paged_mqa_ctx_lens_2d,
                    meta.real_page_table,
                    meta.paged_mqa_schedule_metadata,
                    width,
                    q_offset=TIER,
                )
            selected = indexer_metadata.topk_transform(logits, topk)
            if k31:
                output = backend._forward_flashmla_kv(
                    q, kv, 512, sm_scale, layer, meta, selected, first
                ).view(TIER, heads, 512)
            else:
                output = backend._forward_flashmla_sparse(
                    q,
                    kv,
                    512,
                    selected,
                    sm_scale,
                    topk_length=meta.dsa_cache_seqlens_int32,
                    attn_sink=sink,
                )
            return selected, output, logits

        # Warm the actual native kernels before capture; no replacements or
        # synthetic logits are used in this graph.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                forward()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            graph_topk, graph_output, graph_logits = forward()
        torch.cuda.current_stream().wait_stream(stream)

        for prefix, requests, verify in (
            ([0, 62, long_prefix], [5, 2, 7], [1, 3, 6]),
            ([long_prefix], [3], [1]),
            ([63, long_prefix], [7, 2], [6, 6]),
            ([0, 62, long_prefix], [2, 7, 5], [6, 1, 3]),
        ):
            batch = _batch(prefix, requests, verify)
            backend._init_ragged_verify_metadata(batch, cuda_graph=True)
            self.assertIs(backend.forward_metadata, metadata)
            self.assertEqual(_addresses(metadata), pointers)
            expected_lengths = [
                prefix[r] + j + 1 for r, n in enumerate(verify) for j in range(n)
            ]
            live_tokens = len(expected_lengths)
            self.assertEqual(
                metadata.cache_seqlens_int32.tolist(),
                expected_lengths + [0] * (TIER - live_tokens),
            )
            padded_lengths = expected_lengths + [0] * (TIER - live_tokens)
            self.assertEqual(
                metadata.cu_seqlens_k.tolist(), [0, *accumulate(padded_lengths)]
            )
            self.assertEqual(
                metadata.dsa_cu_seqlens_k.tolist(),
                [0, *accumulate(min(n, topk) for n in padded_lengths)],
            )
            self.assertEqual(metadata.cu_seqlens_q.tolist(), list(range(TIER + 1)))
            backend._init_ragged_verify_metadata(batch, cuda_graph=False)
            eager_topk, eager_output, _ = forward(backend.forward_metadata)
            backend.forward_metadata = metadata
            graph.replay()
            # Folded top-k emits an unordered set through atomic counters.
            # Equivalent sets can change the attention reduction's last bits.
            torch.testing.assert_close(
                graph_topk.sort(dim=-1).values,
                eager_topk.sort(dim=-1).values,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                graph_output, eager_output, rtol=0.015, atol=0.015
            )
            self.assertTrue(torch.isfinite(graph_output).all())
            self.assertTrue((graph_topk[live_tokens:] == -1).all())
            self.assertTrue((graph_output[live_tokens:] == 0).all())

            # Independent causal reference validates page ownership as well as
            # the selected sparse attention, including the learned sink term.
            for token, length in enumerate(expected_lengths):
                row = next(
                    r for r in range(len(verify)) if token < sum(verify[: r + 1])
                )
                chosen = graph_topk[token]
                chosen = chosen[chosen >= 0].long()
                logical = chosen - requests[row] * width
                self.assertTrue(((logical >= 0) & (logical < length)).all())
                self.assertEqual(chosen.numel(), min(length, topk))
                index_keys = index_k.view(SLOTS * width, 128)[
                    requests[row] * width : requests[row] * width + length
                ].float()
                index_scores = (index_q[token].float() @ index_keys.T).relu()
                index_reference = (index_scores * weights[token, :, None]).sum(0)
                torch.testing.assert_close(
                    graph_logits[token, :length],
                    index_reference,
                    atol=0.003,
                    rtol=0.003,
                )
                keys = reference_kv[chosen]
                scores = (q[token].float() @ keys.T) * sm_scale
                if sink is None:
                    probs = torch.softmax(scores, dim=1)
                else:
                    probs = torch.softmax(
                        torch.cat([scores, sink[:, None]], dim=1), dim=1
                    )[:, :-1]
                reference = probs @ keys[:, :512]
                torch.testing.assert_close(
                    graph_output[token].float(), reference, atol=0.025, rtol=0.025
                )


if __name__ == "__main__":
    unittest.main()
