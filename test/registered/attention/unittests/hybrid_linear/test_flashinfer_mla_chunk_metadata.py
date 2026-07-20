"""Regression test: HybridLinearAttnBackend delegates init_mha_chunk_metadata.

Hybrid MLA models (Ring/Ling, Kimi-Linear) run DeepSeek-style MLA on their
full-attention layers. Prefill plans the flashinfer ragged wrapper via
``hasattr(get_attn_backend(), "init_mha_chunk_metadata")`` (forward_mha.py).
For a hybrid model ``get_attn_backend()`` is the HybridLinearAttnBackend wrapper;
when it lacked the hook the guard was False, qo_indptr/kv_indptr were never
planned, and flashinfer raised "q.shape[0] does not match qo_indptr[-1]".

The per-method MLA suite misses this: it drives the bare backend, and its mock
runner sets disable_chunked_prefix_cache=True + flashinfer_mla_disable_ragged=True
so the chunked-MHA path never runs.
"""

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.mla_attention import (
    DEFAULT_KV_LORA_RANK,
    DEFAULT_MAX_CONTEXT_LEN,
    MLAAttentionCase,
    MockMLAModelRunner,
    TinyMLAModelConfig,
    _make_forward_batch,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-large")


_KV_LORA_RANK = DEFAULT_KV_LORA_RANK
_QK_ROPE_HEAD_DIM = 0


class _ChunkKVMLARunner(MockMLAModelRunner):
    """MLA mock runner with chunked-prefix-cache (ragged MHA) enabled."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The fixture's config is already published; adjust it through the
        # audited entry point (bare writes raise under the strict guard).
        self.server_args.override(
            source="attention-unittest",
            disable_chunked_prefix_cache=False,
            flashinfer_mla_disable_ragged=False,
        )


def _make_case() -> MLAAttentionCase:
    return MLAAttentionCase(
        name="hybrid_one_shot_prefix",
        backend="flashinfer",
        forward_mode=ForwardMode.EXTEND,
        num_heads=4,
        page_size=1,
        prefix_lens=(8, 4),
        extend_lens=(5, 3),
    )


def _build_hybrid_backend(testcase, case: MLAAttentionCase):
    model_config = TinyMLAModelConfig(
        num_heads=case.num_heads,
        kv_lora_rank=_KV_LORA_RANK,
        qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
        hidden_size=64,
        context_len=DEFAULT_MAX_CONTEXT_LEN,
    )
    runner = _ChunkKVMLARunner(
        case=case,
        model_config=model_config,
        dtype=torch.float16,
        device="cuda",
        max_context_len=DEFAULT_MAX_CONTEXT_LEN,
        kv_lora_rank=_KV_LORA_RANK,
        qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
        disable_cuda_graph=True,
        disable_piecewise_cuda_graph=True,
        runner_batch_size=None,
        fp8_kv_cache=False,
    )
    try:
        from sglang.srt.layers.attention.flashinfer_mla_backend import (
            FlashInferMLAAttnBackend,
        )

        full_backend = FlashInferMLAAttnBackend(runner)
    except (AssertionError, ImportError, ModuleNotFoundError) as exc:
        testcase.skipTest(f"flashinfer MLA backend is not available: {exc}")

    if not getattr(full_backend, "enable_chunk_kv", False):
        testcase.skipTest("chunk-KV path not enabled on this build")

    # full_attn_layers=[0] routes layer 0 through the MLA backend, as a hybrid
    # model's full-attention layers do. The linear backend is unused here.
    hybrid = HybridLinearAttnBackend(
        full_backend, SimpleNamespace(), full_attn_layers=[0]
    )
    return runner, full_backend, hybrid


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestHybridLinearChunkMetadataDelegation(CustomTestCase):
    def test_wrapper_exposes_chunk_metadata_hook(self):
        _, full_backend, hybrid = _build_hybrid_backend(self, _make_case())
        self.assertTrue(hasattr(hybrid, "init_mha_chunk_metadata"))
        self.assertTrue(hasattr(full_backend, "init_mha_chunk_metadata"))

    def test_delegation_plans_qo_indptr(self):
        case = _make_case()
        runner, full_backend, hybrid = _build_hybrid_backend(self, case)

        forward_batch = _make_forward_batch(
            case,
            runner,
            max_context_len=DEFAULT_MAX_CONTEXT_LEN,
            device="cuda",
        )
        forward_batch.num_prefix_chunks = 0

        bs = case.batch_size
        sentinel = 999
        full_backend.qo_indptr[bs] = sentinel

        # disable_flashinfer_ragged=True plans qo_indptr without flashinfer calls.
        hybrid.init_mha_chunk_metadata(forward_batch, disable_flashinfer_ragged=True)

        # qo_indptr[-1] must equal the extend (query) token count; a stale value
        # here is the root cause of the q.shape[0] != qo_indptr[-1] crash.
        planned = full_backend.mha_chunk_kv_cache.qo_indptr[bs].item()
        self.assertEqual(planned, case.num_input_tokens)
        self.assertNotEqual(planned, sentinel)


if __name__ == "__main__":
    sys_path_parent = str(Path(__file__).resolve().parents[1])
    import sys

    sys.path.insert(0, sys_path_parent)
    unittest.main()
