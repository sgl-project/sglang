"""Regression test for PR #32208 / base-b-test-1-gpu-large CI failure.

FlashAttentionBackend's prefill-aware-SWA scratch buffer
(``_pa_swa_prefill_lens``) is indexed directly by raw ``req_pool_idx`` values
(``self._pa_swa_prefill_lens[forward_batch.req_pool_indices[:batch_size]] =
...``), not by batch position. ``ReqToTokenPool`` reserves row 0 as a padding
slot and hands out real slots ``1..size``, so the valid index range is
``[0, size]`` -- the buffer must hold ``size + 1`` elements, not ``size``.

Sizing it to ``size`` (the pre-fix code) is off-by-one: writing at
``req_pool_idx == size`` overflows the buffer, which crashes on CUDA with an
``index_put`` "index out of bounds" device assertion. This was latent under
the old head-first ``ReqToTokenPool.alloc()`` (index ``size`` was only
reachable once the pool was nearly saturated) and became immediately
reachable once ``alloc()`` switched to popping free slots from the tail
(``memory_pool.py``, "O(1) slot allocation in ReqToTokenPool.alloc()"): the
very first request into a fresh pool gets ``req_pool_idx == size``.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def _make_prefill_aware_swa_runner(*, pool_size: int, max_context_len: int = 64):
    """A minimal fake ModelRunner that reaches FlashAttentionBackend.__init__'s
    is_prefill_aware_swa branch (mirrors how models like
    python/sglang/srt/models/unlimited_ocr.py opt in)."""
    device = "cuda"
    req_to_token_pool = SimpleNamespace(
        size=pool_size,
        req_to_token=torch.zeros(
            pool_size + 1, max_context_len, dtype=torch.int32, device=device
        ),
    )
    model_config = SimpleNamespace(
        is_encoder_decoder=False,
        context_len=max_context_len,
        attention_arch=AttentionArch.MHA,
        is_local_attention_model=False,
        head_dim=8,
        hf_text_config=SimpleNamespace(
            num_attention_heads=2, attn_logit_softcapping=None
        ),
        get_num_kv_heads=lambda tp_size: 2,
    )
    server_args = SimpleNamespace(
        speculative_eagle_topk=None,
        enable_deterministic_inference=False,
        is_embedding=False,
        chunked_prefill_size=8192,
        disable_radix_cache=False,
        enable_prefill_cp=False,
        enable_dp_attention=False,
    )
    return SimpleNamespace(
        sliding_window_size=None,
        model_config=model_config,
        device=device,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool=object(),  # not a SWAKVPool instance -> use_sliding_window_kv_pool=False
        # getattr(..., "full_v2p_page_table", None) is None -> unified_mla_hooks
        # falls back to the static (disabled) hook set.
        token_to_kv_pool_allocator=object(),
        kv_cache_dtype=torch.float16,
        kv_cache_dtype_str="auto",
        page_size=1,
        ps=SimpleNamespace(attn_cp_size=1, tp_size=1),
        is_draft_worker=False,
        server_args=server_args,
        attention_chunk_size=None,
        prefill_aware_swa=True,
    )


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestPrefillAwareSwaPrefillLensBound(CustomTestCase):
    def test_buffer_covers_full_req_pool_idx_range(self):
        pool_size = 8
        runner = _make_prefill_aware_swa_runner(pool_size=pool_size)

        # __init__ reads get_spec().speculative_num_draft_tokens, which comes
        # from the published runtime-context config bag, not model_runner /
        # server_args -- publish a real (dummy) ServerArgs rather than faking
        # the accessor (see the sglang-runtime-context skill).
        with get_context().override_server_args():
            backend = FlashAttentionBackend(runner)

        # req_pool_idx ranges over [0, pool_size] inclusive (row 0 is the
        # reserved CUDA-graph padding slot; real requests use 1..pool_size).
        self.assertEqual(backend._pa_swa_prefill_lens.shape[0], pool_size + 1)

        # This mirrors the exact write that crashed in CI: writing at the
        # maximum valid req_pool_idx must stay in bounds.
        max_req_pool_idx = torch.tensor([pool_size], device=runner.device)
        backend._pa_swa_prefill_lens[max_req_pool_idx] = 7
        self.assertEqual(backend._pa_swa_prefill_lens[pool_size].item(), 7)


if __name__ == "__main__":
    unittest.main()
