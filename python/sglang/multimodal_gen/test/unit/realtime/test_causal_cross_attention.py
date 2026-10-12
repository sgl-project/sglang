# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.kvcache.causal_attention_cache import (
    CrossAttentionKVCache,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.lingbot_world import (
    CausalLingBotWorldTransformerBlock,
)
from sglang.multimodal_gen.runtime.models.dits.longlive2 import (
    LongLive2CausalWanTransformerBlock,
)
from sglang.multimodal_gen.runtime.models.dits.wanvideo import WanT2VCrossAttention
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "cls", [LongLive2CausalWanTransformerBlock, CausalLingBotWorldTransformerBlock]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_causal_cross_attention_cache(cls, dtype, single_process_model_parallel):
    block = cls.__new__(cls)
    torch.nn.Module.__init__(block)
    block.attn2 = WanT2VCrossAttention(
        64, 4, supported_attention_backends={AttentionBackendEnum.TORCH_SDPA}
    ).to(device="cuda", dtype=dtype)
    for parameter in block.parameters():
        parameter.normal_(std=0.02)
    hidden = torch.randn(2, 7, 64, device="cuda", dtype=dtype)
    context = torch.randn(2, 5, 64, device="cuda", dtype=dtype)
    cache = CrossAttentionKVCache(torch.empty(0), torch.empty(0))
    with set_forward_context(current_timestep=0, attn_metadata=None):
        expected = block.attn2(hidden, context, None)
        with patch.object(
            block.attn2.to_k, "forward", wraps=block.attn2.to_k.forward
        ) as project_k:
            first = block._cross_attn_with_cache(hidden, context, cache)
            keys, values = cache.k, cache.v
            repeated = block._cross_attn_with_cache(hidden, context + 1, cache)
            assert project_k.call_count == 1
            assert cache.k is keys and cache.v is values
            torch.testing.assert_close(first, expected, rtol=0, atol=0)
            torch.testing.assert_close(repeated, expected, rtol=0, atol=0)
            cache.reset()
            refreshed = block._cross_attn_with_cache(hidden, context + 1, cache)
            assert project_k.call_count == 2
        expected = block.attn2(hidden, context + 1, None)
        torch.testing.assert_close(refreshed, expected, rtol=0, atol=0)
        uncached = block._cross_attn_with_cache(hidden, context, None)
        torch.testing.assert_close(uncached, first, rtol=0, atol=0)
