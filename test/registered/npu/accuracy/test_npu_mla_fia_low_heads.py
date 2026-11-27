"""Single-card Ascend 950 accuracy for the framework's paged MLA decode path."""

import math
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=90, suite="nightly-acc-test-4-npu-a5", nightly=True)


@pytest.fixture(scope="module")
def a5():
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    from sglang.srt.hardware_backend.npu.attention.mla_preprocess import is_fia_nz

    assert get_hardware().generation is Generation.ARCH35
    with pytest.MonkeyPatch.context() as env:
        env.setenv("SGLANG_USE_FIA_NZ", "0")
        is_fia_nz.cache_clear()
        yield torch.device("npu")
    is_fia_nz.cache_clear()


@pytest.mark.parametrize("heads", range(1, 9), ids=lambda n: f"{n}q-1kv")
@pytest.mark.parametrize("modelslim", [False, True], ids=["bf16-fia-v1", "fp8-fia-v2"])
def test_paged_mla_decode(a5, heads, modelslim):
    from sgl_kernel_npu.operators import attention

    from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
        AscendAttnBackend,
        ForwardMetadata,
    )
    from sglang.srt.layers.quantization.modelslim.schemes import (
        ModelSlimQFP8DynamicKVFP8Scheme,
    )
    from sglang.srt.layers.radix_attention import RadixAttention

    # DeepSeek-V2 MLA uses the pre-absorption QK scale, including YaRN.
    latent, rope, page = 512, 64, 128
    scale = 192**-0.5 * (1 + 0.1 * 0.707 * math.log(40)) ** 2
    layer = RadixAttention(
        heads, latent + rope, scale, 1, layer_id=0, v_head_dim=latent
    )
    lengths = [255, 129]
    table = torch.tensor([[2, 1], [4, 3]], dtype=torch.int32, device="cpu")
    generator = torch.Generator(device="cpu").manual_seed(42)

    def randn(*shape):
        return torch.randn(shape, device="cpu", generator=generator).to(torch.bfloat16)

    query = randn(2, heads, latent + rope)
    cache = randn(5, page, 1, latent + rope)
    q, q_rope = (t.contiguous().to(a5) for t in query.split((latent, rope), dim=-1))
    kv, k_rope = (t.contiguous().to(a5) for t in cache.split((latent, rope), dim=-1))
    ref_q_rope = query[..., latent:].float()
    q_descale = None
    if modelslim:
        scheme = ModelSlimQFP8DynamicKVFP8Scheme("self_attn", "fp8_e4m3")
        scheme.create_weights(
            layer,
            kv_lora_rank=latent,
            qk_rope_head_dim=rope,
            params_dtype=torch.bfloat16,
        )
        layer.to(a5)
        layer.fa_k_scale.data.fill_(1 / 32)
        layer.fa_k_scale.modelslim_loaded = True
        scheme.process_weights_after_loading(layer)
        layer.scheme = scheme
        q, q_descale = scheme.quantize_query(q)
        kv, k_rope, _ = scheme.quantize_cache(kv, k_rope)
        # Match the BF16 RoPE compensation in DeepSeek's MLA prepare step.
        q_rope = (q_rope.float() / (q_descale * scheme.k_scale)).bfloat16()
        ref_q = q.float().cpu() * q_descale.cpu()
        ref_kv = kv.float().cpu() * scheme.k_scale.cpu()
        operator = "fused_infer_attention_score_v2"
    else:
        ref_q = query[..., :latent].float()
        ref_kv = cache[..., :latent].float()
        operator = "fused_infer_attention_score"

    # Only the model runner's pool and its batch metadata are supplied by the
    # test. Dispatch, head padding, the registered FIA kernel and cropping run.
    backend = AscendAttnBackend.__new__(AscendAttnBackend)
    backend.use_mla = backend.use_fia = True
    backend.graph_mode = backend.dense_mla_dcp_target = False
    backend.page_size, backend.kv_lora_rank, backend.qk_rope_head_dim = (
        page,
        latent,
        rope,
    )
    backend.forward_metadata = ForwardMetadata(
        block_tables=table.to(a5), seq_lens_cpu_list=lengths
    )
    backend.token_to_kv_pool = SimpleNamespace(
        get_key_buffer=lambda _: kv, get_value_buffer=lambda _: k_rope
    )
    with patch.object(attention, operator, wraps=getattr(attention, operator)) as fia:
        result = backend.forward_decode(
            q,
            None,
            None,
            layer,
            SimpleNamespace(batch_size=len(lengths)),
            save_kv_cache=False,
            q_rope=q_rope,
            q_descale=q_descale,
        )
    fia.assert_called_once()

    expected = []
    for i, length in enumerate(lengths):
        pages = table[i].long()
        keys = ref_kv[pages].reshape(-1, latent)[:length]
        key_rope = cache[pages, ..., latent:].reshape(-1, rope)[:length].float()
        logits = ref_q[i] @ keys.T + ref_q_rope[i] @ key_rope.T
        expected.append((logits * scale).softmax(dim=-1) @ keys)
    expected = torch.stack(expected).reshape(len(lengths), heads * latent).bfloat16()
    assert result.device.type == "npu"
    torch.testing.assert_close(result.cpu(), expected, atol=4e-3, rtol=6e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
