"""ROCm forward of Inkling2Attention: the Triton backend's relative-bias attention."""

from sglang.kernels.ops.inkling2.attention import gate_sink
from sglang.srt.layers.quantization.unquant import _prefer_triton_a16w16
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_gfx95_supported

_aiter_a16w16 = None
if is_gfx95_supported():
    try:
        from aiter.ops.triton.gemm_a16w16 import gemm_a16w16 as _aiter_a16w16
    except ImportError:
        pass


def forward(attention, q, k, v, r, gate, sinks, sources, positions, forward_batch):
    """ROCm: the Triton backend's relative-bias attention on a compact rel bias.
    Decode and MTP verify take the fused prologue (verify saves the conv windows),
    which also stores K and V into the KV cache when the backend names the target,
    and apply the gate and sink in their attention reductions."""
    tokens = q.shape[0]
    mode = forward_batch.forward_mode
    verify = mode.is_target_verify()
    if attention.fused_decode_prologue and (mode.is_decode() or verify):
        kv_pool = fused_kv_write_buffers(attention.attn, forward_batch, k.dtype)
        q, k, v, rel_bias = attention._decode_prologue(
            q,
            k,
            v,
            r,
            sources,
            positions,
            forward_batch,
            False,
            verify=verify,
            kv_pool=kv_pool,
        )
        verify_kwargs = (
            dict(rel_bias_scale_after_add=True, pack_gqa=False) if verify else {}
        )
        output = attention.attn(
            q,
            k,
            v,
            forward_batch,
            save_kv_cache=kv_pool is None,
            rel_bias=rel_bias,
            gate_sink=(gate, sinks),
            **verify_kwargs,
        )
    else:
        q, k, v, r = attention._prologue(q, k, v, r, sources, positions, forward_batch)
        output, lse = attention.attn(
            q,
            k,
            v,
            forward_batch,
            rel_bias=attention.rel_logits_proj(r),
            return_lse=True,
            rel_bias_scale_after_add=True,
            pack_gqa=False,
        )
        output = gate_sink(
            output.reshape(tokens, -1), gate, lse, sinks, output_dtype=q.dtype
        )
    output = apply_wo_ud(attention.wo_ud, output)
    return get_parallel().attn_tp_group.all_reduce(output)


def apply_wo_ud(layer, x):
    """wo_ud on aiter's Triton BF16 GEMM (gfx95) where SGLang's aiter linear path routes
    decode shapes to it, else the layer's own GEMM; at TP8 (K 768) the two are bitwise
    equal up to 64 rows."""
    if (
        _aiter_a16w16 is not None
        and layer.bias is None
        and _prefer_triton_a16w16(x, layer.weight)
    ):
        return _aiter_a16w16(x, layer.weight, None, dtype=x.dtype)
    return layer(x)[0]


def fused_kv_write_buffers(layer, forward_batch, dtype):
    """The backend's (k_buffer, v_buffer, loc) for layer (through a hybrid backend's
    full-attention backend) when its rows are of dtype, else None."""
    backend = get_attn_backend()
    backend = getattr(backend, "full_attn_backend", backend)
    buffers = backend.get_fused_kv_write_buffers(layer, forward_batch)
    return buffers if buffers is not None and buffers[0].dtype == dtype else None
