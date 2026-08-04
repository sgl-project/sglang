"""Shared plumbing for layer-level backend parity UTs.

Single-process dist bring-up, tp=1 linear-layer builders/loaders, and the
hand-written NVFP4 encode/decode helpers used by the quantization backend
tests. The NVFP4 helpers are deliberately test-side reference code -- do not
replace them with imports from sglang.srt; the tests use them to check srt.
"""

import os

import torch

FLOAT8_E4M3_MAX = 448.0
FLOAT4_E2M1_MAX = 6.0

kE2M1ToFloat = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def init_single_process_dist(master_port: int = 29632):
    """world=1 gloo dist + model-parallel groups; srt layers require them
    even at tp=1."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(master_port))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )

    if not torch.distributed.is_initialized():
        init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
    if not model_parallel_is_initialized():
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            backend="gloo",
        )


def make_tp1_column_parallel_linear(
    quant_config, n: int, k: int, prefix: str = "model.layers.0.mlp.up_proj", **kwargs
):
    from sglang.srt.layers.linear import ColumnParallelLinear

    return ColumnParallelLinear(
        input_size=k,
        output_size=n,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=quant_config,
        prefix=prefix,
        tp_rank=0,
        tp_size=1,
        **kwargs,
    ).cuda()


def load_linear_weights(layer, shard_id=None, **named_weights):
    """Feed checkpoint-format tensors through the real weight_loader."""
    for name, loaded in named_weights.items():
        if shard_id is None:
            layer.weight_loader_v2(getattr(layer, name), loaded)
        else:
            layer.weight_loader_v2(getattr(layer, name), loaded, shard_id)


def assert_output_close(tc, out, ref, cos_threshold=0.99, rtol=None, atol=None):
    """Shape + cosine check, plus assert_close when rtol/atol are given."""
    tc.assertEqual(tuple(out.shape), tuple(ref.shape))
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.flatten(), dim=0
    ).item()
    tc.assertGreater(cos, cos_threshold)
    if rtol is not None:
        torch.testing.assert_close(out.float(), ref, rtol=rtol, atol=atol)


def convert_swizzled_to_linear(a_sf_swizzled: torch.Tensor, m, k, block_size=16):
    m_tiles = (m + 128 - 1) // 128
    f = block_size * 4
    k_tiles = (k + f - 1) // f
    tmp = torch.reshape(a_sf_swizzled, (1, m_tiles, k_tiles, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    out = tmp.reshape(m_tiles * 128, k_tiles * f // block_size)
    # Crop the K-tile padding too: k // block_size scale columns, not k.
    return out[0:m, 0 : k // block_size]


def break_fp4_bytes(a, dtype=torch.float32):
    assert a.dtype == torch.uint8
    m, n = a.shape
    a_flat = a.flatten()
    high = (a_flat & 0xF0) >> 4
    low = a_flat & 0x0F
    combined = torch.stack((low, high), dim=1).flatten()
    signs = (combined & 0x08).to(torch.bool)
    abs_vals = (combined & 0x07).to(torch.long)
    kE2M1 = kE2M1ToFloat.to(device=a.device)
    values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)
    return values.reshape(m, n * 2).to(dtype=dtype)


def dequantize_nvfp4_to_dtype(
    tensor_fp4, tensor_sf, global_scale, dtype, block_size=16
):
    assert tensor_fp4.dtype == torch.uint8
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    tensor_f32 = break_fp4_bytes(tensor_fp4, torch.float32)
    tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)
    tensor_sf = tensor_sf.view(torch.float8_e4m3fn)
    tensor_sf = convert_swizzled_to_linear(tensor_sf, m, k, block_size)
    tensor_sf_dtype = tensor_sf.to(torch.float32) / global_scale
    out = (tensor_f32 * tensor_sf_dtype.unsqueeze(-1)).reshape(m, k)
    return out.to(dtype=dtype)


def quantize_nvfp4_shard(w: torch.Tensor, gs=None):
    """NVFP4-quantize one checkpoint shard; returns (packed, linear sf,
    global scale, fp32 dequant reference)."""
    from flashinfer import fp4_quantize

    n, k = w.shape
    if gs is None:
        gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w.abs().max().to(torch.float32)
    w_q, w_sf_swizzled = fp4_quantize(w, gs)
    sf_linear = convert_swizzled_to_linear(
        w_sf_swizzled.view(torch.float8_e4m3fn), n, k, 16
    )
    w_dequant = dequantize_nvfp4_to_dtype(w_q, w_sf_swizzled, gs, torch.float32)
    return w_q, sf_linear, gs, w_dequant
