"""Real DeepSeek shared MLP fixtures and independent CPU arithmetic."""

import torch
import torch.nn.functional as F

from .triton_compute import _gemm_cpu


def make_shared_mlp(*, shared_experts=2, hidden=2048, intermediate=128, fp8=True):
    from sglang.srt.layers.quantization.fp8 import Fp8Config
    from sglang.srt.models.deepseek_v2 import DeepseekV2MLP
    from sglang.srt.runtime_context import get_parallel

    quant = (
        Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
        if fp8
        else None
    )
    width = shared_experts * intermediate
    # The synthetic pair owns its EP communicator, not SGLang's global TP
    # group. Match the FP8 shape validator to this explicitly unsharded MLP.
    with torch.device("cuda"), get_parallel().override(tp_size=1, tp_rank=0):
        mlp = DeepseekV2MLP(
            hidden,
            width,
            "silu",
            quant_config=quant,
            reduce_results=False,
            tp_rank=0,
            tp_size=1,
        )
    if not fp8:
        mlp = mlp.bfloat16()
    rng = torch.Generator().manual_seed(38683)
    # Exercise the existing checkpoint loaders: two separate gate/up shards
    # and a complete unsharded down projection on every EP rank.
    for shard in range(2):
        value = torch.randn(width, hidden, generator=rng)
        value = (
            value.to(torch.float8_e4m3fn) if fp8 else (value / hidden**0.5).bfloat16()
        )
        parameter = mlp.gate_up_proj.weight
        parameter.weight_loader(parameter, value.cuda(), shard)
    value = torch.randn(hidden, width, generator=rng)
    value = value.to(torch.float8_e4m3fn) if fp8 else (value / width**0.5).bfloat16()
    parameter = mlp.down_proj.weight
    parameter.weight_loader(parameter, value.cuda())
    if fp8:
        for projection, fan_in in ((mlp.gate_up_proj, hidden), (mlp.down_proj, width)):
            projection.weight_scale_inv.data.fill_(fan_in**-0.5)
            projection.quant_method.process_weights_after_loading(projection)
    return mlp.eval().requires_grad_(False)


def shared_reference(mlp, x):
    x = x.cpu().bfloat16()
    if not len(x):
        return x.float()
    first, second = mlp.gate_up_proj, mlp.down_proj
    if first.weight.dtype == torch.float8_e4m3fn:
        gate_up = _gemm_cpu(x, first.weight.cpu(), first.weight_scale_inv.cpu())
    else:
        gate_up = (x.float() @ first.weight.cpu().float().T).bfloat16().float()
    gate, up = gate_up.chunk(2, -1)
    active = (F.silu(gate) * up).bfloat16()
    if second.weight.dtype == torch.float8_e4m3fn:
        return _gemm_cpu(active, second.weight.cpu(), second.weight_scale_inv.cpu())
    return (active.float() @ second.weight.cpu().float().T).bfloat16().float()
