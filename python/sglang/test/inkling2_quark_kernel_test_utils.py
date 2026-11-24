"""Reference dequantizers and synthetic packed experts for Inkling-2 Quark GPU tests."""

import torch

from sglang.srt.layers.quantization.kvfp4_tensor import E2M1_VALUES

AWQ_NIBBLE_TO_COL = (0, 2, 4, 6, 1, 3, 5, 7)
AWQ_COL_TO_NIBBLE = tuple(AWQ_NIBBLE_TO_COL.index(c) for c in range(8))
UINT4_GROUP_SIZE = 128
MXFP4_BLOCK_SIZE = 32
TOP_K = 8


def unpack_awq_int4(packed: torch.Tensor) -> torch.Tensor:
    """int32 [..., R, C/8] -> uint8 [..., R, C] with values 0..15."""
    if packed.dtype != torch.int32:
        raise TypeError(f"expected int32 packed words, got {packed.dtype}")
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=packed.device)
    nibbles = ((packed.unsqueeze(-1) >> shifts) & 0xF).to(torch.uint8)
    nibbles = nibbles[..., list(AWQ_COL_TO_NIBBLE)]
    return nibbles.reshape(*packed.shape[:-1], packed.shape[-1] * 8)


def dequant_uint4_g128(
    qweight, scale, zero_point, group_size=UINT4_GROUP_SIZE, out_dtype=torch.float32
):
    """qweight [..., K, N/8], scale [..., K/g, N] and zero_point [..., K/g, N/8] ->
    W [..., N, K]."""
    q = unpack_awq_int4(qweight).to(torch.float32)
    zp = unpack_awq_int4(zero_point).to(torch.float32)
    s = scale.to(torch.float32)
    k, n = q.shape[-2], q.shape[-1]
    if k % group_size or s.shape[-2] != k // group_size or s.shape[-1] != n:
        raise ValueError(
            f"inconsistent shapes: q {tuple(q.shape)} scale {tuple(s.shape)} "
            f"zp {tuple(zp.shape)} group {group_size}"
        )
    grouped = q.reshape(*q.shape[:-2], k // group_size, group_size, n)
    w = (grouped - zp.unsqueeze(-2)) * s.unsqueeze(-2)
    return w.reshape(*q.shape).transpose(-1, -2).contiguous().to(out_dtype)


def unpack_fp4_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [..., K/2] -> float32 [..., K]; low nibble = even element."""
    if packed.dtype != torch.uint8:
        raise TypeError(f"expected uint8, got {packed.dtype}")
    table = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=packed.device)
    low, high = table[(packed & 0x0F).long()], table[(packed >> 4).long()]
    return torch.stack([low, high], dim=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 2
    )


def dequant_mxfp4(weight, scale, block_size=MXFP4_BLOCK_SIZE, out_dtype=torch.float32):
    """weight uint8 [..., N, K/2], scale uint8 (E8M0) [..., N, K/32] -> [..., N, K]."""
    value = unpack_fp4_e2m1(weight)
    k = value.shape[-1]
    if k % block_size or scale.shape[-1] != k // block_size:
        raise ValueError(
            f"inconsistent shapes: weight {tuple(weight.shape)} scale {tuple(scale.shape)}"
        )
    blocks = value.reshape(*value.shape[:-1], k // block_size, block_size)
    # torch's E8M0 decode is exact for every byte, unlike quark.utils.e8m0_to_f32
    power = scale.view(torch.float8_e8m0fnu).to(torch.float32)
    return (blocks * power.unsqueeze(-1)).reshape(value.shape).to(out_dtype)


def dequant_uint4_experts(data: dict, chunk: int = 16) -> torch.Tensor:
    """dequant_uint4_g128 in BF16 over the experts of `data`, `chunk` at a time."""
    return torch.cat(
        [
            dequant_uint4_g128(
                data["weight"][i : i + chunk],
                data["weight_scale"][i : i + chunk],
                data["weight_zero_point"][i : i + chunk],
                out_dtype=torch.bfloat16,
            )
            for i in range(0, data["weight"].shape[0], chunk)
        ]
    )


def synth_mxfp4(
    experts: int,
    n: int,
    k: int,
    seed: int = 0,
    scale_lo: int = 100,
    scale_hi: int = 140,
    scale_byte: int | None = None,
    device="cpu",
) -> dict:
    """Random codes and E8M0 bytes in [scale_lo, scale_hi) (or all scale_byte): weight,
    weight_scale and `expected`, the dequantized BF16 weights [E, N, K]."""
    g = torch.Generator().manual_seed(seed)
    weight = torch.randint(0, 256, (experts, n, k // 2), generator=g, dtype=torch.uint8)
    if scale_byte is None:
        scale = torch.randint(
            scale_lo, scale_hi, (experts, n, k // 32), generator=g, dtype=torch.uint8
        )
    else:
        scale = torch.full((experts, n, k // 32), scale_byte, dtype=torch.uint8)
    weight, scale = weight.to(device), scale.to(device)
    return {
        "weight": weight,
        "weight_scale": scale,
        "expected": dequant_mxfp4(weight, scale, out_dtype=torch.bfloat16),
    }


def synth_uint4(
    experts: int,
    n: int,
    k: int,
    seed: int = 0,
    scale_dtype=torch.bfloat16,
    device="cpu",
) -> dict:
    """Random words, zero points and positive scales (F32 ones with a full mantissa):
    weight, weight_scale, weight_zero_point and `expected` [E, N, K] in BF16."""
    g = torch.Generator().manual_seed(seed)

    def words(*shape):
        return torch.randint(-(2**31), 2**31, shape, generator=g, dtype=torch.int32)

    groups = k // UINT4_GROUP_SIZE
    scale = torch.rand(experts, groups, n, generator=g) * 0.06 + 0.002
    if scale_dtype == torch.float32:
        scale = scale * (1.0 + torch.rand(experts, groups, n, generator=g) * 1e-3)
    data = {
        "weight": words(experts, k, n // 8).to(device),
        "weight_scale": scale.to(scale_dtype).to(device),
        "weight_zero_point": words(experts, groups, n // 8).to(device),
    }
    data["expected"] = dequant_uint4_experts(data)
    return data


def canonical_bits(t: torch.Tensor) -> torch.Tensor:
    """BF16 bit patterns as int16 with -0 mapped to +0."""
    if t.dtype != torch.bfloat16:
        raise TypeError(f"expected bfloat16, got {t.dtype}")
    bits = t.contiguous().view(torch.int16)
    return torch.where(bits == -32768, torch.zeros_like(bits), bits)


def count_bit_mismatches(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((canonical_bits(a) != canonical_bits(b)).sum())


def triton_gap(*features: str) -> str:
    """The skip reason if triton.language lacks one of `features`, else ""."""
    import triton
    import triton.language as tl

    missing = [f"tl.{name}" for name in features if not hasattr(tl, name)]
    if not missing:
        return ""
    return (
        f"Triton {triton.__version__} lacks {', '.join(missing)}: the MXFP4 tile needs "
        "dot_scaled and the UINT4 tile join (validated with Triton 3.8.0)"
    )


def make_routing(tokens, num_experts, top_k=TOP_K, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    scores = torch.rand(tokens, num_experts, generator=g, device=device)
    return scores.topk(top_k, dim=1).indices.to(torch.int32)


def _kernel():
    from sglang.kernels.ops.moe import inkling2_quark_gemm_hip

    return inkling2_quark_gemm_hip


def build_route(routing, ids, start, num_local, num_experts, block_m):
    """'product': the route with the schedule of block_m; 'launcher': without one."""
    from sglang.kernels.ops.moe import inkling2_experts_hip

    return inkling2_experts_hip.route(
        ids,
        start,
        num_local,
        num_experts,
        block_m=block_m if routing == "product" else None,
    )


def run_gemm(
    fmt, data, x, route, n, gather_div, out_dtype, rows, *, num_experts, config=None
):
    return _kernel().expert_gemm_quark(
        x,
        data["weight"],
        data["weight_scale"],
        data.get("weight_zero_point"),
        route,
        n,
        gather_div,
        out_dtype,
        rows,
        fmt=fmt,
        num_experts=num_experts,
        config=config,
    )


def onehot_check(
    fmt, data, k, n, config, experts=(0,), reps=1, routing="product", device="cuda"
):
    """(mismatching elements, elements checked) of the one-hot probe."""
    num_local = data["weight"].shape[0]
    x = torch.eye(k, dtype=torch.bfloat16, device=device)
    bad = total = 0
    for e in experts:
        ids = torch.full((k, 1), e, dtype=torch.int32, device=device)
        route = build_route(routing, ids, 0, num_local, num_local, config.bm)
        expected = data["expected"][e]
        for _ in range(reps):
            y = run_gemm(
                fmt, data, x, route, n, 1, torch.float32, k,
                num_experts=num_local, config=config,
            )  # fmt: skip
            got = y.sum(0).t().to(torch.bfloat16)
            bad += count_bit_mismatches(got, expected)
            total += expected.numel()
    return bad, total


def reference_grouped(x, expected, ids, gather_div, start=0):
    """float64 x[r // gather_div] @ W_e(r)^T for the rows r routed to the experts
    [start, start + E) of `expected` [E, N, K]; returns (y, local mask)."""
    flat = ids.flatten().long()
    local = (flat >= start) & (flat < start + expected.shape[0])
    ref = torch.zeros(
        flat.numel(), expected.shape[1], dtype=torch.float64, device=x.device
    )
    for e in torch.unique(flat[local]).tolist():
        r = (flat == e).nonzero(as_tuple=True)[0]
        ref[r] = x[r // gather_div].double() @ expected[e - start].double().t()
    return ref, local


def routed_case(
    fmt,
    data,
    kind,
    tokens,
    *,
    num_experts,
    start=0,
    config=None,
    seed=0,
    routing="product",
    out_dtype=torch.float32,
    ids=None,
    route_block_m=None,
    device="cuda",
):
    """One routed grouped GEMM against float64 on the dequantized weights. kind "w13"
    reads one input row per token, "w2" one per routed row. Split-K slabs are summed
    in float64."""
    gemm = _kernel()
    num_local, n, k = data["expected"].shape
    if ids is None:
        ids = make_routing(tokens, num_experts, seed=seed, device=device)
    tokens = ids.shape[0]
    rows = ids.numel()
    gather_div = TOP_K if kind == "w13" else 1
    g = torch.Generator(device=device).manual_seed(1000 + seed)
    x = torch.randn(
        tokens if kind == "w13" else rows, k, generator=g, device=device
    ).mul_(0.5)
    x = x.to(torch.bfloat16)
    if config is None:
        scale = data["weight_scale"]
        block_m = gemm.default_config_quark(fmt, rows, num_experts, scale.dtype).bm
    else:
        block_m = config.bm
    route = build_route(
        routing, ids, start, num_local, num_experts, route_block_m or block_m
    )
    y = run_gemm(
        fmt, data, x, route, n, gather_div, out_dtype, rows,
        num_experts=num_experts, config=config,
    )  # fmt: skip
    ref, local = reference_grouped(x, data["expected"], ids, gather_div, start)
    diff = y.double().sum(0)[local] - ref[local]
    result = {
        "rel_l2": float(diff.norm() / ref[local].norm()),
        "max_abs": float(diff.abs().max()),
        "splits": y.shape[0],
        "rows": rows,
        "local_rows": int(local.sum()),
        "dtype": y.dtype,
    }
    if y.shape[0] > 1:  # slab s holds the partial sums over its own K range
        width = k // y.shape[0]
        result["slab_rel_l2"] = []
        for s in range(y.shape[0]):
            part = slice(s * width, (s + 1) * width)
            slab_ref, _ = reference_grouped(
                x[:, part], data["expected"][:, :, part], ids, gather_div, start
            )
            slab_diff = y[s].double()[local] - slab_ref[local]
            result["slab_rel_l2"].append(
                float(slab_diff.norm() / slab_ref[local].norm())
            )
    return result
