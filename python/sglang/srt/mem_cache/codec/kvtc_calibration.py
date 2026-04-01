import math
import os
from typing import Any, List, Tuple

import torch


def _iter_nodes(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        if getattr(node, "children", None):
            stack.extend(node.children.values())


def _get_vector_numel(host_pool) -> int:
    dp = host_pool.device_pool
    head_num = dp.head_num
    head_dim = dp.head_dim
    v_head_dim = dp.v_head_dim
    layer_num = dp.layer_num
    page_size = host_pool.page_size
    return layer_num * page_size * head_num * (head_dim + v_head_dim)


def _get_shape_meta(host_pool) -> dict[str, Any]:
    dp = host_pool.device_pool
    return {
        "numel": _get_vector_numel(host_pool),
        "page_size": host_pool.page_size,
        "layer_num": dp.layer_num,
        "head_num": dp.head_num,
        "head_dim": dp.head_dim,
        "v_head_dim": dp.v_head_dim,
    }


def collect_page_vectors_from_tree(
    root,
    host_pool,
    max_pages: int,
    seen: set[int] | None = None,
) -> Tuple[List[torch.Tensor], dict[str, Any], set[int]]:
    page_size = host_pool.page_size
    vector_numel = _get_vector_numel(host_pool)
    seen = seen or set()
    vectors: List[torch.Tensor] = []

    for node in _iter_nodes(root):
        host_value = getattr(node, "host_value", None)
        if host_value is None or len(host_value) == 0:
            continue
        host_indices = host_value.detach().cpu().tolist()
        for i in range(0, len(host_indices), page_size):
            base = int(host_indices[i])
            if base in seen:
                continue
            seen.add(base)
            data_page = host_pool.get_data_page(base, flat=True)
            vec = data_page.contiguous()
            if vec.dtype == torch.uint8:
                vec = vec.view(host_pool.dtype)
            else:
                vec = vec.view(host_pool.dtype)

            if getattr(host_pool, "kv_order", "") == "interleaved_per_layer":
                dp = host_pool.device_pool
                page_size = host_pool.page_size
                per_layer_k = page_size * dp.head_num * dp.head_dim
                per_layer_v = page_size * dp.head_num * dp.v_head_dim
                k_parts = []
                v_parts = []
                offset = 0
                for _ in range(dp.layer_num):
                    k_parts.append(vec[offset : offset + per_layer_k])
                    offset += per_layer_k
                    v_parts.append(vec[offset : offset + per_layer_v])
                    offset += per_layer_v
                vec = torch.cat(k_parts + v_parts, dim=0)
            if vec.numel() != vector_numel:
                continue
            vectors.append(vec.to(torch.float32).cpu())
            if max_pages > 0 and len(vectors) >= max_pages:
                return vectors, _get_shape_meta(host_pool), seen

    return vectors, _get_shape_meta(host_pool), seen


def compute_kvtc_params(
    vectors: List[torch.Tensor],
    ratio: float,
    output_dtype: str,
    max_k: int = 256,
    shape: dict[str, Any] | None = None,
) -> dict[str, Any]:
    X = torch.stack(vectors, dim=0)
    m, d = X.shape
    max_k_cap = max(1, int(max_k))
    scale_overhead_bits = max(0, int(os.getenv("SGLANG_KVTC_SCALE_OVERHEAD_BITS", "16")))
    allowed_block_sizes_str = os.getenv(
        "SGLANG_KVTC_ALLOWED_BLOCK_SIZES", "1,16,64,256,1024"
    )
    allowed_block_sizes = [
        int(x.strip())
        for x in allowed_block_sizes_str.split(",")
        if x.strip() and int(x.strip()) > 0
    ]
    allowed_block_sizes = sorted(set(allowed_block_sizes))
    allowed_types = ["none", "int2", "int4", "fp8"]

    def _fit_stream(Xs: torch.Tensor) -> dict[str, Any]:
        mean = Xs.mean(dim=0)
        Xc = Xs - mean
        ms, ds = Xc.shape
        pca_k = min(max_k_cap, max(1, min(ms, ds)))
        _, _, V = torch.pca_lowrank(Xc, q=pca_k, center=False)
        proj = V
        Y = Xc @ proj
        var = torch.var(Y, dim=0, unbiased=False).clamp_min(1e-12)

        max_bits = 8
        bits_budget = int(math.floor(pca_k * 16.0 / max(ratio, 1.0)))
        bits_budget = max(0, min(bits_budget, pca_k * max_bits))

        bs_list = [bs for bs in allowed_block_sizes if bs <= pca_k]
        if 1 not in bs_list:
            bs_list = [1] + bs_list

        type_bits = [0, 2, 4, 8]
        type_names = {0: "none", 2: "int2", 4: "int4", 8: "fp8"}

        dp = [[float("inf")] * (bits_budget + 1) for _ in range(pca_k + 1)]
        prev_i = [[-1] * (bits_budget + 1) for _ in range(pca_k + 1)]
        prev_used = [[-1] * (bits_budget + 1) for _ in range(pca_k + 1)]
        prev_bs = [[-1] * (bits_budget + 1) for _ in range(pca_k + 1)]
        prev_tb = [[-1] * (bits_budget + 1) for _ in range(pca_k + 1)]
        dp[0][0] = 0.0

        var_cpu = var.detach().cpu().to(torch.float64)
        var_prefix = torch.zeros((pca_k + 1,), dtype=torch.float64)
        var_prefix[1:] = torch.cumsum(var_cpu, dim=0)

        range_mult = 3.0
        for i in range(1, pca_k + 1):
            for budget in range(bits_budget + 1):
                if budget > 0 and dp[i][budget] > dp[i][budget - 1]:
                    dp[i][budget] = dp[i][budget - 1]
                    prev_i[i][budget] = prev_i[i][budget - 1]
                    prev_used[i][budget] = prev_used[i][budget - 1]
                    prev_bs[i][budget] = prev_bs[i][budget - 1]
                    prev_tb[i][budget] = prev_tb[i][budget - 1]

                for bs in bs_list:
                    if bs > i:
                        break
                    j0 = i - bs
                    v_sum = float((var_prefix[i] - var_prefix[j0]).item())
                    v_max = float(var_cpu[j0:i].max().item())
                    for tb in type_bits:
                        if tb == 0:
                            used = 0
                            cost = v_sum
                        else:
                            used = bs * tb + scale_overhead_bits
                            if used > budget:
                                continue
                            cost = v_sum * (2.0 ** (-2.0 * tb))
                        if used > budget:
                            continue
                        base = dp[j0][budget - used] if used <= budget else float("inf")
                        if base == float("inf"):
                            continue
                        cand = base + cost
                        if cand < dp[i][budget]:
                            dp[i][budget] = cand
                            prev_i[i][budget] = j0
                            prev_used[i][budget] = used
                            prev_bs[i][budget] = bs
                            prev_tb[i][budget] = tb

        end_budget = min(range(bits_budget + 1), key=lambda b: dp[pca_k][b])
        segments = []
        i = pca_k
        budget = end_budget
        while i > 0:
            j0 = prev_i[i][budget]
            bs = prev_bs[i][budget]
            tb = prev_tb[i][budget]
            used = prev_used[i][budget]
            if j0 < 0 or bs <= 0 or tb < 0 or used < 0:
                break
            segments.append((j0, i, bs, tb))
            i = j0
            budget = budget - used
        segments.reverse()

        bits_list = [0] * pca_k
        scale_list = [1.0] * pca_k
        for start, end, bs, tb in segments:
            if tb <= 0:
                continue
            vmax = float(var_cpu[start:end].max().item())
            qmax = float((2 ** (tb - 1)) - 1)
            s = range_mult * math.sqrt(vmax) / max(qmax, 1.0)
            for t in range(start, end):
                bits_list[t] = tb
                scale_list[t] = s

        bits = torch.tensor(bits_list, dtype=torch.int16)
        scale = torch.tensor(scale_list, dtype=torch.float32).clamp_min(1e-8)

        return {
            "proj": proj.contiguous(),
            "mean": mean.contiguous(),
            "scale": scale.contiguous(),
            "bits": bits.contiguous(),
            "segments": segments,
            "scale_overhead_bits": int(scale_overhead_bits),
            "allowed_block_sizes": bs_list,
            "allowed_types": allowed_types,
            "pca_k": int(pca_k),
            "bits_budget": int(bits_budget),
            "max_bits": int(max_bits),
            "range_mult": float(range_mult),
        }

    if shape is not None:
        k_dim = (
            int(shape["layer_num"])
            * int(shape["page_size"])
            * int(shape["head_num"])
            * int(shape["head_dim"])
        )
        v_dim = (
            int(shape["layer_num"])
            * int(shape["page_size"])
            * int(shape["head_num"])
            * int(shape["v_head_dim"])
        )
        if k_dim + v_dim != d:
            raise ValueError(f"shape-derived dims mismatch: {k_dim}+{v_dim}!={d}")
    else:
        k_dim = d // 2
        v_dim = d - k_dim

    k_stream = _fit_stream(X[:, :k_dim])
    v_stream = _fit_stream(X[:, k_dim : k_dim + v_dim])
    return {
        "k_proj": k_stream["proj"],
        "k_mean": k_stream["mean"],
        "k_scale": k_stream["scale"],
        "k_bits": k_stream["bits"],
        "k_segments": k_stream["segments"],
        "k_scale_overhead_bits": k_stream["scale_overhead_bits"],
        "k_allowed_block_sizes": k_stream["allowed_block_sizes"],
        "k_allowed_types": k_stream["allowed_types"],
        "k_pca_k": k_stream["pca_k"],
        "k_bits_budget": k_stream["bits_budget"],
        "v_proj": v_stream["proj"],
        "v_mean": v_stream["mean"],
        "v_scale": v_stream["scale"],
        "v_bits": v_stream["bits"],
        "v_segments": v_stream["segments"],
        "v_scale_overhead_bits": v_stream["scale_overhead_bits"],
        "v_allowed_block_sizes": v_stream["allowed_block_sizes"],
        "v_allowed_types": v_stream["allowed_types"],
        "v_pca_k": v_stream["pca_k"],
        "v_bits_budget": v_stream["bits_budget"],
        "output_dtype": output_dtype,
        "format_version": 4,
        "entropy": "deflate",
        "packing": "twos_complement",
        "max_bits": int(k_stream["max_bits"]),
        "range_mult": float(k_stream["range_mult"]),
        "scale_overhead_bits": int(scale_overhead_bits),
        "allowed_block_sizes": allowed_block_sizes,
        "allowed_types": allowed_types,
        "samples": int(m),
        "input_dim": int(d),
        "max_k": int(max_k_cap),
    }


def save_kvtc_params(params: dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(params, output_path)
