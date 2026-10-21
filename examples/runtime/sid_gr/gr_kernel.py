"""Load the separately pinned NVIDIA GR decode kernel in the example process."""

import importlib
import math
import os
import sys
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def load_gr_kernel():
    path = Path(os.environ["SGLANG_SID_GR_KERNEL_PATH"]).resolve(strict=True)
    if not (path / "interface.py").is_file():
        raise ValueError("GR kernel path must contain interface.py")
    import cutlass.cute as cute

    # Compatibility with the pinned GR source's CUTLASS DSL 4.4.2 API.
    # These are namespace aliases; they do not change kernel arithmetic.
    if not hasattr(cute.core, "ThrMma"):
        cute.core.ThrMma = cute.ThrMma
    if not hasattr(cute, "make_fragment"):
        cute.make_fragment = cute.make_rmem_tensor
    if "interface" in sys.modules:
        loaded = Path(sys.modules["interface"].__file__).resolve()
        if loaded != path / "interface.py":
            raise RuntimeError(f"Another module already occupies 'interface': {loaded}")
    sys.path.insert(0, str(path))
    return importlib.import_module("interface").beam_decode_attn


def beam_decode_with_reference_splits(
    q, k_context, v_context, k_beam, v_beam, paths, steps, *, reference_heads
):
    """Use the pinned fused GR kernels with the full-head SID split geometry.

    This explicit adapter leaves the upstream module and its heuristic intact.
    Its private launch API is tied to the separately pinned kernel revision.
    Local Q/K/V heads and KV storage remain partitioned across TP ranks.
    """
    load_gr_kernel()
    kernel = importlib.import_module("interface")
    kernel._validate_inputs(q, k_context, v_context, k_beam, v_beam, paths, steps)
    batch, sq, width, heads, dim = q.shape
    if sq != 1 or reference_heads < heads or reference_heads % heads:
        raise ValueError("Invalid full-head SID decode geometry")
    context = k_context.shape[1]
    tile = 128 if dim <= 128 else 64
    mblocks, nblocks = math.ceil(width / tile), math.ceil(context / tile)
    splits = kernel.num_splits_heuristic(
        batch * reference_heads * mblocks,
        torch.cuda.get_device_properties(q.device).multi_processor_count,
        nblocks,
        mblocks,
        context * dim * 4,
    )
    if splits > 1 and nblocks > 0:
        splits = min(splits, nblocks)
    output = torch.empty(batch, width, heads, dim, device=q.device, dtype=q.dtype)
    scale = 1.0 / math.sqrt(dim)
    if splits > 1:
        partial = torch.empty(
            splits, batch, width, heads, dim, device=q.device, dtype=torch.float32
        )
        lse_partial = torch.empty(
            splits, batch, heads, width, device=q.device, dtype=torch.float32
        )
        lse = torch.empty(
            batch, heads, width, device=q.device, dtype=torch.float32
        ).transpose(-1, -2)
        kernel._fused_context_beam(
            q,
            k_context,
            v_context,
            k_beam,
            v_beam,
            paths,
            steps,
            scale,
            splits,
            True,
            out=partial,
            lse=lse_partial,
        )
        kernel._combine(partial, lse_partial.transpose(-1, -2), output, lse)
    else:
        lse = torch.empty(batch, heads, width, device=q.device, dtype=torch.float32)
        kernel._fused_context_beam(
            q,
            k_context,
            v_context,
            k_beam,
            v_beam,
            paths,
            steps,
            scale,
            1,
            False,
            out=output,
            lse=lse,
        )
    return output.unsqueeze(1)
