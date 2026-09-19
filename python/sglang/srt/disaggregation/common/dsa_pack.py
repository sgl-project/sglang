"""Synchronous DSA page packing in the transfer worker's existing MLA buffer."""

import numpy as np
import torch

from sglang.kernels.ops.kvcache.pd_dcp_gather import copy_dsa_pages_into_pack
from sglang.srt.disaggregation.common.dsa_dcp import (
    DSAPackPlan,
    iter_dsa_dcp_pack_plans,
)


def pack_dsa_plan(plan, pack_buffer, dcp_size, dcp_rank):
    if not pack_buffer.fits(plan.packed_bytes):
        raise ValueError("DSA pack plan exceeds the registered buffer")
    # The only device allocation is bounded page metadata (128 KiB + 8 B at
    # DCP8). The payload aliases the existing registered buffer.
    host = np.empty(plan.src_pages.size + 1, dtype=np.int64)
    host[0] = plan.src_ptr
    host[1:] = plan.src_pages
    pack = pack_buffer.buffer.narrow(0, 0, plan.packed_bytes)
    metadata = torch.as_tensor(host, device=pack.device)
    stream = pack_buffer.get_gather_stream()
    stream.wait_stream(torch.cuda.current_stream(pack.device))
    with torch.cuda.stream(stream):
        copy_dsa_pages_into_pack(metadata, pack, plan.num_tokens, dcp_size, dcp_rank)
    # Ensure both the RDMA source and temporary metadata are safe to use/free.
    stream.synchronize()


def iter_packed_dsa_transfer_batches(
    *args,
    pack_buffer,
    transfer_profile=None,
    **kwargs,
):
    """Consume each batch synchronously before advancing and overwriting it."""
    plans = iter_dsa_dcp_pack_plans(
        *args,
        pack_ptr=pack_buffer.get_ptr(),
        pack_bytes=pack_buffer.get_size(),
        **kwargs,
    )
    if transfer_profile is not None:
        plans = transfer_profile.batches(plans)
    for plan in plans:
        pack = pack_dsa_plan
        if transfer_profile is not None:

            def pack(*args):
                return transfer_profile.call("dsa_pack", pack_dsa_plan, *args)

        pack(plan, pack_buffer, kwargs["dcp_size"], kwargs["dcp_rank"])
        yield plan.blocks


def warmup_dsa_pack_buffer(pack_buffer, dcp_size):
    # Disjoint source/output pages in existing scratch storage. Do this before
    # publishing the buffers to transfer workers, not during a long request.
    if pack_buffer.get_size() < 2 * 8448:
        return
    pack_buffer.buffer.narrow(0, 8448, 8448).zero_()
    plan = DSAPackPlan(
        pack_buffer.get_ptr() + 8448,
        np.array([0], dtype=np.int64),
        (64 + dcp_size - 1) // dcp_size,
        8448,
        [],
    )
    pack_dsa_plan(plan, pack_buffer, dcp_size, 0)
