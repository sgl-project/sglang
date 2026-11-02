"""Optional deterministic KV initialization for fake-prefill benchmarks."""

import torch


def zero_fake_mha_kv_cache(*pools):
    """Zero MHA/SWA cache contents in place, then wait for device completion.

    The caller must have drained all requests. Validate every pool before
    writing any tensor so an unsupported cache cannot cause a partial reset.
    """
    tensors = []
    visited_pools = set()
    visited_regions = set()

    def collect(pool):
        if pool is None or id(pool) in visited_pools:
            return
        visited_pools.add(id(pool))
        if hasattr(pool, "full_kv_pool"):
            collect(pool.full_kv_pool)
            collect(getattr(pool, "swa_kv_pool", None))
            return
        for name in ("k_buffer", "v_buffer"):
            buffers = getattr(pool, name, None)
            if not isinstance(buffers, (list, tuple)):
                raise TypeError("Fake-prefill KV reset requires MHA/SWA K/V buffers")
            for tensor in buffers:
                if not isinstance(tensor, torch.Tensor) or not tensor.is_contiguous():
                    raise TypeError("Fake-prefill KV reset requires contiguous tensors")
                if tensor.numel() == 0:
                    continue
                # Distinct contiguous slices may share one storage allocation.
                region = (
                    tensor.device,
                    tensor.data_ptr(),
                    tensor.numel(),
                    tensor.dtype,
                )
                if region not in visited_regions:
                    visited_regions.add(region)
                    tensors.append(tensor)

    for pool in pools:
        collect(pool)
    if not tensors:
        raise ValueError("No KV buffers were found for fake-prefill reset")
    devices = {tensor.device for tensor in tensors if tensor.is_cuda}
    for device in devices:
        torch.cuda.synchronize(device)
    with torch.no_grad():
        for tensor in tensors:
            tensor.zero_()
    for device in devices:
        torch.cuda.synchronize(device)
    return {
        "buffers": len(tensors),
        "bytes": sum(tensor.numel() * tensor.element_size() for tensor in tensors),
    }
