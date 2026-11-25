from __future__ import annotations

import functools
import json
import logging
import math
from contextvars import ContextVar
from functools import wraps
from itertools import groupby
from pathlib import Path
from typing import FrozenSet, List, Optional, Sequence

import torch

from sglang.srt.environ import envs
from sglang.srt.utils.log_utils import print_warning_once

logger = logging.getLogger(__name__)

_SWITCH = "SGLANG_K3_ROUTED_MOE_DEEPGEMM"
_DEFAULT_TABLE = Path(__file__).with_name("k3_die_local_moe_widths.json")
# Rows of the MegaMoE buffer; no dispatched width may exceed it
MAX_VERIFY_ROWS = 256
# (num_experts, num_local_experts, w13 shape, w2 shape) validated: Kimi K3.1, EP8
_VALIDATED_SHAPES = ((896, 112, (112, 6144, 1792), (112, 3584, 1536)),)
_VALIDATED_SM = (10, 7)


def enabled() -> bool:
    return envs.SGLANG_K3_ROUTED_MOE_DEEPGEMM.get()


@functools.lru_cache(maxsize=1)
def ar_fusion() -> bool:
    return bool(envs.SGLANG_K3_AR_FUSION.get())


@functools.lru_cache(maxsize=1)
def table_path() -> Path:
    override = envs.SGLANG_K3_ROUTED_MOE_DEEPGEMM_VERIFY_WIDTHS.get()
    return Path(override) if override else _DEFAULT_TABLE


@functools.lru_cache(maxsize=1)
def verify_widths() -> FrozenSet[int]:
    """Row counts of a target-verify call that run on the MegaMoE kernel."""
    if not enabled():
        return frozenset()
    path = table_path()
    key = "widths_ar_fusion" if ar_fusion() else "widths"
    table = json.loads(path.read_text())
    if key not in table:
        raise ValueError(f"{path} has no {key!r} list")
    widths = table[key]
    if not widths or not all(
        type(w) is int and 1 <= w <= MAX_VERIFY_ROWS for w in widths
    ):
        raise ValueError(
            f"{path}: {key!r} is a non-empty list of row counts in [1, {MAX_VERIFY_ROWS}]"
        )
    return frozenset(widths)


def reset_caches() -> None:
    """Forget what was read from the environment (tests)."""
    for cached in (ar_fusion, table_path, verify_widths):
        cached.cache_clear()
    _seen.clear()


# (the forward is a target verify, the stream is capturing a CUDA graph)
_verify: ContextVar = ContextVar("k3_target_verify_forward", default=(False, False))
_seen = set()
_summary = dict(layers=0, logged=False)


def target_forward(fn):
    """Decorator of the K3 model forward: marks target-verify forwards for `takes`."""
    if not enabled():
        return fn

    @wraps(fn)
    def scoped(self, input_ids, positions, forward_batch, *args, **kwargs):
        _log_startup()
        token = _verify.set(
            (
                forward_batch.forward_mode.is_target_verify(),
                torch.cuda.is_current_stream_capturing(),
            )
        )
        try:
            return fn(self, input_ids, positions, forward_batch, *args, **kwargs)
        finally:
            _verify.reset(token)

    return scoped


def takes(rows: int) -> bool:
    """Whether a routed call of `rows` rows in the current forward runs on MegaMoE."""
    return _verify.get()[0] and rows in verify_widths()


def dispatch(rows: int) -> bool:
    """`takes`, logging each new (rows, mode, kernel) combination once at DEBUG."""
    taken = takes(rows)
    if logger.isEnabledFor(logging.DEBUG) and len(_seen) < 512:
        target, capturing = _verify.get()
        key = (int(rows), target, taken, capturing)
        if key not in _seen:
            _seen.add(key)
            logger.debug(
                "K3 die-local routed MoE dispatch: rows=%d target_verify=%s kernel=%s capturing=%s",
                key[0],
                target,
                "mega_moe" if taken else "trtllm_gen",
                capturing,
            )
    return taken


def _ineligible(layer: torch.nn.Module) -> Optional[str]:
    device = layer.w13_weight.device
    capability = torch.cuda.get_device_capability(device)
    if capability != _VALIDATED_SM:
        return f"SM {capability[0]}.{capability[1]} device, validated on SM 10.7 only"
    shape = (
        layer.num_experts,
        layer.num_local_experts,
        tuple(layer.w13_weight.shape),
        tuple(layer.w2_weight.shape),
    )
    if shape not in _VALIDATED_SHAPES:
        return (
            f"expert shape {shape} (experts, experts per rank, FC1, FC2) is not one "
            f"the shared weight layout was validated for: {_VALIDATED_SHAPES}"
        )
    for weight in (layer.w13_weight, layer.w2_weight):
        reason = _placement_unavailable(
            tuple(weight.shape), weight.element_size(), device
        )
        if reason is not None:
            return reason
    return None


def wanted(layer: torch.nn.Module, with_bias: bool, precision: str) -> bool:
    """Whether this layer keeps die-local weights shared by both kernels."""
    if not enabled():
        return False
    from sglang.srt.layers.moe import k3_deepgemm_routed

    reason = k3_deepgemm_routed.mismatch(layer, with_bias, precision) or _ineligible(
        layer
    )
    if reason is not None:
        print_warning_once(
            f"{_SWITCH} is set but not used ({reason}); the routed experts stay on the default path"
        )
        return False
    verify_widths()  # a bad table fails here, at load
    return True


def page_homes(shape: Sequence[int], itemsize: int, page: int) -> List[int]:
    """Home domain of each page; domain 0 reads rows [0, N / 2) of every expert."""
    total = math.prod(shape) * itemsize
    half = total // (shape[0] * 2)
    homes = []
    for start in range(0, total, page):
        counts = [0, 0]
        pos = start
        while pos < start + page:
            end = min(start + page, (pos // half + 1) * half)
            counts[pos // half % 2] += end - pos
            pos = end
        if counts[0] == counts[1]:
            homes.append((start // page) % 2)
        else:
            homes.append(int(counts[1] > counts[0]))
    return homes


def _domain_props(device_index: int):
    from cuda.bindings import driver as cu

    props = []
    for domain in (0, 1):
        prop = cu.CUmemAllocationProp()
        prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = (
            cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN
        )
        prop.location.localized.deviceId = device_index
        prop.location.localized.localityDomainId = domain
        props.append(prop)
    return props


@functools.lru_cache(maxsize=None)
def _page_bytes(device_index: int) -> int:
    """Allocation granularity of the device's two locality domains."""
    from cuda.bindings import driver as cu

    from sglang.srt.utils.cuda_vmm_utils import get_allocation_granularity

    attr = getattr(
        cu.CUdevice_attribute, "CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT", None
    )
    if attr is None:
        raise RuntimeError("cuda.bindings has no locality-domain API")
    err, domains = cu.cuDeviceGetAttribute(attr, device_index)
    if err != cu.CUresult.CUDA_SUCCESS or domains != 2:
        raise RuntimeError("the device does not have two locality domains")
    first, second = (
        get_allocation_granularity(prop) for prop in _domain_props(device_index)
    )
    if first != second or first <= 0:
        raise RuntimeError(
            f"locality domains differ in allocation granularity ({first}, {second})"
        )
    return first


def _placement_unavailable(shape, itemsize: int, device) -> Optional[str]:
    try:
        page = _page_bytes(device.index)
    except Exception as e:  # bindings, driver or device dependent
        return f"die-local placement is not available: {e}"
    total = math.prod(shape) * itemsize
    if total % page or total % (shape[0] * 2):
        return (
            f"a {shape} weight does not split into {page}-byte pages and expert halves"
        )
    return None


class _PlacedPages:
    """Owns the reservation behind an `_empty_die_local` tensor."""

    def __init__(self, reservation, num_bytes: int):
        self.reservation = reservation
        self.__cuda_array_interface__ = {
            "shape": (num_bytes,),
            "typestr": "|u1",
            "data": (reservation.base, False),
            "version": 3,
        }

    def __del__(self):
        torch.cuda.synchronize()
        self.reservation.close()


def _domain_of(address: int) -> Optional[int]:
    """Locality domain the page at `address` is homed on, or None."""
    from cuda.bindings import driver as cu

    attr = getattr(
        cu.CUpointer_attribute, "CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL", None
    )
    if attr is None:
        return None
    err, value = cu.cuPointerGetAttribute(attr, address)
    return int(value) if err == cu.CUresult.CUDA_SUCCESS else None


def _empty_die_local(shape, dtype: torch.dtype, device) -> torch.Tensor:
    """Uninitialized tensor on driver-allocated pages homed as `page_homes` says."""
    from cuda.bindings import driver as cu

    from sglang.srt.utils.cuda_vmm_utils import VmmReservation, check_drv

    index = device.index
    page = _page_bytes(index)
    itemsize = torch.empty((), dtype=dtype).element_size()
    homes = page_homes(shape, itemsize, page)
    total = len(homes) * page
    props = _domain_props(index)
    reservation = VmmReservation(total, props[0], index, alignment=page)
    try:
        offset = 0
        for home, run in groupby(homes):
            pages = len(list(run))
            # the driver rounds a locality-domain allocation up to a power of two pages
            while pages:
                size = (1 << (pages.bit_length() - 1)) * page
                handle = check_drv(
                    cu.cuMemCreate(size, props[home], 0), "cuMemCreate(locality)"
                )
                try:
                    reservation.map_existing(offset, size, handle)
                finally:
                    check_drv(cu.cuMemRelease(handle), "cuMemRelease(locality)")
                offset += size
                pages -= size // page
    except BaseException:
        reservation.close()
        raise
    out = torch.as_tensor(_PlacedPages(reservation, total), device=device)
    if out.data_ptr() != reservation.base:
        raise RuntimeError("die-local weight pages were copied instead of wrapped")
    for idx, home in enumerate(homes):
        actual = _domain_of(reservation.base + idx * page)
        if actual is None:
            break  # these bindings cannot read a page's home back
        if actual != home:
            raise RuntimeError(
                f"die-local weight page {idx} of {len(homes)} is homed on domain "
                f"{actual}, not {home}"
            )
    return out.view(dtype).view(*shape)


def stack_final(parts: List[torch.Tensor]) -> torch.Tensor:
    """`torch.stack(parts)` on die-local pages."""
    first = parts[0]
    device = first.device
    # the pages come from the driver, not torch's allocator: hand torch's idle
    # blocks back first (the engine sizes its KV pool from free device memory)
    reserved = torch.cuda.memory_reserved(device)
    allocated = torch.cuda.memory_allocated(device)
    free_before = torch.cuda.mem_get_info(device)[0]
    if reserved - allocated >= 2**31 or free_before < 4 * (2**30):
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        logger.debug(
            "K3 die-local routed MoE: released idle blocks, reserved %d -> %d, allocated %d, device free %d -> %d",
            reserved,
            torch.cuda.memory_reserved(device),
            allocated,
            free_before,
            torch.cuda.mem_get_info(device)[0],
        )
    with torch.cuda.device(device):
        dest = _empty_die_local((len(parts), *first.shape), first.dtype, device)
    torch.stack(parts, out=dest)
    return dest


def finish(layer: torch.nn.Module, prefix: str) -> None:
    """Expose the layer's weights and scales to MegaMoE as zero-copy views."""
    w13, s13 = layer.w13_weight, layer.w13_weight_scale
    w2, s2 = layer.w2_weight, layer.w2_weight_scale
    layer.mega_l1_weights = (w13.view(torch.int8), s13.view(torch.int32))
    layer.mega_l2_weights = (w2.view(torch.int8), s2.view(torch.int32))
    for view, param in zip(
        (*layer.mega_l1_weights, *layer.mega_l2_weights), (w13, s13, w2, s2)
    ):
        assert view.untyped_storage().data_ptr() == param.untyped_storage().data_ptr()
    layer.use_cuda_mxfp4_decode_moe = False
    _summary.update(
        layers=_summary["layers"] + 1,
        rank=layer.moe_ep_rank,
        experts=w13.shape[0],
        weight_bytes=w13.nbytes + w2.nbytes,
    )
    logger.debug(
        "K3 die-local routed MoE: %s rank %d on %s, w13 %d bytes, w2 %d bytes",
        prefix,
        layer.moe_ep_rank,
        w13.device,
        w13.nbytes,
        w2.nbytes,
    )


def _log_startup() -> None:
    """One line per process (EP rank), at its first model forward."""
    if _summary["logged"] or not _summary["layers"]:
        return
    _summary["logged"] = True
    widths = sorted(verify_widths())
    logger.info(
        "K3 die-local routed MoE active: rank=%d layers=%d experts_per_rank=%d "
        "weight_bytes_per_layer=%d verify_widths=%d (%s rows) ar_fusion=%s table=%s",
        _summary["rank"],
        _summary["layers"],
        _summary["experts"],
        _summary["weight_bytes"],
        len(widths),
        f"{widths[0]}-{widths[-1]}" if widths else "none",
        ar_fusion(),
        table_path().name,
    )
