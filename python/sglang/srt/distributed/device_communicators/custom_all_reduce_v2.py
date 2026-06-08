import logging
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, TypeVar

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.jit_kernel.all_reduce import AllReduceAlgo, get_custom_all_reduce_cls
from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
    can_use_custom_all_reduce_with_nvlink,
    is_weak_contiguous,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.utils import is_sm100_supported, log_info_on_rank0

logger = logging.getLogger(__name__)

T = TypeVar("T")

_drv = None


def _get_cuda_driver():
    """Lazily import cuda.bindings.driver (cached after first call)."""
    global _drv
    if _drv is None:
        from cuda.bindings import driver

        _drv = driver
    return _drv


def _check_drv(result_tuple, label):
    """Check a cuda.bindings driver call result and return the value."""
    if not isinstance(result_tuple, tuple):
        result_tuple = (result_tuple,)
    err = result_tuple[0]
    drv = _get_cuda_driver()
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{label}: {err}")
    return result_tuple[1] if len(result_tuple) > 1 else None


def _is_vmm_pointer(ptr: int) -> bool:
    """Check if a device pointer is VMM-backed (cuMemCreate/cuMemMap).

    cuMemRetainAllocationHandle succeeds only on pointers from cuMemCreate;
    it fails on cudaMalloc pointers.
    """
    drv = _get_cuda_driver()
    err, handle = drv.cuMemRetainAllocationHandle(ptr)
    if err == drv.CUresult.CUDA_SUCCESS:
        drv.cuMemRelease(handle)
        return True
    return False


INF = 1 << 60


@dataclass(frozen=True)
class ModeConfig:
    one_shot_push_threshold: int  # below this, use one-shot push
    one_shot_pull_threshold: int  # below this, use one-shot pull


class CustomAllReduceV2:
    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        max_pull_size: Optional[int] = None,
        max_push_size: Optional[int] = None,
        max_pull_blocks: Optional[int] = None,
        max_push_blocks: Optional[int] = None,
    ) -> None:
        _maybe_init_config()
        self.disabled = True
        self._fabric_peer_mappings = []
        if not can_use_custom_all_reduce_v2(group=group, device=device):
            return

        self.group = group
        self.rank = dist.get_rank(group=self.group)
        self.world_size = dist.get_world_size(group=self.group)
        if max_pull_size is None:  # default to 16MB
            max_pull_size = 16 * 1024 * 1024
        if max_push_size is None:  # default to recommended size
            config = THRESHOLD_2_SHOT_MAP[self.world_size]
            max_push_size = config.one_shot_push_threshold
        self.max_pull_size = max_pull_size
        self.max_push_size = max_push_size
        self.max_size = max(max_pull_size, max_push_size)
        self.override_shot(None)  # set default config based on world size
        self.override_algo: Optional[AllReduceAlgo] = None
        self.obj = get_custom_all_reduce_cls()(
            rank=self.rank,
            world_size=self.world_size,
            pull_buffer_bytes=self.max_pull_size,
            push_buffer_bytes=self.max_push_size,
            graph_input_count=131072,
            max_pull_blocks=max_pull_blocks,
            max_push_blocks=max_push_blocks,
        )
        self._post_init_obj()
        self.disabled = False
        log_info_on_rank0(logger, "Custom allreduce v2 initialized successfully")

    def override_shot(self, shot: int | None):
        if shot is None:
            config = THRESHOLD_2_SHOT_MAP[self.world_size]
        else:
            assert shot in (1, 2)
            threshold = INF if shot == 1 else 0
            config = replace(self.config, one_shot_pull_threshold=threshold)
        # need to clip the config thresholds to max sizes to avoid invalid config
        push_threshold = min(config.one_shot_push_threshold, self.max_push_size)
        pull_threshold = min(config.one_shot_pull_threshold, self.max_pull_size)
        self.config: ModeConfig = replace(
            config,
            one_shot_push_threshold=push_threshold,
            one_shot_pull_threshold=pull_threshold,
        )

    @contextmanager
    def capture(self):
        if self.disabled:
            yield
            return
        try:
            self.obj.set_cuda_graph_capture(True)
            yield
        finally:
            self.obj.set_cuda_graph_capture(False)
        assert (
            torch.cuda.is_current_stream_capturing() == False
        ), "Cannot register graph inputs while capturing CUDA graph"
        raw_ptrs = self.obj.get_graph_capture_ptrs()
        if raw_ptrs and _is_vmm_pointer(raw_ptrs[0]):
            self._register_graph_inputs_fabric()
        else:
            self._register_graph_inputs_ipc()

    def _register_graph_inputs_ipc(self):
        """Register graph capture inputs via cudaIpcGetMemHandle.

        This is the fast path for cudaMalloc-backed allocations. Fails
        on VMM pointers (expandable_segments).
        """
        pairs = self.obj.share_graph_inputs()
        handles = [handle for _, handle in pairs]
        offsets = [offset for offset, _ in pairs]
        handles_all = self._share_list(handles)
        offsets_all = self._share_list(offsets)
        result = [list(zip(o, h)) for o, h in zip(offsets_all, handles_all)]
        self.obj.register_inputs(result)
        log_info_on_rank0(
            logger, f"Registered {len(pairs)} cuda graph addresses via IPC"
        )

    def _register_graph_inputs_fabric(self):
        """Register graph capture inputs via FABRIC handle exchange.

        VMM-compatible path for expandable_segments. The C++ side deduplicates
        graph capture pointers into unique base allocations via cuMemGetAddressRange.
        Python exports FABRIC handles for each unique base, all-gathers them
        imports + cuMemMaps peer allocations, then registers the peer VAs.
        """
        import struct as _struct
        import time as _time

        drv = _get_cuda_driver()
        FABRIC = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        FABRIC_HANDLE_BYTES = 64
        MAX_FABRIC_BASES = 4096
        MAX_CHUNKS_PER_INPUT = 16

        _t0 = _time.perf_counter()

        bases_info, input_chunk_indices, input_offsets = (
            self.obj.get_graph_capture_bases()
        )
        if not bases_info:
            return
        new_count = len(input_chunk_indices)
        num_bases = len(bases_info)
        device_id = torch.cuda.current_device()

        # Export FABRIC handles for each unique base allocation.
        local_fabric_handles: List[bytes] = []
        retained_handles = []
        for base_ptr, _ in bases_info:
            alloc_h = _check_drv(
                drv.cuMemRetainAllocationHandle(base_ptr),
                "cuMemRetainAllocationHandle",
            )
            retained_handles.append(alloc_h)
            fabric_h = _check_drv(
                drv.cuMemExportToShareableHandle(alloc_h, FABRIC, 0),
                "cuMemExportToShareableHandle(FABRIC)",
            )
            local_fabric_handles.append(bytes(fabric_h.data))

        if num_bases > MAX_FABRIC_BASES:
            raise RuntimeError(
                f"Too many VMM bases to share: {num_bases} > {MAX_FABRIC_BASES}"
            )

        local_input_chunks = [
            [int(idx) for idx in indices] for indices in input_chunk_indices
        ]
        for chunks in local_input_chunks:
            if len(chunks) > MAX_CHUNKS_PER_INPUT:
                raise RuntimeError(
                    "Too many VMM chunks for graph input: "
                    f"{len(chunks)} > {MAX_CHUNKS_PER_INPUT}"
                )

        # All-gather base handles and per-input VMM spans. A captured tensor can
        # cross expandable-segment allocation boundaries, so peer mappings must
        # preserve each input's contiguous virtual-address span.
        header_struct = _struct.Struct("<QQ")
        base_struct = _struct.Struct(f"<QQ{FABRIC_HANDLE_BYTES}s")
        input_struct = _struct.Struct(f"<QQ{MAX_CHUNKS_PER_INPUT}Q")
        base_offset = header_struct.size
        input_offset = base_offset + MAX_FABRIC_BASES * base_struct.size
        payload_size = input_offset + new_count * input_struct.size
        local_payload = bytearray(payload_size)

        header_struct.pack_into(local_payload, 0, num_bases, new_count)
        for i, (base_ptr, alloc_size) in enumerate(bases_info):
            base_struct.pack_into(
                local_payload,
                base_offset + i * base_struct.size,
                int(base_ptr),
                int(alloc_size),
                local_fabric_handles[i],
            )
        for i, (chunks, offset) in enumerate(zip(local_input_chunks, input_offsets)):
            padded_chunks = chunks + [0] * (MAX_CHUNKS_PER_INPUT - len(chunks))
            input_struct.pack_into(
                local_payload,
                input_offset + i * input_struct.size,
                int(offset),
                len(chunks),
                *padded_chunks,
            )

        in_buf = torch.frombuffer(local_payload, dtype=torch.uint8).clone()
        gather_list = [torch.empty_like(in_buf) for _ in range(self.world_size)]
        dist.all_gather(gather_list, in_buf, group=self.group)

        all_base_payload = []
        all_input_chunks = []
        all_input_offsets = []
        for rank, gathered in enumerate(gather_list):
            payload = gathered.numpy().tobytes()
            peer_num_bases, peer_new_count = header_struct.unpack_from(payload, 0)
            if peer_new_count != new_count:
                raise RuntimeError(
                    "Mismatched graph input count across ranks: "
                    f"rank {rank} has {peer_new_count}, expected {new_count}"
                )

            peer_bases = []
            for i in range(peer_num_bases):
                base_ptr, alloc_size, fabric_handle = base_struct.unpack_from(
                    payload, base_offset + i * base_struct.size
                )
                peer_bases.append((base_ptr, fabric_handle, alloc_size))

            peer_chunks = []
            peer_offsets = []
            for i in range(new_count):
                unpacked = input_struct.unpack_from(
                    payload, input_offset + i * input_struct.size
                )
                offset, chunk_count, *chunks = unpacked
                peer_offsets.append(offset)
                peer_chunks.append(list(chunks[:chunk_count]))

            all_base_payload.append(peer_bases)
            all_input_chunks.append(peer_chunks)
            all_input_offsets.append(peer_offsets)

        # Import + map peer allocations. Individual base mappings are kept for
        # single-chunk inputs; span mappings reserve a contiguous VA range and
        # map each chunk at its original relative offset.
        peer_base_va = {}  # (rank, base_idx) -> local VA
        peer_span_va = {}  # (rank, chunk_indices...) -> (local VA, peer base)
        new_mappings = []

        for peer_rank in range(self.world_size):
            if peer_rank == self.rank:
                for idx, (bp, _) in enumerate(bases_info):
                    peer_base_va[(peer_rank, idx)] = int(bp)
                continue

            peer_bases = all_base_payload[peer_rank]
            for idx, (_, fb, alloc_size) in enumerate(peer_bases):

                imp_h = _check_drv(
                    drv.cuMemImportFromShareableHandle(fb, FABRIC),
                    f"cuMemImportFromShareableHandle(rank={peer_rank})",
                )
                prop = _check_drv(
                    drv.cuMemGetAllocationPropertiesFromHandle(imp_h),
                    "cuMemGetAllocationPropertiesFromHandle",
                )
                gran = _check_drv(
                    drv.cuMemGetAllocationGranularity(
                        prop,
                        drv.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
                    ),
                    "cuMemGetAllocationGranularity",
                )
                va = _check_drv(
                    drv.cuMemAddressReserve(alloc_size, int(gran), 0, 0),
                    "cuMemAddressReserve",
                )
                _check_drv(drv.cuMemMap(int(va), alloc_size, 0, imp_h, 0), "cuMemMap")
                access = drv.CUmemAccessDesc()
                access.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
                access.location.id = device_id
                access.flags = drv.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                _check_drv(
                    drv.cuMemSetAccess(int(va), alloc_size, [access], 1),
                    "cuMemSetAccess",
                )
                peer_base_va[(peer_rank, idx)] = int(va)
                new_mappings.append((int(va), alloc_size, [(0, alloc_size)]))
                _check_drv(drv.cuMemRelease(imp_h), "cuMemRelease(peer)")

        # Build per-input peer VA lists and register.
        peer_ptrs = []
        for j in range(new_count):
            ptrs_j = []
            for rank in range(self.world_size):
                chunks = all_input_chunks[rank][j]
                off = all_input_offsets[rank][j]
                if len(chunks) == 1:
                    ptrs_j.append(peer_base_va[(rank, chunks[0])] + off)
                    continue

                span_key = (rank, *chunks)
                if span_key not in peer_span_va:
                    peer_bases = all_base_payload[rank]
                    first_base = peer_bases[chunks[0]][0]
                    last_base, _, last_size = peer_bases[chunks[-1]]
                    span_size = int(last_base) + int(last_size) - int(first_base)
                    if rank == self.rank:
                        span_va = int(first_base)
                    else:
                        span_va = _check_drv(
                            drv.cuMemAddressReserve(span_size, 0, 0, 0),
                            "cuMemAddressReserve(span)",
                        )
                        mapped_chunks = []
                        for chunk_idx in chunks:
                            base_ptr, fb, alloc_size = peer_bases[chunk_idx]
                            rel = int(base_ptr) - int(first_base)
                            imp_h = _check_drv(
                                drv.cuMemImportFromShareableHandle(fb, FABRIC),
                                f"cuMemImportFromShareableHandle(rank={rank}, span)",
                            )
                            _check_drv(
                                drv.cuMemMap(
                                    int(span_va) + rel, int(alloc_size), 0, imp_h, 0
                                ),
                                "cuMemMap(span)",
                            )
                            access = drv.CUmemAccessDesc()
                            access.location.type = (
                                drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
                            )
                            access.location.id = device_id
                            access.flags = (
                                drv.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                            )
                            _check_drv(
                                drv.cuMemSetAccess(
                                    int(span_va) + rel, int(alloc_size), [access], 1
                                ),
                                "cuMemSetAccess(span)",
                            )
                            mapped_chunks.append((rel, int(alloc_size)))
                            _check_drv(drv.cuMemRelease(imp_h), "cuMemRelease(span)")
                        new_mappings.append((int(span_va), span_size, mapped_chunks))
                    peer_span_va[span_key] = (int(span_va), int(first_base))

                span_va, _ = peer_span_va[span_key]
                ptrs_j.append(span_va + off)
            peer_ptrs.append(ptrs_j)

        self.obj.register_peer_mapped_inputs(peer_ptrs)
        self._fabric_peer_mappings.extend(new_mappings)

        for h in retained_handles:
            _check_drv(drv.cuMemRelease(h), "cuMemRelease(retained)")

        _elapsed_ms = (_time.perf_counter() - _t0) * 1000
        log_info_on_rank0(
            logger,
            f"Registered {new_count} cuda graph addresses via "
            f"FABRIC handles ({num_bases} unique allocations) "
            f"in {_elapsed_ms:.1f} ms",
        )

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        """Check if the input tensor is suitable for custom all-reduce."""
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        # custom allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        return inp_size <= self.max_size

    def custom_all_reduce(self, input: torch.Tensor) -> torch.Tensor:
        if is_in_tc_piecewise_cuda_graph():  # disable inplace optimization
            try:
                self.obj.set_cuda_graph_capture(False)
                return self._all_reduce(input)
            finally:
                self.obj.set_cuda_graph_capture(True)
        return self._all_reduce(input)

    def close(self):
        if not self.disabled and hasattr(self, "obj"):
            self.obj.free(self.group)
        self._release_fabric_peer_mappings()

    def _release_fabric_peer_mappings(self):
        if not getattr(self, "_fabric_peer_mappings", None):
            return
        drv = _get_cuda_driver()
        while self._fabric_peer_mappings:
            va, span_size, mapped_chunks = self._fabric_peer_mappings.pop()
            for rel, size in mapped_chunks:
                _check_drv(drv.cuMemUnmap(int(va) + int(rel), int(size)), "cuMemUnmap")
            _check_drv(
                drv.cuMemAddressFree(int(va), int(span_size)), "cuMemAddressFree"
            )

    def _all_reduce(self, input: torch.Tensor) -> torch.Tensor:
        """Perform the actual all-reduce via JIT kernel."""
        algo = self._determine_algo(input)
        return torch.from_dlpack(self.obj.all_reduce(input, algo))

    def _determine_algo(self, input: torch.Tensor) -> AllReduceAlgo:
        if self.override_algo is not None:
            return self.override_algo
        input_bytes = input.numel() * input.element_size()
        if input_bytes <= self.config.one_shot_push_threshold:
            return AllReduceAlgo.ONE_SHOT_PUSH
        if input_bytes <= self.config.one_shot_pull_threshold:
            return AllReduceAlgo.ONE_SHOT_PULL
        else:
            return AllReduceAlgo.TWO_SHOT_PULL

    def _post_init_obj(self):
        handles = [self.obj.share_storage()]
        result = self._share_list(handles)
        assert all(len(r) == 1 for r in result)
        result = [h[0] for h in result]
        self.obj.post_init(result)

    def _share_list(self, input: List[T]) -> List[List[T]]:
        input_tensor = torch.tensor(input, dtype=torch.int64, device="cpu")
        gather_list = [torch.empty_like(input_tensor) for _ in range(self.world_size)]
        dist.all_gather(gather_list, input_tensor, group=self.group)
        return [g.tolist() for g in gather_list]

    def __del__(self):
        self.close()


def _maybe_init_config():
    global THRESHOLD_2_SHOT_MAP
    if THRESHOLD_2_SHOT_MAP:
        return
    KB, MB = 1024, 1024 * 1024

    if is_sm100_supported():
        # NOTE: This result is based on benchmarks on B200 GPUs
        THRESHOLD_2_SHOT_MAP = {
            2: ModeConfig(4 * MB, INF),
            3: ModeConfig(4 * MB, 4 * MB),
            4: ModeConfig(2 * MB, 2 * MB),
            5: ModeConfig(2 * MB, 2 * MB),
            6: ModeConfig(1 * MB, 1 * MB),
            7: ModeConfig(896 * KB, 896 * KB),
            8: ModeConfig(720 * KB, 720 * KB),
        }
    else:
        # NOTE: This result is based on benchmarks on H200 GPUs
        THRESHOLD_2_SHOT_MAP = {
            2: ModeConfig(2 * MB, INF),
            3: ModeConfig(512 * KB, 512 * KB),
            4: ModeConfig(384 * KB, 256 * KB),
            5: ModeConfig(256 * KB, 256 * KB),
            6: ModeConfig(192 * KB, 192 * KB),
            7: ModeConfig(192 * KB, 192 * KB),
            8: ModeConfig(160 * KB, 160 * KB),
        }
    # TODO: tune on more GPUs, e.g A100


def can_use_custom_all_reduce_v2(
    group: ProcessGroup,
    device: torch.device,
) -> bool:
    # call _maybe_init_config() to ensure THRESHOLD_2_SHOT_MAP is initialized, since can_use_custom_all_reduce_v2 can be called before CustomAllReduceV2 is initialized
    _maybe_init_config()
    full_nvlink = can_use_custom_all_reduce_with_nvlink(
        group=group,
        device=device,
        supported_world_size=list(THRESHOLD_2_SHOT_MAP.keys()),
        cls_name="CustomAllReduceV2",
    )
    return full_nvlink is True


THRESHOLD_2_SHOT_MAP: Dict[int, ModeConfig] = {}
