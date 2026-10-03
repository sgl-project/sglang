import logging
import mmap
import os
import queue
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Generator, List, Optional, Sequence, Tuple

import msgspec
import torch

from sglang.srt.utils.offloader import (
    BaseOffloader,
    _get_resident_parameter_ids,
    _hook_module_forward_raw,
    _SubmoduleAccessor,
    _WhitelistParamNamesCreator,
)

logger = logging.getLogger(__name__)

_IO_ALIGNMENT = 4096
_READ_CHUNK_BYTES = 64 << 20
_NUM_STAGING_BUFFERS = 6
_NUM_PARALLEL_READS = 4
_MIN_OFFLOAD_PARAM_BYTES = 1 << 20


def _align_up(value: int) -> int:
    return (value + _IO_ALIGNMENT - 1) // _IO_ALIGNMENT * _IO_ALIGNMENT


def _storage_span(tensor: torch.Tensor) -> int:
    if tensor.numel() == 0:
        return 0
    return 1 + sum(
        (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
    )


class _ParamSlice(msgspec.Struct, frozen=True):
    name: str
    byte_offset: int
    span: int
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    dtype: torch.dtype

    @property
    def nbytes(self) -> int:
        return self.span * self.dtype.itemsize

    def view_of(self, buffer: torch.Tensor) -> torch.Tensor:
        flat = buffer[self.byte_offset : self.byte_offset + self.nbytes]
        return flat.view(self.dtype).as_strided(self.shape, self.stride)


def _plan_layout(
    named_tensors: Sequence[Tuple[str, torch.Tensor]],
) -> Tuple[Tuple[_ParamSlice, ...], int]:
    slices = []
    cursor = 0
    for name, tensor in named_tensors:
        param_slice = _ParamSlice(
            name=name,
            byte_offset=cursor,
            span=_storage_span(tensor),
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            dtype=tensor.dtype,
        )
        slices.append(param_slice)
        cursor = _align_up(cursor + param_slice.nbytes)
    return tuple(slices), max(cursor, _IO_ALIGNMENT)


class _BackingFile:
    def __init__(self, directory: str, nbytes: int):
        self.nbytes = nbytes
        path = os.path.join(directory, f"sglang-offload-{uuid.uuid4().hex}.bin")
        self._sync_fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
        self._direct_fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        os.unlink(path)
        os.ftruncate(self._sync_fd, nbytes)
        self._memory_map = mmap.mmap(self._sync_fd, nbytes, flags=mmap.MAP_SHARED)
        self.mapping: Optional[torch.Tensor] = torch.frombuffer(
            self._memory_map, dtype=torch.uint8
        )

    def evict_page_cache(self):
        self._memory_map.madvise(mmap.MADV_DONTNEED)
        os.fsync(self._sync_fd)
        os.posix_fadvise(self._sync_fd, 0, 0, os.POSIX_FADV_DONTNEED)

    def contains(self, tensor: torch.Tensor, param_slice: _ParamSlice) -> bool:
        return (
            self.mapping is not None
            and tensor.device.type == "cpu"
            and tensor.dtype == param_slice.dtype
            and tuple(tensor.shape) == param_slice.shape
            and tuple(tensor.stride()) == param_slice.stride
            and tensor.data_ptr() == self.mapping.data_ptr() + param_slice.byte_offset
        )

    def drop_mapping(self):
        self.mapping = None
        self.evict_page_cache()

    def read_into(
        self,
        buffer: torch.Tensor,
        file_offset: int,
        nbytes: int,
        executor: Optional[ThreadPoolExecutor] = None,
    ):
        target = memoryview(buffer.numpy())[: _align_up(nbytes)]
        if executor is None:
            self._read_range(target, file_offset)
            return
        piece_nbytes = _align_up(-(-len(target) // _NUM_PARALLEL_READS))
        pieces = [
            executor.submit(
                self._read_range,
                target[start : start + piece_nbytes],
                file_offset + start,
            )
            for start in range(0, len(target), piece_nbytes)
        ]
        for piece in pieces:
            piece.result()

    def _read_range(self, target: memoryview, file_offset: int):
        done = 0
        while done < len(target):
            count = os.preadv(self._direct_fd, [target[done:]], file_offset + done)
            if count <= 0:
                raise IOError(
                    f"Short read from offloaded weights at offset {file_offset + done}"
                )
            done += count

    def close(self):
        self.mapping = None
        os.close(self._direct_fd)
        os.close(self._sync_fd)


class _OffloadedModule:
    def __init__(self, module: torch.nn.Module, param_names: List[str], directory: str):
        self.module = module
        self.param_names = param_names
        self.layout, self.nbytes = _plan_layout(self._named_tensors())
        self.backing: Optional[_BackingFile] = _BackingFile(directory, self.nbytes)
        self.host_buffer: Optional[torch.Tensor] = None
        self._move_params_into(self.backing)

    def _named_tensors(self) -> List[Tuple[str, torch.Tensor]]:
        return [
            (name, self.module.get_parameter(name).data) for name in self.param_names
        ]

    def _move_params_into(self, backing: _BackingFile):
        for param_slice in self.layout:
            param = self.module.get_parameter(param_slice.name)
            host_view = param_slice.view_of(backing.mapping)
            host_view.copy_(param.data)
            param.data = host_view

    def finalize(self, directory: str):
        named_tensors = self._named_tensors()
        layout, nbytes = _plan_layout(named_tensors)
        if layout != self.layout or not all(
            self.backing.contains(tensor, param_slice)
            for (_, tensor), param_slice in zip(named_tensors, layout)
        ):
            stale_backing = self.backing
            self.layout, self.nbytes = layout, nbytes
            self.backing = _BackingFile(directory, nbytes)
            self._move_params_into(self.backing)
            stale_backing.close()

        for param_slice in self.layout:
            self._replace_with_meta_parameter(param_slice)
        self.backing.drop_mapping()

    def _replace_with_meta_parameter(self, param_slice: _ParamSlice):
        owner_name, _, attribute = param_slice.name.rpartition(".")
        owner = self.module.get_submodule(owner_name)
        stale_param = getattr(owner, attribute)
        meta_param = torch.nn.Parameter(
            torch.empty_strided(
                param_slice.shape,
                param_slice.stride,
                dtype=param_slice.dtype,
                device="meta",
            ),
            requires_grad=False,
        )
        meta_param.__dict__.update(stale_param.__dict__)
        setattr(owner, attribute, meta_param)

    def move_to_host_cache(self, host_buffer: torch.Tensor):
        self.host_buffer = host_buffer
        for chunk_start in range(0, self.nbytes, _READ_CHUNK_BYTES):
            chunk_nbytes = min(_READ_CHUNK_BYTES, self.nbytes - chunk_start)
            self.backing.read_into(
                self.host_buffer[chunk_start:], chunk_start, chunk_nbytes
            )
        self.backing.close()
        self.backing = None

    def device_tensors(self, slot: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            param_slice.name: param_slice.view_of(slot) for param_slice in self.layout
        }


class _FetchJob:
    def __init__(
        self,
        record: _OffloadedModule,
        slot_index: int,
        slot_release: Optional[torch.cuda.Event],
    ):
        self.record = record
        self.slot_index = slot_index
        self.slot_release = slot_release
        self.ready: Optional[torch.cuda.Event] = None
        self.error: Optional[BaseException] = None
        self.done = threading.Event()


class _WeightStreamer:
    def __init__(self, device: torch.device, slot_nbytes: int, num_slots: int):
        self._device = device
        self._slots = [
            torch.empty(slot_nbytes, dtype=torch.uint8, device=device)
            for _ in range(num_slots)
        ]
        self._slot_releases: List[Optional[torch.cuda.Event]] = [None] * num_slots
        self._staging = [
            torch.empty(_READ_CHUNK_BYTES, dtype=torch.uint8, pin_memory=True)
            for _ in range(_NUM_STAGING_BUFFERS)
        ]
        self._staging_releases: List[Optional[torch.cuda.Event]] = [
            None
        ] * _NUM_STAGING_BUFFERS
        self._staging_cursor = 0
        self._copy_stream = torch.cuda.Stream(device=device)
        self._read_executor = ThreadPoolExecutor(
            max_workers=_NUM_PARALLEL_READS, thread_name_prefix="sglang-disk-read"
        )
        self._num_submitted = 0
        self._jobs: "queue.SimpleQueue[_FetchJob]" = queue.SimpleQueue()
        threading.Thread(
            target=self._serve, name="sglang-disk-offload", daemon=True
        ).start()

    def submit(self, record: _OffloadedModule) -> _FetchJob:
        slot_index = self._num_submitted % len(self._slots)
        self._num_submitted += 1
        job = _FetchJob(
            record=record,
            slot_index=slot_index,
            slot_release=self._slot_releases[slot_index],
        )
        self._jobs.put(job)
        return job

    def wait(self, job: _FetchJob) -> torch.Tensor:
        job.done.wait()
        if job.error is not None:
            raise RuntimeError("Fetching offloaded weights failed") from job.error
        torch.cuda.current_stream().wait_event(job.ready)
        return self._slots[job.slot_index]

    def release(self, job: _FetchJob):
        slot_release = torch.cuda.Event()
        slot_release.record()
        self._slot_releases[job.slot_index] = slot_release

    def _serve(self):
        torch.cuda.set_device(self._device)
        while True:
            job = self._jobs.get()
            try:
                self._fetch(job)
            except BaseException as error:
                job.error = error
            job.done.set()

    def _fetch(self, job: _FetchJob):
        slot = self._slots[job.slot_index]
        record = job.record
        with torch.cuda.stream(self._copy_stream):
            if job.slot_release is not None:
                self._copy_stream.wait_event(job.slot_release)
            if record.host_buffer is not None:
                slot[: record.nbytes].copy_(record.host_buffer, non_blocking=True)
            else:
                self._stream_from_disk(record, slot)
            job.ready = torch.cuda.Event()
            job.ready.record(self._copy_stream)

    def _stream_from_disk(self, record: _OffloadedModule, slot: torch.Tensor):
        for chunk_start in range(0, record.nbytes, _READ_CHUNK_BYTES):
            chunk_nbytes = min(_READ_CHUNK_BYTES, record.nbytes - chunk_start)
            staging_index = self._staging_cursor % _NUM_STAGING_BUFFERS
            self._staging_cursor += 1
            staging_release = self._staging_releases[staging_index]
            if staging_release is not None:
                staging_release.synchronize()
            staging = self._staging[staging_index]
            record.backing.read_into(
                staging, chunk_start, chunk_nbytes, executor=self._read_executor
            )
            slot[chunk_start : chunk_start + chunk_nbytes].copy_(
                staging[:chunk_nbytes], non_blocking=True
            )
            staging_release = torch.cuda.Event()
            staging_release.record(self._copy_stream)
            self._staging_releases[staging_index] = staging_release


def _default_offload_param_names(module: torch.nn.Module) -> List[str]:
    resident_ids = _get_resident_parameter_ids(module)
    return [
        name
        for name, param in module.named_parameters()
        if id(param) not in resident_ids
        and param.numel() * param.element_size() >= _MIN_OFFLOAD_PARAM_BYTES
    ]


class DiskOffloader(BaseOffloader):
    def __init__(
        self,
        group_size: int,
        num_in_group: int,
        prefetch_step: int,
        storage_dir: str,
        host_cache_bytes: int,
    ):
        self._group_size = group_size
        self._num_in_group = num_in_group
        self._prefetch_step = max(prefetch_step, 1)
        self._storage_dir = os.path.expanduser(storage_dir)
        self._host_cache_bytes = host_cache_bytes
        self._records: List[_OffloadedModule] = []
        self._host_weights: List[_BackingFile] = []
        self._has_wrapped = False
        self._streamer: Optional[_WeightStreamer] = None
        self._pending_jobs: Dict[int, _FetchJob] = {}
        self._active_jobs: Dict[int, _FetchJob] = {}
        os.makedirs(self._storage_dir, exist_ok=True)

    def wrap_modules(
        self,
        all_modules_generator: Generator[torch.nn.Module, None, None],
        submodule_accessor: Optional[_SubmoduleAccessor] = None,
        whitelist_param_names_creator: Optional[_WhitelistParamNamesCreator] = None,
    ):
        if self._has_wrapped:
            return list(all_modules_generator)
        self._has_wrapped = True

        modules = []
        for module_index, module in enumerate(all_modules_generator):
            modules.append(module)
            if module_index % self._group_size < self._group_size - self._num_in_group:
                continue
            submodule = submodule_accessor(module) if submodule_accessor else module
            param_names = (
                whitelist_param_names_creator(submodule)
                if whitelist_param_names_creator
                else _default_offload_param_names(submodule)
            )
            if param_names:
                self._records.append(
                    _OffloadedModule(submodule, param_names, self._storage_dir)
                )
                torch.cuda.empty_cache()
        return modules

    def place_on_host(self, param: torch.nn.Parameter) -> bool:
        nbytes = param.numel() * param.element_size()
        backing = _BackingFile(self._storage_dir, _align_up(nbytes))
        self._host_weights.append(backing)
        param.data = backing.mapping[:nbytes].view(param.dtype).view(param.shape)
        torch.cuda.empty_cache()
        return True

    def post_init(self):
        for host_weight in self._host_weights:
            host_weight.evict_page_cache()
        if not self._records:
            return
        for record in self._records:
            record.finalize(self._storage_dir)
        num_host_cached = self._fill_host_cache()
        torch.cuda.empty_cache()

        num_slots = min(self._prefetch_step, len(self._records)) + 1
        slot_nbytes = max(record.nbytes for record in self._records)
        self._streamer = _WeightStreamer(
            device=torch.device("cuda", torch.cuda.current_device()),
            slot_nbytes=slot_nbytes,
            num_slots=num_slots,
        )
        for index, record in enumerate(self._records):
            self._install_hook(index, record)
        for index in range(num_slots - 1):
            self._pending_jobs[index] = self._streamer.submit(self._records[index])

        total_nbytes = sum(record.nbytes for record in self._records)
        logger.info(
            f"[disk offloader] {len(self._records)} modules, "
            f"{total_nbytes / 2**30:.2f} GiB offloaded; {num_host_cached} in pinned RAM, "
            f"{len(self._records) - num_host_cached} streamed from {self._storage_dir}; "
            f"{num_slots} GPU slots of {slot_nbytes / 2**20:.0f} MiB; "
            f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated, "
            f"{torch.cuda.memory_reserved() / 2**30:.2f} GiB reserved, "
            f"{torch.cuda.mem_get_info()[0] / 2**30:.2f} GiB free on GPU"
        )

    def _fill_host_cache(self) -> int:
        remaining_bytes = self._host_cache_bytes
        uncached = list(self._records)
        num_host_cached = 0
        while uncached and uncached[0].nbytes <= remaining_bytes:
            block_nbytes = 1 << (remaining_bytes.bit_length() - 1)
            if uncached[0].nbytes > block_nbytes:
                break
            block = torch.empty(block_nbytes, dtype=torch.uint8, pin_memory=True)
            cursor = 0
            while uncached and cursor + uncached[0].nbytes <= block_nbytes:
                record = uncached.pop(0)
                record.move_to_host_cache(block[cursor : cursor + record.nbytes])
                cursor += record.nbytes
                num_host_cached += 1
            remaining_bytes -= block_nbytes
        return num_host_cached

    def _install_hook(self, index: int, record: _OffloadedModule):
        lookahead = min(self._prefetch_step, len(self._records))

        def acquire_and_prefetch():
            job = self._pending_jobs.pop(index)
            self._active_jobs[index] = job
            next_index = (index + lookahead) % len(self._records)
            self._pending_jobs[next_index] = self._streamer.submit(
                self._records[next_index]
            )
            return record.device_tensors(self._streamer.wait(job))

        def release():
            self._streamer.release(self._active_jobs.pop(index))

        _hook_module_forward_raw(
            record.module,
            on_forward_end=release,
            get_parameter_and_buffer_dicts=acquire_and_prefetch,
        )

    @property
    def forbid_copy_engine_usage(self):
        return True
