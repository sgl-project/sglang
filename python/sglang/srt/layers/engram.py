"""Engram: gated n-gram hash memory added to the hc residual stream.

The token map, hash multipliers and prime layout must match
sglang.kernels.ops.embeddings.engram_hash to produce identical hash ids.
"""

from __future__ import annotations

import ctypes
import enum
import errno
import functools
import glob
import logging
import mmap
import os
import re
import time
from contextlib import nullcontext
from typing import TYPE_CHECKING, Optional

import msgspec
import numpy as np
import torch
from torch import nn

from sglang.kernels.ops.attention.dsv4.torch_quant import FP8_BLOCK_SIZE
from sglang.kernels.ops.embeddings.engram_gate import fused_engram_gate
from sglang.kernels.ops.embeddings.engram_gather import engram_gather
from sglang.kernels.ops.embeddings.engram_hash import (
    MODE_DECODE,
    MODE_EXTEND,
    MODE_VERIFY,
    engram_commit_history,
    engram_hash_ids,
    engram_hash_ids_and_commit,
)
from sglang.srt.distributed import tensor_model_parallel_all_reduce
from sglang.srt.distributed.device_communicators.cuda_wrapper import (
    find_loaded_library,
)
from sglang.srt.distributed.parallel_state import inplace_all_reduce
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    attn_cp_all_gather_into_tensor,
    dp_gather_replicate,
    dp_reduce_scatter_tensor,
    dp_scatter,
    get_global_dp_buffer_len,
    is_dp_gatherv_active,
)
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.schedule_batch import MM_PAD_SHIFT_VALUE
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_model, get_parallel, get_serving
from sglang.srt.utils import (
    add_prefix,
    is_cuda,
    is_gfx95_supported,
    is_hip,
)
from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

if TYPE_CHECKING:
    from sglang.srt.layers.layernorm import RMSNorm

logger = logging.getLogger(__name__)


_MILLER_RABIN_WITNESSES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

_is_hip = is_hip()


def _cuda_kernels(t: torch.Tensor) -> bool:
    """True where the Triton kernels apply (CUDA and gfx950); other ROCm GPUs and CPU
    take the torch paths."""
    return t.is_cuda and (is_cuda() or is_gfx95_supported())


def _is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin; exact for n < 3.3e24 with these witnesses."""
    if n < 2:
        return False
    for p in _MILLER_RABIN_WITNESSES:
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d, r = d // 2, r + 1
    for a in _MILLER_RABIN_WITNESSES:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _find_next_prime(start: int, seen_primes: set[int]) -> int:
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Token id -> compressed id, plus the compressed vocab size. The size feeds every
    hash multiplier, so a mismatch rehashes the whole table."""
    from tokenizers import Regex, normalizers

    # A private-use sentinel keeps a lone-space token from collapsing to "" under Strip().
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            # A partial UTF-8 byte token has nothing to normalize; key it by its raw form.
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def compute_hash_multipliers(
    layer_ids: tuple[int, ...], max_ngram_size: int, vocab_size: int
) -> torch.Tensor:
    """One odd multiplier per (layer, lookback) from a per-layer RNG, bounded so that
    token_id * multiplier fits in int64."""
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(
            low=0, high=multiplier_bound, size=(max_ngram_size,), dtype=np.int64
        )
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


class EngramLayout(msgspec.Struct, frozen=True):
    max_ngram_size: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    primes: tuple[tuple[tuple[int, ...], ...], ...]  # [layer][n-gram size][head]
    n_heads: int
    head_dim: int

    @classmethod
    def from_config(cls, config) -> Optional[EngramLayout]:
        """Primes are drawn in (layer, n-gram size, head) order from one shared
        ascending sequence starting above engram_vocab_size - 1."""
        layer_ids = tuple(config.engram_layer_ids)
        if not layer_ids:
            return None
        max_ngram_size, n_heads = config.engram_max_ngram_size, config.engram_n_heads
        vocab_size = config.engram_vocab_size
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                sizes, current = [], vocab_size - 1
                for _ in range(n_heads):
                    current = _find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        return cls(
            max_ngram_size=max_ngram_size,
            layer_ids=layer_ids,
            num_embeddings=tuple(config.engram_num_embeddings),
            primes=tuple(primes),
            n_heads=n_heads,
            head_dim=config.engram_head_dim,
        )

    def column_bounds(self, hash_index: int) -> tuple[int, ...]:
        """Prefix sums of one layer's column segments, in the hash's (n-gram
        size, head) column order: hash column c owns rows [bounds[c], bounds[c + 1])."""
        bounds = [0]
        for per_ngram in self.primes[hash_index]:
            for prime in per_ngram:
                bounds.append(bounds[-1] + prime)
        return tuple(bounds)


def compute_engram_hash_ids(
    tokens: torch.Tensor,
    blocked: torch.Tensor,
    pad_id: int,
    token_map: torch.Tensor,
    multipliers: torch.Tensor,
    primes: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """tokens [T, n]: column 0 is the token itself, column s its s-th predecessor;
    blocked [T, n] marks look-back that ran off the sequence start. Returns
    [T, n_engram_layers, (n - 1) * n_heads] row ids into each layer's table."""
    compressed = torch.where(blocked, pad_id, token_map[tokens])
    products = compressed.unsqueeze(1) * multipliers
    # After step i the running xor is the (i + 1)-gram hash, bucketed by that
    # n-gram size's primes.
    rolling, hashes = products[..., 0], []
    for i in range(1, tokens.shape[-1]):
        rolling = torch.bitwise_xor(rolling, products[..., i])
        hashes.append(rolling.unsqueeze(-1) % primes[:, i - 1])
    return torch.cat(hashes, dim=-1) + offsets


class EngramHasher(nn.Module):
    """Hash ids for every token of a forward batch, [T, n_engram_layers, n_hash_cols]."""

    def __init__(
        self,
        layout: EngramLayout,
        tokenizer,
        pad_id: int,
        compressed_vocab_size: int,
    ):
        super().__init__()
        self.max_ngram_size = layout.max_ngram_size
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        assert vocab_size == compressed_vocab_size, (
            f"the tokenizer normalizes to {vocab_size} distinct tokens but the config "
            f"expects {compressed_vocab_size}; every hash multiplier depends on it"
        )
        self.pad_id = token_map[pad_id]
        flat = [
            [p for per_ngram in layer for p in per_ngram] for layer in layout.primes
        ]
        offsets = np.array([np.cumsum([0, *sizes[:-1]]) for sizes in flat])
        multipliers = compute_hash_multipliers(
            layout.layer_ids, layout.max_ngram_size, vocab_size
        )
        self.register_buffer("token_map", torch.tensor(token_map), persistent=False)
        self.register_buffer("multipliers", multipliers, persistent=False)
        self.register_buffer("primes", torch.tensor(layout.primes), persistent=False)
        self.register_buffer("offsets", torch.tensor(offsets), persistent=False)
        self.image_token_id: Optional[int] = None
        self.history: Optional[torch.Tensor] = None
        self.pad_row = 0
        # typing only
        self.token_map: torch.Tensor
        self.multipliers: torch.Tensor
        self.primes: torch.Tensor
        self.offsets: torch.Tensor

    def init_history(self, num_req_slots: int, device) -> None:
        """Allocate oldest-first history with a spare row for graph padding."""
        self.history = torch.zeros(
            num_req_slots + 1,
            self.max_ngram_size - 1,
            dtype=torch.int32,
            device=device,
        )
        self.pad_row = num_req_slots

    @classmethod
    def from_config(
        cls, config, layout: EngramLayout, *, image_token_id: Optional[int] = None
    ) -> EngramHasher:
        # The compressed token map is built with the HF normalizers, so the HF
        # tokenizer backend is used here whatever the serving backend is.
        tokenizer = get_tokenizer(
            get_serving().tokenizer_path,
            tokenizer_mode=get_serving().tokenizer_mode,
            trust_remote_code=get_model().trust_remote_code,
            revision=get_model().revision,
            tokenizer_backend="huggingface",
        )
        result = cls(
            layout,
            tokenizer,
            config.engram_pad_token_id,
            config.engram_compressed_vocab_size,
        )
        result.image_token_id = image_token_id
        return result

    def forward(
        self, input_ids: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        assert self.history is not None, "EngramHasher.init_history was not called"
        n = self.max_ngram_size
        num_tokens = input_ids.shape[0]
        if num_tokens == 0:
            return torch.empty(
                (0, self.primes.shape[0], self.offsets.shape[1]),
                dtype=torch.int64,
                device=input_ids.device,
            )
        mode = forward_batch.forward_mode
        req_slots = forward_batch.req_pool_indices
        bs = req_slots.shape[0]
        device = input_ids.device
        # Tokens at or past num_real are graph padding. History comes from
        # self.history via req_slots unless the scheduler supplied this extend's rows.
        num_real, block, row, starts = num_tokens, 1, None, None
        history, hist_via_slots = self.history, True
        if mode.is_decode():
            kmode = MODE_DECODE
            commit_rows, commit_last = req_slots, None
        elif mode.is_target_verify():
            block = int(forward_batch.spec_info.draft_token_num)
            assert num_tokens == bs * block, (
                "engram target-verify expects one equal block per request, got "
                f"{num_tokens} tokens for {bs} requests of {block}"
            )
            kmode = MODE_VERIFY
            commit_rows = commit_last = None
        else:
            assert mode.is_extend(), (
                f"engram serves extend, target-verify and decode, not {mode}"
            )
            lens = forward_batch.extend_seq_lens.to(torch.int64)
            starts = forward_batch.extend_start_loc.to(torch.int64)
            lens_cpu = forward_batch.extend_seq_lens_cpu
            row = torch.repeat_interleave(
                torch.arange(bs, device=device),
                lens,
                # The host total skips the device sum's sync.
                output_size=sum(lens_cpu) if lens_cpu is not None else None,
            )
            num_real = row.shape[0]
            kmode = MODE_EXTEND
            if forward_batch.engram_history is not None:
                history, hist_via_slots = forward_batch.engram_history, False
            commit_rows = torch.where(lens > 0, req_slots, self.pad_row)
            commit_last = (starts + lens - 1).clamp(0, num_tokens - 1)

        if _cuda_kernels(input_ids):
            if kmode == MODE_DECODE:
                # out_cache_loc 0 marks the CUDA-graph padded rows that must not commit.
                assert forward_batch.out_cache_loc is not None
                return engram_hash_ids_and_commit(
                    input_ids,
                    forward_batch.positions,
                    history=self.history,
                    req_slots=req_slots,
                    out_cache_loc=forward_batch.out_cache_loc,
                    token_map=self.token_map,
                    multipliers=self.multipliers,
                    primes=self.primes,
                    offsets=self.offsets,
                    pad_id=self.pad_id,
                    image_token_id=self.image_token_id,
                    mm_pad_shift=MM_PAD_SHIFT_VALUE,
                )
            hash_ids, tokens = engram_hash_ids(
                input_ids,
                forward_batch.positions,
                mode=kmode,
                history=history,
                token_map=self.token_map,
                multipliers=self.multipliers,
                primes=self.primes,
                offsets=self.offsets,
                pad_id=self.pad_id,
                num_real=num_real,
                req_slots=req_slots if hist_via_slots else None,
                block=block,
                row=row,
                starts=starts,
                image_token_id=self.image_token_id,
                mm_pad_shift=MM_PAD_SHIFT_VALUE,
            )
        else:
            hash_ids, tokens = self._torch_hash_ids(
                input_ids,
                forward_batch.positions,
                kmode,
                history[req_slots] if hist_via_slots else history,
                num_real,
                block,
                row,
                starts,
            )
        if commit_rows is not None:
            # Padded rows must not overwrite a live request's history.
            last_tokens = tokens if commit_last is None else tokens[commit_last]
            out_loc = forward_batch.out_cache_loc
            if out_loc is not None:
                if commit_last is not None:
                    out_loc = out_loc[commit_last]
                commit_rows = torch.where(out_loc == 0, self.pad_row, commit_rows)
            self.history[commit_rows] = (
                last_tokens[:, : n - 1].flip(-1).to(self.history.dtype)
            )
        return hash_ids

    def _torch_hash_ids(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kmode: int,
        history: torch.Tensor,
        num_real: int,
        block: int,
        row: Optional[torch.Tensor],
        starts: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Non-CUDA fallback of the hash kernel: predecessor table [T, n] and hash ids.
        ``history`` is already the per-request [bs, n - 1] rows of this batch."""
        n = self.max_ngram_size
        num_tokens = input_ids.shape[0]
        device = input_ids.device
        input_ids = input_ids.to(torch.int64)
        shifts = torch.arange(n, device=device)
        if kmode == MODE_DECODE:
            tokens = torch.cat(
                [input_ids.unsqueeze(-1), history.to(torch.int64).flip(-1)], dim=-1
            )
        else:
            t = torch.arange(num_real, device=device)
            if kmode == MODE_VERIFY:
                row = t // block
                offset = t - row * block
            else:
                offset = t - starts[row]
            # Predecessor at shift s of token t: an earlier token of the same run
            # when s <= offset, else history[row, -(s - offset)].
            from_batch = shifts.unsqueeze(0) <= offset.unsqueeze(-1)
            in_batch = input_ids[(t.unsqueeze(-1) - shifts).clamp_min(0)]
            hist_col = (n - 2 - (shifts.unsqueeze(0) - offset.unsqueeze(-1) - 1)).clamp(
                0, n - 2
            )
            from_hist = history.to(torch.int64)[row].gather(1, hist_col)
            tokens = torch.where(from_batch, in_batch, from_hist)
        if num_real < num_tokens:
            tokens = torch.cat([tokens, tokens.new_zeros(num_tokens - num_real, n)])
        positions = positions.to(torch.int64)
        blocked = positions.unsqueeze(-1) < shifts
        if num_real < num_tokens:
            blocked[num_real:] = True
        if self.image_token_id is not None:
            # Scheduler-provided history still carries the multimodal pad ids.
            tokens = tokens.masked_fill(
                tokens >= MM_PAD_SHIFT_VALUE, self.image_token_id
            )
            # Once a lookback hits an image, every older predecessor is PAD.
            blocked = (
                (blocked | (tokens == self.image_token_id))
                .to(torch.int32)
                .cummax(-1)
                .values.bool()
            )
        hash_ids = compute_engram_hash_ids(
            tokens,
            blocked,
            self.pad_id,
            self.token_map,
            self.multipliers,
            self.primes,
            self.offsets,
        )
        return hash_ids, tokens

    def commit_after_verify(
        self,
        verify_ids_2d: torch.Tensor,
        req_pool_indices: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> None:
        """Commit anchor + accepted drafts; the bonus is the next block's anchor."""
        assert self.history is not None, "EngramHasher.init_history was not called"
        if _cuda_kernels(self.history):
            engram_commit_history(
                self.history, verify_ids_2d, req_pool_indices, commit_lens
            )
            return
        n1 = self.max_ngram_size - 1
        req = req_pool_indices.to(torch.int64)
        window = torch.cat(
            [self.history[req], verify_ids_2d.to(self.history.dtype)], dim=1
        )
        cols = commit_lens.to(torch.int64).unsqueeze(-1) + torch.arange(
            n1, device=window.device
        )
        self.history[req] = window.gather(1, cols)


_THP_DIR = "/sys/kernel/mm/transparent_hugepage"


def _thp_mode(knob: str) -> str:
    """Active mode of a transparent_hugepage sysfs knob ("" if unreadable)."""
    try:
        with open(f"{_THP_DIR}/{knob}") as f:
            m = re.search(r"\[(\w+)\]", f.read())
        return m.group(1) if m else ""
    except OSError:
        return ""


def _huge_pages_backing(addr: int) -> tuple[int, int]:
    """(mapped_kB, huge_kB) of the VMA holding addr, from /proc/self/smaps.
    The only evidence that the kernel really handed out huge pages."""
    mapped = huge = 0
    inside = False
    try:
        with open("/proc/self/smaps") as f:
            for line in f:
                m = re.match(r"^([0-9a-f]+)-([0-9a-f]+) ", line)
                if m:
                    if inside:
                        break
                    inside = int(m.group(1), 16) <= addr < int(m.group(2), 16)
                elif inside:
                    key, _, rest = line.partition(":")
                    if key == "Rss":
                        mapped = int(rest.split()[0])
                    elif key in ("AnonHugePages", "ShmemPmdMapped", "FilePmdMapped"):
                        huge += int(rest.split()[0])
    except OSError:
        pass
    return mapped, huge


_page_cache_dropped = False


def drop_checkpoint_page_cache() -> tuple[int, int]:
    """posix_fadvise(DONTNEED) on the checkpoint files; returns (files, bytes)."""
    try:
        model_path = get_model().model_path
    except (ValueError, AttributeError):
        # No published runtime context (unit tests, offline tools): nothing to drop.
        return 0, 0
    files, nbytes = 0, 0
    for f in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        try:
            fd = os.open(f, os.O_RDONLY)
        except OSError:
            continue
        try:
            nbytes += os.fstat(fd).st_size
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            files += 1
        finally:
            os.close(fd)
    return files, nbytes


def _drop_page_cache_once(reason: str) -> None:
    global _page_cache_dropped
    if _page_cache_dropped:
        return
    _page_cache_dropped = True
    files, nbytes = drop_checkpoint_page_cache()
    logger.info(
        "engram host table: dropped the page cache of %d checkpoint files (%.0f GiB) %s",
        files,
        nbytes / 2**30,
        reason,
    )


@functools.cache
def _hip_runtime() -> ctypes.CDLL:
    """
    torch.cuda.cudart() does not expose hipHostGetDevicePointer, so call it
    through ctypes to map a registered host address to its device address.
    """
    path = find_loaded_library("libamdhip64")
    if path is None:
        raise RuntimeError("libamdhip64 is not loaded in the current process")
    lib = ctypes.CDLL(path)
    lib.hipHostGetDevicePointer.restype = ctypes.c_int
    lib.hipHostGetDevicePointer.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    return lib


def _registered_device_ptr(host_ptr: int) -> int:
    """
    Address kernels must use for registered host memory.
    UVA makes it the host address on CUDA; HIP may map it elsewhere.
    """
    if not _is_hip:
        return host_ptr
    device_ptr = ctypes.c_void_p()
    err = _hip_runtime().hipHostGetDevicePointer(ctypes.byref(device_ptr), host_ptr, 0)
    if err != 0 or not device_ptr.value:
        raise RuntimeError(f"hipHostGetDevicePointer failed: {err}")
    return device_ptr.value


class _HostTable:
    """Host-memory backing for one engram table ('shared' or 'per_rank' layout).

    Lives for the whole process: the mapping, the memfd and the cudaHostRegister
    pin are never released because the table is read by every forward.
    """

    def __init__(self, layout: str, nbytes: int, name: str, group):
        if layout not in ("shared", "per_rank"):
            raise ValueError(
                f"Invalid SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT={layout!r}; expected "
                "'shared' or 'per_rank'"
            )
        self.layout = layout
        self.nbytes = nbytes
        self.group = group
        self.dirty = False
        if layout == "shared":
            self.fd = self._open_shared_fd(nbytes, name)
            self.mm = mmap.mmap(
                self.fd,
                nbytes,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
        else:
            self.fd = None
            self.mm = mmap.mmap(
                -1,
                nbytes,
                flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
        # Advisory before the first touch: pages are allocated huge at fault time.
        self.mm.madvise(mmap.MADV_HUGEPAGE)
        self.bytes = torch.frombuffer(self.mm, dtype=torch.uint8)
        if layout == "per_rank":
            # Cached checkpoint pages, left by a previous server or by the loader,
            # make the 512 MiB huge-page faults fall back, so empty them first.
            _drop_page_cache_once("before pre-faulting the per-rank shard")
            np.frombuffer(self.mm, dtype=np.uint8)[:: mmap.PAGESIZE] = 0
        if layout == "shared":
            # Every rank holds the fd before rank 0 continues; the /proc path only
            # resolves while rank 0 keeps its descriptor.
            group.barrier()
        err = torch.cuda.cudart().cudaHostRegister(self.bytes.data_ptr(), nbytes, 0)
        if int(err) != 0:
            raise RuntimeError(f"cudaHostRegister({nbytes} bytes) failed: {err}")
        self.device_ptr = _registered_device_ptr(self.bytes.data_ptr())

    def _open_shared_fd(self, nbytes: int, name: str) -> int:
        owner = None
        if self.group.rank_in_group == 0:
            fd = os.memfd_create(name, 0)
            os.ftruncate(fd, nbytes)
            owner = (os.getpid(), fd)
        pid, owner_fd = self.group.broadcast_object(owner, src=0)
        if self.group.rank_in_group == 0:
            return fd
        try:
            return os.open(f"/proc/{pid}/fd/{owner_fd}", os.O_RDWR)
        except OSError as e:
            raise RuntimeError(
                "engram host table: cannot open rank 0's memfd through /proc; the "
                "TP ranks must share a PID namespace"
            ) from e

    def _collapse(self, tries: int = 3) -> None:
        """Synchronously fold whatever is still on base pages into huge pages.
        Anonymous memory only; shmem obeys shmem_enabled and refuses."""
        MADV_COLLAPSE = 25  # Linux >= 6.1; not in Python's mmap module
        libc = ctypes.CDLL(None, use_errno=True)
        libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
        for attempt in range(tries):
            rc = libc.madvise(
                ctypes.c_void_p(self.bytes.data_ptr()),
                ctypes.c_size_t(self.nbytes),
                MADV_COLLAPSE,
            )
            if rc == 0:
                return
            err = ctypes.get_errno()
            if (
                err != errno.EAGAIN or attempt == tries - 1
            ):  # EAGAIN is the only one worth retrying
                logger.info("engram host table: MADV_COLLAPSE errno %d", err)
                return
            time.sleep(1.0)

    def finish_load(self, label: str):
        if not self.dirty:
            return
        self.dirty = False
        if self.layout == "shared":
            self.group.barrier()
        mapped_kb, huge_kb = _huge_pages_backing(self.bytes.data_ptr())
        if self.layout == "per_rank" and huge_kb < mapped_kb * 0.98:
            # The loader's own reads refilled the page cache; empty it again so the
            # collapse can find contiguous memory.
            drop_checkpoint_page_cache()
            self._collapse()
            mapped_kb, huge_kb = _huge_pages_backing(self.bytes.data_ptr())
        pct = 100.0 * huge_kb / mapped_kb if mapped_kb else 0.0
        msg = (
            f"engram host table {label}: layout={self.layout}, "
            f"{mapped_kb / 2**10:.0f} MiB resident, {huge_kb / 2**10:.0f} MiB in huge pages "
            f"({pct:.0f}%)"
        )
        if huge_kb == 0:
            knob = "shmem_enabled" if self.layout == "shared" else "enabled"
            logger.warning(
                "%s. No huge pages: expect ~10x slower lookups (one TLB miss per row); "
                "transparent_hugepage/%s is '%s'",
                msg,
                knob,
                _thp_mode(knob) or "unreadable",
            )
        else:
            logger.info(msg)


def shard_range(
    num_embeddings: int,
    column_bounds: tuple[int, ...],
    tp_rank: int,
    tp_size: int,
) -> tuple[int, int]:
    """This rank's [start, end) rows: on the layer's hash-column boundaries when
    the column count divides evenly, so a rank owns whole per-head subtables,
    else the equal-row split. Exactly one rank owns each row either way, so the
    zero-fill + all-reduce lookup is bitwise independent of the choice."""
    assert column_bounds[-1] == num_embeddings, (
        "the config's table size disagrees with its primes"
    )
    if (len(column_bounds) - 1) % tp_size == 0:
        cols_per_rank = (len(column_bounds) - 1) // tp_size
        return (
            column_bounds[tp_rank * cols_per_rank],
            column_bounds[(tp_rank + 1) * cols_per_rank],
        )
    return (
        num_embeddings * tp_rank // tp_size,
        num_embeddings * (tp_rank + 1) // tp_size,
    )


class EngramEmbedding(nn.Module):
    """One layer's fp8 hash table with e8m0 block scales, dequantized on lookup.

    Rows are sharded over the TP group in device memory, aligned to the hash
    columns where they divide evenly (`shard_range`); with
    SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE they live in host memory instead, as
    one shared copy or one shard per rank (see _HostTable). Loading is sharded
    in every layout: a rank writes only its own row range.
    """

    def __init__(
        self,
        num_embeddings: int,
        dim: int,
        layer_id: int,
        column_bounds: tuple[int, ...],
    ):
        super().__init__()
        self.dim = dim
        self.tp_size = get_parallel().tp_size
        tp_rank = get_parallel().tp_rank
        self.row_start, row_end = shard_range(
            num_embeddings, column_bounds, tp_rank, self.tp_size
        )
        self.rows = row_end - self.row_start
        spans = [
            shard_range(num_embeddings, column_bounds, r, self.tp_size)
            for r in range(self.tp_size)
        ]
        self._shard_starts = tuple(start for start, _ in spans)
        self._gather_world = self.tp_size
        self._symmetric = False
        self._gather_ptrs: Optional[tuple[tuple[int, ...], tuple[int, ...]]] = None
        self.host_table: Optional[_HostTable] = None
        if envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.get():
            self._init_host_table(num_embeddings, dim, layer_id)
        else:
            self._init_device_table(dim, spans)
        # The default lookup reads rows in place into wkv's MXFP8 operand; the
        # device shards need whole hash columns per rank, and ids must fit int32.
        self._mxfp8_capable = (
            is_cuda()  # ROCm reports "cuda" too; the gather kernels are nvcc JIT
            and num_embeddings < 2**31
        ) and (
            self._shared
            or (
                self.host_table is None
                and (len(column_bounds) - 1) % self.tp_size == 0
                and (self.tp_size == 1 or self._symmetric)
            )
        )
        self.weight.weight_loader = self._load_rows
        self.scale.weight_loader = self._load_rows

    def _init_device_table(self, dim: int, spans) -> None:
        """The shard, in symmetric memory so peers can gather from it in place.
        The pool wants every rank to allocate the same shapes, and the column
        shards are ragged, so every rank allocates the widest one."""
        from sglang.srt.distributed.symmetric_memory import symmetric_context

        alloc_rows = max(end - start for start, end in spans)
        device = torch.empty(0).device
        self._symmetric = self.tp_size > 1 and device.type == "cuda"
        ctx = symmetric_context(device) if self._symmetric else nullcontext()
        with ctx:
            self._weight_buf = torch.empty(alloc_rows, dim, dtype=torch.float8_e4m3fn)
            self._scale_buf = torch.empty(
                alloc_rows, dim // FP8_BLOCK_SIZE, dtype=torch.float8_e8m0fnu
            )
        self.weight = nn.Parameter(self._weight_buf[: self.rows], requires_grad=False)
        self.scale = nn.Parameter(self._scale_buf[: self.rows], requires_grad=False)

    def _init_host_table(self, num_embeddings: int, dim: int, layer_id: int):
        layout = envs.SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT.get()
        n = num_embeddings if layout == "shared" else self.rows
        w_bytes = n * dim
        s_bytes = n * (dim // FP8_BLOCK_SIZE)
        self.host_table = _HostTable(
            layout,
            max(1, w_bytes + s_bytes),  # mmap requires storage even for an empty shard.
            f"sglang_engram_{layer_id}",
            get_parallel().tp_group,
        )
        raw = self.host_table.bytes[: w_bytes + s_bytes]
        weight = raw[:w_bytes].view(torch.float8_e4m3fn).view(n, dim)
        scale = raw[w_bytes:].view(torch.float8_e8m0fnu).view(n, dim // FP8_BLOCK_SIZE)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.scale = nn.Parameter(scale, requires_grad=False)
        device_ptr = self.host_table.device_ptr
        self._host_table_ptrs = (device_ptr, device_ptr + w_bytes)

    @property
    def _shared(self) -> bool:
        return self.host_table is not None and self.host_table.layout == "shared"

    def _table_ptrs(self) -> tuple[int, int]:
        if self.host_table is None:
            return self.weight.data_ptr(), self.scale.data_ptr()
        return self._host_table_ptrs

    def _load_rows(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        rows = slice(self.row_start, self.row_start + self.rows)
        if self._shared:
            param.data[rows].copy_(loaded_weight[rows])
        else:
            param.data.copy_(loaded_weight[rows])
        if self.host_table is not None:
            self.host_table.dirty = True

    def finish_load(self, label: str):
        """Collective: every rank calls it after loading. The device table
        exchanges its shard addresses here, before any CUDA graph capture."""
        if self.host_table is not None:
            self.host_table.finish_load(label)
            if self._mxfp8_capable and self._gather_ptrs is None:
                # The shared copy is one whole table: gather as a world of one.
                w_ptr, s_ptr = self._host_table_ptrs
                self._gather_world = 1
                self._shard_starts = (0,)
                self._gather_ptrs = ((w_ptr,), (s_ptr,))
        elif self._mxfp8_capable and self._gather_ptrs is None:
            if self.tp_size == 1:
                self._gather_ptrs = (
                    (self.weight.data_ptr(),),
                    (self.scale.data_ptr(),),
                )
            else:
                import torch.distributed._symmetric_memory as symm_mem

                group = get_parallel().tp_group.device_group
                wh = symm_mem.rendezvous(self._weight_buf, group)
                sh = symm_mem.rendezvous(self._scale_buf, group)
                self._gather_ptrs = (tuple(wh.buffer_ptrs), tuple(sh.buffer_ptrs))

    @property
    def mxfp8_gather_ready(self) -> bool:
        return self._gather_ptrs is not None

    def gather_mxfp8(self, indices: torch.Tensor):
        """The rows as wkv's MXFP8 A operand ``(data [M, cols * dim] fp8,
        swizzled e8m0 scales)``, bytes copied verbatim; no collective, so each
        rank may pass its own ids."""
        from sglang.kernels.ops.embeddings.engram_fusion import (
            engram_gather_mxfp8,
            sf_bytes,
        )
        from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput

        assert self._gather_ptrs is not None
        m, cols = indices.shape
        m_pad = (m + 127) // 128 * 128
        out_a = indices.new_empty(m_pad, cols * self.dim, dtype=torch.uint8)
        out_sf = out_a.new_empty(sf_bytes(m_pad))
        # TODO(perf): emit int32 straight from the hasher and drop this cast.
        engram_gather_mxfp8(
            self._gather_world,
            indices.to(torch.int32),
            out_a,
            out_sf,
            self._gather_ptrs[0],
            self._gather_ptrs[1],
            self._shard_starts,
        )
        return Mxfp8SwizzledInput(out_a[:m].view(torch.float8_e4m3fn), out_sf)

    def gather_bf16(self, indices: torch.Tensor) -> torch.Tensor:
        """The rows dequantized to [M, cols * dim] bf16, exact (e8m0 scales are
        powers of two); for a wkv without an MXFP8 view, same no-collective
        contract as ``gather_mxfp8``."""
        from sglang.kernels.ops.embeddings.engram_fusion import engram_gather_bf16

        assert self._gather_ptrs is not None
        m, cols = indices.shape
        out = indices.new_empty(m, cols * self.dim, dtype=torch.bfloat16)
        # TODO(perf): emit int32 straight from the hasher and drop this cast.
        engram_gather_bf16(
            self._gather_world,
            indices.to(torch.int32),
            out,
            self._gather_ptrs[0],
            self._gather_ptrs[1],
            self._shard_starts,
        )
        return out

    def forward(
        self,
        indices: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        *,
        cp_all_tokens: bool = False,
    ) -> torch.Tensor:
        if self._shared:
            if indices.shape[0] == 0:
                return self._empty(indices)
            out = self._empty(indices)
            engram_gather(
                *self._table_ptrs(),
                indices.reshape(-1),
                out.view(-1, self.dim),
                self.dim,
                FP8_BLOCK_SIZE,
            )
            return out
        if cp_all_tokens and self.tp_size > 1:
            # Prefill CP: gather the hash ids over the CP group first so every
            # TP rank looks up the same indices, then keep this rank's slice.
            parallel = get_parallel()
            local_rows = indices.shape[0]
            all_indices = indices.new_empty(
                (parallel.attn_cp_size * local_rows, *indices.shape[1:])
            )
            attn_cp_all_gather_into_tensor(all_indices, indices.contiguous())
            start = parallel.attn_cp_rank * local_rows
            return self._lookup(all_indices)[start : start + local_rows]
        if self.tp_size > 1 and get_parallel().attn_dp_size > 1:
            return self._dp_sharded_lookup(indices, forward_batch)
        return self._lookup(indices)

    def _lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """Lookup when every TP rank holds the same indices: the device and
        per-rank host shards zero unowned rows and the all-reduce reassembles."""
        if indices.shape[0] == 0:
            return self._empty(indices)
        values = self._owned_rows(indices)
        if self.tp_size > 1:
            values = self._reduce_owned_rows(values)
        return values

    def _reduce_owned_rows(self, values: torch.Tensor) -> torch.Tensor:
        if _is_hip and values.is_cuda:
            # Integer addition preserves all BF16 bits because exactly one shard owns each row.
            inplace_all_reduce(
                values.view(torch.int32), group_name=get_parallel().tp_group.unique_name
            )
            return values
        return tensor_model_parallel_all_reduce(values)

    def _empty(self, indices: torch.Tensor) -> torch.Tensor:
        return torch.empty(
            *indices.shape, self.dim, dtype=torch.bfloat16, device=indices.device
        )

    def _owned_rows(self, indices: torch.Tensor) -> torch.Tensor:
        """Rows of `indices` this rank's shard holds, zero for the rest."""
        if self.rows == 0:
            return self._empty(indices).zero_()
        if self.host_table is None and not _cuda_kernels(indices):
            local = indices - self.row_start
            owned = (local >= 0) & (local < self.rows)
            local = local.masked_fill(~owned, 0)
            rows = self.weight[local].float().unflatten(-1, (-1, FP8_BLOCK_SIZE))
            values = (rows * self.scale[local].float().unsqueeze(-1)).flatten(-2)
            return values.to(torch.bfloat16).masked_fill(~owned.unsqueeze(-1), 0)
        out = self._empty(indices)
        engram_gather(
            *self._table_ptrs(),
            indices.reshape(-1),
            out.view(-1, self.dim),
            self.dim,
            FP8_BLOCK_SIZE,
            row_lo=self.row_start,
            row_hi=self.row_start + self.rows,
        )
        return out

    def _dp_sharded_lookup(
        self, indices: torch.Tensor, forward_batch: Optional[ForwardBatch]
    ) -> torch.Tensor:
        """Gather DP ranks' indices before looking up TP-sharded rows."""
        assert forward_batch is not None, "the DP engram lookup needs the batch"
        rows = get_global_dp_buffer_len()
        ids_global = torch.empty(
            (rows, *indices.shape[1:]), dtype=indices.dtype, device=indices.device
        )
        # The MAX_LEN gather may zero its local input in place, hence the clone.
        dp_gather_replicate(ids_global, indices.clone(), forward_batch)
        if rows == 0:
            return self._empty(indices)
        values = self._owned_rows(ids_global).view(rows, -1)
        local = torch.empty(
            (indices.shape[0], values.shape[1]),
            dtype=values.dtype,
            device=values.device,
        )
        padding = forward_batch.dp_padding_mode
        if (
            padding is not None
            and padding.is_max_len()
            and self.tp_size == get_parallel().attn_dp_size
            and rows == self.tp_size * local.shape[0]
        ) or is_dp_gatherv_active():
            if _is_hip and values.is_cuda:
                dp_reduce_scatter_tensor(
                    local.view(torch.int32), values.view(torch.int32)
                )
            else:
                dp_reduce_scatter_tensor(local, values)
        else:
            dp_scatter(local, self._reduce_owned_rows(values), forward_batch)
        return local.view(*indices.shape, self.dim)


def engram_gate(
    x: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    clamp_value: float,
) -> torch.Tensor:
    """x [T, hc_mult, dim]; kv [T, (hc_mult + 1) * dim] holds one key per hc copy
    followed by the shared value. Adds the gated value to every copy."""
    if (
        _cuda_kernels(x)
        and x.ndim == 3
        and kv.shape == (x.shape[0], (x.shape[1] + 1) * x.shape[2])
        and x.dtype == kv.dtype
        and x.dtype in (torch.bfloat16, torch.float32)
        and q_weight.dtype in (torch.bfloat16, torch.float32)
        and k_weight.dtype in (torch.bfloat16, torch.float32)
        and all(t.is_contiguous() for t in (x, kv, q_weight, k_weight))
    ):
        return fused_engram_gate(x, kv, q_weight, k_weight, eps, clamp_value)
    hc_mult, dim = x.shape[-2:]
    key, value = kv.split([hc_mult * dim, dim], dim=-1)
    key = key.float().unflatten(-1, (hc_mult, dim))
    weight = q_weight.float() * k_weight.float()
    h = x.float()
    # Normalized per (token, hc copy) over dim, not jointly over the copies.
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(
        key.square().mean(-1) + eps
    )
    dot = (h * weight * key).sum(-1) * rstd * dim**-0.5
    # Signed square root before the sigmoid, matching the training kernel.
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot))
    return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


class EngramFusionMode(enum.IntEnum):
    SP = enum.auto()  # sharded rows in, the kernel all-gathers x
    DP = enum.auto()  # purely local rows: tp 1, or each DP-attention rank's own tokens
    TP = enum.auto()  # replicated rows, the gate shares exchanged over staging


class EngramFusionPlan(msgspec.Struct, frozen=True):
    """Which seam kernel ``fused_seam`` runs; made by ``get_fusion_plan``."""

    mode: EngramFusionMode
    sp_total_rows: int = -1


class Engram(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int,
        layout: EngramLayout,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        next_norm: Optional[RMSNorm] = None,
    ) -> None:
        super().__init__()
        self.layer_hash_index = layout.layer_ids.index(layer_id)
        self.eps = config.rms_norm_eps
        self.clamp_value = 1e-6
        # The norm of the sublayer this engram feeds; the fused seam folds it in.
        self.next_norm = next_norm
        # Image tokens keep their residual (see forward's ``ids``); None when the
        # checkpoint has no vision tower.
        self.image_token_id: Optional[int] = (
            config.image_token_id
            if config.model_type == "deepseek_v41" and config.vision_n_layers > 0
            else None
        )
        dim, hc_mult = config.hidden_size, config.hc_mult
        # The seam kernel hardcodes the DSV4.1 shape; nvcc JIT, so CUDA only.
        self._seam_dims_ok = is_cuda() and hc_mult == 4 and dim == 5120
        self.embed = EngramEmbedding(
            layout.num_embeddings[self.layer_hash_index],
            layout.head_dim,
            layer_id,
            column_bounds=layout.column_bounds(self.layer_hash_index),
        )
        n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        self.wkv = ReplicatedLinear(
            n_hash_cols * layout.head_dim,
            dim * (hc_mult + 1),
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wkv", prefix),
        )
        self.q_weight = nn.Parameter(torch.ones(hc_mult, dim), requires_grad=False)
        self.k_weight = nn.Parameter(torch.ones(hc_mult, dim), requires_grad=False)
        # The seam kernel wants bf16 gate weights (the checkpoint's dtype);
        # anything else turns the fused path off rather than failing in-kernel.
        self._seam_dims_ok = (
            self._seam_dims_ok and self.q_weight.dtype == torch.bfloat16
        )

    def forward(
        self,
        x: torch.Tensor,
        hash_ids: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        *,
        ids: Optional[torch.Tensor] = None,
        cp_all_tokens: bool = False,
    ) -> torch.Tensor:
        """x [T, hc_mult, dim]; hash_ids [T, n_hash_cols] for this layer. ``ids``
        [T] are x's token ids: when given, image tokens keep their residual."""
        if self._use_direct_gather:
            # The direct gather is collective-free, so an empty rank just skips.
            if x.shape[0] == 0:
                return x
            kv, _ = self.wkv(self._gather_wkv_input(hash_ids))
        else:
            # The lookup runs first even for an idle DP-attention batch: under DP
            # attention it is a collective every rank has to join.
            emb = self.embed(hash_ids, forward_batch, cp_all_tokens=cp_all_tokens)
            if x.shape[0] == 0:
                return x
            kv, _ = self.wkv(emb.flatten(-2))
        out = engram_gate(
            x, kv, self.q_weight, self.k_weight, self.eps, self.clamp_value
        )
        if ids is not None:
            assert self.image_token_id is not None
            out = torch.where((ids == self.image_token_id)[:, None, None], x, out)
        return out

    @functools.cached_property
    def _use_direct_gather(self) -> bool:
        """The in-place table gather (either output format): collective-free,
        so each rank passes its own ids."""
        return self.embed.mxfp8_gather_ready

    @functools.cached_property
    def _use_mxfp8_gather(self) -> bool:
        """Resolved on first forward (the wkv backend settles after weight
        processing): table shards gathered straight into wkv's MXFP8 operand."""
        from sglang.srt.layers.quantization.fp8_utils import Mxfp8DenseGemmBackend

        method = self.wkv.quant_method
        return (
            self.embed.mxfp8_gather_ready
            and getattr(method, "mxfp8_dense_backend", None)
            in (
                Mxfp8DenseGemmBackend.FLASHINFER_CUTEDSL,
                Mxfp8DenseGemmBackend.FLASHINFER_CUTLASS,
            )
            and (
                getattr(method, "use_mxfp8", False)
                or getattr(self.wkv, "block_fp8_mxfp8_ready", False)
            )
        )

    @functools.cached_property
    def _mhc_norm_fusable(self) -> bool:
        norm = self.next_norm
        assert norm is not None
        return (
            not norm.cast_x_before_out_mul
            and norm.variance_size_override is None
            and norm.weight.dtype == torch.bfloat16
            and norm.weight.is_contiguous()
        )

    def _gather_wkv_input(self, indices: torch.Tensor):
        if self._use_mxfp8_gather:
            return self.embed.gather_mxfp8(indices)
        return self.embed.gather_bf16(indices)

    def get_fusion_plan(
        self,
        x: torch.Tensor,
        pre: Optional[torch.Tensor],
        *,
        sp_rows: Optional[int],
        cp_all_tokens: bool,
    ) -> Optional[EngramFusionPlan]:
        """How ``fused_seam`` replaces this forward's gate + next combine, None
        for the unfused path; ``pre`` is the state's lagged coefficients."""
        if _is_hip:
            return None
        if sp_rows is not None:
            assert self._use_direct_gather or self.embed._shared
            assert self._mhc_norm_fusable and not cp_all_tokens
            return EngramFusionPlan(EngramFusionMode.SP, sp_total_rows=sp_rows)
        if pre is None or not self._seam_dims_ok or not self._mhc_norm_fusable:
            return None
        if cp_all_tokens:
            return None
        parallel = get_parallel()
        if parallel.tp_size == 1 or parallel.attn_dp_size > 1:
            # DP-attention rows are the rank's own at any attn-TP split; the
            # TP exchange below would mix gates of different tokens.
            return EngramFusionPlan(EngramFusionMode.DP)
        if x.shape[0] < envs.SGLANG_OPT_DSV41_ENGRAM_TP_FUSION_MIN_TOKENS.get():
            return None
        from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
            CustomAllReduceV2,
        )

        if not isinstance(parallel.tp_group.ca_comm, CustomAllReduceV2):
            return None
        return EngramFusionPlan(EngramFusionMode.TP)

    def fused_seam(
        self,
        x: torch.Tensor,
        hash_ids: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        *,
        plan: EngramFusionPlan,
        pre: torch.Tensor,
        ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The seam in one launch: ``x`` [M, 4, 5120] gets ``gate * value`` in
        place and the next sublayer's normalized input comes back (all rows).
        ``ids`` makes image tokens keep their residual; under mHC SP the plan
        carries the global row count and ``x`` holds this rank's rows."""
        import torch.distributed._symmetric_memory as symm_mem

        from sglang.kernels.ops.embeddings.engram_fusion import (
            dp_engram_gate_mhc_combine_norm,
            sp_engram_gate_mhc_combine_norm,
            tp_engram_gate_mhc_combine_norm,
            tp_engram_staging_bytes,
        )
        from sglang.srt.distributed.symmetric_memory import symmetric_context
        from sglang.srt.layers.dsv41_mhc_sp import row_split

        norm = self.next_norm
        assert norm is not None
        q_weight, k_weight = self.q_weight, self.k_weight
        parallel = get_parallel()
        world = parallel.tp_size
        num_rows = x.shape[0]
        if num_rows == 0:
            # An idle DP-attention rank launches nothing, but the bf16 lookup
            # is a collective the rank still owes.
            assert plan.mode == EngramFusionMode.DP
            if not self._use_direct_gather:
                self.embed(hash_ids, forward_batch)
            return x, x.new_empty(0, x.shape[-1])
        tp_slice = slice(None)
        if plan.mode == EngramFusionMode.TP:
            start, length = row_split(num_rows, parallel.tp_rank, world)
            tp_slice = slice(start, start + length)

        if self._use_direct_gather:
            emb = self._gather_wkv_input(hash_ids[tp_slice])
        else:
            emb = self.embed(hash_ids, forward_batch)[tp_slice].flatten(-2)
        skip = None
        if ids is not None:
            assert self.image_token_id is not None
            # view, not to(): bool is 1-byte storage, so this is a free reinterpret.
            skip = (ids == self.image_token_id).view(torch.uint8)
        norm_kwargs = {
            "skip": skip,
            "eps": self.eps,
            "clamp": self.clamp_value,
            "rms_eps": norm.variance_epsilon,
        }
        if plan.mode == EngramFusionMode.SP:
            # mHC SP: sharded rows in, the gathered next input out.
            sp_total_rows = plan.sp_total_rows
            kv, _ = self.wkv(emb)
            with symmetric_context(x.device):
                x_out = x.new_empty(sp_total_rows, x.shape[-1])
            handle = symm_mem.rendezvous(x_out, parallel.tp_group.device_group)
            row_offset, _ = row_split(sp_total_rows, parallel.tp_rank, world)
            sp_engram_gate_mhc_combine_norm(
                world,
                parallel.tp_group.ca_comm.obj,
                handle.buffer_ptrs,
                x,
                kv,
                q_weight,
                k_weight,
                pre,
                norm.weight,
                row_offset=row_offset,
                total_rows=sp_total_rows,
                **norm_kwargs,
            )
            return x, x_out
        elif plan.mode == EngramFusionMode.DP:
            x_out = torch.empty(num_rows, x.shape[-1], dtype=x.dtype, device=x.device)
            kv, _ = self.wkv(emb)
            dp_engram_gate_mhc_combine_norm(
                world,
                x,
                kv,
                q_weight,
                k_weight,
                pre,
                norm.weight,
                x_out,
                **norm_kwargs,
            )
            return x, x_out
        else:
            # The kernel exchanges (value, gates), so wkv -- and the gather
            # feeding it -- ran on this rank's token share only.
            kv, _ = self.wkv(emb)
            with symmetric_context(x.device):
                staging = torch.empty(
                    tp_engram_staging_bytes(num_rows),
                    dtype=torch.uint8,
                    device=x.device,
                )
            handle = symm_mem.rendezvous(staging, parallel.tp_group.device_group)
            x_out = torch.empty(num_rows, x.shape[-1], dtype=x.dtype, device=x.device)
            tp_engram_gate_mhc_combine_norm(
                world,
                parallel.tp_group.ca_comm.obj,
                handle.buffer_ptrs,
                staging,
                x,
                kv,
                q_weight,
                k_weight,
                pre,
                norm.weight,
                x_out,
                **norm_kwargs,
            )
            return x, x_out
