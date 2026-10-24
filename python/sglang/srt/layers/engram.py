"""Engram: gated n-gram hash memory added to the hc residual stream.

The token map, hash multipliers and prime layout must match
sglang.kernels.ops.embeddings.engram_hash to produce identical hash ids.
"""

from __future__ import annotations

import ctypes
import errno
import functools
import glob
import logging
import mmap
import os
import re
import time
from typing import Optional

import msgspec
import numpy as np
import torch
from torch import nn

from sglang.kernels.ops.embeddings.engram_gate import fused_engram_gate
from sglang.kernels.ops.embeddings.engram_gather import (
    engram_gather,
    engram_gather_peer,
)
from sglang.kernels.ops.embeddings.engram_hash import (
    MODE_DECODE,
    MODE_EXTEND,
    MODE_VERIFY,
    engram_commit_history,
    engram_hash_extend_and_commit,
    engram_hash_ids,
    engram_hash_ids_and_commit,
)
from sglang.kernels.ops.quantization.fp4_fake_quant import FP8_BLOCK_SIZE
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
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    get_tc_piecewise_forward_context,
)
from sglang.srt.runtime_context import get_model, get_parallel, get_serving
from sglang.srt.utils import add_prefix, is_cuda, is_gfx95_supported, is_hip
from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

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
        # Host copies of the hash buffers, for windows built on the CPU.
        self._cpu_hash_params: Optional[tuple[torch.Tensor, ...]] = None
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

    def history_row(self, ids: list[int]) -> list[int]:
        """Left-pad up to n - 1 predecessors into a history row. The hash reads every
        id at or above MM_PAD_SHIFT_VALUE as an image token, so multimodal pad ids
        clamp to it and fit the int32 history."""
        n1 = self.max_ngram_size - 1
        return [0] * (n1 - len(ids)) + [min(t, MM_PAD_SHIFT_VALUE) for t in ids]

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
        mode = forward_batch.forward_mode
        if num_tokens == 0 or mode.is_idle():
            # A DP-attention idle batch carries pad rows; nothing reads its hashes.
            return torch.zeros(
                (num_tokens, self.primes.shape[0], self.offsets.shape[1]),
                dtype=torch.int64,
                device=input_ids.device,
            )
        if (
            mode.is_extend()
            and not mode.is_target_verify()
            and envs.SGLANG_OPT_ENGRAM_EXTEND_HASH.get()
            and _cuda_kernels(input_ids)
            and forward_batch.extend_seq_lens_cpu is not None
        ):
            history = forward_batch.engram_history
            return engram_hash_extend_and_commit(
                input_ids,
                forward_batch.positions,
                history=self.history if history is None else history,
                commit_history=self.history,
                req_slots=forward_batch.req_pool_indices,
                starts=forward_batch.extend_start_loc,
                lengths=forward_batch.extend_seq_lens,
                num_real=sum(forward_batch.extend_seq_lens_cpu),
                history_via_slots=history is None,
                out_cache_loc=forward_batch.out_cache_loc,
                token_map=self.token_map,
                multipliers=self.multipliers,
                primes=self.primes,
                offsets=self.offsets,
                pad_id=self.pad_id,
                image_token_id=self.image_token_id,
                mm_pad_shift=MM_PAD_SHIFT_VALUE,
            )
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
        ``history`` is already the per-request [bs, n - 1] rows of this batch.
        EngramRowStager builds the same windows on the host from the requests'
        fill ids."""
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
        return self.hash_windows(tokens, blocked)

    def hash_windows(
        self, tokens: torch.Tensor, blocked: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hash ids [T, n_engram_layers, n_hash_cols] of predecessor windows on the
        device of ``tokens``: tokens [T, n] holds the s-th predecessor in column s,
        blocked [T, n] the look-back past the sequence start. Also returns tokens
        with multimodal pad ids read as the image token."""
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
        params = (self.token_map, self.multipliers, self.primes, self.offsets)
        if tokens.device != self.token_map.device:
            if self._cpu_hash_params is None:
                self._cpu_hash_params = tuple(t.cpu() for t in params)
            params = self._cpu_hash_params
        return compute_engram_hash_ids(tokens, blocked, self.pad_id, *params), tokens

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


@functools.cache
def _sorted_host_reads(device: int) -> bool:
    """Whether host-table rows are read in address order. Without ATS (no
    NVLink-C2C) the GPU maps pinned host memory in its own page tables, and
    random rows of a table far larger than its translation reach miss on nearly
    every row; with ATS the host page tables serve the reads and order costs."""
    if _is_hip:
        return False
    from cuda.bindings import driver

    attr = driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES
    err, uses_host_page_tables = driver.cuDeviceGetAttribute(attr, device)
    return err == driver.CUresult.CUDA_SUCCESS and uses_host_page_tables == 0


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


@functools.cache
def _warn_host_table_ignored() -> None:
    logger.warning(
        "SGLANG_OPT_DSV41_ENGRAM_PEER_SHARDS takes precedence: the engram tables "
        "stay in device memory and SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE is ignored"
    )


class _PeerShards:
    """One rank's engram row shard in exportable device memory (CUDA VMM), with
    every TP peer's shard mapped into this process after `connect`.

    Inside one NVLink domain (MNNVL included, through FABRIC handles) a kernel
    reads any peer's rows in place, so a rank looks up only its own tokens: no
    id gather, no [DP x tokens] buffer, no reduce-scatter. The shard holds the
    fp8 rows followed by their e8m0 scales. Lives for the whole process.
    """

    def __init__(self, rows: int, dim: int, group):
        from sglang.srt.utils.cuda_vmm_utils import (
            VmmReservation,
            align_up,
            get_device_granularity,
            make_device_allocation_prop,
        )

        self.rows = rows
        self.dim = dim
        self.group = group
        self.device_id = torch.cuda.current_device()
        self.w_bytes = rows * dim
        self.s_bytes = rows * (dim // FP8_BLOCK_SIZE)
        gran = get_device_granularity(self.device_id)
        self.size = align_up(max(1, self.w_bytes + self.s_bytes), gran)
        self.local = VmmReservation(
            self.size,
            make_device_allocation_prop(self.device_id),
            self.device_id,
            alignment=gran,
        )
        self.handle = self.local.map(0, self.size, retain_handle=True)
        self.base = self.local.base
        self.w_bias: Optional[torch.Tensor] = None
        self.s_bias: Optional[torch.Tensor] = None
        self.starts: Optional[torch.Tensor] = None

    def views(self) -> tuple[torch.Tensor, torch.Tensor]:
        from sglang.srt.utils.cuda_vmm_utils import tensor_from_pointer

        raw = tensor_from_pointer(self.base, self.size, device_id=self.device_id)
        weight = raw[: self.w_bytes].view(torch.float8_e4m3fn).view(self.rows, self.dim)
        scale = (
            raw[self.w_bytes : self.w_bytes + self.s_bytes]
            .view(torch.float8_e8m0fnu)
            .view(self.rows, self.dim // FP8_BLOCK_SIZE)
        )
        return weight, scale

    @property
    def ready(self) -> bool:
        return self.starts is not None

    def connect(self, row_start: int, label: str) -> None:
        """Collective over the TP group: export this shard, map every peer's."""
        if self.ready:
            return
        import torch.distributed as dist

        from sglang.srt.utils.cuda_vmm_utils import (
            exchange_shareable_handles,
            import_and_map_alloc,
        )

        t0 = time.perf_counter()
        torch.cuda.synchronize()
        cpu_group = self.group.cpu_group
        rank, world = self.group.rank_in_group, self.group.world_size
        use_fabric, peer_handles = exchange_shareable_handles(
            [self.handle], cpu_group, rank, world
        )
        meta = [None] * world
        dist.all_gather_object(meta, (row_start, self.rows, self.size), group=cpu_group)
        bases = []
        try:
            for r, (_, _, size) in enumerate(meta):
                if r == rank:
                    bases.append(self.base)
                    continue
                handle = peer_handles[(r, 0)]
                bases.append(
                    import_and_map_alloc(
                        handle if use_fabric else None,
                        None if use_fabric else handle,
                        size,
                        self.device_id,
                        use_fabric=use_fabric,
                        peer_rank=r,
                    )
                )
        finally:
            if not use_fabric:
                for fd in peer_handles.values():
                    try:
                        os.close(fd)
                    except OSError:
                        pass
        padded = 1 << max(0, (world - 1).bit_length())
        int64_max = torch.iinfo(torch.int64).max
        starts, w_bias, s_bias = [], [], []
        n_scales = self.dim // FP8_BLOCK_SIZE
        for r in range(padded):
            if r < world:
                start, rows = meta[r][0], meta[r][1]
                starts.append(start)
                w_bias.append(bases[r] - start * self.dim)
                s_bias.append(bases[r] + rows * self.dim - start * n_scales)
            else:
                starts.append(int64_max)
                w_bias.append(0)
                s_bias.append(0)
        assert starts[0] == 0 and starts[:world] == sorted(starts[:world]), starts
        device = torch.device("cuda", self.device_id)
        self.w_bias = torch.tensor(w_bias, dtype=torch.int64, device=device)
        self.s_bias = torch.tensor(s_bias, dtype=torch.int64, device=device)
        self.starts = torch.tensor(starts, dtype=torch.int64, device=device)
        torch.cuda.synchronize()
        dist.barrier(group=cpu_group)
        logger.info(
            "engram peer shards %s: %d ranks over %s, %.2f GiB local, mapped in %.1f s",
            label,
            world,
            "FABRIC" if use_fabric else "POSIX fd",
            self.size / 2**30,
            time.perf_counter() - t0,
        )

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        flat = indices.reshape(-1, indices.shape[-1])
        if flat.stride(-1) != 1:
            flat = flat.contiguous()
        out = torch.empty(
            *flat.shape, self.dim, dtype=torch.bfloat16, device=indices.device
        )
        engram_gather_peer(
            flat,
            out,
            self.w_bias,
            self.s_bias,
            self.starts,
            self.dim,
            FP8_BLOCK_SIZE,
        )
        return out.view(*indices.shape, self.dim)


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
        self.host_table: Optional[_HostTable] = None
        self._sort_host_reads = False
        self.staged_by_scheduler = False
        self.peer: Optional[_PeerShards] = None
        if (
            envs.SGLANG_OPT_DSV41_ENGRAM_PEER_SHARDS.get()
            and self.tp_size > 1
            and is_cuda()
        ):
            if envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.get() and tp_rank == 0:
                _warn_host_table_ignored()
            self.peer = _PeerShards(self.rows, dim, get_parallel().tp_group)
            weight, scale = self.peer.views()
            self.weight = nn.Parameter(weight, requires_grad=False)
            self.scale = nn.Parameter(scale, requires_grad=False)
        elif envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.get():
            self._init_host_table(num_embeddings, dim, layer_id)
        else:
            self.weight = nn.Parameter(
                torch.empty(self.rows, dim, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.scale = nn.Parameter(
                torch.empty(
                    self.rows, dim // FP8_BLOCK_SIZE, dtype=torch.float8_e8m0fnu
                ),
                requires_grad=False,
            )
        self.weight.weight_loader = self._load_rows
        self.scale.weight_loader = self._load_rows

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
        self._sort_host_reads = _sorted_host_reads(torch.cuda.current_device())

    @property
    def _shared(self) -> bool:
        return self.host_table is not None and self.host_table.layout == "shared"

    @property
    def shared_host_table_without_ats(self) -> bool:
        """The shared host layout (every row mapped here) on a GPU without ATS."""
        return self._shared and self._sort_host_reads

    def enable_scheduler_staging(self) -> None:
        """Read the rows the scheduler stages (EngramRowStager) instead of
        gathering them in the forward."""
        assert self.shared_host_table_without_ats
        self.staged_by_scheduler = True

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
        """Collective in the shared layout and with peer shards: every rank calls
        it after loading."""
        if self.peer is not None:
            self.peer.connect(self.row_start, label)
        if self.host_table is not None:
            self.host_table.finish_load(label)

    def forward(
        self,
        indices: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        *,
        cp_all_tokens: bool = False,
    ) -> torch.Tensor:
        if self.peer is not None:
            assert self.peer.ready, "engram peer shards are mapped in finish_load"
            # Every shard is mapped here: read this rank's own tokens only, which
            # also covers the prefill CP and DP paths below without a collective.
            return self.peer.lookup(indices)
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
                sort_ids=self._sort_host_reads,
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

    def dequant_staged(
        self,
        rows: torch.Tensor,
        scales: torch.Tensor,
        copied: torch.cuda.Event,
        hash_shape: torch.Size,
    ) -> torch.Tensor:
        """Dequantize this layer's staged rows into [*hash_shape, dim]; rows past the
        staged ones (prefill graph padding) are zero."""
        stream = torch.cuda.current_stream()
        stream.wait_event(copied)
        rows.record_stream(stream)
        scales.record_stream(stream)
        out = torch.empty(
            *hash_shape, self.dim, dtype=torch.bfloat16, device=rows.device
        )
        flat = out.view(-1, self.dim)
        assert rows.shape[0] <= flat.shape[0], (rows.shape, flat.shape)
        engram_gather(
            rows.data_ptr(),
            scales.data_ptr(),
            torch.arange(flat.shape[0], dtype=torch.int32, device=rows.device),
            flat,
            self.dim,
            FP8_BLOCK_SIZE,
            row_hi=rows.shape[0],
        )
        return out

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
            sort_ids=self._sort_host_reads,
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
    input_ids: Optional[torch.Tensor] = None,
    image_token_id: Optional[int] = None,
) -> torch.Tensor:
    """x [T, hc_mult, dim]; kv [T, (hc_mult + 1) * dim] holds one key per hc copy
    followed by the shared value. Adds the gated value to every copy, except in rows
    whose ``input_ids`` entry is ``image_token_id``, which keep ``x``."""
    # ``kv`` is the wkv projection of the same tokens, so a shape mismatch is a
    # caller bug that ``fused_engram_gate`` asserts on, not a reason to fall back.
    if (
        _cuda_kernels(x)
        and x.ndim == 3
        and x.dtype == kv.dtype
        and x.dtype in (torch.bfloat16, torch.float32)
        and q_weight.dtype in (torch.bfloat16, torch.float32)
        and k_weight.dtype in (torch.bfloat16, torch.float32)
        and all(t.is_contiguous() for t in (x, kv, q_weight, k_weight))
    ):
        return fused_engram_gate(
            x, kv, q_weight, k_weight, eps, clamp_value, input_ids, image_token_id
        )
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
    out = (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)
    if input_ids is not None:
        out = torch.where((input_ids == image_token_id)[:, None, None], x, out)
    return out


class Engram(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int,
        layout: EngramLayout,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        image_token_id: Optional[int] = None,
    ):
        super().__init__()
        self.layer_hash_index = layout.layer_ids.index(layer_id)
        self.image_token_id = image_token_id
        self.eps = config.rms_norm_eps
        self.clamp_value = 1e-6
        dim, hc_mult = config.hidden_size, config.hc_mult
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

    def forward(
        self,
        x: torch.Tensor,
        hash_ids: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        *,
        input_ids: torch.Tensor,
        cp_all_tokens: bool = False,
    ) -> torch.Tensor:
        """x [T, hc_mult, dim]; hash_ids [T, n_hash_cols] and input_ids [T] for this
        layer."""
        # The lookup runs first even for an idle DP-attention batch: under DP
        # attention it is a collective every rank has to join.
        if (
            self.embed.staged_by_scheduler
            and forward_batch is not None
            and forward_batch.forward_mode.is_extend_without_speculative()
            and is_in_breakable_cuda_graph()
        ):
            # Every replay stages new rows, so the lookup runs between segments.
            emb = _bcg_engram_embed(self, hash_ids, cp_all_tokens)
        else:
            emb = self.embed_rows(hash_ids, forward_batch, cp_all_tokens)
        if x.shape[0] == 0:
            # Nothing to gate, and the MXFP8 quantize behind wkv rejects an
            # empty M.
            return x
        kv, _ = self.wkv(emb.flatten(-2))
        image_ids = input_ids if self.image_token_id is not None else None
        return engram_gate(
            x,
            kv,
            self.q_weight,
            self.k_weight,
            self.eps,
            self.clamp_value,
            image_ids,
            self.image_token_id,
        )

    def embed_rows(
        self,
        hash_ids: torch.Tensor,
        forward_batch: Optional[ForwardBatch],
        cp_all_tokens: bool,
    ) -> torch.Tensor:
        """The rows the scheduler staged for this batch, else the table lookup."""
        staged = None if forward_batch is None else forward_batch.engram_staged_rows
        if staged is not None:
            return self.embed.dequant_staged(
                *staged[self.layer_hash_index], hash_ids.shape
            )
        return self.embed(hash_ids, forward_batch, cp_all_tokens=cp_all_tokens)


@eager_on_graph
def _bcg_engram_embed(
    engram: Engram, hash_ids: torch.Tensor, cp_all_tokens: bool
) -> torch.Tensor:
    # A replay builds a fresh batch; the capture-time one stages no rows.
    forward_batch = get_tc_piecewise_forward_context().forward_batch
    return engram.embed_rows(hash_ids, forward_batch, cp_all_tokens)
