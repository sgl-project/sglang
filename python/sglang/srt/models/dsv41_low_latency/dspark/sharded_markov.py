# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DSpark Markov draft chain in one cooperative kernel per rank, vocab-sharded.

Developed at NVIDIA; the CUDA kernel is unchanged. Rank k keeps rows [k V / world, (k +
1) V / world) of the Markov head (markov_w2) in registers, adds the bias of the previous
token's markov_w1 row to its shard of the base logits, picks its best (score, index) key
and exchanges one 64-bit key per rank, request row and draft step through a mailbox in
symmetric memory; the kept step logits reach every rank through peer staging. All draft
steps run in one launch.

SGLang glue: the Gumbel noise is SGLang's Philox draft noise (markov_sample.py, same key,
counters and offset), drawn for this rank's shard before the chain; request rows map to
themselves (temperature = the row's step temperature, 0 = argmax); a greedy draft keeps no
step logits, so its rows are passed as inert (argmax, no cache write). `ShardedMarkovChain`
binds it to the folded draft sampler (DsparkDraftSampler.markov_chain).
"""

from __future__ import annotations

import ctypes
import logging
from pathlib import Path
from typing import NamedTuple, Optional

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.srt.models.dsv41_low_latency.routed_expert_gemm import (
    _check,
    compile_source,
)

logger = logging.getLogger(__name__)

SOURCE = Path(__file__).resolve().parent / "sharded_markov.cu"
SYMBOL = "dspark_sharded_markov_sample"
RANK_DIM = 256
THREADS = 512
WARPS = THREADS // 32
MAX_STEPS = 8
MAX_RANKS = 8
MAX_ROWS = 8  # MMA columns
MAX_VOCAB = (1 << 17) - 1  # the candidate key's index field
STAGING_OFFSET = 16384  # bytes; mailbox records come first
CONTROL_WORDS = 256  # int32 words of per-instance control state
ARRIVAL_WORDS = slice(128, 136)  # per-step arrival counters (sharded_markov.cu)
NOISE_BLOCK = 1024
NVRTC_OPTIONS = ("-std=c++17", "-default-device")
# Bound on each cross-rank wait in the kernel: past it the kernel reports and traps
# instead of hanging. Arbitrary; far above any step time.
TIMEOUT_MS = 20_000
_C_TYPES = {torch.bfloat16: "__nv_bfloat16", torch.float32: "float"}
# markov_sample.py's Philox draft noise.
_UNIFORM_SCALE = tl.constexpr(2.3283064365386963e-10)
_PHILOX_ROUNDS = tl.constexpr(10)


@triton.jit
def _shard_noise_kernel(
    NOISE,
    OFFSET,
    KEY,
    VOCAB_START,
    VOCAB_LOCAL,
    STEPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Gumbel noise of (row, step) for this rank's vocab shard, as _select_block draws it."""
    sample = tl.program_id(0)
    row = sample // STEPS
    step = sample % STEPS
    part = tl.program_id(1)
    quad = (part * BLOCK + VOCAB_START) // 4 + tl.arange(0, BLOCK // 4)
    lane = tl.zeros([BLOCK // 4], tl.int64)
    w0, w1, w2, w3 = tl.philox(
        KEY,
        quad.to(tl.uint32),
        (lane + row).to(tl.uint32),
        (lane + step).to(tl.uint32),
        ((lane + tl.load(OFFSET)) & 0xFFFFFFFF).to(tl.uint32),
        _PHILOX_ROUNDS,
    )
    bits = tl.interleave(tl.interleave(w0, w2), tl.interleave(w1, w3))
    uniform = (bits.to(tl.float32) + 0.5) * _UNIFORM_SCALE
    exp_noise = -libdevice.log1p(-uniform)
    gumbel = -tl.log(exp_noise)
    cols = part * BLOCK + tl.arange(0, BLOCK)
    tl.store(NOISE + sample * VOCAB_LOCAL + cols, gumbel, mask=cols < VOCAB_LOCAL)


class _Kernel(NamedTuple):
    function: object  # cuda.bindings.driver.CUfunction
    max_ctas: int


def _load_kernel(device, steps: int, base_dtype, cache_dtype) -> _Kernel:
    from cuda.bindings import driver as cu

    major, minor = torch.cuda.get_device_capability(device)
    defines = {
        "STEPS": str(steps),
        "BASE_T": _C_TYPES[base_dtype],
        "CACHE_T": _C_TYPES[cache_dtype],
    }
    options = NVRTC_OPTIONS + tuple(
        f"-D{name}={value}" for name, value in sorted(defines.items())
    )
    cubin = compile_source(
        SOURCE.read_text(), SYMBOL, f"sm_{major}{minor}a", options=options
    )
    with torch.cuda.device(device):
        # A runtime call makes the device's primary context current for the driver loads.
        torch.zeros(1, device=device)
        (module,) = _check(cu.cuModuleLoadData(cubin), "cuModuleLoadData")
        (function,) = _check(cu.cuModuleGetFunction(module, SYMBOL.encode()), SYMBOL)
        (per_sm,) = _check(
            cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(function, THREADS, 0),
            "cuOccupancyMaxActiveBlocksPerMultiprocessor",
        )
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return _Kernel(function, int(per_sm) * sms)


def _launch_cooperative(
    kernel: _Kernel, ctas: int, arguments: list, stream: int
) -> None:
    from cuda.bindings import driver as cu

    addresses = [ctypes.addressof(argument) for argument in arguments]
    pointer_array = (ctypes.c_void_p * len(addresses))(*addresses)
    config = cu.CUlaunchConfig()
    config.gridDimX, config.gridDimY, config.gridDimZ = ctas, 1, 1
    config.blockDimX, config.blockDimY, config.blockDimZ = THREADS, 1, 1
    config.sharedMemBytes = 0
    config.hStream = cu.CUstream(stream)
    attribute = cu.CUlaunchAttribute()
    attribute.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_COOPERATIVE
    attribute.value.cooperative = 1
    config.attrs = [attribute]
    config.numAttrs = 1
    _check(
        cu.cuLaunchKernelEx(
            config, kernel.function, ctypes.addressof(pointer_array), 0
        ),
        "cuLaunchKernelEx",
    )


def _all_ranks_true(group, value: bool) -> bool:
    """``value`` on every rank of the CPU ``group`` (None: this rank only)."""
    if group is None:
        return value
    flag = torch.tensor([int(value)], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def _pointer(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def unsupported_reason(
    *,
    world: int,
    vocab_size: int,
    num_steps: int,
    embed_weight,
    head_weight,
    cache_dtype,
    device,
) -> Optional[str]:
    """Why the chain kernel cannot serve this configuration, or None; the same on every rank."""
    if world not in (1, 2, 4, MAX_RANKS):
        return f"a tensor-parallel size of {world}"
    if torch.cuda.get_device_capability(device) < (10, 0):
        return "a GPU older than SM100"
    if not 1 <= num_steps <= MAX_STEPS:
        return f"{num_steps} draft positions"
    if vocab_size > MAX_VOCAB or vocab_size % (16 * world) != 0:
        return f"a vocabulary of {vocab_size} for {world} ranks"
    if cache_dtype not in _C_TYPES:
        return f"a {cache_dtype} step-logits buffer"
    for name, weight in (("embedding", embed_weight), ("head", head_weight)):
        if (
            weight.dim() != 2
            or weight.shape[1] != RANK_DIM
            or weight.dtype != torch.bfloat16
            or not weight.is_contiguous()
        ):
            return f"a Markov {name} that is not contiguous BF16 [*, {RANK_DIM}]"
    if head_weight.shape[0] < vocab_size or embed_weight.shape[0] < vocab_size:
        return "a Markov table smaller than the vocabulary"
    return None


class DSparkShardedMarkovSampler:
    """Persistent buffers, peer mappings and launchers of the sharded chain."""

    def __init__(
        self,
        *,
        embed_weight: torch.Tensor,
        head_weight: torch.Tensor,
        vocab_size: int,
        num_steps: int,
        cache_dtype: torch.dtype,
        group,
        rank: int,
        world: int,
        max_bs: int,
        timeout_ms: int = 0,
    ):
        device = embed_weight.device
        reason = unsupported_reason(
            world=world,
            vocab_size=vocab_size,
            num_steps=num_steps,
            embed_weight=embed_weight,
            head_weight=head_weight,
            cache_dtype=cache_dtype,
            device=device,
        )
        if reason is not None:
            raise ValueError(f"sharded Markov sampling does not support {reason}")
        self.group, self.rank, self.world = group, rank, world
        self.device = device
        self.vocab_size = vocab_size
        self.num_steps = num_steps
        self.cache_dtype = cache_dtype
        self.timeout_ns = int(timeout_ms) * 1_000_000
        self.vocab_local = vocab_size // world
        self.vocab_offset = rank * self.vocab_local
        self.embed_weight = embed_weight
        self.head_shard = head_weight[
            self.vocab_offset : self.vocab_offset + self.vocab_local
        ]
        tiles = self.vocab_local // 16
        self.ctas = min(
            torch.cuda.get_device_properties(device).multi_processor_count,
            triton.cdiv(tiles, WARPS),
        )
        # The exchange gives each (request row, rank) pair one lane of a warp.
        self.max_rows = max(1, min(MAX_ROWS, 32 // world, max_bs))
        self.noise = torch.empty(
            self.max_rows * num_steps,
            self.vocab_local,
            dtype=torch.float32,
            device=device,
        )
        # Combine maxima, arrival counters and the launch epoch (sharded_markov.cu).
        self.control = torch.zeros(CONTROL_WORDS, dtype=torch.int32, device=device)
        # Request row r reads temperature[r] and keeps its step logits at cache[r].
        rows = torch.arange(self.max_rows, dtype=torch.int32, device=device)
        self.sampled_rows = rows.repeat_interleave(num_steps)
        self.inert_rows = torch.full_like(self.sampled_rows, -1)
        self.anchor = torch.zeros(self.max_rows, dtype=torch.int32, device=device)
        size = (
            STAGING_OFFSET
            + 2 * self.max_rows * num_steps * vocab_size * cache_dtype.itemsize
        )
        # Compile before the collective set-up so a build failure cannot leave peers waiting.
        self._kernels: dict[torch.dtype, _Kernel] = {}
        failure = None
        try:
            for base_dtype in _C_TYPES:
                kernel = _load_kernel(device, num_steps, base_dtype, cache_dtype)
                if self.ctas > kernel.max_ctas:
                    raise RuntimeError(
                        f"sharded Markov sampling needs {self.ctas} co-resident CTAs; the device holds {kernel.max_ctas}"
                    )
                self._kernels[base_dtype] = kernel
        except (RuntimeError, ImportError) as error:
            failure = error
        if not _all_ranks_true(group, failure is None):
            raise RuntimeError(
                f"kernel set-up failed on a rank: {failure or 'another rank'}"
            )
        self._peer_buffer, peer_pointers = self._allocate_peer_buffer(size)
        self.peers = torch.tensor(peer_pointers, dtype=torch.int64, device=device)
        self._noise_stream = torch.cuda.Stream(device)
        self.launches = 0

    def _allocate_peer_buffer(self, size: int) -> tuple[torch.Tensor, list[int]]:
        if self.group is None:
            buffer = torch.zeros(size, dtype=torch.uint8, device=self.device)
            return buffer, [buffer.data_ptr()]
        import torch.distributed._symmetric_memory as symm_mem

        buffer = symm_mem.empty(size, dtype=torch.uint8, device=self.device)
        handle = symm_mem.rendezvous(buffer, self.group.group_name)
        pointers = [
            handle.get_buffer(peer, (size,), torch.uint8, storage_offset=0).data_ptr()
            for peer in range(self.world)
        ]
        buffer.zero_()
        self._handle = handle
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        return buffer, pointers

    def supports(self, base_logits: torch.Tensor) -> bool:
        """``[rows, num_steps, vocab or vocab // world]`` base logits the chain takes; depends
        only on shapes, which tensor parallelism keeps identical across ranks."""
        return (
            base_logits.dim() == 3
            and 1 <= base_logits.shape[0] <= self.max_rows
            and base_logits.shape[1] == self.num_steps
            and base_logits.shape[2] in (self.vocab_size, self.vocab_local)
            and base_logits.dtype in _C_TYPES
            and base_logits.stride(2) == 1
        )

    def launch_noise(self, *, rows: int, key: int, offset: torch.Tensor) -> None:
        samples = rows * self.num_steps
        _shard_noise_kernel[(samples, triton.cdiv(self.vocab_local, NOISE_BLOCK))](
            self.noise,
            offset,
            key,
            self.vocab_offset,
            self.vocab_local,
            STEPS=self.num_steps,
            BLOCK=NOISE_BLOCK,
        )

    def fork_noise(self, *, rows: int, key: int, offset: torch.Tensor) -> None:
        """Draw the noise on a side stream; join_noise must precede sample()."""
        self._noise_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self._noise_stream):
            self.launch_noise(rows=rows, key=key, offset=offset)

    def join_noise(self) -> None:
        torch.cuda.current_stream(self.device).wait_stream(self._noise_stream)

    def sample(
        self,
        *,
        base_logits: torch.Tensor,
        anchor_tokens: torch.Tensor,
        temperatures: torch.Tensor,
        logits_cache: Optional[torch.Tensor],
        draft_tokens: torch.Tensor,
    ) -> None:
        """Draft tokens of every row into ``draft_tokens`` [rows, num_steps] and, when
        ``logits_cache`` [>= rows, num_steps, vocab] is given, every row's full-vocab step
        logits; without it every row is argmaxed. The noise must be drawn already."""
        rows = base_logits.shape[0]
        assert self.supports(base_logits)
        assert temperatures.dtype == torch.float32
        assert draft_tokens.dtype == torch.int64 and draft_tokens.stride(1) == 1
        self.anchor[:rows].copy_(anchor_tokens)
        if logits_cache is None:
            idx_mapping, cache, stride0, stride1 = self.inert_rows, self.noise, 0, 0
        else:
            assert (
                logits_cache.dtype == self.cache_dtype
                and logits_cache.size(2) >= self.vocab_size
            )
            assert logits_cache.stride(2) == 1 and logits_cache.data_ptr() % 16 == 0
            idx_mapping, cache = self.sampled_rows, logits_cache
            stride0, stride1 = logits_cache.stride(0), logits_cache.stride(1)
        base_column = (
            self.vocab_offset if base_logits.shape[2] == self.vocab_size else 0
        )
        arguments = [
            _pointer(base_logits),
            ctypes.c_int64(base_logits.stride(0)),
            ctypes.c_int64(base_logits.stride(1)),
            ctypes.c_int64(base_column),
            _pointer(self.embed_weight),
            _pointer(self.head_shard),
            _pointer(self.anchor),
            ctypes.c_int32(1),
            _pointer(idx_mapping),
            _pointer(temperatures),
            _pointer(self.noise),
            _pointer(cache),
            ctypes.c_int64(stride0),
            ctypes.c_int64(stride1),
            _pointer(draft_tokens),
            ctypes.c_int64(draft_tokens.stride(0)),
            _pointer(self.peers),
            _pointer(self.control),
            ctypes.c_int32(self.rank),
            ctypes.c_int32(self.world),
            ctypes.c_int32(self.vocab_local),
            ctypes.c_int32(rows),
            ctypes.c_int32(self.max_rows),
            ctypes.c_int64(self.timeout_ns),
        ]
        stream = torch.cuda.current_stream(self.device).cuda_stream
        _launch_cooperative(
            self._kernels[base_logits.dtype], self.ctas, arguments, stream
        )
        self.launches += 1


def make_sharded_sampler(
    *, markov_head, gamma: int, max_bs: int, cache_dtype: torch.dtype
) -> Optional[DSparkShardedMarkovSampler]:
    """Build the sampler on every rank of the Markov head's shard group, or on none
    (reason logged). Every rank must call it: it runs CPU collectives."""
    shard = markov_head._tp_shard
    coordinator = markov_head._shard_group
    group = (
        None
        if coordinator is None or coordinator.world_size == 1
        else coordinator.cpu_group
    )
    world = 1 if group is None else coordinator.world_size
    rank = 0 if group is None else coordinator.rank_in_group
    vocab = int(markov_head.vocab_size)
    embed = markov_head.markov_w1.weight
    head = markov_head.markov_w2.weight
    reason = unsupported_reason(
        world=world,
        vocab_size=vocab,
        num_steps=gamma,
        embed_weight=embed,
        head_weight=head,
        cache_dtype=cache_dtype,
        device=embed.device,
    )
    # The kernel's vocab shard must be the LM head's (its local base logits).
    if (
        reason is None
        and shard is not None
        and (
            shard.org_vocab_start != rank * vocab // world
            or shard.num_embeddings_per_partition != vocab // world
        )
    ):
        reason = "an LM-head vocab partition that is not the even split"
    if not _all_ranks_true(group, reason is None):
        logger.warning(
            f"Sharded DSpark Markov sampling is disabled: {reason or 'another rank cannot use it'}."
        )
        return None
    sampler = None
    try:
        sampler = DSparkShardedMarkovSampler(
            embed_weight=embed,
            head_weight=head,
            vocab_size=vocab,
            num_steps=gamma,
            cache_dtype=cache_dtype,
            group=group,
            rank=rank,
            world=world,
            max_bs=max_bs,
            timeout_ms=TIMEOUT_MS,
        )
    except RuntimeError as error:
        logger.warning(f"Sharded DSpark Markov sampling setup failed: {error}")
    if not _all_ranks_true(group, sampler is not None):
        logger.warning(
            "Sharded DSpark Markov sampling is disabled: setup failed on a rank."
        )
        return None
    logger.info(
        f"Sharded DSpark Markov sampling installed ({sampler.world} ranks, {sampler.ctas} CTAs per rank, "
        f"batches of up to {sampler.max_rows} request rows)."
    )
    return sampler


class ShardedMarkovChain:
    """The folded draft sampler's Markov block on the sharded kernel, for the batches
    it takes; ranks agree on every token inside the kernel, so no per-step TP sync."""

    def __init__(self, sampler: DSparkShardedMarkovSampler):
        self._sampler = sampler

    def fork_noise(self, *, rows: int, noise) -> bool:
        """Draw a sampled draft's noise beside the base logits (it depends on neither
        them nor the chain); ``noise`` is None for a greedy draft. Shape-only gate."""
        if noise is None or rows > self._sampler.max_rows:
            return False
        self._sampler.fork_noise(rows=rows, key=noise.key, offset=noise.offset)
        return True

    def sample(
        self,
        *,
        base_logits: torch.Tensor,
        anchor_tokens: torch.Tensor,
        temperatures: torch.Tensor,
        logits_cache: Optional[torch.Tensor],
        noise,
        forked: bool,
        out: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """``out`` [rows, steps] holds the draft tokens, or None when the per-step path
        must run. ``logits_cache`` and ``noise`` are None for a greedy draft."""
        if forked:
            self._sampler.join_noise()
        if not self._sampler.supports(base_logits):
            return None
        self._sampler.sample(
            base_logits=base_logits,
            anchor_tokens=anchor_tokens,
            temperatures=temperatures,
            logits_cache=logits_cache,
            draft_tokens=out,
        )
        if noise is not None:
            # One noise offset per sampled block, as the per-step path advances it.
            noise.offset.add_(1)
        return out


def install_draft_sampler(draft_sampler) -> None:
    """Bind the sharded chain to a DSpark V4 draft sampler; collective over the Markov
    head's shard group, so every rank calls it."""
    from sglang.srt.models.deepseek_v4_dspark import DSparkV4MarkovHead

    markov_head = draft_sampler.markov_head
    if not isinstance(markov_head, DSparkV4MarkovHead):
        return
    # The kernel rounds the bias to BF16 before the FP32 add, as the BF16 w2 path does;
    # unsupported_reason requires a BF16 markov_w2.
    sampler = make_sharded_sampler(
        markov_head=markov_head,
        gamma=draft_sampler.gamma,
        max_bs=draft_sampler.max_bs,
        cache_dtype=markov_head.step_logits_dtype(torch.bfloat16),
    )
    if sampler is not None:
        draft_sampler.markov_chain = ShardedMarkovChain(sampler)
