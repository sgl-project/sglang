"""LiteTopK fused sparse top-k indexer for SGLang's DSA prefill path.

The fixed production route pair-swaps carry HOT12288 into the ordinary
paged-cache gather prefix. DeepGEMM scores and seed prep emit either that full
prefix or the paged candidate route's HOT8192 inline tier once, one
fixed-threshold no-histogram kernel scans only the suffix, and the qualified
GLM K=2048 path uses an h2048 physical selector with an exact overflow
continuation. One winner-only epilogue then maps final TOPK indices back to
original token positions and accumulates carry votes.

Both fused paths avoid materializing and rereading the full ``[Q, S]`` logits
matrix.

Env knobs:
  SGLANG_LITETOPK=1            enable
  SGLANG_LITETOPK_SO            optional prebuilt extension whose basename and
                              module name match the current source digest
  SGLANG_LITETOPK_SO_SHA256     required SHA256 when SGLANG_LITETOPK_SO is set
  SGLANG_LITETOPK_PRODUCTION_MIN_S
                              FP8 fused-path crossover (default 196608); an
                              explicit value also overrides the FP4 default
  SGLANG_LITETOPK_FP4_PRODUCTION_MIN_S
                              FP4 fused-path crossover (default 65536)
  SGLANG_LITETOPK_MERGE_CAP    legacy contiguous-slab capacity (default
                              196608); the paged route's logical cap is S
  SGLANG_LITETOPK_PAGED_CANDIDATES
                              use an 8K inline tier plus a shared overflow-page
                              pool for qualified FP8 Q=4096/4088 calls
  SGLANG_LITETOPK_PAGED_POOL_PAGES_PER_ROW
                              physical shared-pool budget in 4096-record pages
                              per active Q row (default 8; not a logical cap)
  SGLANG_LITETOPK_TIERED_SEED_12K
                              preserve HOT12K while placing its strongest
                              HOT8K in the paged calibration tier
  SGLANG_LITETOPK_CP_GLOBAL_CARRY
                              sum CP-local carry-vote histograms on a dedicated
                              asynchronous NCCL communicator before selecting
                              HOT12K (default 0)
  SGLANG_LITETOPK_CP_PACKED_INDEX_K_AG
                              quantize each rank's index K before the async CP
                              all-gather and transport one 132-byte fp8+scale
                              record instead of 256 bf16 bytes (default 0)
  SGLANG_LITETOPK_HEADROOM     bucket-scale headroom (default 0)
"""

import hashlib
import importlib.util
import os
import sys

import torch

# SGLang vendors the qualified kernel next to this adapter. The source digest
# is part of the extension module name, so stale binaries fail closed.
_DSA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "litetopk_kernels")
_BUILD_DIR = os.environ.get(
    "SGLANG_LITETOPK_BUILD",
    os.path.expanduser("~/.cache/sglang/litetopk_build"),
)

ENABLED = os.environ.get("SGLANG_LITETOPK", "0") == "1"
# Experimental eager CP-v2 pipeline.  The indexer owns the actual stream DAG;
# this process-wide flag only keeps qualification A/Bs reversible.
CP_ASYNC_PREP = os.environ.get("SGLANG_LITETOPK_CP_ASYNC_PREP", "0") == "1"
CP_PACKED_INDEX_K_AG = (
    os.environ.get("SGLANG_LITETOPK_CP_PACKED_INDEX_K_AG", "0") == "1"
)
if CP_PACKED_INDEX_K_AG and not (ENABLED and CP_ASYNC_PREP):
    raise ValueError(
        "SGLANG_LITETOPK_CP_PACKED_INDEX_K_AG=1 requires both "
        "SGLANG_LITETOPK=1 and SGLANG_LITETOPK_CP_ASYNC_PREP=1"
    )
CP_GLOBAL_CARRY = os.environ.get("SGLANG_LITETOPK_CP_GLOBAL_CARRY", "0") == "1"
if CP_GLOBAL_CARRY and not ENABLED:
    raise ValueError("SGLANG_LITETOPK_CP_GLOBAL_CARRY=1 requires SGLANG_LITETOPK=1")
if CP_GLOBAL_CARRY and os.environ.get("SGLANG_LITETOPK_REQUIRED", "0") != "1":
    raise ValueError(
        "SGLANG_LITETOPK_CP_GLOBAL_CARRY=1 requires "
        "SGLANG_LITETOPK_REQUIRED=1 so collective participation fails closed"
    )
_PRODUCTION_MIN_S_OVERRIDE = os.environ.get("SGLANG_LITETOPK_PRODUCTION_MIN_S")
PRODUCTION_MIN_S = int(_PRODUCTION_MIN_S_OVERRIDE or "196608")
FP4_PRODUCTION_MIN_S = int(
    os.environ.get(
        "SGLANG_LITETOPK_FP4_PRODUCTION_MIN_S",
        _PRODUCTION_MIN_S_OVERRIDE or "65536",
    )
)
PRODUCTION_MAX_S = 1 << 20
if not (
    16384 <= PRODUCTION_MIN_S <= PRODUCTION_MAX_S
    and 16384 <= FP4_PRODUCTION_MIN_S <= PRODUCTION_MAX_S
):
    # The exact-once prefix/suffix split needs HOT12288 plus a chunk-step of
    # certified suffix below the crossover (16384 is the compressed-coordinate
    # floor for DeepSeek-V4's ratio-4 indexer; the selector cap floor is
    # enforced K-relative at the call sites).
    raise ValueError(
        "LiteTopK FP8/FP4 production min-S values must be in [16384, 1<<20]"
    )


def production_min_s(use_fp4: bool) -> int:
    """Return the qualified crossover for the selected cache format."""
    return FP4_PRODUCTION_MIN_S if use_fp4 else PRODUCTION_MIN_S


def _permuted_plan_length_metadata(sequence_length, qualification_sequence_length=None):
    """Keep the local scan extent separate from CP-global qualification."""
    sequence_length = int(sequence_length)
    return {
        "sequence_length": sequence_length,
        "qualification_sequence_length": (
            sequence_length
            if qualification_sequence_length is None
            else int(qualification_sequence_length)
        ),
    }


def single_request_production_ready_cpu(seq_lens_cpu, *, use_fp4: bool) -> bool:
    """Check the crossover from scheduler-owned CPU sequence lengths.

    Independent CP streams must make the same launch decision on every rank.
    Use single-request CPU metadata so a raised LiteTopK crossover restores the
    complete stock schedule below the threshold without a GPU synchronization.
    """
    return (
        seq_lens_cpu is not None
        and len(seq_lens_cpu) == 1
        and int(seq_lens_cpu[0].item()) >= production_min_s(use_fp4)
    )


# Qualified scheduler shapes. CP8 with a global 32K chunk produces the
# 4096/4088 pair; the larger shapes retain the FP4 routes.
FUSED_QUERY_LEN = 8192
FUSED_TAIL_QUERY_LEN = 8128
FP8_FUSED_QUERY_LENS = frozenset((4096, 4088))
FP4_FUSED_QUERY_LENS = frozenset((FUSED_QUERY_LEN, FUSED_TAIL_QUERY_LEN, 4096, 4032))
PAGED_CANDIDATES = os.environ.get("SGLANG_LITETOPK_PAGED_CANDIDATES", "0") == "1"
_PAGED_CANDIDATE_INLINE = 8192
_PAGED_CANDIDATE_PAGE = 4096
TIERED_SEED_12K = os.environ.get("SGLANG_LITETOPK_TIERED_SEED_12K", "0") == "1"
_PAGED_POOL_PAGES_PER_ROW = int(
    # GLM 1M/Q4088 can require >8 shared overflow pages per row on average.
    # Leave capacity headroom without changing the logical cap or seed.
    os.environ.get("SGLANG_LITETOPK_PAGED_POOL_PAGES_PER_ROW", "12")
)
if _PAGED_POOL_PAGES_PER_ROW < 1:
    raise ValueError("SGLANG_LITETOPK_PAGED_POOL_PAGES_PER_ROW must be positive")
if TIERED_SEED_12K and not PAGED_CANDIDATES:
    raise ValueError(
        "tiered LiteTopK calibration seeds require SGLANG_LITETOPK_PAGED_CANDIDATES=1"
    )


def supported_fused_query_lens(*, use_fp4: bool) -> frozenset[int]:
    """Return adapter-qualified Q shapes for the selected cache format."""
    return FP4_FUSED_QUERY_LENS if use_fp4 else FP8_FUSED_QUERY_LENS


def supports_fused_query_len(query_length: int, *, use_fp4: bool) -> bool:
    """Return whether the qualified adapter accepts this scheduler Q shape."""
    return int(query_length) in supported_fused_query_lens(use_fp4=use_fp4)


def enforce_required_path(
    *,
    enabled: bool,
    required: bool,
    use_fp4: bool,
    query_length: int,
    sequence_length: int,
    num_reqs: int,
    capturing: bool,
    route: str,
    dispatched: bool | None = None,
    extra_reasons: tuple[str, ...] = (),
) -> bool:
    """Fail closed for a required production LiteTopK dispatch.

    Contexts below the configured crossover intentionally retain the stock
    indexer.  At or above the crossover, however, REQUIRED must reject every
    unsupported prerequisite and every qualified-kernel decline instead of
    silently benchmarking the stock fallback.  The returned bool says whether
    the sequence has reached that production crossover.
    """
    if required and not enabled:
        raise RuntimeError("SGLANG_LITETOPK_REQUIRED=1 requires SGLANG_LITETOPK=1")

    production_ready = int(sequence_length) >= production_min_s(use_fp4)
    if not required or not production_ready:
        return production_ready

    reasons = list(extra_reasons)
    if int(num_reqs) != 1:
        reasons.append(f"requires one request, got {num_reqs}")
    if capturing:
        reasons.append("CUDA graph capture is active")
    if not supports_fused_query_len(query_length, use_fp4=use_fp4):
        supported = sorted(supported_fused_query_lens(use_fp4=use_fp4))
        reasons.append(
            f"unsupported Q={query_length}; supported query lengths are {supported}"
        )
    if dispatched is False:
        reasons.append("qualified kernel declined")
    if reasons:
        raise RuntimeError(
            "SGLANG_LITETOPK_REQUIRED=1 but "
            f"the {route} production path is unavailable: " + "; ".join(reasons)
        )
    return production_ready


HOT_PREFIX = 12288
NB = int(os.environ.get("SGLANG_LITETOPK_NB", "256"))
_TELEMETRY = {"calls": 0, "candidate_max": 0}
# Absolute forward headroom on the bucket scale (fraction of the sample span
# prepended ABOVE the sample max). Pair with a proportionally larger NB to
# keep bucket width unchanged (e.g. HEADROOM=1.0 + NB=512 == today's width).
HEADROOM = float(os.environ.get("SGLANG_LITETOPK_HEADROOM", "0.0"))
# Capacity of the legacy contiguous candidate slab.  It is deliberately not a
# correctness bound for the paged route: there the logical cap is the complete
# sequence length S, while a shared physical page pool fails closed if its
# observed working set does not fit.
MERGE_CAP = int(os.environ.get("SGLANG_LITETOPK_MERGE_CAP", "196608"))
# The K-relative floor (cap >= 32*topk, e.g. 16384 at K=512) is enforced at
# the call sites where topk is known; this import-time check only rejects
# configurations no supported K could satisfy.
if MERGE_CAP < 16384:
    raise ValueError(
        "SGLANG_LITETOPK_MERGE_CAP must be at least 16384 for the "
        "fixed-HOT no-hist production path"
    )
# OVF_LOG: print the running max of sampled per-row candidate counts (from
# the existing deferred 1-in-8 probe; sync-free).  This sizes the legacy slab;
# paged qualification separately records logical cap S and page-pool usage.
OVF_LOG = os.environ.get("SGLANG_LITETOPK_OVF_LOG", "0") == "1"
_HOT_STREAM = {}
_CP_GLOBAL_CARRY_COMM = None
_CP_GLOBAL_CARRY_WORLD_SIZE = 1
_CP_GLOBAL_CARRY_DEVICE = None
_CP_GLOBAL_CARRY_GROUP_ID = None
_CP_GLOBAL_CARRY_GROUP = None
_CP_GLOBAL_CARRY_FP16 = {}
_CP_GLOBAL_CARRY_LOGGED = False
_CP_GLOBAL_CARRY_EPOCHS = 0
PROBE_EVERY = int(os.environ.get("SGLANG_LITETOPK_PROBE_EVERY", "8"))
if PROBE_EVERY < 1:
    raise ValueError("SGLANG_LITETOPK_PROBE_EVERY must be >= 1")
OVF_WATERMARK = int(os.environ.get("SGLANG_LITETOPK_OVF_WATERMARK", "65536"))
# Real adjacent-chunk capture selected this window: all K winners from the last
# 1536 query rows predict the next chunk substantially better than the old
# rotating 1/8 sample over all rows, while adding only atomics to the mandatory
# winner-map pass.
CARRY_RECENT_ROWS = 1536
_CP_CARRY_RECENT_ROWS = int(os.environ.get("SGLANG_LITETOPK_CP_CARRY_RECENT_ROWS", "0"))
RELEASE_SCRATCH_ON_ROLLBACK = (
    os.environ.get("SGLANG_LITETOPK_RELEASE_SCRATCH_ON_ROLLBACK", "0") == "1"
)
if _CP_CARRY_RECENT_ROWS < 0:
    raise ValueError("SGLANG_LITETOPK_CP_CARRY_RECENT_ROWS must be >= 0")


def carry_recent_rows_for_cp(cp_size: int) -> int:
    """Choose the local carry-vote window under interleaved CP.

    Zero keeps the global-window-equivalent default.  A positive experimental
    override lets qualification measure whether more local history produces a
    tighter HOT set for small-Q CP without changing the selector contract.
    The caller still caps the returned value by the actual local Q rows.
    """
    cp_size = int(cp_size)
    if cp_size <= 0:
        raise ValueError(f"cp_size must be positive, got {cp_size}")
    if _CP_CARRY_RECENT_ROWS:
        return _CP_CARRY_RECENT_ROWS
    return max(1, (CARRY_RECENT_ROWS + cp_size - 1) // cp_size)


def configure_cp_global_carry(cp_group) -> None:
    """Record the CP group used by the carry-vote communicator.

    CP8 commonly aliases the attention-CP group to TP. A new
    ``PyNcclCommunicator`` gets its own NCCL unique ID while using only the
    existing CPU group for bootstrap, so carry collectives cannot reorder or
    head-of-line block the model's TP/index-K collectives. This constructor-time
    hook only records the group. ``initialize_cp_global_carry`` materializes it
    at SGLang's explicit all-rank post-pool startup hook, so NCCL's
    process-lifetime buffers neither reduce the profiled maximum context length
    nor bootstrap from a data-dependent carry path.
    """
    global _CP_GLOBAL_CARRY_WORLD_SIZE
    global _CP_GLOBAL_CARRY_DEVICE, _CP_GLOBAL_CARRY_GROUP_ID
    global _CP_GLOBAL_CARRY_GROUP

    if not CP_GLOBAL_CARRY:
        return
    world_size = int(cp_group.world_size)
    if world_size <= 1:
        raise RuntimeError("global CP carry requires CP world_size > 1")
    device = torch.device(cp_group.device)
    group_id = (
        tuple(int(rank) for rank in cp_group.ranks),
        int(cp_group.rank_in_group),
    )
    if _CP_GLOBAL_CARRY_GROUP is not None:
        if (
            _CP_GLOBAL_CARRY_WORLD_SIZE != world_size
            or _CP_GLOBAL_CARRY_DEVICE != device
            or _CP_GLOBAL_CARRY_GROUP_ID != group_id
        ):
            raise RuntimeError(
                "global CP carry communicator was initialized for a different "
                f"group/device: world_size={_CP_GLOBAL_CARRY_WORLD_SIZE}, "
                f"device={_CP_GLOBAL_CARRY_DEVICE}; requested "
                f"world_size={world_size}, device={device}, group={group_id}"
            )
        return

    _CP_GLOBAL_CARRY_WORLD_SIZE = world_size
    _CP_GLOBAL_CARRY_DEVICE = device
    _CP_GLOBAL_CARRY_GROUP_ID = group_id
    _CP_GLOBAL_CARRY_GROUP = cp_group


def _initialize_cp_global_carry_comm() -> None:
    """Materialize the dedicated communicator at an all-rank startup hook."""
    global _CP_GLOBAL_CARRY_COMM

    if _CP_GLOBAL_CARRY_COMM is not None:
        return
    cp_group = _CP_GLOBAL_CARRY_GROUP
    if cp_group is None or _CP_GLOBAL_CARRY_DEVICE is None:
        raise RuntimeError(
            "global CP carry group was not configured during Indexer construction"
        )

    from sglang.srt.distributed.device_communicators.pynccl import (
        PyNcclCommunicator,
    )

    comm = PyNcclCommunicator(group=cp_group.cpu_group, device=_CP_GLOBAL_CARRY_DEVICE)
    if not getattr(comm, "available", False):
        raise RuntimeError(
            "global CP carry requested, but a dedicated PyNccl communicator "
            "could not be initialized"
        )
    _CP_GLOBAL_CARRY_COMM = comm
    if int(cp_group.rank_in_group) == 0:
        print(
            "LITETOPK_CP_GLOBAL_CARRY_COMM initialized "
            f"world_size={_CP_GLOBAL_CARRY_WORLD_SIZE} "
            f"device={_CP_GLOBAL_CARRY_DEVICE}",
            flush=True,
        )


def initialize_cp_global_carry(max_sequence_length: int) -> None:
    """Initialize and preallocate global-carry state after final pool sizing.

    Every CP rank calls this from the scheduler's common startup path.  The
    PyNccl constructor performs its own one-element collective warmup, so no
    CPU-group bootstrap or first-use NCCL synchronization remains in the
    request path.  Preallocating the exact FP16 wire slab also keeps allocator
    work out of the first eligible chunk; the slab itself is overwritten before
    every all-reduce and therefore needs no initialization kernel.
    """
    if not CP_GLOBAL_CARRY:
        return
    max_sequence_length = int(max_sequence_length)
    if max_sequence_length <= 0:
        raise ValueError(
            "global CP carry max_sequence_length must be positive, got "
            f"{max_sequence_length}"
        )

    _initialize_cp_global_carry_comm()
    device = _CP_GLOBAL_CARRY_DEVICE
    assert device is not None
    dev_key = str(device)
    if _HOT_STREAM.get(dev_key) is None:
        _HOT_STREAM[dev_key] = torch.cuda.Stream(device=device)

    configured_global_max = (
        carry_recent_rows_for_cp(_CP_GLOBAL_CARRY_WORLD_SIZE)
        * _CP_GLOBAL_CARRY_WORLD_SIZE
    )
    wire_dtype = "int32"
    wire_values = 0
    if configured_global_max <= 2048:
        live_values = min(max_sequence_length, PRODUCTION_MAX_S)
        cap = max(1024, 1 << (live_values - 1).bit_length())
        wire = _CP_GLOBAL_CARRY_FP16.get(dev_key)
        if wire is None or wire.numel() < cap:
            _CP_GLOBAL_CARRY_FP16[dev_key] = torch.empty(
                cap, dtype=torch.float16, device=device
            )
        wire_dtype = "fp16-exact"
        wire_values = cap

    cp_group = _CP_GLOBAL_CARRY_GROUP
    assert cp_group is not None
    if int(cp_group.rank_in_group) == 0:
        print(
            "LITETOPK_CP_GLOBAL_CARRY_READY "
            f"wire_dtype={wire_dtype} values={wire_values} "
            f"max_vote={configured_global_max}",
            flush=True,
        )


def shutdown_cp_global_carry() -> None:
    """Forget the process-lifetime communicator before parallel re-init.

    PyNccl intentionally has no Python destructor because ncclCommDestroy can
    itself be collective. Synchronize our side stream, then drop references;
    the CUDA context owns final reclamation, matching SGLang's other raw
    PyNccl users.
    """
    global _CP_GLOBAL_CARRY_COMM, _CP_GLOBAL_CARRY_WORLD_SIZE
    global _CP_GLOBAL_CARRY_DEVICE, _CP_GLOBAL_CARRY_GROUP_ID
    global _CP_GLOBAL_CARRY_GROUP
    global _CP_GLOBAL_CARRY_LOGGED
    global _CP_GLOBAL_CARRY_EPOCHS

    if _CP_GLOBAL_CARRY_COMM is not None and _CP_GLOBAL_CARRY_DEVICE is not None:
        side = _HOT_STREAM.get(str(_CP_GLOBAL_CARRY_DEVICE))
        if side is not None:
            side.synchronize()
        cp_group = _CP_GLOBAL_CARRY_GROUP
        if cp_group is not None and int(cp_group.rank_in_group) == 0:
            print(
                f"LITETOPK_CP_GLOBAL_CARRY_SUMMARY epochs={_CP_GLOBAL_CARRY_EPOCHS}",
                flush=True,
            )
    _CP_GLOBAL_CARRY_COMM = None
    _CP_GLOBAL_CARRY_WORLD_SIZE = 1
    _CP_GLOBAL_CARRY_DEVICE = None
    _CP_GLOBAL_CARRY_GROUP_ID = None
    _CP_GLOBAL_CARRY_GROUP = None
    _CP_GLOBAL_CARRY_FP16.clear()
    _CP_GLOBAL_CARRY_LOGGED = False
    _CP_GLOBAL_CARRY_EPOCHS = 0


def _globalize_cp_carry_votes(votes, local_max_vote):
    """Asynchronously sum exact CP votes on the current (HOT) CUDA stream."""
    global _CP_GLOBAL_CARRY_LOGGED, _CP_GLOBAL_CARRY_EPOCHS

    if not CP_GLOBAL_CARRY:
        return int(local_max_vote)
    if _CP_GLOBAL_CARRY_COMM is None:
        raise RuntimeError(
            "global CP carry reached the request path before the all-rank "
            "post-pool initialization hook"
        )
    comm = _CP_GLOBAL_CARRY_COMM
    assert comm is not None
    if votes.device != _CP_GLOBAL_CARRY_DEVICE:
        raise RuntimeError(
            "global CP carry vote tensor is on the wrong device: "
            f"votes={votes.device}, communicator={_CP_GLOBAL_CARRY_DEVICE}"
        )
    if votes.dim() != 1 or votes.dtype != torch.int32 or not votes.is_contiguous():
        raise RuntimeError(
            "global CP carry requires one contiguous int32 vote histogram"
        )

    local_max_vote = int(local_max_vote)
    configured_local_max = carry_recent_rows_for_cp(_CP_GLOBAL_CARRY_WORLD_SIZE)
    if not 0 < local_max_vote <= configured_local_max:
        raise RuntimeError(
            "global CP carry observed an invalid local vote bound: "
            f"actual={local_max_vote}, configured={configured_local_max}"
        )
    # Use the same configured bound on every rank.  A short or eventually
    # uneven tail may contribute fewer rows locally; multiplying an observed
    # rank-local count would give different selector clamp ranges after the
    # global sum and could split the HOT result across ranks.
    global_max_vote = configured_local_max * _CP_GLOBAL_CARRY_WORLD_SIZE
    if not 0 < global_max_vote <= 8192:
        raise RuntimeError(
            "global CP carry selector bound must be in [1, 8192], got "
            f"local={configured_local_max}, "
            f"world_size={_CP_GLOBAL_CARRY_WORLD_SIZE}, "
            f"global={global_max_vote}"
        )

    # Every integer through 2048 is exactly representable in FP16. The
    # default CP8 tree never has an intermediate sum above 192*8=1536, so the
    # 2-MiB wire buffer is an exact compression of the 4-MiB int32 histogram.
    # Larger experimental vote windows retain exactness via int32 NCCL.
    if global_max_vote <= 2048:
        key = str(votes.device)
        wire = _CP_GLOBAL_CARRY_FP16.get(key)
        if wire is None or wire.numel() < votes.numel():
            cap = max(1024, 1 << (votes.numel() - 1).bit_length())
            wire = torch.empty(cap, dtype=torch.float16, device=votes.device)
            _CP_GLOBAL_CARRY_FP16[key] = wire
        wire = wire[: votes.numel()]
        wire.copy_(votes)
        with comm.change_state(enable=True):
            comm.all_reduce(wire)
        votes.copy_(wire)
        wire_dtype = "fp16-exact"
    else:
        with comm.change_state(enable=True):
            comm.all_reduce(votes)
        wire_dtype = "int32"

    _CP_GLOBAL_CARRY_EPOCHS += 1

    if not _CP_GLOBAL_CARRY_LOGGED:
        print(
            "LITETOPK_CP_GLOBAL_CARRY active "
            f"world_size={_CP_GLOBAL_CARRY_WORLD_SIZE} "
            f"wire_dtype={wire_dtype} values={votes.numel()} "
            f"max_vote={global_max_vote}",
            flush=True,
        )
        _CP_GLOBAL_CARRY_LOGGED = True
    return global_max_vote


_HOT_CARRY = {}
# GATE4 writes BUCKET-SPACE high24 candidates (affine order-preserving).
# Both seed-prefix emission and the suffix producer use the same packed score
# contract, so the mapped postpass can process their concatenation directly.

_EXT = None
_FAILED = False
_AUX_CACHE = {}  # (device, head) -> (zeros[Qmax], full_head[Qmax]) int32
_SINGLE_SCAN_LOGGED = False


def _dsa_source_id():
    digest = hashlib.sha256()
    for filename in (
        "dsa_litetopk.cu",
        "sm100_dsa_litetopk.cuh",
    ):
        path = os.path.join(_DSA_DIR, filename)
        digest.update(filename.encode())
        with open(path, "rb") as source:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()[:12]


def _ks0_keh(Q, head, dev):
    """Cached zero-starts and sample-end tensors: torch.zeros/torch.full are a
    kernel launch each and measurably cost ~0.1-0.2ms/chunk on the hot path."""
    key = (str(dev), head)
    entry = _AUX_CACHE.get(key)
    if entry is None or entry[0].shape[0] < Q:
        qmax = max(Q, 1024)
        entry = (
            torch.zeros(qmax, dtype=torch.int32, device=dev),
            torch.full((qmax,), head, dtype=torch.int32, device=dev),
        )
        _AUX_CACHE[key] = entry
    return entry[0][:Q], entry[1][:Q]


def _ext():
    global _EXT, _FAILED
    if _EXT is None and not _FAILED:
        try:
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0a")
            source_id = _dsa_source_id()
            name = f"sglang_litetopk_dsa_b200_production_{source_id}"
            override_path = os.environ.get("SGLANG_LITETOPK_SO", "")
            override_sha256 = os.environ.get("SGLANG_LITETOPK_SO_SHA256", "")
            if bool(override_path) != bool(override_sha256):
                raise RuntimeError(
                    "SGLANG_LITETOPK_SO and SGLANG_LITETOPK_SO_SHA256 must "
                    "be set together"
                )
            if override_path:
                resolved_path = os.path.realpath(os.path.expanduser(override_path))
                expected_basename = f"{name}.so"
                if os.path.basename(resolved_path) != expected_basename:
                    raise RuntimeError(
                        "LiteTopK override basename must be "
                        f"{expected_basename}, got "
                        f"{os.path.basename(resolved_path)}"
                    )
                if not os.path.isfile(resolved_path):
                    raise FileNotFoundError(
                        f"LiteTopK override does not exist: {resolved_path}"
                    )
                expected_sha256 = override_sha256.lower()
                if len(expected_sha256) != 64 or any(
                    c not in "0123456789abcdef" for c in expected_sha256
                ):
                    raise RuntimeError(
                        "SGLANG_LITETOPK_SO_SHA256 must be 64 hexadecimal characters"
                    )
                digest = hashlib.sha256()
                with open(resolved_path, "rb") as binary:
                    for chunk in iter(lambda: binary.read(1 << 20), b""):
                        digest.update(chunk)
                actual_sha256 = digest.hexdigest()
                if actual_sha256 != expected_sha256:
                    raise RuntimeError(
                        "LiteTopK override SHA256 mismatch: expected "
                        f"{expected_sha256}, got {actual_sha256}"
                    )
                spec = importlib.util.spec_from_file_location(name, resolved_path)
                if spec is None or spec.loader is None:
                    raise RuntimeError(f"cannot create module spec for {resolved_path}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                try:
                    spec.loader.exec_module(module)
                except BaseException:
                    sys.modules.pop(name, None)
                    raise
                _EXT = module
                load_kind = "prebuilt"
            else:
                from torch.utils.cpp_extension import load

                dg25 = os.environ.get("DEEPGEMM_DIR", "/opt/glm5_prefill_test/DeepGEMM")
                src = "dsa_litetopk.cu"
                bdir = f"{_BUILD_DIR}_production_{source_id}"
                dg_inc = os.path.join(dg25, "deep_gemm/include")
                cutlass_inc = os.path.join(dg25, "third-party/cutlass/include")
                if os.path.isdir(dg_inc) and os.path.isdir(cutlass_inc):
                    incs = [_DSA_DIR, dg_inc, cutlass_inc]
                else:
                    # A pinned DeepGEMM wheel may bundle both DeepGEMM and
                    # CUTLASS headers under its package include directory.
                    import deep_gemm

                    pkg_inc = os.path.join(
                        os.path.dirname(deep_gemm.__file__), "include"
                    )
                    if not os.path.isfile(
                        os.path.join(pkg_inc, "cutlass/arch/barrier.h")
                    ):
                        raise RuntimeError(
                            "DeepGEMM/CUTLASS headers not found; set "
                            "DEEPGEMM_DIR to a DeepGEMM 2.5 checkout"
                        )
                    incs = [_DSA_DIR, pkg_inc]
                cuda_flags = [
                    "-O3",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-gencode=arch=compute_100a,code=sm_100a",
                ]
                if os.environ.get("LITETOPK_LINEINFO") == "1":
                    cuda_flags.append("-lineinfo")
                os.makedirs(bdir, exist_ok=True)
                _EXT = load(
                    name=name,
                    sources=[os.path.join(_DSA_DIR, src)],
                    extra_include_paths=incs,
                    extra_cuda_cflags=cuda_flags,
                    build_directory=bdir,
                    extra_ldflags=["-lcuda"],
                    verbose=False,
                )
                load_kind = "JIT"
            reported_u16 = getattr(_EXT, "candidate_value_u16_litetopk", None)
            if reported_u16 is None or not bool(reported_u16()):
                raise RuntimeError(
                    "loaded LiteTopK extension is not the U16 candidate ABI"
                )
            reported_fp24 = getattr(_EXT, "candidate_fp24_global_litetopk", None)
            if reported_fp24 is None or not bool(reported_fp24()):
                raise RuntimeError(
                    "loaded LiteTopK extension is not the production high24 ABI"
                )
            required_ops = (
                "plan_and_permuted_paged_gather_out",
                "plan_tiered_and_permuted_paged_gather_out",
                "h2048_safe_topk_out_paged_litetopk_",
            )
            for required_op in required_ops:
                if not hasattr(_EXT, required_op):
                    raise RuntimeError(
                        "loaded LiteTopK extension is missing required op "
                        f"{required_op}"
                    )
            print(
                f"[litetopk] using {load_kind} fixed vendored B200 "
                f"production kernel (source={source_id})",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            _FAILED = True
            print(
                f"[litetopk] extension load/build failed, falling back: {e}",
                flush=True,
            )
    return _EXT


_HINTS_VALIDATED = False
_PENDING_OVF = None  # event, pinned stats, logical limit, K, watermark, paged
_DEEP_GEMM_ACCEPTS_OUT = None
_CAND_ACC = None  # (device running max[1], device over-watermark count[1]):
# accumulated unconditionally every call so the sampled
# probe readback still reports the complete running max
_PROBE_RES = None  # cached (device stats, device status max, pinned
# buffer, event): allocating
# these per arm blocked the CPU ~17ms inside
# cudaHostAlloc-class calls (nsys), starving the GPU
# stream for ~3.4ms/call at 256K/Q=512 when throttled


def _deferred_overflow_poll():
    """Non-blocking check of the previous chunk's candidate-overflow probe."""
    global _PENDING_OVF
    if _PENDING_OVF is not None:
        ev, pinned, limitv, kk, watermark, paged = _PENDING_OVF
        if ev.query():  # finished long ago; no sync
            mx = int(pinned[0])
            st = int(pinned[2])
            run_max = int(pinned[3])
            over = int(pinned[4])
            if st != 0:
                if CP_GLOBAL_CARRY:
                    # A rank-local async event can become queryable at a
                    # different host time on each CP rank. Raising here would
                    # let one worker exit before its peers enqueue the next
                    # global carry collective. Preserve the fatal evidence in
                    # every qualification log and let run_matrix reject the
                    # arm after workers finish instead of risking an NCCL hang.
                    kind = "paged" if paged else "contiguous"
                    print(
                        "LITETOPK_CP_GLOBAL_CARRY_DEFERRED_FATAL "
                        f"kind={kind} selector_status={st} "
                        f"candidate_max={mx} limit={limitv}",
                        flush=True,
                    )
                    _PENDING_OVF = None
                    return
                if paged:
                    raise RuntimeError(
                        f"[litetopk] paged selector status={st} on a probed "
                        f"chunk (candidate max {mx}, logical cap {limitv}); "
                        "the emitted top-k indices are unreliable — inspect "
                        "the overflow pool/page diagnostics"
                    )
                raise RuntimeError(
                    f"[litetopk] selector status={st} on a probed chunk "
                    f"(candidate max {mx}, cap {limitv}); the emitted top-k "
                    "indices are unreliable — raise SGLANG_LITETOPK_MERGE_CAP"
                )
            if not paged and run_max > limitv:
                print(
                    f"[litetopk] WARNING: candidate overflow ({run_max} > "
                    f"cap {limitv}); recall may dip on that chunk — raise "
                    "SGLANG_LITETOPK_MERGE_CAP",
                    flush=True,
                )
            if OVF_LOG and run_max > _TELEMETRY["candidate_max"]:
                # run_max/over are device-accumulated over EVERY call; only
                # the readback is sampled, so the printed max is the true
                # running max as of the probed chunk
                _TELEMETRY["candidate_max"] = run_max
                print(
                    f"[litetopk] cand max -> {run_max} "
                    f"(probed chunk mean {float(pinned[1]) / kk:.2f}xK, "
                    f"row-chunks over {watermark}: {over})",
                    flush=True,
                )
            _PENDING_OVF = None


_PREP_BUFS = {}  # (dev, NB) -> dict of caller-owned seed_prep buffers
_SLOG_SLABS = {}  # dev -> persistent seed-GEMM logits slab (out= reuse)
_OPS_VERIFIED = None  # required-ops hasattr walk, done once per ext load


def _slog_slab(Q, seq_len_kv, dev):
    """Persistent output slab for the seed GEMM: kills the ~392 MiB
    alloc/free per fused call. Sized generously for DeepGEMM's internal
    [align(Q, block_q), align(seq_len_kv + block_kv, 8)] padding."""
    need = (Q + 8) * (seq_len_kv + 512)
    slab = _SLOG_SLABS.get(str(dev))
    if slab is None or slab.numel() < need:
        _SLOG_SLABS[str(dev)] = None
        slab = torch.empty(need, dtype=torch.float32, device=dev)
        _SLOG_SLABS[str(dev)] = slab
    return slab


_CAND_BUFS = {}  # dev -> opaque U16 slab carrying delayed-high24 codes
_PAGED_CAND_BUFS = {}  # dev -> inline tier + shared overflow-page pool
_VOTE_BUF_HOT = {}  # dev -> persistent stash-carry vote histogram
_CARRY_VOTE_BUFS = {}  # (dev, layer) -> selector-fused vote slab + free event
_CARRY_TOPK_WORKSPACE = {}  # dev -> single-side-stream partial/state workspace
_CARRY_TOPK_MAX_BLOCKS = 128
_CARRY_TOPK_STATE_INTS = 136
# One pair-swap workspace is owned by each main CUDA stream.  Planning and the
# paged gather are submitted together through the production extension; no
# side-stream plan, prepared ticket, or per-layer permutation cache exists.
_PAIR_PLAN_BUFS = {}
_PAIR_PLAN_EPOCH = {}


def _stream_id(dev):
    return (
        int(torch.cuda.current_stream(dev).cuda_stream)
        if getattr(dev, "type", None) == "cuda"
        else 0
    )


def _pair_plan_bufs(sequence_length, dev):
    """Persistent pair-swap planner workspace with geometric growth."""
    key = (str(dev), _stream_id(dev))
    state = _PAIR_PLAN_BUFS.get(key)
    if state is None or state["cap"] < sequence_length:
        cap = max(16384, 1 << (sequence_length - 1).bit_length())
        state = {
            "cap": cap,
            "hot_epoch": torch.zeros(cap, dtype=torch.int32, device=dev),
            "permutation": torch.arange(cap, dtype=torch.int32, device=dev),
            "swap_a": torch.empty(HOT_PREFIX, dtype=torch.int32, device=dev),
            "swap_b": torch.empty(HOT_PREFIX, dtype=torch.int32, device=dev),
            # The tiered planner first installs the complete top12K set, then
            # swaps at most 4K misplaced strong/weak entries across the 8K
            # seed boundary. These pairs must be undone before the base swaps
            # are restored on the next epoch.
            "tier_swap_a": torch.empty(4096, dtype=torch.int32, device=dev),
            "tier_swap_b": torch.empty(4096, dtype=torch.int32, device=dev),
            "tier_counts": torch.zeros(4, dtype=torch.int32, device=dev),
            # [previous pair count, A count, B count, status].
            "counts": torch.zeros(4, dtype=torch.int32, device=dev),
        }
        _PAIR_PLAN_BUFS[key] = state
        _PAIR_PLAN_EPOCH[key] = 0
    epoch = _PAIR_PLAN_EPOCH.get(key, 0) + 1
    if epoch >= (1 << 30):
        # This is many years of continuous prefill calls. Recreate the planner
        # state instead of making epoch wrap a correctness concern.
        del _PAIR_PLAN_BUFS[key]
        _PAIR_PLAN_EPOCH.pop(key, None)
        return _pair_plan_bufs(sequence_length, dev)
    _PAIR_PLAN_EPOCH[key] = epoch
    return state, epoch


def _retire_request_state(dev, *, release_scratch=False):
    """Retire all state after a confirmed per-layer carry-extent rollback."""
    device_index = dev.index if dev.index is not None else torch.cuda.current_device()
    dev_key = str(torch.device("cuda", device_index))

    def _tuple_device_keys(cache):
        return [
            key for key in cache if isinstance(key, tuple) and key and key[0] == dev_key
        ]

    request_state = (_HOT_CARRY, _CARRY_VOTE_BUFS, _PAIR_PLAN_BUFS)
    has_request_state = any(bool(_tuple_device_keys(cache)) for cache in request_state)
    side = _HOT_STREAM.get(dev_key)
    if has_request_state and side is not None:
        side.synchronize()

    for cache in request_state:
        for key in _tuple_device_keys(cache):
            cache.pop(key, None)
    for key in _tuple_device_keys(_PAIR_PLAN_EPOCH):
        _PAIR_PLAN_EPOCH.pop(key, None)

    if release_scratch:
        for cache in (_PREP_BUFS, _AUX_CACHE):
            for key in _tuple_device_keys(cache):
                cache.pop(key, None)
        for cache in (
            _SLOG_SLABS,
            _CAND_BUFS,
            _PAGED_CAND_BUFS,
            _VOTE_BUF_HOT,
            _CARRY_TOPK_WORKSPACE,
        ):
            cache.pop(dev_key, None)


_SCRATCH_RETIRE_LOGGED = False


def retire_if_carry_extent_rollback(dev, sequence_length):
    """Release the previous request's fused scratch at the first new chunk.

    Sequence length is scheduler-owned CPU metadata and is monotonic within a
    single request.  A smaller length than any published carry extent therefore
    proves a request epoch rollback without a device synchronization.
    """
    global _SCRATCH_RETIRE_LOGGED
    if not RELEASE_SCRATCH_ON_ROLLBACK:
        return False
    device_index = dev.index if dev.index is not None else torch.cuda.current_device()
    dev_key = str(torch.device("cuda", device_index))
    current = int(sequence_length)
    rollback = any(
        isinstance(key, tuple)
        and key
        and key[0] == dev_key
        and value is not None
        and len(value) >= 2
        and int(value[1]) > current
        for key, value in _HOT_CARRY.items()
    )
    if not rollback:
        return False
    _retire_request_state(dev, release_scratch=True)
    if not _SCRATCH_RETIRE_LOGGED:
        print(
            "LITETOPK_SCRATCH_RETIRED previous-request GPU workspaces released",
            flush=True,
        )
        _SCRATCH_RETIRE_LOGGED = True
    return True


def _vote_hist(nv, dev):
    """Reuse one side-stream vote histogram for official-path carry seeding."""
    cache = _VOTE_BUF_HOT
    key = str(dev)
    b = cache.get(key)
    if b is None or b.numel() < nv:
        b = torch.empty(max(nv, 1024), dtype=torch.int32, device=dev)
        cache[key] = b
    buf = b[:nv]
    buf.zero_()
    return buf


def _cand_bufs(Q, cap, dev):
    key = str(dev)
    b = _CAND_BUFS.get(key)
    if b is None or b["cap"] != cap or b["q"] < Q:
        _CAND_BUFS[key] = None  # drop old slab BEFORE allocating the new one
        del b
        qm = max(Q, 1024)
        b = {
            "q": qm,
            "cap": cap,
            # float16 is opaque 16-bit storage in the packed ABI; CUDA
            # interprets its bits as uint16 rather than doing half arithmetic.
            "cv": torch.empty(qm, cap, dtype=torch.float16, device=dev),
            "ci": torch.empty(qm, cap, dtype=torch.int32, device=dev),
        }
        _CAND_BUFS[key] = b
    return b


def _paged_cand_bufs(Q, logical_cap, dev):
    """Acquire one reusable inline tier and shared overflow-page arena.

    The shared pool has no per-row cap: all rows contend for one arena. The
    page table uses flat maximum-width backing storage so its active view stays
    contiguous. Stale payload pages are unreachable and need no clearing.
    """
    Q = int(Q)
    logical_cap = int(logical_cap)
    if Q <= 0 or not (_PAGED_CANDIDATE_INLINE < logical_cap <= PRODUCTION_MAX_S):
        raise ValueError(
            f"invalid paged candidate shape Q={Q}, logical_cap={logical_cap}"
        )
    table_pages = (
        logical_cap - _PAGED_CANDIDATE_INLINE + _PAGED_CANDIDATE_PAGE - 1
    ) // _PAGED_CANDIDATE_PAGE
    pool_pages = _PAGED_POOL_PAGES_PER_ROW * Q
    key = str(dev)
    b = _PAGED_CAND_BUFS.get(key)
    if b is None or b["q"] < Q:
        _PAGED_CAND_BUFS[key] = None
        del b
        # Grow-only backing avoids Q=4096/tail-Q reallocation.
        qm = max(int(Q), 4096)
        max_table_pages = (
            PRODUCTION_MAX_S - _PAGED_CANDIDATE_INLINE + _PAGED_CANDIDATE_PAGE - 1
        ) // _PAGED_CANDIDATE_PAGE
        b = {
            "q": qm,
            "iv": torch.empty(
                qm,
                _PAGED_CANDIDATE_INLINE,
                dtype=torch.float16,
                device=dev,
            ),
            "ii": torch.empty(
                qm,
                _PAGED_CANDIDATE_INLINE,
                dtype=torch.int32,
                device=dev,
            ),
            "ov": torch.empty(
                _PAGED_POOL_PAGES_PER_ROW * qm,
                _PAGED_CANDIDATE_PAGE,
                dtype=torch.float16,
                device=dev,
            ),
            "oi": torch.empty(
                _PAGED_POOL_PAGES_PER_ROW * qm,
                _PAGED_CANDIDATE_PAGE,
                dtype=torch.int32,
                device=dev,
            ),
            "pt": torch.empty(
                qm * max_table_pages,
                dtype=torch.int32,
                device=dev,
            ),
            "head": torch.empty(1, dtype=torch.int32, device=dev),
            "status": torch.empty(1, dtype=torch.int32, device=dev),
        }
        _PAGED_CAND_BUFS[key] = b
    table_words = Q * table_pages
    return {
        "inline_val": b["iv"][:Q],
        "inline_idx": b["ii"][:Q],
        "overflow_val": b["ov"][:pool_pages],
        "overflow_idx": b["oi"][:pool_pages],
        "page_table": b["pt"][:table_words].view(Q, table_pages),
        "pool_head": b["head"],
        "pool_status": b["status"],
    }


def _carry_vote_hist(nv, dev, hot_key, waited_event=None):
    """Acquire a per-layer histogram for selector-fused carry votes."""
    key = (str(dev), hot_key)
    entry = _CARRY_VOTE_BUFS.get(key)
    if entry is not None and entry["free_event"] is not None:
        free_event = entry["free_event"]
        if free_event is not waited_event:
            torch.cuda.current_stream(dev).wait_event(free_event)
    if entry is None or entry["buf"].numel() < nv:
        cap = max(1024, 1 << (nv - 1).bit_length())
        hot = None if entry is None else entry.get("hot")
        strong = None if entry is None else entry.get("strong")
        ready_event = None if entry is None else entry.get("ready_event")
        if ready_event is None:
            ready_event = torch.cuda.Event()
        entry = {
            # The carry top-k kernel clears every live vote. Zero the whole
            # geometric slab once so future growth inside this capacity also
            # exposes clean, never-before-used tail positions.
            "buf": torch.zeros(cap, dtype=torch.int32, device=dev),
            "free_event": None,
            "ready_event": ready_event,
            "hot": (
                hot
                if hot is not None
                else torch.empty(HOT_PREFIX, dtype=torch.int64, device=dev)
            ),
            "strong": (
                strong
                if strong is not None
                else torch.empty(
                    _PAGED_CANDIDATE_INLINE,
                    dtype=torch.int64,
                    device=dev,
                )
            ),
            "needs_reset": False,
            "dirty_extent": 0,
        }
        _CARRY_VOTE_BUFS[key] = entry
    votes = entry["buf"][:nv]
    if entry.get("needs_reset", False):
        reset_extent = max(nv, entry.get("dirty_extent", 0))
        entry["buf"][:reset_extent].zero_()
        entry["needs_reset"] = False
        entry["dirty_extent"] = 0
    # The selector that follows will dirty the slab. A successfully enqueued
    # custom publisher clears this flag because its K2 owns the reset.
    entry["needs_reset"] = True
    entry["dirty_extent"] = max(entry.get("dirty_extent", 0), nv)
    return votes


def _carry_topk_workspace(max_vote, dev):
    """Caller-owned workspace serialized by the one side stream per device."""
    key = str(dev)
    bins = max_vote + 1
    entry = _CARRY_TOPK_WORKSPACE.get(key)
    if entry is None or entry["partial"].shape[1] < bins:
        entry = {
            "partial": torch.empty(
                (_CARRY_TOPK_MAX_BLOCKS, bins),
                dtype=torch.int16,
                device=dev,
            ),
            # state[0] is the reusable last-block completion ticket.
            "state": torch.zeros(
                _CARRY_TOPK_STATE_INTS,
                dtype=torch.int32,
                device=dev,
            ),
        }
        _CARRY_TOPK_WORKSPACE[key] = entry
    return entry


def _publish_carry(hot_key, votes, nv, min_index, max_vote):
    """Publish the voted HOT set on the per-device side stream."""
    if nv - min_index < HOT_PREFIX:
        return
    dev = votes.device
    key = (str(dev), hot_key)
    entry = _CARRY_VOTE_BUFS[key]
    side = _HOT_STREAM.get(str(dev))
    if side is None:
        side = torch.cuda.Stream(device=dev)
        _HOT_STREAM[str(dev)] = side
    carry_ext = _ext()
    side.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(side):
        votes.record_stream(side)
        max_vote = _globalize_cp_carry_votes(votes, max_vote)
        hot_n = HOT_PREFIX
        use_custom = (
            carry_ext is not None
            and hasattr(carry_ext, "carry_votes_topk_reset_")
            and (not TIERED_SEED_12K or hasattr(carry_ext, "carry_votes_topk_noreset_"))
            and nv <= 1_048_576
            and 0 < max_vote <= 8192
        )
        if use_custom:
            hot = entry["hot"][:hot_n]
            workspace = _carry_topk_workspace(max_vote, dev)
            strong = None
            if TIERED_SEED_12K:
                # Select the calibration tier before the full-set selector
                # clears the vote histogram.  The no-reset selector has the
                # same score/ID tie order as the reset variant, so strong is
                # an exact subset of the full top12K set.
                strong = entry["strong"][:_PAGED_CANDIDATE_INLINE]
                carry_ext.carry_votes_topk_noreset_(
                    votes,
                    strong,
                    workspace["partial"],
                    workspace["state"],
                    _PAGED_CANDIDATE_INLINE,
                    max_vote,
                    min_index,
                )
            carry_ext.carry_votes_topk_reset_(
                votes,
                hot,
                workspace["partial"],
                workspace["state"],
                hot_n,
                max_vote,
                min_index,
            )
            entry["needs_reset"] = False
            entry["dirty_extent"] = 0
        else:
            if min_index > 0:
                votes[:min_index].fill_(torch.iinfo(torch.int32).min)
            strong = (
                votes.topk(_PAGED_CANDIDATE_INLINE).indices if TIERED_SEED_12K else None
            )
            hot = votes.topk(hot_n).indices
        ready = entry["ready_event"]
        ready.record(side)
    entry["free_event"] = ready
    _HOT_CARRY[key] = (
        (hot, nv, ready, min_index, strong)
        if TIERED_SEED_12K
        else (hot, nv, ready, min_index)
    )


def _prep_bufs(Q, nb, dev):
    """Caller-owned seed_prep outputs. Reusing these kills the 0.1-0.5GB
    per-call alloc churn that forced the CUDA allocator into pathological
    behavior at small Q (256K: 4.5ms without an event-paced probe)."""
    key = (str(dev), nb)
    b = _PREP_BUFS.get(key)
    if b is None or b["q"] < Q:
        qm = max(Q, 1024)
        b = {
            "q": qm,
            "o": torch.empty(qm, device=dev),
            "inv": torch.empty(qm, device=dev),
            "th": torch.empty(qm, dtype=torch.int32, device=dev),
            "bc": torch.empty(qm, nb, dtype=torch.int32, device=dev),
            "cc": torch.empty(qm, dtype=torch.int32, device=dev),
            "status": torch.empty(qm, dtype=torch.int32, device=dev),
        }
        _PREP_BUFS[key] = b
    return b


def prepare_permuted_gather(
    kv_cache,
    dst_k,
    dst_scale,
    block_table,
    *,
    sequence_length,
    query_length,
    num_reqs,
    common_end,
    window_start,
    hot_key,
    qualification_sequence_length=None,
):
    """Pair-swap HOT12288 into the paged-gather prefix on the main stream."""
    try:
        length_metadata = _permuted_plan_length_metadata(
            sequence_length, qualification_sequence_length
        )
        S = length_metadata["sequence_length"]
        qualification_S = length_metadata["qualification_sequence_length"]
        Q = int(query_length)
        ks = int(window_start)
        common_end = int(common_end)
        use_fp4 = dst_k.dtype == torch.uint8
        if (
            not ENABLED
            or int(num_reqs) != 1
            or not (production_min_s(use_fp4) <= qualification_S <= PRODUCTION_MAX_S)
            or qualification_S < S
            or not supports_fused_query_len(Q, use_fp4=use_fp4)
            or hot_key is None
            or ks < 0
            or ks % 4 != 0
            or ks + HOT_PREFIX > common_end
            or common_end > S
            or dst_scale.shape != (S, 4)
            or dst_scale.dtype != torch.uint8
            # fp8 rows are 128 fp8 bytes + one packed fp32 scale; fp4 rows are
            # 64 packed e2m1 bytes + 4 ue8m0 scale bytes (same (S, 4) uint8).
            or (tuple(dst_k.shape), dst_k.dtype)
            not in (
                ((S, 128), torch.float8_e4m3fn),
                ((S, 64), torch.uint8),
            )
            or dst_k.device.type != "cuda"
            or torch.cuda.get_device_capability(dst_k.device)[0] != 10
        ):
            return None
        carry = _HOT_CARRY.get((str(dst_k.device), hot_key))
        expected_carry_len = 5 if TIERED_SEED_12K else 4
        carry_valid = not (
            carry is None
            or len(carry) != expected_carry_len
            or carry[1] > common_end
            or carry[3] < ks
            or carry[0].dim() != 1
            or carry[0].numel() < HOT_PREFIX
            or (
                TIERED_SEED_12K
                and (
                    carry[4] is None
                    or carry[4].dim() != 1
                    or carry[4].numel() < _PAGED_CANDIDATE_INLINE
                )
            )
        )
        if not carry_valid:
            return None
        ext = _ext()
        required_planner_ops = (
            ("plan_tiered_and_permuted_paged_gather_out",)
            if TIERED_SEED_12K
            else ("plan_and_permuted_paged_gather_out",)
        )
        if ext is None or not all(hasattr(ext, name) for name in required_planner_ops):
            return None
        hot = carry[0][:HOT_PREFIX]
        strong = carry[4][:_PAGED_CANDIDATE_INLINE] if TIERED_SEED_12K else None
        carry_event = carry[2]
        if carry_event is not None:
            carry_event.wait()
        hot.record_stream(torch.cuda.current_stream(dst_k.device))
        if strong is not None:
            strong.record_stream(torch.cuda.current_stream(dst_k.device))
        state, epoch = _pair_plan_bufs(S, dst_k.device)
        permutation = state["permutation"][:S]
        if TIERED_SEED_12K:
            ext.plan_tiered_and_permuted_paged_gather_out(
                hot,
                strong,
                state["hot_epoch"][:S],
                permutation,
                state["swap_a"],
                state["swap_b"],
                state["counts"],
                state["tier_swap_a"],
                state["tier_swap_b"],
                state["tier_counts"],
                ks,
                common_end,
                epoch,
                kv_cache,
                dst_k.view(torch.uint8),
                dst_scale,
                block_table,
            )
        else:
            ext.plan_and_permuted_paged_gather_out(
                hot,
                state["hot_epoch"][:S],
                permutation,
                state["swap_a"],
                state["swap_b"],
                state["counts"],
                ks,
                common_end,
                epoch,
                kv_cache,
                dst_k.view(torch.uint8),
                dst_scale,
                block_table,
            )
        return {
            "permutation": permutation,
            "carry_event": carry_event,
            "sequence_length": length_metadata["sequence_length"],
            "query_length": Q,
            "window_start": ks,
            "common_end": common_end,
            # CP-interleaved DSV4 ranks can differ by a few compressed KV
            # records at one scheduler-global threshold.  This is threshold
            # metadata only; all buffer shapes and scan extents remain S.
            "qualification_sequence_length": length_metadata[
                "qualification_sequence_length"
            ],
            # Keep the grow-only planner allocation alive until the scan has
            # consumed every asynchronously-produced map entry.
            "planner_state": state,
        }
    except Exception as e:  # noqa: BLE001
        print(
            f"[litetopk] exact-once permuted gather declined: {e}",
            flush=True,
        )
        return None


def stash_carry(hot_key, idx, S, min_index=0, *, recent_rows_hint=None):
    """Seed a layer's hot carry from the OFFICIAL path's topk output, called
    by the container on the LAST official chunk before MIN_S. The
    official->ours boundary is deterministic, so this one seed is all the
    first ours-chunk needs to run HOT (no cold start, no cold prefix). Stored
    compressed (voted hot columns, ~64KB/layer).

    The vote+topk selection and the store run on a per-device SIDE STREAM
    (async): seeding overlaps the model forward instead of stalling the
    official path. The exact-once gather consumer waits on the stored event
    before touching the carry."""
    if hot_key is None:
        return
    dev = idx.device
    nv = int(S)
    min_index = int(min_index)
    if nv - min_index < HOT_PREFIX:
        return
    # max_tokens=1 can finish directly from the final prefill logits, without
    # ever entering the no-prefill/decode branch that normally retires this
    # state.  Across prefill steps a layer's carry extent is strictly
    # increasing, so a *decrease* identifies a new request.  Equal extents are
    # expected when the 2-GiB logits budget splits one prefill step into several
    # internal Q chunks; treating equality as a reset drops the carry already
    # published by the other layers.  The first layer that observes a decrease
    # clears every old per-device planner/carry, and the remaining layers build
    # fresh state.
    previous = _HOT_CARRY.get((str(dev), hot_key))
    if previous is not None and len(previous) >= 2 and int(previous[1]) > nv:
        _retire_request_state(dev, release_scratch=RELEASE_SCRATCH_ON_ROLLBACK)
    # The caller reuses one persistent output tensor across layers.  The next
    # chunk is best predicted by every winner from the most recent query
    # window, so snapshot only that window before the async reader starts.
    # Besides matching the steady-state fused publisher, this cuts the
    # dense->fused boundary copy from Q*K to min(Q,1536)*K indices.
    recent_budget = (
        CARRY_RECENT_ROWS if recent_rows_hint is None else int(recent_rows_hint)
    )
    if recent_budget <= 0:
        raise ValueError(f"recent_rows_hint must be positive, got {recent_budget}")
    recent_rows = min(int(idx.shape[0]), recent_budget)
    idx_snapshot = idx[-recent_rows:].clone()
    ss = _HOT_STREAM.get(str(dev))
    if ss is None:
        ss = torch.cuda.Stream(device=dev)
        _HOT_STREAM[str(dev)] = ss
    carry_ext = _ext()
    ss.wait_stream(torch.cuda.current_stream())  # see the just-written topk
    idx_snapshot.record_stream(ss)  # keep it alive for the read
    with torch.cuda.stream(ss):
        votes = _vote_hist(nv, dev)
        hpf = idx_snapshot.reshape(-1).long().clamp_(0, nv - 1)
        votes.scatter_add_(0, hpf, torch.ones_like(hpf, dtype=torch.int32))
        hot_n = HOT_PREFIX
        # Each row contains every winner at most once, so recent_rows is the
        # exact per-index vote upper bound used by the custom carry selector.
        max_vote = _globalize_cp_carry_votes(votes, recent_rows)
        use_custom = (
            carry_ext is not None
            and hasattr(carry_ext, "carry_votes_topk_reset_")
            and (not TIERED_SEED_12K or hasattr(carry_ext, "carry_votes_topk_noreset_"))
            and nv <= PRODUCTION_MAX_S
            and 0 < max_vote <= 8192
        )
        if use_custom:
            hot = torch.empty(hot_n, dtype=torch.int64, device=dev)
            workspace = _carry_topk_workspace(max_vote, dev)
            strong = None
            if TIERED_SEED_12K:
                strong = torch.empty(
                    _PAGED_CANDIDATE_INLINE,
                    dtype=torch.int64,
                    device=dev,
                )
                carry_ext.carry_votes_topk_noreset_(
                    votes,
                    strong,
                    workspace["partial"],
                    workspace["state"],
                    _PAGED_CANDIDATE_INLINE,
                    max_vote,
                    min_index,
                )
            carry_ext.carry_votes_topk_reset_(
                votes,
                hot,
                workspace["partial"],
                workspace["state"],
                hot_n,
                max_vote,
                min_index,
            )
        else:
            if min_index > 0:
                votes[:min_index].fill_(torch.iinfo(torch.int32).min)
            strong = (
                votes.topk(_PAGED_CANDIDATE_INLINE).indices if TIERED_SEED_12K else None
            )
            hot = votes.topk(hot_n).indices
        ev = torch.cuda.Event()
        ev.record()
    _HOT_CARRY[(str(dev), hot_key)] = (
        (hot, nv, ev, min_index, strong)
        if TIERED_SEED_12K
        else (hot, nv, ev, min_index)
    )


def try_large_exact_once_chunk(
    q,
    k,
    k_scale,
    weights,
    ks,
    ke,
    out_idx,
    topk,
    *,
    permuted_plan,
    num_reqs,
    ke_min_hint,
    hot_key,
    ks_common_hint,
    carry_extent_hint,
    carry_recent_rows_hint=None,
    q_sf=None,
):
    """Run the fixed-HOT producer with one exact-once physical scan.

    The paged gather has pair-swapped the carried HOT set into one physical
    prefix. Normally seed prep emits either HOT12288 or the paged arena's
    HOT8192 inline tier and the single no-hist producer starts after that seed.
    Fixed-logical calibration emits nothing and instead scans the whole legal
    physical interval. Selection emits physical winners, then maps only those
    winners back to corpus order while accumulating carry votes. No scan-time
    threshold update or checkpoint remains.
    """
    global _HINTS_VALIDATED, _SINGLE_SCAN_LOGGED
    global _PENDING_OVF, _PROBE_RES, _CAND_ACC
    global _DEEP_GEMM_ACCEPTS_OUT
    try:
        Q = int(q.shape[0])
        S = int(k.shape[0])
        qualification_S = (
            int(permuted_plan.get("qualification_sequence_length", S))
            if isinstance(permuted_plan, dict)
            else S
        )
        prefix_base = int(ks_common_hint)
        common_end = int(ke_min_hint)
        cap_eff = MERGE_CAP
        # fp4 operands: q/k are packed e2m1 (64 bytes per 128-dim row, uint8
        # or deep_gemm's int8 tag view) with int32 ue8m0 scale streams; the
        # presence of q_sf selects the fp4graft scan.
        use_fp4 = q_sf is not None
        use_paged_candidates = (
            PAGED_CANDIDATES
            and not use_fp4
            and q.dim() == 3
            and int(q.shape[1]) == 32
            and topk == 2048
            and NB == 256
            and S > _PAGED_CANDIDATE_INLINE
        )
        min_s = production_min_s(use_fp4)
        packed_dim = 64 if use_fp4 else 128
        packed_dtypes = (torch.uint8, torch.int8) if use_fp4 else (torch.float8_e4m3fn,)
        if (
            not isinstance(permuted_plan, dict)
            or int(num_reqs) != 1
            or not supports_fused_query_len(Q, use_fp4=use_fp4)
            # FP8 production is exclusively the monolithic paged CP8 route;
            # PR #32094 remains the fallback for other FP8 shapes.
            or (not use_fp4 and not use_paged_candidates)
            or min_s > qualification_S
            or qualification_S < S
            or qualification_S > PRODUCTION_MAX_S
            or S > PRODUCTION_MAX_S
            or q.dim() != 3
            or tuple(q.shape[1:]) not in ((32, packed_dim), (64, packed_dim))
            or q.dtype not in packed_dtypes
            or tuple(k.shape) != (S, packed_dim)
            or k.dtype not in packed_dtypes
            or tuple(k_scale.shape) != (S,)
            or k_scale.dtype != (torch.int32 if use_fp4 else torch.float32)
            or weights.shape != (Q, int(q.shape[1]))
            or ks.shape != (Q,)
            or ke.shape != (Q,)
            or ks.dtype != torch.int32
            or ke.dtype != torch.int32
            or out_idx.shape != (Q, topk)
            or out_idx.dtype != torch.int32
            or topk <= 0
            or topk > 2048
            or (not use_paged_candidates and cap_eff < max(16384, 32 * topk))
            or prefix_base < 0
            or prefix_base % 4 != 0
            or prefix_base + HOT_PREFIX > common_end
            or common_end > S
            or int(permuted_plan.get("sequence_length", -1)) != S
            or int(permuted_plan.get("query_length", -1)) != Q
            or int(permuted_plan.get("window_start", -1)) != prefix_base
            or int(permuted_plan.get("common_end", -1)) != common_end
        ):
            return False
        permutation = permuted_plan.get("permutation")
        if (
            not isinstance(permutation, torch.Tensor)
            or permutation.shape != (S,)
            or permutation.dtype != torch.int32
            or permutation.device != q.device
            or not permutation.is_contiguous()
        ):
            return False
        if not (k.is_contiguous() and k_scale.is_contiguous()):
            return False
        kv_sf_ext = None
        if use_fp4:
            if (
                not isinstance(q_sf, torch.Tensor)
                or q_sf.dtype != torch.int32
                or tuple(q_sf.shape) != (Q, int(q.shape[1]))
                or q_sf.device != q.device
            ):
                return False
            if not q_sf.is_contiguous():
                q_sf = q_sf.contiguous()
            # The fp4graft TMA descriptor declares a 4-aligned kv_sf extent;
            # widen the view inside the backing storage when S % 4 != 0.
            sf_aligned = (S + 3) & ~3
            kv_sf_ext = k_scale
            if kv_sf_ext.numel() < sf_aligned:
                storage_i32 = (
                    kv_sf_ext.untyped_storage().nbytes() // 4
                    - kv_sf_ext.storage_offset()
                )
                if storage_i32 < sf_aligned:
                    return False
                kv_sf_ext = kv_sf_ext.as_strided((sf_aligned,), (1,))
        if not q.is_contiguous():
            q = q.contiguous()
        if weights.dtype != torch.float32:
            weights = weights.float()
        if not weights.is_contiguous():
            weights = weights.contiguous()
        if not (ks.is_contiguous() and ke.is_contiguous() and out_idx.is_contiguous()):
            return False
        if not _HINTS_VALIDATED:
            real_ks_min = int(ks.min().item())
            real_ks_max = int(ks.max().item())
            real_ke_min = int(ke.min().item())
            assert real_ks_min == real_ks_max == prefix_base
            assert real_ke_min == common_end
            _HINTS_VALIDATED = True
            print(
                "[litetopk] CPU hints validated; sync-free path active",
                flush=True,
            )

        _deferred_overflow_poll()
        ext = _ext()
        scan_op = (
            "mqa_logits_dsa_static_hot_nohist_paged_litetopk_"
            if use_paged_candidates
            else "mqa_logits_dsa_static_hot_nohist_fp4graft_litetopk_"
        )
        required_ops = (
            "seed_prep_litetopk_",
            scan_op,
            "map_topk_vote_stats_litetopk_",
            "cand_count_stats_litetopk_",
        )
        required_ops += (
            ("h2048_safe_topk_out_paged_litetopk_",)
            if use_paged_candidates
            else (
                "finalize_static_hot_meta_litetopk_",
                "compact_topk_min_thr_inplace_idx_out_litetopk",
            )
        )
        global _OPS_VERIFIED
        if ext is None:
            return False
        if not isinstance(_OPS_VERIFIED, dict):
            _OPS_VERIFIED = {}
        ops_key = scan_op
        if ops_key not in _OPS_VERIFIED:
            _OPS_VERIFIED[ops_key] = all(hasattr(ext, name) for name in required_ops)
        if not _OPS_VERIFIED[ops_key]:
            return False
        import deep_gemm

        seed_hot = _PAGED_CANDIDATE_INLINE if use_paged_candidates else HOT_PREFIX
        seed_cap = seed_hot if use_paged_candidates else cap_eff
        prefix_end = prefix_base + seed_hot
        prefix_k = k[prefix_base:prefix_end]
        prefix_scale = k_scale[prefix_base:prefix_end]
        sample_start, sample_end = _ks0_keh(Q, seed_hot, q.device)
        if use_fp4:
            seed_q = (q.view(torch.int8), q_sf)
            seed_k = (prefix_k.view(torch.int8), prefix_scale)
        else:
            seed_q = (q, None)
            seed_k = (prefix_k, prefix_scale)
        seed_args = (
            seed_q,
            seed_k,
            weights,
            sample_start,
            sample_end,
        )
        if _DEEP_GEMM_ACCEPTS_OUT is not False:
            try:
                sample_logits = deep_gemm.fp8_fp4_mqa_logits(
                    *seed_args,
                    clean_logits=False,
                    out=_slog_slab(Q, seed_hot, q.device),
                )
                _DEEP_GEMM_ACCEPTS_OUT = True
            except TypeError as exc:
                if "unexpected keyword argument 'out'" not in str(exc):
                    raise
                _DEEP_GEMM_ACCEPTS_OUT = False
                sample_logits = deep_gemm.fp8_fp4_mqa_logits(
                    *seed_args,
                    clean_logits=False,
                )
        else:
            sample_logits = deep_gemm.fp8_fp4_mqa_logits(
                *seed_args,
                clean_logits=False,
            )
        b = _prep_bufs(Q, NB, q.device)
        origin = b["o"][:Q]
        inv = b["inv"][:Q]
        threshold = b["th"][:Q]
        boundary_meta = b["bc"][:Q]
        candidate_count = b["cc"][:Q]
        status = b["status"][:Q]
        paged_cb = None
        if use_paged_candidates:
            paged_cb = _paged_cand_bufs(Q, S, q.device)
            candidate_value = paged_cb["inline_val"]
            candidate_index = paged_cb["inline_idx"]
        else:
            cb = _cand_bufs(Q, cap_eff, q.device)
            candidate_value = cb["cv"][:Q]
            candidate_index = cb["ci"][:Q]

        headroom_eff = HEADROOM
        if headroom_eff < 0.0:
            raise ValueError(f"headroom must be non-negative, got {headroom_eff}")

        # In exact-once mode the historical probe_stride argument is the
        # physical prefix base used for emitted candidate indices.
        ext.seed_prep_litetopk_(
            sample_logits,
            NB,
            topk,
            seed_cap,
            seed_hot,
            headroom_eff,
            prefix_base,
            1,
            origin,
            inv,
            threshold,
            boundary_meta,
            candidate_value,
            candidate_index,
            candidate_count,
        )
        del sample_logits

        # All rows share the physical prefix.  Reuse the immutable cached
        # filled tensor instead of launching an add kernel in every layer.
        suffix_start = _ks0_keh(Q, prefix_end, q.device)[1]
        if use_fp4:
            ext.mqa_logits_dsa_static_hot_nohist_fp4graft_litetopk_(
                q.view(torch.uint8),
                q_sf,
                k.view(torch.uint8),
                kv_sf_ext,
                weights,
                suffix_start,
                ke,
                origin,
                inv,
                threshold,
                candidate_value,
                candidate_index,
                candidate_count,
                boundary_meta,
                NB,
                topk,
            )
        else:
            ext.mqa_logits_dsa_static_hot_nohist_paged_litetopk_(
                q,
                k,
                k_scale,
                weights,
                suffix_start,
                ke,
                origin,
                inv,
                threshold,
                candidate_value,
                candidate_index,
                paged_cb["overflow_val"],
                paged_cb["overflow_idx"],
                paged_cb["page_table"],
                paged_cb["pool_head"],
                paged_cb["pool_status"],
                candidate_count,
                boundary_meta,
                S,
                NB,
                topk,
            )
        if not use_paged_candidates:
            # Compatibility path for LongCat's K=1008, DSV4's K=512, and
            # non-default CAP/NB. It retains the existing certificate and
            # destructive selector.
            ext.finalize_static_hot_meta_litetopk_(
                candidate_value,
                candidate_index,
                candidate_count,
                threshold,
                boundary_meta,
                status,
                NB,
                topk,
                S,
            )
        _TELEMETRY["calls"] += 1

        carry_recent_budget = (
            CARRY_RECENT_ROWS
            if carry_recent_rows_hint is None
            else int(carry_recent_rows_hint)
        )
        if carry_recent_budget <= 0:
            raise ValueError(
                f"carry_recent_rows_hint must be positive, got {carry_recent_budget}"
            )
        carry_recent_rows = min(Q, carry_recent_budget)
        carry_event = permuted_plan.get("carry_event")
        carry_nv = int(carry_extent_hint)
        carry_votes = _carry_vote_hist(carry_nv, q.device, hot_key, carry_event)
        if use_paged_candidates:
            ext.h2048_safe_topk_out_paged_litetopk_(
                candidate_value,
                candidate_index,
                paged_cb["overflow_val"],
                paged_cb["overflow_idx"],
                paged_cb["page_table"],
                candidate_count,
                paged_cb["pool_status"],
                out_idx,
                status,
                boundary_meta,
                S,
                S,
            )
        else:
            ext.compact_topk_min_thr_inplace_idx_out_litetopk(
                candidate_value,
                candidate_index,
                candidate_count,
                threshold,
                boundary_meta,
                NB,
                topk,
                out_idx,
                candidate_count[:0],
                1,
            )
        if _CAND_ACC is None or _CAND_ACC[0].device != q.device:
            _CAND_ACC = (
                torch.zeros(1, dtype=torch.int32, device=q.device),
                torch.zeros(1, dtype=torch.int32, device=q.device),
            )
        run_max, over_events = _CAND_ACC
        ext.map_topk_vote_stats_litetopk_(
            out_idx,
            permutation,
            status,
            carry_votes,
            carry_recent_rows,
            candidate_count,
            run_max,
            over_events,
            OVF_WATERMARK,
        )
        if _PENDING_OVF is None and (
            (_TELEMETRY["calls"] % PROBE_EVERY) == 0 or _TELEMETRY["calls"] == 1
        ):
            # Armed after the selector so the probe also carries its status:
            # a nonzero max fails the run closed at the next poll instead of
            # letting bad-count garbage reach the attention gather.
            if _PROBE_RES is None or _PROBE_RES[0].device != q.device:
                _PROBE_RES = (
                    torch.empty(2, dtype=torch.int32, device=q.device),
                    torch.empty(1, dtype=torch.int32, device=q.device),
                    torch.empty(5, dtype=torch.int32, pin_memory=True),
                    torch.cuda.Event(),
                )
            stats, smax, pinned, event = _PROBE_RES
            ext.cand_count_stats_litetopk_(candidate_count, stats)
            torch.amax(status, dim=0, keepdim=True, out=smax)
            pinned[:2].copy_(stats, non_blocking=True)
            pinned[2:3].copy_(smax, non_blocking=True)
            pinned[3:4].copy_(run_max, non_blocking=True)
            pinned[4:5].copy_(over_events, non_blocking=True)
            event.record()
            _PENDING_OVF = (
                event,
                pinned,
                S if use_paged_candidates else cap_eff,
                topk,
                OVF_WATERMARK,
                use_paged_candidates,
            )
        _publish_carry(
            hot_key,
            carry_votes,
            carry_nv,
            prefix_base,
            carry_recent_rows,
        )
        if not _SINGLE_SCAN_LOGGED:
            if use_paged_candidates:
                if TIERED_SEED_12K:
                    seed_layout = "strongest HOT8192 seed + next-HOT4096 early refresh"
                else:
                    seed_layout = "HOT8192 seed"
                print(
                    f"[litetopk] HOT{HOT_PREFIX} exact-once active: "
                    f"HOT{HOT_PREFIX} carry + {seed_layout} + "
                    "one ring-paged suffix scan + "
                    "page-aware physical select + winner-only map/vote",
                    flush=True,
                )
            else:
                print(
                    "[litetopk] HOT12288 exact-once active: prefix emit + "
                    "one fixed-threshold histogram-free suffix scan + "
                    "physical select + winner-only map/vote",
                    flush=True,
                )
            _SINGLE_SCAN_LOGGED = True
        return True
    except Exception as e:  # noqa: BLE001
        print(
            f"[litetopk] large exact-once declined: {e}",
            flush=True,
        )
        return False
