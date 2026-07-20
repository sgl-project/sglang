from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_server_args
from sglang.srt.utils import is_cuda

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator


# v4 (full one-shot) is out-of-place and drops the exit barrier, so it needs a
# double-buffered input. We carve three regions out of the tail of the enlarged
# comm.buffer -- two rotating input buffers (A/B) + one output -- so a region is
# only reused two ARs later, separated by the intervening AR's entry barrier
# (capture-safe: the A/B alternation bakes into the graph). Region size covers a
# few decode rows at hidden=6144; v4 only fires for num_tokens <= 2.
#
# SAFETY INVARIANT: the reuse-distance-2 argument requires the v4 AR sequence to
# alternate A,B,A,B *globally* -- across forwards, and across graph replays. That
# holds because every forward issues an even number of v4 ARs (attn + MLP per
# layer), so each captured graph starts and ends on opposite regions and replays
# stay aligned with each other and with eager forwards. If a forward could ever
# issue an ODD number of v4 ARs (e.g. a layer taking a reduce-scatter path at
# num_tokens <= 2), a replay boundary would put the same region in consecutive
# ARs and a lagging peer could still be multicast-reading it -- audit this before
# changing which layers all-reduce at decode.
_INKLING_AR_V4_REGION = 16 * 6144  # elems; 16B-aligned (mult of 8)

# v5 (push one-shot) needs a per-rank staging slot on every GPU: two rotating
# staging buffers of world*_INKLING_AR_V5_REGION elems (A/B, same reuse-distance-2
# argument and SAFETY INVARIANT as v4 above -- v5's single barrier plays the
# entry barrier's role) plus one local output region. Sized to cover the
# custom-kernel decode band (<=96 rows at hidden=6144) plus the fused
# target-verify chain (bs*draft_token_num <= 144 rows).
_INKLING_AR_V5_REGION = 160 * 6144  # elems; 16B-aligned (mult of 8)

# The custom kernels reduce one 16B vector (8 bf16 elems) at a time and their
# validate() rejects a num_items that isn't a multiple of this. torch symm-mem
# only enforces 4B alignment, so a non-vector bf16 size (e.g. [1, 2]) must fall
# back to torch multimem instead of hitting the kernel's hard check. (For Inkling
# proper this never bites -- hidden=6144 is a multiple of 8 -- but the utility
# is general.)
_INKLING_AR_VEC = 8

# Fused {AR + scattered sconv} OUT region (elems): the extend kernel broadcasts
# the post-conv [T, H] here (out-of-place -- the conv taps re-read the pristine
# input partials). ONE region suffices (no A/B): the kernel keeps BOTH
# barriers, so the next fused call's ENTRY barrier proves every peer's
# consumers already read the previous OUT. Sized to the chunked-prefill
# ceiling at hidden=6144.
_INKLING_AR_SSCONV_OUT_REGION = 16384 * 6144  # max_prefill_tokens x hidden

# World sizes with torch-multimem NVLink support; the symm-mem fast path (and
# with it the custom kernels) is only taken for these.
_INKLING_AR_WORLD_SIZES = (4, 6, 8)


class _InklingArResources(msgspec.Struct):
    """Per-group custom-AR resources: barrier flags/state + comm.buffer peer and
    multicast pointers + v4 double-buffer region offsets + rotation index."""

    rank: int
    world: int
    buffer_ptrs_dev: int
    multicast_ptr: int
    flag_ptrs_dev: int
    state_ptr: int
    v4_in: tuple[int, int]  # (A, B) input region starts (elems)
    v4_out: int  # output region start (elems)
    v5_in: tuple[int, int]  # (A, B) push-staging region starts (elems)
    v5_out: int  # v5 output region start (elems)
    ssconv_out: int  # fused {AR + scattered sconv} OUT region start (elems)
    v4_cur: int = 0  # rotation index, flips per v4 AR
    v5_cur: int = 0  # rotation index, flips per v5 AR
    refs: tuple = ()  # keep-alive: (flags, state, hdl, hflags)


# Lazily-built per-group resources, keyed by group name. Built once on the
# first eager call (before any capture). comm.buffer itself is enlarged at
# communicator init (a normal, non-inference tensor) so producer GEMMs can
# write into it -- including v4's input regions.
_INKLING_AR_CACHE: dict[str, _InklingArResources] = {}


@functools.cache
def _ar_jit():
    """The inkling_all_reduce JIT wrapper module, imported once on first use (kept
    lazy so importing comm.py doesn't pull in the JIT machinery)."""
    if not is_cuda():
        return None
    from sglang.jit_kernel import inkling_all_reduce

    return inkling_all_reduce


@functools.cache
def _ar_fused_jit():
    if not is_cuda():
        return None
    from sglang.jit_kernel import inkling_ar_fused

    return inkling_ar_fused


def _get_inkling_ar_resources(comm) -> _InklingArResources | None:
    """Return the cached custom-AR resources for ``comm``, or ``None`` if they
    can't be built now (a CUDA-graph capture is active, or on ROCm). The first eager call
    populates the cache before capture."""
    key = comm.group.group_name
    cached = _INKLING_AR_CACHE.get(key)
    if cached is not None:
        return cached
    if torch.cuda.is_current_stream_capturing():
        return None
    if not is_cuda():
        return None
    import torch.distributed._symmetric_memory as torch_symm_mem

    jit = _ar_jit()
    world = comm.world_size
    # The JIT kernels static_assert a power-of-two world (std::has_single_bit).
    # TP=6 is torch-multimem-eligible but would trip that assertion at compile
    # time; return None so it stays on the plain-multimem fallback path.
    if world & (world - 1) != 0:
        return None
    dev = comm.buffer.device
    hdl = torch_symm_mem.rendezvous(comm.buffer, key)  # idempotent (done at init)
    flags = torch_symm_mem.empty(jit.flags_numel(world), device=dev, dtype=torch.uint32)
    flags.zero_()
    hflags = torch_symm_mem.rendezvous(flags, key)
    # Device-side barrier so no peer's first fused-AR kernel can write an epoch
    # into our flags while our zero_ is still pending on the stream (the zero
    # would clobber the signal; the protocol self-heals, but don't rely on it).
    hflags.barrier()
    state = torch.zeros(jit.STATE_SIZE, device=dev, dtype=torch.uint32)
    jit.compile_inkling_all_reduce(comm.dtype, world)
    # v4 + v5 regions at the tail of comm.buffer. A huge in-place v3 AR's [0:n]
    # may reach into these regions; that is safe -- every fused AR's entry
    # barrier proves all peers finished the previous AR before any broadcast
    # touches the buffer, and the staging regions hold no cross-AR state.
    total = comm.buffer.numel()
    v4reg = _INKLING_AR_V4_REGION
    v5stage = world * _INKLING_AR_V5_REGION
    v5_base = total - 3 * v4reg - 2 * v5stage - _INKLING_AR_V5_REGION
    # Fused scattered-sconv OUT sits below the v5 regions; eligibility caps the
    # input [0:n] at ssconv_out so the two never overlap.
    ssconv_out = v5_base - _INKLING_AR_SSCONV_OUT_REGION
    res = _InklingArResources(
        rank=hdl.rank,
        world=world,
        buffer_ptrs_dev=hdl.buffer_ptrs_dev,
        multicast_ptr=hdl.multicast_ptr,
        flag_ptrs_dev=hflags.buffer_ptrs_dev,
        state_ptr=state.data_ptr(),
        v4_in=(total - 3 * v4reg, total - 2 * v4reg),
        v4_out=total - v4reg,
        v5_in=(v5_base, v5_base + v5stage),
        v5_out=v5_base + 2 * v5stage,
        ssconv_out=ssconv_out,
        refs=(flags, state, hdl, hflags),
    )
    _INKLING_AR_CACHE[key] = res
    return res


def ensure_inkling_ar_resources(group: GroupCoordinator) -> None:
    """Eagerly build the custom-AR resources for ``group`` (idempotent).

    Call at model init: the lazy first-call build only works when an eager
    forward runs before CUDA-graph capture (historically guaranteed by the
    prefill BCG capture's eager breaks). With the prefill graph disabled and
    --skip-server-warmup, decode capture would otherwise see no resources and
    silently bake the non-custom fallback into the decode graphs."""
    comm = group.torch_symm_mem_comm
    if (
        comm is not None
        and not comm.disabled
        and group.world_size in _INKLING_AR_WORLD_SIZES
    ):
        _get_inkling_ar_resources(comm)


def _v4_enabled(comm, num_tokens: int) -> bool:
    if not is_cuda():
        return False
    return _ar_jit().select_ar_config(num_tokens, comm.world_size)[0] == "v4"


# Fused decode {MoE AR -> mlp_sconv -> attn_norm} band: bounded by the v5
# staging region size and by one per-block-barrier slot per token row.
_INKLING_AR_FUSED_MAX_TOKENS = 96
# Target-verify band: T = batch * draft_token_num (144 at bs=16, Q=9), bounded
# by the (enlarged) staging region rows and the per-block barrier slots.
_INKLING_AR_FUSED_MAX_TOKENS_VERIFY = 160


# --- fused-AR shared-expert partials hand-off -------------------------------
#
# InklingMoE.forward(reduce=False) produces {routed partials, shared partials}.
# Pre-adding them costs one full [T, H] kernel per MoE layer; the custom AR
# kernels can instead fold the shared term for free (v5-family register fold at
# the push). The partials
# tensor must stay a bare tensor across the layer boundary (BCG narrowing,
# logging, the AttnRes loop), so the shared tensor rides this per-forward
# stash: the producer deposits it, the consuming fused-AR call collects it.
# Python-trace-time only (works identically under CUDA-graph capture); the
# consumer ALWAYS drains it in the same model forward that stashed it.
_PENDING_AR_SHARED: list = []


def stash_ar_shared(shared: torch.Tensor) -> None:
    assert not _PENDING_AR_SHARED, "unconsumed fused-AR shared partials"
    _PENDING_AR_SHARED.append(shared)


def take_ar_shared(num_tokens: int) -> torch.Tensor | None:
    """Collect (and clear) the stashed shared partials, prefix-narrowed to the
    consumer's row count (BCG narrowing slices rows [0:t])."""
    if not _PENDING_AR_SHARED:
        return None
    shared = _PENDING_AR_SHARED.pop()
    if shared.shape[0] != num_tokens:
        shared = shared[:num_tokens]
    return shared


def ar_sconv_norm_fusable(
    group: GroupCoordinator,
    forward_batch,
    num_tokens: int,
    hidden: int,
    dtype: torch.dtype,
) -> bool:
    """True when a decode {all-reduce -> sconv -> add+RMSNorm} chain
    (attn-side: wo_ud AR -> attn_sconv -> mlp_norm; MoE-side: MoE AR ->
    mlp_sconv -> next attn_norm)
    can run as the single fused kernel (jit_kernel/inkling_ar_fused.py). Must be
    evaluated identically by the producing layer (MoE ``reduce=False``) and the
    consuming layer/tail -- it is a pure function of per-forward state."""
    if not is_cuda():
        return False
    if not (
        envs.SGLANG_OPT_USE_INKLING_CUSTOM_AR.get()
        and envs.SGLANG_OPT_USE_INKLING_FUSED_AR_SCONV_NORM.get()
    ):
        return False
    if get_server_args().enable_scattered_sconv:
        # The decode {AR -> sconv -> norm} fusion is full-width; under scattered
        # sconv the output sconvs are hidden-sharded, so it does not apply.
        return False
    fm = forward_batch.forward_mode
    if fm.is_decode():
        max_tokens = _INKLING_AR_FUSED_MAX_TOKENS
    elif fm.is_target_verify():
        max_tokens = _INKLING_AR_FUSED_MAX_TOKENS_VERIFY
    else:
        return False
    comm = group.torch_symm_mem_comm
    if (
        comm is None
        or comm.disabled
        or group.world_size not in _INKLING_AR_WORLD_SIZES
        or dtype != comm.dtype
    ):
        return False
    if (
        num_tokens > max_tokens
        or num_tokens > _ar_jit().MAX_BARRIER_BLOCKS
        or hidden % _INKLING_AR_VEC != 0
        or hidden // _INKLING_AR_VEC > 1024  # one 16B vec per thread, one block/row
        or num_tokens * hidden > _INKLING_AR_V5_REGION
    ):
        return False
    return _get_inkling_ar_resources(comm) is not None


def ar_sconv_norm_fused(
    input: torch.Tensor,
    residual: torch.Tensor,
    sconv,
    norm,
    forward_batch,
    group: GroupCoordinator,
    shared: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused decode {all-reduce -> sconv -> residual-add + RMSNorm}: one kernel
    replacing ``symm_mem_all_reduce`` + ``fused_causal_conv1d_update_decode`` +
    the fused-add RMSNorm. ``input`` holds the UNREDUCED MoE partial sums
    (``InklingMoE.forward(reduce=False)``); returns ``(hs, residual)`` exactly like
    the unfused ``sconv -> norm(hs, res)`` chain. The caller must have checked
    ``ar_sconv_norm_fusable``. Occupies one v5 staging rotation slot (this
    IS a v5 AR with the epilogue seam filled in; same reuse-distance rule)."""
    comm = group.torch_symm_mem_comm
    res = _get_inkling_ar_resources(comm)
    if shared is None:
        shared = take_ar_shared(input.shape[0])
    hs_out = torch.empty_like(input)
    residual_out = torch.empty_like(residual)
    cur = res.v5_cur
    stage_off = res.v5_in[cur]
    esz = comm.buffer.element_size()
    mc = res.multicast_ptr + stage_off * esz
    local = comm.buffer.data_ptr() + stage_off * esz
    if forward_batch.forward_mode.is_target_verify():
        sconv_cache, cache_indices, cache_mask, conv_weight, inter_out = (
            sconv.verify_fused_ar_inputs(forward_batch)
        )
        _ar_fused_jit().inkling_ar_sconv_norm_verify(
            input,
            residual,
            residual_out,
            hs_out,
            norm.weight,
            norm.variance_epsilon,
            sconv_cache,
            cache_indices,
            cache_mask,
            conv_weight,
            inter_out,
            forward_batch.spec_info.draft_token_num,
            mc,
            local,
            res.flag_ptrs_dev,
            res.state_ptr,
            res.rank,
            res.world,
            activation=sconv.activation,
            use_residual=sconv.use_residual,
            shared=shared,
        )
    else:
        sconv_cache, cache_indices, cache_mask, conv_weight = (
            sconv.decode_fused_ar_inputs(forward_batch)
        )
        _ar_fused_jit().inkling_ar_sconv_norm(
            input,
            residual,
            residual_out,
            hs_out,
            norm.weight,
            norm.variance_epsilon,
            sconv_cache,
            cache_indices,
            cache_mask,
            conv_weight,
            mc,
            local,
            res.flag_ptrs_dev,
            res.state_ptr,
            res.rank,
            res.world,
            activation=sconv.activation,
            use_residual=sconv.use_residual,
            track_mask=forward_batch.mamba_track_mask,
            track_indices=forward_batch.mamba_track_indices,
            shared=shared,
        )
    res.v5_cur = 1 - cur
    return hs_out, residual_out


def get_ar_buffer(
    group: GroupCoordinator,
    num_tokens: int,
    hidden: int,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Return a ``[num_tokens, hidden]`` view of a rendezvous'd symm buffer for a
    producer to write into, or ``None`` when the symm-mem fast path is ineligible.

    With ``SGLANG_OPT_USE_INKLING_CUSTOM_AR`` the communicator buffer is enlarged at
    init (256 MiB) so big prefill ARs fit; for the v4 (num_tokens<=2) bucket this
    returns the current rotating input region so ``symm_mem_all_reduce`` can run
    the out-of-place full one-shot.
    """
    comm = group.torch_symm_mem_comm
    if (
        comm is None
        or comm.disabled
        or group.world_size not in _INKLING_AR_WORLD_SIZES
        or dtype != comm.dtype
    ):
        return None
    n = num_tokens * hidden
    nbytes = n * dtype.itemsize
    if nbytes % 4 != 0:
        return None
    if (
        envs.SGLANG_OPT_USE_INKLING_CUSTOM_AR.get()
        # Scattered sconv replaces the AR with reduce_scatter_hidden, which
        # stages from comm.buffer[:n] -- never hand out the v4 region there.
        and not get_server_args().enable_scattered_sconv
    ):
        res = _get_inkling_ar_resources(comm)
        if (
            res is not None
            and _v4_enabled(comm, num_tokens)
            and n <= _INKLING_AR_V4_REGION
            and n % _INKLING_AR_VEC == 0
        ):
            off = res.v4_in[res.v4_cur]
            return comm.buffer[off : off + n].view(num_tokens, hidden)
    if nbytes >= comm.max_size:
        return None
    return comm.buffer[:n].view(num_tokens, hidden)


def symm_mem_all_reduce(
    input: torch.Tensor,
    group: GroupCoordinator,
    *,
    output: torch.Tensor | None = None,
    input_is_ar_buffer: bool = False,
    num_sms: int = 32,
    shared: torch.Tensor | None = None,
) -> torch.Tensor:
    """All-reduce ``input`` across ``group`` in the communicator's symm buffer.

    Default (``--enable-torch-symm-mem``): one-shot NVLink ``multimem_all_reduce_``.
    With ``SGLANG_OPT_USE_INKLING_CUSTOM_AR``: dispatch on shape to the autotuned
    custom kernels -- v5 push one-shot (out-of-place, double-buffered staging)
    for the latency band, v3/v3b two-shot multimem for medium/large -- with
    torch multimem for the remaining small ("mm") bucket. The buffer is enlarged
    at init so large prefill ARs take this path instead of NCCL.

    ``shared``: optional LOCAL shared-expert partials to add into the reduced
    result. Folded IN-KERNEL on the v5 push (in registers) and the v4
    pre-barrier prologue -- the small/decode band; every other bucket and
    fallback PRE-ADDS during its stage-in copy, so passing ``shared`` is
    always numerics-identical to {torch.add -> all_reduce}.
    """
    _ = num_sms
    if group.world_size == 1:
        if shared is not None:
            input = input + shared
        if output is None:
            return input
        output.copy_(input)
        return output

    comm = group.torch_symm_mem_comm
    if (
        output is None
        and comm is not None
        and not comm.disabled
        and group.world_size in _INKLING_AR_WORLD_SIZES
        and comm.should_torch_symm_mem_allreduce(input)
    ):
        n = input.numel()
        num_tokens = input.shape[0] if input.dim() >= 2 else n
        res = (
            _get_inkling_ar_resources(comm)
            if envs.SGLANG_OPT_USE_INKLING_CUSTOM_AR.get()
            else None
        )
        # Custom kernels need a 16B-vector-multiple size (validate() enforces it);
        # a non-vector size falls through to plain multimem below. The "mm" bucket
        # inside this block also uses torch multimem, but it still requires the
        # kernel-eligible size to reach here, so gate the whole block on it.
        if res is not None and n % _INKLING_AR_VEC == 0:
            jit = _ar_jit()
            kernel, nb, bs = jit.select_ar_config(num_tokens, res.world)
            if (
                kernel == "v5"
                and n <= _INKLING_AR_V5_REGION
                and input.data_ptr() % 16 == 0
            ):
                # Push one-shot: multicast-push input into the rotating staging
                # buffer, one per-block barrier, local reduce into the out
                # region. Input is read locally, so it needs NO stage-in copy
                # even when it isn't an AR buffer.
                cur = res.v5_cur
                stage_off = res.v5_in[cur]
                out_view = comm.buffer[res.v5_out : res.v5_out + n]
                esz = comm.buffer.element_size()
                jit.inkling_multimem_push_oneshot(
                    input.view(-1),
                    out_view,
                    res.multicast_ptr + stage_off * esz,
                    comm.buffer.data_ptr() + stage_off * esz,
                    res.flag_ptrs_dev,
                    res.state_ptr,
                    res.rank,
                    res.world,
                    n,
                    nb,
                    bs,
                    per_block_barrier=True,
                    shared=shared.view(-1) if shared is not None else None,
                )
                res.v5_cur = 1 - cur
                return out_view.view(input.shape)
            if kernel == "v4" and n <= _INKLING_AR_V4_REGION:
                # out-of-place full one-shot in the double-buffered tail regions.
                cur = res.v4_cur
                in_off = res.v4_in[cur]
                out_off = res.v4_out
                in_view = comm.buffer[in_off : in_off + n]
                out_view = comm.buffer[out_off : out_off + n]
                v4_shared = shared
                if not input_is_ar_buffer:
                    if v4_shared is not None:
                        torch.add(input.view(-1), v4_shared.view(-1), out=in_view)
                        v4_shared = None
                    else:
                        in_view.copy_(input.view(-1))
                mc = res.multicast_ptr + in_off * comm.buffer.element_size()
                jit.inkling_multimem_full_oneshot(
                    in_view,
                    out_view,
                    mc,
                    res.flag_ptrs_dev,
                    res.state_ptr,
                    res.rank,
                    res.world,
                    n,
                    nb,
                    bs,
                    shared=v4_shared.view(-1) if v4_shared is not None else None,
                )
                res.v4_cur = 1 - cur
                return out_view.view(input.shape)

            buf = comm.buffer[:n]
            if not input_is_ar_buffer:
                if shared is not None:
                    torch.add(input.view(-1), shared.view(-1), out=buf)
                else:
                    buf.copy_(input.view(-1))
            elif shared is not None:
                buf.add_(shared.view(-1))
            if kernel in ("v3", "v3b"):
                jit.inkling_multimem_one_shot_fused(
                    buf,
                    res.multicast_ptr,
                    res.flag_ptrs_dev,
                    res.state_ptr,
                    res.rank,
                    res.world,
                    n,
                    nb,
                    bs,
                    per_block_barrier=(kernel == "v3b"),
                )
                return buf.view(input.shape)
            if kernel == "v2":
                jit.inkling_two_shot_all_reduce_fused(
                    buf,
                    res.buffer_ptrs_dev,
                    res.flag_ptrs_dev,
                    res.state_ptr,
                    res.rank,
                    res.world,
                    n,
                    nb,
                    bs,
                )
                return buf.view(input.shape)
            # "mm" bucket: torch multimem on comm.buffer.
            torch.ops.symm_mem.multimem_all_reduce_(buf, "sum", comm.group.group_name)
            return buf.view(input.shape)

        # flag off, resources unavailable (in capture / non-power-of-two world),
        # or a non-vector size: plain multimem.
        buf = comm.buffer[:n]
        if not input_is_ar_buffer:
            if shared is not None:
                torch.add(input.view(-1), shared.view(-1), out=buf)
            else:
                buf.copy_(input.view(-1))
        elif shared is not None:
            buf.add_(shared.view(-1))
        torch.ops.symm_mem.multimem_all_reduce_(buf, "sum", comm.group.group_name)
        return buf.view(input.shape)

    if shared is not None:
        input = input + shared
    result = group.all_reduce(input)
    if output is None:
        return result
    output.copy_(result)
    return output


# --- scattered sconv (--enable-scattered-sconv) comm helpers -----------------
#
# torch symmetric-memory multimem needs NVLink multicast, which torch supports
# for world sizes {4,6,8} on cc>=9. reduce_scatter/all_gather here reuse the
# group's TorchSymmMemCommunicator buffer -- already allocated and rendezvous'd
# at init, so these one-shot NVLink collectives are CUDA-graph-safe (no
# rendezvous in forward), same as symm_mem_all_reduce. Ineligible cases fall
# back to NCCL.


def _symm_mem_comm(group: GroupCoordinator, input: torch.Tensor, full_numel: int):
    """Return the group's torch-symm-mem communicator if it can multimem this
    tensor (dtype match, full [T, H] fits + 4B-aligned in the rendezvous'd
    buffer, supported world size); else None to signal the NCCL fallback."""
    comm = group.torch_symm_mem_comm
    if comm is None or comm.disabled:
        return None
    if not input.is_cuda or input.dtype != comm.dtype:
        return None
    if group.world_size not in _INKLING_AR_WORLD_SIZES:
        return None
    nbytes = full_numel * input.element_size()  # RS stages [T,H]; AG rebuilds [T,H]
    if nbytes % 4 != 0 or nbytes >= comm.max_size:
        return None
    return comm


def reduce_scatter_hidden(
    input: torch.Tensor,
    group: GroupCoordinator,
    *,
    input_is_ar_buffer: bool = False,
) -> torch.Tensor:
    """Reduce partial-sum [T, H] across the group, scatter hidden -> [T, H/P]."""
    p = group.world_size
    if p == 1:
        return input
    t, h = input.shape
    assert h % p == 0, f"hidden {h} not divisible by tp size {p}"

    comm = _symm_mem_comm(group, input, t * h)
    if comm is not None:
        # Stage [T,H] into the rendezvous'd symm buffer (no-op when the producer
        # already wrote it there via get_ar_buffer); multimem reduce-scatter over
        # the last (hidden) dim -> local [T, H/P] shard. split_last_dim=True
        # avoids the transpose the NCCL (dim-0) path needs.
        symm_in = comm.buffer[: t * h].view(t, h)
        if not input_is_ar_buffer:
            symm_in.copy_(input)
        out = torch.empty((t, h // p), dtype=input.dtype, device=input.device)
        torch.ops.symm_mem.reduce_scatter_out(symm_in, comm.group.group_name, True, out)
        return out

    # one transpose so reduce_scatter_tensor (dim-0) scatters the hidden dim;
    # tensor form uses the optimized backend, one copy instead of P chunk-copies.
    x = input.view(t, p, h // p).movedim(1, 0).reshape(p * t, h // p).contiguous()
    out = torch.empty((t, h // p), dtype=input.dtype, device=input.device)
    group.reduce_scatter_tensor(out, x)
    return out


def all_gather_hidden(input: torch.Tensor, group: GroupCoordinator) -> torch.Tensor:
    """Gather hidden shard [T, H/P] -> [T, H]; inverse of reduce_scatter_hidden."""
    p = group.world_size
    if p == 1:
        return input
    t, hp = input.shape

    comm = _symm_mem_comm(group, input, t * hp * p)
    if comm is not None:
        # multimem all-gather concatenates the P shards along dim 0 into the
        # rendezvous'd symm buffer ([P*T, H/P]); move rank to the middle and
        # flatten to reconstruct [T, H] (hidden chunks in rank order).
        symm_out = comm.buffer[: p * t * hp].view(p * t, hp)
        torch.ops.symm_mem.multimem_all_gather_out(
            input, comm.group.group_name, symm_out
        )
        return symm_out.view(p, t, hp).movedim(0, 1).reshape(t, p * hp)

    return group.all_gather(input, dim=-1)


@functools.cache
def _ar_ssconv_jit():
    if not is_cuda():
        return None
    from sglang.jit_kernel import inkling_ar_scattered_sconv

    return inkling_ar_scattered_sconv


def scattered_ar_sconv_fusable(
    group: GroupCoordinator,
    forward_batch,
    num_tokens: int,
    hidden: int,
    dtype: torch.dtype,
) -> bool:
    """True when an extend {reduce_scatter_hidden -> sconv(shard) ->
    all_gather_hidden} chain can run as the single fused v3/v3b-style kernel
    (jit_kernel/inkling_ar_scattered_sconv.py). Pure function of per-forward
    state -- the producing layer (reduce=False) and the consuming site must
    evaluate it identically."""
    if not is_cuda():
        return False
    if not (
        get_server_args().enable_scattered_sconv
        and envs.SGLANG_OPT_USE_INKLING_CUSTOM_AR.get()
        and envs.SGLANG_OPT_USE_INKLING_FUSED_AR_SCONV.get()
    ):
        return False
    fm = forward_batch.forward_mode
    if fm.is_draft_extend_v2():
        return False  # de-tied per-step metadata semantics; unfused chain
    if not (fm.is_extend() or fm.is_decode()):
        return False
    # Prefill scope: the BCG runner's eager-break sites are not wired (its
    # baked flags would disagree with the break bodies) and tc_piecewise's
    # FX pieces can't carry the cross-layer producer/consumer contract, so
    # both fall back to the unfused chain. The FULL prefill CUDA-graph
    # backend (context.full_graph -- the whole model captured uniformly) IS
    # supported: the kernel is capture-safe (barrier epochs advance across
    # replays; validated capture+replay) and all its metadata (qsl/si/
    # cache_mask/safe_idx/track rows) is recomputed in-graph from the
    # runner's refreshed registry slots. Bucket padding is contained: pad
    # rows only write pad rows of the OUT region (the eager tail slices
    # [:raw]), and sentinel request slots have qlen == 0 so the in-kernel
    # cache update/track skip them.
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        get_tc_piecewise_forward_context,
    )

    tc_ctx = get_tc_piecewise_forward_context()
    if tc_ctx is not None and not tc_ctx.full_graph:
        return False
    comm = group.torch_symm_mem_comm
    if (
        comm is None
        or comm.disabled
        or group.world_size not in (4, 8)  # kernel static_asserts power-of-two
        or dtype != comm.dtype
    ):
        return False
    n = num_tokens * hidden
    if num_tokens == 0 or hidden % (group.world_size * _INKLING_AR_VEC) != 0:
        return False
    res = _get_inkling_ar_resources(comm)
    return res is not None and n <= res.ssconv_out


def ar_scattered_sconv_fused(
    input: torch.Tensor,
    sconv,
    forward_batch,
    group: GroupCoordinator,
    norm=None,
    norm_residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Fused {AR + scattered sconv}: one v3/v3b-style two-shot multimem
    kernel replacing ``reduce_scatter_hidden`` + ``causal_conv1d(shard)`` +
    ``all_gather_hidden``. ``input`` holds the UNREDUCED producer partials
    ([T, H], ``reduce=False``). The caller must have checked
    ``scattered_ar_sconv_fusable``. The reduced pre-conv x shard is stashed
    locally and consumed in-kernel by the fused cache-update + prefix-cache
    track (Phase 3) -- there is no separate update/track kernel call.

    Without ``norm``: returns the gathered post-conv [T, H] (a view of the OUT
    symm region). With ``norm`` (an RMSNorm module) + ``norm_residual`` the
    add+RMSNorm tail is fused in-kernel too (both the chunked and streaming
    kernels carry the tail; validated at all bands) and the call returns
    ``(hidden, residual)`` exactly like ``ar_sconv_norm_fused``. The call
    sites only pass ``norm`` for decode/verify: at extend shapes the in-kernel
    tail is slower than the unfused sgl_kernel fused_add_rmsnorm -- the tail is
    bandwidth-bound and the AR grid is capped by barrier co-residency, while
    fusion only saves a launch. Decode prefix-cache tracking is fused (post-update window
    snapshot), so tracked decode batches are supported."""
    shared = take_ar_shared(input.shape[0])
    if shared is not None:
        # Pre-add on this path rather than folding in-kernel: pre-add keeps the
        # full-occupancy torch.add instead of the barrier-capped grid.
        # Add straight into the AR buffer -- the same single kernel the
        # producer used to run, and the stage-in copy below then no-ops.
        _buf = get_ar_buffer(group, input.shape[0], input.shape[1], input.dtype)
        if _buf is not None:
            torch.add(input, shared, out=_buf)
            input = _buf
        else:
            input = input + shared

    comm = group.torch_symm_mem_comm
    res = _get_inkling_ar_resources(comm)
    jit = _ar_ssconv_jit()
    t, h = input.shape
    n = t * h
    world = res.world
    hc = h // world

    # Design note: everything (cache, conv input/output, exchange) is sharded
    # along the hidden dim -- the pure column layout. Communication volume is
    # optimal (n/P in + n/P out per rank, no window staging) and each rank
    # convs only its own channels. The price, accepted by design: decode pays
    # a second cross-rank sync round (consumers need full rows, so the
    # post-conv shard exchange must publish). The one-shot decode and
    # banded-scattered variants (window-shard push) remain available in the
    # JIT module / bench harness as alternatives.

    buf = comm.buffer[:n].view(t, h)
    if input.data_ptr() != buf.data_ptr():
        buf.copy_(input)

    (
        sconv_cache,
        safe_idx,
        cache_mask,
        cu,
        si,
        weight,
        query_start_loc,
        cache_indices,
        has_initial_state,
        track_rows,
        track_mask,
        track_dst,
    ) = sconv.extend_fused_ar_inputs(forward_batch)
    del query_start_loc  # update + track are fused in-kernel
    fm = forward_batch.forward_mode
    is_verify = fm.is_target_verify()
    kernel_ci = cache_indices.to(torch.int32)  # kernel reads raw int32
    if is_verify:
        # Verify must NOT update the working cache: PAD every row so the
        # in-kernel phase-3 update skips (windows are saved separately below).
        kernel_ci = torch.full_like(cache_indices, -1)

    # Decode prefix-cache track: snapshot the post-update window in-kernel
    # (the unfused fused_causal_conv1d_update_decode semantics). The capture
    # batch always carries the persistent (all-False) mask buffer, so this
    # bakes correctly into decode graphs and stays data-dependent at replay.
    track_from_cache = False
    if fm.is_decode() and forward_batch.mamba_track_mask is not None:
        b = forward_batch.batch_size
        track_mask = forward_batch.mamba_track_mask[:b]
        track_dst = forward_batch.mamba_track_indices[:b]
        track_from_cache = True

    esz = comm.buffer.element_size()

    if norm is not None and fm.is_decode():
        # COLUMN DECODE V2: dedicated small-batch kernel -- one block per
        # token row, block-scoped two-round barriers, prefetch under the
        # entry spin, inline cache update/track, fused full-row norm.
        assert norm_residual is not None
        out_local = comm.buffer[res.ssconv_out : res.ssconv_out + n].view(t, h)
        hs_out = torch.empty_like(norm_residual)
        residual_out = torch.empty_like(norm_residual)
        jit.inkling_ar_col_decode(
            buf,
            out_local,
            norm_residual,
            residual_out,
            hs_out,
            norm.weight,
            float(norm.variance_epsilon),
            sconv_cache,
            kernel_ci,
            cache_mask,
            weight,
            track_mask,
            track_dst,
            res.multicast_ptr,
            res.multicast_ptr + res.ssconv_out * esz,
            res.flag_ptrs_dev,
            res.state_ptr,
            res.rank,
            world,
            activation=sconv.activation,
            use_residual=sconv.use_residual,
        )
        return hs_out, residual_out

    # Launch configs below are keyed by world size (TP4 and TP8 need different
    # configs: Hc and barrier fan-in differ) and by shape. use_stream=True
    # selects the streaming rolling-window kernel for large T; the grid-exit
    # barrier (per_block=False) is used throughout. Chunked prefill caps extend
    # T at max_prefill_tokens=16384, so the T domain is closed.
    per_block = False
    stream_walk = 0
    use_stream = False
    if is_verify:  # target-verify band (decode is intercepted above)
        if world == 8:
            if t <= 48:
                nb, bs = 48, 384
            elif t <= 80:
                nb, bs = 80, 384
            elif t <= 144:
                nb, bs = 80, 768
            else:
                nb, bs = 96, 384
        else:
            if t <= 8:
                nb, bs = 32, 384
            elif t <= 48:
                nb, bs = 48, 384
            elif t <= 80:
                nb, bs = 80, 384
            elif t <= 144:
                nb, bs = 148, 384
            elif t <= 192:
                nb, bs = 96, 768
            else:
                nb, bs = 128, 768
    elif world == 8:
        if t < 3072:  # chunked band
            if t <= 128:
                nb, bs = 48, 384
            elif t <= 256:
                nb, bs = 80, 384
            elif t <= 512:
                nb, bs = 192, 768
            elif t <= 768:
                nb, bs = 96, 512
            elif t <= 1024:
                nb, bs = 96, 768
            elif t <= 1536:
                nb, bs = 192, 768
            else:
                nb, bs = 148, 1024
        elif 8192 <= t < 10240:
            nb, bs = 48, 1024  # chunked edges stream in this band
        else:
            use_stream = True
            if t < 4096:
                nb, bs, stream_walk = 148, 256, 16
            elif t < 6144:
                nb, bs, stream_walk = 96, 512, 24
            elif t < 8192:
                nb, bs, stream_walk = 64, 512, 32
            elif t < 16384:
                nb, bs, stream_walk = 148, 128, 0
            else:
                nb, bs, stream_walk = 96, 192, 0
    elif t >= 3072:  # TP4 streaming band
        use_stream = True
        if t < 4096:
            nb, bs, stream_walk = 148, 256, 16
        elif t < 6144:
            nb, bs, stream_walk = 0, 0, 24
        elif t < 8192:
            nb, bs, stream_walk = 148, 256, 32
        elif t < 10240:
            nb, bs, stream_walk = 96, 512, 0
        else:
            nb, bs, stream_walk = 148, 256, 0
    elif t <= 128:
        nb, bs = 96, 384
    elif t <= 256:
        nb, bs = 148, 512
    elif t <= 768:
        nb, bs = 128, 256
    elif t <= 3072:
        nb, bs = 148, 384
    else:
        nb, bs = 0, 0

    x_scratch = torch.empty((t, hc), dtype=input.dtype, device=input.device)
    out_local = comm.buffer[res.ssconv_out : res.ssconv_out + n].view(t, h)
    if norm is not None:
        assert norm_residual is not None
        norm_kwargs = dict(
            track_from_cache=track_from_cache,
            out_local=out_local,
            norm_gamma=norm.weight,
            norm_residual=norm_residual,
            norm_out=torch.empty_like(norm_residual),
            norm_eps=float(norm.variance_epsilon),
        )
    else:
        norm_kwargs = dict(track_from_cache=track_from_cache)
    jit.inkling_ar_scattered_sconv(
        buf,
        x_scratch,
        sconv_cache,
        safe_idx,
        cache_mask,
        kernel_ci,
        has_initial_state,
        cu,
        si,
        weight,
        track_rows,
        track_mask,
        track_dst,
        res.multicast_ptr,
        res.multicast_ptr + res.ssconv_out * esz,
        res.flag_ptrs_dev,
        res.state_ptr,
        res.rank,
        world,
        activation=sconv.activation,
        use_residual=sconv.use_residual,
        num_blocks=nb,
        block_size=bs,
        per_block_barrier=per_block,
        # Only verify consumes x_scratch (window save); skipping the global
        # scratch writes lets the chunked path stay smem-resident.
        need_scratch=is_verify,
        # Streaming rolling-window path: v3-dataflow, no staging. Below the
        # stream band, walk-length geometry underfills the grid, so
        # chunked/tile is used instead.
        use_stream=use_stream,
        stream_walk=stream_walk,
        **norm_kwargs,
    )
    if is_verify:
        # Save the per-position windows for update_conv_state_after_mtp_verify.
        sconv.verify_fused_ar_finish(forward_batch, x_scratch, cache_indices)
    if norm is not None:
        return norm_kwargs["norm_out"], norm_residual
    return out_local


# Below this token count the unfused non-scattered chain {one-shot AR +
# full-width causal_conv1d + update_sconv_cache} beats the fused kernel.
# The threshold is part of the producer/consumer contract.
_INKLING_AR_FW_MIN_TOKENS = 3072


def fullwidth_ar_sconv_fusable(
    group: GroupCoordinator,
    forward_batch,
    num_tokens: int,
    hidden: int,
    dtype: torch.dtype,
) -> bool:
    """True when a NON-scattered extend {all-reduce -> full-width sconv ->
    cache update} chain can run as the fused column kernel in full-width mode
    (``ar_fullwidth_sconv_fused``). Pure function of per-forward state -- the
    producing layer (``reduce=False``) and the consuming site must evaluate
    it identically. Mutually exclusive with ``ar_sconv_norm_fusable`` by mode
    (extend vs decode/verify) and with ``scattered_ar_sconv_fusable`` by the
    scattered flag."""
    if not is_cuda():
        return False
    if not (
        not get_server_args().enable_scattered_sconv
        and envs.SGLANG_OPT_USE_INKLING_CUSTOM_AR.get()
        and envs.SGLANG_OPT_USE_INKLING_FUSED_AR_SCONV.get()
    ):
        return False
    fm = forward_batch.forward_mode
    if fm.is_draft_extend_v2():
        return False  # de-tied per-step metadata semantics; unfused chain
    if fm.is_target_verify():
        return False  # verify needs the window save (need_scratch); v5 covers it
    if not fm.is_extend():
        return False
    if num_tokens < _INKLING_AR_FW_MIN_TOKENS:
        return False
    # Same prefill-runner scope as the scattered gate: BCG / tc_piecewise
    # pieces can't carry the cross-layer producer contract; the FULL prefill
    # CUDA-graph backend is supported (capture-safe kernel, in-graph metadata).
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        get_tc_piecewise_forward_context,
    )

    tc_ctx = get_tc_piecewise_forward_context()
    if tc_ctx is not None and not tc_ctx.full_graph:
        return False
    comm = group.torch_symm_mem_comm
    if (
        comm is None
        or comm.disabled
        or group.world_size not in (4, 8)
        or dtype != comm.dtype
    ):
        return False
    n = num_tokens * hidden
    if num_tokens == 0 or hidden % (group.world_size * _INKLING_AR_VEC) != 0:
        return False
    res = _get_inkling_ar_resources(comm)
    return res is not None and n <= res.ssconv_out


def ar_fullwidth_sconv_fused(
    input: torch.Tensor,
    sconv,
    forward_batch,
    group: GroupCoordinator,
) -> torch.Tensor:
    """Fused NON-scattered extend {AR + sconv + cache update}: the column
    two-shot kernel in full-width mode. The conv still runs column-sharded
    (1/P of the replicated full-width conv's FLOPs, no x round trip) against
    this rank's column slice of the replicated [slots, W-1, H] cache and
    [H, W] weight, but phase 3 updates/tracks ALL H cache columns on every
    rank (window rows re-ld_reduced full-width -- B*(W-1) rows, negligible)
    so the replicated cache stays coherent for the full-width decode/verify
    consumers. ``input`` holds the UNREDUCED producer partials ([T, H],
    ``reduce=False``); the caller must have checked
    ``fullwidth_ar_sconv_fusable``. Returns the gathered post-conv [T, H]
    (a view of the OUT symm region); the caller runs the norm unfused (the
    in-kernel tail is slower at extend shapes, see
    ``ar_scattered_sconv_fused``)."""
    shared = take_ar_shared(input.shape[0])
    if shared is not None:
        # Pre-add on this path rather than folding in-kernel: pre-add keeps the
        # full-occupancy torch.add instead of the barrier-capped grid.
        # Add straight into the AR buffer -- the same single kernel the
        # producer used to run, and the stage-in copy below then no-ops.
        _buf = get_ar_buffer(group, input.shape[0], input.shape[1], input.dtype)
        if _buf is not None:
            torch.add(input, shared, out=_buf)
            input = _buf
        else:
            input = input + shared

    comm = group.torch_symm_mem_comm
    res = _get_inkling_ar_resources(comm)
    jit = _ar_ssconv_jit()
    t, h = input.shape
    n = t * h
    world = res.world
    hc = h // world
    rank = res.rank

    buf = comm.buffer[:n].view(t, h)
    if input.data_ptr() != buf.data_ptr():
        buf.copy_(input)

    (
        sconv_cache,
        safe_idx,
        cache_mask,
        cu,
        si,
        weight,
        query_start_loc,
        cache_indices,
        has_initial_state,
        track_rows,
        track_mask,
        track_dst,
    ) = sconv.extend_fused_ar_inputs(forward_batch)
    del query_start_loc  # update + track are fused in-kernel
    kernel_ci = cache_indices.to(torch.int32)

    # The streaming kernel is used throughout the fusable band
    # (>= _INKLING_AR_FW_MIN_TOKENS).
    if world == 8:
        if t < 4096:
            nb, bs, walk = 0, 0, 16
        elif t < 6144:
            nb, bs, walk = 96, 512, 24
        elif t < 8192:
            nb, bs, walk = 64, 512, 32
        elif t < 10240:
            nb, bs, walk = 0, 0, 0
        elif t < 16384:
            nb, bs, walk = 148, 128, 0
        else:
            nb, bs, walk = 96, 192, 0
    elif t < 4096:
        nb, bs, walk = 0, 0, 16
    elif t < 6144:
        nb, bs, walk = 96, 512, 24
    elif t < 8192:
        nb, bs, walk = 148, 256, 32
    elif t < 10240:
        nb, bs, walk = 0, 0, 0
    else:
        nb, bs, walk = 148, 256, 0

    esz = comm.buffer.element_size()
    x_scratch = torch.empty((t, hc), dtype=input.dtype, device=input.device)
    out_local = comm.buffer[res.ssconv_out : res.ssconv_out + n].view(t, h)
    jit.inkling_ar_scattered_sconv(
        buf,
        x_scratch,
        sconv_cache,
        safe_idx,
        cache_mask,
        kernel_ci,
        has_initial_state,
        cu,
        si,
        weight[rank * hc : (rank + 1) * hc],
        track_rows,
        track_mask,
        track_dst,
        res.multicast_ptr,
        res.multicast_ptr + res.ssconv_out * esz,
        res.flag_ptrs_dev,
        res.state_ptr,
        rank,
        world,
        activation=sconv.activation,
        use_residual=sconv.use_residual,
        num_blocks=nb,
        block_size=bs,
        per_block_barrier=False,
        need_scratch=False,
        use_stream=True,
        stream_walk=walk,
        full_update=True,
        cache_col0=rank * hc,
    )
    return out_local
