from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple, Optional, Union

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.environ import envs
from sglang.srt.utils import is_cuda

from .utils import make_name

_is_cuda = is_cuda()

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_common_module() -> Module:
    return load_jit(
        make_name("common"),
        cuda_files=["deepseek_v4/common.cuh"],
        cuda_wrappers=[("plan_compress_prefill", "plan_compress_prefill")],
    )


@cache_once
def _jit_compress_128_online_plan_module() -> Module:
    """Host-side plan generator for online compress 128 (no template args)."""
    return load_jit(
        make_name("compress_128_online_plan"),
        cuda_files=["deepseek_v4/c128_online.cuh"],
        cuda_wrappers=[
            ("plan_compress_online_prefill", "plan_compress_online_prefill"),
        ],
    )


@cache_once
def _jit_compress_128_online_module(head_dim: int) -> Module:
    """Online compress 128 kernel: ring_size=1, per-index (max, sum, kv) state."""
    args = make_cpp_args(head_dim, is_arch_support_pdl())
    kernel_class = f"FlashCompress128OnlineKernel<{args}>"
    return load_jit(
        make_name("compress_128_online"),
        *args,
        cuda_files=["deepseek_v4/c128_online.cuh"],
        cuda_wrappers=[
            ("decode", f"{kernel_class}::run_decode"),
            ("prefill", f"{kernel_class}::run_prefill"),
        ],
        extra_cuda_cflags=["-use_fast_math"],
    )


@cache_once
def _jit_norm_rope_module(
    dtype: torch.dtype,
    head_dim: int,
    rope_dim: int,
) -> Module:
    args = make_cpp_args(dtype, head_dim, rope_dim, is_arch_support_pdl())
    return load_jit(
        make_name("fused_norm_rope"),
        *args,
        cuda_files=["deepseek_v4/fused_norm_rope.cuh"],
        cuda_wrappers=[
            ("forward", f"FusedNormRopeKernel<{args}>::forward"),
        ],
    )


@cache_once
def _jit_compress_module(
    head_dim: int,
    dtype_in: torch.dtype,
    dtype_out: torch.dtype,
    ratio: Literal[4, 128],
) -> Module:
    args = make_cpp_args(head_dim, dtype_in, dtype_out, is_arch_support_pdl())
    kernel_class = f"FlashCompress{ratio}Kernel<{args}>"
    return load_jit(
        make_name(f"compress_{ratio}"),
        *args,
        cuda_files=[f"deepseek_v4/c{ratio}.cuh"],
        cuda_wrappers=[
            ("decode", f"{kernel_class}::run_decode"),
            ("prefill", f"{kernel_class}::run_prefill"),
        ],
        extra_cuda_cflags=["-use_fast_math"],
    )


class CompressorPrefillPlan(NamedTuple):
    compress_ratio: int
    compress_plan: torch.Tensor
    write_plan: torch.Tensor

    def copy_(self, other: CompressorPrefillPlan) -> None:
        assert self.compress_ratio == other.compress_ratio
        self.compress_plan.copy_(other.compress_plan)
        self.write_plan.copy_(other.write_plan)

    @staticmethod
    def generate(
        compress_ratio: Literal[4, 128],
        num_q_tokens: int,
        seq_lens: torch.Tensor,
        extend_lens: torch.Tensor,
        device: torch.device,
        use_cuda_graph: bool = False,
    ) -> CompressorPrefillPlan:
        from sglang.srt.environ import envs

        # Online c128 keeps the same NamedTuple shape (compress_plan, write_plan)
        # so call sites that splat `*plan[1:]` continue to work, but the C++
        # plan struct semantics differ (last-token coords + window_len).
        if compress_ratio == 128 and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
            return CompressorPrefillPlan._generate_online(
                num_q_tokens=num_q_tokens,
                seq_lens=seq_lens,
                extend_lens=extend_lens,
                device=device,
                use_cuda_graph=use_cuda_graph,
            )
        assert seq_lens.device == extend_lens.device
        seq_lens = seq_lens.to(torch.int64)
        extend_lens = extend_lens.to(torch.int64)
        plan_tensor = torch.empty(
            (2, num_q_tokens, 16),
            dtype=torch.uint8,
            device=seq_lens.device,
            pin_memory=seq_lens.is_cpu,
        )
        is_overlap = compress_ratio == 4
        if _is_cuda:
            module = _jit_common_module()
            plan_lens = module.plan_compress_prefill(
                extend_lens,
                seq_lens,
                plan_tensor[0],
                plan_tensor[1],
                compress_ratio,
                is_overlap,
                use_cuda_graph,
            )
        else:
            plan_lens = _plan_compress_prefill_torch(
                extend_lens,
                seq_lens,
                plan_tensor[0],
                plan_tensor[1],
                compress_ratio,
                is_overlap,
                use_cuda_graph,
            )
        return CompressorPrefillPlan(
            compress_ratio,
            plan_tensor[0, : plan_lens[0]].to(device, non_blocking=True),
            plan_tensor[1, : plan_lens[1]].to(device, non_blocking=True),
        )

    @staticmethod
    def _generate_online(
        num_q_tokens: int,
        seq_lens: torch.Tensor,
        extend_lens: torch.Tensor,
        device: torch.device,
        use_cuda_graph: bool,
    ) -> CompressorPrefillPlan:
        # Online plan host-side path: only CPU/cuda-host implemented today.
        # Move inputs to CPU pinned memory then bounce the result to device.
        seq_lens_cpu = seq_lens.detach().to(torch.int64).cpu()
        extend_lens_cpu = extend_lens.detach().to(torch.int64).cpu()
        plan_tensor = torch.empty(
            (2, num_q_tokens, 16),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        module = _jit_compress_128_online_plan_module()
        plan_lens = module.plan_compress_online_prefill(
            extend_lens_cpu,
            seq_lens_cpu,
            plan_tensor[0],
            plan_tensor[1],
            use_cuda_graph,
        )
        return CompressorPrefillPlan(
            128,
            plan_tensor[0, : plan_lens[0]].to(device, non_blocking=True),
            plan_tensor[1, : plan_lens[1]].to(device, non_blocking=True),
        )

    @property
    def is_decode(self) -> bool:
        return False


class CompressorDecodePlan(NamedTuple):
    compress_ratio: int
    seq_lens: torch.Tensor

    def copy_(self, other: CompressorDecodePlan) -> None:
        assert self.compress_ratio == other.compress_ratio
        self.seq_lens.copy_(other.seq_lens)

    @property
    def is_decode(self) -> bool:
        return True


def compress_plan(
    compress_ratio: Literal[4, 128],
    num_q_tokens: int,
    seq_lens: torch.Tensor,
    extend_lens: Optional[torch.Tensor],
    device: torch.device,
) -> Union[CompressorDecodePlan, CompressorPrefillPlan]:
    if extend_lens is not None:
        return CompressorPrefillPlan.generate(
            compress_ratio,
            num_q_tokens,
            seq_lens,
            extend_lens,
            device,
        )
    else:
        assert num_q_tokens == len(seq_lens)
        seq_lens = seq_lens.to(device, non_blocking=True)
        return CompressorDecodePlan(compress_ratio, seq_lens)


def compress_forward(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan, None] = None,
    extra_data: Optional[torch.Tensor] = None,
    *,
    head_dim: int,
    compress_ratio: Literal[4, 128],
    out: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    extend_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert head_dim % 128 == 0
    num_q_tokens = kv_score_input.shape[0]
    if out is None:
        out = kv_score_input.new_empty((num_q_tokens, head_dim))
    if plan is None:
        assert seq_lens is not None
        plan = compress_plan(
            compress_ratio,
            num_q_tokens,
            seq_lens,
            extend_lens,
            kv_score_input.device,
        )
    assert plan.compress_ratio == compress_ratio, "Mismatched compress ratio in plan!"
    # Online c128: separate JIT module, fp32 state, no compile-time dtypes.
    if compress_ratio == 128 and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
        online_module = _jit_compress_128_online_module(head_dim=head_dim)
        F = online_module.decode if plan.is_decode else online_module.prefill
        F(kv_score_buffer, kv_score_input, out, ape, indices, *plan[1:], extra_data)
        return out
    if _is_xpu:
        module = _jit_compress_module(
            head_dim,
            kv_score_input.dtype,
            out.dtype,
            compress_ratio,
        )
        F = module.decode if plan.is_decode else module.prefill
        F(kv_score_buffer, kv_score_input, out, ape, indices, *plan[1:], extra_data)
    else:
        # torch fallback for non-CUDA backends. Mirrors
        # FlashCompress{4,128}Kernel in jit_kernel/csrc/deepseek_v4/c{4,128}.cuh.
        _torch_compress_forward(
            kv_score_buffer,
            kv_score_input,
            out,
            ape,
            indices,
            plan,
            extra_data,
            head_dim=head_dim,
            compress_ratio=compress_ratio,
        )
    return out


def compress_fused_norm_rope_inplace(
    kv: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    freq_cis: torch.Tensor,
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan],
) -> None:
    freq_cis = torch.view_as_real(freq_cis).flatten(-2)
    module = _jit_norm_rope_module(kv.dtype, kv.shape[-1], freq_cis.shape[-1])
    module.forward(
        kv,
        weight,
        plan[1],
        freq_cis,
        int(plan.is_decode),
        eps,
        plan.compress_ratio,
    )


def fused_norm_rope_inplace(
    kv: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    freq_cis: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    freq_cis = torch.view_as_real(freq_cis).flatten(-2)
    module = _jit_norm_rope_module(kv.dtype, kv.shape[-1], freq_cis.shape[-1])
    module.forward(
        kv,
        weight,
        positions,
        freq_cis,
        2,
        eps,
        0,
    )


def _plan_compress_prefill_torch(
    extend_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    compress_plan: torch.Tensor,
    write_plan: torch.Tensor,
    compress_ratio: int,
    is_overlap: bool,
    use_cuda_graph: bool,
) -> Tuple[int, int]:
    """Pure-torch fallback for ``plan_compress_prefill``."""
    import struct

    assert compress_plan.dtype == torch.uint8
    assert write_plan.dtype == torch.uint8
    num_tokens = compress_plan.shape[0]
    assert write_plan.shape[0] == num_tokens
    assert compress_plan.shape[1] == 16 and write_plan.shape[1] == 16

    extend_lens_cpu = extend_lens.detach().to("cpu", dtype=torch.int64).tolist()
    seq_lens_cpu = seq_lens.detach().to("cpu", dtype=torch.int64).tolist()
    batch_size = len(extend_lens_cpu)
    assert len(seq_lens_cpu) == batch_size

    ratio = compress_ratio * (2 if is_overlap else 1)
    counter = 0
    compress_entries: list = []
    write_entries: list = []

    for i in range(batch_size):
        seq_len = int(seq_lens_cpu[i])
        extend_len = int(extend_lens_cpu[i])
        assert 0 < extend_len <= seq_len
        prefix_len = seq_len - extend_len
        pos = (seq_len // compress_ratio) * compress_ratio
        if is_overlap:
            start_write_pos = pos - compress_ratio if pos >= compress_ratio else 0
        else:
            start_write_pos = pos
        for j in range(extend_len):
            position = prefix_len + j
            window_len = ratio - min(j + 1, ratio)
            plan = (counter + j, i, position, window_len)
            if (position + 1) % compress_ratio == 0:
                compress_entries.append(plan)
            if position >= start_write_pos:
                write_entries.append(plan)
        counter += extend_len
    assert counter == num_tokens, f"input size {counter} != num_q_tokens {num_tokens}"

    kInvalid = 0xFFFFFFFF
    invalid_row = struct.pack("<IIII", kInvalid, kInvalid, kInvalid, kInvalid)

    def _fill(buf: torch.Tensor, entries: list) -> int:
        n_entries = len(entries)
        n_rows = num_tokens if use_cuda_graph else n_entries
        if n_rows == 0:
            return num_tokens if use_cuda_graph else 0
        payload = bytearray()
        for e in entries:
            payload.extend(struct.pack("<IIII", *e))
        if use_cuda_graph and n_entries < num_tokens:
            for _ in range(num_tokens - n_entries):
                payload.extend(invalid_row)
        cpu_view = torch.frombuffer(payload, dtype=torch.uint8).view(n_rows, 16)
        buf[:n_rows].copy_(cpu_view)
        return num_tokens if use_cuda_graph else n_entries

    compress_count = _fill(compress_plan, compress_entries)
    write_count = _fill(write_plan, write_entries)
    return compress_count, write_count


def _decode_prefill_plan(plan_bytes: torch.Tensor) -> torch.Tensor:
    """Decode packed PrefillPlan tensor ([N, 16] uint8) into [N, 4] int64.

    Each plan slot is 4 little-endian uint32:
    (ragged_id, batch_id, position, window_len). Invalid entries use
    ``0xFFFFFFFF`` for every field.
    """
    cpu = plan_bytes.detach().to("cpu", copy=False).contiguous()
    arr = cpu.numpy().reshape(-1).view("<u4").reshape(-1, 4).astype("int64")
    return torch.from_numpy(arr)


def _describe(x: Any) -> str:
    if isinstance(x, torch.Tensor):
        return f"Tensor(shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device})"
    if isinstance(x, (CompressorDecodePlan, CompressorPrefillPlan)):
        fields = ", ".join(
            f"{name}={_describe(getattr(x, name))}" for name in x._fields
        )
        return f"{type(x).__name__}({fields})"
    if x is None:
        return "None"
    return repr(x)


def _torch_compress_forward(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    out: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan],
    extra_data: Optional[torch.Tensor],
    *,
    head_dim: int,
    compress_ratio: Literal[4, 128],
) -> None:
    if compress_ratio == 4:
        if plan.is_decode:
            _torch_c4_decode(
                kv_score_buffer,
                kv_score_input,
                out,
                ape,
                indices,
                plan.seq_lens,
                extra_data,
                head_dim=head_dim,
            )
        else:
            _torch_c4_prefill(
                kv_score_buffer,
                kv_score_input,
                out,
                ape,
                indices,
                plan.compress_plan,
                plan.write_plan,
                extra_data,
                head_dim=head_dim,
            )
    else:
        assert compress_ratio == 128
        if plan.is_decode:
            _torch_c128_decode(
                kv_score_buffer,
                kv_score_input,
                out,
                ape,
                indices,
                plan.seq_lens,
                head_dim=head_dim,
            )
        else:
            _torch_c128_prefill(
                kv_score_buffer,
                kv_score_input,
                out,
                ape,
                indices,
                plan.compress_plan,
                plan.write_plan,
                extra_data,
                head_dim=head_dim,
            )


def _softmax_weighted_sum(
    kv: torch.Tensor,
    score: torch.Tensor,
    bias: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Safe softmax over dim=-2 then weighted sum of ``kv``.

    Shapes: ``kv``/``score``/``bias`` all ``[..., S, head_dim]``.
    Returns ``[..., head_dim]`` cast to ``out_dtype``.
    """
    s = (score.float() + bias.float())
    m = s.amax(dim=-2, keepdim=True)
    w = (s - m).exp()
    num = (kv.float() * w).sum(dim=-2)
    den = w.sum(dim=-2)
    return (num / den).to(out_dtype)


# ---------------------------------------------------------------------------
# c4 fallback
# ---------------------------------------------------------------------------


def _c4_split_chunks(buf_or_input: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Split last dim ``head_dim*4`` into ``[..., 4, head_dim]``.

    Layout: ``| kv_overlap | kv | score_overlap | score |``.
    """
    return buf_or_input.view(*buf_or_input.shape[:-1], 4, head_dim)


def _torch_c4_decode(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    out: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extra: Optional[torch.Tensor],
    *,
    head_dim: int,
) -> None:
    # Determine page mode from buffer shape.
    page_size = kv_score_buffer.shape[1]
    paged = page_size == 4
    assert paged or page_size == 8
    HD = head_dim
    B = indices.shape[0]
    device = kv_score_input.device
    out_dtype = out.dtype

    indices_i64 = indices.to(torch.int64)
    seq_lens_i64 = seq_lens.to(torch.int64)

    # 1) write current step into the buffer
    write_pos = (seq_lens_i64 + (page_size - 1)) % page_size  # [B]
    kv_score_buffer[indices_i64, write_pos] = kv_score_input.to(kv_score_buffer.dtype)

    # 2) forward only when seq_len % 4 == 0
    do_fwd = (seq_lens_i64 % 4) == 0
    if not bool(do_fwd.any()):
        return

    fwd_idx = do_fwd.nonzero(as_tuple=True)[0]
    fwd_indices = indices_i64[fwd_idx]  # [F]
    fwd_seq_lens = seq_lens_i64[fwd_idx]  # [F]

    # Gather 8 slots from buffer for each forward batch.
    buf4 = _c4_split_chunks(kv_score_buffer, HD)  # [N_idx, page, 4, HD]

    if paged:
        assert extra is not None, "Page4Align mode requires extra tensor"
        index_prev = extra.view(-1)[fwd_idx].to(torch.int64)  # [F]
        # i in [0,8): first 4 use index_prev (overlap), last 4 use index
        kv_chunks = []
        score_chunks = []
        for i in range(8):
            k = i % 4
            page_idx = index_prev if i < 4 else fwd_indices
            chunk_kv = 0 if i < 4 else 1
            chunk_score = 2 if i < 4 else 3
            kv_chunks.append(buf4[page_idx, k, chunk_kv])
            score_chunks.append(buf4[page_idx, k, chunk_score])
    else:
        # Ring buffer of size 8.
        kv_chunks = []
        score_chunks = []
        for i in range(8):
            k = (fwd_seq_lens + i) % 8
            chunk_kv = 0 if i < 4 else 1
            chunk_score = 2 if i < 4 else 3
            kv_chunks.append(buf4[fwd_indices, k, chunk_kv])
            score_chunks.append(buf4[fwd_indices, k, chunk_score])

    kv_stack = torch.stack(kv_chunks, dim=1)  # [F, 8, HD]
    score_stack = torch.stack(score_chunks, dim=1)
    bias = ape.unsqueeze(0).expand(kv_stack.shape[0], -1, -1)  # [F, 8, HD]

    # seq_len == 4 special case: zero overlap kv, -inf overlap score.
    sl4 = (fwd_seq_lens == 4)
    if bool(sl4.any()):
        sl4_b = sl4.view(-1, 1, 1)
        zero = torch.zeros((), dtype=kv_stack.dtype, device=device)
        ninf = torch.full((), -1e9, dtype=score_stack.dtype, device=device)
        head_mask = torch.zeros(8, dtype=torch.bool, device=device)
        head_mask[:4] = True
        full_mask = sl4_b & head_mask.view(1, 8, 1)
        kv_stack = torch.where(full_mask, zero, kv_stack)
        score_stack = torch.where(full_mask, ninf, score_stack)

    result = _softmax_weighted_sum(kv_stack, score_stack, bias, out_dtype)
    out[fwd_idx] = result


def _torch_c4_prefill(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    out: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    compress_plan: torch.Tensor,
    write_plan: torch.Tensor,
    extra: Optional[torch.Tensor],
    *,
    head_dim: int,
) -> None:
    page_size = kv_score_buffer.shape[1]
    paged = page_size == 4
    assert paged or page_size == 8
    HD = head_dim
    device = kv_score_input.device
    indices_i64 = indices.to(torch.int64)

    # Decode plans on CPU then move valid entries to device.
    cplan_cpu = _decode_prefill_plan(compress_plan)  # [Nc, 4] int64
    wplan_cpu = _decode_prefill_plan(write_plan)  # [Nw, 4] int64

    extra_i64: Optional[torch.Tensor] = None
    if paged:
        assert extra is not None, "Page4Align c4 prefill requires extra tensor"
        extra_i64 = extra.to(torch.int64)

    # NOTE: order matches the CUDA kernel launches in c4.cuh: compress
    # (reads buffer) runs BEFORE write (mutates buffer). Reversing the order
    # would feed write-modified slots into compress and corrupt the output.

    # ---- compress plan ----------------------------------------------------
    _INVALID_PLAN = 0xFFFFFFFF
    valid_c = (cplan_cpu[:, 0] != _INVALID_PLAN)
    if bool(valid_c.any()):
        cp = cplan_cpu[valid_c].to(device)
        ragged_ids = cp[:, 0]
        batch_ids = cp[:, 1]
        positions = cp[:, 2]
        window_lens = cp[:, 3]
        seq_lens = positions + 1  # [N]
        N = ragged_ids.shape[0]

        buf4 = _c4_split_chunks(kv_score_buffer, HD)  # [N_idx, page, 4, HD]
        inp4 = _c4_split_chunks(kv_score_input, HD)  # [N_q, 4, HD]

        if paged:
            assert extra_i64 is not None
            load_first_page = extra_i64[batch_ids, 0]
            load_second_page = extra_i64[batch_ids, 1]
            # Choose per-i page: window_len <= 4 means both halves use second
            # page; otherwise overlap (i<4) uses first, normal (i>=4) uses
            # second.
            wl_le_4 = window_lens <= 4

        kv_chunks = []
        score_chunks = []
        for i in range(8):
            chunk_kv = 0 if i < 4 else 1
            chunk_score = 2 if i < 4 else 3
            use_buf = (i < window_lens)  # [N] bool

            # Buffer source.
            if paged:
                if i < 4:
                    page_idx = torch.where(
                        wl_le_4, load_second_page, load_first_page
                    )
                else:
                    page_idx = load_second_page
                k_buf = torch.full_like(positions, i % 4)
                buf_kv = buf4[page_idx, k_buf, chunk_kv]
                buf_score = buf4[page_idx, k_buf, chunk_score]
            else:
                page_idx = indices_i64[batch_ids]
                k_buf = (seq_lens + i) % 8
                buf_kv = buf4[page_idx, k_buf, chunk_kv]
                buf_score = buf4[page_idx, k_buf, chunk_score]

            # Ragged tail source (k = i - 7 <= 0).
            rag_off = (ragged_ids + (i - 7)).clamp(min=0)
            rag_kv = inp4[rag_off, chunk_kv]
            rag_score = inp4[rag_off, chunk_score]

            ub = use_buf.unsqueeze(-1)
            kv_chunks.append(torch.where(ub, buf_kv, rag_kv))
            score_chunks.append(torch.where(ub, buf_score, rag_score))

        kv_stack = torch.stack(kv_chunks, dim=1)  # [N, 8, HD]
        score_stack = torch.stack(score_chunks, dim=1)
        bias = ape.unsqueeze(0).expand(N, -1, -1)

        sl4 = (seq_lens == 4)
        if bool(sl4.any()):
            sl4_b = sl4.view(-1, 1, 1)
            zero = torch.zeros((), dtype=kv_stack.dtype, device=device)
            ninf = torch.full((), -1e9, dtype=score_stack.dtype, device=device)
            head_mask = torch.zeros(8, dtype=torch.bool, device=device)
            head_mask[:4] = True
            full_mask = sl4_b & head_mask.view(1, 8, 1)
            kv_stack = torch.where(full_mask, zero, kv_stack)
            score_stack = torch.where(full_mask, ninf, score_stack)

        result = _softmax_weighted_sum(kv_stack, score_stack, bias, out.dtype)
        out[ragged_ids] = result

    # ---- write plan (must run AFTER compress) ----------------------------
    _torch_c4_prefill_write(
        kv_score_buffer,
        kv_score_input,
        indices_i64,
        extra_i64,
        wplan_cpu,
        paged,
        device,
    )


def _torch_c4_prefill_write(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    indices_i64: torch.Tensor,
    extra_i64: Optional[torch.Tensor],
    wplan_cpu: torch.Tensor,
    paged: bool,
    device: torch.device,
) -> None:
    _INVALID_PLAN = 0xFFFFFFFF
    valid_w = (wplan_cpu[:, 0] != _INVALID_PLAN)
    if not bool(valid_w.any()):
        return
    wp = wplan_cpu[valid_w].to(device)
    ragged_ids = wp[:, 0]
    batch_ids = wp[:, 1]
    positions = wp[:, 2]
    if paged:
        assert extra_i64 is not None
        last_pos = extra_i64[batch_ids, 3]
        write_first_page = extra_i64[batch_ids, 2]
        write_second_page = indices_i64[batch_ids]
        tgt_index = torch.where(
            positions < last_pos, write_first_page, write_second_page
        )
        tgt_pos = positions % 4
    else:
        tgt_index = indices_i64[batch_ids]
        tgt_pos = positions % 8
    kv_score_buffer[tgt_index, tgt_pos] = kv_score_input[ragged_ids].to(
        kv_score_buffer.dtype
    )


# ---------------------------------------------------------------------------
# c128 fallback
# ---------------------------------------------------------------------------


def _c128_split_chunks(buf_or_input: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Split last dim ``head_dim*2`` into ``[..., 2, head_dim]``.

    Layout: ``| kv | score |``.
    """
    return buf_or_input.view(*buf_or_input.shape[:-1], 2, head_dim)


def _torch_c128_decode(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    out: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    head_dim: int,
) -> None:
    HD = head_dim
    device = kv_score_input.device
    indices_i64 = indices.to(torch.int64)
    seq_lens_i64 = seq_lens.to(torch.int64)

    # 1) write current step at (seq_len + 127) % 128.
    write_pos = (seq_lens_i64 + 127) % 128
    kv_score_buffer[indices_i64, write_pos] = kv_score_input.to(kv_score_buffer.dtype)

    # 2) forward only when seq_len % 128 == 0; window_len = 128 (all from buf).
    do_fwd = (seq_lens_i64 % 128) == 0
    if not bool(do_fwd.any()):
        return
    fwd_idx = do_fwd.nonzero(as_tuple=True)[0]
    fwd_indices = indices_i64[fwd_idx]

    buf2 = _c128_split_chunks(kv_score_buffer, HD)  # [N_idx, 128, 2, HD]
    gathered = buf2[fwd_indices]  # [F, 128, 2, HD]
    kv = gathered[..., 0, :]  # [F, 128, HD]
    score = gathered[..., 1, :]
    bias = ape.unsqueeze(0).expand(kv.shape[0], -1, -1)
    out[fwd_idx] = _softmax_weighted_sum(kv, score, bias, out.dtype)


def _torch_c128_prefill(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    out: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    compress_plan: torch.Tensor,
    write_plan: torch.Tensor,
    extra: Optional[torch.Tensor],
    *,
    head_dim: int,
) -> None:
    HD = head_dim
    device = kv_score_input.device
    indices_i64 = indices.to(torch.int64)
    # extra is optional load_indices; falls back to indices when absent.
    load_indices_i64 = (
        extra.to(torch.int64) if extra is not None else indices_i64
    )

    cplan_cpu = _decode_prefill_plan(compress_plan)
    wplan_cpu = _decode_prefill_plan(write_plan)

    # NOTE: order matches the CUDA kernel launches in c128.cuh: compress
    # (reads buffer) runs BEFORE write (mutates buffer). Reversing the order
    # would feed write-modified slots into compress and corrupt the output.

    # ---- compress plan (uses `load_indices`) -----------------------------
    _INVALID_PLAN = 0xFFFFFFFF
    valid_c = (cplan_cpu[:, 0] != _INVALID_PLAN)
    if bool(valid_c.any()):
        cp = cplan_cpu[valid_c].to(device)
        ragged_ids = cp[:, 0]
        batch_ids = cp[:, 1]
        window_lens = cp[:, 3]
        N = ragged_ids.shape[0]

        buf2 = _c128_split_chunks(kv_score_buffer, HD)  # [N_idx, 128, 2, HD]
        inp2 = _c128_split_chunks(kv_score_input, HD)  # [N_q, 2, HD]

        page_idx = load_indices_i64[batch_ids]  # [N]
        buf_slot = buf2[page_idx]  # [N, 128, 2, HD]
        buf_kv = buf_slot[..., 0, :]  # [N, 128, HD]
        buf_score = buf_slot[..., 1, :]

        j = torch.arange(128, device=device, dtype=torch.int64)
        rag_off = (ragged_ids.unsqueeze(1) + (j.unsqueeze(0) - 127)).clamp(
            min=0
        )  # [N, 128]
        rag_kv = inp2[..., 0, :][rag_off]  # [N, 128, HD]
        rag_score = inp2[..., 1, :][rag_off]

        use_buf = (j.unsqueeze(0) < window_lens.unsqueeze(1)).unsqueeze(
            -1
        )  # [N,128,1]
        kv = torch.where(use_buf, buf_kv, rag_kv)
        score = torch.where(use_buf, buf_score, rag_score)
        bias = ape.unsqueeze(0).expand(N, -1, -1)  # [N, 128, HD]

        out[ragged_ids] = _softmax_weighted_sum(kv, score, bias, out.dtype)

    # ---- write plan (uses `indices`, must run AFTER compress) ------------
    valid_w = (wplan_cpu[:, 0] != _INVALID_PLAN)
    if bool(valid_w.any()):
        wp = wplan_cpu[valid_w].to(device)
        ragged_ids = wp[:, 0]
        batch_ids = wp[:, 1]
        positions = wp[:, 2]
        tgt_index = indices_i64[batch_ids]
        tgt_pos = positions % 128
        kv_score_buffer[tgt_index, tgt_pos] = kv_score_input[ragged_ids].to(
            kv_score_buffer.dtype
        )
