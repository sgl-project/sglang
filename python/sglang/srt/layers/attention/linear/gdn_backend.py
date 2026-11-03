from functools import cache
from typing import Optional, Tuple, Union

import msgspec
import torch
import torch.distributed as dist

from sglang.kernels.ops.attention.linear.fused_gdn_gating import fused_gdn_gating
from sglang.kernels.ops.mamba.causal_conv1d_triton import (
    CAUSAL_CONV1D_FWD_BLOCK_M,
    causal_conv1d_fn,
    causal_conv1d_fn as causal_conv1d_fn_triton,
    causal_conv1d_peer_with_halo,
    causal_conv1d_update,
    causal_conv1d_with_halo,
    supports_causal_conv1d_peer,
)
from sglang.kernels.ops.mamba.gdn_prefill_track import (
    gdn_track_conv_window,
    gdn_track_ssm_state,
)
from sglang.srt.configs.hybrid_arch import hybrid_gdn_config
from sglang.srt.distributed.device_communicators.torch_symm_mem import (
    TorchSymmMemCommunicator,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear.gdn_prefill_graph import (
    GDNPrefillGraphMetadata,
    buckets_keep_chunk_o_block_size,
)
from sglang.srt.layers.attention.linear.kda_prefill_graph import (
    KDAPrefillTrackRows,
    build_track_rows,
)
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.kernels.state_pool_io import (
    stage_pool_states,
    unstage_pool_states,
)
from sglang.srt.layers.attention.linear.utils import (
    LinearAttnBackends,
    LinearAttnKernelBackend,
    build_verify_intermediate_state_indices,
    pp_spec_stable_rows_enabled,
    select_verify_intermediate_state_indices,
)
from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
from sglang.srt.layers.cp.utils import _cp_metadata_tensor
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.mem_cache.kv.router import STATE_PART
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import (
    get_exec,
    get_memory,
    get_schedule,
    mamba_cache_chunk_size,
)
from sglang.srt.utils import is_cpu, is_cuda, is_hip, is_npu, is_xpu
from sglang.srt.utils.common import is_pin_memory_available, rank0_log

_is_hip = is_hip()
from sglang.srt.utils.nvtx_pytorch_hooks import profile_method, profile_range

if not is_cpu():
    from sglang.kernels.ops.attention.fla.chunk_delta_h import (
        CHUNK_SIZE as FLA_CHUNK_SIZE,
    )

if is_cuda() or is_hip() or is_xpu():
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
        can_use_fused_qkvzba_causal_conv1d_update_contiguous,
        fused_qkv_split_gdn_prefill,
        fused_qkvzba_causal_conv1d_update_contiguous,
        fused_qkvzba_split_reshape_cat_contiguous,
    )

MAX_FUSED_QKV_SPLIT_DIM = 8192
_fused_decode_proj_conv_logged = False
_fused_decode_proj_conv_fallback_logged = False
_fused_decode_proj_conv_layers_logged: set[int] = set()
_fused_decode_real_tensor_verified_layers: set[int] = set()
_fused_decode_log_layer_hits = envs.SGLANG_GDN_DECODE_FUSION_LOG_LAYER_HITS.get()
_fused_decode_verify_real_tensors = (
    envs.SGLANG_GDN_DECODE_FUSION_VERIFY_REAL_TENSORS.get()
)


class GDNMISMetadata(msgspec.Struct, frozen=True):
    query_token_indices: torch.Tensor
    query_cu_seqlens: torch.Tensor
    query_seq_lens_cpu: list[int]
    query_request_indices: torch.Tensor
    item_token_indices: torch.Tensor
    item_cu_seqlens: torch.Tensor
    item_seq_lens_cpu: list[int]
    item_request_indices: torch.Tensor


def build_gdn_mis_metadata(forward_batch: ForwardBatch) -> GDNMISMetadata:
    """Build compact query/item segments from request-local MIS delimiters."""
    if not forward_batch.is_prefill_only:
        raise ValueError("GDN MIS is only supported for prefill-only requests")

    prefix_lens = forward_batch.extend_prefix_lens_cpu
    if isinstance(prefix_lens, torch.Tensor):
        prefix_lens = prefix_lens.tolist()
    if any(int(prefix_len) != 0 for prefix_len in prefix_lens):
        raise ValueError("GDN MIS does not support cached prefixes")

    seq_lens = forward_batch.extend_seq_lens_cpu
    if isinstance(seq_lens, torch.Tensor):
        seq_lens = seq_lens.tolist()
    seq_lens = [int(seq_len) for seq_len in seq_lens]
    delimiter_indices = forward_batch.multi_item_delimiter_indices
    if delimiter_indices is None or len(delimiter_indices) != len(seq_lens):
        raise ValueError("GDN MIS requires delimiter indices for every request")
    if sum(seq_lens) > forward_batch.input_ids.numel():
        raise ValueError("GDN MIS sequence lengths exceed the input tokens")

    query_token_indices: list[int] = []
    query_seq_lens_cpu: list[int] = []
    query_request_indices: list[int] = []
    item_token_indices: list[int] = []
    item_seq_lens_cpu: list[int] = []
    item_request_indices: list[int] = []

    request_start = 0
    for request_idx, (seq_len, request_delimiters) in enumerate(
        zip(seq_lens, delimiter_indices)
    ):
        delimiters = [int(index) for index in request_delimiters.tolist()]
        if len(delimiters) < 2:
            raise ValueError("GDN MIS requires at least two delimiters per request")
        if any(
            current >= following
            for current, following in zip(delimiters, delimiters[1:])
        ):
            raise ValueError("GDN MIS delimiter indices must be strictly increasing")
        if delimiters[0] < 0 or delimiters[-1] >= seq_len:
            raise ValueError("GDN MIS delimiter index is outside the request")
        if delimiters[-1] != seq_len - 1:
            raise ValueError("GDN MIS final delimiter must be the last request token")

        query_len = delimiters[0]
        if query_len > 0:
            query_token_indices.extend(range(request_start, request_start + query_len))
            query_seq_lens_cpu.append(query_len)
            query_request_indices.append(request_idx)

        branch_ends = delimiters[1:] + [seq_len]
        for branch_start, branch_end in zip(delimiters, branch_ends):
            branch_len = branch_end - branch_start
            item_token_indices.extend(
                range(request_start + branch_start, request_start + branch_end)
            )
            item_seq_lens_cpu.append(branch_len)
            item_request_indices.append(request_idx)

        request_start += seq_len

    device = forward_batch.input_ids.device

    def _indices(values: list[int], dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(values, dtype=dtype, device=device)

    def _cu_seqlens(lengths: list[int]) -> torch.Tensor:
        result = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
        if lengths:
            result[1:] = torch.tensor(lengths, dtype=torch.int32, device=device).cumsum(
                dim=0
            )
        return result

    return GDNMISMetadata(
        query_token_indices=_indices(query_token_indices, torch.int64),
        query_cu_seqlens=_cu_seqlens(query_seq_lens_cpu),
        query_seq_lens_cpu=query_seq_lens_cpu,
        query_request_indices=_indices(query_request_indices, torch.int64),
        item_token_indices=_indices(item_token_indices, torch.int64),
        item_cu_seqlens=_cu_seqlens(item_seq_lens_cpu),
        item_seq_lens_cpu=item_seq_lens_cpu,
        item_request_indices=_indices(item_request_indices, torch.int64),
    )


def validate_gdn_mis_backend(prefill_backend: LinearAttnKernelBackend) -> None:
    if not get_exec().features.enable_mis:
        return
    if not prefill_backend.is_triton():
        raise ValueError(
            "GDN multi-item scoring requires the Triton linear-attention prefill "
            "backend. Set --linear-attn-prefill-backend triton."
        )
    if get_memory().enable_page_major_kv_layout:
        raise ValueError("GDN multi-item scoring does not support page-major layout")


if is_cuda():
    from sglang.srt.layers.attention.mamba.causal_conv1d import (
        causal_conv1d_fn as causal_conv1d_fn_cuda,
    )

    causal_conv1d_fn = causal_conv1d_fn_cuda
elif is_npu():
    from sgl_kernel_npu.fla.fused_gdn_gating import fused_gdn_gating_npu
    from sgl_kernel_npu.mamba.causal_conv1d import (
        causal_conv1d_fn_npu,
        causal_conv1d_update_npu,
    )

    fused_gdn_gating = fused_gdn_gating_npu
    causal_conv1d_fn = causal_conv1d_fn_npu
    causal_conv1d_update = causal_conv1d_update_npu
elif is_cpu():
    from sgl_kernel.mamba import causal_conv1d_fn_cpu, causal_conv1d_update_cpu

    causal_conv1d_fn = causal_conv1d_fn_cpu
    causal_conv1d_update = causal_conv1d_update_cpu
    fused_gdn_gating = torch.ops.sgl_kernel.fused_gdn_gating_cpu


def flashinfer_gdn_prefill_default(model_runner: ModelRunner) -> Optional[str]:
    """FlashInfer for the narrow SM90/SM100 GDN prefill domains we validated, else None."""
    sm_major = torch.cuda.get_device_capability()[0] if is_cuda() else 0
    if (
        get_exec().mamba.linear_attn_prefill_backend is not None
        or get_exec().mamba.linear_attn_backend != "triton"
        or get_exec().deterministic.enable_deterministic_inference
        or sm_major not in (9, 10)
    ):
        return None

    # SM100 runs the CUDA>=13 CuTe-DSL chunk kernel on a bf16 state pool;
    # SM90 runs the fused Hopper kernel on an fp32 state pool and tolerates
    # larger chunks. Everything outside these validated domains keeps Triton.
    cuda_version = torch.version.cuda
    if sm_major == 10:
        if cuda_version is None or int(cuda_version.split(".", 1)[0]) < 13:
            return None
        max_chunk = 8192
        expected_state_dtype = torch.bfloat16
    else:
        max_chunk = 32768
        expected_state_dtype = torch.float32

    chunk_size = get_schedule().chunked_prefill_size
    config = hybrid_gdn_config(model_runner.model_config)
    if (
        get_schedule().enable_dynamic_chunking
        or chunk_size is None
        or not 1 <= chunk_size <= max_chunk
        or getattr(config, "linear_key_head_dim", None) != 128
        or getattr(config, "linear_value_head_dim", None) != 128
        or model_runner.token_to_kv_pool.layer_router.part_view(
            STATE_PART
        ).mamba_cache.temporal.dtype
        != expected_state_dtype
    ):
        return None

    from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
        is_flashinfer_gdn_prefill_available,
    )

    if not is_flashinfer_gdn_prefill_available():
        return None

    rank0_log(f"Defaulting SM{sm_major}0 GDN prefill backend to FlashInfer.")
    return "flashinfer"


def _validate_gdn_linear_attn_backends(backends: LinearAttnBackends) -> None:
    if (
        get_exec().deterministic.enable_deterministic_inference
        and backends.prefill.is_flashinfer()
    ):
        raise ValueError(
            "FlashInfer GDN prefill is not supported with "
            "--enable-deterministic-inference. Use "
            "--linear-attn-prefill-backend triton."
        )


def _tail(history, rows, width):
    # Copy only the convolution halo, not the full projected chunk.
    if width == 0:
        return history[:, :0].clone()
    if rows.shape[0] >= width:
        return rows[-width:].T.contiguous()
    return torch.cat((history[:, -(width - rows.shape[0]) :], rows.T), dim=1)


def _padded_inputs(inputs, width):
    padded = []
    for x in inputs:
        if x.shape[0] == width:
            padded.append(x.unsqueeze(0).contiguous())
            continue
        y = x.new_zeros((1, width, *x.shape[1:]))
        y[0, : x.shape[0]] = x
        padded.append(y)
    return padded


@profile_method("sglang.cp.gdn_pack_qkv")
def _prepare_qkv_inputs(convolved, layer, width, *, fused=True):
    """Split once into FlashQLA's packed inputs, retaining zero-padded tails."""
    count = convolved.shape[0]
    if count == width and count > 0 and convolved.shape[-1] <= 16384 and fused:
        from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
            fused_qkv_split_gdn_prefill,
        )

        # Full-head CP has a wider projection than TP. The 16K-element split
        # fits the supported Hopper/Blackwell kernel without changing values.
        return list(
            fused_qkv_split_gdn_prefill(
                convolved,
                layer.num_q_heads,
                layer.num_k_heads,
                layer.num_v_heads,
                layer.head_q_dim,
                layer.head_k_dim,
                layer.head_v_dim,
            )
        )
    q, k, v = convolved.split((layer.q_dim, layer.k_dim, layer.v_dim), dim=-1)
    return _padded_inputs(
        [
            q.reshape(count, layer.num_q_heads, layer.head_q_dim),
            k.reshape(count, layer.num_k_heads, layer.head_k_dim),
            v.reshape(count, layer.num_v_heads, layer.head_v_dim),
        ],
        width,
    )


def make_gdn_cp_conv_metadata(frame, *, device=None):
    """Cache length-dependent launch tensors to avoid per-layer blocking H2D copies."""
    device = frame.device if device is None else device
    has_initial_state = _cp_metadata_tensor([True], device=device, dtype=torch.bool)
    by_count, metadata = {}, []
    for length in frame.lengths:
        width = (length + 64 * frame.world - 1) // (64 * frame.world) * 64
        count = min(width, max(0, length - frame.rank * width))
        if count not in by_count:
            by_count[count] = dict(
                query_start_loc=_cp_metadata_tensor(
                    [0, count], device=device, dtype=torch.int32
                ),
                has_initial_state=has_initial_state,
                seq_lens_cpu=[count],
            )
        metadata.append(by_count[count])
    return metadata


def convolve_gdn_context_shard(
    frame, layer, local_raw, history, length, width, metadata, *, halo_exchange=None
):
    """Exchange raw convolution halos while evaluating independent local blocks."""
    overlap = halo_exchange is not None
    count = local_raw.shape[0]
    history_width = layer.conv_weights.shape[-1] - 1
    # Only the last kernel_width-1 raw projection rows cross a rank boundary.
    local_tail = _tail(history, local_raw, history_width)
    convolved = None
    if overlap:
        with profile_range("sglang.cp.conv_halo.prepare"):
            pending = halo_exchange.prepare(local_tail)
        if count > CAUSAL_CONV1D_FWD_BLOCK_M:
            with profile_range("sglang.cp.conv_interior"):
                convolved = causal_conv1d_with_halo(
                    local_raw.T,
                    history.unsqueeze(0),
                    layer.conv_weights,
                    bias=layer.bias,
                    activation=layer.activation,
                    token_block_range=(
                        1,
                        (count + CAUSAL_CONV1D_FWD_BLOCK_M - 1)
                        // CAUSAL_CONV1D_FWD_BLOCK_M,
                    ),
                    **metadata,
                )
        with profile_range("sglang.cp.conv_halo.submit"):
            pending.start()
        with profile_range("sglang.cp.conv_halo.wait"):
            tails = pending.wait()
    else:
        tails = [torch.empty_like(local_tail) for _ in range(frame.world)]
        with profile_range("sglang.cp.conv_halo.all_gather"):
            dist.all_gather(tails, local_tail, group=frame.group)
    # A final shard can have fewer rows than the convolution history. Carry
    # earlier rows through it; padding must never replace real history.
    terminal_history = history
    for peer, tail in enumerate(tails):
        if peer == frame.rank:
            preceding = terminal_history
        peer_count = min(width, max(0, length - peer * width))
        if peer_count:
            valid = min(peer_count, history_width)
            terminal_history = torch.cat((terminal_history, tail[:, -valid:]), dim=1)[
                :, -history_width:
            ].contiguous()
    if count:
        with profile_range("sglang.cp.conv_boundary" if overlap else "sglang.cp.conv"):
            convolved = causal_conv1d_with_halo(
                local_raw.T,
                preceding.unsqueeze(0),
                layer.conv_weights,
                bias=layer.bias,
                activation=layer.activation,
                out=convolved,
                token_block_range=(0, 1) if overlap else None,
                **metadata,
            ).T
    else:
        convolved = local_raw
    return convolved, preceding, terminal_history


class GDNKernelDispatcher:
    """Dispatches GDN kernel calls to the appropriate backend per mode."""

    def __init__(
        self,
        decode_backend: LinearAttnKernelBackend,
        prefill_backend: LinearAttnKernelBackend,
        verify_backend: Optional[LinearAttnKernelBackend] = None,
    ):
        self.use_context_parallel = False
        triton_kernel = TritonGDNKernel()
        self.tree_verify_kernel = triton_kernel

        cutedsl_kernel = None
        if decode_backend.is_triton():
            self.decode_kernel = triton_kernel
        elif decode_backend.is_intel_xpu():
            if not is_xpu():
                raise ValueError("--linear-attn-backend intel_xpu requires Intel XPU")
            # The fused SYCL kernel is dispatched via XpuGDNAttnBackend.forward_fused_gdn,
            # outside this dispatcher; Triton is the dispatcher-level kernel for requests
            # that hook doesn't handle (e.g. verify).
            self.decode_kernel = triton_kernel
        elif decode_backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("GDN CuTe DSL backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.gdn_cutedsl import (
                CuteDSLGDNKernel,
            )

            cutedsl_kernel = CuteDSLGDNKernel()
            self.decode_kernel = cutedsl_kernel
        elif decode_backend.is_flashinfer():
            if not is_cuda():
                raise ValueError("FlashInfer GDN backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                FlashInferGDNKernel,
            )

            flashinfer_kernel = FlashInferGDNKernel()
            self.decode_kernel = flashinfer_kernel
        elif decode_backend.is_helion():
            raise ValueError(
                "The Helion linear-attention backend supports KDA only, not GDN."
            )
        else:
            raise ValueError(f"Unsupported GDN decode backend: {decode_backend}")

        if prefill_backend.is_triton():
            self.extend_kernel = triton_kernel
        elif prefill_backend.is_intel_xpu():
            if not is_xpu():
                raise ValueError("--linear-attn-backend intel_xpu requires Intel XPU")
            # See the decode branch above: intel_xpu uses Triton as its
            # dispatcher-level fallback kernel.
            self.extend_kernel = triton_kernel

        elif prefill_backend.is_flashqla():
            from sglang.srt.distributed import get_attn_cp_group, get_attn_tp_group
            from sglang.srt.layers.attention.linear.kernels.gdn_flashqla import (
                FlashQLAGDNKernel,
            )
            from sglang.srt.layers.cp.utils import is_gdn_cp_enabled

            group = None
            if prefill_backend == LinearAttnKernelBackend.FLASHQLA_CP:
                self.use_context_parallel = is_gdn_cp_enabled()
                group = (
                    get_attn_cp_group()
                    if self.use_context_parallel
                    else get_attn_tp_group()
                ).device_group
            self.extend_kernel = FlashQLAGDNKernel(
                fallback=triton_kernel, cp_group=group
            )
        elif prefill_backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("GDN CuTe DSL backend requires CUDA")
            # Reuse the CuteDSL kernel if already created for decode
            if cutedsl_kernel is None:
                from sglang.srt.layers.attention.linear.kernels.gdn_cutedsl import (
                    CuteDSLGDNKernel,
                )

                cutedsl_kernel = CuteDSLGDNKernel()
            # The CuteDSL prefill kernel only exists on SM10x. Elsewhere
            # (SM90, SM120) fall back to Triton so users can pick
            # `cutedsl` uniformly across hardware.
            if cutedsl_kernel.supports_prefill:
                self.extend_kernel = cutedsl_kernel
            else:
                rank0_log(
                    "CuTe DSL GDN prefill is not supported on this GPU "
                    "(requires SM10x). Falling back to Triton for prefill."
                )
                self.extend_kernel = triton_kernel
        elif prefill_backend.is_flashinfer():
            if not is_cuda():
                raise ValueError("FlashInfer GDN backend requires CUDA")
            # Reuse the FlashInfer kernel if already created for decode
            if decode_backend.is_flashinfer():
                self.extend_kernel = flashinfer_kernel
            else:
                from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                    FlashInferGDNKernel,
                )

                flashinfer_kernel = FlashInferGDNKernel()
                self.extend_kernel = flashinfer_kernel
        elif prefill_backend.is_helion():
            raise ValueError(
                "The Helion linear-attention backend supports KDA only, not GDN."
            )
        else:
            raise ValueError(f"Unsupported GDN prefill backend: {prefill_backend}")

        # Verify kernel. An explicitly configured verify backend wins; the
        # historical auto rule (FlashInfer when the selected FlashInfer kernel
        # supports MTP verify) only applies when no explicit choice was made.
        # SM90 FlashInfer verify requires a fp32 SSM state, so e.g.
        # --mamba-ssm-dtype bfloat16 setups must be able to force Triton here.
        if verify_backend is not None and verify_backend.is_triton():
            self.verify_kernel = triton_kernel
            self.verify_kernel_is_flashinfer = False
        elif (
            decode_backend.is_flashinfer() or prefill_backend.is_flashinfer()
        ) and flashinfer_kernel.supports_target_verify:
            self.verify_kernel = flashinfer_kernel
            self.verify_kernel_is_flashinfer = True
        else:
            self.verify_kernel = triton_kernel
            self.verify_kernel_is_flashinfer = False

        self.supports_packed_decode = getattr(
            self.decode_kernel, "supports_packed_decode", False
        )

        rank0_log(
            f"GDN kernel dispatcher: decode={self.decode_kernel.__class__.__name__}, "
            f"extend={self.extend_kernel.__class__.__name__}, "
            f"verify={self.verify_kernel.__class__.__name__} "
            f"packed_decode={self.supports_packed_decode}"
        )

    @property
    def extend_uses_state_checkpoints(self) -> bool:
        return self.extend_kernel.uses_state_checkpoints

    def packed_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """Attempt packed decode. Returns output tensor or None if
        the decode kernel does not support packed decode."""
        if not self.supports_packed_decode:
            return None
        return self.decode_kernel.packed_decode(
            mixed_qkv,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            **kwargs,
        )

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.decode_kernel.decode(
            q,
            k,
            v,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> tuple:
        return self.extend_kernel.extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        verify_kernel = self._get_target_verify_kernel(
            kwargs.get("retrieve_parent_token")
        )
        return verify_kernel.target_verify(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def target_verify_supports_strided_qkv(
        self, retrieve_parent_token: Optional[torch.Tensor]
    ) -> bool:
        verify_kernel = self._get_target_verify_kernel(retrieve_parent_token)
        return (
            getattr(verify_kernel, "supports_strided_target_verify_qkv", False) is True
        )

    def _get_target_verify_kernel(self, retrieve_parent_token: Optional[torch.Tensor]):
        # Tree drafts use Triton even when linear MTP verification uses FlashInfer.
        return (
            self.tree_verify_kernel
            if retrieve_parent_token is not None
            else self.verify_kernel
        )


class GDNAttnBackend(MambaAttnBackendBase):
    """Attention backend for GDN (Gated Delta Network) linear attention."""

    needs_cpu_seq_lens: bool = False
    supports_mis: bool = True
    requires_contiguous_prefill_state: bool = True

    # Set while a breakable prefill CUDA graph captures or replays a bucket:
    # forward_extend then runs the graph-safe chain on these static tables.
    prefill_graph_metadata: Optional[GDNPrefillGraphMetadata] = None

    def _create_kernel_dispatcher(self, backends):
        return GDNKernelDispatcher(backends.decode, backends.prefill, backends.verify)

    def __init__(self, model_runner: ModelRunner):
        _validate_gdn_linear_attn_backends(model_runner.linear_attn_backends)
        super().__init__(model_runner)
        self.enable_mis = get_exec().features.enable_mis
        self.mis_metadata: Optional[GDNMISMetadata] = None
        self.conv_states_shape = self.state_pool.mamba_cache.conv[0].shape
        if not is_cpu() and not is_npu():
            assert self.conv_states_shape[-1] < FLA_CHUNK_SIZE, (
                f"{self.conv_states_shape[-1]=} should be less than {FLA_CHUNK_SIZE}"
            )

        backends = model_runner.linear_attn_backends
        validate_gdn_mis_backend(backends.prefill)
        self.linear_attn_backends = backends
        self._gdn_cp_plan = None
        self.kernel_dispatcher = self._create_kernel_dispatcher(backends)
        # Sized past the pool for attn_tp-padded warmup/MLP-sync batches (see helper).
        self.verify_intermediate_state_indices = (
            build_verify_intermediate_state_indices(
                self.req_to_token_pool.size,
                model_runner.device,
            )
        )
        self._conv_workspace = None
        self._use_strided_target_verify_qkv = False
        # A prefill batch never holds more requests than the pool.
        self.prefill_graph_max_seqs = min(
            self.req_to_token_pool.size, envs.SGLANG_GDN_PREFILL_GRAPH_MAX_SEQS.get()
        )

    def supports_prefill_graph_extend(self) -> bool:
        # Read here, not in __init__: the support check reads model_config.
        return (
            envs.SGLANG_ENABLE_GDN_PREFILL_GRAPH_EXTEND.get()
            # drafts keep their eager GDN extend; only the target's is captured
            and not self.is_draft_worker
            and not self.kernel_dispatcher.use_context_parallel
            and self._prefill_graph_extend_supported()
        )

    def _prefill_graph_extend_supported(self) -> bool:
        from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
            get_prefill_num_tokens_to_capture,
        )

        mamba_cache = self.state_pool.mamba_cache
        capture_tokens = get_exec().graph.cuda_graph_config.prefill.bs or []
        supported = (
            is_cuda()
            and isinstance(self.kernel_dispatcher.extend_kernel, TritonGDNKernel)
            and not self.enable_mis
            # The track plan's h rows assume the delta-rule chunk size.
            and self.mamba_chunk_size == FLA_CHUNK_SIZE
            and all(conv.is_contiguous() for conv in mamba_cache.conv)
            and mamba_cache.temporal.is_contiguous()
            and buckets_keep_chunk_o_block_size(
                get_prefill_num_tokens_to_capture(capture_tokens)
            )
        )
        rank0_log(
            "GDN prefill extend inside breakable prefill CUDA graphs: "
            + ("on" if supported else "unsupported here, layers keep breaking")
        )
        return supported

    def prefill_graph_extend_active(self) -> bool:
        return self.prefill_graph_metadata is not None

    def init_prefill_graph_metadata(
        self, forward_batch: ForwardBatch
    ) -> GDNPrefillGraphMetadata:
        cache_indices = self._prefill_graph_cache_indices(forward_batch)
        meta = GDNPrefillGraphMetadata.allocate(
            num_tokens=forward_batch.positions.shape[0],
            max_seqs_cap=self.prefill_graph_max_seqs,
            # Any replay of this bucket may carry prefix-cache snapshots (the
            # extra buffer is only on with the radix cache).
            track=get_exec().mamba.enable_mamba_extra_buffer,
            cache_index_dtype=cache_indices.dtype,
            device=self.device,
        )
        self._fill_prefill_graph_metadata(
            meta=meta, forward_batch=forward_batch, cache_indices=cache_indices
        )
        return meta

    def refresh_prefill_graph_metadata(
        self, meta: GDNPrefillGraphMetadata, forward_batch: ForwardBatch
    ) -> None:
        self._fill_prefill_graph_metadata(
            meta=meta,
            forward_batch=forward_batch,
            cache_indices=self._prefill_graph_cache_indices(forward_batch),
        )

    def _prefill_graph_cache_indices(self, forward_batch: ForwardBatch):
        # Same slot resolution as MambaAttnBackendBase._forward_metadata.
        cache_indices = self._translate_mamba_indices(
            self.req_to_token_pool.get_mamba_indices(forward_batch.req_pool_indices)
        )
        real_bs = forward_batch._original_batch_size
        if real_bs is not None and real_bs < cache_indices.shape[0]:
            cache_indices = cache_indices.clone()
            cache_indices[real_bs:] = -1
        return cache_indices

    def _fill_prefill_graph_metadata(
        self,
        *,
        meta: GDNPrefillGraphMetadata,
        forward_batch: ForwardBatch,
        cache_indices: torch.Tensor,
    ) -> None:
        if forward_batch.mamba_track_indices is not None:
            # Same one-time in-place translation as the eager metadata path.
            forward_batch.mamba_track_indices = self._translate_mamba_indices(
                forward_batch.mamba_track_indices
            )
        track_rows = None
        if forward_batch.mamba_track_mask is not None:
            assert meta.track, (
                "prefix-cache track batch replayed on a bucket captured without "
                "track tables"
            )
            track_rows = self._prefill_graph_track_rows(forward_batch)
        meta.fill(
            extend_seq_lens=list(forward_batch.extend_seq_lens_cpu),
            cache_indices=cache_indices,
            extend_prefix_lens=forward_batch.extend_prefix_lens,
            track_rows=track_rows,
        )
        if track_rows is None and forward_batch.mamba_track_indices is not None:
            meta.track_dst[: forward_batch.batch_size].copy_(
                forward_batch.mamba_track_indices
            )
        self.prefill_graph_metadata = meta
        self.forward_metadata = ForwardMetadata(
            query_start_loc=meta.cu_seqlens,
            mamba_cache_indices=meta.cache_indices,
            mamba_track_indices=None
            if forward_batch.mamba_track_indices is None
            else meta.track_dst[: forward_batch.batch_size],
        )

    def _prefill_graph_track_rows(
        self, forward_batch: ForwardBatch
    ) -> Optional[KDAPrefillTrackRows]:
        bs = forward_batch.batch_size
        if self._has_cpu_prefill_track_metadata(forward_batch):
            track_mask = list(forward_batch.mamba_prefill_track_mask_cpu)
            track_seq_lens = list(forward_batch.mamba_track_seqlens_cpu)
            extend_prefix_lens = list(forward_batch.extend_prefix_lens_cpu)
        else:
            # The same host copies the eager path takes without CPU metadata.
            track_mask = forward_batch.mamba_track_mask[:bs].tolist()
            track_seq_lens = forward_batch.mamba_track_seqlens[:bs].tolist()
            extend_prefix_lens = forward_batch.extend_prefix_lens[:bs].tolist()
        return build_track_rows(
            track_mask=track_mask,
            track_seq_lens=track_seq_lens,
            extend_seq_lens=list(forward_batch.extend_seq_lens_cpu),
            extend_prefix_lens=extend_prefix_lens,
            state_chunk_size=self.mamba_chunk_size,
            cache_chunk_size=mamba_cache_chunk_size(),
            conv_len=self.conv_states_shape[-1],
            dst=forward_batch.mamba_track_indices,
        )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        super().init_forward_metadata_out_graph(forward_batch, in_capture=in_capture)
        self._init_target_verify_qkv_routing(forward_batch)

    def prepare_cp_metadata(self, batch, *, lengths=None, prefixes=None):
        """Plan GDN's contiguous compute rows before the forward is submitted."""
        from sglang.srt.distributed import get_attn_cp_group
        from sglang.srt.layers.cp.utils import ShardedContextParallelMetadata

        if not batch.forward_mode.is_extend_without_speculative():
            return
        if batch.gdn_cp_metadata is not None:
            return
        lengths = batch.extend_lens if lengths is None else lengths
        prefixes = batch.prefix_lens if prefixes is None else prefixes
        world = get_attn_cp_group().world_size
        attention_indices = tuple(
            tuple(range(rank, sum(lengths), world)) for rank in range(world)
        )
        metadata = ShardedContextParallelMetadata.plan_prefill(
            lengths,
            prefixes,
            self.device,
            cp_mode="round-robin-split",
            attention_indices=attention_indices,
            previous=self._gdn_cp_plan,
        )
        metadata.gdn_conv_metadata = tuple(
            make_gdn_cp_conv_metadata(metadata, device="cpu")
        )
        metadata.upload()
        batch.gdn_cp_metadata = metadata
        self._gdn_cp_plan = metadata

    def warmup(self, model_runner: ModelRunner):
        if not self.kernel_dispatcher.use_context_parallel:
            return
        args = get_schedule()
        # Dynamic chunking and unchunked first requests can exceed the ordinary
        # batch budget. Cover their full supported sequence length at startup.
        tokens = (
            model_runner.model_config.context_len
            if args.enable_dynamic_chunking or (args.chunked_prefill_size or 0) <= 0
            else args.chunked_prefill_size
        )
        self.kernel_dispatcher.extend_kernel.warmup_cp_contexts(
            device=torch.device(model_runner.device),
            max_tokens=tokens,
            value_heads=model_runner.model_config.hf_text_config.linear_num_value_heads,
        )
        pool = self.state_pool
        if pool.cp_size > 1:
            # Slot zero is reserved: compile peer gather, owner store and COW
            # without modifying any request or cached prefix.
            index = torch.zeros(1, dtype=torch.int64, device=model_runner.device)
            pool.clear_slots(index)
            pool.copy_from(index, index)
            state = pool.load_layer_state(0, index)
            pool.store_layer_state(0, index, conv=state.conv, temporal=state.temporal)
            pool.synchronize()

    @profile_method("sglang.attention.prepare_gdn_metadata")
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # An eager forward never runs on a graph bucket's tables.
        self.prefill_graph_metadata = None
        cp_prefill = forward_batch.forward_mode.is_extend_without_speculative()
        cp_metadata = forward_batch.gdn_cp_metadata or forward_batch.prefill_cp_metadata
        if cp_metadata is not None and not cp_prefill:
            raise ValueError("Sharded GDN metadata is only valid for ordinary prefill")
        if cp_prefill and cp_metadata is not None:
            if not self.kernel_dispatcher.use_context_parallel:
                raise ValueError(
                    "GDN CP metadata requires the flashqla_cp prefill backend and sharded CP"
                )
            # Previous batches may have written the source on another owner.
            # Finish those stores before COW, then publish COW/clear before reads.
            self.state_pool.synchronize()
            # CP exports its own recurrence checkpoints. The ordinary GDN
            # tracking indices are unused here and their mask/nonzero queries
            # would synchronize the forward stream with the host.
            self.forward_metadata = None
            cp_metadata.record_stream(torch.cuda.current_stream())
            self.cp_metadata = cp_metadata
            self.cp_conv_metadata = cp_metadata.gdn_conv_metadata
            if len(self.cp_conv_metadata) != forward_batch.batch_size:
                raise ValueError(
                    "GDN convolution metadata must be prepared with the CP batch"
                )
            self.cp_state_slots = self.req_to_token_pool.get_mamba_indices(
                forward_batch.req_pool_indices
            ).long()
            self.cp_checkpoints = forward_batch.mamba_track_checkpoints
            if self.cp_checkpoints is None:
                if forward_batch.mamba_track_mask is not None:
                    raise ValueError("GDN CP requires scheduler checkpoint metadata")
                self.cp_checkpoints = ((),) * forward_batch.batch_size
            if len(self.cp_checkpoints) != forward_batch.batch_size:
                raise ValueError("GDN CP checkpoint metadata must match the batch")
            return
        super().init_forward_metadata(forward_batch)
        self._init_target_verify_qkv_routing(forward_batch)
        self.mis_metadata = None
        if forward_batch.multi_item_delimiter_indices is not None:
            if not self.enable_mis:
                raise ValueError("GDN MIS metadata requires --enable-mis")
            self.mis_metadata = build_gdn_mis_metadata(forward_batch)
        if self.forward_metadata.has_mamba_track_mask:
            if getattr(self.forward_metadata, "mamba_track_mask_indices", None) is None:
                self.forward_metadata.mamba_track_mask_indices = (
                    forward_batch.mamba_track_mask.nonzero(as_tuple=True)[0]
                )
            self.forward_metadata.conv_states_mask_indices = (
                forward_batch.mamba_track_indices[
                    self.forward_metadata.mamba_track_mask_indices
                ]
            )
            if self.kernel_dispatcher.extend_uses_state_checkpoints:
                from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                    maybe_build_flashinfer_checkpoint_plan,
                )

                maybe_build_flashinfer_checkpoint_plan(
                    forward_batch, self.forward_metadata, self.device
                )
        self._init_prefill_conv_block_table(forward_batch)

    def _init_prefill_conv_block_table(self, forward_batch: ForwardBatch) -> None:
        metadata = self.forward_metadata
        metadata.conv_block_table = None
        if (
            not is_cuda()
            or forward_batch.forward_mode not in (ForwardMode.EXTEND, ForwardMode.MIXED)
            or self.mis_metadata is not None
            or forward_batch.tbo_parent_token_range is not None
        ):
            return
        seq_lens = forward_batch.extend_seq_lens_cpu
        if seq_lens is None or len(seq_lens) != metadata.query_start_loc.numel() - 1:
            return
        block_counts = [
            (length + CAUSAL_CONV1D_FWD_BLOCK_M - 1) // CAUSAL_CONV1D_FWD_BLOCK_M
            for length in seq_lens
        ]
        # Amortize table construction only when most of the rectangular grid
        # would be empty: a 4x ratio means over 75% of its blocks do no work.
        # This conservative crossover avoids charging mildly ragged batches
        # for the host table and H2D copy, measured at ~280 us per forward.
        if not block_counts or len(block_counts) * max(block_counts) <= 4 * sum(
            block_counts
        ):
            return

        # Build once per forward for all GDN layers. A rectangular launch on a
        # mixed batch otherwise schedules every decode row for the longest prefill.
        pairs = [
            (seq, block)
            for seq, count in enumerate(block_counts)
            for block in range(count)
        ]
        metadata.conv_block_table = torch.tensor(
            pairs, dtype=torch.int32, device="cpu", pin_memory=is_pin_memory_available()
        ).to(self.device, non_blocking=True)

    def _init_target_verify_qkv_routing(self, forward_batch: ForwardBatch) -> None:
        # CUDA-graph metadata leaves mode and draft count at their defaults.
        if not forward_batch.forward_mode.is_target_verify():
            self._use_strided_target_verify_qkv = False
            return

        metadata = self.forward_metadata
        mamba_pool = self.state_pool
        mamba_cache = mamba_pool.mamba_cache
        is_gdn_replayssm = not mamba_pool.replayssm_is_kda
        use_replayssm_fold = (
            mamba_cache.replayssm_rawv is not None
            and mamba_pool.replayssm_spec_fold
            and is_gdn_replayssm
        )
        use_replayssm_spec = (
            mamba_cache.replayssm_d is not None
            and mamba_pool.replayssm_cache_base is not None
            and is_gdn_replayssm
        )
        self._use_strided_target_verify_qkv = self._target_verify_supports_strided_qkv(
            retrieve_parent_token=metadata.retrieve_parent_token,
            use_replayssm_fold=use_replayssm_fold,
            use_replayssm_spec=use_replayssm_spec,
            ssm_dtype=mamba_cache.temporal.dtype,
            draft_token_num=forward_batch.spec_info.draft_token_num,
        )

    def _replayssm_fold_uses_cutedsl(
        self, ssm_dtype: torch.dtype, draft_token_num: int
    ) -> bool:
        return (
            self.kernel_dispatcher.verify_kernel_is_flashinfer
            and ssm_dtype == torch.bfloat16
            and draft_token_num >= 3
        )

    def _target_verify_supports_strided_qkv(
        self,
        *,
        retrieve_parent_token: Optional[torch.Tensor],
        use_replayssm_fold: bool,
        use_replayssm_spec: bool,
        ssm_dtype: torch.dtype,
        draft_token_num: int,
    ) -> bool:
        # ReplaySSM Triton routes accept strides; the CuTeDSL fold does not.
        if use_replayssm_fold:
            return not self._replayssm_fold_uses_cutedsl(ssm_dtype, draft_token_num)
        if use_replayssm_spec:
            return True
        return self.kernel_dispatcher.target_verify_supports_strided_qkv(
            retrieve_parent_token
        )

    def forward_decode(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        global _fused_decode_proj_conv_fallback_logged
        global _fused_decode_proj_conv_logged
        global _fused_decode_proj_conv_layers_logged
        global _fused_decode_real_tensor_verified_layers

        if _is_hip and isinstance(mixed_qkv, torch.Tensor) and mixed_qkv.shape[0] == 0:
            return mixed_qkv.new_zeros((1, 0, layer.num_v_heads, layer.head_v_dim))

        layer_cache = self.state_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices
        # GDN ReplaySSM (slice 1a): per-layer ring slices + the once-per-forward
        # per-row write cursor. All None unless --enable-linear-replayssm, so the
        # legacy dispatch below is byte-identical when the flag is off.
        replayssm_write_pos = self.forward_metadata.replayssm_write_pos
        # GDN ReplaySSM (slice 2b): per-row force-flush at radix track
        # boundaries (None unless --enable-linear-replayssm). When present the
        # kernel folds the ring into temporal[slot] on the snapshot steps.
        replayssm_force_flush = self.forward_metadata.replayssm_force_flush
        replayssm_d = layer_cache.replayssm_d
        replayssm_k = layer_cache.replayssm_k
        replayssm_g = layer_cache.replayssm_g

        return_z = False
        conv_already_applied = False
        if isinstance(mixed_qkv, tuple):
            if len(mixed_qkv) != 2:
                raise ValueError(
                    "Fused GDN decode projection input must be "
                    "(projected_qkvz, projected_ba)"
                )
            projected_qkvz, projected_ba = mixed_qkv
            eligible, eligibility_reason = (
                can_use_fused_qkvzba_causal_conv1d_update_contiguous(
                    projected_qkvz,
                    projected_ba,
                    conv_states,
                    layer.conv_weights,
                    layer.bias,
                    activation=layer.activation,
                )
            )
            if eligible:
                qkv_dim = layer.q_dim + layer.k_dim + layer.v_dim
                fused_backend = "triton_direct_oracle_exact"
                if not _fused_decode_proj_conv_logged:
                    rank0_log("Using fused GDN decode QKVZ/BA unpack + indexed Conv1D.")
                    _fused_decode_proj_conv_logged = True
                if (
                    _fused_decode_log_layer_hits or _fused_decode_verify_real_tensors
                ) and layer.layer_id not in _fused_decode_proj_conv_layers_logged:
                    rank0_log(
                        "GDN_FUSED_DECODE_BACKEND "
                        f"layer_id={layer.layer_id} backend={fused_backend} "
                        f"batch={projected_qkvz.shape[0]} "
                        f"qkv_dim={qkv_dim} state_shape={tuple(conv_states.shape)} "
                        f"state_indices_dtype={cache_indices.dtype}"
                    )
                    _fused_decode_proj_conv_layers_logged.add(layer.layer_id)

                # Compare real activations against the direct-Triton update on a
                # compact state copy, leaving the live cache to the candidate.
                verify_real_tensors = (
                    _fused_decode_verify_real_tensors
                    and layer.layer_id not in _fused_decode_real_tensor_verified_layers
                )
                if verify_real_tensors:
                    if bool(torch.any(cache_indices < 0).item()):
                        raise AssertionError(
                            "Real-tensor GDN fusion verification requires "
                            "non-padding cache indices"
                        )
                    ref_indices = torch.arange(
                        cache_indices.numel(),
                        device=cache_indices.device,
                        dtype=torch.int32,
                    )
                    ref_state = torch.index_select(
                        conv_states, 0, cache_indices.to(torch.int64)
                    )
                    ref_mixed_qkv, ref_z, ref_b, ref_a = (
                        fused_qkvzba_split_reshape_cat_contiguous(
                            projected_qkvz,
                            projected_ba,
                            layer.num_q_heads,
                            layer.num_v_heads,
                            layer.head_q_dim,
                            layer.head_v_dim,
                        )
                    )
                    ref_mixed_qkv = causal_conv1d_update(
                        ref_mixed_qkv,
                        ref_state,
                        layer.conv_weights,
                        layer.bias,
                        layer.activation,
                        conv_state_indices=ref_indices,
                    )

                mixed_qkv, z, b, a = fused_qkvzba_causal_conv1d_update_contiguous(
                    projected_qkvz,
                    projected_ba,
                    conv_states,
                    layer.conv_weights,
                    layer.bias,
                    cache_indices,
                    qkv_dim=qkv_dim,
                    v_dim=layer.v_dim,
                    num_v_heads=layer.num_v_heads,
                    head_v_dim=layer.head_v_dim,
                    activation=layer.activation,
                )
                if verify_real_tensors:
                    candidate_state = torch.index_select(
                        conv_states, 0, cache_indices.to(torch.int64)
                    )
                    named_pairs = (
                        ("qkv", mixed_qkv, ref_mixed_qkv),
                        ("z", z, ref_z),
                        ("b", b, ref_b),
                        ("a", a, ref_a),
                        ("state", candidate_state, ref_state),
                    )
                    report = []
                    mismatch = False
                    for tensor_name, candidate, reference in named_pairs:
                        diff = (candidate.float() - reference.float()).abs()
                        nonzero = int(torch.count_nonzero(diff).item())
                        mismatch |= nonzero != 0
                        report.append(
                            f"{tensor_name}_nonzero={nonzero}/"
                            f"{diff.numel()} {tensor_name}_max="
                            f"{diff.max().item()}"
                        )
                    rank0_log(
                        "GDN_FUSED_REAL_TENSOR_PARITY "
                        f"layer_id={layer.layer_id} backend={fused_backend} "
                        + " ".join(report)
                    )
                    _fused_decode_real_tensor_verified_layers.add(layer.layer_id)
                    if mismatch:
                        raise AssertionError(
                            "GDN fused real-tensor parity failed at "
                            f"layer_id={layer.layer_id}; " + " ".join(report)
                        )
                conv_already_applied = True
            else:
                if not _fused_decode_proj_conv_fallback_logged:
                    rank0_log(
                        "Falling back from fused GDN decode projection/Conv1D: "
                        f"{eligibility_reason}"
                    )
                    _fused_decode_proj_conv_fallback_logged = True
                mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat_contiguous(
                    projected_qkvz,
                    projected_ba,
                    layer.num_q_heads,
                    layer.num_v_heads,
                    layer.head_q_dim,
                    layer.head_v_dim,
                )
            return_z = True
        else:
            assert isinstance(mixed_qkv, torch.Tensor)

        if not conv_already_applied:
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                conv_states,
                layer.conv_weights,
                layer.bias,
                layer.activation,
                conv_state_indices=cache_indices,
            )

        # Skip split + reshape + separate gating kernel by consuming
        # the packed mixed_qkv directly in a single fused Triton kernel.
        if self.kernel_dispatcher.supports_packed_decode:
            core_attn_out = self.kernel_dispatcher.packed_decode(
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                scale=layer.head_k_dim**-0.5,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                num_v_heads=layer.num_v_heads,
                head_v_dim=layer.head_v_dim,
                replayssm_d=replayssm_d,
                replayssm_k=replayssm_k,
                replayssm_g=replayssm_g,
                replayssm_write_pos=replayssm_write_pos,
                replayssm_force_flush=replayssm_force_flush,
            )
            self._track_mamba_state_decode(
                forward_batch, conv_states, ssm_states, cache_indices, layer.layer_id
            )
            return (core_attn_out, z) if return_z else core_attn_out

        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )
        # Reshape from [bs, h*d] to [1, bs, h, d]
        bs = forward_batch.batch_size
        query = query.view(1, bs, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, bs, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, bs, layer.num_v_heads, layer.head_v_dim)

        core_attn_out = self.kernel_dispatcher.decode(
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )

        self._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices, layer.layer_id
        )

        return (core_attn_out, z) if return_z else core_attn_out

    def _convolve_prefill(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        conv_states: torch.Tensor,
        cache_indices: torch.Tensor,
    ) -> torch.Tensor:
        block_table = self.forward_metadata.conv_block_table
        conv_kwargs = {"block_table": block_table} if block_table is not None else {}
        return causal_conv1d_fn(
            mixed_qkv.transpose(0, 1),
            layer.conv_weights,
            layer.bias,
            activation=layer.activation,
            conv_states=conv_states,
            has_initial_state=forward_batch.extend_prefix_lens > 0,
            cache_indices=cache_indices,
            query_start_loc=self.forward_metadata.query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            **conv_kwargs,
        ).transpose(0, 1)[: mixed_qkv.shape[0]]

    @staticmethod
    def _prefill_gates(
        layer: RadixLinearAttention, a: torch.Tensor, b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)

    @staticmethod
    def _split_prefill_qkv(
        layer: RadixLinearAttention, mixed_qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """forward_extend's q / k / v split for a plain extend on CUDA."""
        if layer.q_dim + layer.k_dim + layer.v_dim <= MAX_FUSED_QKV_SPLIT_DIM:
            return fused_qkv_split_gdn_prefill(
                mixed_qkv,
                layer.num_q_heads,
                layer.num_k_heads,
                layer.num_v_heads,
                layer.head_q_dim,
                layer.head_k_dim,
                layer.head_v_dim,
            )
        seq_len = mixed_qkv.shape[0]
        query, key, value = torch.split(
            mixed_qkv, [layer.q_dim, layer.k_dim, layer.v_dim], dim=-1
        )
        return (
            query.view(1, seq_len, layer.num_q_heads, layer.head_q_dim),
            key.view(1, seq_len, layer.num_k_heads, layer.head_k_dim),
            value.view(1, seq_len, layer.num_v_heads, layer.head_v_dim),
        )

    def _forward_extend_in_graph(
        self,
        *,
        layer: RadixLinearAttention,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """forward_extend's plain-extend chain on a captured bucket's static
        tables, so every launch grid depends only on the bucket. Rows past the
        batch belong to no sequence; chunk_fwd_o leaves them zero."""
        meta = self.prefill_graph_metadata
        layer_cache = self.state_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        if meta.track:
            # Snapshot the conv windows before the conv kernel updates the cache.
            gdn_track_conv_window(
                conv_pool=conv_states,
                x=mixed_qkv,
                track_mode=meta.track_mode,
                track_conv_start=meta.track_conv_start,
                track_dst=meta.track_dst,
                num_tokens=meta.num_logical_tokens,
            )
        mixed_qkv = causal_conv1d_fn_triton(
            mixed_qkv.transpose(0, 1),
            layer.conv_weights,
            layer.bias,
            conv_states=conv_states,
            query_start_loc=meta.cu_seqlens,
            seq_lens_cpu=None,
            cache_indices=meta.cache_indices,
            has_initial_state=meta.has_initial_state,
            activation=layer.activation,
            block_table=meta.conv_block_table,
        ).transpose(0, 1)
        query, key, value = self._split_prefill_qkv(layer, mixed_qkv)
        g, beta = self._prefill_gates(layer, a, b)
        core_attn_out, _, h = self.kernel_dispatcher.extend(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            ssm_states=ssm_states,
            cache_indices=meta.cache_indices,
            query_start_loc=meta.cu_seqlens,
            chunk_indices=meta.chunk_indices,
            chunk_offsets=meta.chunk_offsets,
        )
        if meta.track:
            gdn_track_ssm_state(
                ssm_pool=ssm_states,
                h=h,
                track_mode=meta.track_mode,
                track_h_row=meta.track_h_row,
                track_dst=meta.track_dst,
                cache_indices=meta.cache_indices,
            )
        return core_attn_out
    def _forward_extend_cp(self, layer, batch, raw, a, b):
        """GDN core only: convolution halos, FlashQLA recurrence and radix state."""
        frame = self.cp_metadata
        kernel = self.kernel_dispatcher.extend_kernel
        cache = self.state_layer_cache(layer.layer_id)
        pool = self.state_pool
        cache_layer = self.req_to_token_pool.mamba_map[layer.layer_id]
        if raw.shape[0]:
            g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
            g, beta = g[0], beta[0]
        else:
            g = beta = raw.new_empty((0, layer.num_v_heads), dtype=torch.float32)
        results, start = [], 0
        history_width = layer.conv_weights.shape[-1] - 1
        head_geometry = (
            layer.num_q_heads,
            layer.num_k_heads,
            layer.num_v_heads,
            layer.head_q_dim,
            layer.head_k_dim,
            layer.head_v_dim,
        )
        for sequence, length in enumerate(frame.lengths):
            if not length:
                continue
            width = (length + 64 * frame.world - 1) // (64 * frame.world) * 64
            count = min(width, max(0, length - frame.rank * width))
            local_raw = raw[start : start + count]
            # Keep pool indices on device; Python slicing a CUDA scalar would
            # read it back and wait for previously queued forward work.
            slot = self.cp_state_slots[sequence : sequence + 1]
            state = pool.load_layer_state(cache_layer, slot, temporal=frame.rank == 0)
            initial = state.temporal.float()
            history = state.conv[0][0]
            if frame.prefixes[sequence] == 0:
                initial = torch.zeros_like(initial)
                history = torch.zeros_like(history)

            # The global length check makes every rank choose the same exchange.
            # Ragged and empty shards retain the general convolution path.
            if (
                frame.world == 4
                and length % (64 * frame.world) == 0
                and kernel.symm_state_enabled()
                and supports_causal_conv1d_peer(
                    local_raw,
                    history,
                    layer.conv_weights,
                    head_geometry=head_geometry,
                    bias=layer.bias,
                    activation=layer.activation,
                )
                and not torch.cuda.is_current_stream_capturing()
            ):
                if self._conv_workspace is None:
                    communicator = TorchSymmMemCommunicator.get(frame.group, raw.device)
                    self._conv_workspace = communicator.allocate_peer_workspace(
                        (raw.shape[1], history_width), raw.dtype
                    )
                qkv, preceding, terminal_history = causal_conv1d_peer_with_halo(
                    local_raw,
                    history,
                    layer.conv_weights,
                    self._conv_workspace,
                    head_geometry=head_geometry,
                    bias=layer.bias,
                    activation=layer.activation,
                )
            else:
                convolved, preceding, terminal_history = convolve_gdn_context_shard(
                    frame,
                    layer,
                    local_raw,
                    history,
                    length,
                    width,
                    self.cp_conv_metadata[sequence],
                )
                qkv = _prepare_qkv_inputs(convolved, layer, width)
            inputs = qkv + _padded_inputs(
                [g[start : start + count], beta[start : start + count]], width
            )
            checkpoints = self.cp_checkpoints[sequence]
            internal_positions = tuple(
                position for position, _ in checkpoints if 0 < position < length
            )
            output, final, states = self.kernel_dispatcher.extend(
                *inputs,
                ssm_states=cache.temporal,
                cache_indices=slot,
                query_start_loc=self.cp_conv_metadata[sequence]["query_start_loc"],
                cp_metadata=frame,
                initial_state=initial,
                inplace_update=False,
                checkpoint_indices=internal_positions,
                state_dtype=cache.temporal.dtype,
            )
            results.append(output[0, :count])
            pool.store_layer_state(
                cache_layer, slot, conv=[terminal_history.unsqueeze(0)], temporal=final
            )

            state_iter = iter(states)
            for target_length, target_slot in checkpoints:
                if target_length == 0:
                    checkpoint_state, checkpoint_conv = initial, history
                elif target_length == length:
                    checkpoint_state, checkpoint_conv = final, terminal_history
                elif 0 < target_length < length:
                    checkpoint_state = next(state_iter)
                    owner = (target_length - 1) // width
                    local_end = min(count, max(0, target_length - frame.rank * width))
                    checkpoint_conv = _tail(
                        preceding, local_raw[:local_end], history_width
                    )
                    kernel.broadcast_state(checkpoint_conv, owner)
                else:
                    raise ValueError("Invalid GDN radix checkpoint inside CP chunk")
                pool.store_layer_state(
                    cache_layer,
                    target_slot,
                    conv=[checkpoint_conv.unsqueeze(0)],
                    temporal=checkpoint_state,
                )
            start += count
        if start != raw.shape[0]:
            raise ValueError("GDN CP layout does not cover local token rows")
        if not start:
            return raw.new_empty((1, 0, layer.num_v_heads, layer.head_v_dim))
        return (
            results[0] if len(results) == 1 else torch.cat(results, dim=0)
        ).unsqueeze(0)

    @profile_method("sglang.attention.gdn.extend")
    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        assert isinstance(mixed_qkv, torch.Tensor)
        if (
            forward_batch.forward_mode.is_extend_without_speculative()
            and self.kernel_dispatcher.use_context_parallel
        ):
            if (
                forward_batch.prefill_cp_metadata is None
                and forward_batch.gdn_cp_metadata is None
            ):
                raise ValueError("GDN CP prefill requires initialized batch metadata")
            return self._forward_extend_cp(layer, forward_batch, mixed_qkv, a, b)
        seq_len = mixed_qkv.shape[0]

        if _is_hip and seq_len == 0:
            return mixed_qkv.new_zeros((1, 0, layer.num_v_heads, layer.head_v_dim))

        is_target_verify = forward_batch.forward_mode.is_target_verify()
        if self.prefill_graph_metadata is not None and not is_target_verify:
            return self._forward_extend_in_graph(
                layer=layer, mixed_qkv=mixed_qkv, a=a, b=b
            )
        forward_metadata = self.forward_metadata

        query_start_loc = forward_metadata.query_start_loc
        cache_indices = forward_metadata.mamba_cache_indices
        retrieve_next_token = forward_metadata.retrieve_next_token
        retrieve_next_sibling = forward_metadata.retrieve_next_sibling
        retrieve_parent_token = forward_metadata.retrieve_parent_token

        mamba_cache_params = self.state_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal
        if self.mis_metadata is not None:
            if is_target_verify:
                raise ValueError("GDN MIS does not support target verify")
            return self._forward_extend_mis(
                layer=layer,
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                conv_states=conv_states,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
            )
        if is_target_verify:
            assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)
            intermediate_state_cache = mamba_cache_params.intermediate_ssm
            intermediate_conv_window_cache = (
                mamba_cache_params.intermediate_conv_window[0]
            )
            intermediate_state_indices = select_verify_intermediate_state_indices(
                self.verify_intermediate_state_indices,
                forward_batch.req_pool_indices,
                cache_indices[: query_start_loc.shape[0] - 1] >= 0,
                self.req_to_token_pool.size,
            )
            mamba_pool = self.state_pool
            use_replayssm_fold = (
                mamba_cache_params.replayssm_rawv is not None
                and mamba_pool.replayssm_spec_fold
                and not mamba_pool.replayssm_is_kda
            )
            use_replayssm_spec = (
                mamba_cache_params.replayssm_d is not None
                and mamba_pool.replayssm_cache_base is not None
                and not mamba_pool.replayssm_is_kda
            )

        # Page-major envelope: the prefill kernels (CUDA causal_conv1d_fwd,
        # chunk_gated_delta_rule) write state back in place assuming a contiguous
        # slot layout, so they silently drop the write to the strided envelope
        # pool. Run them on contiguous per-sequence copies and scatter the result
        # back. No-op for the default contiguous pool.
        # CPU kernels (causal_conv1d_fwd_cpu, chunk_gated_delta_rule_cpu) use
        # proper indexed writes and handle non-contiguous pools directly via
        # cache_indices, so the gather/scatter round-trip is unnecessary on CPU.
        # TODO(ch-wan): drop these .contiguous() copies by making the prefill conv
        # and chunk_gated_delta_rule kernels honor the pool's real slot stride +
        # int64 indexing, like packed_decode / causal_conv1d_update already do.
        needs_state_gather = (
            self.requires_contiguous_prefill_state
            and (not is_target_verify)
            and (not is_cpu())
            and (not conv_states.is_contiguous() or not ssm_states.is_contiguous())
        )
        if needs_state_gather:
            # Row 0 of the copies is the padding sink, as in the pool: the
            # prefill kernels treat state index 0 as padding.
            (
                (conv_states_contig, ssm_states_contig),
                state_cache_indices,
                staged_pool_rows,
            ) = stage_pool_states((conv_states, ssm_states), cache_indices)
        else:
            conv_states_contig = conv_states
            ssm_states_contig = ssm_states
            state_cache_indices = cache_indices

        if is_target_verify:
            batch_size = seq_len // forward_batch.spec_info.draft_token_num
            draft_token_num = forward_batch.spec_info.draft_token_num
            mixed_qkv_reshaped = mixed_qkv.view(
                batch_size, draft_token_num, -1
            ).transpose(1, 2)
            mixed_qkv_processed = causal_conv1d_update(
                mixed_qkv_reshaped,
                conv_states,
                layer.conv_weights,
                layer.bias,
                layer.activation,
                conv_state_indices=cache_indices[:batch_size],
                intermediate_conv_window=intermediate_conv_window_cache,
                intermediate_state_indices=intermediate_state_indices[:batch_size],
                retrieve_next_token=retrieve_next_token,
                retrieve_next_sibling=retrieve_next_sibling,
                retrieve_parent_token=retrieve_parent_token,
            )
            mixed_qkv = mixed_qkv_processed.transpose(1, 2).view(seq_len, -1)
        else:
            if forward_metadata.has_mamba_track_mask:
                mixed_qkv_to_track = mixed_qkv[
                    forward_metadata.track_conv_indices
                ].transpose(1, 2)
                conv_states[forward_metadata.conv_states_mask_indices] = (
                    mixed_qkv_to_track
                )

            mixed_qkv = self._convolve_prefill(
                layer,
                forward_batch,
                mixed_qkv,
                conv_states_contig,
                state_cache_indices,
            )

        actual_seq_len = mixed_qkv.shape[0]
        qkv_dim = layer.q_dim + layer.k_dim + layer.v_dim
        if (
            (is_cuda() or is_hip() or is_xpu())
            and qkv_dim <= MAX_FUSED_QKV_SPLIT_DIM
            and not self._use_strided_target_verify_qkv
        ):
            query, key, value = fused_qkv_split_gdn_prefill(
                mixed_qkv,
                layer.num_q_heads,
                layer.num_k_heads,
                layer.num_v_heads,
                layer.head_q_dim,
                layer.head_k_dim,
                layer.head_v_dim,
            )
        else:
            query, key, value = torch.split(
                mixed_qkv,
                [layer.q_dim, layer.k_dim, layer.v_dim],
                dim=-1,
            )
            query = query.view(1, actual_seq_len, layer.num_q_heads, layer.head_q_dim)
            key = key.view(1, actual_seq_len, layer.num_k_heads, layer.head_k_dim)
            value = value.view(1, actual_seq_len, layer.num_v_heads, layer.head_v_dim)

        if is_target_verify:
            # ReplaySSM verify protocols: fold-every-commit (ring-write during
            # verify, fold on commit), circular ring, or the snapshotting
            # fallback when neither ring is allocated.
            if use_replayssm_fold:
                core_attn_out = self._replayssm_fold_target_verify(
                    layer=layer,
                    query=query,
                    key=key,
                    value=value,
                    a=a,
                    b=b,
                    layer_cache=mamba_cache_params,
                    ssm_states=ssm_states,
                    cache_indices=cache_indices,
                    query_start_loc=query_start_loc,
                    retrieve_parent_token=retrieve_parent_token,
                )
            elif use_replayssm_spec:
                core_attn_out = self._replayssm_target_verify(
                    layer=layer,
                    query=query,
                    key=key,
                    value=value,
                    a=a,
                    b=b,
                    mamba_pool=mamba_pool,
                    layer_cache=mamba_cache_params,
                    cache_indices=cache_indices,
                    replay_indices=forward_batch.req_pool_indices,
                    query_start_loc=query_start_loc,
                    draft_token_num=forward_batch.spec_info.draft_token_num,
                )
            else:
                # The recurrent fallback needs the per-draft snapshots, which
                # the pool gates OFF under --enable-linear-replayssm-spec (the
                # same flag that makes `use_replayssm_spec` true above), so
                # this branch is unreachable with a None buffer by
                # construction -- keep it loud rather than silently frozen.
                assert intermediate_state_cache is not None, (
                    "recurrent target_verify fallback requires intermediate_ssm, "
                    "which is not allocated under --enable-linear-replayssm-spec"
                )
                core_attn_out = self.kernel_dispatcher.target_verify(
                    A_log=layer.A_log,
                    dt_bias=layer.dt_bias,
                    q=query,
                    k=key,
                    v=value,
                    a=a,
                    b=b,
                    ssm_states=ssm_states,
                    cache_indices=cache_indices,
                    query_start_loc=query_start_loc,
                    intermediate_states_buffer=intermediate_state_cache,
                    intermediate_state_indices=intermediate_state_indices,
                    cache_steps=forward_batch.spec_info.draft_token_num,
                    retrieve_parent_token=retrieve_parent_token,
                    stable_rows=pp_spec_stable_rows_enabled(),
                )
        else:
            g, beta = self._prefill_gates(layer, a, b)
            h_track_buf = None
            if (
                forward_metadata.has_mamba_track_mask
                and forward_metadata.track_ssm_h_src.numel() > 0
                and self.kernel_dispatcher.extend_kernel.supports_track_state_snapshot
            ):
                # Snapshot-capable kernels can supply just the selected FP32
                # checkpoint instead of materializing all per-chunk states.
                assert forward_metadata.track_chunk_idx is not None
                h_track_buf = torch.empty(
                    (len(cache_indices), *ssm_states.shape[1:]),
                    dtype=torch.float32,
                    device=ssm_states.device,
                )
            core_attn_out, last_recurrent_state, h = self.kernel_dispatcher.extend(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                ssm_states=ssm_states_contig,
                cache_indices=state_cache_indices,
                query_start_loc=query_start_loc,
                return_cp_output=kwargs.get("return_cp_output", False),
                seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
                output_intermediate_states=(
                    forward_metadata.has_mamba_track_mask
                    and forward_metadata.track_ssm_h_src.numel() > 0
                ),
                state_checkpoint_cu_starts=(
                    forward_metadata.state_checkpoint_cu_starts
                ),
                num_state_checkpoints=forward_metadata.num_state_checkpoints,
                state_checkpoint_every_n_tokens=(
                    forward_metadata.state_checkpoint_every_n_tokens
                ),
                output=kwargs.get("linear_attn_output"),
                layer_id=layer.layer_id,
                extend_prefix_lens=forward_batch.extend_prefix_lens,
                track_state=h_track_buf,
                track_chunk_idx=(
                    forward_metadata.track_chunk_idx
                    if h_track_buf is not None
                    else None
                ),
            )

            if is_npu() and last_recurrent_state is not None:
                last_recurrent_state = last_recurrent_state.to(
                    ssm_states.dtype, copy=False
                )
                ssm_states[cache_indices] = last_recurrent_state

            if needs_state_gather:
                # Scatter the in-place-updated contiguous copies back to the
                # strided envelope pool (advanced indexing handles the strides).
                unstage_pool_states(
                    (conv_states, ssm_states),
                    (conv_states_contig, ssm_states_contig),
                    staged_pool_rows,
                )

            if forward_metadata.has_mamba_track_mask:
                self._track_mamba_state_extend(
                    forward_batch,
                    h,
                    ssm_states,
                    forward_metadata,
                    h_track_buf=h_track_buf,
                )

        return core_attn_out

    def _forward_extend_mis(
        self,
        *,
        layer: RadixLinearAttention,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
    ) -> torch.Tensor:
        metadata = self.mis_metadata
        assert metadata is not None

        conv_states[cache_indices] = 0
        ssm_states[cache_indices] = 0
        output = mixed_qkv.new_zeros(
            1, mixed_qkv.shape[0], layer.num_v_heads, layer.head_v_dim
        )

        if metadata.query_token_indices.numel() > 0:
            query_cache_indices = cache_indices[metadata.query_request_indices]
            query_output = self._forward_mis_segments(
                layer=layer,
                mixed_qkv=mixed_qkv[metadata.query_token_indices],
                a=a[metadata.query_token_indices],
                b=b[metadata.query_token_indices],
                conv_states=conv_states,
                conv_cache_indices=query_cache_indices,
                has_initial_states=torch.zeros(
                    len(metadata.query_seq_lens_cpu),
                    dtype=torch.bool,
                    device=mixed_qkv.device,
                ),
                query_start_loc=metadata.query_cu_seqlens,
                seq_lens_cpu=metadata.query_seq_lens_cpu,
                ssm_states=ssm_states,
                ssm_cache_indices=query_cache_indices,
                inplace_update=True,
            )
            output[:, metadata.query_token_indices] = query_output

        item_ssm_indices = cache_indices[metadata.item_request_indices]
        item_conv_states = conv_states[item_ssm_indices]
        item_conv_indices = torch.arange(
            item_ssm_indices.shape[0],
            dtype=cache_indices.dtype,
            device=cache_indices.device,
        )
        item_output = self._forward_mis_segments(
            layer=layer,
            mixed_qkv=mixed_qkv[metadata.item_token_indices],
            a=a[metadata.item_token_indices],
            b=b[metadata.item_token_indices],
            conv_states=item_conv_states,
            conv_cache_indices=item_conv_indices,
            has_initial_states=torch.ones(
                len(metadata.item_seq_lens_cpu),
                dtype=torch.bool,
                device=mixed_qkv.device,
            ),
            query_start_loc=metadata.item_cu_seqlens,
            seq_lens_cpu=metadata.item_seq_lens_cpu,
            ssm_states=ssm_states,
            ssm_cache_indices=item_ssm_indices,
            inplace_update=False,
        )
        output[:, metadata.item_token_indices] = item_output
        return output

    def _forward_mis_segments(
        self,
        *,
        layer: RadixLinearAttention,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        conv_states: torch.Tensor,
        conv_cache_indices: torch.Tensor,
        has_initial_states: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens_cpu: list[int],
        ssm_states: torch.Tensor,
        ssm_cache_indices: torch.Tensor,
        inplace_update: bool,
    ) -> torch.Tensor:
        mixed_qkv = causal_conv1d_fn(
            mixed_qkv.transpose(0, 1),
            layer.conv_weights,
            layer.bias,
            activation=layer.activation,
            conv_states=conv_states,
            has_initial_state=has_initial_states,
            cache_indices=conv_cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=seq_lens_cpu,
        ).transpose(0, 1)

        qkv_dim = layer.q_dim + layer.k_dim + layer.v_dim
        if (is_cuda() or is_hip()) and qkv_dim <= MAX_FUSED_QKV_SPLIT_DIM:
            query, key, value = fused_qkv_split_gdn_prefill(
                mixed_qkv,
                layer.num_q_heads,
                layer.num_k_heads,
                layer.num_v_heads,
                layer.head_q_dim,
                layer.head_k_dim,
                layer.head_v_dim,
            )
        else:
            query, key, value = torch.split(
                mixed_qkv, [layer.q_dim, layer.k_dim, layer.v_dim], dim=-1
            )
            num_tokens = mixed_qkv.shape[0]
            query = query.view(1, num_tokens, layer.num_q_heads, layer.head_q_dim)
            key = key.view(1, num_tokens, layer.num_k_heads, layer.head_k_dim)
            value = value.view(1, num_tokens, layer.num_v_heads, layer.head_v_dim)

        g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
        output, _, _ = self.kernel_dispatcher.extend(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            ssm_states=ssm_states,
            cache_indices=ssm_cache_indices,
            query_start_loc=query_start_loc,
            inplace_update=inplace_update,
        )
        return output

    def _replayssm_fold_target_verify(
        self,
        *,
        layer: RadixLinearAttention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        layer_cache: "MambaPool.SpeculativeState",
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        retrieve_parent_token: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Ring-writing verify; the commit fold replays the accepted prefix
        into ``temporal``. Uses the vendored CuTe DSL MTP kernel when the
        dispatcher selected the FlashInfer bf16-state verify, else the Triton
        recurrent kernel (both store the same raw window)."""
        from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (
            fused_sigmoid_gating_delta_rule_update,
        )

        assert retrieve_parent_token is None, (
            "ReplaySSM fold-every-commit supports a linear draft chain only "
            "(topk <= 1); EAGLE tree verify must use the recurrent verify."
        )
        seq_len = query.shape[1]
        batch_size = query_start_loc.shape[0] - 1
        draft_token_num = seq_len // batch_size
        if self._replayssm_fold_uses_cutedsl(ssm_states.dtype, draft_token_num):
            from sglang.kernels.ops.attention.cutedsl_gdn_mtp_ring import (
                gated_delta_rule_mtp,
            )

            num_v_heads = value.shape[2]
            head_v_dim = value.shape[3]
            out = gated_delta_rule_mtp(
                A_log=layer.A_log.detach(),
                a=a.view(batch_size, draft_token_num, num_v_heads),
                dt_bias=layer.dt_bias.detach(),
                q=query.view(batch_size, draft_token_num, *query.shape[2:]),
                k=key.view(batch_size, draft_token_num, *key.shape[2:]),
                v=value.view(batch_size, draft_token_num, num_v_heads, head_v_dim),
                b=b.view(batch_size, draft_token_num, num_v_heads),
                initial_state_source=ssm_states,
                initial_state_indices=cache_indices,
                use_qk_l2norm_in_kernel=True,
                disable_state_update=True,
                cache_ring=True,
                replayssm_rawv=layer_cache.replayssm_rawv,
                replayssm_rawk=layer_cache.replayssm_rawk,
                replayssm_g=layer_cache.replayssm_g,
                replayssm_beta=layer_cache.replayssm_beta,
            )
            return out.view(1, seq_len, num_v_heads, head_v_dim)
        return fused_sigmoid_gating_delta_rule_update(
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=False,
            disable_state_update=True,
            cache_ring=True,
            replayssm_rawv=layer_cache.replayssm_rawv,
            replayssm_rawk=layer_cache.replayssm_rawk,
            replayssm_g=layer_cache.replayssm_g,
            replayssm_beta=layer_cache.replayssm_beta,
        )

    def _replayssm_target_verify(
        self,
        *,
        layer: RadixLinearAttention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        mamba_pool: MambaPool,
        layer_cache: "MambaPool.SpeculativeState",
        cache_indices: torch.Tensor,
        replay_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        draft_token_num: int,
    ) -> torch.Tensor:
        """ReplaySSM GDN spec-verify (Part B of #28511).

        Reconstructs the verify output for the whole draft window from the frozen
        checkpoint (``temporal``) + the per-slot circular ``(d, k, g)`` ring, and
        appends this window's drafts to the compact ``(d, k, g)`` rings. BF16
        checkpoints also keep low parts for compensated materialization. The rings
        are per-layer (sliced via ``mamba2_layer_cache``), while the cursors
        (write_pos, cache_base, is_flush) are request-slot pool attributes shared
        by all GDN layers of the step. The cursors advance once per accepted step;
        here we only read them and write this step's ring entries. GDN has K == V,
        so ``temporal``
        ([slots, HV, K, V]) is consumed directly as the kernel's [slots, HV, V, K]
        checkpoint.
        """
        from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
            gdn_replayssm_spec_decode,
        )

        H, K = layer.num_k_heads, layer.head_k_dim
        HV, V = layer.num_v_heads, layer.head_v_dim
        # q/k/v may be [1, seq, *] (fallback split) or [seq, *] (fused split);
        # derive the packed token count from numel so both layouts flatten.
        seq_len = query.numel() // (H * K)
        q = query.reshape(seq_len, H, K)
        k = key.reshape(seq_len, H, K)
        v = value.reshape(seq_len, HV, V)
        a = a.reshape(seq_len, HV)
        b = b.reshape(seq_len, HV)
        d_cache = layer_cache.replayssm_d  # [slots, HV, L, V]
        max_cache_len = d_cache.shape[-2]  # ring length L
        out = q.new_empty(seq_len, HV, V)
        gdn_replayssm_spec_decode(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            checkpoint_state=layer_cache.temporal,
            d_cache=d_cache,
            k_cache=layer_cache.replayssm_k,
            g_cache=layer_cache.replayssm_g,
            # BF16 compact D/K low parts; None for fp32 checkpoints.
            rawv_cache=layer_cache.replayssm_rawv,
            rawk_cache=layer_cache.replayssm_rawk,
            beta_cache=layer_cache.replayssm_beta,
            out=out,
            query_start_loc=query_start_loc,
            ssm_state_indices=cache_indices,
            replay_indices=replay_indices,
            # Request-slot cursors live on the pool (shared across all GDN layers)
            # and advance only after acceptance, so the decode-path
            # forward_metadata.replayssm_write_pos snapshot is not used here.
            write_pos=mamba_pool.replayssm_spec_write_pos,
            cache_base=mamba_pool.replayssm_cache_base,
            is_flush=mamba_pool.replayssm_is_flush,
            max_cache_len=max_cache_len,
            max_spec_len=draft_token_num,
            scale=K**-0.5,
            use_qk_l2norm_in_kernel=True,
            # SGLang marks invalid/padding requests with a negative mamba slot
            # index (valid slots start at 0), so the kernel's "null block"
            # sentinel is -1, not the vLLM default of 0.
            null_block_id=-1,
            # Capacity folds are committed once across every GDN layer after
            # acceptance; the active path is therefore a single launch/layer.
            launch_mode="verify",
        )
        # Match the recurrent target_verify output shape (== value.shape).
        return out.reshape(value.shape)


@cache
def warmup_gdn_prefill(
    *,
    device,
    dtype,
    key_heads,
    value_heads,
    key_head_dim,
    value_head_dim,
    conv_kernel_size,
    state_dtype,
    group,
    projection_stride,
    local_tokens,
):
    """Warm the selected prefill operators before the runner allocates live KV/state."""
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
        warmup_gdn_projection,
    )
    from sglang.kernels.ops.mamba.causal_conv1d_triton import (
        autotune_causal_conv1d_peer,
        warmup_causal_conv1d_context_parallel,
    )
    from sglang.srt.layers.attention.linear.kernels.gdn_flashqla import (
        _warmup_flashqla_gdn,
    )

    geometry = (key_heads, value_heads, key_head_dim, value_head_dim)
    channels = 2 * key_heads * key_head_dim + value_heads * value_head_dim
    with torch.cuda.device(device), torch.inference_mode():
        warmup_gdn_projection(device, dtype, geometry, projection_stride)
        warmup_causal_conv1d_context_parallel(
            device, dtype, channels, conv_kernel_size, projection_stride
        )
        _warmup_flashqla_gdn(
            device,
            geometry=geometry,
            dtype=dtype,
            state_dtype=state_dtype,
            cp_group=group,
            cp_size=dist.get_world_size(group),
            cp_rank=dist.get_rank(group),
        )
        if TorchSymmMemCommunicator.multicast_unavailable_reason(group) is None:
            communicator = TorchSymmMemCommunicator.get(group, device)
            history = torch.zeros(
                (channels, conv_kernel_size - 1), device=device, dtype=dtype
            )
            workspace = communicator.allocate_broadcast_workspace(
                history.numel() * history.element_size() * 2
            )
            try:
                for owner in range(dist.get_world_size(group)):
                    communicator.broadcast(workspace, history, owner)
            finally:
                workspace.close()
            if (
                dist.get_world_size(group) == 4
                and (*geometry, conv_kernel_size) == (16, 64, 128, 128, 4)
                and dtype == torch.bfloat16
            ):
                raw = torch.zeros(
                    (max(64, local_tokens), projection_stride),
                    device=device,
                    dtype=dtype,
                )[:, :channels]
                weight = torch.zeros(
                    (channels, conv_kernel_size), device=device, dtype=dtype
                )
                workspace = communicator.allocate_peer_workspace(history.shape, dtype)
                try:
                    autotune_causal_conv1d_peer(
                        raw,
                        history,
                        weight,
                        workspace,
                        head_geometry=(
                            key_heads,
                            key_heads,
                            value_heads,
                            key_head_dim,
                            key_head_dim,
                            value_head_dim,
                        ),
                    )
                finally:
                    workspace.close()
        torch.cuda.synchronize(device)
