import logging
from typing import Optional, Union

import torch
from sgl_kernel_npu.mamba.mamba_state_update_triton import (
    conv_state_rollback,
    move_intermediate_cache,
)

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    MambaAttnBackendBase,
)
from sglang.srt.layers.attention.mamba.mamba2_metadata import (
    ForwardMetadata,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.spec_info import SpecInput

logger = logging.getLogger(__name__)

# One-shot logs, keyed by reason: these paths run per layer per forward.
_logged_once: set = set()


def _log_once(key: str, msg: str, *, warning: bool = False) -> None:
    if key not in _logged_once:
        _logged_once.add(key)
        (logger.warning if warning else logger.info)(msg)


class ReplaySSMDecodeMixin:
    """ReplaySSM decode ring (--enable-linear-replayssm) for an Ascend backend.

    The ring kernel is the gate-generic Triton implementation shared with CUDA
    (``sglang.kernels...fla.fused_recurrent_linear_replayssm``); it runs on
    Ascend through triton-ascend, so this only has to detect that the ring is
    live for the pool + layer and hand the *packed* post-conv qkv to that kernel
    instead of splitting q/k/v for the per-step decode. Cursor advance and
    force-flush stay in the shared metadata path.
    """

    def _replayssm_ring_args(self, layer, layer_cache, is_kda: bool):
        """Ring tensors + cursor for this layer, or None to keep the legacy path."""
        mamba_pool = getattr(self.req_to_token_pool, "mamba_pool", None)
        if mamba_pool is None or not getattr(
            mamba_pool, "enable_linear_replayssm", False
        ):
            return None
        if getattr(mamba_pool, "replayssm_spec_fold", False):
            return None
        write_pos = getattr(self.forward_metadata, "replayssm_write_pos", None)
        if write_pos is None:
            _log_once(
                "no_cursor",
                "--enable-linear-replayssm on NPU: no ring cursor in this forward "
                "metadata; falling back to the per-step decode.",
                warning=True,
            )
            return None
        if is_kda and getattr(layer, "lower_bound", None) is not None:
            # Bounded (safe) gate: the ring kernel implements the plain softplus
            # gate only, so this layer keeps the per-step decode (as on CUDA).
            _log_once(
                "kda_lower_bound",
                "--enable-linear-replayssm on NPU: this KDA layer uses a bounded "
                "gate, which the ring kernel does not implement; decoding per-step.",
                warning=True,
            )
            return None
        d_cache = layer_cache.replayssm_d
        k_cache = layer_cache.replayssm_k
        g_cache = layer_cache.replayssm_g
        if d_cache is None or k_cache is None or g_cache is None:
            return None
        return (
            d_cache,
            k_cache,
            g_cache,
            write_pos,
            getattr(self.forward_metadata, "replayssm_force_flush", None),
        )

    def replayssm_decode_npu(
        self, *, layer, layer_cache, packed_qkv, a, b, ssm_states, cache_indices, is_kda
    ):
        """Ring-buffered decode of one batch, or None when the ring is not usable.

        ``packed_qkv`` is the post-conv packed q|k|v the legacy path splits.
        """
        args = self._replayssm_ring_args(layer, layer_cache, is_kda)
        if args is None:
            return None
        d_cache, k_cache, g_cache, write_pos, force_flush = args
        # Geometry the kernel infers from the state pool [slots, HV, V, K].
        hv, v_dim, k_dim = ssm_states.shape[1:]
        if is_kda:
            # Per-K gate: a arrives [B, HV*K], dt_bias [HV*K], A_log [1, 1, HV, 1].
            B = packed_qkv.shape[0]
            a = a.reshape(B, hv, k_dim).contiguous()
            dt_bias = layer.dt_bias.reshape(hv, k_dim).contiguous()
            A_log = layer.A_log.reshape(-1)
            b = b.reshape(B, hv).contiguous()
        else:
            dt_bias, A_log = layer.dt_bias, layer.A_log
        _log_once(
            f"engaged_kda={is_kda}",
            "--enable-linear-replayssm: ReplaySSM ring decode engaged on NPU "
            f"(triton-ascend ring kernel, is_kda={is_kda}, L={d_cache.shape[2]}).",
        )
        from sglang.kernels.ops.attention.fla.fused_recurrent_linear_replayssm import (
            fused_recurrent_linear_replayssm_decode,
        )

        out = packed_qkv.new_empty(packed_qkv.shape[0], 1, hv, v_dim)
        fused_recurrent_linear_replayssm_decode(
            mixed_qkv=packed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=k_dim**-0.5,
            initial_state=ssm_states,
            d_cache=d_cache,
            k_cache=k_cache,
            g_cache=g_cache,
            out=out,
            ssm_state_indices=cache_indices,
            write_pos=write_pos,
            force_flush=force_flush,
            use_qk_l2norm_in_kernel=True,
            is_kda=is_kda,
        )
        # [B, 1, HV, V] -> the [1, B, ...] layout the packed decode returns.
        return out.transpose(0, 1)


class AscendMambaAttnBackendBase(MambaAttnBackendBase):
    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self.state_indices_list_gdn = []

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        assert max_num_tokens % max_bs == 0, (
            f"max_num_tokens={max_num_tokens} must be divisible by max_bs={max_bs}"
        )
        draft_token_num = max_num_tokens // max_bs
        # Static per-bs ReplaySSM ring buffers (None unless the flag is on), refreshed
        # in place by _replay_metadata before each replay.
        self.replayssm_write_pos_list = [] if self._replayssm_enabled() else None
        self.replayssm_force_flush_list = [] if self._replayssm_enabled() else None
        for i in range(max_bs):
            self.state_indices_list.append(
                torch.full(
                    (i + 1,), self.pad_slot_id, dtype=torch.int32, device=self.device
                )
            )
            self.state_indices_list_gdn.append(
                torch.full(
                    ((i + 1) * draft_token_num,),
                    self.pad_slot_id,
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            self.query_start_loc_list.append(
                torch.zeros((i + 2,), dtype=torch.int32, device=self.device)
            )
            self.retrieve_next_token_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
            self.retrieve_next_sibling_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
            self.retrieve_parent_token_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
            if self.replayssm_write_pos_list is not None:
                self.replayssm_write_pos_list.append(
                    torch.zeros((i + 1,), dtype=torch.int32, device=self.device)
                )
            if self.replayssm_force_flush_list is not None:
                self.replayssm_force_flush_list.append(
                    torch.zeros((i + 1,), dtype=torch.int32, device=self.device)
                )
        self.cached_cuda_graph_decode_query_start_loc = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=self.device
        )
        self.cached_cuda_graph_verify_query_start_loc = torch.arange(
            0,
            max_bs * draft_token_num + 1,
            step=draft_token_num,
            dtype=torch.int32,
            device=self.device,
        )

    def _capture_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)
        if forward_mode.is_decode_or_idle():
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
            )
        elif forward_mode.is_target_verify():
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
            )
            ssm_state_indices = torch.arange(
                mamba_indices.shape[0] * spec_info.draft_token_num,
                dtype=torch.int32,
                device=mamba_indices.device,
            )
            self.state_indices_list_gdn[bs - 1][
                : len(mamba_indices) * spec_info.draft_token_num
            ].copy_(ssm_state_indices)
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")

        # ReplaySSM: capture the static ring-buffer pointers (None unless the flag is on).
        replayssm_write_pos = (
            self.replayssm_write_pos_list[bs - 1]
            if self.replayssm_write_pos_list is not None
            else None
        )
        replayssm_force_flush = (
            self.replayssm_force_flush_list[bs - 1]
            if self.replayssm_force_flush_list is not None
            else None
        )

        # If topk > 1, we need to use retrieve_next_token and retrieve_next_sibling to handle the eagle tree custom attention mask
        if forward_mode.is_target_verify() and spec_info.topk > 1:
            # They are None during cuda graph capture so skip the copy_...
            # self.retrieve_next_token_list[bs - 1].copy_(spec_info.retrive_next_token)
            # self.retrieve_next_sibling_list[bs - 1].copy_(spec_info.retrive_next_sibling)
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                retrieve_next_token=self.retrieve_next_token_list[bs - 1],
                retrieve_next_sibling=self.retrieve_next_sibling_list[bs - 1],
                retrieve_parent_token=self.retrieve_parent_token_list[bs - 1],
                replayssm_write_pos=replayssm_write_pos,
                replayssm_force_flush=replayssm_force_flush,
            )
        else:
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                mamba_cache_indices_gdn=self.state_indices_list_gdn[bs - 1],
                replayssm_write_pos=replayssm_write_pos,
                replayssm_force_flush=replayssm_force_flush,
            )

    def _replay_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        num_padding: Optional[int] = None,
        in_capture: bool = False,
        mamba_track_indices: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        # out_graph passes seq_lens_cpu=None at capture; mirror the base guard.
        if seq_lens_cpu is None:
            num_padding = 0
        else:
            num_padding = torch.count_nonzero(
                seq_lens_cpu == self.get_cuda_graph_seq_len_fill_value()
            )
        # Make sure forward metadata is correctly handled for padding reqs
        req_pool_indices[bs - num_padding :] = 0
        mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        # Pad rows must carry the pad sentinel (kernel skips state_idx < 0),
        # NOT 0: with 0 they would read/write physical slot 0's conv/ring
        # state under graph replay (same sentinel as the CUDA base).
        mamba_indices[bs - num_padding :] = self.pad_slot_id
        self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)
        track_buf = None
        if mamba_track_indices is not None:
            track_buf = mamba_track_indices
        if forward_mode.is_decode_or_idle():
            if num_padding == 0:
                self.query_start_loc_list[bs - 1].copy_(
                    self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
                )
            else:
                self.query_start_loc_list[bs - 1][: bs - num_padding].copy_(
                    self.cached_cuda_graph_decode_query_start_loc[: bs - num_padding]
                )
                self.query_start_loc_list[bs - 1][bs - num_padding :].fill_(
                    bs - num_padding
                )
        elif forward_mode.is_target_verify():
            ssm_state_indices = torch.arange(
                bs * spec_info.draft_token_num,
                dtype=torch.int32,
                device=mamba_indices.device,
            )
            self.state_indices_list_gdn[bs - 1].copy_(ssm_state_indices)
            if num_padding == 0:
                self.query_start_loc_list[bs - 1].copy_(
                    self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
                )
            else:
                self.query_start_loc_list[bs - 1][: bs - num_padding].copy_(
                    self.cached_cuda_graph_verify_query_start_loc[: bs - num_padding]
                )
                self.query_start_loc_list[bs - 1][bs - num_padding :].fill_(
                    (bs - num_padding) * spec_info.draft_token_num
                )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")

        # ReplaySSM: snapshot + advance the ring cursor for this replay.
        replayssm_write_pos, replayssm_force_flush = self._replayssm_graph_buffers(
            bs, mamba_indices, forward_mode, seq_lens_cpu, in_capture
        )

        # If topk > 1, we need to use retrieve_next_token and retrieve_next_sibling to handle the eagle tree custom attention mask
        if forward_mode.is_target_verify() and spec_info.topk > 1:
            bs_without_pad = spec_info.retrive_next_token.shape[0]
            self.retrieve_next_token_list[bs - 1][:bs_without_pad].copy_(
                spec_info.retrive_next_token
            )
            self.retrieve_next_sibling_list[bs - 1][:bs_without_pad].copy_(
                spec_info.retrive_next_sibling
            )
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                mamba_track_indices=track_buf,
                retrieve_next_token=self.retrieve_next_token_list[bs - 1],
                retrieve_next_sibling=self.retrieve_next_sibling_list[bs - 1],
                retrieve_parent_token=self.retrieve_parent_token_list[bs - 1],
                replayssm_write_pos=replayssm_write_pos,
                replayssm_force_flush=replayssm_force_flush,
            )
        else:
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                mamba_cache_indices_gdn=self.state_indices_list_gdn[bs - 1],
                mamba_track_indices=track_buf,
                replayssm_write_pos=replayssm_write_pos,
                replayssm_force_flush=replayssm_force_flush,
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return 0  # Mamba attn does not use seq lens to index kv cache


class AscendMamba2AttnBackend(AscendMambaAttnBackendBase):
    pass


class AscendHybridLinearAttnBackend(HybridLinearAttnBackend):
    def __init__(
        self,
        full_attn_backend: AttentionBackend,
        linear_attn_backend: AscendMambaAttnBackendBase,
        full_attn_layers: list[int],
    ):
        super().__init__(full_attn_backend, linear_attn_backend, full_attn_layers)

    def update_mamba_state_after_mtp_verify(
        self,
        last_correct_step_indices: torch.Tensor,
        mamba_track_indices: Optional[torch.Tensor],
        mamba_steps_to_track: Optional[torch.Tensor],
        model,
        req_pool_indices: Optional[torch.Tensor] = None,
    ):
        """
        Update mamba states after MTP verify using fully fused Triton kernel.

        This replaces the original advanced indexing operations with a single fused
        gather-scatter kernel that also handles masking internally, avoiding:
        - index_elementwise_kernel from tensor[bool_mask]
        - index_select kernel launches
        - nonzero kernel launches
        """
        del req_pool_indices  # accepted for hook parity; slots come from metadata
        request_number = last_correct_step_indices.shape[0]

        state_indices_tensor = (
            self.linear_attn_backend.forward_metadata.mamba_cache_indices[
                :request_number
            ]
        )

        mamba_caches = self.linear_attn_backend.req_to_token_pool.get_speculative_mamba2_params_all_layers()

        conv_states = mamba_caches.conv[0]
        ssm_states = mamba_caches.temporal
        intermediate_state_cache = mamba_caches.intermediate_ssm
        dst_indices_tensor = state_indices_tensor.to(torch.int64)  # [N]
        src_indices_tensor = torch.arange(
            dst_indices_tensor.shape[0],
            device=dst_indices_tensor.device,
            dtype=torch.int64,
        )
        last_steps = last_correct_step_indices.to(torch.int64)  # [N]

        move_intermediate_cache(
            ssm_states,
            intermediate_state_cache,
            dst_indices_tensor,
            src_indices_tensor,
            last_steps,
        )

        draft_token_num = intermediate_state_cache.shape[2]
        if mamba_track_indices is not None:
            assert mamba_steps_to_track is not None
            mamba_track_indices = mamba_track_indices.to(torch.int64)
            mamba_steps_to_track = mamba_steps_to_track.to(torch.int64)

            move_intermediate_cache(
                ssm_states,
                intermediate_state_cache,
                mamba_track_indices,
                src_indices_tensor,
                mamba_steps_to_track,
            )

            track_mask = mamba_steps_to_track >= 0
            # Track conv state from the verify-time window before rolling back
            # the working slot; NPU does not keep per-step conv intermediates.
            src_slots = torch.where(track_mask, dst_indices_tensor, mamba_track_indices)
            conv_states[:, mamba_track_indices] = conv_states[:, src_slots]

        if dst_indices_tensor.numel() > 0:
            conv_state_rollback(
                conv_states,
                dst_indices_tensor,
                last_steps,
                draft_token_num,
            )

        if mamba_track_indices is not None and mamba_track_indices.numel() > 0:
            conv_state_rollback(
                conv_states,
                mamba_track_indices,
                mamba_steps_to_track,
                draft_token_num,
            )

        return

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        pass
