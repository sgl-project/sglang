import torch
from sgl_kernel import gdn_attention as sgl_kernel_gdn_attention

from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class XpuGDNAttnBackend(GDNAttnBackend):
    """XPU specialization of ``GDNAttnBackend``.

    Adds an optional fused path that dispatches the whole conv1d + gating +
    delta-rule pipeline to the vendored vLLM SYCL kernel exposed as
    ``torch.ops.sgl_kernel.gdn_attention``. This is opt-in via
    ``--linear-attn-backend intel_xpu`` (the default remains ``triton``, same
    as other platforms).
    """

    def supports_fused_gdn(self, layer, forward_batch: ForwardBatch) -> bool:
        """Conservative guard: only the plain decode / non-prefix-cached,
        non-speculative extend cases are handled by the fused kernel."""
        mode = forward_batch.forward_mode
        backends = self.linear_attn_backends
        selected = (
            backends.verify
            if mode.is_target_verify()
            else (backends.decode if mode.is_decode_or_idle() else backends.prefill)
        )
        if not selected.is_intel_xpu():
            return False
        if not hasattr(torch.ops.sgl_kernel, "gdn_attention"):
            # User explicitly asked for intel_xpu but the op isn't built.
            raise RuntimeError(
                "--linear-attn-backend intel_xpu requires the "
                "torch.ops.sgl_kernel.gdn_attention op, but it is not "
                "available. Rebuild sgl-kernel-xpu or use "
                "--linear-attn-backend triton."
            )
        if mode.is_target_verify() or mode.is_draft_extend_v2():
            return False
        fm = self.forward_metadata
        if getattr(fm, "has_mamba_track_mask", False):
            # chunked prefix-cache intermediate-state tracking unsupported
            return False
        if getattr(fm, "query_start_loc", None) is None:
            return False
        # GDN (not KDA) shared weights must be plain tensors
        if not isinstance(layer.conv_weights, torch.Tensor):
            return False
        if layer.bias is not None and not isinstance(layer.bias, torch.Tensor):
            return False
        return True

    def forward_fused_gdn(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        projected_states_qkvz: torch.Tensor,
        projected_states_ba: torch.Tensor,
    ):
        """Run the fused SYCL GDN op and return ``(core_attn_out, z)``.

        Caches stay in the SGLang pool layout and are updated in place. The conv
        pool is ``[cache, dim, width-1]``; we pass a transposed view so the op sees
        its logical ``[cache, width-1, dim]`` layout while the kernels index via
        explicit width/dim strides (no gather/transpose/scatter copies). The ssm
        pool already matches the op layout. ``mamba_cache_indices`` indexes the
        full pool directly for both conv and ssm.
        """
        fm = self.forward_metadata
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]  # [cache, dim, width-1]
        ssm_states = layer_cache.temporal  # [cache, nv, hv, hk]
        cache_indices = fm.mamba_cache_indices
        query_start_loc = fm.query_start_loc

        device = projected_states_qkvz.device
        dtype = projected_states_qkvz.dtype
        bs = forward_batch.batch_size
        num_actual_tokens = projected_states_qkvz.shape[0]

        if forward_batch.forward_mode.is_decode_or_idle():
            num_decodes, num_prefills = bs, 0
            has_initial_state = torch.ones(bs, dtype=torch.bool, device=device)
        else:
            num_decodes, num_prefills = 0, bs
            has_initial_state = forward_batch.extend_prefix_lens > 0

        # Full-pool, zero-copy: transposed view for conv + native ssm pool, indexed
        # directly by the full-pool cache indices.
        conv_view = conv_states.transpose(1, 2)  # [cache, width-1, dim] view
        state_idx = cache_indices.to(torch.int32).contiguous()

        core_attn_out = torch.empty(
            num_actual_tokens,
            layer.num_v_heads,
            layer.head_v_dim,
            dtype=dtype,
            device=device,
        )
        z = torch.empty_like(core_attn_out)

        sgl_kernel_gdn_attention(
            core_attn_out=core_attn_out,
            z=z,
            projected_states_qkvz=projected_states_qkvz,
            projected_states_ba=projected_states_ba,
            num_k_heads=layer.num_k_heads,
            num_v_heads=layer.num_v_heads,
            head_k_dim=layer.head_k_dim,
            head_v_dim=layer.head_v_dim,
            conv_state=conv_view,
            ssm_state=ssm_states,
            conv_weights=layer.conv_weights,
            conv_bias=layer.bias,
            activation=layer.activation,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            num_spec_decodes=0,
            has_initial_state=has_initial_state,
            non_spec_query_start_loc=query_start_loc,
            non_spec_token_indx=None,
            non_spec_state_indices_tensor=state_idx,
            spec_query_start_loc=None,
            spec_token_indx=None,
            spec_state_indices_tensor=None,
            num_accepted_tokens=None,
            num_actual_tokens=num_actual_tokens,
            # Heads/tensors are already per-rank sharded; kernel needs no
            # further in-kernel sharding, so this is always 1, not --tp-size.
            tp_size=1,
            reorder_input=True,
        )

        # conv/ssm states were updated in place via the pool views.
        return core_attn_out, z
