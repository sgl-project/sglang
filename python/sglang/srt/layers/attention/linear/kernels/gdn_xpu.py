import torch
from sgl_kernel import gated_delta_rule, gated_delta_rule_verify
from sgl_kernel.mamba import causal_conv1d_update_verify_xpu

from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel


def causal_conv1d_verify_xpu(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    *,
    conv_state_indices: torch.Tensor,
    intermediate_conv_window: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    retrieve_next_token: torch.Tensor | None,
    retrieve_next_sibling: torch.Tensor | None,
    retrieve_parent_token: torch.Tensor | None,
) -> torch.Tensor:
    return causal_conv1d_update_verify_xpu(
        x,
        conv_state,
        weight,
        intermediate_conv_window,
        intermediate_state_indices,
        retrieve_next_token,
        retrieve_next_sibling,
        retrieve_parent_token,
        bias=bias,
        activation=activation,
        cache_indices=conv_state_indices,
    )


class XpuGDNKernel(TritonGDNKernel):
    """Standalone SYCL GDN recurrence, with Triton for unsupported prefill modes."""

    supports_packed_decode = False
    supports_strided_target_verify_qkv = True

    @staticmethod
    def _tokens(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, x.shape[-2], x.shape[-1]).contiguous()

    def decode(
        self,
        q,
        k,
        v,
        a,
        b,
        *,
        A_log,
        dt_bias,
        ssm_states,
        cache_indices,
        query_start_loc,
        **kwargs,
    ):
        out = v.new_empty(v.shape)
        gated_delta_rule(
            out.reshape(-1, *out.shape[-2:]),
            self._tokens(q),
            self._tokens(k),
            self._tokens(v),
            b,
            a,
            A_log,
            dt_bias,
            ssm_states,
            query_start_loc,
            cache_indices,
            num_prefills=0,
            num_decodes=cache_indices.numel(),
        )
        return out

    def extend(
        self,
        q,
        k,
        v,
        g,
        beta,
        *,
        ssm_states,
        cache_indices,
        query_start_loc,
        **kwargs,
    ):
        a, b = kwargs.get("a"), kwargs.get("b")
        if (
            a is None
            or b is None
            or not kwargs.get("inplace_update", True)
            or kwargs.get("track_state") is not None
        ):
            return super().extend(
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
        out = v.new_empty(v.shape)
        gated_delta_rule(
            out.reshape(-1, *out.shape[-2:]),
            self._tokens(q),
            self._tokens(k),
            self._tokens(v),
            b,
            a,
            kwargs["A_log"],
            kwargs["dt_bias"],
            ssm_states,
            query_start_loc,
            cache_indices,
            num_prefills=cache_indices.numel(),
            num_decodes=0,
            has_initial_state=kwargs["has_initial_state"],
        )
        return out, None, None

    def target_verify(
        self,
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        *,
        ssm_states,
        cache_indices,
        query_start_loc,
        intermediate_states_buffer,
        intermediate_state_indices,
        cache_steps,
        retrieve_parent_token,
        **kwargs,
    ):
        out = v.new_empty(v.shape)
        gated_delta_rule_verify(
            out.reshape(-1, *out.shape[-2:]),
            self._tokens(q),
            self._tokens(k),
            self._tokens(v),
            b,
            a,
            A_log,
            dt_bias,
            ssm_states,
            cache_indices[: query_start_loc.numel() - 1],
            intermediate_states_buffer,
            intermediate_state_indices[: query_start_loc.numel() - 1],
            cache_steps,
            retrieve_parent_token=retrieve_parent_token,
        )
        return out
