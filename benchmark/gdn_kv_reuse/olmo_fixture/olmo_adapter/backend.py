"""Retain SGLang conv/state tracking; use Olmo's beta=2*sigmoid(b)."""

import torch.nn.functional as F

from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_update,
)
from sglang.srt.layers.attention.linear.gdn_backend import (
    GDNAttnBackend,
    GDNKernelDispatcher,
)
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel


def gates(A_log, a, b, dt_bias):
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    beta = b.sigmoid() * 2.0
    return g.unsqueeze(0), beta.unsqueeze(0)


class OlmoTritonGDNKernel(TritonGDNKernel):
    supports_packed_decode = False

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
        g, beta = gates(A_log, a, b, dt_bias)
        return fused_recurrent_gated_delta_rule_update(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
        )


class OlmoGDNAttnBackend(GDNAttnBackend):
    def _create_kernel_dispatcher(self, backends):
        assert backends.decode.is_triton() and backends.prefill.is_triton()
        dispatcher = GDNKernelDispatcher(
            backends.decode, backends.prefill, backends.verify
        )
        dispatcher.decode_kernel = OlmoTritonGDNKernel()
        dispatcher.supports_packed_decode = False
        return dispatcher

    @staticmethod
    def _prefill_gates(layer, a, b):
        return gates(layer.A_log, a, b, layer.dt_bias)
