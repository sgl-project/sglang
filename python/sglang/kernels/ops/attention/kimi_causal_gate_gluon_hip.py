"""Opt-in Kimi-K3 fresh causal prefill with fused sigmoid output gating."""

import functools
import os
from types import MethodType, SimpleNamespace

import torch

from sglang.kernels.ops.attention.kda_whole_layer_gluon_hip import (
    _has_required_gluon_api,
    _rocm_arch,
)
from sglang.srt.utils import is_hip


def enabled() -> bool:
    return (
        os.environ.get("SGLANG_ROCM_K3_CAUSAL_GATE_FUSED_BACKEND", "").lower()
        == "gluon"
    )


def supported_shape(rows: int, requests: int) -> bool:
    """Use only current-upstream wins from the 24-point MI355X matrix."""
    return (rows, requests) in {
        (1024, 1),
        (1024, 2),
        (1024, 4),
        (1024, 8),
        (1024, 16),
        (1024, 32),
        (2048, 8),
        (2048, 16),
        (2048, 32),
        (4096, 4),
        (4096, 8),
        (4096, 16),
        (4096, 32),
        (8192, 16),
        (8192, 32),
    }


def can_prepare(attn, parallel, server_args) -> bool:
    weight = getattr(attn, "w_kc", None)
    return (
        enabled()
        and is_hip()
        and _has_required_gluon_api()
        and isinstance(weight, torch.Tensor)
        and _rocm_arch(weight.device) == "gfx950"
        and type(attn).__name__ == "KimiK3MLAAttention"
        and type(attn).__module__ == "sglang.srt.models.kimi_k3"
        and parallel.attn_tp_size == 8
        and not parallel.dcp_enabled
        and not parallel.dcp_replicate_q_proj
        and not getattr(server_args, "enable_lora", False)
        and not getattr(server_args, "speculative_algorithm", None)
        and attn.use_output_gate
        and not attn.use_dsa
        and not attn.use_deep_gemm_bmm
    )


def host_unit(x):
    return x is None or (type(x) in (int, float) and x == 1)


def layout(lengths, prefix, m):
    if not isinstance(lengths, (list, tuple)) or not isinstance(prefix, (list, tuple)):
        return None
    if not all((type(x) is int and x > 0 for x in lengths)):
        return None
    if len(lengths) != len(prefix) or not all(
        (type(x) is int and x == 0 for x in prefix)
    ):
        return None
    value = tuple(lengths)
    return value if value and sum(value) == m else None


def native_runtime():
    from sglang.srt.model_executor.forward_context import get_attn_backend
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        get_tc_piecewise_forward_context,
    )
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        is_in_breakable_cuda_graph,
    )
    from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mha import (
        resolve_attn_backend,
    )
    from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
    from sglang.srt.layers.cp.utils import is_mla_cp_active
    from sglang.srt.runtime_context import get_exec, get_parallel, get_server_args

    return SimpleNamespace(
        is_tensor=torch.is_tensor,
        piecewise_context=get_tc_piecewise_forward_context,
        breakable=is_in_breakable_cuda_graph,
        capturing=torch.cuda.is_current_stream_capturing,
        compiling=torch.compiler.is_compiling,
        backend=get_attn_backend,
        resolve_backend=resolve_attn_backend,
        parallel=get_parallel,
        server_args=get_server_args,
        execution=get_exec,
        mla_cp=is_mla_cp_active,
        dsa_cp=dsa_use_prefill_cp,
        wait_stream=lambda stream: torch.cuda.current_stream().wait_stream(stream),
    )


def eligible(attn, q, k, v, fb, rt):
    if (
        type(attn).__name__ != "KimiK3MLAAttention"
        or type(attn).__module__ != "sglang.srt.models.kimi_k3"
    ):
        return None
    if (
        rt.compiling()
        or rt.capturing()
        or rt.breakable()
        or (rt.piecewise_context() is not None)
    ):
        return None
    if not fb.forward_mode.is_extend_without_speculative():
        return None
    if not all((rt.is_tensor(t) for t in (q, k, v))) or q.ndim != 3:
        return None
    m = q.shape[0]
    ls = layout(fb.extend_seq_lens_cpu, fb.extend_prefix_lens_cpu, m)
    if ls is None or fb.batch_size != len(ls) or not supported_shape(m, len(ls)):
        return None
    if any(
        (
            t.device.type != "cuda" or t.device != q.device or t.dtype != torch.bfloat16
            for t in (q, k, v)
        )
    ):
        return None
    if (
        tuple(q.shape) != (m, 12, 192)
        or tuple(k.shape) != (m, 12, 192)
        or tuple(v.shape) != (m, 12, 128)
    ):
        return None
    if (
        q.stride() != (2304, 192, 1)
        or k.stride() != (2304, 192, 1)
        or v.stride() != (3072, 256, 1)
        or (v.storage_offset() != 128)
    ):
        return None
    if (
        attn.current_attention_backend != "triton"
        or not attn.use_output_gate
        or attn.rotary_emb is not None
        or attn.use_dsa
        or attn.use_deep_gemm_bmm
    ):
        return None
    par = rt.parallel()
    server = rt.server_args()
    execution = rt.execution()
    if (
        par.dcp_enabled
        or par.dcp_replicate_q_proj
        or par.attn_tp_size != 8
        or rt.mla_cp(fb)
        or rt.dsa_cp(fb)
    ):
        return None
    if (
        getattr(server, "enable_lora", False)
        or getattr(server, "speculative_algorithm", None)
        or execution.deterministic.enable_deterministic_inference
    ):
        return None
    if (
        getattr(fb, "_attn_output", None) is not None
        or getattr(fb, "mha_return_lse", False)
        or getattr(fb, "mha_one_shot", False)
    ):
        return None
    if not host_unit(attn.w_scale):
        return None
    backend = rt.backend()
    full = getattr(backend, "full_attn_backend", backend)
    if (
        type(full).__name__ != "TritonAttnBackend"
        or type(full).__module__ != "sglang.srt.layers.attention.triton_backend"
    ):
        return None
    if (
        hasattr(backend, "full_attn_backend")
        and attn.layer_id not in backend.full_attn_layers
    ):
        return None
    resolved = rt.resolve_backend(fb)
    if resolved is not backend or hasattr(resolved, "prepare_prefill_qkv"):
        return None
    if (
        full.dcp_size != 1
        or full.enable_deterministic
        or full.page_size != 1
        or (full._translate_kv_loc is not None)
    ):
        return None
    meta = full.forward_metadata
    radix = attn.attn_mha
    if meta.custom_mask is not None or meta.mask_indptr is not None:
        return None
    if (
        meta.out_cache_loc_full_physical is not None
        or meta.swa_out_cache_loc is not None
    ):
        return None
    if type(meta.max_extend_len) is not int or meta.max_extend_len != max(ls):
        return None
    if meta.kv_indices.numel() != 0:
        return None
    if radix.is_cross_attention or getattr(radix.attn_type, "value", None) != "decoder":
        return None
    if (
        radix.sliding_window_size not in (None, -1)
        or radix.logit_cap != 0
        or radix.xai_temperature_len != -1
    ):
        return None
    if not all(
        (
            host_unit(x)
            for x in (
                radix.k_scale,
                radix.v_scale,
                radix.k_scale_float,
                radix.v_scale_float,
            )
        )
    ):
        return None
    if (
        radix.tp_q_head_num,
        radix.tp_k_head_num,
        radix.tp_v_head_num,
        radix.qk_head_dim,
        radix.v_head_dim,
    ) != (12, 12, 12, 192, 128):
        return None
    if type(radix.scaling) not in (int, float) or radix.scaling != 192 ** (-0.5):
        return None
    ip = meta.qo_indptr
    if (
        not rt.is_tensor(ip)
        or ip.dtype != torch.int64
        or ip.device != q.device
        or (tuple(ip.shape) != (len(ls) + 1,))
        or (not ip.is_contiguous())
    ):
        return None
    hidden = attn._gate_hidden_states
    if (
        not rt.is_tensor(hidden)
        or hidden.dtype != torch.bfloat16
        or hidden.device != q.device
        or (tuple(hidden.shape) != (m, 7168))
    ):
        return None
    if attn._gate_precomputed is not None:
        return None
    return (ls, ip)


def run(q, k, v, gate, indptr, *, scale, max_query_len):
    from sglang.kernels.ops.attention.mla_gluon.causal_attention_gate import (
        kimi_causal_attention_gate,
    )

    return kimi_causal_attention_gate(
        q,
        k,
        v,
        gate,
        indptr,
        scale=scale,
        max_query_len=max_query_len,
    )


def bind(attn, *, _runtime=None):
    if getattr(attn, "_causal_gate_gluon_bound", False):
        return True
    rt = _runtime or native_runtime()
    original = attn.forward_normal_core

    @functools.wraps(original)
    def core(self, q, k, v, forward_batch):
        chosen = eligible(self, q, k, v, forward_batch, rt)
        if chosen is None:
            return original(q, k, v, forward_batch)
        lengths, indptr = chosen
        k = k.contiguous()
        v = v.contiguous()
        gate = self.g_proj(self._gate_hidden_states)[0]
        if not (
            rt.is_tensor(gate)
            and gate.dtype == torch.bfloat16
            and gate.device == q.device
            and tuple(gate.shape) == (q.shape[0], 1536)
            and gate.is_contiguous()
        ):
            raise RuntimeError("Unsupported causal MLA output gate layout")
        output = run(
            q,
            k,
            v,
            gate,
            indptr,
            scale=self.attn_mha.scaling,
            max_query_len=max(lengths),
        )
        if not (
            rt.is_tensor(output)
            and output.dtype == torch.bfloat16
            and output.device == q.device
            and tuple(output.shape) == (q.shape[0], 1536)
            and output.is_contiguous()
        ):
            raise RuntimeError("Unsupported causal MLA output layout")
        self._gate_hidden_states = None
        self._gate_precomputed = None
        return self.o_proj(output)[0]

    attn.forward_normal_core = MethodType(core, attn)
    attn._causal_gate_gluon_bound = True
    return True
