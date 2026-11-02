# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""C shims for CPU custom ops under TorchInductor cpp_wrapper."""

import ctypes
import logging
import os
from typing import Dict, List

import torch

logger = logging.getLogger(__name__)

_CPU_C_SHIMS_LOADED = False


def _get_common_ops_path() -> str:
    try:
        import sgl_kernel

        pkg_dir = os.path.dirname(sgl_kernel.__file__)
        for f in os.listdir(pkg_dir):
            if "common_ops" in f and f.endswith(".so"):
                return os.path.join(pkg_dir, f)
    except Exception as e:
        logger.warning(f"Failed to find sgl_kernel common_ops.so: {e}")
    return ""


def ensure_cpu_c_shims_loaded() -> None:
    """Ensure CPU C shims are loaded into the global symbol table (RTLD_GLOBAL)."""
    global _CPU_C_SHIMS_LOADED
    if _CPU_C_SHIMS_LOADED:
        return

    common_ops_so = _get_common_ops_path()
    if common_ops_so:
        try:
            # Ensure common_ops symbols are visible globally for runtime lookup
            ctypes.CDLL(common_ops_so, mode=ctypes.RTLD_GLOBAL)
        except Exception as e:
            logger.warning(f"Failed to load {common_ops_so} via RTLD_GLOBAL: {e}")

    # Check if aoti_torch_cpu_multimodal_rotary_embedding_cpu is already exported by common_ops
    handle = ctypes.CDLL(None)
    if hasattr(handle, "aoti_torch_cpu_multimodal_rotary_embedding_cpu"):
        _CPU_C_SHIMS_LOADED = True
        return

    # If not yet exported (e.g. running against older prebuilt wheel), compile inline from source
    shim_cpp_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../../kernels/aot/csrc/cpu/aoti_c_shims.cpp",
        )
    )
    if not os.path.exists(shim_cpp_path):
        logger.warning(
            f"CPU C shims source not found at {shim_cpp_path}. "
            "TorchInductor cpp_wrapper custom op calls may fall back."
        )
        return

    try:
        with open(shim_cpp_path, "r", encoding="utf-8") as f:
            shim_source = f.read()

        from torch.utils.cpp_extension import load_inline

        extra_ldflags = [common_ops_so] if common_ops_so else []
        shim_lib_path = load_inline(
            name="sgl_cpu_aoti_shims",
            cpp_sources=shim_source,
            extra_ldflags=extra_ldflags,
            is_python_module=False,
            verbose=False,
        )
        ctypes.CDLL(shim_lib_path, mode=ctypes.RTLD_GLOBAL)
        _CPU_C_SHIMS_LOADED = True
        logger.info("Successfully loaded SGLang CPU AOTI C shims.")
    except Exception as e:
        logger.warning(
            f"Failed to compile/load SGLang CPU AOTI C shims inline: {e}. "
            "TorchInductor cpp_wrapper custom op calls may fall back."
        )


def get_cpu_custom_ops_to_c_shims() -> Dict[object, List[str]]:
    """Return Inductor custom_ops_to_c_shims dictionary for CPU ops."""
    shims = {}
    sgl_ops = getattr(torch.ops, "sgl_kernel", None)
    if sgl_ops is None:
        return shims

    def register(op_name: str, c_signature: str):
        op_obj = getattr(sgl_ops, op_name, None)
        if op_obj is not None and hasattr(op_obj, "default"):
            shims[op_obj.default] = [c_signature]

    # 1. multimodal_rotary_embedding_cpu (List[int]? mrope_section)
    register(
        "multimodal_rotary_embedding_cpu",
        "int aoti_torch_cpu_multimodal_rotary_embedding_cpu("
        "AtenTensorHandle positions, AtenTensorHandle query, AtenTensorHandle key, "
        "int64_t head_size, AtenTensorHandle cos_sin, "
        "const int64_t* const* mrope_section, int64_t mrope_section_size, "
        "bool mrope_interleaved, bool is_neox, void* out)",
    )

    # 2. fp8_scaled_mm_cpu (List[int] block_size)
    register(
        "fp8_scaled_mm_cpu",
        "int aoti_torch_cpu_fp8_scaled_mm_cpu("
        "AtenTensorHandle mat1, AtenTensorHandle mat2, AtenTensorHandle scales2, "
        "const int64_t* block_size, int64_t block_size_len, "
        "AtenTensorHandle bias, int32_t out_dtype, bool is_vnni, "
        "AtenTensorHandle* out)",
    )

    # 3. fused_experts_cpu (List[int]? block_size, str? activation)
    register(
        "fused_experts_cpu",
        "int aoti_torch_cpu_fused_experts_cpu("
        "AtenTensorHandle hidden_states, AtenTensorHandle w1, AtenTensorHandle w2, "
        "AtenTensorHandle topk_weights, AtenTensorHandle topk_ids, "
        "bool inplace, int64_t moe_comp_method, "
        "AtenTensorHandle w1_scale, AtenTensorHandle w2_scale, "
        "AtenTensorHandle w1_zero, AtenTensorHandle w2_zero, "
        "const int64_t* const* block_size, int64_t block_size_len, "
        "AtenTensorHandle w1_bias, AtenTensorHandle w2_bias, "
        "const double* alpha, const double* limit, bool is_vnni, "
        "const char* activation, AtenTensorHandle* out)",
    )

    # 4. shared_expert_cpu (List[int]? block_size)
    register(
        "shared_expert_cpu",
        "int aoti_torch_cpu_shared_expert_cpu("
        "AtenTensorHandle hidden_states, AtenTensorHandle w1, AtenTensorHandle w2, "
        "AtenTensorHandle fused_experts_out, const double* routed_scaling_factor, "
        "bool inplace, bool use_int8_w8a8, bool use_fp8_w8a16, "
        "AtenTensorHandle w1_scale, AtenTensorHandle w2_scale, "
        "const int64_t* const* block_size, int64_t block_size_len, "
        "bool is_vnni, AtenTensorHandle* out)",
    )

    # 5. qkv_proj_with_rope (List[int]? block_size)
    register(
        "qkv_proj_with_rope",
        "int aoti_torch_cpu_qkv_proj_with_rope("
        "AtenTensorHandle hidden_states, AtenTensorHandle q_a_proj_weight, "
        "AtenTensorHandle q_b_proj_weight, AtenTensorHandle kv_a_proj_weight, "
        "AtenTensorHandle w_kc, AtenTensorHandle q_a_layernorm_weight, "
        "AtenTensorHandle kv_a_layernorm_weight, AtenTensorHandle positions, "
        "AtenTensorHandle cos_sin_cache, double eps, bool use_int8_w8a8, bool use_fp8_w8a16, "
        "AtenTensorHandle q_a_proj_scale, AtenTensorHandle q_b_proj_scale, "
        "AtenTensorHandle kv_a_proj_scale, AtenTensorHandle w_scale, "
        "bool is_vnni, const int64_t* const* block_size, int64_t block_size_len, "
        "AtenTensorHandle* out0, AtenTensorHandle* out1, AtenTensorHandle* out2)",
    )

    # 6. qkv_proj_with_rope_fused_weight (List[int]? block_size)
    register(
        "qkv_proj_with_rope_fused_weight",
        "int aoti_torch_cpu_qkv_proj_with_rope_fused_weight("
        "AtenTensorHandle hidden_states, AtenTensorHandle qkv_a_proj_weight, "
        "AtenTensorHandle q_b_proj_weight, AtenTensorHandle w_kc, "
        "AtenTensorHandle q_a_layernorm_weight, AtenTensorHandle kv_a_layernorm_weight, "
        "AtenTensorHandle positions, AtenTensorHandle cos_sin_cache, "
        "double eps, bool use_int8_w8a8, bool use_fp8_w8a16, "
        "AtenTensorHandle qkv_a_proj_scale, AtenTensorHandle q_b_proj_scale, "
        "AtenTensorHandle w_scale, bool is_vnni, "
        "const int64_t* const* block_size, int64_t block_size_len, "
        "int64_t q_lora_rank, int64_t kv_lora_rank, int64_t qk_rope_head_dim, "
        "AtenTensorHandle* out0, AtenTensorHandle* out1, AtenTensorHandle* out2)",
    )

    # Core high-frequency compute ops
    register(
        "rmsnorm_cpu",
        "int aoti_torch_cpu_rmsnorm_cpu("
        "AtenTensorHandle input, AtenTensorHandle weight, double eps, "
        "AtenTensorHandle* out)",
    )
    register(
        "fused_add_rmsnorm_cpu",
        "int aoti_torch_cpu_fused_add_rmsnorm_cpu("
        "AtenTensorHandle input, AtenTensorHandle residual, AtenTensorHandle weight, "
        "double eps, void* unused_out)",
    )
    register(
        "silu_and_mul_cpu",
        "int aoti_torch_cpu_silu_and_mul_cpu("
        "AtenTensorHandle input, AtenTensorHandle* out)",
    )

    return shims


def enable_cpu_cpp_wrapper() -> None:
    """Configure TorchInductor for CPU C++ wrapper with direct C shims."""
    ensure_cpu_c_shims_loaded()
    shims = get_cpu_custom_ops_to_c_shims()
    if shims:
        torch._inductor.config.aot_inductor.custom_ops_to_c_shims.update(shims)
    torch._inductor.config.cpp_wrapper = True
    logger.info("TorchInductor CPU cpp_wrapper enabled with custom op C shims.")
