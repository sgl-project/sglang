# SPDX-License-Identifier: Apache-2.0
"""XPU (Intel GPU) quantization kernels.

Dense int4 GPTQ/AWQ linear methods that lower to the native torch XPU
``_weight_int4pack_mm`` op (see ``kernel-plan/xpu_quant_implementation_plan.md``).
"""
