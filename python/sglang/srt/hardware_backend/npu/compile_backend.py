"""npugraph_ex compilation inside SGLang-owned NPU graph capture."""

import functools
import logging
import os
from importlib import import_module

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _register_npu_mxfp8_fusion_patterns() -> None:
    """Register Ascend 950 (arch35) MXFP8 FX fusion patterns into npugraph_ex.

    Fuses consecutive memory-bound vector ops into single Ascend 950 CANN kernels
    to eliminate BF16 HBM round-trips (critical on HiBL 1.0 HBM):
      1. npu_add_rms_norm + npu_dynamic_mx_quant -> npu_add_rms_norm_dynamic_mx_quant
      2. npu_rms_norm + npu_dynamic_mx_quant     -> npu_rms_norm_dynamic_mx_quant
      3. npu_swiglu + npu_dynamic_mx_quant       -> npu_swiglu_mx_quant
    """
    if not envs.SGLANG_NPU_ENABLE_MXFP8_FUSION.get():
        return
    from sglang.srt.hardware_backend.npu.utils import is_npu_arch35

    npu_ops = getattr(torch.ops, "npu", None)
    if not is_npu_arch35() or not all(
        hasattr(npu_ops, name)
        for name in (
            "npu_add_rms_norm_dynamic_mx_quant",
            "npu_rms_norm_dynamic_mx_quant",
            "npu_swiglu_mx_quant",
        )
    ):
        return
    try:
        from npugraph_ex.patterns.pattern_pass_manager import (
            DEFAULT_EPSILON,
            _pattern_manager,
        )
        from torch._subclasses.fake_tensor import FakeTensorMode
    except Exception as exc:
        logger.debug("Skipping npugraph_ex MXFP8 pattern registration: %s", exc)
        return

    pm = _pattern_manager()
    fp8_dtype = torch.float8_e4m3fn

    def add_rms_mx_search(x1, x2, gamma, epsilon):
        y, _, res_out = torch.ops.npu.npu_add_rms_norm.default(x1, x2, gamma, epsilon)
        qx, xs = torch.ops.npu.npu_dynamic_mx_quant.default(y, dst_type=fp8_dtype)
        return qx, xs, res_out

    def add_rms_mx_replace(x1, x2, gamma, epsilon):
        qx, res_out, xs, _ = torch.ops.npu.npu_add_rms_norm_dynamic_mx_quant.default(
            x1, x2, gamma, epsilon=epsilon, dst_type=fp8_dtype
        )
        return qx, xs, res_out

    def rms_mx_search(x, gamma, epsilon):
        y, _ = torch.ops.npu.npu_rms_norm.default(x, gamma, epsilon)
        qx, xs = torch.ops.npu.npu_dynamic_mx_quant.default(y, dst_type=fp8_dtype)
        return qx, xs

    def rms_mx_replace(x, gamma, epsilon):
        qx, xs, _ = torch.ops.npu.npu_rms_norm_dynamic_mx_quant.default(
            x, gamma, epsilon=epsilon, dst_type=fp8_dtype
        )
        return qx, xs

    def swiglu_mx_search(x):
        y = torch.ops.npu.npu_swiglu.default(x)
        qx, xs = torch.ops.npu.npu_dynamic_mx_quant.default(y, dst_type=fp8_dtype)
        return qx, xs

    def swiglu_mx_replace(x):
        qx, xs = torch.ops.npu.npu_swiglu_mx_quant.default(
            x, activate_dim=-1, activate_left=True, dst_type=fp8_dtype
        )
        return qx, xs

    try:
        with FakeTensorMode():
            for dtype in (torch.bfloat16, torch.float16):
                x2d = torch.empty((2, 64), dtype=dtype, device="npu")
                r2d = torch.empty((2, 64), dtype=dtype, device="npu")
                w1d = torch.empty((64,), dtype=dtype, device="npu")
                g2d = torch.empty((2, 128), dtype=dtype, device="npu")
                pm.register_pattern(
                    search_fn=add_rms_mx_search,
                    replace_fn=add_rms_mx_replace,
                    example_inputs=(x2d, r2d, w1d, DEFAULT_EPSILON),
                    scalar_workaround={"epsilon": DEFAULT_EPSILON},
                    skip_duplicates=True,
                )
                pm.register_pattern(
                    search_fn=rms_mx_search,
                    replace_fn=rms_mx_replace,
                    example_inputs=(x2d, w1d, DEFAULT_EPSILON),
                    scalar_workaround={"epsilon": DEFAULT_EPSILON},
                    skip_duplicates=True,
                )
                pm.register_pattern(
                    search_fn=swiglu_mx_search,
                    replace_fn=swiglu_mx_replace,
                    example_inputs=(g2d,),
                    skip_duplicates=True,
                )
        logger.info("Registered Ascend 950 MXFP8 FX fusion patterns in npugraph_ex.")
    except Exception as exc:
        logger.warning("Failed to register npugraph_ex MXFP8 patterns: %s", exc)


def validate_npugraph_ex_config(*, enable_compile, graph_enabled, world_size):
    if not envs.SGLANG_NPU_ENABLE_STATIC_KERNEL.get():
        return
    if not enable_compile:
        raise ValueError("statickernel requires --enable-torch-compile")
    if not graph_enabled:
        raise ValueError("statickernel requires decode NPU graph capture")
    if world_size > 1 and "LOCAL_WORLD_SIZE" not in os.environ:
        # torch_npu installs the compiled run packages once per node and reads
        # the node grouping from LOCAL_WORLD_SIZE; without it static_kernel.py
        # turns the feature off behind a warning. SGLang does not launch through
        # torchrun, so nothing sets the variable on its own.
        raise ValueError(
            "statickernel on more than one rank requires LOCAL_WORLD_SIZE in "
            "the environment; set it to the number of ranks this node launches."
        )


def get_npugraph_ex_backend():
    import npugraph_ex

    # torch_npu distributions ship either the standalone config or the older
    # torchair-compatible layout. Do not hide a missing dependency inside either.
    module_name = "npugraph_ex.configs.compiler_config"
    try:
        config_module = import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        config_module = None

    process_options = getattr(config_module, "_process_kwargs_options", None)
    if process_options is None:
        # Some legacy packages also contain compiler_config, but keep the
        # flat-option translator only in npugraphex_config.
        config_module = import_module("npugraph_ex.configs.npugraphex_config")
        process_options = config_module._process_kwargs_options

    _register_npu_mxfp8_fusion_patterns()

    enable_static_kernel = envs.SGLANG_NPU_ENABLE_STATIC_KERNEL.get()
    enable_super_kernel = envs.SGLANG_NPU_ENABLE_SUPER_KERNEL.get()

    if enable_static_kernel:
        if "LOCAL_WORLD_SIZE" not in os.environ:
            os.environ["LOCAL_WORLD_SIZE"] = "1"
        try:
            from npugraph_ex._acl_concrete_graph.static_kernel import (
                _set_static_kernel_blacklist,
            )

            _set_static_kernel_blacklist(
                [
                    "TransposeQuantBatchMatMul",
                    "QuantLightningIndexer",
                    "PoolKeyIndexer",
                ]
            )
        except Exception as exc:
            logger.debug("Skipping static kernel blacklist setup: %s", exc)

    config = npugraph_ex.CompilerConfig()
    options = {
        # NPUGraphRunner owns capture/replay; nested backend replay is invalid.
        "force_eager": True,
        "inplace_pass": False,
        "clone_input": False,
        "clone_output": False,
        "frozen_parameter": True,
        "remove_noop_ops": True,
        "remove_cat_ops": True,
        "pattern_fusion_pass": True,
        "static_kernel_compile": enable_static_kernel,
    }
    if enable_super_kernel:
        options["super_kernel_optimize"] = True
        options["super_kernel_optimize_options"] = {
            "dcci_after_kernel_end": [".*"],
        }
    process_options(config, {"options": options})
    torch.npu.set_compile_mode(jit_compile=False)
    logger.info(
        "Configured npugraph_ex backend: static_kernel=%s, super_kernel=%s, mxfp8_fusion=%s",
        enable_static_kernel,
        enable_super_kernel,
        envs.SGLANG_NPU_ENABLE_MXFP8_FUSION.get(),
    )
    return npugraph_ex.get_npu_backend(compiler_config=config)
