"""npugraph_ex compilation inside SGLang-owned NPU graph capture."""

import os
from importlib import import_module

import torch

from sglang.srt.environ import envs


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

    config = npugraph_ex.CompilerConfig()
    options = {
        # NPUGraphRunner owns capture/replay; nested backend replay is invalid.
        "force_eager": True,
        "inplace_pass": False,
        "clone_input": False,
        "clone_output": False,
        "static_kernel_compile": envs.SGLANG_NPU_ENABLE_STATIC_KERNEL.get(),
    }
    process_options(config, {"options": options})
    torch.npu.set_compile_mode(jit_compile=False)
    return npugraph_ex.get_npu_backend(compiler_config=config)
