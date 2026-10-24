# SPDX-License-Identifier: Apache-2.0
"""Load the gfx1250 FA3 implementation only when its Gluon APIs are present."""

import importlib


def load_forward():
    try:
        gluon = importlib.import_module("triton.experimental.gluon")
        language = importlib.import_module("triton.experimental.gluon.language")
    except ImportError as exc:
        return None, str(exc)

    # JIT bodies resolve these APIs only at compilation time. Probe the generic
    # and persistent kernel requirements before dispatch so older Triton
    # installations can fall back without hiding compilation or runtime errors.
    required_apis = (
        (
            gluon,
            "triton.experimental.gluon",
            ("aggregate", "constexpr_function", "jit"),
        ),
        (
            language,
            "triton.experimental.gluon.language",
            (
                "PaddedSharedLayout.with_identity_for",
                "DotOperandLayout",
                "SliceLayout",
                "SwizzledSharedLayout",
                "allocate_shared_memory",
                "shared_memory_descriptor",
                "amd.AMDWMMALayout",
                "amd.warp_pipeline_stage",
                "amd.slice",
                "amd.cdna5.buffer_load",
                "amd.cdna5.buffer_store",
                "amd.cdna5.wmma",
                "amd.cdna5.tdm.tensor_descriptor",
                "amd.cdna5.tdm.make_tensor_descriptor",
                "amd.cdna5.tdm.async_load",
                "amd.cdna5.tdm.async_wait",
                "amd.cdna5.tdm.async_store",
                "amd.cdna5.tdm.async_load_fused",
                "amd.cdna5.tdm.update_tensor_descriptor",
                "amd.cdna5.tdm.prefetch",
            ),
        ),
    )
    for module, module_name, paths in required_apis:
        for path in paths:
            api = module
            for attribute in path.split("."):
                api = getattr(api, attribute, None)
                if api is None:
                    return None, f"missing required Gluon API {module_name}.{path}"

    try:
        kernel_module = importlib.import_module(
            "sglang.kernels.ops.attention.gluon.fa3.gfx1250.forward"
        )
    except ImportError as exc:
        return None, str(exc)
    return kernel_module.flash_attention, None
