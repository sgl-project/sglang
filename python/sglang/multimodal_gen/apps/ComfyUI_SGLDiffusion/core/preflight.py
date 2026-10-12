# SPDX-License-Identifier: Apache-2.0
"""Reject accelerations this environment cannot run before the SGLD worker starts.

The worker is spawned from this interpreter, so asking SGLang's own backend
resolution here gives the same answer the worker would get after loading the
model; a missing kernel then fails in the loader node instead of at the first
sampling step, or silently as another backend.
"""

from pkgutil import resolve_name

import torch

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.platforms.interface import AttentionBackendEnum

_INSTALL_HINTS = {
    AttentionBackendEnum.SAGE_ATTN: "install SageAttention 2: https://github.com/thu-ml/SageAttention",
    AttentionBackendEnum.SAGE_ATTN_3: (
        "SageAttention 3 targets Blackwell GPUs: "
        "https://github.com/thu-ml/SageAttention/tree/main/sageattention3_blackwell"
    ),
}


def requested_attention_backends(sgld_options: dict) -> dict[str, str]:
    """Option name -> requested backend, for every attention backend the options select."""
    requested = {}
    if sgld_options.get("attention_backend"):
        requested["attention_backend"] = sgld_options["attention_backend"]
    components = sgld_options.get("component_attention_backends") or {}
    if isinstance(components, str):
        components = dict(
            item.split("=", 1) for item in components.split(",") if "=" in item
        )
    for component, backend in components.items():
        requested[f"component_attention_backends.{component.strip()}"] = backend
    return requested


def _resolved_backend(backend: AttentionBackendEnum) -> AttentionBackendEnum:
    # Head size and dtype of a typical DiT layer; only availability matters here.
    cls_path = current_platform.get_attn_backend_cls_str(backend, 128, torch.bfloat16)
    return resolve_name(cls_path).get_enum()


def check_attention_backends(sgld_options: dict) -> None:
    for option, name in requested_attention_backends(sgld_options).items():
        try:
            backend = AttentionBackendEnum[name.strip().upper()]
        except KeyError:
            raise ValueError(
                f"{option}={name!r} is not an SGLang attention backend"
            ) from None
        try:
            resolved = _resolved_backend(backend)
        except (ImportError, ValueError) as error:
            raise ValueError(
                f"{option}={name} cannot run in this environment: {error}"
            ) from error
        if resolved is not backend:
            hint = _INSTALL_HINTS.get(backend, "install it or choose another backend")
            raise ValueError(
                f"{option}={name} is not available in this environment (SGLang "
                f"would run {resolved.name.lower()} instead); {hint}"
            )


def check_parallel_layout(sgld_options: dict) -> None:
    dp = sgld_options.get("dp_size") or 1
    if dp > 1:
        # Each sampler step is one request carrying CUDA IPC tensors from this
        # process, and ComfyUI runs one prompt at a time.
        raise ValueError(
            "dp_size > 1 is not supported in ComfyUI integrated mode: replicas never "
            "overlap behind ComfyUI's single prompt queue. Run one ComfyUI instance "
            "per GPU, or use tp_size / sp_degree for one request."
        )
    tp = sgld_options.get("tp_size")
    sp = sgld_options.get("sp_degree")
    if (
        sp is None
        and sgld_options.get("ulysses_degree")
        and sgld_options.get("ring_degree")
    ):
        sp = sgld_options["ulysses_degree"] * sgld_options["ring_degree"]
    if tp is None or sp is None:
        return  # SGLang fills unset degrees from num_gpus
    cfg = (
        (sgld_options.get("cfg_parallel_degree") or 2)
        if sgld_options.get("enable_cfg_parallel")
        else 1
    )
    num_gpus = sgld_options.get("num_gpus") or 1
    if dp * tp * sp * cfg != num_gpus:
        # A GPU outside the layout gets no process group; its worker hangs at startup.
        raise ValueError(
            f"num_gpus ({num_gpus}) must equal tp_size * sp_degree"
            f"{' * cfg_parallel_degree' if cfg > 1 else ''} = {dp * tp * sp * cfg}; "
            "set the degrees to fill every GPU or leave sp_degree at -1 (auto)"
        )


def check_sgld_options(sgld_options: dict) -> None:
    check_attention_backends(sgld_options)
    check_parallel_layout(sgld_options)
