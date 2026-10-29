# SPDX-License-Identifier: Apache-2.0
"""Default-off TP4 FN router producer, loaded from architecture-qualified cubins."""

import hashlib
import logging
import os
from functools import cache
from pathlib import Path

import torch

from sglang.kernels.ops.moe import q38_router_jit as jit
from sglang.kernels.ops.moe.q38_router_front import Nvfp4RouterInput, Params
from sglang.kernels.ops.moe.q38_moe_aot import bundle_manifest

logger = logging.getLogger(__name__)
_seen = set()


@cache
def _bundle(device_index):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Q38C router AOT must load before CUDA graph capture")
    split = 4 if os.getenv('SGLANG_Q38C_ROUTER_K4', '0') == '1' else 2
    variable = 'SGLANG_Q38C_ROUTER_K4_AOT' if split == 4 else 'SGLANG_Q38C_ROUTER_AOT'
    root, manifest = bundle_manifest(variable, torch.cuda.get_device_capability(device_index))
    source = Path(__file__).resolve().parents[2] / "jit/csrc/moe/q38_router_front.cu"
    assert manifest["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest["shape"] == [2560, 512, split]
    kernels = {}
    for capacity in (1, 8):
        name = f"route-m{capacity}.cubin"
        data = (root / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == manifest["sha256"][name]
        shared = (8 + capacity) * (2560 * 2 + 16) + split * 16 * 8 * 4 + 16
        kernel = jit.load(data, "q38_router_gemm_quant", (256, 1, 1), shared)
        # M1's static+dynamic total also requires opt-in, despite dynamic <48KiB.
        jit.check(
            jit.driver().cuFuncSetAttribute(
                kernel.function,
                jit.driver().CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared,
            ),
            "Q38C router dynamic shared opt-in",
        )
        kernels[capacity] = kernel
    logger.info("STACK8_ROUTER_BUNDLE arch=%s split=%d root=%s", manifest["arch"], split, root)
    return kernels


@cache
def _wide_bundle(device_index):
    """Load the qualified K2 wide kernels without changing the M1/M4 K4 bundle."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Q38C wide router AOT must load before CUDA graph capture")
    root, manifest = bundle_manifest(
        "SGLANG_Q38C_ROUTER_WIDE_AOT", torch.cuda.get_device_capability(device_index)
    )
    source = Path(__file__).resolve().parents[2] / "jit/csrc/moe/q38_router_wide.cu"
    assert manifest["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest["shape"] == [2560, 512, 2]
    kernels = {}
    for capacity in (8, 16):
        name = f"route-m{capacity}.cubin"
        data = (root / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == manifest["sha256"][name]
        shared = (8 + capacity) * (2560 * 2 + 16) + 2 * 16 * 8 * 4 + 16
        kernel = jit.load(data, "q38_router_gemm_quant", (256, 1, 1), shared)
        jit.check(
            jit.driver().cuFuncSetAttribute(
                kernel.function,
                jit.driver().CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared,
            ),
            "Q38C wide router dynamic shared opt-in",
        )
        kernels[capacity] = kernel
    return kernels


def initialize(module, config):
    """Validate the opt-in target and load both capacities during eager init."""
    from sglang.srt.layers.moe.utils import get_moe_runner_backend
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod
    from sglang.srt.runtime_context import get_parallel

    if not (
        config.model_type == "qwen4_exp_text"
        and config.hidden_size == 2560
        and config.num_experts == 512
        and module.tp_size == 4
        and get_parallel().moe_ep_size == 1
        and get_moe_runner_backend().is_flashinfer_trtllm()
        and not module.enable_shared_expert_fusion
        and isinstance(module.experts.quant_method, ModelOptNvFp4FusedMoEMethod)
        and not module.experts.quant_method.quant_config.use_per_token_activation
    ):
        raise ValueError(
            "Q38C fused router requires FN H2560/E512 NVFP4 TP4/EP1 TRTLLM "
            "and separate shared experts"
        )
    if module._q38_router_front:
        raise ValueError("Choose Q38C fused router or the old TP1 router experiment")
    selected = os.getenv("SGLANG_Q38C_ROUTER_LAYERS", "all")
    module._q38c_router_layers = (
        None if selected == "all" else frozenset(int(v) for v in selected.split(","))
    )
    _bundle(torch.cuda.current_device())
    module._q38c_router_wide = os.getenv("SGLANG_Q38C_ROUTER_WIDE", "0") == "1"
    if module._q38c_router_wide:
        _wide_bundle(torch.cuda.current_device())


def produce(module, hidden):
    """Issue on the caller's main stream, before the existing alt-stream event."""
    m = hidden.shape[0]
    if not (m in (1, 4) or (module._q38c_router_wide and m in (8, 16))) or (
        module._q38c_router_layers is not None
        and module.layer_id not in module._q38c_router_layers
    ):
        return None
    # Padding is established by post-load weight processing, after initialize().
    if module.experts.trtllm_padded_hidden_size is not None:
        raise ValueError("Q38C fused router requires unpadded H2560 expert weights")
    kernels = _bundle(hidden.device.index) if m in (1, 4) else _wide_bundle(hidden.device.index)
    logits = torch.empty((m, 512), dtype=torch.bfloat16, device=hidden.device)
    packed = torch.empty((m, 1280), dtype=torch.uint8, device=hidden.device)
    scales = torch.empty((m, 160), dtype=torch.float8_e4m3fn, device=hidden.device)
    params = Params(
        x=hidden.data_ptr(),
        weight=module.gate.weight.data_ptr(),
        global_scale=module.experts.w13_input_scale_quant.data_ptr(),
        logits=logits.data_ptr(),
        x_fp4=packed.data_ptr(),
        x_sf=scales.data_ptr(),
        num_tokens=m,
        mode=1,
    )
    stream = torch.cuda.current_stream()
    jit.launch(kernels[1 if m == 1 else 8 if m <= 8 else 16], (64, 1, 1), [params], stream.cuda_stream)
    signature = module.layer_id, m, torch.cuda.is_current_stream_capturing()
    if signature not in _seen:
        _seen.add(signature)
        logger.info(
            "Q38C_ROUTER_FUSED_ACTIVE layer=%d M=%d capture=%s producer_stream=%d",
            *signature,
            stream.cuda_stream,
        )
    return logits, Nvfp4RouterInput(packed, scales)
