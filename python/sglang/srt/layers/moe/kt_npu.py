# SPDX-License-Identifier: Apache-2.0
# Ascend KT stream/callback handling adapted from Pan-Boyi/sglang,
# dsv4-cann9-no-patch at ff46d660d16cbf11ef393c80cc6b3ab7f6259780.
"""Ascend-specific helpers for KT CPU-expert offload.

Keep torch_npu imports lazy so CUDA installations can import the KT wrapper.
"""

import os

import torch


def validate_kt_npu_config(tp_size, ep_size, num_gpu_experts, a2a_backend):
    experimental_tp2 = os.environ.get("KT_NPU_EXPERIMENTAL_TP2") == "1"
    if ep_size != 1 or tp_size not in ((1, 2) if experimental_tp2 else (1,)):
        raise ValueError(
            "KT offload on NPU requires TP=1/EP=1 by default; "
            "experimental TP=2/EP=1 requires KT_NPU_EXPERIMENTAL_TP2=1"
        )
    if num_gpu_experts is None or num_gpu_experts < 1:
        raise ValueError("KT offload on NPU requires --kt-num-gpu-experts >= 1")
    if a2a_backend != "none":
        raise ValueError("KT offload on NPU requires --moe-a2a-backend none")
    # SGLang owns the ACL report subscriber used by graph host callbacks.
    # The companion kt-kernel must not subscribe to the same stream twice.
    os.environ["KT_EXTERNAL_NPU_REPORT_SUBSCRIBER"] = "1"


def mask_cpu_expert_routing_npu(topk_ids, topk_weights, num_gpu_experts):
    """Use a valid resident slot with zero weight for each CPU-only route.

    Ascend routing cannot consume CUDA's -1 expert sentinel. CPU submission
    must read the original ids/weights before this non-mutating masking step.
    Only the contiguous resident prefix is supported.
    """
    resident = (topk_ids >= 0) & (topk_ids < num_gpu_experts)
    return (
        torch.where(resident, topk_ids, torch.zeros_like(topk_ids)),
        torch.where(resident, topk_weights, torch.zeros_like(topk_weights)),
    )


def resolve_kt_weight_path(weight_path, layer_idx):
    if "{layer_idx}" in weight_path:
        return weight_path.format(layer_idx=layer_idx)
    if weight_path.count("{}") == 1:
        return weight_path.replace("{}", str(layer_idx), 1)
    return weight_path


def npu_graph_capturing():
    import torch_npu  # noqa: F401

    from sglang.srt.model_executor.runner import get_is_capture_mode

    return torch.npu.is_current_stream_capturing() or get_is_capture_mode()


def subscribe_npu_report(stream):
    import torch_npu

    try:
        torch_npu.npu._subscribe_report(stream)
    except RuntimeError as exc:
        # torch_npu 2.10 may already subscribe NPUGraph capture streams.
        if "107011" not in str(exc):
            raise


@torch.no_grad()
def _npu_graph_host_forward(args):
    wrapper, hidden_states, stream_handle = args
    wrapper.run_pinned_forward_sync(hidden_states, stream_handle)


def submit_npu_graph(wrapper, hidden_states, topk_ids, topk_weights):
    import torch_npu

    stream = torch.npu.current_stream(hidden_states.device)
    # Subscription must happen before capture, in process_weights_after_loading.
    wrapper.copy_inputs_to_cpu_buffers(hidden_states, topk_ids, topk_weights)
    torch_npu.npu._launch_host_func(
        stream,
        _npu_graph_host_forward,
        (wrapper, hidden_states, int(stream.npu_stream)),
    )
