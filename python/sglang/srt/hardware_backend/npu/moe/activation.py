import torch


class NPUSwiglu:
    """
    Standard SwiGLU activation (NPU backend).
    """

    def _apply_activation(self, hidden_states: torch.Tensor):
        hidden_states = torch.ops.npu.npu_swiglu(hidden_states)
        return hidden_states


class NPUSwigluQuant:
    """
    SwiGLU activation with quantization
    Return (hidden_states, swiglu_out_scale).
    """

    def _apply_activation(self, hidden_states: torch.Tensor):
        hidden_states, swiglu_out_scale = torch.ops.npu.npu_dequant_swiglu_quant(
            hidden_states,
            quant_mode=1,
            activate_left=True,
        )
        return hidden_states, swiglu_out_scale
