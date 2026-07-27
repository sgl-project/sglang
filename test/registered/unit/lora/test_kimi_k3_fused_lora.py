import torch
from torch import nn

from sglang.srt.lora.layers import RowParallelLinearWithLoRA
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _IdentityQuantMethod:
    def apply(self, layer, input_, bias=None):
        return input_.clone()


def test_row_parallel_lora_preserves_k3_input_transform():
    """K3 applies its MLA output gate to o_proj's input via
    ``lora_input_transform`` rather than inside ``o_proj.forward``, so the LoRA
    A-side sees the same gated input the base GEMM does. If the wrapper skips
    the transform, the adapter is trained against gated input but served
    against ungated input -- a silent train/serve skew, not a crash.
    """
    base_layer = nn.Module()
    base_layer.input_is_parallel = True
    base_layer.tp_size = 1
    base_layer.tp_rank = 0
    base_layer.bias = None
    base_layer.skip_bias_add = False
    base_layer.reduce_results = False
    base_layer.quant_method = _IdentityQuantMethod()
    base_layer.lora_input_transform = lambda tensor: tensor * 3

    wrapper = RowParallelLinearWithLoRA.__new__(RowParallelLinearWithLoRA)
    nn.Module.__init__(wrapper)
    wrapper.base_layer = base_layer
    wrapper.set_lora = False

    output, output_bias = wrapper(torch.tensor([[1.0, 2.0]]))

    assert torch.equal(output, torch.tensor([[3.0, 6.0]]))
    assert output_bias is None
