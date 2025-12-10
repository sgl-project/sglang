import sgl_kernel_npu.norm.l1_norm
import torch


@torch.library.custom_op("sglang::l1_norm", mutates_args=())
def l1_norm(input: torch.Tensor) -> torch.Tensor:
    return sgl_kernel_npu.norm.l1_norm.l1_norm(input)


@l1_norm.register_fake
def l1_norm_fake(input: torch.Tensor) -> torch.Tensor:
    batch_size = input.shape[0]
    hidden_size = input.shape[1]
    output = torch.empty(
        batch_size, hidden_size, device=input.device, dtype=torch.float32
    )
    return output
