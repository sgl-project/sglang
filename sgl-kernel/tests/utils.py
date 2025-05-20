from typing import Optional

import torch
from sgl_kernel.scalar_type import scalar_types


def marlin_make_empty_g_idx(device: torch.device) -> torch.Tensor:
    return torch.nn.Parameter(
        torch.empty(0, dtype=torch.int, device=device), requires_grad=False
    )


def marlin_make_workspace_new(
    device: torch.device, max_blocks_per_sm: int = 1
) -> torch.Tensor:
    # In the new marlin kernel, we use the num of threadblocks as workspace
    # size. The num of threadblocks is is sms_count * max_blocks_per_sm.
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(
        sms * max_blocks_per_sm, dtype=torch.int, device=device, requires_grad=False
    )


# For binary size and compile time, we don't support the same types for with and
#  without runtime zero-point. We support common cases, i.e. AWQ and GPTQ.
#  TODO: we may want to move this into the C++ so its closer to the actual impl
def query_marlin_supported_quant_types(
    has_zp: bool, device_capability: Optional[int] = None
):

    if has_zp:
        # AWQ style, unsigned + runtime zero-point
        return [scalar_types.uint4, scalar_types.uint8]
    else:
        # GPTQ style, unsigned + symmetric bias
        # TODO: once fp8_marlin is merged into "gptq_marlin" we should be able
        #  to add `scalar_types.float8_e4m3fn` here
        return [scalar_types.uint4b8, scalar_types.uint8b128]
