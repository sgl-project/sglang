"""Padding-preserving logical-to-physical HiSparse slot translation."""

import torch
import triton
import triton.language as tl


@triton.jit
def _translate_padded_hisparse_locations(
    mapping,
    locations,
    output,
    count,
    stride,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    loc = tl.load(locations + offset * stride, offset < count, other=-1)
    physical = tl.load(
        mapping + tl.maximum(loc, 0), (offset < count) & (loc >= 0), other=0
    )
    tl.store(output + offset, tl.where(loc >= 0, physical, loc), offset < count)


def translate_padded_hisparse_locations(
    mapping: torch.Tensor, locations: torch.Tensor
) -> torch.Tensor:
    """Translate logical token locations into physical GPU cache rows.

    Tensor layout (all tensors are 1D):
      mapping: contiguous [num_mapping_entries].
        mapping[logical_slot] = physical GPU cache row for that logical slot.
      locations: possibly strided [num_tokens].
        locations[i] = logical slot for token i; negative values mean padding.
      output: contiguous [num_tokens].
        output[i] = mapping[locations[i]] for a nonnegative location;
        otherwise output[i] = locations[i], preserving the padding value.

    There is no layer axis: each layer uses the row numbers in its own KV buffer.
    For example, mapping[17] = 3 and mapping[18] = 5 translate locations
    [17, 18, -1] into [3, 5, -1]. Nonnegative locations must index within mapping.

    Inputs are int32/int64 tensors on the same device and remain unchanged.
    Output uses that device and the promoted integer dtype of both inputs.
    """
    assert mapping.ndim == locations.ndim == 1 and mapping.is_contiguous()
    assert mapping.device == locations.device
    assert mapping.dtype in (torch.int32, torch.int64)
    assert locations.dtype in (torch.int32, torch.int64)
    output = torch.empty(
        locations.shape,
        device=locations.device,
        dtype=torch.promote_types(mapping.dtype, locations.dtype),
    )
    if locations.numel():
        _translate_padded_hisparse_locations[(triton.cdiv(locations.numel(), 128),)](
            mapping, locations, output, locations.numel(), locations.stride(0), 128
        )
    return output
