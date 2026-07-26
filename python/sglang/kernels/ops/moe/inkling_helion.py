"""Optional Helion activation kernels for Inkling."""

from functools import partial

try:
    import helion
    import helion.language as hl
except ModuleNotFoundError as error:
    if error.name == "helion" or (
        error.name is not None and error.name.startswith("helion.")
    ):
        raise ImportError(
            "Inkling's Helion kernels require the optional 'helion' package. "
            'Install it with `pip install "sglang[helion]"`.'
        ) from error
    raise
import torch

from sglang.srt.layers.moe.moe_runner.triton_utils.helion_utils import (
    get_model_depths,
    helion_aot_autotune,
)


def silu_and_mul_key(
    gateup_output: torch.Tensor,
    topk_weights: torch.Tensor | None,
    out_dtype: object | None = None,
):
    # Keep this stable across inputs for Helion AOT autotune.
    del out_dtype
    return gateup_output.shape[1], (gateup_output.dtype,), (topk_weights is not None,)


def silu_and_mul_inputs(sizes: list[int]):
    # Used only for Helion autotune input generation.
    inputs = []
    numel = 2**30
    with torch.device("cuda"):
        for size in sizes:
            x = torch.randn(numel // size, 2 * size, dtype=torch.bfloat16)
            inputs.append((x, None, None))
            inputs.append((x, torch.randn(numel // size, dtype=torch.bfloat16), None))
    return inputs


@helion_aot_autotune(
    "silu_and_mul_interleaved",
    kernel_key=silu_and_mul_key,
    primary_inputs=partial(silu_and_mul_inputs, sizes=[512, 2048, 48 * 96, 6144, 8192]),
    secondary_inputs=partial(
        silu_and_mul_inputs,
        sizes=[512]
        + [i * 96 for i in get_model_depths()]
        + list(range(1024, 8192 + 1, 1024)),
    ),
)
@helion.kernel(static_shapes=False)
def _silu_and_mul_helion_interleaved_kernel(
    gateup_output,
    topk_weights: torch.Tensor | None = None,
    out_dtype: hl.constexpr | None = None,
):
    """Apply SiLU to interleaved gate/up values and multiply them."""
    batch_size, hidden_size = gateup_output.shape
    hidden_size = hl.specialize(hidden_size)
    assert hidden_size % 2 == 0, f"{hidden_size=}"

    half_hidden_size = hidden_size // 2
    down_input = gateup_output.new_empty(
        batch_size, half_hidden_size, dtype=out_dtype or gateup_output.dtype
    )
    for batch_tile, hidden_tile in hl.tile([batch_size, half_hidden_size]):
        gate_output = gateup_output[batch_tile, 2 * hidden_tile.index].to(torch.float32)
        up_output = gateup_output[batch_tile, 2 * hidden_tile.index + 1].to(
            torch.float32
        )
        silu_mul_output = gate_output * torch.sigmoid(gate_output) * up_output
        if topk_weights is not None:
            weight_scale = topk_weights[batch_tile, None].to(torch.float32)
            silu_mul_output = silu_mul_output * weight_scale
        down_input[batch_tile, hidden_tile] = silu_mul_output
    return down_input


@helion_aot_autotune(
    "silu_and_mul",
    kernel_key=silu_and_mul_key,
    primary_inputs=partial(silu_and_mul_inputs, sizes=[512, 2048, 48 * 96, 6144, 8192]),
    secondary_inputs=partial(
        silu_and_mul_inputs,
        sizes=[512]
        + [i * 96 for i in get_model_depths()]
        + list(range(1024, 8192 + 1, 1024)),
    ),
)
@helion.kernel(static_shapes=False)
def _silu_and_mul_helion_non_interleaved_kernel(
    gateup_output,
    topk_weights: torch.Tensor | None = None,
    out_dtype: hl.constexpr | None = None,
):
    """Apply SiLU to separate gate/up halves and multiply them."""
    batch_size, hidden_size = gateup_output.shape
    hidden_size = hl.specialize(hidden_size)
    assert hidden_size % 2 == 0, f"{hidden_size=}"

    half_hidden_size = hidden_size // 2
    down_input = gateup_output.new_empty(
        batch_size, half_hidden_size, dtype=out_dtype or gateup_output.dtype
    )
    for batch_tile, hidden_tile in hl.tile([batch_size, half_hidden_size]):
        gate_output = gateup_output[batch_tile, hidden_tile.index].to(torch.float32)
        up_output = gateup_output[batch_tile, hidden_tile.index + half_hidden_size].to(
            torch.float32
        )
        silu_mul_output = gate_output * torch.sigmoid(gate_output) * up_output
        if topk_weights is not None:
            weight_scale = topk_weights[batch_tile, None].to(torch.float32)
            silu_mul_output = silu_mul_output * weight_scale
        down_input[batch_tile, hidden_tile] = silu_mul_output
    return down_input


def silu_and_mul_helion(
    gateup_output: torch.Tensor,
    topk_weights: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    use_interleaved: bool = True,
) -> torch.Tensor:
    """Dispatch to the interleaved or non-interleaved Helion activation."""
    if use_interleaved:
        return _silu_and_mul_helion_interleaved_kernel(
            gateup_output, topk_weights, out_dtype
        )
    return _silu_and_mul_helion_non_interleaved_kernel(
        gateup_output, topk_weights, out_dtype
    )
