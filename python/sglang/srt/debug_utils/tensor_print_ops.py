import torch

from sglang.srt.compilation.compilation_config import register_split_op
from sglang.srt.compilation.piecewise_context_manager import get_forward_context
from sglang.srt.utils.custom_op import register_custom_op


@register_split_op()
@register_custom_op(mutates_args=[])
def print_tensor_meta(t: torch.Tensor, name: str) -> None:
    """
    Debug operator for printing tensor meta info.

    Usage:
        torch.ops.sglang.print_tensor_stats(qkv, "qkv")

    Output Effect:

        ================================================================================
        [TensorStats] <name>
        --------------------------------------------------------------------------------
          shape     : (<dimensions>)
          dtype     : <type>
          data_ptr  : <address>
          min / max : <val> / <val>
          mean      : <val>
          has_nan   : <bool>
          has_inf   : <bool>
        ================================================================================

    """

    ctx = get_forward_context()
    if ctx is None:
        return

    if not t.is_cuda or torch.cuda.current_device() != 0:
        return

    torch.cuda.synchronize()

    is_fp = torch.is_floating_point(t)
    min_v = t.min()
    max_v = t.max()
    mean_v = t.mean() if is_fp else None
    has_nan = torch.isnan(t).any().item() if is_fp else False
    has_inf = torch.isinf(t).any().item() if is_fp else False

    print("\n" + "=" * 80, flush=True)
    print(f"[TensorMeta] {name}", flush=True)
    print("-" * 80, flush=True)

    print(f"  shape     : {tuple(t.shape)}", flush=True)
    print(f"  dtype     : {t.dtype}", flush=True)
    print(f"  data_ptr  : {t.data_ptr()}", flush=True)

    print(f"  min / max : {min_v} / {max_v}", flush=True)
    if mean_v is not None:
        print(f"  mean      : {mean_v}", flush=True)

    if is_fp:
        print(f"  has_nan   : {has_nan}", flush=True)
        print(f"  has_inf   : {has_inf}", flush=True)

    print("=" * 80 + "\n", flush=True)


@register_split_op()
@register_custom_op(mutates_args=[])
def print_tensor_data(t: torch.Tensor, name: str) -> None:
    """
    Debug operator for printing tensor contents.

    Usage:
        torch.ops.sglang.print_tensor_data(qkv[:5,], "qkv[:5,]")
        torch.ops.sglang.print_tensor_data(qkv[,:5], "qkv[,:5]")

    Output Effect:

        ================================================================================
        [TensorData] <name>
        --------------------------------------------------------------------------------
          shape: (<dimensions>)
          <tensor_data>
        ================================================================================

    """
    ctx = get_forward_context()
    if ctx is None:
        return

    if not t.is_cuda or torch.cuda.current_device() != 0:
        return

    torch.cuda.synchronize()

    print("\n" + "=" * 80, flush=True)
    print(f"[TensorData] {name}", flush=True)
    print("-" * 80, flush=True)

    print(f"  shape: {tuple(t.shape)}", flush=True)
    print(f"{t}", flush=True)

    print("=" * 80 + "\n", flush=True)
