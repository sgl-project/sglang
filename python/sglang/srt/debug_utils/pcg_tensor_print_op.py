from sglang.srt.compilation.piecewise_context_manager import get_forward_context
from sglang.srt.utils.custom_op import register_custom_op
import torch
@register_custom_op(mutates_args=[],eager=True)
def print_tensor_debug(
    t: torch.Tensor,
    name: str,
) -> None:
    ctx = get_forward_context()
    if ctx is None:
        return

    if not t.is_cuda or torch.cuda.current_device() != 0:
        return


    if torch.cuda.is_available():
        torch.cuda.synchronize()

    is_fp = torch.is_floating_point(t)

    min_v = t.min()
    max_v = t.max()
    mean_v = t.mean() if is_fp else None

    has_nan = torch.isnan(t).any().item() if is_fp else False
    has_inf = torch.isinf(t).any().item() if is_fp else False

    print("\n" + "=" * 80, flush=True)
    print(f"[TensorDebug] {name}", flush=True)
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

