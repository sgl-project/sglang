"""Row-bounded FP32 mHC residual mixing for Ascend 910B."""

import torch


def _post_torch_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Accumulate residual mixing and the sublayer output in FP32."""
    return (
        torch.bmm(comb.float().transpose(1, 2), residual.float())
        + post_mix.float().unsqueeze(-1) * x.float().unsqueeze(1)
    ).to(x.dtype)


def post_910b(x, residual, post_mix, comb):
    rows = x.shape[0]
    if rows <= 512:
        return _post_torch_impl(x, residual, post_mix, comb)
    output = torch.empty_like(residual, dtype=x.dtype)
    for start in range(0, rows, 512):
        batch = slice(start, start + 512)
        output[batch].copy_(
            _post_torch_impl(x[batch], residual[batch], post_mix[batch], comb[batch])
        )
    return output
