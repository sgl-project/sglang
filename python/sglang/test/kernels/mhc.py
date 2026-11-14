"""Pure-torch references for the mHC (hyper-connection) kernels."""

import torch


def hc_split_sinkhorn_torch(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """Pure-torch equivalent of ``mhc.hc_split_sinkhorn_kernel``.

    Layout mirrors the kernel exactly: the flattened ``mixes`` row holds ``pre``
    (hc), ``post`` (hc) and the ``comb`` matrix (hc * hc) consecutively.
    """
    b, s, _ = mixes.size()
    hc = hc_mult
    flat = mixes.reshape(-1, (2 + hc) * hc).float()
    scale = hc_scale.float()
    base = hc_base.float()

    pre = torch.sigmoid(flat[:, :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(flat[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc])

    comb = flat[:, 2 * hc :] * scale[2] + base[2 * hc :]
    comb = comb.reshape(-1, hc, hc)

    # Initial row softmax (numerically stabilized) then column normalize.
    row_max = comb.amax(dim=2, keepdim=True)
    comb = torch.exp(comb - row_max)
    comb = comb / comb.sum(dim=2, keepdim=True) + eps
    comb = comb / (comb.sum(dim=1, keepdim=True) + eps)

    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=2, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=1, keepdim=True) + eps)

    pre = pre.reshape(b, s, hc).to(mixes.dtype)
    post = post.reshape(b, s, hc).to(mixes.dtype)
    comb = comb.reshape(b, s, hc, hc).to(mixes.dtype)
    return pre, post, comb
