import torch


class DFlyRequestState:
    """Per-request persistent state carried between target forwards for DFly2.

    Stores:
      - ``hidden``: post-final-norm target hidden state at each request's last
        accepted token (``[pool_size, hidden_size]``).
      - ``topk``: per-draft-layer target sparse-indexer top-k pool slots
        captured at each request's last accepted token
        (``[num_layers, pool_size, target_topk]`` int32, ``-1`` padded).
    """

    def __init__(
        self,
        *,
        pool_size: int,
        hidden_size: int,
        num_layers: int,
        target_topk: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.hidden = torch.zeros((pool_size, hidden_size), dtype=dtype, device=device)
        self.topk = torch.full(
            (num_layers, pool_size, target_topk), -1, dtype=torch.int32, device=device
        )

    def write(
        self,
        *,
        req_pool_indices: torch.Tensor,
        hidden: torch.Tensor,
        topk: torch.Tensor,
    ) -> None:
        self.hidden[req_pool_indices] = hidden
        self.topk[:, req_pool_indices] = topk
