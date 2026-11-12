from dataclasses import dataclass, field

import torch
from torch import nn


def block_attn_res(
    blocks: list[torch.Tensor],
    partial_block: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
    queries: torch.Tensor,
    *,
    normalize_values: bool = False,
    scaled_queries: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    if queries.ndim == 1:
        queries = queries.unsqueeze(0)
    assert queries.ndim == 2
    partial_addend = None
    if isinstance(partial_block, tuple):
        partial_block, partial_addend = partial_block
    sources = [*blocks, partial_block] if partial_block is not None else list(blocks)
    hidden_size = sources[0].shape[-1]
    assert queries.shape[-1] == hidden_size
    assert all(
        x.shape == sources[0].shape and x.dtype == sources[0].dtype for x in sources
    )
    effective_queries = (
        scaled_queries.view(queries.shape)
        if scaled_queries is not None
        else queries.float() * (hidden_size**-0.5)
    )
    from sglang.kernels.ops.inkling2.attn_res import attn_res

    return attn_res(
        [x.contiguous() for x in sources],
        partial_addend.contiguous() if partial_addend is not None else None,
        effective_queries.contiguous(),
        normalize_values=normalize_values,
    )


class Inkling2AttnRes(nn.Module):
    def __init__(
        self,
        n_layers: int,
        hidden_dim: int,
        n_layers_per_attn_res_block: int,
        local_layer_ids: list[int],
        normalize_values: bool = False,
    ):
        super().__init__()
        assert 0 < n_layers_per_attn_res_block <= n_layers
        self.n_layers_per_attn_res_block = n_layers_per_attn_res_block
        self.local_layer_ids = frozenset(local_layer_ids)
        self.normalize_values = normalize_values
        self.attn_queries = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(
                        hidden_dim if i in self.local_layer_ids else (3, hidden_dim),
                        dtype=torch.float32,
                    )
                )
                for i in range(1, n_layers)
            ]
        )
        self.mlp_queries = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float32))
                for _ in range(n_layers)
            ]
        )
        self.output_query = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float32))

    def scale_queries(self) -> None:
        """Cache queries * hidden**-0.5, updating in place on weight reload."""
        queries = [*self.attn_queries, *self.mlp_queries, self.output_query]
        scaled = getattr(self, "_scaled_queries", None)
        if scaled is None or len(scaled) != len(queries):
            self._scaled_queries = [q.float() * (q.shape[-1] ** -0.5) for q in queries]
            return
        for target, q in zip(scaled, queries):
            target.copy_(q.float() * (q.shape[-1] ** -0.5))

    def scaled_query(self, kind: str, layer_idx: int) -> torch.Tensor | None:
        scaled = getattr(self, "_scaled_queries", None)
        if scaled is None:
            return None
        n_attn = len(self.attn_queries)
        if kind == "attn":
            return scaled[layer_idx - 1]
        if kind == "mlp":
            return scaled[n_attn + layer_idx]
        return scaled[-1]

    def is_block_boundary(self, layer_idx: int) -> bool:
        return layer_idx % self.n_layers_per_attn_res_block == 0

    def attn_query(self, layer_idx: int) -> torch.Tensor:
        assert layer_idx > 0
        return self.attn_queries[layer_idx - 1]

    def compute_output(
        self, blocks: list[torch.Tensor], partial: torch.Tensor
    ) -> torch.Tensor:
        return block_attn_res(
            blocks,
            partial,
            self.output_query,
            normalize_values=self.normalize_values,
            scaled_queries=self.scaled_query("output", 0),
        )[0]


@dataclass
class Inkling2AttnResState:
    attn_res: Inkling2AttnRes
    partial: torch.Tensor
    blocks: list[torch.Tensor] = field(default_factory=list)

    def attention_input(self, layer_idx: int) -> tuple[torch.Tensor, ...]:
        if not self.blocks:
            return (self.partial,) * (
                1 if layer_idx in self.attn_res.local_layer_ids else 3
            )
        return block_attn_res(
            self.blocks,
            self.partial,
            self.attn_res.attn_query(layer_idx),
            normalize_values=self.attn_res.normalize_values,
            scaled_queries=self.attn_res.scaled_query("attn", layer_idx),
        )

    def mlp_input(self, layer_idx: int, attn_output: torch.Tensor) -> torch.Tensor:
        if self.attn_res.is_block_boundary(layer_idx):
            blocks, partial = [*self.blocks, self.partial], attn_output
        else:
            blocks, partial = self.blocks, (self.partial, attn_output)
        return block_attn_res(
            blocks,
            partial,
            self.attn_res.mlp_queries[layer_idx],
            normalize_values=self.attn_res.normalize_values,
            scaled_queries=self.attn_res.scaled_query("mlp", layer_idx),
        )[0]

    def finalize_layer(
        self, layer_idx: int, attn_output: torch.Tensor, mlp_output: torch.Tensor
    ) -> None:
        if self.attn_res.is_block_boundary(layer_idx):
            self.blocks.append(self.partial)
            self.partial = attn_output + mlp_output
        else:
            from sglang.kernels.ops.inkling2.residual import residual_add

            self.partial = residual_add(self.partial, attn_output, mlp_output)

    def finalize_layer_with(self, layer_idx: int, update) -> None:
        """Like finalize_layer, with update(partial or None) computing
        partial + attention + mlp (or attention + mlp at a block boundary)."""
        if self.attn_res.is_block_boundary(layer_idx):
            self.blocks.append(self.partial)
            self.partial = update(None)
        else:
            self.partial = update(self.partial)

    def compute_output(self) -> torch.Tensor:
        return self.attn_res.compute_output(self.blocks, self.partial)
