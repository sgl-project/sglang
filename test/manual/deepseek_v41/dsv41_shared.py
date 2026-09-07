import torch


class SharedAttentionRuntime:
    """Tensors attention layers hand down the stack: compress_kv and index_k from the
    kv_source layers, topk_idxs from the index_source layers, candidates from the
    candidate_source layer. Layers run in order, so a source always writes before
    its consumers read."""

    def __init__(self):
        self.compress_kv: torch.Tensor | None = None
        self.index_k: torch.Tensor | None = None
        self.topk_idxs: torch.Tensor | None = None
        self.candidates: torch.Tensor | None = None
