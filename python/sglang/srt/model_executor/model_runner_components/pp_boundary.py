"""Inspect the local layer stack's pipeline-boundary row layouts."""


def _boundary_stages(model):
    first_attn = last_ffn = None
    for module in model.modules():
        attn = getattr(module, "attn_boundary", None)
        if attn is not None and first_attn is None:
            first_attn = attn
        ffn = getattr(module, "ffn_boundary", None)
        if ffn is not None:
            last_ffn = ffn
    return first_attn, last_ffn


def input_scattered(*, model, pp_size: int, pp_rank: int) -> bool:
    """Whether the first local attention stage consumes attn-TP-local rows."""
    if pp_size <= 1 or pp_rank == 0:
        return False
    first_attn = _boundary_stages(model)[0]
    return first_attn is not None and first_attn.input_on_attn_tp_slices


def output_scattered(*, model, pp_size: int, pp_rank: int) -> bool:
    """Whether the last local FFN stage produces attn-TP-local rows."""
    if pp_size <= 1 or pp_rank == pp_size - 1:
        return False
    from sglang.srt.layers.layer_boundary.contracts import BatchVariant
    from sglang.srt.layers.layer_boundary.layout import TokenAxis

    last_ffn = _boundary_stages(model)[1]
    edges = (
        last_ffn.plan.edges.get(BatchVariant.ORDINARY) if last_ffn is not None else None
    )
    produced = getattr(edges, "outgoing", None)
    produced = getattr(produced, "produced", None) if produced else None
    layout = getattr(produced, "layout", None)
    return layout is not None and TokenAxis.ATTN_TP in layout.sharded
