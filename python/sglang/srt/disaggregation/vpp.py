"""VPP ownership on the PD wire and compressed-MLA buffer selection."""

from __future__ import annotations

from sglang.srt.disaggregation.base.conn import StateType


def vpp_transfer_layers(model) -> list[list[int]] | None:
    layout = getattr(model, "vpp_layout", None)
    if layout is None:
        return None
    return [
        list(layout.layer_ids_for_rank(rank)) for rank in range(layout.physical_size)
    ]


def check_vpp_transfer_layers(
    prefill_layers, decode_layers, *, prefill_tp, decode_tp, decode_pp_size
) -> None:
    if prefill_layers is None and decode_layers is None:
        return
    if prefill_tp != decode_tp:
        raise RuntimeError("VPP PD requires the same TP size on prefill and decode")
    if decode_pp_size == 1 and decode_layers is None:
        return
    if prefill_layers is None or prefill_layers != decode_layers:
        raise RuntimeError(
            "VPP PD requires identical per-rank layer ownership on pipelined "
            "peers, or a PP1 decode peer"
        )


def slice_vpp_mla_entries(kv_args, src, dst, state_type=None):
    """Select each buffer group's noncontiguous global layers on a PP1 peer.

    Every group follows increasing global layer id, but C4 KV, its indexer,
    C128 KV and request states have separate groups. Selecting one enclosing
    interval would silently write the other physical ranks' layers as well.
    """
    if len(src) == len(dst):
        # Bootstrap checked identical rank ownership for pipelined peers.
        return src, dst
    ratios = kv_args.mla_compression_ratios
    local = set(kv_args.vpp_layers[kv_args.pp_rank])
    # The ratio vector can include MTP layers absent from the target KV pool.
    layers = sorted({layer for rank in kv_args.vpp_layers for layer in rank})
    c4 = [layer for layer in layers if ratios[layer] == 4]
    c128 = [layer for layer in layers if ratios[layer] == 128]
    sources = getattr(kv_args, "mla_kv_source_layers", None) or []
    c1 = [layer for layer in layers if layer in sources and ratios[layer] == 1]
    c2 = [layer for layer in layers if layer in sources and ratios[layer] == 2]
    if state_type == StateType.SWA_RING:
        candidates = ([layers], [layers, layers])
    elif state_type == StateType.DSV4_REQUEST_STATE:
        candidates = ([c128, c2],)
    elif state_type == StateType.SWA:
        candidates = ([layers, c4, c4],)
    elif state_type is None:
        candidates = (
            [c4, c4, c128, c1, c1, c2, c2],
            [c4, c4, c128, c1, c1, c1, c2, c2, c2],
            [c4, c4, c4, c128, c128, c1, c1, c2, c2],
        )
    else:
        raise ValueError(f"Unsupported VPP compressed-MLA state: {state_type}")
    matches = set()
    for groups in candidates:
        full = [layer for group in groups for layer in group]
        indices = [i for i, layer in enumerate(full) if layer in local]
        if len(full) == len(dst) and len(indices) == len(src):
            matches.add(tuple(indices))
    if len(matches) != 1:
        raise ValueError(
            "VPP PD compressed-MLA buffer layout differs between peers: "
            f"src={len(src)}, dst={len(dst)}, state={state_type}"
        )
    return src, [dst[i] for i in next(iter(matches))]
