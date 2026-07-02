"""Paged expert pager: the per-step residency decision over the K-slot GPU pool.

The pager owns *which* expert lives in *which* slot and when — a host-side keep-warm + LRU decision each
decode step — and hands the resulting ``(src_experts, dst_slots)`` plan to its ``ExpertStore``
(``store.py``), which owns the host backing and the actual byte movement (pinned ``transfer_kv`` or a
pageable copy). Slots 0..K-1 start holding experts 0..K-1 (what sglang's native loader put there);
``logical_to_gpu_index[e]`` is the slot of expert e (-1 if not resident) and its device mirror drives the
forward remap. Store fill from the checkpoint (marlin repack for gptq-int4, direct copy for bf16 — no
offline artifact) lives in ``setup_pager`` below.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional

import torch

from sglang.srt.layers.moe.paged_experts.policy import (
    ResidencyPolicy,
    make_residency_policy,
)
from sglang.srt.layers.moe.paged_experts.store import ExpertStore, make_expert_store

logger = logging.getLogger(__name__)

# ALL pagers in model-layer order (appended by setup_pager) — the wave path's next-layer prefetch.
_ALL_PAGERS: list = []


# --- Replay-twice registry (captured pinned-window fallback) -------------------------------------------
# Each windowed layer registers its pager here and gets a slot in a shared device miss-vector. After a
# captured decode replay, the post-replay hook polls the whole vector in ONE D2H: if every layer hit its
# window (count 0) the token is correct and we stop; otherwise each missed layer stages its deferred cold
# experts into their GPU slots out-of-graph and we replay the SAME graph again (the residency maps it reads
# are fixed-address, so the next replay sees them resident). Converges in ~1 extra replay.


class ExpertPager:
    """Per-step residency decision over the K-slot pool; delegates backing + page-in to an ``ExpertStore``
    and the eviction choice to a ``ResidencyPolicy``.

    The positional constructor ``(layer, E, K, device, pin_host=...)`` builds the matching store, or
    pass a prebuilt ``store=`` to compose one directly (what ``setup_pager`` does). ``eviction`` selects
    the residency policy (``lru`` default | ``lfu``).
    """

    def __init__(
        self,
        layer=None,
        num_experts_E: int = 0,
        num_resident_K: int = 0,
        device=None,
        pin_host: bool = True,
        *,
        store: Optional[ExpertStore] = None,
        eviction: str = "lru",
    ):
        self.store = store or make_expert_store(
            layer, num_experts_E, num_resident_K, device, pin_host=pin_host
        )
        self.E = self.store.E
        self.K = self.store.K
        self.device = self.store.device

        # Residency state (host-side decide; the store does the device transfer). Slots 0..K-1 start
        # holding experts 0..K-1 (what the native loader put there). logical_to_gpu_index[e] is the slot
        # of expert e (-1 if not resident); its device mirror drives the remap each step. The policy owns
        # the eviction choice + its recency/frequency bookkeeping (see policy.py).
        self.policy: ResidencyPolicy = make_residency_policy(eviction, self.K, self.E)
        self.slot_expert = list(range(self.K))  # slot -> expert id (-1 == empty)
        self.logical_to_gpu_index = torch.full((self.E,), -1, dtype=torch.int32)
        self.logical_to_gpu_index[: self.K] = torch.arange(self.K, dtype=torch.int32)
        self.logical_to_gpu_index_cuda = self.logical_to_gpu_index.to(self.device)

    # --- backing delegated to the store (exposed on the pager for the fill code) ---
    @property
    def gpu(self) -> Dict[str, torch.Tensor]:
        return self.store.gpu

    @property
    def host(self) -> Dict[str, torch.Tensor]:
        return self.store.host

    @property
    def item_bytes(self) -> Dict[str, int]:
        return self.store.item_bytes

    @property
    def pin_host(self) -> bool:
        return self.store.pinned

    def page_in(self, src_experts: torch.Tensor, dst_slots: torch.Tensor) -> None:
        """Page the chosen experts into their slots via the store (transport-specific; a no-op if empty)."""
        self.store.page_in(src_experts, dst_slots)

    def distinct_active(self, topk_ids: torch.Tensor):
        """Sorted distinct active (>=0) expert ids this step, as a host list (one host sync)."""
        return [int(e) for e in torch.unique(topk_ids).tolist() if e >= 0]

    def decide_keep_warm(self, topk_ids: torch.Tensor, distinct=None):
        """Host-side residency decision (eager keep-warm): for each distinct active expert not resident,
        evict a non-needed slot (chosen by ``self.policy`` — LRU/LFU) and assign it. Updates the maps in
        place and returns ``(src_experts, dst_slots)`` (device int64) for ``page_in``. **Requires
        ``len(distinct) <= K``** — the caller routes steps with more distinct experts to the wave path
        (see forward.py). Data-dependent -> not capturable (the eager path).
        """
        self.policy.begin_step()
        if distinct is None:
            distinct = self.distinct_active(topk_ids)
        l2g = self.logical_to_gpu_index
        needed = set(distinct)
        for e in distinct:  # touch recency/frequency of resident hits
            s = int(l2g[e])
            if s >= 0:
                self.policy.record_use(e, s)
        src, dst = [], []
        for e in distinct:
            if int(l2g[e]) >= 0:
                continue  # already resident (or just assigned)
            victim = self.policy.pick_victim(self.slot_expert, needed)
            if victim < 0:
                continue  # pool too small (shouldn't happen: distinct <= K)
            old = self.slot_expert[victim]
            if old >= 0:
                l2g[old] = -1
            self.slot_expert[victim] = e
            l2g[e] = victim
            self.policy.record_use(e, victim)  # the fresh assignment counts as a use
            src.append(e)
            dst.append(victim)
        self.logical_to_gpu_index_cuda.copy_(l2g)
        return (
            torch.tensor(src, dtype=torch.int64, device=self.device),
            torch.tensor(dst, dtype=torch.int64, device=self.device),
        )

    def set_residency(self, experts) -> None:
        """Force slot ``i`` to hold ``experts[i]`` and rebuild the maps. Called after the wave path so
        the next keep-warm step's residency state matches what is physically in the slots.
        """
        experts = list(experts)
        self.slot_expert = experts + [-1] * (self.K - len(experts))
        self.logical_to_gpu_index.fill_(-1)
        for i, e in enumerate(experts):
            self.logical_to_gpu_index[e] = i
        self.logical_to_gpu_index_cuda.copy_(self.logical_to_gpu_index)


def _snapshot_dir(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    from huggingface_hub import snapshot_download

    return snapshot_download(model_path, local_files_only=True)


def _weight_map(snap: str) -> Dict[str, str]:
    """{tensor_name: shard_file}; falls back to the single .safetensors when there's no index.json
    (small/quantized checkpoints are often one file)."""
    import glob

    idx = os.path.join(snap, "model.safetensors.index.json")
    if os.path.exists(idx):
        return json.load(open(idx))["weight_map"]
    from safetensors import safe_open

    files = glob.glob(os.path.join(snap, "*.safetensors"))
    assert len(files) == 1, f"no index.json and != 1 safetensors shard: {files}"
    with safe_open(files[0], framework="pt") as f:
        return {k: os.path.basename(files[0]) for k in f.keys()}


def _experts_prefix(wmap: Dict[str, str], layer_idx: int) -> str:
    """The checkpoint name prefix of this layer's routed experts. Text-only checkpoints use
    ``model.layers.N.mlp.experts.``; VL checkpoints (e.g. Qwen3.5/3.6 MoE) nest the text model under
    ``model.language_model.``. Probed against the weight map so new nestings fail loudly.
    """
    for pre in (
        f"model.layers.{layer_idx}.mlp.experts.",
        f"model.language_model.layers.{layer_idx}.mlp.experts.",
    ):
        if any(
            k.startswith(pre) for k in (wmap.keys() if hasattr(wmap, "keys") else wmap)
        ):
            return pre
    raise RuntimeError(
        f"[paged-experts] no expert tensors found for layer {layer_idx} under known prefixes "
        "(model.layers. / model.language_model.layers.) — unsupported checkpoint layout."
    )


def _fill_gptq_marlin_from_checkpoint(
    store: ExpertStore, model_path: str, layer_idx: int
) -> None:
    """gptq-int4: repack the GPTQ checkpoint into the on-GPU marlin layout for ALL E experts, using
    sglang's own ops, straight into the host store. sglang's loader repacks only the K resident slots
    (num_local_experts=K); we repack all E so the paged experts match. This is the per-layer repack the
    offline builder did, moved to load time -> no offline store artifact needed. (At runtime the
    quantization package is already imported, so the gptq_kernels/wNa16 circular import doesn't apply.)
    """
    from safetensors import safe_open

    # Load the quantization package fully before importing gptq_kernels directly — gptq_kernels and
    # compressed_tensors_wNa16_moe form an import cycle that only fails when gptq_kernels is the entry
    # point. At server runtime it is already imported; this makes the order-independent too.
    import sglang.srt.layers.quantization  # noqa: F401
    from sglang.srt.hardware_backend.gpu.quantization.gptq_kernels import (
        gptq_marlin_moe_repack,
    )
    from sglang.srt.layers.quantization.marlin_utils import marlin_moe_permute_scales

    snap = _snapshot_dir(model_path)
    cfg = json.load(open(os.path.join(snap, "config.json")))
    tcfg = cfg.get("text_config", cfg)
    inter = tcfg["moe_intermediate_size"]
    qc = cfg["quantization_config"]
    bits, group = qc["bits"], qc["group_size"]
    pack = 32 // bits
    assert not qc.get(
        "desc_act", False
    ), "desc_act=True needs g_idx paging (unsupported)"
    wmap = _weight_map(snap)
    pre = _experts_prefix(wmap, layer_idx)
    dev = store.device

    from contextlib import ExitStack

    _shard_stack = ExitStack()
    open_shards: Dict[str, object] = {}

    def get(name: str) -> torch.Tensor:
        sh = wmap[name]
        if sh not in open_shards:
            open_shards[sh] = _shard_stack.enter_context(
                safe_open(os.path.join(snap, sh), framework="pt")
            )
        return open_shards[sh].get_tensor(name)

    w13_qw, w2_qw, w13_s, w2_s, w13_qz, w2_qz = [], [], [], [], [], []
    for e in range(store.E):
        p = f"{pre}{e}."
        w13_qw.append(
            torch.cat([get(p + "gate_proj.qweight"), get(p + "up_proj.qweight")], dim=1)
        )
        w2_qw.append(get(p + "down_proj.qweight"))
        w13_s.append(
            torch.cat([get(p + "gate_proj.scales"), get(p + "up_proj.scales")], dim=1)
        )
        w2_s.append(get(p + "down_proj.scales"))
        w13_qz.append(
            torch.cat([get(p + "gate_proj.qzeros"), get(p + "up_proj.qzeros")], dim=1)
        )
        w2_qz.append(get(p + "down_proj.qzeros"))
    _shard_stack.close()  # release shard handles before the (GPU) repack
    w13_qw, w2_qw = torch.stack(w13_qw).to(dev), torch.stack(w2_qw).to(dev)
    w13_s, w2_s = torch.stack(w13_s).to(dev), torch.stack(w2_s).to(dev)
    sort = torch.empty((store.E, 0), dtype=torch.int32, device=dev)
    marlin = {
        "w13_qweight": gptq_marlin_moe_repack(
            w13_qw, sort, w13_qw.shape[1] * pack, w13_qw.shape[2], bits
        ),
        "w2_qweight": gptq_marlin_moe_repack(
            w2_qw, sort, w2_qw.shape[1] * pack, w2_qw.shape[2], bits
        ),
        "w13_scales": marlin_moe_permute_scales(
            s=w13_s, size_k=inter, size_n=w13_s.shape[2], group_size=group
        ),
        "w2_scales": marlin_moe_permute_scales(
            s=w2_s, size_k=w2_s.shape[1] * group, size_n=w2_s.shape[2], group_size=group
        ),
        "w13_qzeros": torch.stack(w13_qz),  # carried unrepacked (sym); kernel ignores
        "w2_qzeros": torch.stack(w2_qz),
    }
    for name in store.gpu:
        t = marlin[name].contiguous().cpu()
        expected = (store.E, *store.gpu[name].shape[1:])
        assert tuple(t.shape) == expected, (name, t.shape, expected)
        store.fill_tensor(name, t)


def _fill_bf16_from_checkpoint(
    store: ExpertStore, model_path: str, layer_idx: int
) -> None:
    """bf16: host w13_weight=[E,2*inter,hidden]=concat(gate,up), w2_weight=[E,hidden,inter]."""
    from safetensors import safe_open

    snap = _snapshot_dir(model_path)
    wmap = _weight_map(snap)
    pre = _experts_prefix(wmap, layer_idx)
    dt = store.gpu["w13_weight"].dtype
    by_shard: Dict[str, list] = {}
    for e in range(store.E):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            by_shard.setdefault(wmap[f"{pre}{e}.{proj}.weight"], []).append((e, proj))
    for shard, items in by_shard.items():
        with safe_open(os.path.join(snap, shard), framework="pt") as f:
            for e, proj in items:
                t = f.get_tensor(f"{pre}{e}.{proj}.weight").to(dt)
                if proj == "down_proj":
                    store.row("w2_weight", e).copy_(t)
                    continue
                # w13 packs gate (first half of dim 0) then up (second half)
                row = store.row("w13_weight", e)
                half = row.shape[0] // 2
                if proj == "gate_proj":
                    row[:half].copy_(t)
                else:  # up_proj
                    row[half:].copy_(t)


def _fill_fp8_block_from_checkpoint(
    store: ExpertStore, model_path: str, layer_idx: int
) -> None:
    """fp8 block-quant: direct copy, like bf16 but with the per-block scales as paged tensors too.
    Host ``w13_weight=[E,2*inter,hidden]`` e4m3 (concat gate,up), ``w2_weight=[E,hidden,inter]``;
    ``w13_weight_scale_inv``/``w2_weight_scale_inv`` are the [E, ceil(rows/block), ceil(cols/block)]
    float32 block scales, concatenated the same way. The CUDA triton path applies no post-load
    transform (no repack); layouts that DO transform (deepgemm ue8m0, mxfp8) are rejected by the
    dtype assertions below.
    """
    from safetensors import safe_open

    assert store.gpu["w13_weight"].dtype == torch.float8_e4m3fn, (
        "fp8 fill expects e4m3fn weights",
        store.gpu["w13_weight"].dtype,
    )
    assert store.gpu["w13_weight_scale_inv"].dtype == torch.float32, (
        "fp8 fill expects float32 block scales (ue8m0/mxfp8 layouts unsupported)",
        store.gpu["w13_weight_scale_inv"].dtype,
    )
    snap = _snapshot_dir(model_path)
    wmap = _weight_map(snap)
    pre = _experts_prefix(wmap, layer_idx)
    by_shard: Dict[str, list] = {}
    for e in range(store.E):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for suffix in ("weight", "weight_scale_inv"):
                by_shard.setdefault(wmap[f"{pre}{e}.{proj}.{suffix}"], []).append(
                    (e, proj, suffix)
                )
    for shard, items in by_shard.items():
        with safe_open(os.path.join(snap, shard), framework="pt") as f:
            for e, proj, suffix in items:
                t = f.get_tensor(f"{pre}{e}.{proj}.{suffix}")
                base = "w2_weight" if proj == "down_proj" else "w13_weight"
                name = base if suffix == "weight" else base + "_scale_inv"
                row = store.row(name, e)
                if proj == "down_proj":
                    row.copy_(t)
                    continue
                # w13 packs gate (first half of dim 0) then up (second half); the block scales
                # follow the same row split, so the same halving works for both suffixes.
                half = row.shape[0] // 2
                if proj == "gate_proj":
                    row[:half].copy_(t)
                else:  # up_proj
                    row[half:].copy_(t)


def setup_pager(method, layer) -> ExpertPager:
    """Build the host store and fill it from the checkpoint (all E experts), then return the pager wrapping
    it. ``method`` carries E, K, and the resident map. gptq-int4 is repacked to marlin at load time; bf16 is
    copied directly. No offline artifact."""
    from sglang.srt.server_args import get_global_server_args

    dev = next(layer.parameters()).device
    store = make_expert_store(
        layer,
        method.E,
        method.num_resident,
        dev,
        pin_host=getattr(method, "pin_host", True),
        window_W=getattr(method, "window", 0),
        cold_backing=getattr(method, "cold_backing", "ram"),
        cold_dir=getattr(method, "cold_dir", None),
    )

    layer_idx = getattr(layer, "layer_id", getattr(layer, "layer_idx", 0))
    model_path = get_global_server_args().model_path
    try:
        if any(n.endswith("qweight") for n in store.gpu):  # gptq-marlin int4
            _fill_gptq_marlin_from_checkpoint(store, model_path, layer_idx)
        elif (
            "w13_weight_scale_inv" in store.gpu
        ):  # fp8 block-quant (weights + block scales)
            _fill_fp8_block_from_checkpoint(store, model_path, layer_idx)
        elif "w13_weight" in store.gpu:  # bf16
            _fill_bf16_from_checkpoint(store, model_path, layer_idx)
        else:
            raise RuntimeError(
                f"[paged-experts] no fill for params {list(store.gpu)} "
                "(supported: gptq-marlin int4, fp8 block-quant, unquantized bf16)"
            )
    except KeyError as e:
        raise RuntimeError(
            f"[paged-experts] checkpoint tensor {e} not found for layer {layer_idx}: the fill expects "
            "per-expert {gate,up,down}_proj tensors; fused-expert checkpoint layouts (e.g. a packed "
            "experts.gate_up_proj) are unsupported."
        ) from e
    logger.debug(
        "[paged-experts] L%d host store filled: E=%d, %d tensors %s",
        layer_idx,
        store.E,
        len(store.gpu),
        list(store.gpu),
    )
    pager = ExpertPager(store=store, eviction=getattr(method, "eviction", "lru"))
    # Layer-ordered registry (setup runs per layer in model order): the wave path uses it to prefetch
    # the NEXT layer's cold tier while the current layer transfers/computes. A weight reload
    # (update_weights_from_disk) re-runs setup on the same method — REPLACE the old pager in place so
    # the registry doesn't grow a stale duplicate holding a second full host store.
    old_pager = getattr(method, "_pager", None)
    if old_pager is not None and 0 <= getattr(old_pager, "_layer_ord", -1) < len(
        _ALL_PAGERS
    ):
        pager._layer_ord = old_pager._layer_ord
        _ALL_PAGERS[pager._layer_ord] = pager
    else:
        pager._layer_ord = len(_ALL_PAGERS)
        _ALL_PAGERS.append(pager)
    return pager


def next_layer_pager(pager) -> Optional[ExpertPager]:
    """The pager of the next MoE layer in model order (None at the last layer / unknown order)."""
    i = getattr(pager, "_layer_ord", -1)
    if 0 <= i < len(_ALL_PAGERS) - 1:
        return _ALL_PAGERS[i + 1]
    return None
