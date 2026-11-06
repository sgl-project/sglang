"""Compare what PD peers and HiCache see of a unified pool between checkouts.

PD peers exchange the buffers they register and the order and shape of their
state entries; HiCache names its host pools, and so the suffixes of its L3
keys, by pool name. Both are wire formats, so a change that must not alter
them is checked by recording them on the base checkout and comparing on the
candidate:

    # on the base checkout
    python test/manual/mem_cache/unified_transfer_compare.py record --out base.json
    # on the candidate checkout
    python test/manual/mem_cache/unified_transfer_compare.py compare --base base.json

Every pool chain is built from the unified pool fixtures on the CPU; HiCache
host pools register host memory with CUDA, so a CUDA device must be visible.
Pointers are recorded relative to the unified buffer, eviction callbacks by
what they ask of a recording cache, allocation callbacks by their names.

Pools built without unified memory register with PD peers too (configurations
named ``static_*``): they are built on the CUDA device, and their pointers are
recorded relative to the lowest one they register.
"""

import argparse
import dataclasses
import inspect
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.unified_allocator_fixtures import (
    UnifiedGeometry,
    build_unified_bundle,
    setup_allocator_context,
)

_P16 = dict(page_size=16, full_tokens=64 * 16, swa_tokens=64 * 16)
GEOMETRIES = {
    "swa_p1": UnifiedGeometry(chain="swa", full_layer_ids=(0, 1), swa_layer_ids=(2, 3)),
    "swa_p16": UnifiedGeometry(
        chain="swa", full_layer_ids=(0, 1), swa_layer_ids=(2, 3), **_P16
    ),
    "swa_p16_asym": UnifiedGeometry(
        chain="swa",
        full_layer_ids=(0, 1, 2),
        swa_layer_ids=(3, 4),
        v_head_dim=16,
        swa_head_num=2,
        **_P16,
    ),
    "mamba_p1": UnifiedGeometry(
        chain="mamba", full_layer_ids=(0, 1), state_layer_ids=(2, 3)
    ),
    "mamba_p16": UnifiedGeometry(
        chain="mamba", full_layer_ids=(0, 1), state_layer_ids=(2, 3), **_P16
    ),
    "mamba_mla_p16": UnifiedGeometry(
        chain="mamba", full_layer_ids=(0, 1), state_layer_ids=(2, 3), mla=True, **_P16
    ),
    "tri_p1": UnifiedGeometry(
        chain="tri", full_layer_ids=(0, 1), swa_layer_ids=(2, 3), state_layer_ids=(0, 2)
    ),
    "tri_p16": UnifiedGeometry(
        chain="tri",
        full_layer_ids=(0, 1),
        swa_layer_ids=(2, 3),
        state_layer_ids=(0, 2),
        **_P16,
    ),
}
_COMPONENTS = {
    "swa": {ComponentType.FULL, ComponentType.SWA},
    "mamba": {ComponentType.FULL, ComponentType.MAMBA},
    "tri": {ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA},
}
# Pools built without unified memory, registered with PD peers.
STATIC_CONFIGS = (
    "static_mha",
    "static_mla",
    "static_swa",
    "static_hybrid",
    "static_hybrid_mla",
    "static_tri",
)
# (hicache_mem_layout, hicache_storage_backend): the page-envelope host needs
# no backend or one that keeps envelopes; "file" does not.
HICACHE_VARIANTS = (("page_first", None), ("page_first", "file"), ("layer_first", None))


def _state_pool(bundle):
    # The state pool moved from the request table to the allocator; a tool run
    # on both sides of that move reads whichever this checkout has.
    state = getattr(bundle.token_to_kv_pool_allocator, "state", None)
    if state is not None:
        return state.pool
    table = getattr(bundle, "req_to_token_pool", None)
    return getattr(table, "mamba_pool", None)


def _buffer_base(bundle) -> int:
    # The unified buffer moved from the pool bundle to the arena; a tool run on
    # both sides of that move reads whichever this checkout has.
    arena = getattr(bundle, "arena", None)
    if arena is not None:
        return arena.raw.data_ptr()
    return bundle.unified_memory_pool._raw.data_ptr()


def _rel(ptrs, base):
    return [p - base for p in ptrs]


def pd_summary(bundle, total_layers: int) -> dict:
    from sglang.srt.disaggregation.base.conn import KVArgs

    pool = bundle.token_to_kv_pool
    allocator = bundle.token_to_kv_pool_allocator
    base = _buffer_base(bundle)
    storage = getattr(allocator, "storage", None)
    kv_args = KVArgs()
    kv, layer_ids = _register(
        kv_args,
        pool,
        storage,
        total_layers,
        state_pool=_state_pool(bundle),
        req_to_token_pool=getattr(bundle, "req_to_token_pool", None),
    )
    return _pd_record(kv_args, kv, layer_ids, base)


def _register(kv_args, pool, storage, total_layers, *, state_pool, req_to_token_pool):
    """Register ``pool`` as this checkout's PD code does: through the storage
    where registration takes one (``storage`` is None for a static pool), else
    through the pool and its keywords. Returns the KV entries and layer ids."""
    from sglang.srt.disaggregation import utils as pd

    setup = inspect.signature(pd.setup_state_kv_args).parameters
    if list(setup)[1] == "storage":
        if storage is None:
            from sglang.srt.mem_cache.static_kv_storage import StaticKVStorage

            allocator = SimpleNamespace(
                storage=None,
                state=None if state_pool is None else SimpleNamespace(pool=state_pool),
                get_kvcache=lambda: pool,
            )
            storage = StaticKVStorage(None, allocator, num_hidden_layers=total_layers)
        kv = pd.get_kv_transfer_buf_infos(storage)
        pd.setup_state_kv_args(kv_args, storage, None, total_layers)
        layer_id_kwargs = {"storage": storage}
    else:
        kwargs = {}
        if "storage" in setup:
            kwargs["storage"] = storage
        if "state_pool" in setup:
            kwargs["state_pool"] = state_pool
        else:
            kwargs["req_to_token_pool"] = req_to_token_pool
        if "storage" in inspect.signature(pd.get_kv_transfer_buf_infos).parameters:
            kv = pd.get_kv_transfer_buf_infos(pool, storage)
        else:
            kv = pd.get_kv_transfer_buf_infos(pool)
        pd.setup_state_kv_args(kv_args, pool, None, total_layers, **kwargs)
        layer_id_kwargs = {}
        if "storage" in inspect.signature(pd.build_kv_layer_ids).parameters:
            layer_id_kwargs["storage"] = storage
    layer_ids = pd.build_kv_layer_ids(
        token_to_kv_pool=pool,
        draft_token_to_kv_pool=None,
        num_draft_entries=0,
        num_hidden_layers=total_layers,
        **layer_id_kwargs,
    )
    return kv, layer_ids


def _pd_record(kv_args, kv, layer_ids, base) -> dict:
    return {
        "kv": {"ptrs": _rel(kv[0], base), "lens": kv[1], "item_lens": kv[2]},
        "kv_layer_ids": layer_ids,
        "state_types": [t.value for t in kv_args.state_types],
        "state_ptrs": [_rel(p, base) for p in kv_args.state_data_ptrs],
        "state_lens": kv_args.state_data_lens,
        "state_item_lens": kv_args.state_item_lens,
        "state_dim_per_tensor": kv_args.state_dim_per_tensor,
        "state_conv_shard_groups": kv_args.state_conv_shard_groups,
        "state_slice_outer_counts": kv_args.state_slice_outer_counts,
        "state_layer_ids": kv_args.state_layer_ids,
        "is_hybrid_mla_backend": kv_args.is_hybrid_mla_backend,
        "swa_kv_cache_layout": kv_args.swa_kv_cache_layout,
        "mla_compression_ratios": kv_args.mla_compression_ratios,
    }


def _static_mamba(layer_ids):
    import torch

    from sglang.srt.configs.inkling import (
        InklingConvCacheParams,
        InklingConvStateShape,
        InklingStateDType,
    )
    from sglang.srt.mem_cache.memory_pool import MambaPool

    return MambaPool(
        size=16,
        spec_state_size=4,
        cache_params=InklingConvCacheParams(
            shape=InklingConvStateShape(
                conv=[(3, 256), (3, 256), (3, 128), (3, 128), (3, 256), (3, 256)],
                temporal=(2, 8, 16),
            ),
            layers=layer_ids,
            dtype=InklingStateDType(conv=torch.bfloat16, temporal=torch.bfloat16),
        ),
        mamba_layer_ids=layer_ids,
        device="cuda",
    )


def build_static(name: str):
    """A static pool, and the recurrent state pool beside it when the KV pool
    does not hold that state itself."""
    import torch

    from sglang.srt.mem_cache.memory_pool import (
        HybridLinearKVPool,
        MHATokenToKVPool,
        MLATokenToKVPool,
    )
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

    common = dict(size=256, page_size=16, dtype=torch.bfloat16, device="cuda")
    if name == "static_mha":
        pool = MHATokenToKVPool(
            head_num=2, head_dim=64, layer_num=4, enable_memory_saver=False, **common
        )
        return pool, None
    if name == "static_mla":
        pool = MLATokenToKVPool(
            kv_lora_rank=64,
            qk_rope_head_dim=16,
            layer_num=4,
            enable_memory_saver=False,
            **common,
        )
        return pool, None
    if name in ("static_swa", "static_tri"):
        pool = SWAKVPool(
            size_swa=128,
            head_num=2,
            head_dim=64,
            full_attention_layer_ids=[0, 1],
            swa_attention_layer_ids=[2, 3],
            token_to_kv_pool_class=MHATokenToKVPool,
            swa_kv_pool_kwargs={
                "head_num": 1,
                "head_dim": 64,
                "page_size": 16,
                "device": "cuda",
                "enable_memory_saver": False,
            },
            **common,
        )
        return pool, _static_mamba([0, 2]) if name == "static_tri" else None
    mla = name == "static_hybrid_mla"
    pool = HybridLinearKVPool(
        head_num=2,
        head_dim=64,
        full_attention_layer_ids=[0, 2],
        mamba_pool=_static_mamba([1, 3]),
        use_mla=mla,
        kv_lora_rank=64 if mla else None,
        qk_rope_head_dim=16 if mla else None,
        **common,
    )
    return pool, None


def static_pd_summary(name: str) -> dict:
    from sglang.srt.disaggregation.base.conn import KVArgs

    pool, state_pool = build_static(name)
    # Before the state moved to the allocator, the request table described it.
    table = (
        None
        if state_pool is None
        else SimpleNamespace(
            get_state_buf_infos=state_pool.get_contiguous_buf_infos,
            get_state_dim_per_tensor=state_pool.get_state_dim_per_tensor,
            get_state_conv_shard_groups=state_pool.get_state_conv_shard_groups,
            get_state_slice_outer_counts=state_pool.get_state_slice_outer_counts,
        )
    )
    kv_args = KVArgs()
    kv, layer_ids = _register(
        kv_args, pool, None, 4, state_pool=state_pool, req_to_token_pool=table
    )
    base = min(list(kv[0]) + [p for ptrs in kv_args.state_data_ptrs for p in ptrs])
    return _pd_record(kv_args, kv, layer_ids, base)


class _RecordingCache:
    def __init__(self, bundle, page_size, components):
        self.token_to_kv_pool_allocator = bundle.token_to_kv_pool_allocator
        self.req_to_token_pool = getattr(bundle, "req_to_token_pool", None)
        self.page_size = page_size
        self.components = {c: None for c in components}
        self.calls = []

    def evict_host(self, n, component):
        self.calls.append(["evict_host", n, int(component)])

    def evict_for_alloc(self, params):
        self.calls.append(["evict_for_alloc", dataclasses.asdict(params)])


def _fn_name(fn):
    return None if fn is None else getattr(fn, "__qualname__", repr(fn))


def _calls(cache, fn, arg):
    if fn is None:
        return None
    cache.calls.clear()
    fn(arg)
    return list(cache.calls)


def _view_name(bundle, device_pool):
    """Named as main's record names them: the owner's view `full_kv_pool`, the
    window's `swa_kv_pool` (the attributes of main's pool classes)."""
    pool = bundle.token_to_kv_pool
    for name in ("full_kv_pool", "swa_kv_pool"):
        if getattr(pool, name, None) is device_pool:
            return name
    router = getattr(pool, "layer_router", None)
    for sub_pool, name in (("full", "full_kv_pool"), ("swa", "swa_kv_pool")):
        if router is not None and router.sub_pool(sub_pool) is not None:
            if router.view(sub_pool) is device_pool:
                return name
    if device_pool is _state_pool(bundle):
        return "state_pool"
    return type(device_pool).__name__


def _group_name(group, host_pool):
    for name, entry in group.entry_map.items():
        if entry.host_pool is host_pool:
            return name.value
    return None


def hicache_summary(bundle, chain: str, page_size: int, layout, backend) -> dict:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as hpa
    from sglang.srt.runtime_context import publish, reset_context
    from sglang.srt.server_args import ServerArgs

    reset_context()
    publish(
        ServerArgs(
            model_path="dummy",
            enable_unified_memory=True,
            hicache_mem_layout=layout,
            hicache_storage_backend=backend,
            hicache_ratio=2.0,
            hicache_size=0,
        ),
        role="tokenizer",
    )
    allocator = bundle.token_to_kv_pool_allocator
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=getattr(bundle, "req_to_token_pool", None),
        token_to_kv_pool_allocator=allocator,
        page_size=page_size,
        tp_cache_group="tp",
        attn_cp_cache_group="attn_cp",
        attn_tp_cache_group="attn_tp",
        pp_cache_group="pp",
    )
    cache = _RecordingCache(bundle, page_size, _COMPONENTS[chain])
    controller_args = {}

    def record_init(self, *args, **kwargs):
        controller_args["args"] = [
            a if isinstance(a, (int, str)) else None for a in args
        ]
        controller_args.update(
            {
                k: v if isinstance(v, (int, float, str, type(None))) else repr(type(v))
                for k, v in kwargs.items()
            }
        )
        self.transfer_layer_id_max = kwargs.get("transfer_layer_id_max")

    with patch.object(hpa.HybridCacheController, "__init__", record_init):
        storage = getattr(allocator, "storage", None)
        if storage is not None and hasattr(hpa, "build_unified_host_stack"):
            result = hpa.build_unified_host_stack(
                storage=storage,
                params=params,
                cache=cache,
                load_cache_event=None,
                storage_backend=backend,
            )
        else:
            kvcache = allocator.get_kvcache()
            strategy = hpa._select_strategy(kvcache, set(_COMPONENTS[chain]))
            result = strategy.build(
                cache=cache,
                kvcache=kvcache,
                params=params,
                server_args=None,
                load_cache_event=None,
                storage_backend=backend,
            )
    group = result.host_pool_group
    limit = controller_args["transfer_layer_id_max"] + 4
    entries = []
    for entry in group.entry_map.values():
        host = entry.host_pool
        entries.append(
            {
                "name": entry.name.value,
                "host_cls": type(host).__name__,
                "host_size": host.size,
                "host_page_size": host.page_size,
                "host_layout": host.layout,
                "host_label": getattr(host, "pool_label", None),
                "device_pool": _view_name(bundle, entry.device_pool),
                "layer_map": [entry.layer_mapper(t) for t in range(limit)],
                "anchor": entry.is_primary_index_anchor,
                "packed_drafts": len(entry.packed_draft_device_pools),
                "host_evict": _calls(cache, entry.host_evict_fn, 3),
                "device_evict": _calls(cache, entry.device_evict_fn, 1 << 30),
                "device_alloc": _fn_name(entry.device_alloc_fn),
                "device_free": _fn_name(entry.device_free_fn),
            }
        )
    return {
        "entries": entries,
        "components": sorted(
            [int(c), _group_name(group, pool)]
            for c, pool in result.component_host_pools.items()
        ),
        "sidecars": [
            [s.pool_name.value, s.indices_from_pool.value, str(s.hit_policy)]
            for s in result.sidecars
        ],
        # Results from checkouts older than this field record None.
        "loads_state_layer_wise": getattr(result, "loads_state_layer_wise", None),
        "controller": controller_args,
    }


def summarize_one(name: str, variant: int) -> dict:
    setup_allocator_context()
    if name in STATIC_CONFIGS:
        try:
            return static_pd_summary(name)
        except Exception as e:  # recorded: an error on both sides agrees
            return {"error": f"{type(e).__name__}: {e}"}
    geometry = GEOMETRIES[name]
    bundle = build_unified_bundle(geometry)
    if variant < 0:
        layers = (
            geometry.full_layer_ids + geometry.swa_layer_ids + geometry.state_layer_ids
        )
        return pd_summary(bundle, max(layers) + 1)
    layout, backend = HICACHE_VARIANTS[variant]
    try:
        return hicache_summary(
            bundle, geometry.chain, geometry.page_size, layout, backend
        )
    except Exception as e:  # recorded: an error on both sides agrees
        return {"error": f"{type(e).__name__}: {e}"}


def summarize() -> dict:
    # One process per configuration and HiCache variant: host pools register
    # device memory with CUDA, which a second stack over the same pool repeats.
    def run(name, variant):
        out = subprocess.run(
            [sys.executable, __file__, "one", name, str(variant)],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(out.stdout.splitlines()[-1])

    keys = {-1: "pd"}
    keys.update(
        (i, f"hicache:{layout}:{backend}")
        for i, (layout, backend) in enumerate(HICACHE_VARIANTS)
    )
    jobs = [(name, variant) for name in GEOMETRIES for variant in keys]
    jobs += [(name, -1) for name in STATIC_CONFIGS]
    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(lambda job: run(*job), jobs))
    out = {name: {} for name in (*GEOMETRIES, *STATIC_CONFIGS)}
    for (name, variant), record in zip(jobs, records):
        out[name][keys[variant]] = record
    return out


def _diff(a, b, path=""):
    if type(a) is not type(b):
        return [f"{path}: {a!r} != {b!r}"]
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'base' if k in a else 'candidate'}")
            else:
                out += _diff(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}: {a!r} vs {b!r}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += _diff(x, y, f"{path}[{i}]")
        return out
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    record = sub.add_parser("record")
    record.add_argument("--out", required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--base", required=True)
    compare.add_argument("--out", default=None)
    one = sub.add_parser("one")
    one.add_argument("name", choices=sorted((*GEOMETRIES, *STATIC_CONFIGS)))
    one.add_argument("variant", type=int, help="-1 for PD, else a HiCache variant")
    args = parser.parse_args()
    if args.cmd == "one":
        print(json.dumps(summarize_one(args.name, args.variant), default=str))
        return
    current = json.loads(json.dumps(summarize(), default=str))
    if args.cmd == "record":
        with open(args.out, "w") as f:
            json.dump(current, f, indent=1, sort_keys=True)
        print(f"recorded {len(current)} configurations to {args.out}")
        return
    if args.out:
        with open(args.out, "w") as f:
            json.dump(current, f, indent=1, sort_keys=True)
    with open(args.base) as f:
        base = json.load(f)
    diffs = _diff(base, current)
    for line in diffs:
        print(line)
    print(f"{len(current)} configurations, {len(diffs)} differences")
    sys.exit(1 if diffs else 0)


if __name__ == "__main__":
    main()
