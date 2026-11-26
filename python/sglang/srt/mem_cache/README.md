# `mem_cache/`

Everything that owns KV / SSM-state memory: who hands out slots, who holds the bytes on
the device, who mirrors them to host and disk, and which radix cache decides what
to keep. The layout is specified in
[#25371](https://github.com/sgl-project/sglang/issues/25371).

## Layers

```
        scheduler / model_runner / attention backend
                          |
                          v
        allocation.py                  per-batch allocation policy
                          |
                          v
        hybrid_cache/                  multi-pool router (layer_id -> pool)
                          |
                          v
        allocator/                     "give me N slots"  (need_size -> indices)
                          | holds a reference to
                          v
        pool/ (device, L1)  --hicache-->  pool_host/ (host, L2)  -->  storage/ (L3)
        (layer_id, indices)               device_indices <->
             <-> tensor                   host_indices
```

| Layer | Cares about | In -> Out |
|---|---|---|
| `allocation.py` | per-batch allocation policy | `batch` -> `out_cache_loc` |
| `hybrid_cache/` | per-layer routing across pools | `layer_id` -> pool |
| `allocator/` | which slots are free | `need_size` -> `indices` |
| `pool/` | physical KV / SSM state layout | `(layer_id, indices)` <-> tensor |
| `pool_host/` | host mirror + H2D/D2H | `device_indices` <-> `host_indices` |
| `storage/` | L3 backends (file, NIXL, HF3FS, Mooncake, ...) | hash -> bytes |
| radix cache | what to keep and what to evict | token prefix -> node |

Two groups sit outside that stack:

- **Radix cache** is its own axis. The per-model variants (`radix_cache.py`,
  `pure_swa_radix_cache.py`) are converging onto the **Unified Radix Cache** (`unified_cache/`,
  [#20415](https://github.com/sgl-project/sglang/issues/20415)), whose Full/SWA/Mamba
  component model is documented in
  [`unified_cache/components/README.md`](unified_cache/components/README.md).
- **Construction** cuts across every layer rather than sitting in it:
  `kv_cache_configurator.py`, `kv_cache_builder.py`, `cache_init_params.py`,
  `allocation_sizing.py`, `kv_cache_dtype.py`, `kv_vmm_backing.py`, and
  `hybrid_cache/hybrid_pool_assembler.py` decide the shapes and build the objects above.

## Unified memory (`kv/`)

With `--enable-unified-memory`, a hybrid model's full-attention KV, sliding-window KV
and recurrent state live in one byte arena instead of the pools above. `kv/` owns that
whole stack: declarations and plan (`decl.py`, `plan.py`, `sizing.py`), the arena and
layouts (`memory/`, `layout/`), one allocator over every sub-pool (`alloc/chain.py`,
`alloc/band.py`), the request table (`requests.py`), and `storage.py::KVStorage`, the
interface the scheduler, the prefix tree, HiCache and PD use.

Two rules hold anywhere its pages are touched:

- **Freeing is logical at once; physical reuse is not.** A freed page stays pending
  until the forward that may still touch it is processed, or a stream wait on that
  forward settles it (`alloc/forward_steps.py`), and
  every write into a page clears what the band's ledger records on it
  (`alloc/page_usage.py::CLEARANCE`).
- **Allocator decisions read logical state only** (counts, forwards processed,
  transfers released), never whether an event fired, so TP ranks agree. Under
  pipeline parallelism every stage asks each capacity question of every stage's
  geometry (`alloc/stage_capacity.py`), so the stages agree too.

Capacity decisions ask a concrete demand (`alloc/demand.py::KVDemand`, through
`ChainAllocator.admits`, `units_fit` or `capacity.py`): sub-pools share bytes, so a
scalar availability read is only an estimate for observers.

Design, debugging tools and known limitations:
[Unified Memory Design](https://docs.sglang.io/docs/advanced_features/unified_memory_design).

## Where does my class go?

By base class, never by name:

| Inherits from | Home |
|---|---|
| `BaseTokenToKVPoolAllocator` | `allocator/<family>.py` |
| `KVCache`, `BaseSWAKVPool`, `ReqToTokenPool`, `MambaPool` | `pool/<family>.py` |
| `HostKVCache` | `pool_host/<family>.py` |
| `HiCacheStorage` | `storage/<backend>/` |
| `BasePrefixCache` | one module at the `mem_cache/` root |

`<family>` is the attention or state family: `mha`, `mla`, `dsa`, `mamba`, `swa`,
`hisparse`, `deepseek_v4`. A new quantization or layout variant of an existing family is
a new file in that family's module, not a new class in a catch-all one.

`Allocator` means two different things and they do not share a directory:

- **slot allocator** -- a `BaseTokenToKVPoolAllocator` subclass, hands out KV slots,
  lives in `allocator/`.
- **host tensor allocator** -- `HostTensorAllocator` and its subclasses, hands out pinned
  host memory, lives in `pool_host/common.py` and `storage/`.

## Conventions

- **Names drop affixes that do not differentiate.** If every file in a directory shares
  the role the directory already names, the affix carries nothing: `pool_host/mha.py`,
  not `pool_host/mha_pool_host.py`. Keep a role affix only where same-directory siblings
  have different roles.
- **A family is a module; a module may be a package.** One file per family by default;
  past ~1500 lines the family becomes a package.
- **Layers do not import upwards.** `pool/` and `pool_host/` must not import
  `allocator/`, `hybrid_cache/`, or `allocation.py`; `allocator/` may hold the pool it
  allocates into, not the reverse; none of the three may import the construction layer.
