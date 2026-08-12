# Removing the staging memcpy from the unified L3 KV layout adapter

Branch: `unified_kv_l3` @ `037d1dc4c6`. Every number below was measured in the
`zsk-sglang` dev container against the branch's own pool code — span matrices and
conversion rates on CPU, transfer rates on an H200, transport costs against a
live `mooncake_master`.

---

## Answers up front

**Q: "Can we only support `page_first` and `page_first_direct`?"**
Half right, and the useful half is not the one you'd expect. Restricting the
layout set changes nothing on its own: under the byte order the PR ships,
`layer_first`, `page_first`, `page_first_direct` and `page_head` all fragment an
MHA chunk into **256-byte pieces** (one `head_dim` vector each). All four rows of
the table are identical. But restricting to **`page_first_direct` alone, *and*
flipping the chunk byte order to match it**, is exactly the fix — see below.

**Q: "Do we have a better way?"**
Yes: **change the unified chunk order from `(head, layer, token, dim)` to
`(layer, token, head, dim)`.** That is `page_first_direct`'s own page-block
order, so every layer-range chunk becomes **one contiguous span** — including
DeepSeek-V3's short tail chunk at prime `L=61`. No new host layout, no new CUDA,
no padding, no keyspace flip beyond the format version.

**Q: "Can we save the copy when reusing KV across parallel configs?"**
For **cross-PP (layer fan-out): completely** — zero copy, one descriptor. For
**MLA at any TP: completely**, because rank-replicated KV has no head axis. For
**cross-TP GQA (head fan-out): no**, and that is a real limit, not an oversight —
a linear layout makes exactly one axis outermost, and buying the head axis costs
more elsewhere than it saves (§4).

---

## 1. The two candidate byte orders, measured

Contiguous spans per chunk. `1 x N` = one range of `N` bytes = zero copy.
MHA at L=61, 8 local kv heads, D=128, P=64, bf16, per K half.

**Order A — `(head, layer, token, dim)`, what the branch ships:**

| host layout | all-L x all-H | L[0:8] x all-H | L[56:61] tail | all-L x H[0:2] |
|---|---|---|---|---|
| `layer_first` | 31232 x 256 B | 4096 x 256 B | 2560 x 256 B | 7808 x 256 B |
| `page_first` | 31232 x 256 B | 4096 x 256 B | 2560 x 256 B | 7808 x 256 B |
| `page_first_direct` | 31232 x 256 B | 4096 x 256 B | 2560 x 256 B | 7808 x 256 B |
| `page_head` | 31232 x 256 B | 4096 x 256 B | 2560 x 256 B | 7808 x 256 B |

**Order B — `(layer, token, head, dim)`:**

| host layout | all-L x all-H | L[0:8] x all-H | L[56:61] tail | all-L x H[0:2] |
|---|---|---|---|---|
| `layer_first` | 61 x 128 KB | 8 x 128 KB | 5 x 128 KB | 3904 x 512 B |
| `page_first` | 3904 x 2 KB | 512 x 2 KB | 320 x 2 KB | 3904 x 512 B |
| **`page_first_direct`** | **1 x 7.6 MB** | **1 x 1.0 MB** | **1 x 640 KB** | 3904 x 512 B |
| `page_head` | 31232 x 256 B | 4096 x 256 B | 2560 x 256 B | 7808 x 256 B |

MLA has no head axis, so both orders are `(layer, token, dim)` and
`page_first_direct` is already 1 span for any layer range — which is why MLA is
the one case the PR already gets right. Order B generalizes that from a lucky
coincidence to the rule.

What the conversion costs today, 671 MB batch, 112 threads (`--full` mode):

| host layout | fan-out | staged | gather GB/s | scatter GB/s |
|---|---|---|---|---|
| `page_first` | none | 100 % | 52.7 | 51.9 |
| `page_first` | layer+head | 100 % | **13.2** | **12.9** |
| `page_first_direct` | none | 100 % | 54.9 | 62.3 |
| `page_first_direct` | layer+head | 100 % | **13.9** | **13.5** |

At 13 GB/s the conversion is slower than the fabric it feeds, and it does not
overlap: gather and put run on one thread against one staging buffer reused from
offset 0 (`hicache_key_scheme.py:560-571`).

## 2. Why order B and not a new host layout

Order A can be made zero-copy, but only by adding the host layout it was designed
for — `page_head_layer_direct` = `(2, page_num, head, layer, page, dim)`, which
the PR itself names as the target (`PR_unified_kv_l3.md:107`). Measured, that
layout class is expensive where it hurts most. `page_head` is the shipped
head-outermost layout and the closest analogue; both put `head` outside the token
axis, which is what sets the transfer item size:

| host layout | D2H all-layer | H2D per-layer |
|---|---|---|
| `page_first` (token-major) | 48.4 GB/s | 50.4 GB/s |
| `page_head` (head-major) | 30.6 GB/s (**1.6x slower**) | 11.3 GB/s (**4.5x slower**) |

H2D per-layer is the L2-hit critical path — `cache_controller.py` drives it layer
by layer under a `layer_done_counter`. A head-major host pool taxes **every L2
hit**, which is far more frequent than an L3 hit, to make **some L3 chunks**
free. That trade does not pay unless cross-TP GQA reuse is the dominant workload.

Order B takes the opposite side and needs nothing new: `page_first_direct` is
already a supported layout, is already what `_resolve_storage_layout_compatibility`
steers mooncake deployments to (`server_args.py:7313-7331`), and already has its
transfer kernels.

## 3. The transport side: spans, and when they pay

The pool can hand the transport a list of `(ptr, size)` spans instead of one
staged buffer; mooncake concatenates them. This is **already wired end to end** —
`_uses_multi_buffer` routes list-valued pointers to
`batch_put_from_multi_buffers` / `batch_get_into_multi_buffers`
(`mooncake_store.py:966,1406,1421`), the legacy MLA `layer_first` path already
uses it, and the whole host pool is already registered
(`mooncake_store.py:697`), so span pointers are directly DMA-able. The only
missing piece is a pool method that emits spans.

But spans are not free. Measured against a live `mooncake_master`, per-span cost
is ~0.7-0.8 µs, which sets a break-even span size of `c x B_stage` ≈ **1.8 KB**
for MHA and ≈ **9.6 KB** for MLA:

| case | span size | spans vs staging |
|---|---|---|
| MHA any shipped layout, order A | 256 B | **5.4x worse** |
| MLA `page_first` | 1152 B | **3.7x worse** |
| MLA `layer_first` | 73.7 KB | **1.3x better** |

So the rule is span size, not layout: emit spans when the minimum span clears
~16 KB, stage otherwise. Under order B that sends `page_first_direct` (1 span)
and `layer_first` (128 KB spans) down the copy-free path and leaves everything
else exactly as it is today.

## 4. What order B costs

Head fan-out. Under order B a head-subgroup chunk is `lg x P` spans of `hg x D`
(512 B at `hg=2`) on every layout — below the break-even, so cross-TP GQA reuse
keeps today's staged conversion. Concretely:

| transition | order B result |
|---|---|
| MLA any TP (rank-replicated) | zero copy, 1 descriptor |
| PP2 <-> PP1, any layer partition, incl. prime L | zero copy, 1 descriptor |
| GQA TP8 <-> TP4 (head fan-out) | staged, as today |
| GQA TP4/PP2 -> TP2/PP1 | staged, as today |

For the models where L3 KV reuse actually matters at scale — DeepSeek/Kimi-class
MLA — there is no trade at all: no head axis exists, so order B is strictly
better. For GQA it buys cross-PP and leaves cross-TP where it is.

### 4.1 How much the head split actually costs

Measured on `page_first_direct` under order B, GQA L=80, 4 local kv heads,
D=128, P=64, 671 MB batch. `e2e@N` is save throughput with an N GB/s fabric in
series with the conversion, `1/(1/gather + 1/N)`:

| layer grid | head_group | staged | gather GB/s | scatter GB/s | e2e@50 | e2e@200 |
|---|---|---|---|---|---|---|
| none | 4 (no split) | **0 %** | — | — | **50.0** | **200.0** |
| none | 2 | 100 % | 35.9 | 35.8 | 20.9 (**-58 %**) | 30.4 (**-85 %**) |
| none | 1 | 100 % | 20.5 | 20.0 | 14.5 (**-71 %**) | 18.6 (**-91 %**) |
| lg=8 | 4 (no split) | **0 %** | — | — | **50.0** | **200.0** |
| lg=8 | 2 | 100 % | 4.6 | 4.6 | 4.2 (**-92 %**) | 4.5 (**-98 %**) |
| lg=8 | 1 | 100 % | 2.3 | 2.2 | 2.2 (**-96 %**) | 2.2 (**-99 %**) |

Three things to read off it:

1. **Not splitting heads is free** — zero bytes copied, with or without a layer
   split, so the fabric is the only limit.
2. **The coarsest legal `head_group` is worth ~1.75x.** `hg=2` gathers at
   35.9 GB/s vs 20.5 at `hg=1`. Set `head_group` to the *gcd of local kv-head
   counts across the fleet*, not to 1: a TP2+TP4 fleet on 8 kv heads needs
   `gcd(4,2) = 2`; only adding TP8 forces 1.
3. **Stacking a layer split on top of a head split costs another ~7.8x** —
   35.9 -> 4.6 GB/s at `hg=2`. This corrects a natural assumption: the span
   *granularity* is unchanged (512 B either way), but the conversion is one
   `copy_` per (page, layer-chunk, head-chunk, K/V), so a 10-way layer split
   makes ten times as many slabs, each a tenth the size. Below roughly 512 KB
   per slab the copy stops amortizing thread-pool dispatch and falls back to
   about single-core speed (~4 GB/s measured with 1 thread, flat across
   configs). Batching the gather into one `index_select` per slab type over all
   pages recovers only ~1.2x, so this is not pure dispatch overhead.

### 4.2 The head split need not cost a copy at all: fold the offset into the H2D/D2H

The staging memcpy exists only because the conversion is being done by the CPU as
a *separate* step. But a copy across PCIe already happens on both paths — L2
write-back on save, L2->L1 load on read — and those transfers are DMA-bound, not
compute-bound. They already compute a per-(token, layer, head) address; making
them compute the head-partitioned unified-order address instead is free. The
CPU copy then disappears entirely rather than being made faster.

Measured on an H200, 671 MB, L=80, 4 local kv heads, D=128, P=64, `head_group=2`
(`benchmark/hicache/bench_unified_l3_gpu_permute.py`):

| stage | ms | GB/s |
|---|---|---|
| contiguous D2H (no-split ceiling) | 12.16 | 55.2 |
| contiguous H2D (no-split ceiling) | 12.05 | 55.7 |
| CPU gather, host pool -> staging (today) | 20.76 | 32.3 |
| CPU scatter, staging -> host pool (today) | 18.97 | 35.4 |
| **D2H device pool -> staging, offsets folded in** | **14.24** | **47.1** |
| **H2D staging -> device pool, offsets folded in** | **12.81** | **52.4** |

End to end for head-partitioned chunks:

| path | ms | vs today |
|---|---|---|
| save today: D2H to L2 + CPU gather | 32.92 | 1.00x |
| save: D2H to L2 + D2H to staging | 26.40 | **1.25x** |
| save, L3-only (no L2 copy) | 14.24 | **2.31x** |
| load today: CPU scatter + H2D | 31.02 | 1.00x |
| **load: staging -> device pool, one transfer** | **12.81** | **2.42x** |

**Additional copies: 1 today, 0 with this.** The head partition costs only the
gap against the contiguous ceiling — **6 % on the read path** (52.4 vs 55.7) and
**15 % on the write path** (47.1 vs 55.2) — on a transfer that has to happen
regardless. Compare that to §4.1, where the same split costs 100 % staged at
32-35 GB/s and drives end-to-end save down 58-71 %.

The RDMA still lands in a contiguous pinned buffer rather than scattering
straight into the pool: a head-partitioned chunk is `lg x P` spans of `hg x D`
(512 B at `hg=2`), so a GPUDirect scatter would reintroduce exactly the
tiny-descriptor problem of §3. One contiguous RDMA plus one offset-computing
H2D is the efficient decomposition.

**This is already expressible in the existing kernels.**
`get_global_offset_ph` (`kernels/aot/csrc/kvcacheio/transfer.cu:104-119`) already
computes a `(page, head, token, layer)` address from scalar parameters, and
`transfer_page_head_kernel_impl` (`:123`) is templated on
`<SrcOffsetFn, DstOffsetFn>`. A unified-order head-partitioned functor is one
more `__device__` function plus a template instantiation, registered the same
way as the other 13 wrappers. The GPU-side-gather-then-contiguous-DMA shape also
already exists as the staged write-back path
(`_init_write_back_staging_buffers`, `staging_k_buffer`,
`jit_transfer_hicache_all_layer_staged_lf_pf`).

**What it costs to adopt — scheduling, not kernels.** The blocker is where the
data lives when the L3 transfer happens:

- *Save*: L3 backup runs after the D2H ack, from `node.host_value` (host
  indices) — the device copy is already released (`hiradix_cache.py:840-937`).
  A device-sourced L3 write needs the L3 admission decision fused into
  write-back so both transfers are issued while the page is still on the GPU.
  Without that fusion the source must be L2, and reading L2 -> GPU -> staging is
  two PCIe passes (~26 ms), worse than the CPU gather.
- *Load*: prefetch is asynchronous into `host_indices`; device indices do not
  exist yet. Landing L3 straight in the device pool means either doing the L3
  fetch at load time (putting the L3 round trip on the critical path) or keeping
  the async prefetch but landing it in a pinned unified-order area and deferring
  the offset-computing H2D to schedule time. The second keeps today's latency
  profile and is the one to build.

The read path is both the bigger win (2.42x) and the easier change, so it is the
one to do first.

Practical rule: **if a fleet needs cross-TP GQA reuse, do not also set a fine
`layer_partition`** — take cross-PP reuse from a coarse partition or not at all.
And if cross-TP GQA is the fleet's primary axis, the `page_head` + order-A route
is the better home for it (§2): there cross-TP is the free axis and cross-PP is
the one that pays.

## 5. Rejected: tiling the host pool to the fleet grid

Allocating `(2, pn, H/hg, ceil(L/lg), hg, lg, P, D)` makes every chunk one range,
and it does work (`nspans == 1` for every grid tried). It fails on the model that
motivates the feature: the chunk name encodes its own extent
(`hicache_key_scheme.py:277` emits `L{a}-{min(a+lg, end)}`), so a padded tile
cannot match a peer's object, and **L = 61 is prime** — exact tiling admits only
`lg in {1, 61}`, i.e. no layer fan-out or 16 KB objects at 61x the key count. It
also bakes the grid into pool allocation, turning a grid change into a restart
plus a re-registration of the whole multi-hundred-GB buffer. Order B keeps the
layer grid renegotiable at runtime (`cache_controller.py:431-475`).

## 6. Recommendation

1. **Flip the unified chunk order to `(layer, token, head, dim)`** and bump the
   namespace schema version. Objects stay one-chunk-per-key and byte-identical
   across fleets; only the intra-chunk order changes.
2. **Require `page_first_direct` for adapter-mode fleets** (fail fast otherwise,
   or fall back to staging). This is your instinct, applied to the one layout
   that matches the new order.
3. **Add span emission with a `min_span >= 16 KB` gate**, so `layer_first` also
   goes copy-free and everything else keeps today's path. Cache the span template
   per grid — offsets are page-index-independent, and the naive per-chunk walk is
   ~600x slower.
4. **Do not add `page_head_layer_direct`** until cross-TP GQA reuse is shown to
   be hot enough to justify 1.6x D2H / 4.5x H2D on every L1<->L2 move.

## 7. Other findings on the branch

- **`gather` runs before the existence check.** `_batch_set_adapter` gathers the
  whole sub-batch (`mooncake_store.py:1064`) then calls `_batch_exist` (`:1067`),
  so pages already in L3 pay the full conversion memcpy for nothing. On a warm
  cache that is most pages. Worth reordering regardless of everything above.
- **`_uses_multi_buffer` inspects only `buffer_ptrs[0]`** (`:966`). A hybrid
  batch — some slabs spanned, some staged — would mis-dispatch. Normalize to
  1-element lists before enabling spans.
- **The zero-copy claim is understated.** `unified_zero_copy` also returns True
  for MLA + `layer_first` + `layer_partition=1`, and for MHA +
  `page_first_direct` + `local_kv_heads == 1` at any layer partition — verified
  against the branch's own `_slab_schedule`. The PR names only MLA +
  `page_first_direct`.
- **Adapter metrics are not comparable to the legacy path**: the adapter's timing
  window spans `gather + exists + put` (`:1060`, `:1098-1102`) while the legacy
  window covers only the store call.

## 8. Benchmark

`test/registered/unit/mem_cache/test_hicache_unified_layout_perf.py` — CPU-only,
11 tests, ~14 s, `base-a-test-cpu`. It prices save (`gather_unified_chunks`) and
load (`scatter_unified_chunks`) per `layout x fan-out`, drives the real
`MooncakeStore` adapter path against a loopback transport for an end-to-end
save/load number, prints the order-A-vs-order-B descriptor matrix that decides
this design, and proves the span path executably —
`test_spans_reproduce_the_staged_bytes_without_copying` asserts that reading a
chunk's spans straight from the pool reproduces the staging buffer byte for byte
on every layout.

```bash
python3 test/registered/unit/mem_cache/test_hicache_unified_layout_perf.py --full
```
