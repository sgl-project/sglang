# Dense LoRA plan tables (`--lora-backend triton_v2`)

`<arch>.plans.json` names, per device architecture, the kernels that serve a
dense LoRA site for one (phase, token count, pool rank, site geometry). Rows
are tried in order and the first match wins; a row without a bound on some
key matches every value of it. Without a table, or for a query no row
matches, the in-code defaults in `../plan.py` serve. `sm100` serves every
capability-10 device (B200, GB300), `sm90` capability 9, `default` the rest.
A row's `name` is a label only: its match fields in short form (phase, kind,
then bounds: `decode.rank_le16.tokens_le8.k_le2290`, `prefill.vocab`).

## Tuning a new workload

Use the [dense tuning workflow](../../../../../../benchmark/kernels/lora_dense/README.md).
The tuner returns independently checked plan candidates for explicit local
shapes; sparse samples are not automatically widened into production rules.
Review every affected selector region and verify model throughput before
changing these tables.

## Historical derivation of the shipped rows

The following records the original exploration, not the current tuning command.
The retired `derive_plans.py` and `bench_dense_site.py` are available in Git
history. Their aggregate fitting and competitor comparisons are not part of
the maintained tuning workflow.

Tables combine generated rules with serving-validated overrides. This command
derives a candidate, not the final Blackwell hybrid:

    python benchmark/kernels/lora_dense/derive_plans.py \
        --sweep-dir <sweep dir> --suffix _b200 --decode-suffix _b200_x1c --out sm100.plans.json

from `bench_dense_site.py` sweeps (every v2 arm on every site of the
Qwen3.5-35B-A3B, Qwen3-1.7B, GLM-4.7-Flash and Inkling-Small presets, ranks
8-256, TP 1-8, decode at 1-64 tokens under the decode graph, prefill at
512-8192 tokens under the prefill graph, each arm checked against the Triton
reference). Decode cells come from a single-forward sweep (`--graph-repeat 1`,
the `--decode-suffix` files): a graph holding eight forwards lets the next
forward's side-stream shrink start during this forward's expand and
over-credits arms with long exposed expands, which the model pays in full.
Prefill cells keep the eight-forward replay (their kernels dwarf the launch
gaps). The table is a grid: one cell per measured (tokens, pool rank) with the
arm that minimizes the summed time ratio to each measurement's best arm,
subject to the exp guard (no member more than 2 percent behind the experimental
path's measurement, else the smallest worst loss), plus up to two geometry
boxes (in-feature band x out-feature band) accepted risk first: a box that
removes exp losses without creating one is taken even if it lowers the summed
gain; otherwise a box needs 2 percent of summed gain over its members. Rows are written pool-major,
tokens ascending, boxes before their cell's default, so a query lands in the
smallest measured cell that bounds it. Band edges sit at the geometric mean of
neighbouring measured widths. Split-K candidates are the two deterministic
modes, serial and planes (an fp32-atomic mode was faster on some decode cells
but not repeatable run to run; no table ever selected it, and it was removed).

Table candidates are compared on end-to-end servers with the same node,
adapters and graph settings. The recorded pairs run the previous table first,
then the candidate, with five benchmark repeats within each server. This
measures within-server spread, not variation across independent server starts;
small differences need a reversed-order comparison. Kernel timings rank arms,
but do not alone establish a serving improvement.

The first two rows of every table serve the standalone shrink/expand entry
points (the `embedding` and `lm_head` kinds), one for decode and one for prefill, with
the in-code default plans (grouped shrink and expand; sorted route and split-K
4 at decode, block 64 at prefill). Their evidence is the Qwen3-1.7B
vocab-adapter servers (embed_tokens + lm_head, batch 64, output tokens/s): with
a per-row head row the row ran 14320 on H200 and 20910 on GB300 against the
Triton backend's 14999 and 22710; with these rows 14905 and 23060. The
per-row pair had been chosen from a B200 comparison against a grouped head
row without the sorted route and the split-K, which lost as well.

## End-to-end re-check (2026-09-23)

The table options were re-measured in serving as an ablation (remove or merge
one distinction, compare end to end; not a retune of the surviving rows):
all-module adapters (attention, linear attention, MLA, dense and shared MLP,
routed experts, Inkling sink, embed_tokens, lm_head) on Qwen3.5-35B-A3B
bf16/FP8, GLM-4.7-Flash, Qwen3-1.7B, Inkling-Small NVFP4/bf16, Inkling NVFP4,
Qwen3.5-397B FP8/NVFP4 and (GB300) DeepSeek-V3 FP4, at the cookbook TP (capped
at 4 on GB300), CUDA graphs on in both phases, pool ranks 8-256, H200 for
`sm90` and GB300 for `sm100`, three rounds with the arms alternating on one
GPU. An option was kept only if removing it cost 3% or more in some cell, or
2% consistently across jobs. Cells reported as ties include noisy ones (GB300
512- and 2048-token prefill varies up to 15% between server starts); the
envelope is the models and ranks listed here.

Changed: the `ab_delta` overlap (A and B on the side stream into a delta) now
serves one row, the `sm100` decode row for up to 16 tokens at pool rank 16
with K >= 2805 and N <= 2290: overlap `a` there cost Qwen3.5-397B NVFP4 with
shared-outer adapters 6.5% decode at 16 requests on GB300. Every other
`ab_delta` row runs overlap `a` (a tie at ranks 8-16 and on the Inkling sink
rows, 2-6% faster decode at ranks 128 and 256). The per-row shrink rows run
the grouped shrink (tie). The
fp32-atomic split mode is gone (no table ever selected it). Split-K in
`sm90.plans.json` is serial (the last program at a tile sums the fp32 planes)
everywhere but two Inkling sink rows: 1-2% faster decode than the planes rows
on Qwen3-1.7B and a tie on every other H200 job, with identical outputs.
`sm100.plans.json` keeps its planes rows: serial everywhere cost Qwen3-1.7B
4-6% decode at 8-64 requests on GB300, and planes on its three serial
rank-128/256 rows cost 9.4% decode at 64 requests. The Inkling sink rows stay
in both tables. Without them, per-expert Inkling-Small NVFP4 on H200 lost up
to 2.1% at 512-token prefill and Inkling-Small bf16 with shared-outer adapters
on GB300 lost up to 3% decode at 16-32 requests. The same removal made
512-token prefill with shared-outer adapters 7.5% faster on H200. Dropping only
the prefill row for up to 1024 tokens at rank above 16 is a wash: per-expert
Inkling-Small lost 2.2% (NVFP4) and 2.5% (bf16) at 512 tokens on H200, while
shared-outer NVFP4 gained 2.6%, on the same site.

Kept, with the cost of removing each: the all-slots shrink (4-19% decode at 1-32
tokens), the per-row expand (grouping it cost up to 11% decode at rank 256;
the all-slots shrink needs it anyway), split-K (4-29% decode), overlap `a`
(3-23% decode), the tuned tiles (2-9% prefill), the block sizes (3-6% prefill),
the head rows (vocab sites), the rank bands (serving every rank with the
rank-64 rows lost 17-27% decode at rank 256 and 2-5% at 128; with the rank-16
rows, 3-13% at rank 64), and the geometry boxes.

Sharing one row across models: at ranks up to 16 the boxes can go (one row per
phase, token band and rank band served Qwen3.5-35B, GLM-4.7-Flash and
Qwen3-1.7B within noise on both architectures). Above that they still pay:
per-model boxes win 3-8% at some cells at ranks 128 and 256 (Qwen3-1.7B), and
dropping the boxes at ranks up to 64 together with the MoE per-row change cost
1-3% decode on the Qwen3.5 per-expert models, so the boxes stay.

The legacy backends, on the one model they can wrap whole (Qwen3-1.7B; they
refuse the MoE LoRA runner): the SGEMM backend (`--lora-backend triton`) runs
21-36% slower decode and up to 16% slower prefill (a tie at 512-2k prefill tokens
on GB300); chunked SGMV 26-39% slower decode and 8-35% slower prefill; the
experimental SGEMM path is within noise at 16-64 decode tokens on H200 but 5-6%
slower at 1-8 tokens, 9-17% slower at every decode cell on GB300, and 10-14%
slower prefill at 4k+ tokens on both. It won one cell against the old tables
(H200, 32 requests over two adapters, +2.2%).

## Evidence (2026-09-07 sweeps; the sweep log has every number)

`sm100.plans.json` (87 rows: two head rows, 75 site rows after removing seven wholly shadowed rows, four short-request band rows with three kept-row copies, and three serial K/N exceptions) is the
table the paired servers chose on B200 and
GB300: its 1-token rows and its prefill rows are the rows the batch-1 and
prefill A/Bs of the first table settled, its rows for 2-64 tokens come from the
single-forward decode sweep taken with the current expand kernels
(`--decode-suffix _b200_x1c`). Against the first table, both arms on the same
kernels, output tokens/s,
median of five benchmark repeats within one server per arm (spread in parentheses):

| Qwen3.5-35B-A3B, a16 + a64 | B200 first table | B200 this table | GB300 first table | GB300 this table |
|---|---|---|---|---|
| decode, batch 1 | 279.0 (277.8..279.9) | 279.4 (278.2..282.2) | 281.4 (281.2..283.3) | 278.9 (278.0..280.8) |
| decode, batch 4 | 895.2 (880.3..916.6) | 890.9 (882.3..903.5) | 914.4 (891.9..925.3) | 918.6 (896.6..938.8) |
| decode, batch 16 | 2430.7 (2396.7..2614.4) | 2482.2 (2451.4..2676.0) | 2517.3 (2456.3..2659.3) | 2543.3 (2502.3..2688.1) |
| decode, batch 64 | 5738.6 (5496.0..5861.6) | 5869.2 (5635.9..5968.9) | 5843.9 (5647.7..5888.5) | 6031.3 (5808.1..6113.2) |
| mixed a8 + a16 + a64, batch 64 | 5692.1 (5475.6..5763.3) | 5795.7 (5595.9..5925.2) | 5760.5 (5618.2..5798.2) | 5954.9 (5770.1..5983.4) |
| prefill-heavy, 16 x 4096 in / 32 out | 525.6 (522.4..526.9) | 527.6 (525.7..528.5) | 555.0 (552.3..555.9) | 558.4 (556.2..560.8) |
| TP 2, batch 64 | 7987.1 (7876.1..8038.5) | 8251.7 (8134.9..8341.3) | 8159.5 (8027.1..8231.6) | 8420.4 (8280.9..8495.9) |
| TP 4, batch 64 | 10621.7 (10472.9..10735.6) | 10885.7 (10747.6..11026.4) | 10786.3 (10626.2..10948.1) | 11074.5 (10915.7..11217.3) |

Both tables select the same rows at one token; the GB300 batch-1 difference
therefore does not isolate a selector change. Batch-4 spreads overlap. The
reported medians from batch 16 up improve by 1-4 percent.

The two wide Blackwell prefill rows (block 64 for N <= 10240, block 128 above)
take the sorted route instead of the segment route. The segment route pads
every request to the block and sizes the grouped expand's grid from a static
capacity of tokens + requests x block, so a batch of 32 short requests launches
up to three times the CTAs of the sorted route (capacity tokens + slots x
block), most of them exiting after their route loads. Same tiles, sorted route,
32 requests x 64 tokens under the prefill graph: in_proj_qkvz (K 2048, N 12288)
205 -> 143 us at rank 8 in a pool of 64 on GB300 (exp 154), qkv 128 -> 112;
4 x 512 tokens 142 -> 120 and 129 -> 112; 4 x 4096 tokens unchanged. The
block-32 and 512-token rows kept the segment route (mixed, -5 to +4 percent)
until every prefill row moved to the sorted route (2026-09-22, below).
Pool-rank-64 short-request rows (2026-09-09). The band rows above cover pools
8-32; pools 33-64 fell through to the generic block-16 / block-32 rows, which the
kernel sweeps on GB300 and B200 put level with or behind exp per site at 4 x 128
and 32 x 64 tokens (o_proj 0.99-1.00, down 1.09-1.14 of exp). Three rows for
pools 33-64: <= 512 tokens with N <= 4096 -> nt16_n16 (qkv 0.66, o_proj 0.83-0.85,
down 0.92 of exp); <= 2048 tokens with N <= 2048 -> g64s_ov (sorted route, block
64, shrink on the side stream: o_proj 0.63-0.64, down 0.66-0.67); <= 2048 tokens
with 2048 < N <= 4096 -> nt16_n16 (qkv 0.84-0.85). Servers (first-token protocol,
pool rank 64, four per row, 21 seeds, pooled median; "of 21" = seeds where the
candidate's per-seed median was lower), B200 against the same tree without the
rows: Qwen3-1.7B 4 x 128 -5.9 percent (21 of 21), 32 x 64 -6.4 (20 of 21),
32 x 16 -5.4 (21 of 21), Qwen3.5 32 x 64 -2.5 (20 of 21). GB300 (servers pinned
to GPU 0's socket; that node's first-token servers otherwise sit 5-6 percent
apart at 4 x 128 even on one tree): Qwen3 32 x 64 -4.8 percent (20 of 21),
Qwen3.5 32 x 64 -1.6 (19 of 21), 4 x 128 level (six servers per arm, -1.0
percent, 17 of 21 on the eight-server run), 32 x 16 level (-0.3 percent). The
fair check of the 4 x 128 cell against the experimental path: B200 -9.5 percent
(21 of 21), GB300 -0.7 percent (19 of 21). A block-32 overlap row for the <= 512-token gate_up / in_proj geometry
(g32_ov, 0.47 of exp in the kernel sweep) was measured and not taken: with the
per-request route it pads 32 x 16-token requests to twice their rows (+0.3
percent at 32 x 16), and its sorted variant lost 7 percent at 4 x 128 on GB300.

Every block route is the sorted route: tokens grouped by adapter slot, capacity
tokens + (slots + 1) x (block - 1), independent of the request count, so the
same captured plan serves any request count within its token bucket. The
per-request segment route (padding each request to whole blocks in place) was
measured against it on 2026-09-22 and removed with its builder and the `route`
plan field: site harness on GB300, H200 and B200, sorted at parity or faster on
every shipped prefill row under the prefill graph (the block-16 rows at 512
tokens by 5-12 percent) and in eager prefill above 2048 tokens (parity; a 1-2 us
loss on 512-token single-request forwards that serving runs under a graph);
server level, GLM-4.7-Flash on H200 and Qwen3.5 on GB300, 18 cells within 3
percent.
Routing is built on first use per forward and reused by sites with the same key.
CUDA graphs capture those kernels and replay them against current metadata;
the executing tensor's token count selects the plan and route extent. LoRA
initialization needs only buffer capacities, not the graph runner's bucket lists.
Host preparation refreshes metadata; replay needs no Python planning or routing.
The following historical server results used the former 32-request switch and
do not establish performance of this graph-capacity change.
Servers, 32 requests x 64 input tokens x 1 output token, prefix cache off,
prefill graph at 2048 tokens, batch first-token latency pooled over two servers
per arm: B200 45.46 -> 43.58 ms (-4.1 percent), GB300 47.49 -> 45.53 ms (-4.1
percent), identical output tokens. The eight standard Qwen3.5 rows (decode
batches 1-64, three adapters, prefill-heavy 16 x 4096, TP 2 and 4) move by -0.6
to +0.7 percent, inside their run-to-run ranges, on B200 and GB300.

Short-request band rows (Blackwell). At the shapes the first sweeps never
measured, 32 requests x 16 or 64 tokens and 4 x 128, the block-16 and block-32
rows were 4-33 percent behind exp on o_proj, out_proj, gate_up, down and
in_proj_qkvz for pools 8-64. A 22-arm sweep at those shapes on B200 and GB300
(ranks 8-64) picked one arm per (pool, tokens) band, judged on every cell of the
band on both nodes: no cell may regress by more than 2 percent against the row
it replaces and every losing cell must come within 2 percent of exp, with one
accepted exception (GB300 out_proj at rank 8 in a pool of 8, 512 tokens, 1.027x
exp under the pool-8 row). Four rows
ship (plus the three copies): pools 8, 16 and 32 at <= 512 tokens take block 16 with a 16 x 256 shrink
tile and a 128 x 32 expand tile; pool 16 at <= 2048 tokens the same with a
32 x 256 shrink tile. Kernel time over a band falls 16-25 percent (geometric
mean of its cells). Paired servers (32 requests x 16 or 64 input tokens x 1
output token, four alternating servers per arm, the 21 payloads paired by
seed): pool 16 at 32 x 16 tokens 32.55 -> 31.71 ms on B200 (faster on 21 of 21
seeds) and 44.08 -> 42.89 on GB300 (18 of 21); pool 16 at 32 x 64 41.93 ->
40.88 on B200 (21 of 21), a tie on GB300 (13 of 21); pool 8 at 32 x 16 32.59 ->
31.99 on B200 (20 of 21), a tie on GB300; pool 32 at 32 x 16 32.70 -> 31.92 on
B200 (21 of 21) and 43.16 -> 42.92 on GB300 (12 of 21). Two candidates from the same sweep
were not shipped: pool 8 at <= 2048 tokens (a 32 x 256 shrink tile), 0.81x the
shipped rows in the kernel bench yet slower on B200 servers (faster on 5 and 7
of 21 seeds), and block 32 with the sorted route for pool 64 at <= 2048, a tie
on both nodes. The kernel bench times one layer's base GEMM plus its LoRA
kernels under a CUDA graph, and these prefill plans run the LoRA kernels
serially (no overlap), so the disagreement is not a two-stream effect; graph
capture, batch composition, route preparation and the rest of the model differ
between the bench and a server, and the servers decide. A query
matches every row whose pool and token bounds cover it, first match wins, so a
band that keeps the shipped rows needs bounded copies of them whenever a
larger band's row would catch it: pool 8 at <= 2048 tokens carries three such
copies ahead of the pool-16 row (without them a pool-8 request at 2048 tokens
would take the pool-16 row, the very candidate the servers rejected for pool
8). Pools above 32 and batches above 2048 tokens see none of these rows.
Three serial K/N exceptions follow from the same sweep, on the exact Qwen3.5
geometries where the remaining losses sat: o_proj/out_proj (K 4096, N 2048)
for pool 8 at <= 2048 tokens, and in_proj_qkvz (K 2048, N 12288) plus
o_proj/out_proj for pool 64 at <= 512 tokens, all with the 16 x 256 shrink and
128 x 32 expand tiles (13-18 percent behind exp before, within 8 percent ahead
after in the layer bench). Servers, eight alternating per shape, paired by seed:
pool 8 at 32 x 64 tokens faster on 18 of 21 seeds on B200 (-1.3 percent) and 17
of 21 on GB300 (-1.7 percent); pool 64 at 32 x 16 tokens 21 of 21 on B200
(-2.0 percent) and a tie on GB300 (9 of 21).

`sm90.plans.json` (133 rows: two head rows, 129 site rows and two serial
in_proj_qkvz exceptions for pools 33-64: block 16 to 2048 tokens, from the
same 22-arm sweep;
servers with 32 requests x 64 tokens faster on 21 of 21 seeds, -1.7 percent,
and 32 x 16 tokens on 17 of 21, -0.5 percent) is derived from the H200 eight-forward sweep with
planes split-K. The recorded Qwen3.5 batch-1 comparison gains about 10 percent
over the in-code defaults. A single-forward candidate tied this table on the
recorded Qwen3.5 rows (-1.1 .. +1.6 percent, within-server spreads overlapping),
so the eight-forward table stays. The older GLM and DeepSeek comparisons were
not repeated with the final table; they do not establish a current all-model
win against the experimental two-stream path.

`default.plans.json` is the first Blackwell table (18 rows with the two head rows), kept for
architectures without a sweep.

The sweep's best-arm and GPU-critical-path reports are screening evidence,
not guarantees about the shipped selector. Some older critical-path comparisons
used the since-removed atomic mode. Compare the actual
table-selected arm against the experimental path to assess remaining losses;
the single-forward timer's launch floor also makes small percentage differences
at short sites inconclusive without a paired rerun or a GPU timeline.

## MLA corrections (2026-09-24)

The absorbed-MLA q/v corrections (`dense/mla_correction.py`) do not read these
tables. They run one plan per phase: grouped shrink and expand without split-K,
block 16 with the shrink overlapping the base BMM at decode, block 64 in line at
prefill. The alternatives never paid. At kernel level
(`bench_mla_correction.py` sweeps under CUDA graphs, rank 16, 1/8/16/32 tokens,
GLM-4.7-Flash and DeepSeek-V3 at 16 and 32 local heads, H200 and GB300) nothing
beat this plan by more than 2.5% in any cell, and at 8 tokens and up serial
split-K cost up to 20%, the per-row expand up to 57% and the all-slots shrink up
to 88%. End to end (GLM-4.7-Flash per-expert on H200 and GB300, DeepSeek-V3
per-expert on GB300, 2 rounds, 1-32 decode requests) no alternative won a cell
except one DeepSeek-V3 all-slots cell between -18% and -20% cells of the same
arm. The per-row shrink is about 0.4 us faster at 1 token (about +1.3% end to
end at 1 request, under the bar) and slower above.

## Inkling sink rows (2026-09-12)

The `sink_gate_up` and `sink_down` rows at the top of `sm100.plans.json` and
`sm90.plans.json` serve the Inkling shared-expert sink's two site kinds and nothing else
(`kinds` filters): `sink_gate_up` (the gate/up expand of both adapter layouts,
looked up at the pool rank of B) and `sink_down` (per-expert B: the windowed
shrink over the pool's compact A, looked up at E*R). The shared-B down (one
shrink over the concatenated per-slot A) has no sink row; it runs as an
ordinary `linear` site. A row whose shrink family is `all_slots` never serves
`sink_down` (one dense matmul cannot window per rank block): the resolver
passes over it and the next matching row serves. They come
from the bounded sink search in `~/Desktop/lora_refactor/inkling_sink_20260911/`
(`sink_plan_search.py`, `search_batch*.sh`, results in `results/search/`): per
cell of (site, TP-local shape, layout, pool rank 16/64, tokens) the complete
sink forward under the CUDA graph for every family x overlap x split-K
candidate with the tables' default tiles, checked against the torch reference;
a row is a candidate within a few percent of every covered cell's best (worst
gaps 0.7-5.9 percent).

Decode (1-32 tokens, both tables): two rows per site keyed on the local
geometry, the TP>=2 shapes (gate/up N <= 4096, down K <= 2048) and the TP1
shape. Prefill (512 and 2048 tokens searched): the two tables differ because
their generic rows differ. On GB300 the rank-64 generic rows carry narrow B
tiles and trail the block-64 grouped/grouped/overlap-a plan by 10-25%, so the
table gets explicit prefill rows for gate/up (rank <=16 up to 1024 tokens,
rank <=16, rank >16, TP1 rank <=16 up to 512 tokens = the generic plan kept
explicit because the TP1 row loses 16% there on B200, TP1) and for the
windowed down; on H200 the generic
rank-16 rows already carry rank-matched narrow A tiles and win at 2048 tokens,
so only two gate/up rows up to 1024 tokens are added. Cells without a sink row
(other token counts, other geometries, the shared-B down at prefill) fall
through to the generic rows at the true pool rank; the legacy wrapper's padded
expand had been served by the 2R band.

One pair of rows is not from the harness. The single-layer search prefers overlap
`a` (the shrink overlaps the base GEMM, the expand follows it) for every sink
decode row; in the B200 TP4 server, where the sink runs on its own stream next to
the routed experts, that expand is a serial tail and decode at 8 requests ran
1.5% below the baseline with every overlap-`a` row, grouped or not, while the
baseline's `all_slots/per_row` rows with the delta overlap (shrink and expand
on the side stream, the delta added to the base output) were at parity. So the
sm100 table's TP>=2 decode rows up to 16 tokens carry the baseline's families
with `ab_delta` on the compact operands (6-14% slower per layer in isolation,
at parity in the server); from 17 to 32 tokens the grouped overlap-`a` rows
stay, where a 32-request server showed no gap. A single-layer harness cannot
choose an overlap mode; only the server can (`e2e_ab*.sh` in the sink folder).

## Wider Blackwell tiles for the generic prefill rows (2026-09-26)

The `sm100` row `prefill.n_le4096` (block 32) shrinks with a 256-deep K tile
(was 128) and expands with a 128-wide N tile (was 32); `prefill.n_le10240`
(block 64) expands with the 128-wide N tile (was 64). These rows serve every
rank above 2048 tokens and, from 513 tokens, the ranks above 64 (ranks 17-64
only when the site is wider than 4096). Site harness on GB300 (30 calls under
the prefill graph, LoRA kernel sum, candidate over the previous tiles): rank
16 over K 2048 and 4096 with sites of N 2048, 4096, 2048+128+128, 5120 and
8192 at 1k-8k tokens, 0 of 40 points slower than 2 percent, 1k-2k tokens
level (0.99-1.00), 4k tokens 0.75-0.89, 8k tokens 0.80-0.89; rank 32 and 64
at N 5120 and 8192, 0.86-0.93; rank 128 at N 4096 0.79-0.87, at N 2048+128+128
0.83-0.96, at N 8192 0.98-0.99 up to 2k tokens and 1.02 at 4k-8k (the one
point over parity); rank 256 at N 4096 0.82-0.88, at N 8192 0.93-0.94, at
N 2048+128+128 0.79-0.91. The rows kept at 512 tokens and below and the
rank-banded rows up to 2048 tokens are unchanged.

## Batch-1 shrink in planes mode (2026-09-26)

The two `sm100` rows for one token (`decode.tokens_le1`, `decode.tokens_le1.n_ge8192`)
split K 8 ways as before but in planes mode (the expand sums the planes on
load, as the rows for 2-32 tokens do) with 8 warps and 2 stages. In serial
mode no tile beat a 5-6 us shrink at one token: the last-arriving program's
plane reduction is the latency. Site harness on GB300 at 1 token, rank 16,
shrink + expand per apply, new over shipped: K 2048 at N 2048-12288 (seven
site widths) 0.69-0.81; K 4096 0.75-0.85; K 8192 0.82-0.93; no site slower
(`decode.tokens_le1` sweeps: serial best 5.0-6.0 us shrink, planes 2.8-3.4 us
with the expand up 0.3-1 us). The Qwen3.5-397B TP4 attention block has three
such sites per layer, about 7 us per layer at batch 1. Ranks 64 and 128 gain
as well (0.76-0.94 over the same sites); at rank 256 the sites wider than
8192 lose 16-18 percent in planes mode (the grouped expand re-reads eight
256-wide planes per N tile), so the wide row keeps its serial tiles above
rank 128 (`decode.rank_le128.tokens_le1.n_ge8192` carries the planes tiles,
`decode.tokens_le1.n_ge8192` the shipped serial ones), while the narrower row
stays in planes mode at every rank (rank 256 at N 2048-4096: 0.89-0.92).

## Batch-1 expand tiles (2026-09-27)

The one-token `sm100` rows (`decode.tokens_le1*`, `decode.rank_le128.tokens_le1.n_ge8192`)
shipped the expand (B) with 128x64 tiles and 4 warps. At rank 128 and 256 the
batch-1 decode of the dense job on GB300 read 2 percent under the pre-rebase
engine, and its per-step profile put the difference on the expand kernel
(per-row B 13-20 percent slower per launch than the old per-pair kernel).
A sweep of the expand tiles on the Qwen3-1.7B sites (qkv, o, gate_up, down;
GB300, tokens 1; `tools/dense_b_tile_sweep.py` of the sweep harness) over
N 32/64/128 x K 64/128/256 x 4/8 warps: 64x256 with 4 warps is neutral or
better at every rank (LoRA-side kernel sum over the four sites, new over
shipped: rank 16 0.98, 32 0.98, 64 0.98, 128 0.85, 256 0.80; per site at
rank 128: 0.82-0.87, at rank 256: 0.77-0.88). 32x256 is the best single tile
at rank 256 (0.79) but 1 percent slower at rank 32, so the rows take 64x256.
The K tile covers the whole rank up to 256 (a rank below the tile is masked,
which is why 64x128 and 64x256 read the same at rank 128).
