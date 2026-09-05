# Changing the MoE LoRA config tables

The file format is documented next to the tables, in
`python/sglang/srt/lora/moe/configs/README.md`. This file is about the other
half: how selection actually cascades, how to change a table without breaking
a case you did not think about, and how to tell a real measurement from a
hopeful one.

Read this before editing `{arch}.plans.json` or `{arch}.tiles.json` by hand.

## The cascade, in the order it runs

Four things pick what a layer executes, and they run at two different times.

**At weight bind (once per layer, per server):**

1. **Plan row.** `resolve_plans` walks `scenarios` top to bottom and takes the
   FIRST row whose `layout`, `phase` and `max_rank` all admit this layer. An
   absent key is a wildcard. If the model's geometry is outside `domain`, or no
   row matches, the `fallback` rows are walked the same way instead.
2. **Provider.** Joined from two places: the matched row's `base_gemm_rows`
   (`expert_major` / `route_major`) says which row order the plan needs, and
   `--moe-lora-base-gemm` says which vendor implements it (`cutedsl` default
   or route-major-only `triton`). A geometry a vendor cannot admit fails at
   attach.
3. **Tile rules.** `resolve_tiles` takes that row's rule list and drops every
   rule whose `max_rank` is below the bound rank. What survives is a ladder.

**Per forward:**

4. **Tile pick.** `TileTable.config_for(num_tokens)` walks the surviving ladder
   and takes the first rule whose `max_tokens` admits this batch.

Two consequences that are easy to miss, and both have bitten:

- **A rule with no `max_tokens` terminates the ladder.** Everything below it is
  unreachable for any rank that rule survives at. `decode.per_expert`'s
  `max_rank: 16` rule has no `max_tokens`, so at rank 16 the whole rest of the
  ladder is dead. If you add a rule and it seems to do nothing, check whether an
  earlier rule ends the ladder first.
- **Ordering is the only precedence.** There is no specificity scoring. A
  catch-all placed above a narrow rule silently wins.

## Before you change a table

Work out, on paper, the full cross product of (rank × tokens) the change
touches, and which cells must be **unchanged**. You need that list twice: to
convince yourself the edit is scoped, and to read the benchmark afterwards.

`test_moe_lora_plan_tables.py` has a helper shape worth copying — resolve the
table for several ranks and assert the whole matrix, e.g. rank 32 walking
tiny/mid/mid/large across 4/16/32/33 tokens. Asserting the matrix catches an
ordering mistake that asserting one cell will not.

## Measuring a change

Point `SGLANG_LORA_MOE_CONFIG_DIR` at a directory holding **only the file you
are changing**. Lookup falls back per file, so a directory with just
`gb300.tiles.json` still gets its plans from the package. Do not copy files you
are not changing — a stale copy is worse than none.

Two different questions, two different setups, do not mix them:

- **"Is this candidate faster?"** — force the candidate on one row and compare
  against the shipped table. Fine for exploring.
- **"Does the table I am about to commit do what I claim?"** — run the actual
  edited file against the actual old file, same tree, same flags, nothing else
  different. This is the one that belongs in a commit message.

### Reading the result

**The cells your change cannot reach are the control.** If the change only
touches tokens 17–32 at rank ≤32, then batches 1, 8 and 16 must come back flat.
When they do, they also measure the run-to-run noise for that machine and model
on that day, which is what makes the cell that did move interpretable. When they
do not, something else moved — stop and find out what.

A candidate that is byte-identical to the incumbent is a **self-check, not a
test**. It should reproduce the baseline; it tells you the forcing mechanism
works. It does not tell you the incumbent is right. Check before concluding
anything: two rules in a shipped ladder may already hold the same values.

Protocol used by every number in the campaign docs: four passes per batch size,
median after discarding the first, batch sizes 1/8/16/32, 4096 in / 1024 out,
`--max-loras-per-batch 4`. Noise is about ±2%, wider at batch 1. **Effects under
about 1% cannot be resolved by this protocol** — take more passes at one batch
size instead of arguing about the fourth digit.

## Traps that have cost real time

- **`SGLANG_LORA_MOE_CONFIG_DIR` also moves the base-GEMM search root.** The
  `base_gemm/` bucket tables resolve under the same root. Lookup falls back per
  file, so an override directory without a `base_gemm/` subtree still finds the
  packaged ones — but check `configs/base_gemm/` before assuming a forced run
  and its baseline used the same base-GEMM tables.
- **Do not index rules positionally in tooling.** A script that says "rule 2 is
  the mid-M one" breaks the moment anyone inserts a rule. Match on the
  predicate instead.
- **Some rules are duplicated on purpose, and nothing enforces it.** See the
  note in `configs/README.md`. If you retune one of a duplicated pair, retune
  the other in the same edit.
- **Vendor and row order are different axes, and the config only holds one.**
  `base_gemm_rows` in the table is the row order; the vendor is the server flag.
  Changing the row order changes the memory footprint and can make a server
  fail to start; changing the vendor does not. Measure them separately or you
  will not know which one moved the number — an earlier sweep forced whole
  providers and conflated the two, which is how a 30% decode result got
  attributed to the wrong axis.

## Scripts here

- `tune_lora_config.py` — report which shipped rows serve a model (`--check`) and emit a seed plan table whose `domain`
  covers it (`--emit-seed`) into an output directory that
  `SGLANG_LORA_MOE_CONFIG_DIR` can point at directly. The seed reuses existing
  plans: validate admission and correctness on the new geometry, then tune it
  end to end with the campaign protocol. The former `--sweep` mode was retired.
- `sweep_masked_gemm_configs.py` — sweep masked base-GEMM launch configs for one
  geometry on the current device and emit the M-bucketed store.
- `make_config_variant.py` — build a one-file override directory for a
  forcing run, matching rows by name rather than by position.
