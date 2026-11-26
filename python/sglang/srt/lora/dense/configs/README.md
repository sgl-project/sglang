# Dense LoRA execution plans

`<arch>.plans.json` selects kernels, routing tiles and overlap for
`--lora-backend triton_v2`. Capability major 9 selects `sm90`, major 10 or
above selects `sm100`, and other devices select `default`.

## Selection and overrides

Rows are tried in file order; the first compatible match wins. Selectors cover
phase, layer kind, token count, pool rank and input/output widths. Put narrow
rules before broad ones. A row name is only a label, not a selector. The
windowed Inkling sink-down site cannot use an all-slots shrink.

`SGLANG_LORA_DENSE_CONFIG_DIR` can supply an override file for an architecture.
Missing files or unmatched queries use the defaults in `../plan.py`. Tables
are cached within the process; use a fresh server to evaluate a changed file.
Malformed tables fail loading, including invalid rows that the current model
would not select.

Each plan contains:

- `a_family`: grouped, per-row or all-slots shrink;
- `b_family`: grouped or per-row expand;
- `overlap`: `none`, `a`, or `ab_delta`;
- `block_size`: aligned routing tile;
- `a_tiles` and `b_tiles`: complete launch configurations.

Grouped stages use aligned routes; per-row stages use raw routes. All-slots A
writes slot planes and requires per-row B. Split-K is grouped-A only, with
deterministic `serial` or `planes` reduction. These compatibility rules are
independent of which configuration is fastest.

## Tuning

Use the [dense tuner](../../../../../../benchmark/kernels/lora_dense/README.md)
for explicit resident shapes and devices. Independent validation of a kernel
candidate does not qualify an entire production selector region. Check every
affected kind/rank/geometry/phase, then measure model throughput with the actual
graphs and base backend before promoting a row.

Historical exploration and measurements remain in Git history; they are not
instructions for the maintained tuner or a current no-regression guarantee.
