# gfx950 mono common (shared device and plan code)

What every mono kernel family on gfx950 shares: `device/` (traced inside kernels: tagged mailbox, wave / math /
packing primitives, MX helpers, timeline stamps) and `plan/` (plain Python: execution model, region layout, build
keys, arithmetic on ints and traced values alike, the traced hand-off contract). It imports no model and holds no
model constant. Host-side mechanisms (peer memory, TP consensus, step fence, compile-only builds) live in
`sglang.srt.layers.mono.runtime`.

## Origin

Vendored from [ROCm/ATOM](https://github.com/ROCm/ATOM) @ 454d0e3023, `atom/mono/{device,plan}` (MIT, header kept
in every file).

## Local changes

- `device/sync.py` `Mailbox`: `tag=` for kernels that tag hand-offs with a step epoch instead of `layer + 1` (nothing
  cleared between steps); a raw address (no `Addr`) is accepted as device-scope scratch and reported by the hand-off
  check as unnamed (`plan.trace.note_unnamed`).
- `device/ops.py` `ld_i32`: optional cache policy `cm` (default 0 = the previous load).
- `plan/arith.py`: `imin`, `imax`, `i32`, and a `cdiv` that works on traced values; `plan/shard.py` uses it.
- `plan/build_key.py` `source_digest`: paths relative to this `gfx950` package (a family passes its own directory and
  `"common"`).

## Families using it

- `deepseek_v41/`: every device helper (its private copy of this code was removed; see its SOURCE.md).
- MiniMax-M3 (not ported yet): of the 30 helpers in zijiecode/sglang-mono-amd `kernels/ops/mono/gfx950/common.py`,
  22 are identical to `device/`, 1 differs only in its docstring (`unlikely`), 5 differ (`kernel_symbol`, `xshfl`,
  `block_max`, `bf2_f32`, `Mailbox`) and 2 are M3-only (`release_stores`, `ar_pack_sumsq`).

## Not used by any family yet

`device/{ranks,select}` beyond `peer_bases` / `sum_partials`, `plan/check`, parts of `plan/trace`, and helpers such as
`fp8x8_bf16_pk` (which the M3 kernels use). Kept until the MiniMax-M3 port decides what it needs.
