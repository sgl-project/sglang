# gfx950 mono common (shared device and plan code)

What every mono kernel family on gfx950 shares: `device/` (traced inside kernels: the tagged mailbox, wave / math /
packing primitives, MX helpers, the top-k select stage) and `plan/` (plain Python: the execution model, region
layouts, the traced hand-off record). It imports no model and holds no model constant. Host-side mechanisms (peer
memory, TP consensus) live in `sglang.srt.layers.mono.runtime`.

## Origin

Vendored from [ROCm/ATOM](https://github.com/ROCm/ATOM) @ 454d0e3023, `atom/mono/device/{ops,sync,select,mx}.py` and
`atom/mono/plan/{execution,layout,trace}.py` (MIT, header kept in every file). ATOM's other `device` / `plan` modules
are not vendored until a family here uses them.

## Local changes

- `device/sync.py` `Mailbox`: `tag=` for kernels that tag hand-offs with a step epoch instead of `layer + 1` (nothing
  cleared between steps); a raw address (no `Addr`) is accepted as device-scope scratch (`plan.trace.note_unnamed`).
- `device/ops.py` `ld_i32`: optional cache policy `cm` (default 0 = the previous load); `kernel_symbol` demangles to
  `sglang::<stem>_<params>`.

## Families using it

- `minimax_m3/`: every device helper but `release_stores` and `ar_pack_sumsq` (its `device.py`), the `select` stage
  helpers, and the `Mailbox` with a step-epoch tag (see its SOURCE.md).
