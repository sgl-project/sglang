# evalscope[all] deps without downgrading sglang

`sweep.sh` installs `evalscope[all]` and supports a `PIP_NO_DEPS=1` prefix:

```bash
PIP_NO_DEPS=1 ./sweep.sh
```

With that prefix, pip installs **only the evalscope package** — none of its
dependency tree. The sweep then relies on those deps already being present in
the image. This directory is what puts them there, without disturbing the
image's bleeding-edge pins.

## Why `PIP_NO_DEPS` is needed

evalscope's dependencies formally cap several packages *below* what the
inference image pins, so there is **no resolution** that installs evalscope[all]
and keeps the image intact. The hard conflicts (evalscope commit `acd09b44`,
versions measured against the sglang image):

| evalscope dep (extra)                    | caps to                    | image has        |
| ---------------------------------------- | -------------------------- | ---------------- |
| `datasets` (via `modelscope[datasets]`)  | `datasets<=4.8.4`          | datasets 4.8.5   |
| `ms-opencompass` (opencompass)           | `numpy<2.0.0`              | numpy 2.3.5      |
| `gradio` (app)                           | `markupsafe~=2.0`          | markupsafe 3.0.3 |
| `gradio` (app)                           | `pillow<12.0`              | pillow 12.2.0    |
| `gradio` (app)                           | `python-multipart==0.0.12` | 0.0.29           |
| `langchain-core` (rag)                   | `packaging<26.0.0`         | packaging 26.2   |

A plain `pip install evalscope[all]` resolves these by **downgrading numpy
(2.3.5→1.26.4), datasets, markupsafe, pillow, python-multipart, packaging** and
the starlette/fastapi web stack — all sglang runtime pins. `PIP_NO_DEPS=1`
sidesteps that: the newer versions work in practice for the only thing the
sweep runs (`evalscope perf`); the caps above are soft in reality. (Note
`openai` and `safetensors` are *upgrade*-wants, not downgrades, so they are not
a concern.)

## Install the deps (run once per image)

```bash
evalscope-deps/scripts/install_evalscope_deps.sh
```

It freezes the current env into a temporary constraints file (so nothing
installed can change) and `pip install --no-deps` the curated list in
`evalscope-all-nodeps.txt` — the evalscope[all] closure minus everything the
image already ships. After it runs, `PIP_NO_DEPS=1 ./sweep.sh` works.

`pip check` will then list a handful of mismatches — exactly the table above.
They are **expected and harmless**: every one is in the `app`/`rag`/`opencompass`
extras that `evalscope perf` never imports.

## Verify

```bash
evalscope perf --help                                                # exit 0
python3 -c "from evalscope.perf.plugin.datasets import swe_smith"    # imports
python3 -c "import importlib.metadata as m; print(m.version('numpy'), m.version('transformers'))"
# -> 2.3.5 5.8.1  (unchanged)
```

## Platform-restricted entries

A few packages in `evalscope-all-nodeps.txt` carry PEP 508 environment markers
because they have no aarch64 wheel and pure-pip can't build them from source on
ARM hosts. Pip skips marker-gated lines on non-matching platforms, so the same
list installs cleanly on both x86_64 and aarch64 sglang images.

| package        | marker                            | reason                                                                 |
| -------------- | --------------------------------- | ---------------------------------------------------------------------- |
| `decord==0.6.0`| `platform_machine == "x86_64"`    | Last released 2022; only x86_64 manylinux2010 wheel — no aarch64 wheel.|

These markers are preserved by `regen_nodeps_list.sh` (it reads the existing
file before rewriting it), so a re-resolve after bumping `EVALSCOPE_COMMIT`
won't drop them. If you add a new platform-restricted dep, set the marker in
the file by hand once — regen will carry it forward.

`evalscope perf` (the only command the sweep runs) does not import any of
these packages, so skipping them on aarch64 is safe in practice.

## When the sglang image is updated

The install *mechanism* is image-agnostic: it freezes whatever is live and uses
`--no-deps`, so it can never downgrade a pin regardless of the image's versions.

The package list (`evalscope-all-nodeps.txt`) is a snapshot, though, so it can
drift when a new image changes which packages are pre-installed:

- **A listed package is now pre-installed** (newer image bundles more): the
  install script detects it and **skips it**, keeping the image's version. No
  error, no downgrade.
- **A previously pre-installed transitive dep was removed** from the image: it
  is neither installed nor in the list, so `evalscope perf` hits an
  `ImportError` at runtime. Fix by regenerating the list (below).
- **A kept package jumped versions** (newer torch/transformers/etc.): same
  practical-compatibility assumption as the original install; usually fine.

Rule of thumb: after a meaningful base-image bump, **regenerate the list** so it
matches the new pre-installed set, then re-run the install.

## After bumping EVALSCOPE_COMMIT

The package list is pinned to evalscope commit `acd09b44`. If you change
`EVALSCOPE_COMMIT` in `sweep.sh`, regenerate the list **on a clean base image**
(so "already-installed" == the image's set, not a dev box that already has
evalscope deps):

```bash
EVALSCOPE_COMMIT=<new_sha> evalscope-deps/scripts/regen_nodeps_list.sh
```

This re-resolves `evalscope[all]` (dry-run, installs nothing) and rewrites
`evalscope-all-nodeps.txt` with the new closure minus the currently-installed
packages.

## Files

- `evalscope-all-nodeps.txt` — curated `--no-deps` install list (112 packages).
- `scripts/install_evalscope_deps.sh` — freeze current env + install the list.
- `scripts/regen_nodeps_list.sh` — recompute the list for a new commit.
