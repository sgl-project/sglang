# AKO Loop Checklist

Use this checklist after `scripts/ensure_ako4all_clean.sh` succeeds.

## Minimum Repo Layout

Inside `AKO4ALL/`, prefer these files for a diffusion kernel task:

- `input/reference.py`
- `input/<kernel>.py`
- `solution/<kernel>.py`
- `bench/bench_<kernel>.py`
- `context/<kernel>_notes.md`

## Baseline Checklist

- Reproduce the current SGLang kernel exactly in AKO first.
- Run the custom microbench before making edits.
- Record one representative `ncu` report on a real hot shape.
- Note the baseline bottleneck in plain language.

## Iteration Discipline

- One optimization idea per iteration.
- Re-benchmark after every code change.
- Log the result in `ITERATIONS.md`.
- Keep the best candidate easy to identify.

Stop a direction early when:

- 3 consecutive iterations do not beat the best runtime
- correctness gets fragile
- AKO-only gains stop transferring to real denoise runs

## Real Validation Gate

Before calling a kernel "done", validate all of:

- syntax or import checks
- targeted unit test or regression test
- kernel or op-level benchmark
- model-level denoise benchmark with perf dumps
- one generated image if the PR needs production proof

## PR Artifact Checklist

Prepare these artifacts:

- microbench table
- denoise-stage table
- end-to-end table
- one `ncu` before or after pair
- one short explanation of why the kernel got faster
- one generated output image when applicable
