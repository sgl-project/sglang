# Fast-dVLM validation

Run these commands from the repository root in a Linux CUDA environment with
this branch installed. Use the same checkpoint revision on every run. The
harness records input/output token IDs, finish reasons, cache counts, launch
settings, and package versions. It also records the GPU and the current
directory's Git head.

## Prepare prompts on CPU

The checkpoint revision below is the one used by the original port. Select a
different immutable revision if needed, then use the resulting manifest for
every comparison.

```bash
python test/manual/dllm/validate_fast_dvlm.py prepare \
  --revision d7977da26e374f3ef7c96c1700b2ebab50ff62fc \
  --out /tmp/fast-dvlm-prompts.json
```

This creates four aligned prompts and 31 unaligned prompts, covering length
remainders from 0 to 31 modulo the block size. All prompts share a textual prefix. Add
`--local-files-only` to use an existing tokenizer cache.

## Compare cache and graph behavior

Each command starts and stops its own engine. It warms the first prompt before
measurement, which also primes the shared prefix when radix caching is on.
Run on an otherwise idle GPU. The default memory fraction is 0.42, matching
the earlier A40 experiment. Use `--mem-fraction-static` to adjust it.

```bash
python test/manual/dllm/validate_fast_dvlm.py run \
  --prompts /tmp/fast-dvlm-prompts.json --radix off --out /tmp/off.json
python test/manual/dllm/validate_fast_dvlm.py run \
  --prompts /tmp/fast-dvlm-prompts.json --radix on --out /tmp/on.json
python test/manual/dllm/validate_fast_dvlm.py compare \
  /tmp/off.json /tmp/on.json --require-cache-hit

python test/manual/dllm/validate_fast_dvlm.py run \
  --prompts /tmp/fast-dvlm-prompts.json --radix on --eager \
  --out /tmp/eager.json
python test/manual/dllm/validate_fast_dvlm.py compare \
  /tmp/on.json /tmp/eager.json --require-cache-hit
```

Exact output IDs and finish reasons should match for each prompt. The cache
check also requires a positive cached-token count in every result from the
second run. A missing result, duplicate ID, different input, changed checkpoint,
or absent output token IDs fails validation. Text equality cannot substitute
for token equality.

## Check batching

```bash
python test/manual/dllm/validate_fast_dvlm.py run \
  --prompts /tmp/fast-dvlm-prompts.json --radix on \
  --max-running 4 --batched --out /tmp/batch4.json
python test/manual/dllm/validate_fast_dvlm.py compare \
  /tmp/on.json /tmp/batch4.json --require-cache-hit --allow-differences
```

Different batch sizes can change floating-point results near the confidence
threshold. This diagnostic reports differences without treating them as a pass
for equivalence. Invalid or incomplete artifacts still fail. Keep the full
report and inspect differences by prompt remainder. Measure output quality
separately on unaligned prompts with a fixed GSM8K evaluation set.

## Compare the reference fork

Copy the harness and the same prompt manifest into the reference fork's
environment. Its SGLang package must be the one imported by that Python
interpreter. Run from that checkout so the artifact records its Git head.
`--fork` uses the reference launch settings and requires one request at a time
with radix caching off.

```bash
python test/manual/dllm/validate_fast_dvlm.py run \
  --prompts /tmp/fast-dvlm-prompts.json --fork --aligned-only \
  --out /tmp/fork-aligned.json
```

Back in the port's environment:

```bash
python test/manual/dllm/validate_fast_dvlm.py run \
  --prompts /tmp/fast-dvlm-prompts.json --aligned-only \
  --out /tmp/port-aligned.json
python test/manual/dllm/validate_fast_dvlm.py compare \
  /tmp/fork-aligned.json /tmp/port-aligned.json --allow-differences
```

The fork starts generation blocks after the prompt, whereas upstream aligns
them to absolute positions. Use aligned prompts to isolate numerical
differences between stacks, then evaluate quality on unaligned prompts.
Matching decoding logic can still produce different tokens across kernel
versions.

## Check the comparison tool on CPU

```bash
python -m unittest discover -s test/manual/dllm -p 'test_fast_dvlm_validation.py'
```
