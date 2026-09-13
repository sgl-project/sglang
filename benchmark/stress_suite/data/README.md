# Synthetic trace fixture

`synthetic_trace.jsonl` is hand-authored test data in the existing custom
dataset format. Every prompt and answer is fictional and exists only to exercise
the benchmark harness.

Do not add production requests, transformed production requests, customer data,
or private dataset metadata to this directory. Private traces must remain outside
the repository and be passed to the benchmark by local path.
