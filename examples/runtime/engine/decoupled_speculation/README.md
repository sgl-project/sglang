# Decoupled Speculation Example

This directory has two CLI entrypoints for decoupled speculative decoding:

- `multi-node.py`: run the Ray-backed multi-node benchmark from either `--prompt` or `--dataset-path`.
- `single-node.py`: run the local-process single-node benchmark without Ray.
- `common/`: shared helpers split by function: runtime/Ray topology, prompt loading, metrics/output, and shared types.

Decoupled-spec engines use static bind/connect endpoint configuration. Each
verifier or drafter instance receives one local bind endpoint, an ordered list
of peer connect endpoints, and a role-local rank. The helper builds a full mesh:
every verifier connects to every drafter control endpoint, and every drafter
connects to every verifier result endpoint. The script first chooses verifier
placement groups and drafter nodes, then reserves bind endpoints on those nodes,
and finally launches engines with the completed endpoint configs. When multiple
verifier replicas are used, `--batch-size` must be divisible by the verifier
replica count; each verifier receives one equal contiguous slice of the batch.

Common modes:

```bash
# Single prompt, compare decoupled speculation against normal decode.
python examples/runtime/engine/decoupled_speculation/multi-node.py \
  --prompt "Write a short haiku about distributed systems." \
  --target-model-path Qwen/Qwen3-32B \
  --draft-model-path Qwen/Qwen3-0.6B \
  --target-tp-size 4 \
  --draft-tp-size 1 \
  --max-new-tokens 128

# Dataset batch, decoupled speculation only.
python examples/runtime/engine/decoupled_speculation/multi-node.py \
  --dataset-path /path/to/prompts.parquet \
  --batch-size 16 \
  --skip-decode \
  --target-model-path /path/to/target \
  --draft-model-path /path/to/draft \
  --target-tp-size 8 \
  --draft-tp-size 1 \
  --max-new-tokens 1024

# Multi-node mesh. Here 16 GPUs are reserved for verifier replicas and 4 GPUs
# are reserved for drafter replicas; the remaining GPUs stay idle.
python examples/runtime/engine/decoupled_speculation/multi-node.py \
  --dataset-path /path/to/prompts.parquet \
  --batch-size 64 \
  --skip-decode \
  --target-model-path /path/to/target \
  --draft-model-path /path/to/draft \
  --nnodes 4 \
  --n-gpu-per-node 8 \
  --target-tp-size 8 \
  --draft-tp-size 1 \
  --verify-ngpus 16 \
  --draft-ngpus 4 \
  --dist-init-port 30000 \
  --max-new-tokens 1024

# Print responses and write per-mode CSV/JSON outputs.
python examples/runtime/engine/decoupled_speculation/multi-node.py \
  --prompt "Explain speculative decoding." \
  --show-responses \
  --output-dir ./decoupled_spec_outputs \
  --target-model-path /path/to/target \
  --draft-model-path /path/to/draft \
  --target-tp-size 4 \
  --draft-tp-size 1 \
  --max-new-tokens 256
```
