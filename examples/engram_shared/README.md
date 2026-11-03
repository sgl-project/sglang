# SGLang P/D with an external Engram owner

This example uses the owner protocol from
[vLLM #64082](https://github.com/vllm-project/vllm/pull/64082). Install that
draft in a separate vLLM environment and set `VLLM_PYTHON` to its Python
executable. SGLang workers import the same immutable generation but keep
their native NIXL P/D KV transfer and router. This four-GPU recipe uses P TP2
on GPUs 0/1 and D TP2 on GPUs 2/3.

```bash
export MODEL_DIR=/models/DeepSeek-V4.1-Flash
export TOKEN_FILE=/shared/private/engram.token
export OWNER_ENDPOINT=127.0.0.1:56000
export VLLM_PYTHON=/path/to/vllm/.venv/bin/python
export SGLANG_PYTHON=/path/to/sglang/.venv/bin/python
umask 077
mkdir -p "$(dirname "$TOKEN_FILE")"
openssl rand -hex 32 > "$TOKEN_FILE"
```

Run `./launch.sh owner` and wait for `ENGRAM_OWNER_READY`. Copy its
`generation=` value to `GENERATION` in the P and D environments. Then run
`./launch.sh prefill`, `./launch.sh decode`, and `./launch.sh router` in
separate terminals. Once all three services are ready, use
`$SGLANG_PYTHON request.py` to send a request through the router.

Every process needs the same checkpoint identity. The owner and consumers
need access to the private token file. Stop router, D, and P by sending
SIGTERM to each launcher PID, not the entire process group. Wait for all
detach events before stopping the owner. A process-group signal to the
SGLang workers caused uncertain leases in the local test. For different
hosts, set routable owner and P/D URLs, adjust GPU IDs and ports, and use
one IMEX/NVLink domain. Cross-host operation has not been validated yet.
