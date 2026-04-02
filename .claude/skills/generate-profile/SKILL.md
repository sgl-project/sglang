---
name: generate-profile
description: Generate an e2e profiling trace of an SGLang server run. Launches a server, validates accuracy, captures a Chrome-compatible trace, and returns the profile path.
---

# Generate an E2E Profile of an SGLang Server Run

This skill launches an SGLang server, validates it with a quick accuracy test, generates a profiling trace, and returns the profile file path.

## Prerequisites

- A working SGLang installation (`pip install -e .` or equivalent)
- At least one available CUDA GPU

## Step-by-step Workflow

### Step 1: Launch the server

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> sglang serve --model-path <model> --port <port> &
```

- Default model: `Qwen/Qwen3-8B` (good balance of speed and quality)
- Default port: `30000`
- The server runs in the background. Save the PID for cleanup.
- Use the GPU specified by the user's preferences (check memory files for GPU preferences).

### Step 2: Wait for server readiness

Poll the health endpoint until the server is ready:

```bash
for i in $(seq 1 120); do
  if curl -s http://127.0.0.1:<port>/health 2>/dev/null | grep -q "ok\|healthy"; then
    echo "Server ready"
    break
  fi
  sleep 5
done
```

The server prints **"The server is fired up and ready to roll!"** to stdout when ready. The health endpoint returns 200 once the server can accept requests.

Typical startup time: 30-90 seconds depending on model size and whether CUDA graphs are being compiled.

### Step 3: Validate accuracy (sanity check)

```bash
python3 -m sglang.test.few_shot_gsm8k --num-q 20
```

- Expected accuracy: **> 0.8** for capable models (Qwen3-8B, Llama-3.1-8B-Instruct, etc.)
- This is a quick sanity check, not a rigorous benchmark.
- If accuracy is unexpectedly low, something is wrong — do not proceed to profiling.

### Step 4: Generate the profile

```bash
python3 -m sglang.test.send_one --profile
```

This command:
1. Sends a request to the server
2. Triggers the profiler for 5 steps (default)
3. Generates a trace file under `/tmp/<timestamp>/`
4. The trace directory contains:
   - `<timestamp>-TP-0.trace.json.gz` — Chrome trace format (open in `chrome://tracing` or Perfetto)
   - `server_args.json` — the server configuration used

**Output format:**
```
Dump profiling traces to /tmp/<timestamp>
```

The profile path is printed to stdout. Parse it from the output.

**Optional flags:**
- `--profile-steps N` — number of profiling steps (default: 5)
- `--profile-by-stage` — profile by stage (prefill/decode separately)
- `--profile-prefix <path>` — custom output prefix

### Step 5: Kill the server

```bash
pkill -9 -f "sglang.launch_server\|sglang serve\|sglang.srt"
```

Wait a moment and verify no sglang processes remain:
```bash
sleep 2 && pgrep -af "sglang serve" || echo "Server killed"
```

### Step 6: Report the profile path

Return the profile directory path (e.g., `/tmp/1773999986.4769795`) and list its contents so the user knows what files were generated.

## Example Full Run

```bash
# 1. Launch server
source cleanup/bin/activate
CUDA_VISIBLE_DEVICES=1 sglang serve --model-path Qwen/Qwen3-8B --port 30000 &

# 2. Wait for ready
for i in $(seq 1 120); do
  curl -s http://127.0.0.1:30000/health | grep -q "ok" && break
  sleep 5
done

# 3. Accuracy check
python3 -m sglang.test.few_shot_gsm8k --num-q 20
# Expected: Accuracy > 0.8

# 4. Profile
python3 -m sglang.test.send_one --profile
# Output: "Dump profiling traces to /tmp/1773999986.4769795"

# 5. Cleanup
pkill -9 -f "sglang.launch_server\|sglang serve\|sglang.srt"
sleep 2

# 6. Check output
ls -la /tmp/1773999986.4769795/
# 1773999986.4851577-TP-0.trace.json.gz  (Chrome trace)
# server_args.json                        (server config)
```

## Customization

- **Different port**: Pass `--port <port>` and use `--host 127.0.0.1 --port <port>` for test commands
- **Multi-GPU**: Use `--tp <N>` for tensor parallelism; trace files will be generated per TP rank
- **Longer profile**: Use `--profile-steps 10` for more steps in the trace
- **Stage profiling**: Use `--profile-by-stage` to separate prefill and decode phases

## Viewing the Profile

Open the `.trace.json.gz` file in:
- **Perfetto UI**: https://ui.perfetto.dev/ (drag and drop the file)
- **Chrome tracing**: `chrome://tracing` (load the file)

Both support the gzipped Chrome trace format natively.
