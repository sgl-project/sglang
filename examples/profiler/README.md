# SGLang Shape Profiler

Tool to profile tensor shapes during model inference. Supports both single-GPU and multi-GPU (TP) setups.

## Two Modes

### Mode 1: Standalone (TP=1 only)

Simple standalone profiler using PyTorch dispatch mode. **Only works with TP=1.**

```bash
python profile_shapes.py \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --tp-size 1 \
  --num-prompts 3 \
  --max-tokens 50
```

**Limitation:** With TP > 1, captures 0 operations (worker processes can't be intercepted).

### Mode 2: TP Worker Integration (Works with TP > 1)

Enables profiling inside TP workers via environment variables. Profiles operations on a specific GPU rank.

```bash
# Profile GPU 0 with TP=8
SGLANG_PROFILE_SHAPES=1 \
SGLANG_PROFILE_SHAPES_RANK=0 \
SGLANG_PROFILE_SHAPES_FILE=gpu0_shapes.jsonl \
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-14B-Instruct \
  --tp-size 8 \
  --port 30000
```

**Environment Variables:**
- `SGLANG_PROFILE_SHAPES=1` - Enable profiling
- `SGLANG_PROFILE_SHAPES_RANK=0` - Which GPU rank to profile (0-7 for TP=8)
- `SGLANG_PROFILE_SHAPES_FILE=shapes.jsonl` - Output file path

**Advantages:**
- ✅ Works with any TP size
- ✅ Captures actual tensor operations in workers
- ✅ Minimal overhead (only profiles one rank)
- ✅ No code modifications needed

**Example with client:**
```bash
# Terminal 1: Start server with profiling
SGLANG_PROFILE_SHAPES=1 SGLANG_PROFILE_SHAPES_RANK=0 \
SGLANG_PROFILE_SHAPES_FILE=shapes.jsonl \
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-14B-Instruct \
  --tp-size 8

# Terminal 2: Send requests
python -c "
import openai
client = openai.Client(base_url='http://localhost:30000/v1', api_key='none')
response = client.chat.completions.create(
    model='default',
    messages=[{'role': 'user', 'content': 'Hello!'}],
    max_tokens=50
)
print(response.choices[0].message.content)
"

# Profiling output will be in shapes.jsonl
```

## Which Mode to Use?

| Use Case | Mode | Command |
|----------|------|---------|
| Debug small model | Standalone (TP=1) | `python profile_shapes.py --tp-size 1` |
| Profile production setup | TP Worker (TP > 1) | `SGLANG_PROFILE_SHAPES=1 python -m sglang.launch_server --tp-size 8` |
| Quick testing | Standalone (TP=1) | `python profile_shapes.py --tp-size 1` |
| Real workload analysis | TP Worker (TP > 1) | Environment variables + launch_server |

## Analyzing Results

Both modes produce JSONL output files:

```bash
# Basic analysis
python analyze_shapes.py shapes.jsonl

# Detailed analysis with shape information
python analyze_shapes.py shapes.jsonl --show-shapes --top 20

# Filter specific operations
python analyze_shapes.py shapes.jsonl --filter-op "matmul"
```

## Files

- `profile_shapes.py` - Standalone profiler (TP=1 only)
- `torch_shape_logger.py` - Base logger implementation
- `torch_shape_logger_rank.py` - TP-aware logger for worker integration
- `analyze_shapes.py` - Analysis utilities

## Technical Details

**Standalone Mode:**
- Uses PyTorch dispatch mode in main process
- Can't intercept worker process operations
- Zero overhead when not used

**TP Worker Mode:**
- Hooks into tp_worker.py via environment variables
- Activates logger on first forward pass
- Only logs on specified rank to minimize overhead
- Automatic cleanup on shutdown
