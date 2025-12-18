# Async Disaggregated Execution - Usage Guide

## Quick Start

### 1. Enable Disaggregated Mode

```bash
# Launch server with disaggregated execution
python -m sglang.multimodal_gen.launch_server \
    --model-path /path/to/model \
    --enable-disagg \
    --num-gpus 4 \
    --num-non-dit-ranks 1
```

This configuration will:
- Use 4 GPUs total
- Assign 1 GPU to Non-DiT (Encoder/VAE)
- Assign 3 GPUs to DiT (Denoising)
- Enable async communication automatically

### 2. Verify Async Communication is Working

Check the logs for these messages:

```
[INFO] DisaggregatedExecutor initialized on rank 0 with role dit, max_pending=2
[INFO] DisaggregatedExecutor initialized on rank 3 with role non_dit, max_pending=2
[DEBUG] [Async Send to DiT] Initiated 5 async sends (2 single tensors, 1 tensor lists)
[DEBUG] [Async Recv from Non-DiT] Initiated 5 async recvs (2 single tensors, 1 tensor lists)
```

### 3. Run Test Script

```bash
# Basic communication test (requires 4 GPUs)
torchrun --nproc_per_node=4 \
    python/sglang/multimodal_gen/runtime/communication/test_async_basic.py
```

Expected output:
```
============================================================
Running Async Communication Tests
Rank: 0/4
============================================================

--- Test: Basic Async Send/Recv ---
[Rank 0] Starting basic async test
[Rank 0] Sending tensor with sum=123.4567
[Rank 0] Async send initiated
[Rank 0] Did some computation: 456.7890
[Rank 0] Send completed
[Rank 0] Basic async test passed!

--- Test: Disagg Topology ---
...

============================================================
Test Summary:
  ✓ PASSED: Basic Async Send/Recv
  ✓ PASSED: Disagg Topology
  ✓ PASSED: Async Batch Transfer
============================================================

All tests passed! ✓
```

## Configuration Options

### Server Arguments

```python
from sglang.multimodal_gen.runtime.server_args import ServerArgs

server_args = ServerArgs(
    model_path="/path/to/model",

    # Disaggregation settings
    enable_disagg=True,           # Enable disaggregated execution
    num_gpus=4,                   # Total number of GPUs
    num_non_dit_ranks=1,          # Number of Non-DiT ranks (Encoder/VAE)

    # Parallelism (applied to DiT ranks only in disagg mode)
    sp_degree=2,                  # Sequence parallelism degree
    tp_size=1,                    # Tensor parallelism size

    # Other settings
    attention_backend="fa",       # Flash attention
    enable_torch_compile=False,   # Torch compile (may cause issues with async)
)
```

### Executor Settings

In `disaggregated_executor.py`:

```python
class DisaggregatedExecutor(PipelineExecutor):
    def __init__(self, server_args: ServerArgs):
        super().__init__(server_args)
        # ...

        # Adjust this to control memory vs throughput tradeoff
        self.max_pending_transfers = 2  # Max in-flight async transfers
```

**Tuning `max_pending_transfers`:**
- **Lower (1)**: Less memory usage, more synchronization overhead
- **Higher (3-4)**: More memory usage, better throughput
- **Recommended**: 2 (good balance)

## Performance Monitoring

### Enable Detailed Logging

```python
import logging

# Enable debug logging for communication
logging.getLogger("sglang.multimodal_gen.runtime.communication").setLevel(logging.DEBUG)

# Enable debug logging for executor
logging.getLogger("sglang.multimodal_gen.runtime.pipelines_core.executors").setLevel(logging.DEBUG)
```

### Key Metrics to Monitor

1. **Latency per Request**
   - Look for: `[INFO] Request completed in X.XX seconds`
   - Expected improvement: 30-50% reduction vs sync

2. **Throughput (Requests/Second)**
   - Measure: Total requests / Total time
   - Expected improvement: 40-60% increase vs sync

3. **Memory Usage**
   - Monitor: `nvidia-smi` during execution
   - Should be bounded by `max_pending_transfers`

4. **Communication Overhead**
   - Look for: `[DEBUG] Waited for send completion: batch_X`
   - If frequent: Consider increasing `max_pending_transfers`

## Troubleshooting

### Issue 1: Deadlock

**Symptoms:**
- Process hangs indefinitely
- No progress in logs

**Causes:**
- Mismatch in send/recv pairs
- Missing barrier or broadcast

**Solution:**
```bash
# Enable NCCL debug logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Run with timeout
timeout 60s torchrun --nproc_per_node=4 your_script.py
```

### Issue 2: Memory Leak

**Symptoms:**
- GPU memory grows over time
- OOM errors after many requests

**Causes:**
- `PendingTransfer` objects not cleaned up
- Tensors not released after communication

**Solution:**
```python
# Check pending queues are being drained
logger.debug(f"Pending sends: {len(self.pending_sends)}")
logger.debug(f"Pending recvs: {len(self.pending_recvs)}")

# Ensure all works are waited on
while self.pending_sends:
    pending = self.pending_sends.popleft()
    self.comm.wait_all_works(pending.works)
```

### Issue 3: Data Corruption

**Symptoms:**
- Output images/videos are incorrect
- Numerical errors in results

**Causes:**
- Tensor modified while communication in flight
- Incorrect tensor shapes/dtypes

**Solution:**
```python
# Verify tensor is not modified after send
work = self.comm.isend_to_dit(tensor)
# DO NOT modify tensor here!
work.wait()
# Now safe to modify

# Verify shapes match
logger.debug(f"Sending tensor: shape={tensor.shape}, dtype={tensor.dtype}")
```

### Issue 4: Performance Regression

**Symptoms:**
- Async is slower than sync
- High communication overhead

**Causes:**
- `max_pending_transfers` too low
- Frequent synchronization points

**Solution:**
```python
# Increase max pending transfers
self.max_pending_transfers = 3  # or 4

# Reduce synchronization points
# - Batch multiple sends together
# - Avoid unnecessary barriers
```

## Best Practices

### 1. Tensor Management

```python
# ✓ GOOD: Keep tensor alive until communication completes
tensor = batch.latents
work = comm.isend_to_dit(tensor)
pending = PendingTransfer(
    batch_id=batch.request_id,
    works=[work],
    tensors={"latents": tensor}  # Keep reference
)
self.pending_sends.append(pending)

# ✗ BAD: Tensor may be garbage collected
work = comm.isend_to_dit(batch.latents)
# batch.latents may be GC'd before send completes!
```

### 2. Error Handling

```python
try:
    works = self._async_send_batch_to_dit(batch)
    # ... do other work ...
    self.comm.wait_all_works(works)
except Exception as e:
    logger.error(f"Communication failed: {e}")
    # Clean up pending transfers
    self.pending_sends.clear()
    raise
```

### 3. Flow Control

```python
# Implement adaptive flow control
if len(self.pending_sends) >= self.max_pending_transfers:
    # Wait for oldest send
    oldest = self.pending_sends.popleft()
    self.comm.wait_all_works(oldest.works)

    # Optional: Adjust max_pending based on memory
    if gpu_memory_usage > 0.9:
        self.max_pending_transfers = max(1, self.max_pending_transfers - 1)
```

### 4. Testing

```python
# Test with single request first
def test_single_request():
    batch = create_test_batch()
    output = executor.execute(stages, batch, server_args)
    assert output.error is None
    assert output.output is not None

# Then test with multiple requests
def test_multiple_requests():
    for i in range(10):
        batch = create_test_batch(id=f"batch_{i}")
        output = executor.execute(stages, batch, server_args)
        assert output.error is None
```

## Migration from Sync to Async

If you have existing code using sync communication:

### Before (Sync)
```python
# Non-DiT rank
self._send_batch_to_dit(batch)  # Blocks until send completes

# DiT rank
batch = self._recv_batch_from_non_dit(batch)  # Blocks until recv completes
```

### After (Async)
```python
# Non-DiT rank
works = self._async_send_batch_to_dit(batch)  # Returns immediately
# ... can do other work here ...
self.comm.wait_all_works(works)  # Wait when needed

# DiT rank
batch, works = self._async_recv_batch_from_non_dit(batch)  # Returns immediately
self.comm.wait_all_works(works)  # Wait before using batch
```

**Key Changes:**
1. Methods return `Work` handles instead of blocking
2. Must explicitly wait before using received data
3. Can do other work between send/recv and wait

## FAQ

**Q: Does async work with all parallelism modes (SP, TP, DP)?**
A: Yes, async communication is orthogonal to parallelism. It works with SP, TP, and their combinations.

**Q: Can I disable async and use sync?**
A: Yes, the old sync methods are still available. Set `use_async_comm=False` in executor (not implemented yet, but easy to add).

**Q: What's the memory overhead of async?**
A: Approximately `max_pending_transfers * batch_size * tensor_sizes`. With default settings (~2 pending), overhead is ~2x a single batch.

**Q: Does async work with CPU offloading?**
A: Yes, but benefits are reduced since CPU offloading already overlaps computation and data movement.

**Q: Can I use async with multiple Non-DiT ranks?**
A: Current implementation assumes 1 Non-DiT rank. Multi-Non-DiT support requires additional coordination.

## Support

For issues or questions:
1. Check logs with `DEBUG` level enabled
2. Run `test_async_basic.py` to verify setup
3. Review `ASYNC_COMMUNICATION_DESIGN.md` for architecture details
4. Open an issue on GitHub with logs and configuration
