# Hyperparameter Tuning

## Achieving high throughput for offline batch inference

Achieving a large batch size is the most important thing for attaining high throughput in offline batch inference.
When the server is running at full load in a steady state, look for the following in the log:

```Decode batch. #running-req: 233, #token: 370959, token usage: 0.82, cuda graph: True, gen throughput (token/s): 4594.01, #queue-req: 317```

### Adjust the request submission speed to control `#queue-req`

`#queue-req` indicates the number of requests in the queue.
If you frequently see `#queue-req: 0`, it suggests that your client code is submitting requests too slowly.
A healthy range for `#queue-req` is `100 - 2000`.
However, avoid making `#queue-req` too large, as this will increase the scheduling overhead on the server.

### Achieve a high `token usage`

`token usage` indicates the KV cache memory utilization of the server. `token usage > 0.9` means good utilization.

If you frequently see `token usage < 0.9` and `#queue-req > 0`, it means the server is too conservative about taking in new requests. You can decrease `--schedule-conservativeness` to a value like 0.3.
The case of a server being too conservative can happen when users send many requests with a large `max_new_tokens` but the requests stop very early due to EOS or stop strings.

On the other hand, if you see `token usage` very high and you frequently see warnings like
`KV cache pool is full. Retract requests. #retracted_reqs: 1, #new_token_ratio: 0.9998 -> 1.0000`, you can increase `--schedule-conservativeness` to a value like 1.3.
If you see `KV cache pool is full. Retract requests.` occasionally but not frequently, it is okay.

### Tune `--mem-fraction-static` to increase the KV cache pool capacity
GPU memory capacity = model weights + KV cache pool + activations + CUDA graph buffers

mem_fraction_static = (model weights + KV cache pool) / GPU memory capacity.

We want to increase the KV cache pool capacity to support a larger concurrency, so
we want `--mem-fraction-static` to be as large as possible but still have enough room
for activations and CUDA graph buffers.

A simple strategy is to increase `--mem-fraction-static` by 0.01 each time until you encounter out-of-memory errors.

### Avoid out-of-memory errors by tuning `--chunked-prefill-size`, `--mem-fraction-static`, and `--max-running-requests`

If you encounter out-of-memory (OOM) errors, you can adjust the following parameters:

- If OOM occurs during prefill, try reducing `--chunked-prefill-size` to `4096` or `2048`. This saves memory but slows down the prefill speed for long prompts.
- If OOM occurs during decoding, try lowering `--max-running-requests`.
- You can also reduce `--mem-fraction-static` to a smaller value, such as 0.8 or 0.7. This decreases the memory usage of the KV cache memory pool and helps prevent OOM errors during both prefill and decoding. However, it limits maximum concurrency and reduces peak throughput.

### Tune `--cuda-graph-max-bs`
By default, CUDA graph is enabled only for small batch sizes (e.g., less than 160 or 256).
However, for some models, especially at large tensor parallelism sizes, CUDA graph can be useful for batch sizes up to 512 or 768.
Therefore, it may be beneficial to increase `--cuda-graph-max-bs` to a larger value.
Note that CUDA graph consumes more memory, so you may need to reduce `--mem-fraction-static` at the same time.

### Tune `--dp-size` and `--tp-size`

Data parallelism is better for throughput. When there is enough GPU memory, always favor data parallelism for throughput. Refer to [sglang router](../router/router.md) for a better data parallelism rather than using `dp_size` parameter.

### Try other options

- `torch.compile` accelerates small models on small batch sizes. You can enable it with `--enable-torch-compile`.
- Try other quantization (e.g. FP8 quantization with `--quantization fp8`)
- Try other parallelism strategies (e.g. expert parallelism) or DP attention for deepseek models (with `--enable-dp-attention --dp-size 8`).
- If the workload has many shared prefixes, try `--schedule-policy lpm`. Here, `lpm` stands for longest prefix match. It reorders requests to encourage more cache hits but introduces more scheduling overhead.
