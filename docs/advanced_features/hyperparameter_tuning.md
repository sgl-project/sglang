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
If you see `KV cache pool is full. Retract requests.` occasionally but not frequently (~1 time per minute), it is okay.

### Tune `--mem-fraction-static` to increase KV cache pool capacity
SGLang allocates memory as follows:

Total memory usage = model weights + KV cache pool + CUDA graph buffers + activations

The `--mem-fraction-static` parameter determines how much memory is allocated to the first two components:

mem_fraction_static = (model weights + KV cache pool) / GPU memory capacity

To support higher concurrency, you should maximize the KV cache pool capacity by setting `--mem-fraction-static` as high as possible while still reserving enough memory for activations and CUDA graph buffers.

SGLang uses simple heuristics to set the default value of `--mem-fraction-static`, but you can optimize it for your use cases.
As a rule of thumb, reserving 5–8 GB of memory for activations is typically sufficient. You can check this by inspecting the logs just before the server is ready.
Look for log entries like this:

```
[2025-08-11 17:17:03] max_total_num_tokens=665690, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=4096, context_len=65536, available_gpu_mem=13.50 GB
```

Check the `available_gpu_mem` value.
- If it is between 5–8 GB, the setting is good.
- If it is too high (e.g., 10 - 20 GB), increase `--mem-fraction-static` to allocate more memory to the KV cache.
- If it is too low, you risk out-of-memory (OOM) errors later, so decrease `--mem-fraction-static`.

Another straightforward approach is to increase `--mem-fraction-static` in increments of 0.01 until you encounter OOM errors for your workloads.

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

Data parallelism is better for throughput. When there is enough GPU memory, always favor data parallelism for throughput. Refer to [sglang router](../advanced_features/router.md) for a better data parallelism rather than using `dp_size` parameter.

### Try other options

- `torch.compile` accelerates small models on small batch sizes. You can enable it with `--enable-torch-compile`.
- Try other quantization (e.g. FP8 quantization with `--quantization fp8`)
- Try other parallelism strategies (e.g. [expert parallelism](https://lmsys.org/blog/2025-05-05-large-scale-ep/)) or DP attention for deepseek models (with `--enable-dp-attention --dp-size 8`).
- If the workload has many shared prefixes, try `--schedule-policy lpm`. Here, `lpm` stands for longest prefix match. It reorders requests to encourage more cache hits but introduces more scheduling overhead.
