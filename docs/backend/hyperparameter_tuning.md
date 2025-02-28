# Hyperparameter Tuning

## Achieving Peak Throughput
Achieving a large batch size is the most important thing for attaining high throughput.

When the server is running at full load, look for the following in the log:

```Decode batch. #running-req: 233, #token: 370959, token usage: 0.82, gen throughput (token/s): 4594.01, #queue-req: 317```

### Tune Your Request Submission Speed
`#queue-req` indicates the number of requests in the queue. If you frequently see `#queue-req == 0`, it suggests you are bottlenecked by the request submission speed.
A healthy range for `#queue-req` is `50 - 500`.
On the other hand, do not make `#queue-req` too large because it will also increase the scheduling overhead on the server, especially when using the default longest-prefix-match schedule policy (`--schedule-policy lpm`).

### Tune `--schedule-conservativeness`
`token usage` indicates the KV cache memory utilization of the server. `token usage > 0.9` means good utilization.
If you frequently see `token usage < 0.9` and `#queue-req > 0`, it means the server is too conservative about taking in new requests. You can decrease `--schedule-conservativeness` to a value like 0.3.
The case of server being too conservative can happen when users send many requests with a large `max_new_tokens` but the requests stop very early due to EOS or stop strings.

On the other hand, if you see `token usage` very high and you frequently see warnings like
`decode out of memory happened, #retracted_reqs: 1, #new_token_ratio: 0.9998 -> 1.0000`, you can increase `--schedule-conservativeness` to a value like 1.3.
If you see `decode out of memory happened` occasionally but not frequently, it is okay.

### Tune `--dp-size` and `--tp-size`
Data parallelism is better for throughput. When there is enough GPU memory, always favor data parallelism for throughput.

### Avoid out-of-memory by Tuning `--chunked-prefill-size`, `--mem-fraction-static`, `--max-running-requests`
If you see out of memory (OOM) errors, you can try to tune the following parameters.
- If OOM happens during prefill, try to decrease `--chunked-prefill-size` to `4096` or `2048`.
- If OOM happens during decoding, try to decrease `--max-running-requests`.
- You can also try to decrease `--mem-fraction-static`, which reduces the memory usage of the KV cache memory pool and helps both prefill and decoding.

### Try Advanced Options
- To enable torch.compile acceleration, add `--enable-torch-compile`. It accelerates small models on small batch sizes. This does not work for FP8 currently.

### Tune `--schedule-policy`
If the workload has many shared prefixes, use the default `--schedule-policy lpm`. `lpm` stands for longest prefix match.
When you have no shared prefixes at all or you always send the requests with the shared prefixes together,
you can try `--schedule-policy fcfs`. `fcfs` stands for first come first serve. `fcfs` has a lower scheduling overhead.
