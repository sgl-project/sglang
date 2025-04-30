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

Data parallelism is better for throughput. When there is enough GPU memory, always favor data parallelism for throughput. Refer to [sglang router](../router/router.md) for a better data parallelism rather than using `dp_size` parameter.

## Avoid out-of-memory by Tuning `--chunked-prefill-size`, `--mem-fraction-static`, `--max-running-requests`

If you see out of memory (OOM) errors, you can try to tune the following parameters.

- If OOM happens during prefill, try to decrease `--chunked-prefill-size` to `4096` or `2048`.
- If OOM happens during decoding, try to decrease `--max-running-requests`.
- You can also try to decrease `--mem-fraction-static`, which reduces the memory usage of the KV cache memory pool and helps both prefill and decoding.

## Enabling cache for `torch.compile`

To enable `torch.compile` acceleration, add `--enable-torch-compile`. It accelerates small models on small batch sizes. By default, `torch.compile` will automatically cache the FX graph and Triton in `/tmp/torchinductor_root`, which might be cleared according to the [system policy](https://serverfault.com/questions/377348/when-does-tmp-get-cleared). You can export the environment variable `TORCHINDUCTOR_CACHE_DIR` to save compilation cache in your desired directory to avoid unwanted deletion. You can also share the cache with other machines to reduce the compilation time.

SGLang uses `max-autotune-no-cudagraphs` mode of `torch.compile`. The auto-tuning can be slow.
If you want to deploy a model on many different machines, you can ship the `torch.compile` cache to these machines and skip the compilation steps. This is based on [PyTorch official documentation](https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html).

*Examples*：

1. Generate the cache by setting `TORCHINDUCTOR_CACHE_DIR` and running the model once.

   ```bash
   TORCHINDUCTOR_CACHE_DIR=/root/inductor_root_cache python3 -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct --enable-torch-compile
   ```

2. Copy the cache folder to other machines and launch the server with `TORCHINDUCTOR_CACHE_DIR`.

## Tune `--schedule-policy`

If the workload has many shared prefixes, use the default `--schedule-policy lpm`. Where `lpm` stands for longest prefix match.

When you have no shared prefixes at all or you always send the requests with the shared prefixes together,
you can try `--schedule-policy fcfs`. Where `fcfs` stands for first come first serve. This policy has a lower scheduling overhead.
