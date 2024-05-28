# Guide on Hyperparameter Tuning

## Achieving Peak Throughput
Achieving a large batch size is the most important factor for attaining high throughput.

When the server is running at full load, look for the following in the log:
```[gpu_id=0] #running-req: 233, #token: 370959, token usage: 0.82, gen throughput (token/s): 4594.01, #queue-req: 417```

### Tune Your Request Submission Speed
`#queue-req` indicates the number of requests in the queue. If you frequently see `#queue-req == 0`, it suggests you are bottlenecked by the request submission speed.
A healthy range for `#queue-req` is `100 - 3000`.

### Tune `--schedule-conservativeness`
`token usage` indicates the KV cache memory utilization of the server. `token usage > 0.9` means good utilization.
If you frequently see `token usage < 0.9` and `#queue-req > 0`, it means the server is too conservative about taking in new requests. You can decrease `--schedule-conservativeness` to a value like 0.5.

### Tune `--dp-size` and `--tp-size`
Data parallelism is better for throughput. When there is enough GPU memory, always favor data parallelism for throughput.

### (Minor) Tune `--schedule-heuristic`
If you have many shared prefixes, use the default `--schedule-heuristic lpm`. `lmp` stands for longest prefix match.
When you have no shared prefixes at all, you can try `--schedule-heuristic fcfs`. `fsfc` stands for first come first serve.