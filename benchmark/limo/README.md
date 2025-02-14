## Run benchmark

This benchmark is primarily intended to be used with reasoning models like `DeepSeek-R1-Distill-Qwen-1.5B`.

### Benchmark sglang

Launch server

```bash
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --port 30000
```

Note that depending on the GPU this benchmark will take quiet some time. To employ data parallelism please use:

```bash
python3 -m sglang_router.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --port 30000 --dp-size 4
```

Benchmark

We use [suggested](https://github.com/deepseek-ai/DeepSeek-R1) parameters of `temperature=0.6`, `top_p=.95`, `max_new_tokens=32768`. The command line argument `num-tries` can be used to evaluate the model multiple times on the same question. We use the suggested `64` from the repo.

By default evaluate on LIMO dataset.
```bash
python3 bench_sglang.py --parallel 256 --num-tries 64
```

Evaluate on AIME 2024 dataset.
```bash
python3 bench_sglang.py --parallel 256 --port 33333 --data-path Maxwell-Jia/AIME_2024 --question-key Problem --answer-key Answer --num-tries 64
```
Accuracy: 32.2% that is more than the reported 28.9% in the paper.