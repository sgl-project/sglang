### Download MT-Bench data

```sh
wget -O question.jsonl https://raw.githubusercontent.com/lm-sys/FastChat/d04ce6453ae016d9e03626b679c07aa1388dcbee/fastchat/llm_judge/data/mt_bench/question.jsonl
```

### Benchmark Context QA

```sh
python3 bench_sglang.py --mode contextqa
```

### Benchmark MT-Bench

```sh
python3 bench_sglang.py --mode mtbench
```