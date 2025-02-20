# Usage for Real World Dataset Benchmarks
## Download Dataset
```bash
./download.sh {sharegpt|ultragpt|loogle|nextqa|all}
```
This script will automatically download the required dataset to the current working directory

## Multiturn Benchmark
### Supported Datasets
- sharegpt
- ultrachat
- loogle
- nextqa (WIP)
### Example Usage:
```bash
python3 bench_serving.py --model mistralai/Mistral-7B-Instruct-v0.3 --backend sglang \
--dataset-path longdep_qa.json --dataset-name loogle --request-rate 10 --num-prompts 10  \
--port 8001 --enable-multiturn --disable-shuffle
```
This uses `mistralai/Mistral-7B-Instruct-v0.3` model with `sglang` as backend. The dataset
is `longdep_qa.json`. We send `10 conversations` with `10 req/s` to port 8001. We enable
multiturn chat without shuffling the order of conversations (i.e. following the original
order in the dataset file).

### Note:
The requests of multiple conversations are sent in a round robin fashion.
For example, if we have 3 conversations A, B, C whose rounds are `[2, 3, 4]` correspondingly,
multiturn chat will send the requests to the backend in the following order: `[A1, B1, C1, A2, B2, C2, B3, C3, C4]`
This has implications on the cache reuse patterns: the cache reuse distance is the largest
under this request pattern (which means a prefix-aware local scheduler in the backend can
yield the most benefit compared to a FIFO scheduler)

## Shared Prefix Benchmark
### Supported Datasets
- loogle
- nextqa (WIP)
### Example Usage:
```bash
python3 bench_serving.py --model mistralai/Mistral-7B-Instruct-v0.3 --backend sglang \
--dataset-path longdep_qa.json --dataset-name loogle --request-rate 10 --num-prompts 10  \
--port 8001 --enable-shared-prefix --disable-shuffle
```
### Note:
Shared Prefix benchmark sends the questions for the same prompt together. For example,
if we have 3 shared prefix A, B, C, which have [2, 3, 4] questions correspondingly,
the shared prefix benchmark will send the requests to the
backend in the following order: `[A+Q1, A+Q2, B+Q1, B+Q2, B+Q3, C+Q1, C+Q2, C+Q3]`.


## Multi Modality Benchmark (WIP)

## Supported Backend
- sglang
- sglang-native
- sglang-oai
- vllm (oai)
- lmdeploy (oai)
