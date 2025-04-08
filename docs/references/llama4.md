# Llama4 Usage

[Llama 4](https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md) is Meta's latest generation of open-source LLM model with industry-leading performance.

SGLang has supported Llama 4 Scout (109B) and Llama 4 Maverick (400B) since [v0.4.5](https://github.com/sgl-project/sglang/releases/tag/v0.4.5).

Ongoing optimizations are tracked in the [Roadmap](https://github.com/sgl-project/sglang/issues/5118).

## Launch Llama 4 with SGLang

To serve Llama 4 models on 8xH100/H200 GPUs:

```bash
python3 -m sglang.launch_server --model-path meta-llama/Llama-4-Scout-17B-16E-Instruct --tp 8 --context-length 1000000
```

### Configuration Tips

- **OOM Mitigation**: Adjust `--context-length` to avoid a GPU out-of-memory issue. For the Scout model, we recommend setting this value up to 1M on 8\*H100 and up to 2.5M on 8\*H200. For the Maverick model, we don't need to set context length on 8\*H200.

- **Chat Template**: Add `--chat-template llama-4` for chat completion tasks.

## Benchmarking Results

### Accuracy Test with `lm_eval`

The accuracy on SGLang for both Llama4 Scout and Llama4 Maverick can match the [official benchmark numbers](https://ai.meta.com/blog/llama-4-multimodal-intelligence/).

Benchmark results on MMLU Pro dataset with 8*H100:
|                    | Llama-4-Scout-17B-16E-Instruct | Llama-4-Maverick-17B-128E-Instruct  |
|--------------------|--------------------------------|-------------------------------------|
| Official Benchmark | 74.3                           | 80.5                                |
| SGLang             | 75.2                           | 80.7                                |

Commands:

```bash
# Llama-4-Scout-17B-16E-Instruct model
python -m sglang.launch_server --model-path meta-llama/Llama-4-Scout-17B-16E-Instruct --port 30000 --tp 8 --mem-fraction-static 0.8 --context-length 65536
lm_eval --model local-chat-completions --model_args model=meta-llama/Llama-4-Scout-17B-16E-Instruct,base_url=http://localhost:30000/v1/chat/completions,num_concurrent=128,timeout=999999,max_gen_toks=2048 --tasks mmlu_pro --batch_size 128 --apply_chat_template --num_fewshot 0

# Llama-4-Maverick-17B-128E-Instruct
python -m sglang.launch_server --model-path meta-llama/Llama-4-Maverick-17B-128E-Instruct --port 30000 --tp 8 --mem-fraction-static 0.8 --context-length 65536
lm_eval --model local-chat-completions --model_args model=meta-llama/Llama-4-Maverick-17B-128E-Instruct,base_url=http://localhost:30000/v1/chat/completions,num_concurrent=128,timeout=999999,max_gen_toks=2048 --tasks mmlu_pro --batch_size 128 --apply_chat_template --num_fewshot 0
```

Details can be seen in [this PR](https://github.com/sgl-project/sglang/pull/5092).
