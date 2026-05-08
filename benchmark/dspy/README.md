## Install

```
pip3 install dspy-ai
```

Turn off cache at https://github.com/stanfordnlp/dspy/blob/34d8420383ec752037aa271825c1d3bf391e1277/dsp/modules/cache_utils.py#L10.
```
cache_turn_on = False
```

or set the environment variable

```
export DSP_CACHEBOOL=false
```

## Benchmark SGLang
```
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
```

```
python3 bench_dspy_intro.py --backend sglang
```


## Benchmark TGI
```
docker run --name tgi --rm -ti --gpus all --network host \
  -v /home/ubuntu/model_weights/Llama-2-7b-chat-hf:/Llama-2-7b-chat-hf \
  ghcr.io/huggingface/text-generation-inference:1.3.0 \
  --model-id /Llama-2-7b-chat-hf --num-shard 1  --trust-remote-code \
  --max-input-length 2048 --max-total-tokens 4096 \
  --port 24000
```

```
python3 bench_dspy_intro.py --backend tgi
```



## Benchmark vLLM
```
python3 -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-chat-hf --disable-log-requests  --port 21000
```

```
python3 bench_dspy_intro.py --backend vllm
```
