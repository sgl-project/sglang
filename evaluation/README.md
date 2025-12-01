# Set Environment

1. Docker image:
   For MI30X:
   ```
   rocm/sgl-dev:v0.5.3.post1-rocm700-mi30x-20251011
   ```
   For MI35X:
   ```
   rocm/sgl-dev:v0.5.3.post1-rocm700-mi35x-20251011
   ```
2. Install aiter dev/perf branch:
   ```
   pip uninstall aiter
   git clone -b dev/perf git@github.com:ROCm/aiter.git
   cd aiter
   git submodule sync && git submodule update --init --recursive
   # for MI308
   PREBUILD_KERNELS=1 GPU_ARCHS=gfx942 python3 setup.py install
   # for MI355
   PREBUILD_KERNELS=1 GPU_ARCHS=gfx950 python3 setup.py install
   ```
4. Install zejun/sglang dev/perf branch:
   ```
   git clone -b dev/perf https://github.com/zejunchen-zejun/sglang.git
   cd sglang
   pip install --upgrade pip
   cd sgl-kernel
   pip uninstall sgl-kernel
   python setup_rocm.py install
   export PYTHONPATH=<you_sglang_path/sglang/python>
   ```

# Launch server
1. deepseek-r1 FP8/MXFP4
- download PTPC FP8 model weight: https://huggingface.co/EmbeddedLLM/deepseek-r1-FP8-Dynamic
    ```
    huggingface-cli download EmbeddedLLM/deepseek-r1-FP8-Dynamic --local-dir EmbeddedLLM/deepseek-r1-FP8-Dynamic
    ```
- download MXFP4 model weight: https://huggingface.co/amd/DeepSeek-R1-MXFP4-Preview
    '''
    huggingface-cli download amd/DeepSeek-R1-MXFP4-Preview --local-dir amd/DeepSeek-R1-MXFP4-Preview
    '''
- launch server:
    ```
    bash launch_deepseekr1_fp4.sh
    ```
    The example command:
    ```
    model=/data/pretrained-models/DeepSeek-R1-MXFP4-Preview
    TP=8
    EP=1
    python3 -m sglang.launch_server \
        --model-path ${model} \
        --host localhost \
        --port 9000 \
        --tp-size ${TP} \
        --ep-size ${EP} \
        --trust-remote-code \
        --chunked-prefill-size 196608 \
        --mem-fraction-static 0.9 \
        --disable-radix-cache \
        --max-prefill-tokens 196608 \
        --cuda-graph-max-bs 128 \
        2>&1 | tee log.server.log &
    ```

    Or if you want to use the PTPC FP8 model, you can change the `model` argument. Due to w8a8fp8 kernel scale limitations, the `chunked-prefill-size` and `max-prefill-tokens` argument must be restricted to 32k in the command like this:
    ```
    bash launch_deepseekr1_fp8.sh
    ```
    The example command:
    ```
    model=/data/models/Deepseek-r1-FP8-Dynamic
    TP=8
    EP=1
    python3 -m sglang.launch_server \
        --model-path ${model} \
        --host localhost \
        --port 9000 \
        --tp-size ${TP} \
        --ep-size ${EP} \
        --trust-remote-code \
        --chunked-prefill-size 32768 \
        --mem-fraction-static 0.9 \
        --disable-radix-cache \
        --max-prefill-tokens 32768 \
        --cuda-graph-max-bs 128 \
        2>&1 | tee log.server.log &
    ```

# Curl request
1. curl a single request to quickly check the functionality
   ```
    curl -X POST "http://localhost:9000/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "prompt": "The capital of China", "temperature": 0, "top_p": 1, "top_k": 0, "repetition_penalty": 1.0, "presence_penalty": 0, "frequency_penalty": 0, "stream": false, "ignore_eos": false, "n": 1, "seed": 123
    }'
   ```
   The result should be:
   ```
    [2025-10-14 00:58:19] INFO:     127.0.0.1:57592 - "POST /v1/completions HTTP/1.1" 200 OK
    {"id":"e2ac826727bb4977bf0f4ee84eac92bc","object":"text_completion","created":1760403499,"model":"default","choices":[{"index":0,"text":" is Beijing, and the capital of Japan is Tokyo. Both Beijing and Tokyo are","logprobs":null,"finish_reason":"length","matched_stop":null}],"usage":{"prompt_tokens":5,"total_tokens":21,"completion_tokens":16,"prompt_tokens_details":null,"reasoning_tokens":0},"metadata":{"weight_version":"default"}}
   ```

# Benchmark
1. Take deepseek as example, you can use the following command to benchmark serve.
    ```
    model=/data/pretrained-models/DeepSeek-R1-MXFP4-Preview

    input_tokens=3584
    output_tokens=1024
    max_concurrency=64
    num_prompts=128

    python3 -m sglang.bench_serving \
        --host localhost \
        --port 9000 \
        --model ${model} \
        --dataset-name random \
        --random-input ${input_tokens} \
        --random-output ${output_tokens} \
        --random-range-ratio 1.0 \
        --max-concurrency ${max_concurrency} \
        --num-prompt ${num_prompts} \
        2>&1 | tee log.client.log
    ```

# Profile
1. set the env flags
    ```
    export SGLANG_TORCH_PROFILER_DIR=./
    export SGLANG_PROFILE_WITH_STACK=1
    export SGLANG_PROFILE_RECORD_SHAPES=1
    <launch the server>
    <launch the client with the additional --profile argument>
    ```

# Evaluation

## Text Model Evaluation

Text model is evaluated using lm-eval (<https://github.com/EleutherAI/lm-evaluation-harness.git>).

1. Install dependencies. `pip install lm_eval[api]`.
2. Start lm-eval. Example:

    ```bash
    addr=localhost
    port=9000

    url=http://${addr}:${port}/v1/completions
    model=/data/pretrained-models/DeepSeek-R1-MXFP4-Preview
    bs=auto
    task=gsm8k

    echo "model=${model}"
    echo "task=${task}"

    lm_eval \
        --model local-completions \
        --tasks ${task} \
        --model_args model=${model},base_url=${url} \
        --batch_size ${bs} \
        --num_fewshot 5 \
        --limit 250 \
        --seed 123 \
        2>&1 | tee log.lmeval.log
    ```
    The result of TP8 should be:
    ```
    |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
    |-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
    |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.960|±  |0.0124|
    |     |       |strict-match    |     5|exact_match|↑  |0.956|±  |0.0130|
    ```
    The result of TP8 and EP8 should be:
    ```
    |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
    |-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
    |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.956|±  |0.0130|
    |     |       |strict-match    |     5|exact_match|↑  |0.952|±  |0.0135|
    ```
