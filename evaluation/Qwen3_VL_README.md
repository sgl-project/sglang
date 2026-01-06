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
   python setup_rocm.py install
   export PYTHONPATH=<you_sglang_path/sglang/python>
   ```
5. Other requirements:
    ```
    # The Qwen3-VL model requires transformers >= 4.57.0
    pip install "transformers>=4.57.0"
    # only for MI355
    pip install xgrammar==0.1.25
    ```

# Launch server
1. Qwen3-VL-235B-A22B-Instruct-FP8-dynamic
- download PTPC FP8 model weight: https://huggingface.co/RedHatAI/Qwen3-VL-235B-A22B-Instruct-FP8-dynamic
    ```bash
    huggingface-cli download RedHatAI/Qwen3-VL-235B-A22B-Instruct-FP8-dynamic --local-dir /data/models/Qwen3-VL-235B-A22B-Instruct-FP8-dynamic
    ```

- launch server:
    ```
    bash launch_qwen3vl_fp8_ptpc.sh
    ```
    The example command:
    ```bash
    model=/data/models/Qwen3-VL-235B-A22B-Instruct-FP8-dynamic/
    TP=8
    EP=8

    echo "launching ${model}"
    echo "TP=${TP}"
    echo "EP=${EP}"

    python3 -m sglang.launch_server \
        --model-path ${model} \
        --host localhost \
        --port 9000 \
        --tp-size ${TP} \
        --ep-size ${EP} \
        --trust-remote-code \
        --chunked-prefill-size 32768 \
        --mem-fraction-static 0.6 \
        --disable-radix-cache \
        --max-prefill-tokens 32768 \
        --cuda-graph-max-bs 128 \
        2>&1 | tee log.server.log &

    ```

# Curl request
1. curl a single request to quickly check the functionality
    First, download the test picture.
    ```bash
    wget https://sf-maas-uat-prod.oss-cn-shanghai.aliyuncs.com/dog.png
    ```
    Then curl a single quickly request
   ```
    curl --request POST \
        --url "http://localhost:9000/v1/chat/completions" \
        --header "Content-Type: application/json" \
        --data '{
            "model": "/data/models/Qwen3-VL-235B-A22B-Instruct-FP8-dynamic/",
            "messages": [
                {
                "role": "user",
                "content": [
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": "/workspace/my_sgl/sglang/evaluation/dog.png"
                    }
                    },
                    {
                    "type": "text",
                    "text": "请简要描述图片是什么内容？"
                    }
                ]
                }
            ],
            "temperature": 0.0,
            "top_p": 0.0001,
            "top_k": 1,
            "max_tokens": 100
        }'
   ```
   The result should be:
   ```
    {"id":"0cc14f2dd6cf45ac99a022b0be5a2952","object":"chat.completion","created":1762762377,"model":"/data/models/Qwen3-VL-235B-A22B-Instruct-FP8-dynamic/","choices":[{"index":0,"message":{"role":"assistant","content":"这是一张棕色狗狗的特写肖像照。狗狗毛发蓬松柔软，耳朵下垂，眼神温柔而专注地望向镜头，表情略带忧郁或沉思。背景是模糊的绿色，突出了狗狗的面部细节，整体画面温馨、富有情感。","reasoning_content":null,"tool_calls":null},"logprobs":null,"finish_reason":"stop","matched_stop":151645}],"usage":{"prompt_tokens":1042,"total_tokens":1104,"completion_tokens":62,"prompt_tokens_details":null,"reasoning_tokens":0},"metadata":{"weight_version":"default"}}
   ```

# Benchmark
1. To benchmark image dataset with 1 images per request, 128 prompts, 1000 input length, and 2000 output length, you can run:
    ```bash
    model=/data/models/Qwen3-VL-235B-A22B-Instruct-FP8-dynamic/

    input_tokens=1000
    output_tokens=2000
    num_prompts=128
    max_concurrency=64
    image_count=1
    image_resolution=800x800

    echo "bench model: ${model}"
    echo "input tokens: ${input_tokens}"
    echo "output tokens: ${output_tokens}"
    echo "image-count: ${image_count}"
    echo "image-resolution: ${image_resolution}"
    echo "max concurrency: ${max_concurrency}"
    echo "num prompts: ${num_prompts}"

    python -m sglang.bench_serving \
        --backend sglang-oai-chat \
        --dataset-name image \
        --num-prompts ${num_prompts} \
        --image-count ${image_count} \
        --image-resolution ${image_resolution} \
        --random-input-len ${input_tokens} \
        --random-output-len ${output_tokens} \
        --max-concurrency ${max_concurrency} \
        2>&1 | tee log.client.log
    ```

# Profile
1. set the env flags
    ```bash
    export SGLANG_TORCH_PROFILER_DIR=./
    export SGLANG_PROFILE_WITH_STACK=1
    export SGLANG_PROFILE_RECORD_SHAPES=1
    <launch the server>
    <launch the client with the additional --profile argument>
    ```
Please make sure that the `SGLANG_TORCH_PROFILER_DIR` should be set at both server and client side, otherwise the trace file cannot be generated correctly.

# Evaluation

## Vision Model Evaluation

Vision model is evaluated on MMMU dataset.

1. First, install the lmms-eval package:
    ```bash
    git clone --branch v0.5 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
    cd lmms-eval
    pip3 install -e .
    ```
2. Start evaluating:
    ```bash
    export OPENAI_API_KEY=EMPTY
    export OPENAI_API_BASE=http://localhost:9000/v1
    export PYTHONPATH=/the/path/to/your/sglang/python

    python3 -m lmms_eval \
        --model=openai_compatible \
        --model_args model_version=/mnt/raid0/models/Qwen3-VL-235B-A22B-Instruct-FP8-dynamic/ \
        --tasks mmmu_val   \
        --batch_size 16 \
    ```
    The result of TP8 on MI308 should be:
    ```bash
    openai_compatible (model_version=/mnt/raid0/models/Qwen3-VL-235B-A22B-Instruct-FP8-dynamic/), gen_kwargs: (), limit: None, num_fewshot: None, batch_size: 16
    | Tasks  |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
    |--------|------:|------|-----:|--------|---|-----:|---|------|
    |mmmu_val|      0|none  |     0|mmmu_acc|↑  |0.607 |±  |   N/A|
    ```

3. Other

    We can also use benchmark to evaluate VLM accuracy.  More information you can find in the [benchmark/mmmu/README.md](../benchmark/mmmu/README.md).
    ```bash
    python benchmark/mmmu/bench_sglang.py --port 9000 --concurrency 16
    ```
    The result of TP8 & EP8 on MI308 should be:
    ```
    Evaluating...
    answers saved to: ./answer_sglang.json
    {'Accounting': {'acc': 0.4, 'num': 30},
    'Agriculture': {'acc': 0.567, 'num': 30},
    'Architecture_and_Engineering': {'acc': 0.3, 'num': 30},
    'Art': {'acc': 0.767, 'num': 30},
    'Art_Theory': {'acc': 0.9, 'num': 30},
    'Basic_Medical_Science': {'acc': 0.733, 'num': 30},
    'Biology': {'acc': 0.567, 'num': 30},
    'Chemistry': {'acc': 0.567, 'num': 30},
    'Clinical_Medicine': {'acc': 0.8, 'num': 30},
    'Computer_Science': {'acc': 0.667, 'num': 30},
    'Design': {'acc': 0.9, 'num': 30},
    'Diagnostics_and_Laboratory_Medicine': {'acc': 0.4, 'num': 30},
    'Economics': {'acc': 0.633, 'num': 30},
    'Electronics': {'acc': 0.2, 'num': 30},
    'Energy_and_Power': {'acc': 0.4, 'num': 30},
    'Finance': {'acc': 0.367, 'num': 30},
    'Geography': {'acc': 0.6, 'num': 30},
    'History': {'acc': 0.8, 'num': 30},
    'Literature': {'acc': 0.833, 'num': 30},
    'Manage': {'acc': 0.433, 'num': 30},
    'Marketing': {'acc': 0.567, 'num': 30},
    'Materials': {'acc': 0.433, 'num': 30},
    'Math': {'acc': 0.4, 'num': 30},
    'Mechanical_Engineering': {'acc': 0.233, 'num': 30},
    'Music': {'acc': 0.4, 'num': 30},
    'Overall': {'acc': 0.578, 'num': 900},
    'Overall-Art and Design': {'acc': 0.742, 'num': 120},
    'Overall-Business': {'acc': 0.48, 'num': 150},
    'Overall-Health and Medicine': {'acc': 0.68, 'num': 150},
    'Overall-Humanities and Social Science': {'acc': 0.767, 'num': 120},
    'Overall-Science': {'acc': 0.54, 'num': 150},
    'Overall-Tech and Engineering': {'acc': 0.4, 'num': 210},
    'Pharmacy': {'acc': 0.767, 'num': 30},
    'Physics': {'acc': 0.567, 'num': 30},
    'Psychology': {'acc': 0.733, 'num': 30},
    'Public_Health': {'acc': 0.7, 'num': 30},
    'Sociology': {'acc': 0.7, 'num': 30}}
    eval out saved to ./val_sglang.json
    Overall accuracy: 0.578
    ```
