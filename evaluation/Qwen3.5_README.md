# Set Environment

1. Docker image:
   ```bash
   rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260301
   ```

2. Install aiter:
   ```bash
   pip uninstall -y aiter
   git clone -b dev/perf https://github.com/sammysun0711/aiter.git
   cd aiter
   git checkout <target_commit>
   git submodule sync && git submodule update --init --recursive
   python3 setup.py develop
   ```

3. Setup Triton
    ```bash
    pip uninstall -y triton
    git clone https://github.com/ROCm/triton -b gluon_ext
    cd triton
    pip install -r python/requirements.txt
    python setup.py install
    cd ..
    ```

4. Setup pyhip
    ```bash
    pip uninstall -y pyhip
    git clone https://github.com/tingqli/pyhip.git
    cd pyhip && git checkout fed8898cfd79f3db9474750c22891e704a8329b9
    pip install -e .
    cd ..
    ```

5. Install sglang:
   ```bash
   pip uninstall -y sglang
   git clone -b dev/perf https://github.com/sammysun0711/sglang.git
   cd sglang
   git checkout <target_commit>
   # Compile sgl-kernel
   pip install --upgrade pip
   cd sgl-kernel
   python3 setup_rocm.py install

   # Install sglang python package
   cd ..
   rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
   pip install -e "python[all_hip]"
   ```

# Launch server

    download model Qwen/Qwen3.5-397B-A17B

    You can either:
    - Download the original BF16 model from Hugging Face:
    ```bash
    huggingface-cli download Qwen/Qwen3.5-397B-A17B --local-dir /models/Qwen/Qwen3.5-397B-A17B
    ```

    - Download the original FP8 model from Hugging Face:
    huggingface-cli download Qwen/Qwen3.5-397B-A17B-FP8 --local-dir /models/Qwen/Qwen3.5-397B-A17B-FP8

- launch server:

    The example command:
    ```bash

    OPTFLAG="w8a8_gemm,moe" ROCM_QUICK_REDUCE_QUANTIZATION=INT4 python -m sglang.launch_server \
        --model-path /models/Qwen3.5-397B-A17B \
        --port 9000 \
        --tp-size 8 \
        --mem-fraction-static 0.8 \
        --context-length 262144 \
        --reasoning-parser qwen3 \
        --attention-backend triton \
        --disable-radix-cache \
        --cuda-graph-max-bs 64
    ```

# Curl request
   curl a single request to quickly check the functionality

   Then curl a single quickly request
   ```
    curl --request POST \
    --url "http://localhost:9000/v1/chat/completions" \
    --header "Content-Type: application/json" \
    --data '{
        "model": "/models/Qwen3.5-397B-A17B",
        "messages": [
        {
            "role": "user",
            "content": [
            {
                "type": "image_url",
                "image_url": {
                "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/CI_Demo/mathv-1327.jpg"
                }
            },
            {
                "type": "text",
                "text": "Describe this image in detail."
            }
            ]
        }
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 100
    }'
   ```
   The result should be:
   ```
   {"id":"d1037684c09942d59afffa50dd4a0d90","object":"chat.completion","created":1772764449,"model":"/models/Qwen3.5-397B-A17B","choices":[{"index":0,"message":{"role":"assistant","content":null,"reasoning_content":"The user wants a detailed description of the provided image.\n\n1.  **Identify the main components:**\n    *   There is a central square.\n    *   There are four circles of varying sizes.\n    *   The shapes are outlined in black on a white background. It's a line drawing.\n\n2.  **Analyze the arrangement and relationships:**\n    *   **The Square:** It's in the center. It's upright (sides are vertical and","tool_calls":null},"logprobs":null,"finish_reason":"length","matched_stop":null}],"usage":{"prompt_tokens":174,"total_tokens":274,"completion_tokens":100,"prompt_tokens_details":null,"reasoning_tokens":0},"metadata":{"weight_version":"default"}}
   ```

# Benchmark
The benchmarks guide for Qwen3.5-397B-A17B 

## Pure Text
    ```bash
    model=/models/Qwen3.5-397B-A17B # or Use /models/Qwen3.5-397B-A17B-FP8

    input_tokens=8000
    output_tokens=500
    num_prompts=32
    max_concurrency=1
    dataset_name="random"

    echo "bench model: ${model}"
    echo "input tokens: ${input_tokens}"
    echo "output tokens: ${output_tokens}"
    echo "max concurrency: ${max_concurrency}"
    echo "num prompts: ${num_prompts}"
    echo "dataset-name: ${dataset_name}"

    python3 -m sglang.bench_serving \
        --backend sglang \
        --model ${model} \
        --dataset-name ${dataset_name} \
        --host localhost \
        --port 9000 \
        --num-prompts ${num_prompts} \
        --random-input ${input_tokens} \
        --random-output ${output_tokens} \
        --random-range-ratio 1.0 \
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

Vision model is evaluated on MMMU dataset. More information you can find in the [benchmark/mmmu/README.md](../benchmark/mmmu/README.md).


1. Start evaluating. Example:

    ```bash
    python3 benchmark/mmmu/bench_sglang.py --port 9000 --concurrency 64
    ```
    The result of TP8 on MI350 should be:
    ```
    {'Accounting': {'acc': 1.0, 'num': 1},
    'Agriculture': {'acc': 1.0, 'num': 4},
    'Architecture_and_Engineering': {'acc': 1.0, 'num': 1},
    'Art': {'acc': 1.0, 'num': 8},
    'Art_Theory': {'acc': 1.0, 'num': 7},
    'Basic_Medical_Science': {'acc': 1.0, 'num': 4},
    'Biology': {'acc': 1.0, 'num': 1},
    'Clinical_Medicine': {'acc': 1.0, 'num': 1},
    'Computer_Science': {'acc': 1.0, 'num': 2},
    'Design': {'acc': 1.0, 'num': 13},
    'Economics': {'acc': 0.75, 'num': 4},
    'Finance': {'acc': 1.0, 'num': 4},
    'Geography': {'acc': 1.0, 'num': 3},
    'History': {'acc': 1.0, 'num': 1},
    'Literature': {'acc': 0.944, 'num': 18},
    'Manage': {'acc': 1.0, 'num': 3},
    'Marketing': {'acc': 1.0, 'num': 4},
    'Math': {'acc': 0.5, 'num': 2},
    'Overall': {'acc': 0.971, 'num': 103},
    'Overall-Art and Design': {'acc': 1.0, 'num': 28},
    'Overall-Business': {'acc': 0.938, 'num': 16},
    'Overall-Health and Medicine': {'acc': 1.0, 'num': 14},
    'Overall-Humanities and Social Science': {'acc': 0.967, 'num': 30},
    'Overall-Science': {'acc': 0.875, 'num': 8},
    'Overall-Tech and Engineering': {'acc': 1.0, 'num': 7},
    'Pharmacy': {'acc': 1.0, 'num': 5},
    'Physics': {'acc': 1.0, 'num': 2},
    'Psychology': {'acc': 1.0, 'num': 4},
    'Public_Health': {'acc': 1.0, 'num': 4},
    'Sociology': {'acc': 1.0, 'num': 7}}
    eval out saved to ./val_sglang.json
    Overall accuracy: 0.971
    ```
