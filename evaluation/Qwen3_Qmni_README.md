# Set Environment

1. Docker image:

   ```
   rocm/ali-private:ubuntu22.04_rocm6.4.3.127_sglang_$(SGLANG_HASH)_aiter_$(AITER_HASH)_$(DATE)
   ```
2. Install aiter:
   ```
   pip uninstall aiter
   git clone -b qwen3vl-project https://github.com/ZLkanyo009/aiter.git
   cd aiter
   git checkout <target_commit>
   git submodule sync && git submodule update --init --recursive
   python3 setup.py develop
   ```
4. Install sglang:
   ```
   pip uninstall sglang
   git clone -b dev/perf https://github.com/zejunchen-zejun/sglang.git
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
1. download model Qwen3-Omni-30B-A3B-Instruct

    ```bash
    huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct --local-dir /models/Qwen3-Omni-30B-A3B-Instruct
    ```

- launch server:

    The example command:
    ```bash
    model=/models/Qwen3-Omni-30B-A3B-Instruct/
    TP=4
    DP=2

    echo "launching ${model}"
    echo "TP=${TP}"
    echo "DP=${DP}"

     SGLANG_VLM_CACHE_SIZE_MB=0 \
     SGLANG_USE_AITER=1 \
     USE_PA=1 \
     SGLANG_ROCM_USE_AITER_PA_ASM_PRESHUFFLE_LAYOUT=0 \
     SGLANG_ROCM_USE_AITER_LINEAR_SHUFFLE=1 \
     python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --host localhost    \
        --port 9000 \
        --tensor-parallel-size ${TP} \
        --data-parallel-size ${DP} \
        --trust-remote-code \
        --chunked-prefill-size 32768 \
        --mem-fraction-static 0.85 \
        --mm-attention-backend "aiter_attn" \
        --max-prefill-tokens 32768 \
        --disable-radix-cache \
        --page-size 64 \
        --mm-enable-dp-encoder \
        --cuda-graph-max-bs 8
        2>&1 | tee log.server.log &

    ```
    You can add `--mm-enable-dp-encoder` when launch server, this command can reduces TTFT for multi-modal workloads under some testing conditions.

# Curl request
1. curl a single request to quickly check the functionality

 
   Then curl a single quickly request
   ```
    curl --request POST \
    --url "http://localhost:9000/v1/chat/completions" \
    --header "Content-Type: application/json" \
    --data '{
        "model": "Qwen3-Omni-30B-A3B-Instruct",
        "messages": [
        {
            "role": "user",
            "content": [
            {
                "type": "image_url",
                "image_url": {
                "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"
                }
            },
            {
                "type": "audio_url",
                "audio_url": {
                "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav"
                }
            },
            {
                "type": "text",
                "text": "What can you see and hear? Answer in one short sentence."
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
    {"id":"a9c3b01f7bc34604af405061e0320c81","object":"chat.completion","created":1765787394,"model":"Qwen3-Omni-30B-A3B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"The image displays four luxury vehicles—a white Rolls-Royce, a red Ferrari Portofino M, a white Porsche 911, and a grey Mercedes-Benz GLE SUV—while the audio contains the sound of a person coughing.","reasoning_content":null,"tool_calls":null},"logprobs":null,"finish_reason":"stop","matched_stop":151645}],"usage":{"prompt_tokens":6105,"total_tokens":6156,"completion_tokens":51,"prompt_tokens_details":null,"reasoning_tokens":0},"metadata":{"weight_version":"default"}}r
   ```

# Benchmark
The benchmarks for Qwen3-Omni are conducted in three distinct scenarios.

## Image+Text
    ```bash
    model=/models/Qwen3-Omni-30B-A3B-Instruct/

    input_tokens=8000
    output_tokens=500
    num_prompts=64
    max_concurrency=1
    image_count=10
    image_resolution=960x1280
    dataset_name="image"

    echo "bench model: ${model}"
    echo "input tokens: ${input_tokens}"
    echo "output tokens: ${output_tokens}"
    echo "image-count: ${image_count}"
    echo "image-resolution: ${image_resolution}"
    echo "max concurrency: ${max_concurrency}"
    echo "num prompts: ${num_prompts}"
    echo "dataset-name: ${dataset_name}"

    python3 -m sglang.bench_serving \
        --backend sglang-oai-chat \
        --model ${model} \
        --dataset-name ${dataset_name} \
        --host localhost \
        --port 9000 \
        --num-prompts ${num_prompts} \
        --image-count ${image_count} \
        --image-resolution ${image_resolution} \
        --random-input ${input_tokens} \
        --random-output ${output_tokens} \
        --random-range-ratio 1.0 \
        --max-concurrency ${max_concurrency} \
        --flush-cache \
        --skip-special-tokens \
        2>&1 | tee log.client.log
    ```
## Pure Text
    ```bash
    model=/models/Qwen3-Omni-30B-A3B-Instruct/

    input_tokens=8000
    output_tokens=500
    num_prompts=64
    max_concurrency=1
    dataset_name="random-ids"

    echo "bench model: ${model}"
    echo "input tokens: ${input_tokens}"
    echo "output tokens: ${output_tokens}"
    echo "max concurrency: ${max_concurrency}"
    echo "num prompts: ${num_prompts}"
    echo "dataset-name: ${dataset_name}"

    python3 -m sglang.bench_serving \
        --backend sglang-oai-chat \
        --model ${model} \
        --dataset-name ${dataset_name} \
        --host localhost \
        --port 9000 \
        --num-prompts ${num_prompts} \
        --random-input ${input_tokens} \
        --random-output ${output_tokens} \
        --random-range-ratio 1.0 \
        --max-concurrency ${max_concurrency} \
        --flush-cache \
        2>&1 | tee log.client.log
    ```

## Audio+Text
    ```bash
    model=/models/Qwen3-Omni-30B-A3B-Instruct/

    input_tokens=2000
    output_tokens=20
    num_prompts=64
    max_concurrency=1
    dataset_name="random-omni"
    audio_length=10

    echo "bench model: ${model}"
    echo "input tokens: ${input_tokens}"
    echo "output tokens: ${output_tokens}"
    echo "max concurrency: ${max_concurrency}"
    echo "num prompts: ${num_prompts}"
    echo "audio lengths: ${audio_length}"
    echo "dataset-name: ${dataset_name}"

    python3 -m sglang.bench_serving \
        --backend sglang-oai-chat \
        --dataset-name ${dataset_name} \
        --model ${model} \
        --host localhost \
        --port 9000 \
        --num-prompts ${num_prompts} \
        --random-input ${input_tokens} \
        --random-output ${output_tokens} \
        --random-range-ratio 1.0 \
        --max-concurrency ${max_concurrency} \
        --flush-cache \
        --skip-special-tokens \
        --audio-length ${audio_length} \
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
    python3 benchmark/mmmu/bench_sglang.py --port 9000 --concurrency 16
    ```
    The result of TP4 on MI308 should be:
    ```
    {'Accounting': {'acc': 0.467, 'num': 30},
    'Agriculture': {'acc': 0.667, 'num': 30},
    'Architecture_and_Engineering': {'acc': 0.167, 'num': 30},
    'Art': {'acc': 0.7, 'num': 30},
    'Art_Theory': {'acc': 0.833, 'num': 30},
    'Basic_Medical_Science': {'acc': 0.733, 'num': 30},
    'Biology': {'acc': 0.5, 'num': 30},
    'Chemistry': {'acc': 0.433, 'num': 30},
    'Clinical_Medicine': {'acc': 0.633, 'num': 30},
    'Computer_Science': {'acc': 0.567, 'num': 30},
    'Design': {'acc': 0.8, 'num': 30},
    'Diagnostics_and_Laboratory_Medicine': {'acc': 0.367, 'num': 30},
    'Economics': {'acc': 0.6, 'num': 30},
    'Electronics': {'acc': 0.433, 'num': 30},
    'Energy_and_Power': {'acc': 0.4, 'num': 30},
    'Finance': {'acc': 0.267, 'num': 30},
    'Geography': {'acc': 0.6, 'num': 30},
    'History': {'acc': 0.733, 'num': 30},
    'Literature': {'acc': 0.867, 'num': 30},
    'Manage': {'acc': 0.533, 'num': 30},
    'Marketing': {'acc': 0.5, 'num': 30},
    'Materials': {'acc': 0.233, 'num': 30},
    'Math': {'acc': 0.4, 'num': 30},
    'Mechanical_Engineering': {'acc': 0.367, 'num': 30},
    'Music': {'acc': 0.333, 'num': 30},
    'Overall': {'acc': 0.547, 'num': 900},
    'Overall-Art and Design': {'acc': 0.667, 'num': 120},
    'Overall-Business': {'acc': 0.473, 'num': 150},
    'Overall-Health and Medicine': {'acc': 0.62, 'num': 150},
    'Overall-Humanities and Social Science': {'acc': 0.725, 'num': 120},
    'Overall-Science': {'acc': 0.507, 'num': 150},
    'Overall-Tech and Engineering': {'acc': 0.405, 'num': 210},
    'Pharmacy': {'acc': 0.7, 'num': 30},
    'Physics': {'acc': 0.6, 'num': 30},
    'Psychology': {'acc': 0.667, 'num': 30},
    'Public_Health': {'acc': 0.667, 'num': 30},
    'Sociology': {'acc': 0.633, 'num': 30}}
    eval out saved to ./val_sglang.json
    Overall accuracy: 0.547
    ```
