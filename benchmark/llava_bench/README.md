## Download benchmark images

```
python3 download_images.py
```

image benchmark source: https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild

### Other Dependency
```
pip3 install "sglang[all]"
pip3 install "torch>=2.1.2" "transformers>=4.36" pillow
```

## Run benchmark

### Benchmark sglang
Launch a server
```
python3 -m sglang.launch_server --model-path liuhaotian/llava-v1.6-vicuna-7b --tokenizer-path llava-hf/llava-1.5-7b-hf --port 30000
```

Run benchmark
```
# Run with local models
python3 bench_sglang.py --num-questions 60

# Run with OpenAI models
python3 bench_sglang.py --num-questions 60 --backend gpt-4-vision-preview
```

### Bench LLaVA original code
```
git clone git@github.com:haotian-liu/LLaVA.git
cd LLaVA
git reset --hard 9a26bd1435b4ac42c282757f2c16d34226575e96
pip3 install -e .

cd ~/sglang/benchmark/llava_bench
CUDA_VISIBLE_DEVICES=0 bash bench_hf_llava_bench.sh
```


### Benchmark llama.cpp

```
# Install
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
pip install sse_starlette starlette_context pydantic_settings

# Download weights
mkdir -p ~/model_weights/llava-v1.5-7b/
wget https://huggingface.co/mys/ggml_llava-v1.5-7b/resolve/main/ggml-model-f16.gguf -O ~/model_weights/llava-v1.5-7b/ggml-model-f16.gguf
wget https://huggingface.co/mys/ggml_llava-v1.5-7b/resolve/main/mmproj-model-f16.gguf -O ~/model_weights/llava-v1.5-7b/mmproj-model-f16.gguf
```

```
python3 -m llama_cpp.server --model ~/model_weights/llava-v1.5-7b/ggml-model-f16.gguf --clip_model_path ~/model_weights/llava-v1.5-7b/mmproj-model-f16.gguf --chat_format llava-1-5 --port 23000

OPENAI_BASE_URL=http://localhost:23000/v1 python3 bench_sglang.py --backend gpt-4-vision-preview --num-q 1
```
