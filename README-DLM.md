# build
```
git clone https://github.com/hutm/sglang sglang-diffusion && cd sglang-diffusion && git checkout dev-dllm
uv venv --python 3.12 --seed && source .venv/bin/activate
pip install -e "python[all]"
```

# start server
```
python -m sglang.launch_server --model-path <HF_MODEL_PATH> --trust-remote-code --diffusion-algorithm=FastDiffuser
```

# test inference
```
curl -s http://localhost:30000/generate -H "Content-Type: application/json" -d '{"text": "User: What is the capital of France?", "sampling_params": {"temperature": 0.7, "max_new_tokens": 25}}'
```
