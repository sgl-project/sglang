
unset http_proxy
unset https_proxy


export PYTHONPATH=/home/l00993641/sglang/python:$PYTHONPATH

evalscope eval \
    --model /home/weights/DeepSeek-V4-Flash-0731-w8a8 \
    --api-url http://127.0.0.1:30100/v1 \
    --api-key EMPTY \
    --eval-type openai_api \
    --generation-config '{
        "max_tokens": 84000,
        "seed": 3407,
        "top_p": 1.0,
        "temperature": 1.0,
        "n": 1,
        "timeout": 3600,
        "stream": true,
        "extra_body": {
            "chat_template_kwargs": {
                "thinking": true,
                "reasoning_effort": "max"
            }
        }
    }' \
    --datasets aime26 \
    --dataset-hub local \
    --dataset-args '{
        "aime26": { "local_path": "/home/datasets/aime26}}' \
    --eval-batch-size 30 \
    --limit 30 \
    --ignore-error