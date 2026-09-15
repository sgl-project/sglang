evalscope eval \
    --model dsv \
    --api-url http://127.0.0.1:8001/v1 \
    --api-key EMPTY \
    --eval-type openai_api \
    --generation-config '{
        "max_tokens": 64000,
        "seed": 3407,
        "top_p": 1.0,
        "temperature": 1.0,
        "n": 1,
        "timeout": 3600,
        "stream": false,
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": true,
                "reasoning_effort": "max"
            }
        }
    }' \
    --datasets gpqa_diamond  \
    --dataset-hub local \
    --dataset-args '{
        "gpqa_diamond": {
            "local_path": "/home/gpqa_diamond",
            "eval_split": "train"
        }
    }' \
    --eval-batch-size 64 \
    --ignore-error \


    --model /home/weights/DeepSeek-V4-Flash-0731-w8a8 \
