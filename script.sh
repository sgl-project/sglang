/home/peiyuan_zhang_ntu_edu_sg/miniconda3/envs/sglang/bin/python -m sglang.launch_server \
--model-path lmms-lab/llama3-llava-next-8b \
--tokenizer-path lmms-lab/llama3-llava-next-8b-tokenizer \
--port=12000 --host="127.0.0.1" \
--tp-size=1 --chat-template llava_llama_3


/home/peiyuan_zhang_ntu_edu_sg/miniconda3/envs/sglang/bin/python -m sglang.launch_server \
--model-path lmms-lab/llava-onevision-qwen2-7b-ov \
--tokenizer-path lmms-lab/llavanext-qwen-siglip-tokenizer \
--port=30000 \
--host=127.0.0.1 \
--tp-size=1 \
--chat-template=chatml-llava


/home/peiyuan_zhang_ntu_edu_sg/miniconda3/envs/sglang/bin/python test/srt/test_multi_image_openai_server.py