# Assuming the model is downdloaded at /home/ubuntu/model_weights/Llama-2-7b-chat-hf
docker run --name tgi --rm -ti --gpus all --network host \
  -v /home/ubuntu/model_weights/Llama-2-7b-chat-hf:/Llama-2-7b-chat-hf \
  ghcr.io/huggingface/text-generation-inference:1.1.0 \
  --model-id /Llama-2-7b-chat-hf --num-shard 1  --trust-remote-code \
  --max-input-length 2048 --max-total-tokens 4096 \
  --port 24000
