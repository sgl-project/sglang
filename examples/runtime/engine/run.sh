python offline_batch_inference_vlm.py --model-path  google/paligemma-3b-pt-224 --chat-template=paligemma > pali.log 2>&1 


python offline_batch_inference_vlm.py --model-path Qwen/Qwen2-VL-7B-Instruct --chat-template=qwen2-vl > qwen.log 2>&1


python offline_batch_inference_vlm.py --model-path Intel/llava-llama-3-8b --chat-template=llava_llama_3 > llava.log 2>&1