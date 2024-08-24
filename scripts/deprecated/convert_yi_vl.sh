# For 34B Model
mkdir ~/model_weights
cd ~/model_weights
git clone https://huggingface.co/01-ai/Yi-VL-34B
cp ~/model_weights/Yi-VL-34B/vit/clip-vit-H-14-laion2B-s32B-b79K-yi-vl-34B-448/preprocessor_config.json ~/model_weights/Yi-VL-34B
python3 convert_yi_vl.py --model-path ~/model_weights/Yi-VL-34B

# For 6B Model
mkdir ~/model_weights
cd ~/model_weights
git clone https://huggingface.co/01-ai/Yi-VL-6B
cp ~/model_weights/Yi-VL-6B/vit/clip-vit-H-14-laion2B-s32B-b79K-yi-vl-6B-448/preprocessor_config.json ~/model_weights/Yi-VL-6B
python3 convert_yi_vl.py --model-path ~/model_weights/Yi-VL-6B
