addr=localhost
port=8080

url=http://${addr}:${port}/v1/completions
model=/data/models/Qwen/Qwen3-Next-80B-A3B-Instruct
bs=auto
task=gsm8k

echo "model=${model}"
echo "task=${task}"

lm_eval \
    --model local-completions \
    --tasks ${task} \
    --model_args model=${model},base_url=${url} \
    --batch_size ${bs} \
    --num_fewshot 5 \
    --limit 250 \
    --seed 123 \
    2>&1 | tee log.lmeval.log
