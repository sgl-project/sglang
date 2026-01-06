export OPENAI_API_KEY=EMPTY
export OPENAI_API_BASE=http://localhost:9000/v1
export PYTHONPATH=/the/path/to/your/sglang/python

python3 -m lmms_eval \
    --model=openai_compatible \
    --model_args model_version=/data/models/Qwen3-VL-235B-A22B-Instruct-FP8-dynamic/ \
    --tasks mmmu_val   \
    --batch_size 16 \
    # --log_samples \
	# --output_path=./logs/ \
	# --verbosity=INFO
