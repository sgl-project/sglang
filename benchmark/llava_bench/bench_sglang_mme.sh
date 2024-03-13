MME_FOLDER=./mme_pack
python3 bench_sglang.py --num-questions 5000 --question-file $MME_FOLDER/llava_mme_bench_replace.jsonl --answer-file answer_mme.jsonl --image-folder $MME_FOLDER/MME_Benchmark_release_version --max-tokens 4
