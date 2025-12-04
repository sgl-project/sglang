```shell
# Serve Qwen-Image
export SGLANG_DIFFUSION_STAGE_LOGGING=1
CUDA_VISIBLE_DEVICES=0 \
sglang serve \
    --model-path /home/t4/models/lvm-data/Qwen-Image \
    --pin-cpu-memory \
    --num-gpus 1 \
    --port 30010

# Serve VL
CUDA_VISIBLE_DEVICES=1 \
python3 -m sglang.launch_server \
    --model-path /home/t4/models/lvm-data/Qwen2.5-VL-7B-Instruct \
    --tensor-parallel-size 1 \
    --port 30000

# Start benchmark
# You can get the --scheduler-port from the Qwen-Image server `Rank 0 scheduler listening on tcp://*:5640`
# You can get the datasets from https://github.com/A113N-W3I/TIIF-Bench/tree/d875cd8/data
python python/sglang/multimodal_gen/benchmarks/bench_t2i_models.py \
    --model /home/t4/models/lvm-data/Qwen-Image \
    --file_prefix color+2d \
    --input_folder /sgl-workspace/sglang/python/sglang/multimodal_gen/benchmarks/data/testmini_prompts \
    --eval_folder /sgl-workspace/sglang/python/sglang/multimodal_gen/benchmarks/data/testmini_eval_prompts \
    --scheduler-port 5640

# Output
Total questions: 40, Correct answers: 29, Accuracy: 0.725
                                  mean            p99
        InputValidationStage:   0.030720       0.049098
           TextEncodingStage:  663.408453     668.544694
           ConditioningStage:   0.011137       0.012469
    TimestepPreparationStage:   2.625129       2.707312
      LatentPreparationStage:   5.993766       6.101789
              DenoisingStage: 56800.014939   57035.165373
               DecodingStage:  696.569516     723.747299
                       steps:  1135.911028    1144.564617
           total_duration_ms: 58170.304780   58420.672795
```
