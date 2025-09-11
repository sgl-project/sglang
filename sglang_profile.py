import sglang as sgl
import os
import time

def profile_with_custom_id_example():
    # 设置环境变量
    os.environ["SGLANG_TORCH_PROFILER_DIR"] = "./profile_output"

    # 创建 engine
    engine = sgl.Engine(
        model_path="qwen/qwen2.5-0.5b-instruct",
        log_level="info"
    )

    try:
        # 使用自定义profile_id
        custom_profile_id = "my_experiment_v1.0"

        print(f"Starting profiler with custom ID: {custom_profile_id}")
        engine.start_profile(
            profile_id=custom_profile_id,
            output_dir="./profile_output",  # 可选，会覆盖环境变量
            activities=["CPU", "GPU"],
            with_stack=True
        )

        # 执行推理
        prompts = ["The future of AI is"]
        outputs = engine.generate(prompts, {"max_new_tokens": 50})

        # 停止profiling
        engine.stop_profile()

        print("Profile saved with custom ID")
        # 文件名将是: my_experiment_v1.0-TP-0.trace.json.gz

    finally:
        # 清理资源
        engine.shutdown()

if __name__ == "__main__":
    profile_with_custom_id_example()