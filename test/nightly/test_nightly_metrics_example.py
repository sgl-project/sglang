"""Example usage of nightly_metrics.py

This shows how to use the generic run_metrics() function to run both
performance and accuracy tests on custom model configurations.
"""

from nightly_metrics import run_metrics

from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, ModelLaunchSettings

# Example 1: Text models - run both perf + accuracy
print("\n" + "=" * 60)
print("Example 1: Text Models - Performance + Accuracy")
print("=" * 60)

text_models = [
    ModelLaunchSettings("meta-llama/Llama-3.1-8B-Instruct", tp_size=1),
    ModelLaunchSettings("Qwen/Qwen2-57B-A14B-Instruct", tp_size=2),
]

result = run_metrics(
    models=text_models,
    run_perf=True,
    run_accuracy=True,
    is_vlm=False,
    base_url=DEFAULT_URL_FOR_TEST,
    profile_dir="performance_profiles_example_text",
    test_name="ExampleTextMetrics",
)

print(f"\n✓ All passed: {result['all_passed']}")
for model_result in result["results"]:
    print(f"  - {model_result['model']}")
    print(f"    Perf: {model_result['perf_passed']}")
    print(f"    Accuracy: {model_result['accuracy_passed']}")
    if model_result["accuracy_metrics"]:
        print(f"    Metrics: {model_result['accuracy_metrics']}")


# Example 2: VLM models - run both perf + accuracy with latency
print("\n" + "=" * 60)
print("Example 2: VLM Models - Performance + Accuracy with Latency")
print("=" * 60)

vlm_models = [
    ModelLaunchSettings("Qwen/Qwen2.5-VL-7B-Instruct"),
    ModelLaunchSettings("Qwen/Qwen3-VL-30B-A3B-Instruct", tp_size=2),
]

result = run_metrics(
    models=vlm_models,
    run_perf=True,
    run_accuracy=True,
    is_vlm=True,
    return_latency=True,
    profile_dir="performance_profiles_example_vlm",
    test_name="ExampleVLMMetrics",
)

print(f"\n✓ All passed: {result['all_passed']}")


# Example 3: Accuracy only - custom eval (GPQA with thinking mode)
print("\n" + "=" * 60)
print("Example 3: Custom Accuracy - GPQA with Thinking Mode")
print("=" * 60)

gpqa_models = [
    ModelLaunchSettings(
        "deepseek-ai/DeepSeek-V3.2",
        tp_size=8,
        extra_args=[
            "--enable-dp-attention",
            "--dp=8",
            "--tool-call-parser=deepseekv32",
            "--reasoning-parser=deepseek-v3",
        ],
    ),
]

result = run_metrics(
    models=gpqa_models,
    run_perf=False,  # Only run accuracy
    run_accuracy=True,
    is_vlm=False,
    eval_name="gpqa",
    num_examples=198,
    max_tokens=120000,
    thinking_mode="deepseek-v3",
    temperature=0.1,
    repeat=4,
)

print(f"\n✓ All passed: {result['all_passed']}")
for model_result in result["results"]:
    print(f"  - {model_result['model']}")
    print(f"    Metrics: {model_result['accuracy_metrics']}")


# Example 4: Performance only - custom batch sizes
print("\n" + "=" * 60)
print("Example 4: Performance Only - Custom Batch Sizes")
print("=" * 60)

perf_models = [
    ModelLaunchSettings("meta-llama/Llama-3.1-8B-Instruct", tp_size=1),
]

result = run_metrics(
    models=perf_models,
    run_perf=True,
    run_accuracy=False,  # Only run performance
    is_vlm=False,
    batch_sizes=[1, 4, 16, 32, 128],
    input_lens=(2048, 4096),
    output_lens=(256, 512),
    profile_dir="performance_profiles_example_custom",
)

print(f"\n✓ All passed: {result['all_passed']}")
