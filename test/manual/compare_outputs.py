#!/usr/bin/env python3
"""Compare SGLang vs Transformers outputs for OpenVLA."""

import os
os.environ["SGLANG_DISABLE_CUDNN_CHECK"] = "1"

import gc
import numpy as np
import requests
import torch
from io import BytesIO
from PIL import Image

# Test image and prompt
TEST_URL = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/images/man_ironing_on_back_of_suv.png"
PROMPT = "In: What action should the robot take to pick up the red block?\nOut:"


def load_image(url):
    response = requests.get(url, timeout=30)
    return Image.open(BytesIO(response.content)).convert("RGB")


def decode_tokens_to_actions(tokens, vocab_size=32000):
    """Decode action tokens to bins and normalized actions using HF formula."""
    bins = [max(0, min(255, vocab_size - t - 1)) for t in tokens]
    actions = [(2.0 * b + 1.0) / 256.0 - 1.0 for b in bins]
    return bins, actions


def run_transformers(image):
    """Run Transformers inference."""
    from transformers import AutoProcessor, AutoModelForVision2Seq

    print("=" * 70)
    print("Transformers Inference")
    print("=" * 70)

    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)

    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda:0")
    model.eval()

    inputs = processor(PROMPT, image).to("cuda:0", dtype=torch.bfloat16)

    print(f"Input pixel_values shape: {inputs['pixel_values'].shape}")
    print(f"Input IDs shape: {inputs['input_ids'].shape}")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=7,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    output_ids = outputs[0, input_len:].cpu().tolist()
    print(f"Output tokens: {output_ids}")

    bins, actions = decode_tokens_to_actions(output_ids)
    print(f"Bins:          {bins}")
    print(f"Actions:       {[f'{a:.4f}' for a in actions]}")

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    return output_ids, bins, actions


def run_sglang(image):
    """Run SGLang inference."""
    from sglang import Engine

    print("\n" + "=" * 70)
    print("SGLang Inference")
    print("=" * 70)

    engine = Engine(
        model_path="openvla/openvla-7b",
        trust_remote_code=True,
        enable_multimodal=True,
        mem_fraction_static=0.80,
        disable_cuda_graph=True,
    )

    output = engine.generate(
        prompt=PROMPT,
        image_data=[image],
        sampling_params={
            "temperature": 0,
            "max_new_tokens": 7,
        },
    )

    output_ids = output.get("output_ids", [])
    print(f"Output tokens: {output_ids}")

    bins, actions = decode_tokens_to_actions(output_ids)
    print(f"Bins:          {bins}")
    print(f"Actions:       {[f'{a:.4f}' for a in actions]}")

    engine.shutdown()

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return output_ids, bins, actions


def compare_results(hf_results, sglang_results):
    """Compare and print results."""
    hf_tokens, hf_bins, hf_actions = hf_results
    sg_tokens, sg_bins, sg_actions = sglang_results

    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    print("\nToken-by-token comparison:")
    print(f"{'Dim':<5} {'HF Token':<15} {'SGLang Token':<15} {'Match':<8} {'Bin Diff':<10}")
    print("-" * 60)

    total_match = 0
    total_bin_diff = 0
    for i in range(min(len(hf_tokens), len(sg_tokens))):
        match = "YES" if hf_tokens[i] == sg_tokens[i] else "NO"
        bin_diff = abs(hf_bins[i] - sg_bins[i])
        total_bin_diff += bin_diff
        if hf_tokens[i] == sg_tokens[i]:
            total_match += 1
        print(f"{i:<5} {hf_tokens[i]:<15} {sg_tokens[i]:<15} {match:<8} {bin_diff:<10}")

    print("-" * 60)

    n = len(hf_tokens)
    print(f"\nToken match rate: {total_match}/{n} ({100*total_match/n:.1f}%)")
    print(f"Average bin difference: {total_bin_diff/n:.2f}")

    print("\nAction value comparison:")
    print(f"{'Dim':<5} {'HF Action':<15} {'SGLang Action':<15} {'Diff':<15}")
    print("-" * 55)

    max_action_diff = 0
    for i in range(min(len(hf_actions), len(sg_actions))):
        diff = abs(hf_actions[i] - sg_actions[i])
        max_action_diff = max(max_action_diff, diff)
        print(f"{i:<5} {hf_actions[i]:<15.4f} {sg_actions[i]:<15.4f} {diff:<15.6f}")

    print("-" * 55)
    print(f"\nMax action difference: {max_action_diff:.6f}")

    if total_match == n:
        print("\n" + "*" * 50)
        print("*** PERFECT MATCH: All tokens identical! ***")
        print("*" * 50)
    elif max_action_diff < 0.01:
        print("\n*** NEAR MATCH: Action difference < 0.01 ***")


if __name__ == "__main__":
    print("Loading test image...")
    image = load_image(TEST_URL)
    print(f"Image size: {image.size}")
    print(f"Prompt: {PROMPT[:60]}...")
    print()

    # Run Transformers FIRST
    hf_results = run_transformers(image)

    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Run SGLang
    sglang_results = run_sglang(image)

    # Compare
    compare_results(hf_results, sglang_results)
