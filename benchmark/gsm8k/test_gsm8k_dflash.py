#!/usr/bin/env python3
"""
GSM8K benchmark following DFlash format:
- Uses chat template with enable_thinking=False
- Uses \boxed{} format for answers
- Sends to SGLang server
- Records acceptance length for speculative decoding
"""

import argparse
import re
import time
import requests
from datasets import load_dataset
from transformers import AutoTokenizer


def extract_boxed_answer(text):
    """Extract the last \boxed{...} answer from the response."""
    # Find all \boxed{...} patterns
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    if matches:
        answer = matches[-1].strip()
        # Remove commas and extract number
        answer = answer.replace(",", "").replace(" ", "")
        numbers = re.findall(r'-?\d+\.?\d*', answer)
        if numbers:
            return numbers[-1]
    return None


def extract_ground_truth(answer_str):
    """Extract the numeric answer from GSM8K ground truth (after ####)."""
    if "####" in answer_str:
        answer = answer_str.split("####")[-1].strip()
        answer = answer.replace(",", "")
        return answer
    return None


def parse_accept_length_from_logs(log_file):
    """Parse average acceptance length from SGLang server logs."""
    accept_lengths = []
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if 'accept len:' in line:
                    # Parse "accept len: X.XX" from log line
                    match = re.search(r'accept len:\s*([\d.]+)', line)
                    if match:
                        accept_lengths.append(float(match.group(1)))
    except FileNotFoundError:
        return None
    
    if accept_lengths:
        return sum(accept_lengths) / len(accept_lengths)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-questions", type=int, default=100)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--server-log", type=str, default=None,
                        help="Path to server log file to parse acceptance length from")
    args = parser.parse_args()
    
    # Load tokenizer for chat template
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Load GSM8K dataset
    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    
    # DFlash prompt format (matching original DFlash benchmark)
    prompt_fmt = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    
    # Prepare questions
    questions = []
    labels = []
    for i, item in enumerate(dataset):
        if i >= args.num_questions:
            break
        
        user_content = prompt_fmt.format(question=item["question"])
        messages = [{"role": "user", "content": user_content}]
        
        # Apply chat template with enable_thinking=False (DFlash style)
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        questions.append(prompt)
        labels.append(extract_ground_truth(item["answer"]))
    
    print(f"Prepared {len(questions)} questions")
    
    # Get stop tokens from tokenizer (matching original DFlash)
    stop_tokens = []
    if tokenizer.eos_token:
        stop_tokens.append(tokenizer.eos_token)
    if hasattr(tokenizer, 'added_tokens_encoder') and '<|im_end|>' in tokenizer.added_tokens_encoder:
        stop_tokens.append('<|im_end|>')
    if not stop_tokens:
        stop_tokens = ["<|im_end|>", "<|endoftext|>"]
    print(f"Using stop tokens: {stop_tokens}")
    
    base_url = f"http://localhost:{args.port}"
    
    # Flush cache before starting
    try:
        requests.post(f"{base_url}/flush_cache", timeout=10)
        print("Cache flushed")
    except Exception as e:
        print(f"Warning: Could not flush cache: {e}")
    
    # Run requests using native /generate endpoint to get spec stats
    print(f"Running with parallel={args.parallel}...")
    
    import concurrent.futures
    
    # Per-request spec stats
    all_spec_accept_lengths = []
    all_spec_accept_rates = []
    
    def process_question(idx):
        prompt = questions[idx]
        try:
            # Use native /generate endpoint which returns meta_info with spec stats
            payload = {
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": args.max_tokens,
                    "temperature": 0,
                    "stop": stop_tokens,
                }
            }
            resp = requests.post(f"{base_url}/generate", json=payload, timeout=300)
            if resp.status_code != 200:
                print(f"Error on question {idx}: HTTP {resp.status_code}")
                return idx, "", 0, 0, None, None
            
            result = resp.json()
            text = result.get("text", "")
            meta_info = result.get("meta_info", {})
            
            prompt_tokens = meta_info.get("prompt_tokens", 0)
            completion_tokens = meta_info.get("completion_tokens", 0)
            
            # Extract per-request spec stats
            spec_accept_length = meta_info.get("spec_accept_length")
            spec_accept_rate = meta_info.get("spec_accept_rate")
            
            return idx, text, prompt_tokens, completion_tokens, spec_accept_length, spec_accept_rate
        except Exception as e:
            print(f"Error on question {idx}: {e}")
            return idx, "", 0, 0, None, None
    
    tic = time.perf_counter()
    
    responses = [None] * len(questions)
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = [executor.submit(process_question, i) for i in range(len(questions))]
        for future in concurrent.futures.as_completed(futures):
            idx, text, prompt_tokens, completion_tokens, spec_len, spec_rate = future.result()
            responses[idx] = text
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            if spec_len is not None:
                all_spec_accept_lengths.append(spec_len)
            if spec_rate is not None:
                all_spec_accept_rates.append(spec_rate)
            # Progress
            done = sum(1 for r in responses if r is not None)
            print(f"\rProgress: {done}/{len(questions)}", end="", flush=True)
    
    print()
    latency = time.perf_counter() - tic
    total_tokens = total_prompt_tokens + total_completion_tokens
    
    # Aggregate spec stats from per-request results
    spec_stats = {}
    if all_spec_accept_lengths:
        spec_stats["avg_accept_length"] = sum(all_spec_accept_lengths) / len(all_spec_accept_lengths)
    if all_spec_accept_rates:
        spec_stats["avg_accept_rate"] = sum(all_spec_accept_rates) / len(all_spec_accept_rates)
    
    # Parse acceptance length from server log if provided
    if args.server_log:
        avg_accept_len = parse_accept_length_from_logs(args.server_log)
        if avg_accept_len:
            spec_stats["avg_accept_length_from_log"] = avg_accept_len
    
    # Evaluate
    correct = 0
    invalid = 0
    
    for i, (response, label) in enumerate(zip(responses, labels)):
        pred = extract_boxed_answer(response)
        
        if pred is None:
            invalid += 1
            if i < 5:  # Print first few failures for debugging
                print(f"\n[{i}] No \\boxed{{}} found. Response snippet: {response[:200] if response else '(empty)'}...")
        else:
            try:
                # Compare as floats to handle decimal answers
                if float(pred) == float(label):
                    correct += 1
                elif i < 5:
                    print(f"\n[{i}] Wrong: pred={pred}, label={label}")
            except ValueError:
                if pred == label:
                    correct += 1
                else:
                    invalid += 1
    
    accuracy = correct / len(questions) * 100
    
    print(f"\n{'='*50}")
    print(f"GSM8K Benchmark Results (DFlash format)")
    print(f"{'='*50}")
    print(f"Questions: {len(questions)}")
    print(f"Correct: {correct}")
    print(f"Invalid (no \\boxed{{}}): {invalid}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Latency: {latency:.2f}s")
    print(f"Output token throughput: {total_completion_tokens/latency:.2f} tok/s")
    print(f"Throughput: {len(questions)/latency:.2f} q/s")
    print(f"{'='*50}")
    print(f"Token Statistics")
    print(f"{'='*50}")
    print(f"Total prompt tokens: {total_prompt_tokens}")
    print(f"Total completion tokens: {total_completion_tokens}")
    print(f"Total tokens: {total_tokens}")
    print(f"Prompt throughput: {total_prompt_tokens/latency:.2f} tok/s")
    print(f"Total throughput: {total_tokens/latency:.2f} tok/s")
    print(f"Avg completion tokens/question: {total_completion_tokens/len(questions):.1f}")
    
    # Print speculative decoding stats if available
    if spec_stats:
        print(f"{'='*50}")
        print(f"Speculative Decoding Statistics")
        print(f"{'='*50}")
        for key, value in spec_stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
    else:
        print(f"\n(Tip: Use --server-log <path> to parse acceptance length from server logs)")


if __name__ == "__main__":
    main()
