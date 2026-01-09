#!/usr/bin/env python3
"""
GSM8K benchmark following DFlash format:
- Uses chat template with enable_thinking=False
- Uses \boxed{} format for answers
- Sends to SGLang server
"""

import argparse
import re
import time
from datasets import load_dataset
from transformers import AutoTokenizer
import openai


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-questions", type=int, default=100)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    args = parser.parse_args()
    
    # Load tokenizer for chat template
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Load GSM8K dataset
    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    
    # DFlash prompt format
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
    
    # Setup OpenAI client pointing to SGLang
    client = openai.Client(
        base_url=f"http://localhost:{args.port}/v1",
        api_key="EMPTY"
    )
    
    # Run requests
    print(f"Running with parallel={args.parallel}...")
    
    import concurrent.futures
    
    def process_question(idx):
        prompt = questions[idx]
        try:
            response = client.completions.create(
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=0,
                stop=["<|im_end|>", "<|endoftext|>"]
            )
            text = response.choices[0].text
            # Get token counts from response usage
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            return idx, text, prompt_tokens, completion_tokens
        except Exception as e:
            print(f"Error on question {idx}: {e}")
            return idx, "", 0, 0
    
    tic = time.perf_counter()
    
    responses = [None] * len(questions)
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = [executor.submit(process_question, i) for i in range(len(questions))]
        for future in concurrent.futures.as_completed(futures):
            idx, text, prompt_tokens, completion_tokens = future.result()
            responses[idx] = text
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            # Progress
            done = sum(1 for r in responses if r is not None)
            print(f"\rProgress: {done}/{len(questions)}", end="", flush=True)
    
    print()
    latency = time.perf_counter() - tic
    total_tokens = total_prompt_tokens + total_completion_tokens
    
    # Evaluate
    correct = 0
    invalid = 0
    
    for i, (response, label) in enumerate(zip(responses, labels)):
        pred = extract_boxed_answer(response)
        
        if pred is None:
            invalid += 1
            if i < 5:  # Print first few failures for debugging
                print(f"\n[{i}] No \\boxed{{}} found. Response snippet: {response[:200]}...")
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


if __name__ == "__main__":
    main()

