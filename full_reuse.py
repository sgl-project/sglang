#!/usr/bin/env python3
"""
Test FusionRAG using curl requests instead of direct model loading

This file keeps the data processing logic from test_fusionrag_reflect.py
but replaces all model loading and inference with curl-based API calls.
"""

import json
import os
import sys
import csv
import shutil
import torch
import numpy as np
import requests
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, TypeAlias, Union
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer
from FlagEmbedding import BGEM3FlagModel


class PreprocessScope(Enum):
    """
    Enum to control the scope of document retrieval during preprocessing

    GLOBAL: Retrieve similar documents from ALL examples globally (original behavior)
    PER_EXAMPLE: Retrieve similar documents only within each example's documents
    SKIP_UNTESTED: Skip retrieval for documents from untested examples (should_test=False)
    """
    GLOBAL = "global"
    PER_EXAMPLE = "per_example"
    SKIP_UNTESTED = "skip_untested"


def load_system_prompt(model_family: str, dataset_type: str = "2wikimqa") -> str:
    """
    Load system prompt from config file
    """
    # Special handling for locomo dataset - use conversation-based prompt
    
    return "\n\n<｜begin▁of▁sentence｜>You are a helpful assistant. Based on the provided conversation history, answer the question accurately and concisely.\n"




def prepare_reflect_data(
    data_path: str,
    tokenizer,
    bge_model_path: str,
    model_type: str = 'qwen2',
    topk: int = 10,
    max_main_questions: int = None,
    preprocess: bool = True,
    preprocess_scope: PreprocessScope = PreprocessScope.GLOBAL,
    dataset_name: str = '2wikimqa'
) -> Tuple[List, torch.Tensor, List, List]:
    """
    Prepare data from result_reflect.json with configurable document corpus scope

    Args:
        model_type: Type of model to determine system prompt
        topk: Top-k similar documents for each document
        preprocess: Whether to compute context_rank
        preprocess_scope: Scope of document retrieval
            - GLOBAL: All documents from all questions (original behavior)
            - PER_EXAMPLE: Only retrieve within each example's documents
            - SKIP_UNTESTED: Exclude documents from untested examples (should_test=False)
        dataset_name: Dataset name to select appropriate system prompt

    Returns:
        questions_data: List of dicts for each main question
        system_tensor: Tokenized system prompt
        context_rank: [total_docs x topk] array of similar document indices
        corpus_lens: List of document counts per question
    """
    print(f"Loading dataset from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if max_main_questions:
        dataset = dataset[:max_main_questions]
        print(f"Limited to first {max_main_questions} main questions")

    # Map model_type to model_family for system prompt
    model_family_map = {
        'qwen': 'Qwen2.5',
        'qwen2': 'Qwen2.5',
        'qwen3': 'Qwen3',
        'mistral': 'Mistral',
        'llama': 'Llama',
        'pangu': 'Pangu',
        'deepseek-v32': 'DeepSeek-V3.2',
        
    }
    model_family = model_family_map.get(model_type, 'Qwen2.5')

    # Tokenize system prompt (shared across all questions) - use dataset_name instead of hardcoded "2wikimqa"
    system_prompt = load_system_prompt(model_family, dataset_name)
    system_tokens = tokenizer.encode(system_prompt, add_special_tokens=True)
    system_tensor = torch.tensor(system_tokens, dtype=torch.long)

    # STEP 1: Build document corpus based on preprocess_scope
    print("\n" + "="*80)
    print(f"Building document corpus with scope: {preprocess_scope.value}")
    print("="*80)

    global_corpus = []  # Documents based on scope
    corpus_lens = []  # Number of docs per question
    questions_data = []

    # First pass: collect all documents globally and build question metadata
    for main_q_idx, data_item in enumerate(dataset):
        main_question = data_item["question"]
        main_answer = data_item["answer"]
        intermediate_context = data_item.get("intermediate_context", [])

        question_docs = []  # Documents for THIS question only
        doc_to_idx = {}  # Local doc -> chunk_id mapping for this question
        sub_questions_info = []

        # Check if this main question should be tested
        should_test_main_question = True

        for sub_q_idx, sub_q in enumerate(intermediate_context):
            docs = sub_q.get("retrieve docs", [])
            doc_chunk_ids = []  # chunk_ids for this sub-question (local to this question)

            for doc in docs:
                if doc not in doc_to_idx:
                    # New document for this question
                    question_docs.append(doc)
                    chunk_id = len(question_docs)  # chunk_id starts from 1
                    doc_to_idx[doc] = chunk_id
                    doc_chunk_ids.append(chunk_id)
                else:
                    # Document already seen in this question
                    doc_chunk_ids.append(doc_to_idx[doc])

            # Remove "Intermediate queryXXX:" prefix from query
            query = sub_q['query']
            if query.startswith("Intermediate query"):
                # Find the colon and extract text after it
                colon_pos = query.find(":")
                if colon_pos != -1:
                    query = query[colon_pos + 1:].strip()

            # Remove "Intermediate answerXXX:" prefix from answer
            answer = sub_q['answer']
            if answer.startswith("Intermediate answer"):
                # Find the colon and extract text after it
                colon_pos = answer.find(":")
                if colon_pos != -1:
                    answer = answer[colon_pos + 1:].strip()

            # Check if any sub-question has problematic answer
            # If so, skip the entire main question
            if "No relevant information found" in answer or "没有相关信息" in answer:
                should_test_main_question = False

            sub_questions_info.append({
                'query': query,
                'answer': answer,
                'chunk_ids': doc_chunk_ids,  # chunk_ids for docs used by this sub-question
            })

        print(f"  Main question {main_q_idx + 1}: {len(question_docs)} unique documents, {len(sub_questions_info)} sub-questions")

        # Tokenize documents for this main question
        doc_tensors = []

        # 对 locomo 数据集进行特殊处理
        if 'locomo' in dataset_name:
            # locomo 数据集：去掉 "Document:" 前缀，按 conversation time 排序
            import re
            from datetime import datetime

            def parse_conversation_time(doc_text):
                """提取 conversation time 并解析为 datetime 对象"""
                match = re.search(r'Conversation Time: (.+?)\.', doc_text)
                if match:
                    time_str = match.group(1)
                    try:
                        # 解析时间格式: "7:55 pm on 9 June, 2023"
                        dt = datetime.strptime(time_str, '%I:%M %p on %d %B, %Y')
                        return dt
                    except:
                        pass
                return None

            # 给每个文档添加时间戳用于排序
            docs_with_time = []
            for doc in question_docs:
                dt = parse_conversation_time(doc)
                docs_with_time.append((f"{doc}\n", dt))
            # 按时间排序（没有时间戳的放在最后）
            docs_with_time.sort(key=lambda x: x[1] if x[1] is not None else datetime.max)

            # 按排序后的顺序 tokenize（只加换行符，不加 "Document:" 前缀）
            for doc, _ in docs_with_time:
                doc_tokens = tokenizer.encode(doc_text, add_special_tokens=False)
                doc_tensor = torch.tensor(doc_tokens, dtype=torch.long)
                doc_tensors.append(doc_tensor)
            docs_changed = docs_with_time
        else:
            docs_changed = []
            # 其他数据集：保留 "Document:" 前缀
            for doc in question_docs:
                doc_text = f"Document: {doc}\n"
                doc_tokens = tokenizer.encode(doc_text, add_special_tokens=False)
                doc_tensor = torch.tensor(doc_tokens, dtype=torch.long)
                doc_tensors.append(doc_tensor)
                docs_changed.append(doc_text)

        # Add this question's docs to global corpus based on scope
        # For SKIP_UNTESTED, only add docs if should_test is True
        if preprocess_scope == PreprocessScope.SKIP_UNTESTED:
            if should_test_main_question:
                global_corpus.extend(question_docs)
                corpus_lens.append(len(question_docs))
            else:
                corpus_lens.append(0)  # No docs added for this question
        else:
            # GLOBAL and PER_EXAMPLE: add all docs
            global_corpus.extend(question_docs)
            corpus_lens.append(len(question_docs))

        # 获取 gold_docs（用于 long_decode 模式的支撑材料评估）
        gold_docs = data_item.get('gold_docs', [])

        questions_data.append({
            'main_question': main_question,
            'main_answer': main_answer,
            'sub_questions': sub_questions_info,
            'docs': docs_changed,
            'doc_tensors': doc_tensors,
            'should_test': should_test_main_question,  # Whether to test this main question
            'gold_docs': gold_docs,  # 用于 long_decode 模式的支撑材料评估
        })

    # Statistics
    total_main_q = len(questions_data)
    testable_main_q = sum(1 for q in questions_data if q['should_test'])
    skipped_main_q = total_main_q - testable_main_q

    total_sub_q = sum(len(q['sub_questions']) for q in questions_data)
    testable_sub_q = sum(len(q['sub_questions']) for q in questions_data if q['should_test'])
    skipped_sub_q = total_sub_q - testable_sub_q

    total_docs = sum(len(q['docs']) for q in questions_data)

    print(f"\n{'='*80}")
    print("DATASET STATISTICS")
    print(f"{'='*80}")
    print(f"Total main questions: {total_main_q}")
    print(f"  - Testable: {testable_main_q}")
    print(f"  - Skipped (llm_judge=False or problematic answers): {skipped_main_q}")
    print(f"\nTotal sub-questions: {total_sub_q}")
    print(f"  - Testable: {testable_sub_q}")
    print(f"  - Skipped: {skipped_sub_q}")
    print(f"\nTotal documents (across all questions): {total_docs}")
    print(f"{'='*80}")

    # STEP 2: Build FAISS index and compute context_rank based on scope
    context_rank = []
    if preprocess and len(global_corpus) > 0:
        print("\n" + "="*80)
        print(f"Computing document similarity with BGE + FAISS (scope: {preprocess_scope.value})...")
        print("="*80)

        import faiss
        from FlagEmbedding import FlagModel

        # Load BGE model
        print(f"Loading BGE model from {bge_model_path}...")
        bgem3 = FlagModel(bge_model_path, use_fp16=True)

        if preprocess_scope == PreprocessScope.PER_EXAMPLE:
            # Build separate FAISS index for EACH example
            print("Building per-example FAISS indices...")
            context_rank = []

            for q_idx, q_data in enumerate(questions_data):
                example_docs = q_data['docs']

                if len(example_docs) == 0:
                    continue

                print(f"  Example {q_idx + 1}: {len(example_docs)} documents")

                # Encode this example's documents
                example_embeddings = bgem3.encode(example_docs)
                example_embeddings = example_embeddings.astype(np.float32)

                # Build FAISS index for this example
                dim = example_embeddings.shape[-1]
                index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
                index.train(example_embeddings)
                index.add(example_embeddings)

                # Search within this example only
                example_embeddings_query = bgem3.encode_queries(example_docs)
                example_embeddings_query = example_embeddings_query.astype(np.float32)
                actual_k = min(topk, len(example_docs))
                score, idx = index.search(example_embeddings_query, k=actual_k)

                # Convert local indices to global indices
                global_offset = sum(corpus_lens[:q_idx])
                global_idx = idx + global_offset

                # Pad to topk if needed
                if actual_k < topk:
                    pad_width = ((0, 0), (0, topk - actual_k))
                    global_idx = np.pad(global_idx, pad_width, mode='constant', constant_values=-1)

                context_rank.append(global_idx)

            if len(context_rank) > 0:
                context_rank = np.vstack(context_rank)
                print(f"Per-example context rank computed: {context_rank.shape}")

        else:
            # GLOBAL or SKIP_UNTESTED: Build single FAISS index for all corpus
            print(f"Encoding {len(global_corpus)} documents for FAISS index...")
            corpus_embeddings = bgem3.encode(global_corpus)
            print(f"Corpus embeddings shape: {corpus_embeddings.shape}")

            # Build FAISS index
            dim = corpus_embeddings.shape[-1]
            index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
            corpus_embeddings = corpus_embeddings.astype(np.float32)
            index.train(corpus_embeddings)
            index.add(corpus_embeddings)
            print(f"FAISS index built with {index.ntotal} vectors")

            # Search for similar documents
            print(f"Searching for top-{topk} similar documents for each document...")
            corpus_embeddings_query = bgem3.encode_queries(global_corpus)
            corpus_embeddings_query = corpus_embeddings_query.astype(np.float32)
            score, idx = index.search(corpus_embeddings_query, k=topk)
            context_rank = idx  # Shape: [total_docs, topk]

            print(f"Context rank computed: {context_rank.shape}")

        bgem3 = None  # Free memory

    return questions_data, system_prompt, system_tensor, context_rank, corpus_lens


# ==================== CURL REQUEST FUNCTIONS ====================

def build_messages(texts):
    """Build messages for API request"""
    if not texts:
        return []

    messages = []
    if len(texts) > 1:
        # 第一个是system消息
        messages.append({"role": "system", "content": texts[0]})
        # 其余都是user消息
        for text in texts[1:]:
            messages.append({"role": "user", "content": text})
    elif len(texts) == 1:
        messages.append({"role": "user", "content": texts[0]})

    return messages


def send_llm_request(
    model: str,
    texts: List[str],
    max_tokens: int = 50,
    temperature: float = 0,
    api_url: str = None,
    recompute_positions: List[int] = None,
    file_name: Optional[str] = None,
    save: Optional[bool] = None,
    load: Optional[bool] = None,
) -> str:
    """
    Send LLM request via curl (HTTP POST)

    Args:
        model: Model name
        texts: List of text strings [system_prompt, doc1, doc2, ..., query]
        max_tokens: Maximum tokens to generate
        temperature: Temperature for sampling
        api_url: API endpoint URL
        recompute_positions: Positions to recompute
        file_name: File name for saving/loading KV cache
        save: Whether to save KV cache
        load: Whether to load KV cache

    Returns:
        Generated text response
    """
    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": build_messages(texts),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "recompute_positions": recompute_positions,
        "file_name": file_name,
        "save": save,
        "load": load,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=5000000)
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except KeyError as e:
        print(f"Response format error: {e}")
        return None


def send_concurrent_inference_requests(
    model_name: str,
    prompt_data: List,
    max_tokens: int,
    api_url: str,
    max_workers: int = 5,
    save: bool = False,
    file_name: str = None
) -> Tuple[List, List]:
    """
    Send concurrent inference requests

    Returns:
        tuple: (answers, ttft_list) - Answer list and TTFT list
    """
    def send_single_inference_request(prompt_item, index, save, file_name):
        """Send single inference request"""
        try:
            result = send_llm_request(
                model_name,
                prompt_item,
                max_tokens,
                api_url=api_url,
                save=save,
                file_name=file_name,
            )
            return True, result, None, index
        except Exception as e:
            print(f"Inference request {index} failed: {e}")
            return False, ' ', None, index

    successful_requests = 0
    failed_requests = 0
    results = [None] * len(prompt_data)
    ttft_list = [None] * len(prompt_data)

    print(f"Starting {len(prompt_data)} concurrent inference requests, max_workers: {max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(send_single_inference_request, prompt_data[i], i, save, f"{file_name}{i+1}" if file_name else None): i
            for i in range(len(prompt_data))
        }

        for future in as_completed(future_to_index):
            original_index = future_to_index[future]
            try:
                success, result, ttft, index = future.result()
                if success:
                    successful_requests += 1
                    results[index] = result if result else ' '
                    ttft_list[index] = ttft
                    print(f"✅ Inference request {index + 1}/{len(prompt_data)} completed")
                else:
                    failed_requests += 1
                    results[index] = ' '
                    ttft_list[index] = None
                    print(f"❌ Inference request {index + 1}/{len(prompt_data)} failed")
            except Exception as e:
                failed_requests += 1
                results[original_index] = ' '
                ttft_list[original_index] = None
                print(f"❌ Inference request {original_index + 1}/{len(prompt_data)} exception: {e}")

    print(f"Concurrent inference completed: successful {successful_requests}, failed {failed_requests}")

    return results, ttft_list


def send_concurrent_cache_requests(
    model_name: str,
    system_prompt: str,
    text_chunks: List,
    api_url: str,
    max_workers: int = 5,
    save: bool = False,
    file_name: str = None,
    load: bool = False
) -> int:
    """
    Send concurrent cache warmup requests

    Returns:
        Number of successful requests
    """
    def send_single_cache_request(text_chunk, save, file_name, load=load):
        """Send single cache warmup request"""
        try:
            result = send_llm_request(
                model_name,
                [system_prompt, text_chunk],
                1,
                api_url=api_url,
                save=save,
                file_name=file_name,
                load=load
            )
            return True, result
        except Exception as e:
            print(f"Cache request failed: {e}")
            return False, str(e)

    successful_requests = 0
    failed_requests = 0

    print(f"Starting {len(text_chunks)} concurrent cache requests, max_workers: {max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {
            executor.submit(
                send_single_cache_request,
                chunk,
                save,
                f"{file_name}{i+1}" if file_name is not None and isinstance(file_name, str) else (file_name[i] if file_name is not None else None),
                load
            ): i
            for i, chunk in enumerate(text_chunks)
        }

        for future in as_completed(future_to_chunk):
            chunk_index = future_to_chunk[future]
            try:
                success, result = future.result()
                if success:
                    successful_requests += 1
                else:
                    failed_requests += 1
            except Exception as e:
                failed_requests += 1

    return successful_requests


def flush_cache(port: int = 30002, host: str = "localhost", timeout: int = 30, verbose: bool = True):
    """Flush server cache"""
    url = f"http://{host}:{port}/flush_cache"

    try:
        print(f"Sending POST request to: {url}")
        response = requests.post(url, timeout=timeout)
        response.raise_for_status()

        if verbose:
            print(f"✅ Success! Status: {response.status_code}")
            try:
                print(f"Response: {response.json()}")
            except:
                print(f"Response: {response.text}")

        return {"success": True}

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"success": False}


# ==================== MAIN FUNCTION ====================

def main(
    # Model and API config
    model_name: str = 'DeepSeek-V3.2',
    model_type: str = 'DeepSeek',
    model_path: str ="/mnt/data/models/DeepSeek-V3.2",
    api_url: str = "http://127.0.0.1:30000/v1/chat/completions",

    # Data config
    data_path: str = '/mnt/data/wjh/data/result_reflect.json',
    data_name: str = 'musique',
    bge_model_path: str = '/mnt/data/models/bge-m3-FP16',
    max_main_questions: int = None,

    # Cache config
    cache_path: str = '/mnt/data/wjh/data/cache_reflect/',

    # Experiment config
    rate: float = 1.0,
    topk: int = 10,
    max_tokens: int = 50,
    revert_rope: bool = True,
    preprocess: bool = True,
    preprocess_scope: PreprocessScope = PreprocessScope.GLOBAL,

    # Execution config
    enable_concurrent: bool = False,
    save: bool = False,
    max_workers: int = 10,
):
    """
    Main function using curl-based API calls

    Args:
        rate: Recompute rate (1.0 = full attention, < 1.0 = partial recompute)
        Other args: same as test_fusionrag_reflect.py
    """

    # Initialize tokenizer (only for data processing)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Prepare data
    questions_data, system_prompt, system_tensor, context_rank, corpus_lens = prepare_reflect_data(
        data_path=data_path,
        tokenizer=tokenizer,
        bge_model_path=bge_model_path,
        model_type=model_type,
        topk=topk,
        max_main_questions=max_main_questions,
        preprocess=preprocess,
        preprocess_scope=preprocess_scope,
        dataset_name=data_name
    )

    # Create save path
    os.makedirs(cache_path, exist_ok=True)
    csv_file = f"{cache_path}/results_rate_{rate}_topk_{topk}_preprocess_{preprocess}.csv"

    # Initialize CSV file
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Question', 'Real Answer', 'Pred Answer'])

    # Flush cache
    flush_cache()

    answer_list = []
    system_len = len(system_tensor)

    # ==================== RATE = 1 BRANCH (Full Attention) ====================
    if rate == 1:
        print("\n" + "="*80)
        print("RUNNING WITH RATE = 1 (FULL ATTENTION)")
        print("="*80)
        # Sequential mode
        for i, q_data in enumerate(questions_data):
            if not q_data['should_test']:
                continue

            # Build full prompt
            full_texts = [q_data['docs'][0] for doc in q_data['docs']]  # TODO: Build proper prompt structure

            save_file_path = f"{cache_path}/fullattention_{i+1}"
            full_texts.insert(0, system_prompt)
            full_texts.append(q_data['main_question'])
            # full_texts.append(f"<｜User｜>{q_data['main_question']}      <｜Assistant｜>        </think>")
            # Send full attention request
            answer = send_llm_request(
                model_name,
                texts=full_texts,
                max_tokens=max_tokens,
                api_url=api_url,
            )

            if "<|im_end|>" in answer:
                answer = answer.split("<|im_end|>")[-1]

            print(f"\n{'='*60}")
            print(f"Question {i+1}: {q_data['main_question']}")
            print(f"Real Answer: {q_data['main_answer']}")
            print(f"Pred Answer: {answer}")
            print(f"{'='*60}")

            answer_list.append(answer)

            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    q_data['main_question'],
                    q_data['main_answer'],
                    answer
                ])

            flush_cache()

    # ==================== RATE != 1 BRANCH (Partial Recompute) ====================
    else:
        print("\n" + "="*80)
        print(f"RUNNING WITH RATE = {rate} (PARTIAL RECOMPUTE)")
        print("="*80)

        # TODO: Implement partial recompute logic
        # This should:
        # 1. Warm up KV cache for each document chunk
        # 2. Compute recompute_index based on rate
        # 3. Send recompute requests with recompute_positions

        for i, q_data in enumerate(questions_data):
            if not q_data['should_test']:
                continue

            # Warm up KV cache
            system_prompt = load_system_prompt('Qwen2.5', data_name)

            for doc_idx, doc in enumerate(q_data['docs']):
                save_file_path = f"{cache_path}/{i+1}_{doc_idx+1}"

                # Send cache warmup request
                send_llm_request(
                    model_name,
                    texts=[system_prompt, doc],
                    max_tokens=1,
                    api_url=api_url,
                    save=save,
                    file_name=save_file_path
                )

            # Compute recompute positions (placeholder - implement actual logic)
            total_docs = len(q_data['docs'])
            num_recompute = int(rate * total_docs)
            recompute_positions = list(range(system_len, system_len + num_recompute))

            # Build full prompt
            full_texts = [system_prompt] + q_data['docs']

            # Send recompute request
            answer = send_llm_request(
                model_name,
                texts=full_texts,
                max_tokens=max_tokens,
                api_url=api_url,
                recompute_positions=recompute_positions
            )

            if "<|im_end|>" in answer:
                answer = answer.split("<|im_end|>")[-1]

            print(f"\n{'='*60}")
            print(f"Question {i+1}: {q_data['main_question']}")
            print(f"Real Answer: {q_data['main_answer']}")
            print(f"Pred Answer: {answer}")
            print(f"Recompute positions: {len(recompute_positions)} tokens")
            print(f"{'='*60}")

            answer_list.append(answer)

            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    q_data['main_question'],
                    q_data['main_answer'],
                    answer
                ])

            flush_cache()

    print("\n" + "="*80)
    print("TESTING COMPLETED")
    print(f"Results saved to: {csv_file}")
    print(f"Total questions processed: {len(answer_list)}")
    print("="*80)


if __name__ == '__main__':
    # Example usage
    main(
        model_name='DeepSeek-V3.2',
        model_type='deepseek-v32',
        model_path='/mnt/data/models/DeepSeek-V3.2',
        api_url="http://127.0.0.1:30000/v1/chat/completions",
        data_path='/mnt/data/wjh/data/result_reflect.json',
        data_name='musique',
        rate=1.0,  # Change this to 0.2 for partial recompute
        topk=10,
        max_tokens=500,
        preprocess=False,
        enable_concurrent=False,
        save=True,
        max_workers=10
    )
