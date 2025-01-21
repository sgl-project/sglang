import copy
import os
import time
import random
import threading
import unittest
from types import SimpleNamespace
from typing import List

import pandas
from transformers import AutoModelForCausalLM, AutoTokenizer

import sglang as sgl
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.simple_eval_common import format_multichoice_question
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    STDOUT_FILENAME,
    STDERR_FILENAME,
    popen_launch_server,
    read_output,
)

MODEL_PATH = DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN
CHAT_PROMPTS = [{"role": "system", "content": "You are a helpful assistant."}, 
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is"}]


class TestBeamSearch(unittest.TestCase):
    def _longest_common_subset(self, arr1: List[List[int]], arr2: List[List[int]]):
        def match(subarr1: List[int], subarr2: List[int]):
            for a, b in zip(subarr1, subarr2):
                if a!=b:
                    return False
            return True
        
        N = len(arr1)
        L = [[0] * (N + 1) for _ in range(N + 1)]
        for i in range(1, N+1):
            for j in range(1, N+1):
                if match(arr1[i-1], arr2[j-1]):
                    L[i][j] = L[i-1][j-1] + 1
                else:
                    L[i][j] = max(L[i-1][j], L[i][j-1])

        return L[N][N]/N

    def _replace_content(new_content):
        new_chat_prompts = copy.deepcopy(CHAT_PROMPTS)
        new_chat_prompts[1]['content'] = new_content
        return new_chat_prompts

    def _run_one_test(self, cur_prompt, tokenizer, max_tokens=5, caching=True, top_logprobs_num=-1):
        if(top_logprobs_num!=-1):
            is_logprob = True
        else:
            is_logprob = False

        llm = sgl.Engine(model_path=MODEL_PATH,
                        disable_radix_cache=not caching,
                        disable_jump_forward=True,
                        disable_overlap_schedule=True)

        sampling_params = {"temperature": 0.0, "max_new_tokens": max_tokens}
        t1 = time.time()
        outputs = llm.generate(cur_prompt, sampling_params=sampling_params, 
                               return_logprob=is_logprob, top_logprobs_num=top_logprobs_num)
        print(f"######time consumption with max_tokens={max_tokens}: {time.time()-t1}######")
        llm.shutdown()

        return [(beam.tokens, tokenizer.decode(beam.tokens)) 
                for output in outputs for beam in output["meta_info"]["beam_search_outputs"].sequences]
    
    def _run_one_bench(self, cur_prompt, tokenizer, max_tokens=[5], caching=True, top_logprobs_num=-1):
        if(top_logprobs_num!=-1):
            is_logprob = True
        else:
            is_logprob = False

        llm = sgl.Engine(model_path=MODEL_PATH,
                        disable_jump_forward=True,
                        disable_overlap_schedule=True)
        
        # warmup
        sampling_params = {"temperature": 0.0, "max_new_tokens": 5}
        warmup_prompt = CHAT_PROMPTS
        warmup_prompt = tokenizer.apply_chat_template(warmup_prompt, tokenize=False, add_generation_prompt=True)
        sampling_params = {"temperature": 0.0, "max_new_tokens": 5}
        output = llm.generate([warmup_prompt], sampling_params)

        time_consume = []
        for max_token in max_tokens:
            sampling_params = {"temperature": 0.0, "max_new_tokens": max_token}
            t1 = time.time()
            outputs = llm.generate(cur_prompt, sampling_params=sampling_params, 
                                return_logprob=is_logprob, top_logprobs_num=top_logprobs_num)
            t2 = time.time()
            print(f"######time consumption with max_tokens={max_token}: {t2-t1}######")
            time_consume.append(t2-t1)
        llm.shutdown()

        return time_consume

    def _bench_vllm_fork(self, cur_prompt, tokenizer, max_tokens=[5], caching=True, top_logprobs_num=-1):
        from sglang.srt.beam_search import BeamSearchSequence, BeamSearchList, sort_by_beam_search_score
        import itertools
        from typing import List, Tuple

        if(top_logprobs_num!=-1):
            is_logprob = True
        else:
            is_logprob = False

        llm = sgl.Engine(model_path=MODEL_PATH,
                        disable_jump_forward=True,
                        disable_overlap_schedule=True)
        
        # warmup
        sampling_params = {"temperature": 0.0, "max_new_tokens": 5}
        warmup_prompt = CHAT_PROMPTS
        warmup_prompt = tokenizer.apply_chat_template(warmup_prompt, tokenize=False, add_generation_prompt=True)
        sampling_params = {"temperature": 0.0, "max_new_tokens": 5}
        output = llm.generate([warmup_prompt], sampling_params)

        time_consume = []
        for max_token in max_tokens:
            sampling_params = {"temperature": 0.0, "max_new_tokens": 1}
            '''
            fork from: https://github.com/vllm-project/vllm/blob/5f0ec3935a0118fee8cf2764728f765c8cc53d2a/vllm/entrypoints/llm.py#L507'''
            prompt_token_indices = [tokenizer.encode(prompt) for prompt in cur_prompt]
            requests = [
                BeamSearchList(
                    incompleted=[BeamSearchSequence(last_token=prompt_token_ids[-1], 
                                                    tokens=prompt_token_ids,
                                                    text=cur_prompt[i])]
                ) for i, prompt_token_ids in enumerate(prompt_token_indices)
            ]
            beam_width = 5
            t1 = time.time()
            for i in range(max_token):
                pos = [0] + list(
                    itertools.accumulate(
                        len(req.incompleted) for req in requests))
                instance_start_and_end: List[Tuple[int, int]] = list(
                    zip(pos[:-1], pos[1:]))
                
                outputs = llm.generate(input_ids=[beam.tokens for req in requests for beam in req.incompleted], 
                                    sampling_params=sampling_params, return_logprob=is_logprob, 
                                    top_logprobs_num=top_logprobs_num)
                
                for (start, end), req in zip(instance_start_and_end, requests):
                    incompleted = []
                    for i, (cur_beam, cur_output) in enumerate(zip(req.incompleted, outputs[start:end])):
                        beam_tokens = [token_ids for logprob, token_ids, _ in cur_output['meta_info']['output_top_logprobs'][0]]
                        beam_logprobs = [logprob for logprob, token_ids, _ in cur_output['meta_info']['output_top_logprobs'][0]]
                        
                        for token, logprob in zip(beam_tokens, beam_logprobs):
                            new_beam = BeamSearchSequence(
                                last_token=token, tokens=cur_beam.tokens+[token],
                                # text=cur_beam.text+text,
                                cum_logprob=cur_beam.cum_logprob+logprob
                            )
                            if token!=tokenizer.eos_token_id:
                                incompleted.append(new_beam)
                            else:
                                req.completed.append(new_beam)
                    incompleted = sorted(incompleted, key=sort_by_beam_search_score, reverse=True)
                    req.incompleted = incompleted[:beam_width]
            t2 = time.time()
            print(f"######time consumption with max_tokens={max_token}: {t2-t1}######")
            time_consume.append(t2-t1)
        llm.shutdown()

        return time_consume


    def _run_transformer(self, cur_prompt, tokenizer, max_tokens=5, caching=True, top_logprobs_num=-1):
        llm = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to("cuda")
        llm.eval()

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        tokenizer.padding_side = "left"
        input_ids = tokenizer(cur_prompt, return_tensors="pt", padding=True).to("cuda")
        outputs = llm.generate(
                    temperature=0.00001,
                    max_new_tokens=max_tokens,
                    num_beams=top_logprobs_num,
                    num_return_sequences=int(top_logprobs_num/2),
                    early_stopping=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    do_sample=False, # -inf score if True
                    **input_ids
                )
        prompt_len = len(input_ids[0])
        ret = [(output[prompt_len:].tolist(), tokenizer.decode(output[prompt_len:])) for output in outputs.sequences]
        return ret
    
    def _bench_transformer(self, cur_prompt, tokenizer, max_tokens=[5], caching=True, top_logprobs_num=-1):
        llm = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to("cuda")
        llm.eval()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        tokenizer.padding_side = "left"

        #warm up
        warmup_prompt = CHAT_PROMPTS
        warmup_prompt = [tokenizer.apply_chat_template([prompt], tokenize=False, add_generation_prompt=True) for prompt in warmup_prompt]
        warmup_prompt = tokenizer(warmup_prompt, return_tensors="pt", padding=True).to("cuda")
        outputs = llm.generate(
                    temperature=0.00001,
                    max_new_tokens=5,
                    num_beams=top_logprobs_num,
                    num_return_sequences=int(top_logprobs_num/2),
                    early_stopping=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    do_sample=False, # -inf score if True
                    **warmup_prompt
                )

        input_ids = tokenizer(cur_prompt, return_tensors="pt", padding=True).to("cuda")
        time_consume = []
        for max_token in max_tokens:
            t1 = time.time()
            outputs = llm.generate(
                        temperature=0.00001,
                        max_new_tokens=max_token,
                        num_beams=top_logprobs_num,
                        num_return_sequences=int(top_logprobs_num/2),
                        early_stopping=True,
                        return_dict_in_generate=True,
                        output_scores=True,
                        do_sample=False, # -inf score if True
                        **input_ids
                    )
            t2 = time.time()
            print(f"######time consumption with max_tokens={max_token}: {t2-t1}######")
            time_consume.append(t2-t1)
        return time_consume

    def test_beam_search_accuracy_by_offline_mmlu(self):
        os.environ["SGLANG_TEST_BEAM_WIDTH"] = "5"
        filename = "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv"
        df = pandas.read_csv(filename)
        examples = [row.to_dict() for _, row in df.iterrows()]
        examples = random.Random(0).sample(examples, 32)

        prompt_messages = [
                {"role": "user", "content": format_multichoice_question(row)}
                    for row in examples
            ]
        MODEL_PATH = DEFAULT_MODEL_NAME_FOR_TEST
        tokenizer = get_tokenizer(MODEL_PATH)

        cur_prompt = [tokenizer.apply_chat_template([prompt], tokenize=False, add_generation_prompt=True) for prompt in prompt_messages]

        beam_width = 5
        params = {"max_tokens": 5, "caching": True, "top_logprobs_num": beam_width*2}

        # our beam search
        output_text = self._run_one_test(cur_prompt, tokenizer, **params)

        # transformers beam search
        transformer_text = self._run_transformer(cur_prompt, tokenizer, **params)

        # longest common subset
        match_rates = []
        for i in range(len(cur_prompt)):
            match_rate = self._longest_common_subset(
                [
                    text[0]
                    for text in transformer_text[i * beam_width : (i + 1) * beam_width]
                ],
                [
                    text[0]
                    for text in output_text[i * beam_width : (i + 1) * beam_width]
                ],
            )
            match_rates.append(match_rate)
        # print(f"Average Match Rate: {sum(match_rates)/len(match_rates)*100}%")
        assert sum(match_rates)/len(match_rates)>=0.8

    def bench_beam_search_overhead_by_offline_mmlu(self):
        filename = "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv"
        df = pandas.read_csv(filename)
        examples = [row.to_dict() for _, row in df.iterrows()]
        examples = random.Random(0).sample(examples, 32)

        prompt_messages = [
                {"role": "user", "content": format_multichoice_question(row)}
                    for row in examples
            ]
        MODEL_PATH = DEFAULT_MODEL_NAME_FOR_TEST
        tokenizer = get_tokenizer(MODEL_PATH)

        cur_prompt = [tokenizer.apply_chat_template([prompt], tokenize=False, add_generation_prompt=True) for prompt in prompt_messages]

        beam_width = 5
        
        os.environ["SGLANG_TEST_BEAM_WIDTH"] = "5"
        os.environ["SGLANG_TEST_BEAM_FIXED_MAX_TOKEN"] = "1"
        time_consume = self._run_one_bench(cur_prompt, tokenizer, 
                                          max_tokens=[8, 16, 32, 64, 128, 512, 1024],
                                          caching=False,
                                          top_logprobs_num=beam_width*2)

        time_consume = self._bench_transformer(cur_prompt, tokenizer, 
                                          max_tokens=[8, 16, 32, 64, 128, 512, 1024],
                                          caching=False,
                                          top_logprobs_num=beam_width*2)

        os.environ["SGLANG_TEST_BEAM_WIDTH"] = "0"
        time_consume = self._bench_vllm_fork(cur_prompt, tokenizer, 
                                          max_tokens=[8, 16, 32, 64, 128, 512, 1024],
                                          caching=False,
                                          top_logprobs_num=beam_width*2)

    def test_beam_search_memory_leak_via_mmlu(self):
        def workload_func(base_url, model):
            # Run the eval
            args = SimpleNamespace(
                base_url=base_url,
                model=model,
                eval_name="mmlu",
                num_examples=128,
                num_threads=16,
            )
            try:
                metrics = run_eval(args)
                assert metrics["score"] >= 0.65, f"{metrics=}"
            finally:
                pass

        # chunked prefilling is supported
        other_args = [
            "--log-level",
            "debug",
            "--trust-remote-code",
            "--disable-overlap-schedule",
            "--disable-jump-forward",
        ]

        model = DEFAULT_MODEL_NAME_FOR_TEST
        port = random.randint(4000, 5000)
        base_url = f"http://127.0.0.1:{port}"

        # Create files and launch the server
        stdout = open(STDOUT_FILENAME, "w")
        stderr = open(STDERR_FILENAME, "w")
        env = os.environ.copy()
        env["SGLANG_TEST_BEAM_WIDTH"] = "5"
        env["SGLANG_TEST_BEAM_KTH_AS_OUTPUT"] = "0"
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            return_stdout_stderr=(stdout, stderr),
            env=env
        )

        # Launch a thread to stream the output
        output_lines = []
        t = threading.Thread(target=read_output, args=(output_lines,))
        t.start()

        # Run the workload
        workload_func(base_url, model)

        # Clean up everything
        kill_process_tree(process.pid)
        kill_process_tree(process.pid)
        stdout.close()
        stderr.close()
        if os.path.exists(STDOUT_FILENAME):
            os.remove(STDOUT_FILENAME)
        if os.path.exists(STDERR_FILENAME):
            os.remove(STDERR_FILENAME)
        t.join()
        # Assert success
        has_new_server = False
        has_leak = False
        has_abort = False
        for line in output_lines:
            if "The server is fired" in line:
                has_new_server = True
            if "leak" in line:
                has_leak = True
            if "Abort" in line:
                has_abort = True
        assert has_new_server
        assert not has_leak

if __name__ == "__main__":
    unittest.main()