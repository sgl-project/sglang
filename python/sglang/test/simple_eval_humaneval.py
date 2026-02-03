# Adapted from https://github.com/openai/simple-evals/

"""
HumanEval: Evaluating Large Language Models Trained on Code
Mark Chen and Jerry Tworek and Heewoo Jun and Qiming Yuan and Henrique Ponde de Oliveira Pinto and Jared Kaplan and Harri Edwards and Yuri Burda and Nicholas Joseph and Greg Brockman and Alex Ray and Raul Puri and Gretchen Krueger and Michael Petrov and Heidy Khlaaf and Girish Sastry and Pamela Mishkin and Brooke Chan and Scott Gray and Nick Ryder and Mikhail Pavlov and Alethea Power and Lukasz Kaiser and Mohammad Bavarian and Clemens Winter and Philippe Tillet and Felipe Petroski Such and Dave Cummings and Matthias Plappert and Fotios Chantzis and Elizabeth Barnes and Ariel Herbert-Voss and William Hebgen Guss and Alex Nichol and Alex Paino and Nikolas Tezak and Jie Tang and Igor Babuschkin and Suchir Balaji and Shantanu Jain and William Saunders and Christopher Hesse and Andrew N. Carr and Jan Leike and Josh Achiam and Vedant Misra and Evan Morikawa and Alec Radford and Matthew Knight and Miles Brundage and Mira Murati and Katie Mayer and Peter Welinder and Bob McGrew and Dario Amodei and Sam McCandlish and Ilya Sutskever and Wojciech Zaremba
https://arxiv.org/abs/2107.03374 https://github.com/openai/human-eval/
"""

import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np

try:
    from human_eval.data import read_problems
    from human_eval.evaluation import estimate_pass_at_k
    from human_eval.execution import check_correctness  # , unsafe_execute
except (ImportError, ModuleNotFoundError):
    print("\nPlease install human-eval at https://github.com/openai/human-eval.\n")
    raise

from sglang.test import simple_eval_common as common
from sglang.test.simple_eval_common import (
    HTML_JINJA,
    Eval,
    EvalResult,
    SamplerBase,
    SingleEvalResult,
)


def evaluate_functional_correctness(
    sample: Dict[str, str],
    completions: List[str],
    n_workers: int = 4,
    timeout: float = 3.0,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """

    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i, completion in enumerate(completions):
            args = (sample, completion, timeout, i)
            future = executor.submit(check_correctness, *args)
            futures.append(future)
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    passed = [int(r["passed"]) for r in results]
    return passed


class HumanEval(Eval):
    def __init__(
        self,
        num_examples: Optional[int],
        num_threads: int,
        num_samples_per_task: int = 5,
        ks_passes: List[int] = [1, 2, 5],
        timeout: int = 120,
    ):
        self.seed = 0
        self.examples = read_problems()
        self.examples = list(self.examples.values())

        self._num_examples = num_examples
        if self._num_examples:
            self.examples = random.Random(self.seed).sample(self.examples, num_examples)
        self._num_samples_per_task = num_samples_per_task
        self._ks_passes = ks_passes
        self._timeout = timeout
        self._num_threads = num_threads

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        instruction = "Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\n"

        def find_code(completion):
            completion = completion or ""
            pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
            matches = pattern.findall(completion)
            extracted_answer = matches[0] if len(matches) >= 1 else completion
            extracted_answer = extracted_answer[
                extracted_answer.find(":\n    ") + 2 :
            ]  # remove signature
            return extracted_answer

        def fn(sample: Dict[str, str]):
            prompt_messages = [
                sampler._pack_message(
                    role="user", content=instruction + sample["prompt"]
                )
            ]
            completions = [
                find_code(sampler(prompt_messages))
                for _ in range(self._num_samples_per_task)
            ]
            results = evaluate_functional_correctness(sample, completions)
            total = len(results)
            correct = sum(results)
            score = sum(results) / len(results)
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=completions[0], role="assistant"),
                score=score,
                correct_answer=[1] * len(results),
                extracted_answer=results,
            )
            convo = prompt_messages + [
                dict(content=completion, role="assistant") for completion in completions
            ]
            return SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                metrics={
                    f"pass@{k}": estimate_pass_at_k([total], [correct], k)
                    # this will be aggregated so no need of .mean()
                    for k in self._ks_passes
                    if total >= k
                },
            )

        results = common.map_with_progress(
            fn, self.examples, num_threads=self._num_threads
        )
        
        aggregated = common.aggregate_results(results)
        
        # Compute acceptance length (for speculative decoding)
        # Access metadata from sampler if available
        if hasattr(sampler, 'last_response_metadata') and sampler.last_response_metadata:
            metadata_list = sampler.last_response_metadata
            has_verify = any('spec_verify_ct' in meta for meta in metadata_list)
            
            if has_verify:
                num_sd_tokens = 0
                num_sd_verify = 0
                num_sd_answers = 0
                num_non_sd_answers = 0
                acceptance_lengths = []  # Per-question acceptance lengths
                
                for meta in metadata_list:
                    verify_ct = meta.get('spec_verify_ct') or 0
                    
                    # Use sd_completion_tokens if available (excludes non-SD tokens)
                    # Otherwise fall back to completion_tokens for backwards compatibility
                    if 'sd_completion_tokens' in meta:
                        sd_tokens = meta['sd_completion_tokens'] or 0
                    else:
                        # Fallback: only count completion_tokens if SD was used
                        sd_tokens = meta.get('completion_tokens', 0) if verify_ct > 0 else 0
                    
                    # Only count answers where SD was actually used
                    if verify_ct > 0:
                        num_sd_tokens += sd_tokens
                        num_sd_verify += verify_ct
                        num_sd_answers += 1
                        acceptance_lengths.append(sd_tokens / verify_ct)
                    else:
                        num_non_sd_answers += 1
                
                print(f"[DEBUG] SD answers: {num_sd_answers}, Non-SD answers: {num_non_sd_answers}")
                print(f"[DEBUG] SD tokens: {num_sd_tokens}, SD verify steps: {num_sd_verify}")
                
                if acceptance_lengths:
                    print(f"[DEBUG] Per-question acceptance length:")
                    print(f"  Min: {min(acceptance_lengths):.2f}")
                    print(f"  Max: {max(acceptance_lengths):.2f}")
                    print(f"  Median: {np.median(acceptance_lengths):.2f}")
                    print(f"  Mean: {np.mean(acceptance_lengths):.2f}")
                
                if num_sd_verify > 0:
                    accept_length = num_sd_tokens / num_sd_verify
                    print(f"Acceptance length: {accept_length:.3f}")
                else:
                    accept_length = 1.0  # No SD was used
                    print(f"Acceptance length: {accept_length:.3f} (No SD used)")
        
        return aggregated
