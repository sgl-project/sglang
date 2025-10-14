import pickle
import random
from typing import List

from tqdm.asyncio import tqdm

from sglang.benchmark.datasets.common import BaseDatasetLoader, DatasetRow
from sglang.benchmark.utils import gen_prompt, get_gen_prefix_cache_path


class GeneratedSharedPrefixLoader(BaseDatasetLoader):
    def load(self) -> List[DatasetRow]:
        assert not self.args.tokenize_prompt
        """Generate benchmark requests with shared system prompts using random tokens and caching."""
        num_groups = self.args.gsp_num_groups
        prompts_per_group = self.args.gsp_prompts_per_group
        system_prompt_len = self.args.gsp_system_prompt_len
        question_len = self.args.gsp_question_len
        output_len = self.args.gsp_output_len

        cache_path = get_gen_prefix_cache_path(self.args, self.tokenizer)

        # Try to load from cache first
        if cache_path.exists():
            print(f"\nLoading cached generated input data from {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        print("\nGenerating new input data...")

        # Generate system prompts for each group
        system_prompts = []
        for _ in range(num_groups):
            system_prompt = gen_prompt(self.tokenizer, system_prompt_len)
            system_prompts.append(system_prompt)

        # Generate questions
        questions = []
        for _ in range(num_groups * prompts_per_group):
            question = gen_prompt(self.tokenizer, question_len)
            questions.append(question)

        # Combine system prompts with questions
        input_requests = []
        total_input_tokens = 0
        total_output_tokens = 0

        for group_idx in tqdm(range(num_groups), desc="Generating system prompt"):
            system_prompt = system_prompts[group_idx]
            for prompt_idx in tqdm(
                range(prompts_per_group), desc="Generating questions", leave=False
            ):
                question = questions[group_idx * prompts_per_group + prompt_idx]
                full_prompt = f"{system_prompt}\n\n{question}"
                prompt_len = len(self.tokenizer.encode(full_prompt))

                input_requests.append(
                    DatasetRow(
                        prompt=full_prompt, prompt_len=prompt_len, output_len=output_len
                    )
                )
                total_input_tokens += prompt_len
                total_output_tokens += output_len

        # Shuffle questions
        random.shuffle(input_requests)

        # Print statistics
        print(f"\nGenerated shared prefix dataset statistics:")
        print(f"Number of groups: {num_groups}")
        print(f"Prompts per group: {prompts_per_group}")
        print(f"Total prompts: {len(input_requests)}")
        print(f"Total input tokens: {total_input_tokens}")
        print(f"Total output tokens: {total_output_tokens}")
        print(
            f"Average system prompt length: {sum(len(self.tokenizer.encode(sp)) for sp in system_prompts) / len(system_prompts):.1f} tokens"
        )
        print(
            f"Average question length: {sum(len(self.tokenizer.encode(q)) for q in questions) / len(questions):.1f} tokens\n"
        )

        # Save to cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Caching generated input data to {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(input_requests, f)

        return input_requests
