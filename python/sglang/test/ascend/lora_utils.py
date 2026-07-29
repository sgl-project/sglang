import dataclasses
import random
from typing import List, Optional

import torch

from sglang.test.ascend.test_ascend_utils import (
    QWEN3_4B_LORA_V2_WEIGHTS_PATH,
    QWEN3_4B_LORA_ZH_WEBNOVELTY_V0_0_WEIGHTS_PATH,
    QWEN3_4B_WEIGHTS_PATH,
)
from sglang.test.runners import HFRunner, SRTRunner
from sglang.test.test_utils import calculate_rouge_l


@dataclasses.dataclass
class LoRAAdaptor:
    name: str
    prefill_tolerance: float = None
    decode_tolerance: float = None
    rouge_l_tolerance: float = None


@dataclasses.dataclass
class LoRAModelCase:
    base: str
    adaptors: List[LoRAAdaptor]
    tp_size: int = 1
    prefill_tolerance: float = 1e-1
    decode_tolerance: float = 1e-1
    rouge_l_tolerance: float = 1.0
    max_loras_per_batch: int = 1
    max_loaded_loras: Optional[int] = None
    skip_long_prompt: bool = False

    def __post_init__(self):
        if len(self.adaptors) > self.max_loras_per_batch:
            raise ValueError(
                f"For base '{self.base}', number of adaptors ({len(self.adaptors)}) "
                f"must be <= max_loras_per_batch ({self.max_loras_per_batch})"
            )


TORCH_DTYPES = [torch.float16]

LORA_MODELS_QWEN3 = [
    LoRAModelCase(
        base=QWEN3_4B_WEIGHTS_PATH,
        adaptors=[
            LoRAAdaptor(
                name=QWEN3_4B_LORA_V2_WEIGHTS_PATH,
                prefill_tolerance=3e-1,
            ),
            LoRAAdaptor(
                name=QWEN3_4B_LORA_ZH_WEBNOVELTY_V0_0_WEIGHTS_PATH,
                prefill_tolerance=3e-1,
            ),
        ],
        max_loras_per_batch=2,
        max_loaded_loras=64,
    ),
]

TEST_MULTIPLE_BATCH_PROMPTS = [
    """
    ### Instruction:
    Tell me about llamas and alpacas
    ### Response:
    Llamas are large, long-necked animals with a woolly coat. They have two toes on each foot instead of three like other camelids (camels, dromedaries). Llamas live in the Andean mountains of South America where they graze on grasses and shrubs. Alpaca is another name for domesticated llama. The word "alpaca" comes from an Incan language meaning "golden fleece." Alpacas look very similar to llamas but are smaller than their wild relatives. Both species were used by ancient people as pack animals and for meat. Today both llamas and alpacas are raised primarily for their fiber which can be spun into yarn or knitted into clothing.
    ### Question 2:
    What do you know about llamas?
    ### Answer:
    """,
    """
    ### Instruction:
    Write a poem about the transformers Python library.
    Mention the word "large language models" in that poem.
    ### Response:
    The Transformers are large language models,
    They're used to make predictions on text.
    """,
    "AI is a field of computer science focused on",
    "Computer science is the study of",
    "Write a short story.",
    "What are the main components of a computer?",
]


def create_multiple_batch_test_samples(
    prompts: List[str], lora_adapter_paths: List[str]
):
    random.seed(42)

    test_cases = [
        (
            [
                random.choice(prompts),
                random.choice(prompts),
                random.choice(prompts),
            ],
            [
                None,
                lora_adapter_paths[0],
                lora_adapter_paths[1],
            ],
        ),
        (
            [
                random.choice(prompts),
                random.choice(prompts),
                random.choice(prompts),
            ],
            [lora_adapter_paths[0], lora_adapter_paths[1], None],
        ),
    ]

    return test_cases


def ensure_reproducibility():
    seed = 42
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def run_lora_multiple_batch_on_model_cases(
    model_cases: List[LoRAModelCase],
    use_spec_decoding: bool = False,
    attention_backend: str = "ascend",
    disable_cuda_graph: bool = True,
    enable_deterministic_inference: bool = False,
    disable_radix_cache: bool = True,
    enable_lora_overlap_loading: Optional[bool] = None,
):
    if not torch.npu.is_available():
        raise RuntimeError(
            "NPU device not available. Please ensure NPU environment is properly configured."
        )

    for model_case in model_cases:
        for torch_dtype in TORCH_DTYPES:
            max_new_tokens = 32
            base_path = model_case.base
            lora_adapter_paths = [a.name for a in model_case.adaptors]
            assert len(lora_adapter_paths) >= 2

            batches = create_multiple_batch_test_samples(
                TEST_MULTIPLE_BATCH_PROMPTS, lora_adapter_paths
            )

            print(
                f"\n========== Testing multiple batches on base '{base_path}', dtype={torch_dtype} ---"
            )

            # Initialize runners
            ensure_reproducibility()
            spec_args = (
                {}
                if not use_spec_decoding
                else {
                    "speculative_algorithm": "NGRAM",
                    "speculative_num_draft_tokens": 5,
                }
            )
            srt_runner = SRTRunner(
                base_path,
                torch_dtype=torch_dtype,
                model_type="generation",
                lora_paths=[lora_adapter_paths[0], lora_adapter_paths[1]],
                enable_lora_overlap_loading=enable_lora_overlap_loading,
                max_loras_per_batch=len(lora_adapter_paths) + 1,
                max_loaded_loras=model_case.max_loaded_loras,
                sleep_on_idle=True,  # Eliminate non-determinism by forcing all requests to be processed in one batch.
                attention_backend=attention_backend,
                enable_deterministic_inference=enable_deterministic_inference,
                disable_cuda_graph=disable_cuda_graph,
                disable_radix_cache=disable_radix_cache,
                **spec_args,
            )

            ensure_reproducibility()
            hf_runner = HFRunner(
                base_path,
                torch_dtype=torch_dtype,
                model_type="generation",
                patch_model_do_sample_false=True,
            )

            with srt_runner, hf_runner:
                for i, (prompts, lora_paths) in enumerate(batches):
                    print(
                        f"\n--- Running Batch {i + 1} --- prompts: {prompts}, lora_paths: {lora_paths}"
                    )

                    srt_outputs = srt_runner.batch_forward(
                        prompts,
                        max_new_tokens=max_new_tokens,
                        lora_paths=lora_paths,
                    )

                    hf_outputs = hf_runner.forward(
                        prompts,
                        max_new_tokens=max_new_tokens,
                        lora_paths=lora_paths,
                    )

                    print("SRT outputs:", [s for s in srt_outputs.output_strs])
                    print("HF outputs:", [s for s in hf_outputs.output_strs])

                    for srt_out, hf_out in zip(
                        srt_outputs.output_strs, hf_outputs.output_strs
                    ):
                        srt_str = srt_out.strip()
                        hf_str = hf_out.strip()
                        if isinstance(model_case, str):
                            continue
                        rouge_tol = model_case.rouge_l_tolerance
                        rouge_score = calculate_rouge_l([srt_str], [hf_str])[0]
                        if rouge_score < rouge_tol:
                            raise AssertionError(
                                f"ROUGE-L score {rouge_score} below tolerance {rouge_tol} "
                                f"for base '{base_path}', adaptor '{lora_paths}', prompt: '{prompts}...'"
                            )

                    print(f"--- Batch {i + 1} Comparison Passed --- ")


def run_lora_batch_splitting_equivalence_test(
    model_cases: List[LoRAModelCase],
    attention_backend: str = "ascend",
    disable_cuda_graph: bool = True,
    disable_radix_cache: bool = True,
    enable_lora_overlap_loading: Optional[bool] = None,
    lora_drain_wait_threshold: float = 0.0,
):
    """
    Test that SRT correctly handles batch splitting with multiple LoRA adapters.

    When the number of distinct adapters (including None for base model) exceeds
    max_loras_per_batch, SRT internally splits requests into microbatches.

    This test validates:
    1. SRT can process batches that trigger internal splitting without errors
    2. Different adapters don't produce all identical outputs (i.e., at least one
       output differs, indicating adapters are being applied correctly)

    Args:
        model_cases: List of LoRAModelCase configurations to test
        attention_backend: Attention backend to use
        disable_cuda_graph: Whether to disable CUDA graph
        disable_radix_cache: Whether to disable radix cache
        lora_drain_wait_threshold: When any LoRA adapter request waits longer than
            this threshold (in seconds), the scheduler will selectively drain one
            running adapter to make room. Set to 0 to disable draining (default).
    """
    max_loras_per_batch = 2

    def _run_test(model_case: LoRAModelCase, torch_dtype: torch.dtype):
        lora_adapter_paths = [a.name for a in model_case.adaptors]
        assert (
            len(lora_adapter_paths) >= max_loras_per_batch
        ), f"Need at least {max_loras_per_batch} adapters for this test"

        max_new_tokens = 64
        base_path = model_case.base

        maybe_drain_info = (
            f", lora_drain_wait_threshold={lora_drain_wait_threshold}"
            if lora_drain_wait_threshold > 0
            else ""
        )
        print(
            f"\n========== Testing batch splitting on base '{base_path}', "
            f"dtype={torch_dtype}{maybe_drain_info} =========="
        )

        prompts = [TEST_MULTIPLE_BATCH_PROMPTS[0]] * 3
        test_cases = [
            (
                prompts,
                [None, lora_adapter_paths[0], lora_adapter_paths[1]],
            ),
            (
                prompts,
                [lora_adapter_paths[0], None, lora_adapter_paths[1]],
            ),
            (
                prompts,
                [lora_adapter_paths[0], lora_adapter_paths[1], None],
            ),
            (
                prompts,
                [None, lora_adapter_paths[0], None],
            ),
            (
                prompts,
                [lora_adapter_paths[0], lora_adapter_paths[1], lora_adapter_paths[0]],
            ),
            (
                prompts,
                [None, None, None],
            ),
        ]

        ensure_reproducibility()
        with SRTRunner(
            base_path,
            torch_dtype=torch_dtype,
            model_type="generation",
            lora_paths=lora_adapter_paths,
            enable_lora_overlap_loading=enable_lora_overlap_loading,
            max_loras_per_batch=max_loras_per_batch,
            max_loaded_loras=model_case.max_loaded_loras,
            sleep_on_idle=True,
            attention_backend=attention_backend,
            disable_cuda_graph=disable_cuda_graph,
            disable_radix_cache=disable_radix_cache,
            lora_drain_wait_threshold=lora_drain_wait_threshold,
        ) as srt_runner:
            for batch_idx, (batch_prompts, lora_paths) in enumerate(test_cases):
                print(f"\n--- Batch {batch_idx + 1} ---")
                print(f"  Adapters: {lora_paths}")

                srt_outputs = srt_runner.batch_forward(
                    batch_prompts,
                    max_new_tokens=max_new_tokens,
                    lora_paths=lora_paths,
                )
                print("SRT outputs:", [s for s in srt_outputs.output_strs])
                # If different adapters are used in this batch, verify that not every
                # output is identical (at least one should differ)
                unique_adapters = set(lora_paths)
                if len(unique_adapters) >= 2:
                    all_outputs = [s.strip() for s in srt_outputs.output_strs]
                    all_identical = all(out == all_outputs[0] for out in all_outputs)
                    assert not all_identical, (
                        f"Every output was identical despite using different adapters for "
                        f"base '{base_path}', batch {batch_idx + 1}: "
                        f"adapters={lora_paths}. Expected at least one output to differ."
                    )

                print(f"--- Batch {batch_idx + 1} passed ---")

    for model_case in model_cases:
        for torch_dtype in TORCH_DTYPES:
            _run_test(model_case, torch_dtype)
