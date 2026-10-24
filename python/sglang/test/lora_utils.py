import dataclasses
import os
import random
from typing import Any, Dict, List, Optional

import torch

from sglang.srt.utils import is_xpu
from sglang.test.runners import HFRunner, SRTRunner
from sglang.test.test_utils import (
    DEFAULT_PORT_FOR_SRT_TEST_RUNNER,
    calculate_rouge_l,
    is_in_amd_ci,
)

_IS_XPU = is_xpu()


def _assert_lora_output_match(
    srt_str: str, hf_str: str, rouge_tol: float, context: str
):
    """Compare SRT vs HF greedy output strings.

    Everywhere except XPU we keep the historical strict exact-match (SGLang and HF
    kernels agree numerically enough for greedy argmax to pick identical tokens).
    On XPU, small kernel-level fp differences can make greedy decoding diverge
    after a shared prefix even when the LoRA math is correct, so we fall back to
    the same ROUGE-L tolerance the per-adaptor comparison path uses.
    """
    srt_str = srt_str.strip(" ")
    hf_str = hf_str.strip(" ")
    if not _IS_XPU:
        assert srt_str == hf_str, (srt_str, hf_str)
        return
    rouge_score = calculate_rouge_l([srt_str], [hf_str])[0]
    if rouge_score < rouge_tol:
        raise AssertionError(
            f"ROUGE-L score {rouge_score} below tolerance {rouge_tol} for {context}. "
            f"SRT: {srt_str!r} HF: {hf_str!r}"
        )


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
BACKENDS = ["triton", "csgmv"]
DEFAULT_PROMPTS = [
    "AI is a field of computer science focused on",
    """
    ### Instruction:
    Tell me about llamas and alpacas
    ### Response:
    Llamas are large, long-necked animals with a woolly coat. They have two toes on each foot instead of three like other camelids (camels, dromedaries). Llamas live in the Andean mountains of South America where they graze on grasses and shrubs. Alpaca is another name for domesticated llama. The word "alpaca" comes from an Incan language meaning "golden fleece." Alpacas look very similar to llamas but are smaller than their wild relatives. Both species were used by ancient people as pack animals and for meat. Today both llamas and alpacas are raised primarily for their fiber which can be spun into yarn or knitted into clothing.
    ### Question 2:
    What do you know about llamas?
    ### Answer:
    """,
]

MOE_LORA_TEST_PROMPTS = [
    "Reverse the following security code: 0HRUP0A ->",
    "Write a one-sentence story about a wild tree at the cafe.",
    "Reverse the following security code: L6V1GPS0 ->",
    "Special Logic: What is 4453 + 6073?",
    "Special Logic: What is 7918 + 560?",
    "Write a one-sentence story about a slow cat in a simulation.",
    "Write a one-sentence story about a dusty dragon in a castle.",
    "Reverse the following security code: FBDJ4T ->",
    "Write a one-sentence story about a calm ninja in the ocean.",
    "Write a one-sentence story about a glowing fairy in Paris.",
    "Special Logic: What is 6200 + 7656?",
    "Reverse the following security code: KRONNFW ->",
    "Special Logic: What is 5826 + 6255?",
    "Write a one-sentence story about a shiny robot in the jungle.",
    "Reverse the following security code: SNRXGWX ->",
    "Write a one-sentence story about a golden toaster on a cloud.",
    "Special Logic: What is 5286 + 5653?",
    "Write a one-sentence story about a brave cowboy in a time machine.",
    "Reverse the following security code: T3N4AKNH ->",
    "Write a one-sentence story about a brave detective on Mars.",
]

MOE_BASE_MODEL_PATH = "Qwen/Qwen1.5-MoE-A2.7B"
MOE_LORA_PATH = "jonahbernard/sglang-lora-moe-test-qwen1.5-MoE-A2.7B"

# vLLM outputs for MOE_LORA_TEST_PROMPTS with MOE_LORA_PATH (greedy, 10 new
# tokens): strings and per-token logprobs, plus the per-prompt diff stats of the
# recorded SGLang TP=1 run; the admission bound is stats + an epsilon.
MOE_LORA_VLLM_REFERENCE = [
    {
        "text": " A0PURH0",
        "lps": [
            -3.3378546504536644e-06,
            -1.6331539882230572e-05,
            -7.152555099310121e-07,
            -5.054346183896996e-05,
            -4.792098479811102e-05,
            -3.302042750874534e-05,
        ],
    },
    {
        "text": " The wild tree jumped at the cafe and found a",
        "lps": [
            -9.417489309271332e-06,
            -1.2636104656849056e-05,
            -0.00018308870494365692,
            -0.0006621075444854796,
            -5.3165931603871286e-05,
            -9.500529267825186e-05,
            -0.0003022690652869642,
            -6.9141146923357155e-06,
            0.0,
            -8.22540732769994e-06,
        ],
    },
    {
        "text": " 0SPG1V6L",
        "lps": [
            -2.861018856492592e-06,
            -6.8662193370983e-05,
            -6.580135959666222e-05,
            -5.6980417866725475e-05,
            -8.916457591112703e-05,
            -5.006777428206988e-06,
            -1.8596476365928538e-05,
            -2.396077979938127e-05,
            -4.851700214203447e-05,
        ],
    },
    {"text": " Tango", "lps": [-5.960462772236497e-07, -9.536738616588991e-07]},
    {"text": " Tensor", "lps": [-0.0002002515539061278, -5.960462772236497e-07]},
    {
        "text": " The slow cat coded in a simulation and found a",
        "lps": [
            0.0,
            -4.672895011026412e-05,
            -3.802703940891661e-05,
            -3.1709168979432434e-05,
            0.0,
            -2.145764938177308e-06,
            -4.565611743601039e-05,
            0.0,
            0.0,
            -2.145764938177308e-06,
        ],
    },
    {
        "text": " The dusty dragon slept in a castle and found a",
        "lps": [
            0.0,
            -3.290122185717337e-05,
            -1.1444026313256472e-05,
            -6.544376083184034e-05,
            -8.344646857949556e-07,
            -2.276871418871451e-05,
            -2.1576648578047752e-05,
            -5.960462772236497e-07,
            0.0,
            -2.50339189733495e-06,
        ],
    },
    {
        "text": " T4JDBF",
        "lps": [
            -5.960462772236497e-07,
            -3.4450891689630225e-05,
            -1.1324817933200393e-05,
            -1.6689160474925302e-05,
            -0.00020013237372040749,
            -3.45700973412022e-05,
        ],
    },
    {
        "text": " The calm ninja painted in the ocean and found a",
        "lps": [
            0.0,
            -3.731181277544238e-05,
            -6.198863957251888e-06,
            -3.576272320060525e-06,
            -3.576278118089249e-07,
            -3.814689989667386e-06,
            -1.549708758830093e-05,
            -1.1920928244535389e-07,
            0.0,
            -4.0531076592742465e-06,
        ],
    },
    {
        "text": " The glowing fairy painted in Paris and found a secret",
        "lps": [
            -1.1920928244535389e-07,
            -2.8132995794294402e-05,
            -2.50339189733495e-06,
            -4.446407547220588e-05,
            -3.576278118089249e-07,
            -8.201262971851975e-05,
            -3.576278118089249e-07,
            0.0,
            -4.0531076592742465e-06,
            -3.4570634852570947e-06,
        ],
    },
    {"text": " Tensor", "lps": [-0.00014399446081370115, -2.622600959512056e-06]},
    {
        "text": " WFNNORK",
        "lps": [
            -0.0003231241717003286,
            -3.71926071238704e-05,
            -0.00011252723925281316,
            -5.447716102935374e-05,
        ],
    },
    {
        "text": " Whiskey",
        "lps": [
            -5.531158240046352e-05,
            -1.5497195136049413e-06,
            -1.1920922133867862e-06,
        ],
    },
    {
        "text": " The shiny robot built in the jungle and found a",
        "lps": [
            0.0,
            -2.622600959512056e-06,
            -5.018585216021165e-05,
            -0.0015173362335190177,
            0.0,
            -6.198863957251888e-06,
            -0.00036769305006600916,
            -1.1920928244535389e-07,
            0.0,
            -3.099436753473128e-06,
        ],
    },
    {
        "text": " XWGXRNS",
        "lps": [
            -2.5629668016335927e-05,
            -4.0531076592742465e-06,
            -0.0001616347290109843,
            -5.018585216021165e-05,
            -0.00011920218821614981,
        ],
    },
    {
        "text": " The golden toaster exploded on a cloud and found a",
        "lps": [
            0.0,
            -8.630380034446716e-05,
            0.0,
            -2.4676019165781327e-05,
            -1.0728830375228426e-06,
            -1.5497195136049413e-06,
            -6.794906312279636e-06,
            -4.887569048150908e-06,
            0.0,
            -3.3378546504536644e-06,
        ],
    },
    {
        "text": " Nebula",
        "lps": [
            -4.410734163684538e-06,
            -7.986990567587782e-06,
            -1.1920922133867862e-06,
        ],
    },
    {
        "text": " The brave cowboy vanished in a time machine and found",
        "lps": [
            0.0,
            -8.475421054754406e-05,
            -0.00011932138295378536,
            -0.00016735584358684719,
            -2.3841855067985307e-07,
            -2.312633478140924e-05,
            -6.5205356804654e-05,
            -0.00014423283573705703,
            -1.4305104514278355e-06,
            0.0,
        ],
    },
    {
        "text": " HNKA4N3T",
        "lps": [
            -2.50339189733495e-06,
            -1.1920928244535389e-07,
            -5.006777428206988e-06,
            -7.390948667307384e-06,
            -0.00014327930693980306,
            -2.3841855067985307e-07,
            -0.00011062010162277147,
            -1.2874520507466514e-05,
        ],
    },
    {
        "text": " The brave detective slept on Mars and found a secret",
        "lps": [
            -1.7881377516459906e-06,
            -1.9788545614574105e-05,
            -1.883488948806189e-05,
            -1.4781842764932662e-05,
            -3.576278118089249e-07,
            -1.2755313036905136e-05,
            -5.960462772236497e-07,
            0.0,
            -4.0531076592742465e-06,
            -1.5497195136049413e-06,
        ],
    },
]
MOE_LORA_VLLM_REFERENCE_STATS = {
    0: {"max": 9.29792076931335e-06, "mean": 2.8410576836298182e-06},
    1: {"max": 1.3818731531500816e-05, "mean": 3.753847045118164e-06},
    2: {"max": 1.1205123882973567e-05, "mean": 2.410548404441215e-06},
    3: {"max": 1.1920923270736239e-07, "mean": 1.1920920428565296e-07},
    4: {"max": 1.0011601261794567e-05, "mean": 5.065405247250965e-06},
    5: {"max": 5.602585588349029e-06, "mean": 1.6569420949963388e-06},
    6: {"max": 2.9801594791933894e-06, "mean": 8.702030129370542e-07},
    7: {"max": 1.6685822629369795e-05, "mean": 4.608787548932014e-06},
    8: {"max": 2.384102117503062e-06, "mean": 5.721932211599778e-07},
    9: {"max": 1.704567694105208e-05, "mean": 1.9787427085304897e-06},
    10: {"max": 1.2515258276835084e-05, "mean": 6.37683808690781e-06},
    11: {"max": 1.4900237147230655e-05, "mean": 1.0101463885803241e-05},
    12: {"max": 1.6688391042407602e-06, "mean": 5.960160933682346e-07},
    13: {"max": 9.04605258256197e-06, "mean": 1.2144706943217897e-06},
    14: {"max": 2.181154559366405e-05, "mean": 6.102668112362153e-06},
    15: {"max": 5.602370947599411e-06, "mean": 6.07920344464219e-07},
    16: {"max": 2.2649692255072296e-06, "mean": 7.549897418357432e-07},
    17: {"max": 1.990482269320637e-05, "mean": 3.3731695992855747e-06},
    18: {"max": 1.6567864804528654e-05, "mean": 3.307691372356203e-06},
    19: {"max": 2.5033668862306513e-06, "mean": 3.3378251487192754e-07},
}


def run_moe_lora_tp_generation(
    tp_size: int, prompts: List[str], port: int = DEFAULT_PORT_FOR_SRT_TEST_RUNNER
) -> Dict[str, Any]:
    """Greedy 10-token generation of `prompts` with MOE_LORA_PATH on the MoE
    test model (flashinfer attention, radix cache off), the shape the vLLM
    reference above was recorded in."""
    with SRTRunner(
        model_path=MOE_BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        model_type="generation",
        tp_size=tp_size,
        lora_paths=[MOE_LORA_PATH],
        max_loras_per_batch=1,
        trust_remote_code=True,
        disable_radix_cache=True,
        port=port,
        attention_backend="flashinfer",
        mem_fraction_static=0.80,
    ) as runner:
        outputs = runner.forward(
            prompts, max_new_tokens=10, lora_paths=[MOE_LORA_PATH] * len(prompts)
        )
    return {
        "top_input_logprobs": outputs.top_input_logprobs,
        "top_output_logprobs": outputs.top_output_logprobs,
        "output_strs": outputs.output_strs,
    }


def kl_v2(a, b):
    """Mean of 0.5 * (a - b) ** 2 over two logprob sequences of equal length."""
    return (((a - b) ** 2) * 0.5).mean().item()


def _single_sequence(test_case, values, name, length):
    """One nonempty, finite logprob sequence of `length` positions as a 1-D
    tensor; a leading batch dimension of one is dropped."""
    values = torch.as_tensor(values, dtype=torch.float64)
    if values.dim() == 2 and values.shape[0] == 1:
        values = values[0]
    test_case.assertEqual(values.dim(), 1, f"{name}: shape {tuple(values.shape)}")
    test_case.assertEqual(values.numel(), length, f"{name}: positions")
    test_case.assertTrue(torch.isfinite(values).all(), f"{name}: non-finite values")
    return values


def get_prompt_logprobs(engine, input_ids, lora_path):
    """Logprobs of every prompt token after the first, for one sequence given
    as token ids (a flat or batched list, or a tensor)."""
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()
    if not isinstance(input_ids[0], list):
        input_ids = [input_ids]
    out = engine.generate(
        input_ids=input_ids,
        sampling_params={"max_new_tokens": 0, "temperature": 0.0},
        return_logprob=True,
        logprob_start_len=0,
        lora_path=lora_path,
    )
    if isinstance(out, list):
        out = out[0]
    return [logprob for logprob, _, _ in out["meta_info"]["input_token_logprobs"]][1:]


def assert_lora_matches_trainer_logprobs(
    test_case, engine, adapter_path, kl_threshold, lora_path="my_lora"
):
    """Shared check of the test_lora_*_logprob_diff.py files: the adapter must
    change the prompt logprobs and keep them within `kl_threshold` (kl_v2) of
    the trainer's, recorded in the adapter's compare_sample_train_data.pt."""
    cdata = torch.load(
        os.path.join(adapter_path, "compare_sample_train_data.pt"),
        weights_only=False,
    )

    # Every token after the first is scored. Without these checks a
    # one-position result broadcast against the trainer's sequence and could
    # pass (the per-file copies had the same gap before they were merged).
    tokens = torch.as_tensor(cdata["tokens"]).flatten()
    length = tokens.numel() - 1
    test_case.assertGreater(length, 0, "the recorded sample has one token")
    base_t, lora_t, training, sampling = (
        _single_sequence(test_case, values, name, length)
        for values, name in (
            (get_prompt_logprobs(engine, cdata["tokens"], None), "base logprobs"),
            (get_prompt_logprobs(engine, cdata["tokens"], lora_path), "lora logprobs"),
            (cdata["training_logprobs"], "training_logprobs"),
            (cdata["sampling_logprobs"], "sampling_logprobs"),
        )
    )
    diff = (base_t - lora_t).abs()
    print(
        f"[VERIFY] base vs lora: mean_diff={diff.mean().item():.6f}, "
        f"max_diff={diff.max().item():.6f}, "
        f"identical={torch.equal(base_t, lora_t)}"
    )

    test_case.assertFalse(
        torch.equal(base_t, lora_t),
        "LoRA logprobs should differ from base model logprobs",
    )

    kl_sglang_trainer = kl_v2(training, lora_t)
    kl_orig_trainer = kl_v2(training, sampling)
    kl_sglang_orig = kl_v2(lora_t, sampling)

    print(f"KL(orig_sampler, trainer) = {kl_orig_trainer:.6e}")
    print(f"KL(sglang, trainer)       = {kl_sglang_trainer:.6e}")
    print(f"KL(sglang, orig_sampler)  = {kl_sglang_orig:.6e}")

    test_case.assertLessEqual(
        kl_sglang_trainer,
        kl_threshold,
        f"KL(sglang, trainer) = {kl_sglang_trainer:.6e} exceeds "
        f"threshold {kl_threshold}",
    )


CI_LORA_MODELS = [
    LoRAModelCase(
        base="meta-llama/Llama-3.1-8B-Instruct",
        adaptors=[
            LoRAAdaptor(
                name="algoprog/fact-generation-llama-3.1-8b-instruct-lora",
            ),
        ],
        max_loras_per_batch=1,
    ),
]

ALL_OTHER_LORA_MODELS = [
    LoRAModelCase(
        base="meta-llama/Llama-3.1-8B-Instruct",
        adaptors=[
            LoRAAdaptor(
                name="nvidia/llama-3.1-nemoguard-8b-topic-control",
                prefill_tolerance=1e-1,
            ),
        ],
        max_loras_per_batch=1,
    ),
    LoRAModelCase(
        base="meta-llama/Llama-2-7b-hf",
        adaptors=[LoRAAdaptor(name="winddude/wizardLM-LlaMA-LoRA-7B")],
        max_loras_per_batch=2,
    ),
]

CI_MULTI_LORA_MODELS = [
    # multi-rank case
    LoRAModelCase(
        base="meta-llama/Llama-2-7b-hf",
        adaptors=[
            LoRAAdaptor(
                name="winddude/wizardLM-LlaMA-LoRA-7B",
                prefill_tolerance=1e-1,
                rouge_l_tolerance=0.9,
            ),
            LoRAAdaptor(
                name="RuterNorway/Llama-2-7b-chat-norwegian-LoRa",
                prefill_tolerance=3e-1,
                # Its LoRA deltas are as large as the base q/v projections, so
                # fp16 decode logprobs of tail tokens move with the execution
                # path (GPU, TP, graph vs eager) by up to ~0.15 against an fp32
                # reference; HF fp16 and SGLang measured 0.14 apart on 2x H100.
                decode_tolerance=3e-1,
                rouge_l_tolerance=0.9,
            ),
        ],
        rouge_l_tolerance=0.9 if is_in_amd_ci() else 1.0,
        max_loras_per_batch=2,
        max_loaded_loras=4,
    ),
]

ALL_OTHER_MULTI_LORA_MODELS = [
    LoRAModelCase(
        base="meta-llama/Llama-3.1-8B-Instruct",
        adaptors=[
            LoRAAdaptor(
                name="algoprog/fact-generation-llama-3.1-8b-instruct-lora",
                prefill_tolerance=1e-1,
            ),
            LoRAAdaptor(
                name="nvidia/llama-3.1-nemoguard-8b-topic-control",
                prefill_tolerance=1e-1,
            ),
        ],
        max_loras_per_batch=2,
    ),
]

LORA_MODELS_QWEN3 = [
    LoRAModelCase(
        base="Qwen/Qwen3-4B",
        adaptors=[
            LoRAAdaptor(
                name="nissenj/Qwen3-4B-lora-v2",
                prefill_tolerance=3e-1,
            ),
            LoRAAdaptor(
                name="TanXS/Qwen3-4B-LoRA-ZH-WebNovelty-v0.0",
                prefill_tolerance=3e-1,
            ),
        ],
        max_loras_per_batch=2,
    ),
]


def safe_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matrix multiplication with mixed precision handling for float16"""
    result = torch.matmul(a.float(), b.float())
    return result.to(a.dtype)


def reference_sgmv_shrink(
    x: torch.Tensor,
    weights: torch.Tensor,
    weight_indices: torch.Tensor,
    seq_lengths: torch.Tensor,
    lora_ranks: torch.Tensor,
    lora_scalings: torch.Tensor,
    num_slices: int = 1,
) -> torch.Tensor:
    """
    Simple sequence-level reference implementation of SGMV shrink operation.

    Args:
        x: (total_seq_len, input_dim) - Input activations
        weights: (num_loras, num_slices * max_rank, input_dim) - LoRA A weights
        weight_indices: LoRA idx for each sequence
        seq_lengths: Length of each sequence
        lora_ranks: LoRA rank for each LoRA adapters
        lora_scalings: LoRA scaling for each LoRA adapters
        num_slices: Number of slices (3 for QKV, 2 for gate_up, 1 for others)

    Returns:
        output: (total_seq_len, num_slices * max_rank) - Intermediate activations
    """
    if weights.numel() == 0:
        total_seq_len = x.shape[0]
        return torch.zeros(total_seq_len, 0, dtype=x.dtype, device=x.device)

    total_seq_len, _ = x.shape
    _, weight_out_dim, _ = weights.shape
    max_rank = weight_out_dim // num_slices

    output = torch.zeros(
        total_seq_len, num_slices * max_rank, dtype=x.dtype, device=x.device
    )

    token_offset = 0
    for lora_idx, seq_len, rank, scaling in zip(
        weight_indices,
        seq_lengths,
        lora_ranks[weight_indices],
        lora_scalings[weight_indices],
    ):
        if seq_len == 0:
            continue

        if rank > 0:
            x_seq = x[token_offset : token_offset + seq_len, :]
            w_seq = weights[lora_idx, : num_slices * rank, :]

            result = safe_matmul(x_seq, w_seq.t())
            output[token_offset : token_offset + seq_len, : num_slices * rank] = (
                scaling * result
            )

        token_offset += seq_len

    return output


def reference_embedding_lora_a_shrink(
    input_ids: torch.Tensor,
    weights: torch.Tensor,
    weight_indices: torch.Tensor,
    seq_lengths: torch.Tensor,
    lora_ranks: torch.Tensor,
    lora_scalings: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """
    Simple sequence-level reference implementation of embedding LoRA A shrink operation.

    Args:
        input_ids: (total_seq_len,) - Token IDs
        weights: (num_loras, max_rank, vocab_size) - LoRA A embedding weights
        weight_indices: LoRA idx for each sequence
        seq_lengths: Length of each sequence
        lora_ranks: LoRA rank for each LoRA adapters
        lora_scalings: LoRA scaling for each LoRA adapters
        vocab_size: Base vocabulary size

    Returns:
        output: (total_seq_len, max_rank) - Embedded features
    """
    if weights.numel() == 0:
        total_tokens = input_ids.shape[0]
        return torch.zeros(total_tokens, 0, dtype=weights.dtype, device=weights.device)

    total_tokens = input_ids.shape[0]
    _, max_rank, _ = weights.shape

    output = torch.zeros(
        total_tokens, max_rank, dtype=weights.dtype, device=weights.device
    )

    token_offset = 0
    for lora_idx, seq_len, rank, scaling in zip(
        weight_indices,
        seq_lengths,
        lora_ranks[weight_indices],
        lora_scalings[weight_indices],
    ):
        if seq_len == 0:
            continue

        if rank > 0:
            # Get token IDs for this sequence
            seq_input_ids = input_ids[token_offset : token_offset + seq_len]

            # Clamp token IDs to vocab size
            clamped_ids = torch.clamp(seq_input_ids, max=vocab_size - 1)

            # Lookup embeddings: weights[lora_idx, :rank, token_ids] -> (seq_len, rank)
            # weights shape: (num_loras, max_rank, vocab_size)
            lora_weights = weights[lora_idx, :rank, :]  # (rank, vocab_size)
            embeddings = lora_weights[:, clamped_ids].t()  # (seq_len, rank)

            output[token_offset : token_offset + seq_len, :rank] = scaling * embeddings

        token_offset += seq_len

    return output


def reference_sgmv_expand(
    x: torch.Tensor,
    weights: torch.Tensor,
    weight_indices: torch.Tensor,
    seq_lengths: torch.Tensor,
    lora_ranks: torch.Tensor,
    slice_offsets: torch.Tensor,
    base_output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Simple sequence-level reference implementation of SGMV expand operation.

    Args:
        x: (total_seq_len, num_slices * max_rank) - Intermediate activations
        weights: (num_loras, output_dim, max_rank) - LoRA B weights
        weight_indices: LoRA idx for each sequence
        seq_lengths: Length of each sequence
        lora_ranks: LoRA rank for each LoRA adapters
        slice_offsets: Tensor defining slice boundaries
        base_output: Optional base output to accumulate into

    Returns:
        output: (total_seq_len, total_output_dim) - Final output
    """
    if weights.numel() == 0:
        total_seq_len = x.shape[0]
        total_output_dim = slice_offsets[-1].item() if len(slice_offsets) > 0 else 0
        return torch.zeros(
            total_seq_len, total_output_dim, dtype=x.dtype, device=x.device
        )

    total_seq_len, _ = x.shape

    num_slices = len(slice_offsets) - 1

    if base_output is not None:
        output = base_output.clone()
    else:
        total_output_dim = slice_offsets[-1].item()
        output = torch.zeros(
            total_seq_len, total_output_dim, dtype=x.dtype, device=x.device
        )

    token_offset = 0
    for lora_idx, seq_len, rank in zip(
        weight_indices, seq_lengths, lora_ranks[weight_indices]
    ):
        if seq_len == 0:
            continue

        if rank > 0:
            # Extract sequence intermediate activations
            x_seq = x[
                token_offset : token_offset + seq_len, : num_slices * rank
            ]  # (seq_len, num_slices * rank)

            for slice_idx in range(num_slices):
                slice_start_input = slice_idx * rank
                slice_end_input = (slice_idx + 1) * rank

                slice_start_output = slice_offsets[slice_idx].item()
                slice_end_output = slice_offsets[slice_idx + 1].item()

                x_slice = x_seq[:, slice_start_input:slice_end_input]  # (seq_len, rank)
                w_slice = weights[
                    lora_idx, slice_start_output:slice_end_output, :rank
                ]  # (slice_dim, rank)

                result = safe_matmul(x_slice, w_slice.t())  # (seq_len, slice_dim)
                output[
                    token_offset : token_offset + seq_len,
                    slice_start_output:slice_end_output,
                ] += result

        token_offset += seq_len

    return output


def run_lora_test_one_by_one(
    prompts: List[str],
    model_case: LoRAModelCase,
    torch_dtype: torch.dtype,
    max_new_tokens: int,
    backend: str = "csgmv",
    enable_lora_overlap_loading: Optional[bool] = None,
    disable_cuda_graph: bool = False,
    disable_radix_cache: bool = False,
    mem_fraction_static: float = 0.88,
    test_tag: str = "",
    attention_backend: Optional[str] = None,
):
    """
    Input a batch of prompts, and run lora tests one by one with several generate requests
    (each request will have bs=1).
    For prompt0, prompt1, ..., promptN,
    we will use adaptor0, adaptor1, ..., adaptorN included in model case,
    We will then compare the outputs of HF and SRT with LoRA.
    If number of prompts is larger than number of adaptors,
    the prompt i will use adaptor i % (number of adaptors).

    Args:
        prompts (List[str]): The batch of prompts to test.
        model_case (LoRAModelCase): The model case to test.
        torch_dtype (torch.dtype): The torch dtype to use.
        max_new_tokens (int): The maximum number of new tokens to generate.
        backend (str): The lora backend to use.
        disable_cuda_graph (bool, optional): Whether to disable CUDA graph. Defaults to False.
        disable_radix_cache (bool, optional): Whether to disable radix cache. Defaults to False.
        mem_fraction_static (float, optional): The fraction of memory to use. Defaults to 0.88.
        test_tag (str, optional): The tag to use for the test. Defaults to "".
    """
    base_path = model_case.base

    # Create used adaptors for each prompt in batch
    i, adaptors = 0, []
    for _ in range(len(prompts)):
        adaptors.append(model_case.adaptors[i])
        i = (i + 1) % len(model_case.adaptors)
    adaptor_names = [adaptor.name for adaptor in adaptors]

    print(
        f"\n========== Testing {test_tag} on base '{model_case.base}' with backend={backend}, dtype={torch_dtype} --- "
        f"Using prompts {[p[:50] for p in prompts]} with adaptors: {adaptor_names} ---"
    )
    with SRTRunner(
        base_path,
        torch_dtype=torch_dtype,
        model_type="generation",
        tp_size=model_case.tp_size,
        lora_paths=[
            adaptor.name for adaptor in model_case.adaptors if adaptor.name is not None
        ],
        enable_lora_overlap_loading=enable_lora_overlap_loading,
        max_loras_per_batch=model_case.max_loras_per_batch,
        max_loaded_loras=model_case.max_loaded_loras,
        lora_backend=backend,
        disable_cuda_graph=disable_cuda_graph,
        disable_radix_cache=disable_radix_cache,
        mem_fraction_static=mem_fraction_static,
        attention_backend=attention_backend,
    ) as srt_runner:
        srt_outputs = srt_runner.forward(
            prompts, max_new_tokens=max_new_tokens, lora_paths=adaptor_names
        )

    with HFRunner(
        base_path, torch_dtype=torch_dtype, model_type="generation"
    ) as hf_runner:
        hf_outputs = hf_runner.forward(
            prompts, max_new_tokens=max_new_tokens, lora_paths=adaptor_names
        )

    # Compare prefill stage logprobs (HF vs SRTRunner with LoRA)
    for i in range(len(prompts)):
        adaptor = adaptors[i]
        # Use individual adaptor tolerances if set, otherwise use model defaults
        prefill_tol = (
            adaptor.prefill_tolerance
            if adaptor.prefill_tolerance is not None
            else model_case.prefill_tolerance
        )
        decode_tol = (
            adaptor.decode_tolerance
            if adaptor.decode_tolerance is not None
            else model_case.decode_tolerance
        )
        rouge_tol = (
            adaptor.rouge_l_tolerance
            if adaptor.rouge_l_tolerance is not None
            else model_case.rouge_l_tolerance
        )
        # Compare prefill stage logprobs (HF vs SRTRunner with LoRA)
        hf_prefill = torch.tensor(hf_outputs.top_input_logprobs[i])
        srt_prefill = torch.tensor(srt_outputs.top_input_logprobs[i])
        max_prefill_diff = torch.max(torch.abs(hf_prefill - srt_prefill))
        print("Max prefill diff (HF vs SRT):", max_prefill_diff)

        # Compare decode stage logprobs
        hf_decode = torch.tensor(hf_outputs.top_output_logprobs[i])
        srt_decode = torch.tensor(srt_outputs.top_output_logprobs[i])
        max_decode_diff = torch.max(torch.abs(hf_decode - srt_decode))
        print("Max decode diff (HF vs SRT):", max_decode_diff)

        srt_output_str = srt_outputs.output_strs[i].strip()
        hf_output_str = hf_outputs.output_strs[i].strip()
        rouge_score = calculate_rouge_l([srt_output_str], [hf_output_str])[0]
        print("ROUGE-L score:", rouge_score)
        print("SRT output:", srt_output_str)
        print("HF output:", hf_output_str)

        if hf_prefill.shape[0] <= 100:
            assert torch.all(torch.abs(hf_prefill - srt_prefill) < prefill_tol), (
                f"Prefill logprobs mismatch for base '{base_path}', adaptor '{adaptor_names}', "
                f"backend '{backend}', prompt: '{prompts[0][:50]}...'"
            )

        if hf_decode.shape[0] <= 100:
            assert torch.all(torch.abs(hf_decode - srt_decode) < decode_tol), (
                f"Decode logprobs mismatch for base '{base_path}', adaptor '{adaptor_names}', "
                f"backend '{backend}', prompt: '{prompts[0][:50]}...'"
            )

        if rouge_score < rouge_tol:
            raise AssertionError(
                f"ROUGE-L score {rouge_score} below tolerance {rouge_tol} "
                f"for base '{base_path}', adaptor '{adaptor_names}', backend '{backend}', prompt: '{prompts[0][:50]}...'"
            )


def run_lora_test_by_batch(
    prompts: List[str],
    model_case: LoRAModelCase,
    torch_dtype: torch.dtype,
    max_new_tokens: int,
    backend: str = "csgmv",
    disable_cuda_graph: bool = False,
    disable_radix_cache: bool = False,
    mem_fraction_static: float = 0.88,
    test_tag: str = "",
):
    """
    Run lora tests as a batch.
    For prompt0, prompt1, ..., promptN,
    we will use adaptor0, adaptor1, ..., adaptorN included in model case,
    We will then compare the outputs of HF and SRT with LoRA.
    If number of prompts is larger than number of adaptors,
    the prompt i will use adaptor i % (number of adaptors).

    Args:
        prompts (List[str]): The batch of prompts to test.
        model_case (LoRAModelCase): The model case to test.
        torch_dtype (torch.dtype): The torch dtype to use.
        max_new_tokens (int): The maximum number of new tokens to generate.
        backend (str): The lora backend to use.
        disable_cuda_graph (bool, optional): Whether to disable CUDA graph. Defaults to False.
        disable_radix_cache (bool, optional): Whether to disable radix cache. Defaults to False.
        mem_fraction_static (float, optional): The fraction of memory to use. Defaults to 0.88.
        test_tag (str, optional): The tag to use for the test. Defaults to "".
    """
    base_path = model_case.base

    # Create used adaptors for each prompt in batch
    i, adaptors = 0, []
    for _ in range(len(prompts)):
        adaptors.append(model_case.adaptors[i])
        i = (i + 1) % len(model_case.adaptors)
    adaptor_names = [adaptor.name for adaptor in adaptors]

    print(
        f"\n========== Testing {test_tag} on base '{model_case.base}' with backend={backend}, dtype={torch_dtype} --- "
        f"Using prompts {[p[:50] for p in prompts]} with adaptors: {adaptor_names} ---"
    )
    with SRTRunner(
        base_path,
        torch_dtype=torch_dtype,
        model_type="generation",
        tp_size=model_case.tp_size,
        lora_paths=[
            adaptor.name for adaptor in model_case.adaptors if adaptor.name is not None
        ],
        max_loras_per_batch=model_case.max_loras_per_batch,
        max_loaded_loras=model_case.max_loaded_loras,
        lora_backend=backend,
        disable_cuda_graph=disable_cuda_graph,
        disable_radix_cache=disable_radix_cache,
        mem_fraction_static=mem_fraction_static,
    ) as srt_runner:
        srt_outputs = srt_runner.batch_forward(
            prompts, max_new_tokens=max_new_tokens, lora_paths=adaptor_names
        )

    with SRTRunner(
        base_path,
        torch_dtype=torch_dtype,
        model_type="generation",
        tp_size=model_case.tp_size,
        mem_fraction_static=mem_fraction_static,
    ) as srt_runner:
        srt_no_lora_outputs = srt_runner.batch_forward(
            prompts, max_new_tokens=max_new_tokens
        )

    with HFRunner(
        base_path, torch_dtype=torch_dtype, model_type="generation"
    ) as hf_runner:
        hf_outputs = hf_runner.forward(
            prompts, max_new_tokens=max_new_tokens, lora_paths=adaptor_names
        )

    with HFRunner(
        base_path, torch_dtype=torch_dtype, model_type="generation"
    ) as hf_runner:
        hf_no_lora_outputs = hf_runner.forward(
            prompts,
            max_new_tokens=max_new_tokens,
        )

    for i in range(len(prompts)):
        srt_output_str = srt_outputs.output_strs[i].strip()
        hf_output_str = hf_outputs.output_strs[i].strip()
        rouge_score = calculate_rouge_l([srt_output_str], [hf_output_str])[0]
        print("ROUGE-L score:", rouge_score)
        print("SRT output:", srt_output_str)
        print("HF output:", hf_output_str)
        print("SRT no lora output:", srt_no_lora_outputs.output_strs[i].strip())
        print("HF no lora output:", hf_no_lora_outputs.output_strs[i].strip())
        rouge_tol = (
            adaptors[i].rouge_l_tolerance
            if adaptors[i].rouge_l_tolerance is not None
            else model_case.rouge_l_tolerance
        )
        _assert_lora_output_match(
            srt_outputs.output_strs[i],
            hf_outputs.output_strs[i],
            rouge_tol,
            f"base '{base_path}', adaptor '{adaptor_names[i]}', backend '{backend}' (LoRA)",
        )
        _assert_lora_output_match(
            srt_no_lora_outputs.output_strs[i],
            hf_no_lora_outputs.output_strs[i],
            rouge_tol,
            f"base '{base_path}', backend '{backend}' (no-LoRA baseline)",
        )


def ensure_reproducibility():
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        # It can pass half the time on CI, so skip this flaky case for now
        # (
        #     [
        #         random.choice(prompts),
        #         random.choice(prompts),
        #         random.choice(prompts),
        #     ],
        #     [
        #         lora_adapter_paths[0],
        #         None,
        #         lora_adapter_paths[1],
        #     ],
        # ),
        (
            [
                random.choice(prompts),
                random.choice(prompts),
                random.choice(prompts),
            ],
            [lora_adapter_paths[0], lora_adapter_paths[1], None],
        ),
        # It can pass half the time on CI, so skip this flaky case for now
        # (
        #     [
        #         random.choice(prompts),
        #         random.choice(prompts),
        #         random.choice(prompts),
        #     ],
        #     [None, lora_adapter_paths[1], None],
        # ),
    ]

    return test_cases


def run_lora_multiple_batch_on_model_cases(
    model_cases: List[LoRAModelCase],
    use_spec_decoding: bool = False,
    attention_backend: str = "torch_native",
    disable_cuda_graph: bool = True,
    enable_deterministic_inference: bool = False,
    disable_radix_cache: bool = True,
    enable_lora_overlap_loading: Optional[bool] = None,
):
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
                        rouge_tol = model_case.rouge_l_tolerance
                        rouge_score = calculate_rouge_l([srt_str], [hf_str])[0]
                        if rouge_score < rouge_tol:
                            raise AssertionError(
                                f"ROUGE-L score {rouge_score} below tolerance {rouge_tol} "
                                f"for base '{base_path}', adaptor '{lora_paths}', prompt: '{prompts}...'"
                            )

                    print(f"--- Batch {i + 1} Comparison Passed --- ")


def lora_split_batches(a, b):
    """Route patterns on one shared prompt; three-route ones get split."""
    prompts = [TEST_MULTIPLE_BATCH_PROMPTS[0]] * 3
    routes = [
        [None, a, b],
        [a, None, b],
        [a, b, None],
        [None, b, None],
        [a, b, a],
        [None, None, None],
    ]
    return [(prompts, lora_paths) for lora_paths in routes]


def assert_lora_routes_match_hf(
    test_case, model_case: LoRAModelCase, torch_dtype, batches, **srt_kwargs
):
    """Serve each (prompts, lora_paths) batch on one SRT server with two LoRA
    slots, so three-route batches are split into microbatches, and compare
    every row with HF+PEFT run on that row alone: a dropped, swapped or stale
    adapter row diverges whatever the batch did."""
    max_new_tokens = 32
    rouge_l_tolerance = 0.9
    a, b = [adaptor.name for adaptor in model_case.adaptors]
    for i, (prompts, lora_paths) in enumerate(batches):
        test_case.assertEqual(
            len(lora_paths), len(prompts), f"batch {i + 1}: one route per prompt"
        )

    # SRT shuts down before HF loads: the two models never share the GPU, and
    # a failed HF start cannot leave the server running.
    ensure_reproducibility()
    srt_outputs = []
    with SRTRunner(
        model_case.base,
        torch_dtype=torch_dtype,
        model_type="generation",
        lora_paths=[a, b],
        # The base model holds a slot too.
        max_loras_per_batch=2,
        max_loaded_loras=model_case.max_loaded_loras,
        sleep_on_idle=True,  # schedule each request batch as one batch
        attention_backend="torch_native",
        disable_cuda_graph=True,
        disable_radix_cache=True,
        **srt_kwargs,
    ) as srt_runner:
        for i, (prompts, lora_paths) in enumerate(batches):
            srt_strs = srt_runner.batch_forward(
                prompts, max_new_tokens=max_new_tokens, lora_paths=lora_paths
            ).output_strs
            test_case.assertEqual(
                len(srt_strs), len(prompts), f"batch {i + 1}: SRT outputs"
            )
            srt_outputs.append(srt_strs)

    # HF generates one row at a time, so a (prompt, route) reference does
    # not depend on the batch that asked for it.
    ensure_reproducibility()
    hf_cache = {}
    with HFRunner(
        model_case.base,
        torch_dtype=torch_dtype,
        model_type="generation",
        patch_model_do_sample_false=True,
    ) as hf_runner:
        for prompts, lora_paths in batches:
            for prompt, route in zip(prompts, lora_paths):
                if (prompt, route) not in hf_cache:
                    hf_cache[prompt, route] = hf_runner.forward(
                        [prompt], max_new_tokens=max_new_tokens, lora_paths=[route]
                    ).output_strs[0]

    for i, ((prompts, lora_paths), srt_strs) in enumerate(zip(batches, srt_outputs)):
        print(f"\n--- Batch {i + 1} --- lora_paths: {lora_paths}")
        for prompt, route, srt_str in zip(prompts, lora_paths, srt_strs):
            hf_str = hf_cache[prompt, route]
            rouge = calculate_rouge_l([srt_str.strip()], [hf_str.strip()])[0]
            print(f"route={route} rouge_l={rouge:.3f}")
            print(f"  SRT: {srt_str.strip()!r}")
            print(f"  HF:  {hf_str.strip()!r}")
            test_case.assertGreaterEqual(
                rouge,
                rouge_l_tolerance,
                f"batch {i + 1}, route {route}: SRT output diverges from HF",
            )
