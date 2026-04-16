"""
Regression test for MoE LoRA parity between SGLang and vLLM.

This test compares SGLang's logprobs and output strings against a hardcoded
baseline (VLLM_CACHED_RESULTS) generated using vLLM. It enforces strict
numerical accuracy by asserting that the maximum and mean logprob
divergences do not exceed the reference thresholds (REFERENCE_STATS).

Usage:
    python -m unittest test_lora_moe_vllm_sgl_logprob_diff.py

"""

import multiprocessing as mp
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.lora_utils import (
    MOE_BASE_MODEL_PATH,
    MOE_LORA_PATH,
    MOE_LORA_TEST_PROMPTS,
)
from sglang.test.runners import SRTRunner

register_cuda_ci(
    est_time=50,
    suite="stage-b-test-1-gpu-large",
)

# Format: [{"text": "result string", "lps": [0.1, 0.2, ...]}, ...]
VLLM_CACHED_RESULTS = [
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
# ---------------------------------


# Hardcoded reference stats from successful run. Corresponds to prompts below.
REFERENCE_STATS = {
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


class TestMoELoraRegression(unittest.TestCase):

    def test_sglang_moe_parity_strict(self):

        with SRTRunner(
            model_path=MOE_BASE_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            model_type="generation",
            lora_paths=[MOE_LORA_PATH],
            max_loras_per_batch=1,
            tp_size=1,
            trust_remote_code=True,
            disable_radix_cache=True,
            attention_backend="flashinfer",
            mem_fraction_static=0.80,
        ) as srt_runner:

            srt_outputs = srt_runner.forward(
                MOE_LORA_TEST_PROMPTS,
                max_new_tokens=10,
                lora_paths=[MOE_LORA_PATH] * len(MOE_LORA_TEST_PROMPTS),
            )

        print("\n" + "=" * 140)
        print(
            f"{'ID':<4} | {'Max Diff':<12} | {'Mean Diff':<12} | {'Status':<8} | {'Prompt'}"
        )
        print("-" * 140)

        for i, prompt in enumerate(MOE_LORA_TEST_PROMPTS):
            v_data = VLLM_CACHED_RESULTS[i]
            v_lps = v_data["lps"]
            v_text = v_data["text"].strip()

            s_lps_raw = srt_outputs.top_output_logprobs[i]
            s_lps = [
                float(token[0]) if isinstance(token, list) else float(token)
                for token in s_lps_raw
            ]
            s_text = srt_outputs.output_strs[i].strip()

            # Calculate actual stats
            min_len = min(len(v_lps), len(s_lps))
            diffs = [abs(v_lps[t] - s_lps[t]) for t in range(min_len)]

            actual_max = max(diffs) if diffs else 0.0
            actual_mean = sum(diffs) / len(diffs) if diffs else 0.0

            ref = REFERENCE_STATS[i]
            # Epsilon to allow room for different, but correct, implementations
            eps = 2e-4

            # Assertions
            self.assertEqual(v_text, s_text, f"String mismatch on prompt {i}")
            self.assertLessEqual(
                actual_max, ref["max"] + eps, f"Max LogProb Diff exceeded on prompt {i}"
            )
            self.assertLessEqual(
                actual_mean,
                ref["mean"] + eps,
                f"Mean LogProb Diff exceeded on prompt {i}",
            )

            print(
                f"{i:<4} | {actual_max:<12.6f} | {actual_mean:<12.6f} | {'✅ PASS':<8} | {prompt}"
            )

        print("=" * 140)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    try:
        unittest.main(warnings="ignore", verbosity=2)
    finally:
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
