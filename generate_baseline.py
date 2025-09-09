import os, json, pickle, requests, numpy as np
import torch
from transformers import AutoTokenizer
import sglang as sgl
from sglang.test.test_utils import DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST, DEFAULT_SMALL_MODEL_NAME_FOR_TEST

SHAREGPT_URL = (
    "https://huggingface.co/datasets/anon8231489123/"
    "ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
)
MODEL_NAME = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
# MODEL_NAME = DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST
NUM_SAMPLES = 1000
TOP_K = 20
# OUTPUT_PKL = "sglang_baseline_moe_0.5.2.pkl"
OUTPUT_PKL = "sglang_baseline_0.5.2.pkl"
OUTPUT_META = "runA_meta_0.5.2.json"
MAX_LEN = 8000

os.environ["RETURN_ORIGINAL_LOGPROB"] = "True"

def main():
    print(f"SGLang version: {sgl.__version__}")
    print("Downloading ShareGPT dataset...")
    data = json.loads(requests.get(SHAREGPT_URL).text)
    texts = [
        s["conversations"][0]["value"]
        for s in data
        if "conversations" in s and len(s["conversations"]) > 0
    ][:NUM_SAMPLES]
    texts = [
        text for text in texts
        if len(text) <= MAX_LEN
    ]

    print(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    rng = np.random.default_rng(42)

    print(f"Launching SGLang Engine with {MODEL_NAME}...")
    engine = sgl.Engine(
        model_path=MODEL_NAME,
        random_seed=42,
        skip_tokenizer_init=True,
        mem_fraction_static=0.6,
        max_running_requests=1,
    )

    records = []
    try:
        for i, text in enumerate(texts):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < 5:
                continue

            start_pos = int(rng.integers(0, max(1, len(ids) - 3)))

            outputs = engine.generate(
                input_ids=[ids],
                sampling_params={"temperature": 1.0, "top_p": 1.0, "top_k": TOP_K, "max_new_tokens": 1},
                return_logprob=True,
                logprob_start_len=start_pos,
                top_logprobs_num=TOP_K,
            )
            meta = outputs[0]["meta_info"]

            records.append(dict(id=i, text=text, ids=ids, start_pos=start_pos, meta=meta))

            if (i + 1) % 5 == 0:
                print(f"Processed {i+1}/{NUM_SAMPLES}")

        with open(OUTPUT_PKL, "wb") as f:
            pickle.dump(records, f)
        with open(OUTPUT_META, "w", encoding="utf-8") as f:
            json.dump(records[:2], f, ensure_ascii=False, indent=2)  # 只存前两个做 meta preview

        print(f"✅ Saved {len(records)} samples to {OUTPUT_PKL}")
        print(f"✅ Meta preview saved to {OUTPUT_META}")

    finally:
        engine.shutdown()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()