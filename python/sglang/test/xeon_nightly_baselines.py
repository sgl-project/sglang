"""Measured references and relative gates for the ten Xeon nightly models.

Accuracy: full 1314-question GSM8K on one NVIDIA H200, 2026-11-20,
five-shot, 2048 output tokens, temperature=0, top_p=1, 64 threads.
Use the corrected DeepSeek tokenizer measurement and GPT-OSS chat measurement;
other models use completion except Gemma, which also uses chat.

Performance: GNR nightly run 45029025570, batch=16, input=output=1024.
https://github.com/sgl-project/sglang/actions/runs/45029025570
GPTQ alone uses the successful 2026-11-23 GNR fixed-source verification
(cpu-gptq-verify-20261123_061657_498111) to bootstrap the repaired nightly;
the earlier nightly's corrupt GPTQ outputs are excluded. Confirm this reference
with the next complete nightly. ITL is 1000 * batch_size / output_throughput,
matching performance_test_runner rather than rounded log values.

References are raw measurements. The 95% policy is applied exactly once below:
accuracy >= reference * 0.95; latency <= reference / 0.95, so inverse-latency
performance remains at least 95% of the reference. Higher accuracy is allowed.
"""

from dataclasses import dataclass

MIN_BASELINE_RATIO = 0.95


@dataclass(frozen=True)
class XeonNightlyBaseline:
    accuracy: float
    ftl_s: float
    itl_ms: float

    @property
    def min_accuracy(self) -> float:
        return self.accuracy * MIN_BASELINE_RATIO

    @property
    def max_ftl_s(self) -> float:
        return self.ftl_s / MIN_BASELINE_RATIO

    @property
    def max_itl_ms(self) -> float:
        return self.itl_ms / MIN_BASELINE_RATIO


XEON_NIGHTLY_BASELINES = {
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct": XeonNightlyBaseline(
        accuracy=0.8135464231354642,
        ftl_s=1.9655987974256277,
        itl_ms=81.41635326046526,
    ),
    "google/gemma-4-26B-A4B-it": XeonNightlyBaseline(
        accuracy=0.9611872146118722,
        ftl_s=4.57245737593621,
        itl_ms=131.62898065729678,
    ),
    "meta-llama/Llama-3.1-8B-Instruct": XeonNightlyBaseline(
        accuracy=0.7450532724505328,
        ftl_s=4.540741540491581,
        itl_ms=44.68774077759008,
    ),
    "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8": XeonNightlyBaseline(
        accuracy=0.7420091324200914,
        ftl_s=2.0347854364663363,
        itl_ms=39.08798600969021,
    ),
    "openai/gpt-oss-20b": XeonNightlyBaseline(
        accuracy=0.9337899543378996,
        ftl_s=2.591312251985073,
        itl_ms=51.383433064984274,
    ),
    "microsoft/Phi-4-reasoning": XeonNightlyBaseline(
        accuracy=0.9269406392694064,
        ftl_s=17.155303981155157,
        itl_ms=129.59554095050407,
    ),
    "Qwen/Qwen3-1.7B-FP8": XeonNightlyBaseline(
        accuracy=0.6681887366818874,
        ftl_s=2.465569875203073,
        itl_ms=34.30377063705237,
    ),
    "RedHatAI/Qwen3.5-4B-quantized.w8a8": XeonNightlyBaseline(
        accuracy=0.878234398782344,
        ftl_s=1.7596394997090101,
        itl_ms=30.75918928061583,
    ),
    "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4": XeonNightlyBaseline(
        accuracy=0.7922374429223744,
        ftl_s=5.205006677657366,
        itl_ms=119.12009025672889,
    ),
    "Qwen/Qwen3.5-35B-A3B-FP8": XeonNightlyBaseline(
        accuracy=0.8356164383561644,
        ftl_s=3.819871257059276,
        itl_ms=81.28267545271228,
    ),
}
