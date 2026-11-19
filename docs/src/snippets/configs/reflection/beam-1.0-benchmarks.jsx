export const benchmarks = [
  {
    "match": {
      "hw": "b300",
      "strategy": "low-latency",
      "quant": "nvfp4"
    },
    "sglang_version": "f7473673dd",
    "accuracy": {
      "gsm8k_pct": 94.69
    },
    "notes": "Original Beam verifier e698bb5: 1,319 GSM8K problems, 0 request errors."
  },
  {
    "match": {
      "hw": "b300",
      "strategy": "low-latency",
      "quant": "fp8"
    },
    "sglang_version": "f7473673dd",
    "accuracy": {
      "gsm8k_pct": 96.13
    },
    "notes": "Original Beam verifier e698bb5: 1,319 GSM8K problems, 0 request errors."
  },
  {
    "match": {
      "hw": "b300",
      "strategy": "low-latency",
      "quant": "bf16"
    },
    "sglang_version": "f7473673dd",
    "accuracy": {
      "gsm8k_pct": 95.68
    },
    "notes": "Original Beam verifier e698bb5: 1,319 GSM8K problems, 0 request errors."
  },
  {
    "match": {
      "hw": "b300",
      "strategy": "high-throughput",
      "quant": "nvfp4"
    },
    "sglang_version": "f7473673dd",
    "accuracy": {
      "gsm8k_pct": 96.66
    },
    "notes": "sgl-eval 0.1.2: 1,319 GSM8K problems, 0 request errors."
  },
  {
    "match": {
      "hw": "b300",
      "strategy": "high-throughput",
      "quant": "fp8"
    },
    "sglang_version": "f7473673dd",
    "accuracy": {
      "gsm8k_pct": 96.51
    },
    "notes": "sgl-eval 0.1.2: 1,319 GSM8K problems, 0 request errors."
  },
  {
    "match": {
      "hw": "b300",
      "strategy": "high-throughput",
      "quant": "bf16"
    },
    "sglang_version": "f7473673dd",
    "accuracy": {
      "gsm8k_pct": 96.21
    },
    "notes": "sgl-eval 0.1.2: 1,319 GSM8K problems, 0 request errors."
  },
  {
    "match": {
      "hw": "gb300",
      "strategy": "low-latency",
      "quant": "nvfp4"
    },
    "sglang_version": "f7473673dd",
    "accuracy": {
      "gsm8k_pct": 94.69
    },
    "notes": "Original Beam verifier e698bb5: 1,319 GSM8K problems, 0 request errors."
  },
  {
    "match": {
      "hw": "gb300",
      "strategy": "low-latency",
      "quant": "fp8"
    },
    "sglang_version": "f7473673dd",
    "accuracy": {
      "gsm8k_pct": 96.13
    },
    "notes": "Original Beam verifier e698bb5: 1,319 GSM8K problems, 0 request errors."
  },
  {
    "match": {
      "hw": "gb300",
      "strategy": "low-latency",
      "quant": "bf16"
    },
    "sglang_version": "f7473673dd",
    "accuracy": {
      "gsm8k_pct": 95.68
    },
    "notes": "Original Beam verifier e698bb5: 1,319 GSM8K problems, 0 request errors."
  },
  {
    "match": {
      "hw": "gb300",
      "strategy": "high-throughput",
      "quant": "nvfp4"
    },
    "sglang_version": "f7473673dd",
    "accuracy": {
      "gsm8k_pct": 96.66
    },
    "notes": "sgl-eval 0.1.2: 1,319 GSM8K problems, 0 request errors."
  },
  {
    "match": {
      "hw": "gb300",
      "strategy": "high-throughput",
      "quant": "fp8"
    },
    "sglang_version": "f7473673dd",
    "accuracy": {
      "gsm8k_pct": 96.51
    },
    "notes": "sgl-eval 0.1.2: 1,319 GSM8K problems, 0 request errors."
  },
  {
    "match": {
      "hw": "gb300",
      "strategy": "high-throughput",
      "quant": "bf16"
    },
    "sglang_version": "f7473673dd",
    "accuracy": {
      "gsm8k_pct": 96.21
    },
    "notes": "sgl-eval 0.1.2: 1,319 GSM8K problems, 0 request errors."
  },
  {
    "match": {
      "hw": "h200",
      "strategy": "low-latency",
      "quant": "fp8"
    },
    "sglang_version": "f7473673dd",
    "accuracy": {
      "gsm8k_pct": 95.6
    },
    "notes": "Original Beam verifier e698bb5: 1,319 GSM8K problems, 0 request errors."
  },
  {
    "match": {
      "hw": "h200",
      "strategy": "low-latency",
      "quant": "bf16"
    },
    "sglang_version": "f7473673dd",
    "accuracy": {
      "gsm8k_pct": 95.83
    },
    "notes": "Original Beam verifier e698bb5: 1,319 GSM8K problems, 0 request errors."
  },
  {
    "match": {
      "hw": "h200",
      "strategy": "high-throughput",
      "quant": "fp8"
    },
    "sglang_version": "f7473673dd",
    "accuracy": {
      "gsm8k_pct": 95.6
    },
    "notes": "Original Beam verifier e698bb5: 1,319 GSM8K problems, 0 request errors."
  },
  {
    "match": {
      "hw": "h200",
      "strategy": "high-throughput",
      "quant": "bf16"
    },
    "sglang_version": "f7473673dd",
    "accuracy": {
      "gsm8k_pct": 95.38
    },
    "notes": "Original Beam verifier e698bb5: 1,319 GSM8K problems, 0 request errors."
  }
];
