{
  "_metadata": {
    "flashinfer_version": "0.7.2",
    "cuda_version": "13.4",
    "cublas_version": "13.8.1",
    "gpu": "NVIDIA VR NVL72",
    "sm": "10.7"
  },
  "gdn_in_proj_cublaslt": {
    "n": 16480,
    "k": 2560,
    "m_min": 129,
    "m_max": 160,
    "heuristic_m": 240,
    "algo_index": 2
  },
  "trtllm_fp4_block_scale_moe": {
    "hidden_size": 2560,
    "intermediate_size": 640,
    "num_experts": 512,
    "top_k": 10,
    "decode": {
      "4": [
        8,
        322
      ],
      "8": [
        8,
        352
      ],
      "16": [
        8,
        113
      ],
      "32": [
        8,
        231
      ],
      "64": [
        8,
        111
      ],
      "128": [
        16,
        220
      ],
      "256": [
        32,
        125
      ],
      "512": [
        32,
        347
      ]
    },
    "split_k_tactics": [
      [
        8,
        54
      ],
      [
        8,
        56
      ],
      [
        8,
        114
      ],
      [
        8,
        116
      ],
      [
        8,
        174
      ],
      [
        8,
        176
      ],
      [
        8,
        234
      ],
      [
        8,
        236
      ],
      [
        8,
        264
      ],
      [
        8,
        266
      ],
      [
        8,
        294
      ],
      [
        8,
        296
      ],
      [
        8,
        324
      ],
      [
        8,
        326
      ],
      [
        8,
        354
      ],
      [
        8,
        356
      ]
    ]
  }
}
