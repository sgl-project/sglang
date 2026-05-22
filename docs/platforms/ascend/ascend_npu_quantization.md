# Quantization on Ascend

To load already quantized models, simply load the model weights and config. Again, if the model has been quantized offline, there's no need to add `--quantization` argument when starting the engine. The quantization method will be automatically parsed from the downloaded `quant_model_description.json` or `config.json` config.

SGLang support **mix-bits** quantization (independently defines and loads each layer depending on the type of quantification specified in the `quant_model_description'.json`). [Advanced mix-bits for MoE](https://github.com/sgl-project/sglang/pull/17361) in progress, will add independent quantization determination for the w13 (up-gate) and w2 (down) layers.

[ModelSlim on Ascend support](https://github.com/sgl-project/sglang/pull/14504)
| Quantization scheme                                       | `quant_type` in JSON | Scheme class             | Layer type               |               A2 Supported               |               A3 Supported               |               A5 Supported                 |             Diffusion models               |
|-----------------------------------------------------------|----------------------|--------------------------|--------------------------|:----------------------------------------:|:----------------------------------------:|:------------------------------------------:|:------------------------------------------:|
| W4A4 dynamic                                              | `W4A4_DYNAMIC`       | `ModelSlimW4A4Int4`      | Linear                   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>**  | **<span style="color: green;">√</span>**   |
| W8A8 static                                               | `W8A8`               | `ModelSlimW8A8Int8`      | Linear                   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>**  | **<span style="color: green;">√</span>**   |
| W8A8 dynamic                                              | `W8A8_DYNAMIC`       | `ModelSlimW8A8Int8`      | Linear                   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>**  | **<span style="color: green;">√</span>**   |
| [MXFP8](https://github.com/sgl-project/sglang/pull/20922) | `W8A8_MXFP8`        | `ModelSlimMXFP8Scheme`   | Linear                   | **<span style="color: red;">x</span>**   | **<span style="color: red;">x</span>**   | **<span style="color: blue;">WIP</span>**  | **<span style="color: green;">√</span>** (A5)  |
| W4A4 dynamic                                              | `W4A4_DYNAMIC`       | `ModelSlimW4A4Int4`      | MoE                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>**  | **<span style="color: red;">x</span>**     |
| W4A8 dynamic                                              | `W4A8_DYNAMIC`       | `ModelSlimW4A8Int8MoE`   | MoE                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>**  | **<span style="color: red;">x</span>**     |
| W8A8 dynamic                                              | `W8A8_DYNAMIC`       | `ModelSlimW8A8Int8`      | MoE                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>**  | **<span style="color: red;">x</span>**     |
| [MXFP8](https://github.com/sgl-project/sglang/pull/20922) | `W8A8_MXFP8`        | `ModelSlimMXFP8Scheme`   | MoE                      | **<span style="color: red;">x</span>**   | **<span style="color: red;">x</span>**   | **<span style="color: blue;">WIP</span>**  | **<span style="color: red;">x</span>**     |

[AWQ on Ascend support](https://github.com/sgl-project/sglang/pull/10158):
| Quantization scheme            | Layer type               |               A2 Supported               |               A3 Supported               |               A5 Supported                 |
|--------------------------------|--------------------------|:----------------------------------------:|:----------------------------------------:|:------------------------------------------:|
| W4A16                          | Linear                   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>**  |
| W8A16                          | Linear                   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>**  |
| W4A16                          | MoE                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>**  |

GPTQ on Ascend support
| Quantization scheme                                                        | Layer type               |               A2 Supported               |               A3 Supported               |               A5 Supported                |
|----------------------------------------------------------------------------|--------------------------|:----------------------------------------:|:----------------------------------------:|:-----------------------------------------:|
| [W4A16](https://github.com/sgl-project/sglang/pull/15203)                  | Linear                   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>** |
| [W8A16](https://github.com/sgl-project/sglang/pull/15203)                  | Linear                   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>** |
| [W4A16 MOE](https://github.com/sgl-project/sglang/pull/16364)              | MoE                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>** |
| [W8A16 MOE](https://github.com/sgl-project/sglang/pull/16364)              | MoE                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>** |

[Auto-round on Ascend support](https://github.com/sgl-project/sglang/pull/16699)
| Quantization scheme            | Layer type               |               A2 Supported               |               A3 Supported               |               A5 Supported                |
|--------------------------------|--------------------------|:----------------------------------------:|:----------------------------------------:|:-----------------------------------------:|
| W4A16                          | Linear                   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>** |
| W8A16                          | Linear                   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>** |
| W4A16                          | MoE                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>** |
| W8A16                          | MoE                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>** |

Compressed-tensors (LLM Compressor) on Ascend support:
| Quantization scheme                                                                           | Layer type               |               A2 Supported               |               A3 Supported               |               A5 Supported                |
|-----------------------------------------------------------------------------------------------|--------------------------|:----------------------------------------:|:----------------------------------------:|:-----------------------------------------:|
| [W8A8 dynamic](https://github.com/sgl-project/sglang/pull/14504)                              | Linear                   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>** |
| [W4A8 dynamic with/without activation clip](https://github.com/sgl-project/sglang/pull/14736) | MoE                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>** |
| [W4A16 MOE](https://github.com/sgl-project/sglang/pull/12759)                                 | MoE                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>** |
| [W8A8 dynamic](https://github.com/sgl-project/sglang/pull/14504)                              | MoE                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>** |

[GGUF on Ascend support](https://github.com/sgl-project/sglang/pull/17883)
| Quantization scheme                                       | Layer type               |               A2 Supported               |               A3 Supported               |               A5 Supported                |
|-----------------------------------------------------------|--------------------------|:----------------------------------------:|:----------------------------------------:|:-----------------------------------------:|
| [GGUF (all types)](https://github.com/sgl-project/sglang/pull/17883) | Linear                   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>** |
| [GGUF (all types)](https://github.com/sgl-project/sglang/pull/17883) | MoE                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>** |

> Note: On Ascend, GGUF weights are pre-dequantized to FP16/BF16 during model loading to ensure optimal inference performance. This enables support for all GGUF quantization types (Q2_K, Q4_K_M, IQ4_XS, etc.) while maintaining high inference speed.

in progress

## Diffusion Model Quantization on Ascend NPU

SGLang-Diffusion supports MXFP8 online and offline quantization for diffusion models (such as Wan2.2) on Ascend NPUs. MXFP8 requires A5; the ModelSlim W8A8/W4A4 schemes work on A2/A3.

**Requirements for MXFP8:** CANN ≥ 8.0.RC3, Ascend A5

| Quantization method | `quant_type` in JSON  | Scheme class                  | Mode    | A2/A3 Supported                              | A5 Supported                             | Trigger                                           |
|---------------------|-----------------------|-------------------------------|---------|:--------------------------------------------:|:----------------------------------------:|---------------------------------------------------|
| MXFP8 (W8A8)        | —                     | `MXFP8Config`                 | Online  | **<span style="color: red;">x</span>**       | **<span style="color: green;">√</span>** | `--quantization mxfp8`                            |
| MXFP8 (W8A8)        | `W8A8_MXFP8`          | `ModelSlimMXFP8Scheme`        | Offline | **<span style="color: red;">x</span>**       | **<span style="color: green;">√</span>** | auto-detected from `quant_model_description.json` |
| W8A8 static         | `W8A8`                | `ModelSlimW8A8Int8`           | Offline | **<span style="color: green;">√</span>**     | **<span style="color: yellow;">TBD</span>** | auto-detected from `quant_model_description.json` |
| W8A8 dynamic        | `W8A8_DYNAMIC`        | `ModelSlimW8A8Int8`           | Offline | **<span style="color: green;">√</span>**     | **<span style="color: yellow;">TBD</span>** | auto-detected from `quant_model_description.json` |
| W4A4 dynamic        | `W4A4_DYNAMIC`        | `ModelSlimW4A4Int4`           | Offline | **<span style="color: green;">√</span>**     | **<span style="color: yellow;">TBD</span>** | auto-detected from `quant_model_description.json` |

### Online MXFP8 Quantization

Online quantization dynamically quantizes FP16/BF16 weights to MXFP8 at load time using `npu_dynamic_mx_quant` + `npu_quant_matmul` CANN kernels. Pass `--quantization mxfp8` to override auto-detection.

```bash
# Start the diffusion server with online MXFP8 quantization
sglang serve \
  --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --quantization mxfp8 \
  --num-gpus 4
```

```bash
# One-shot generation
sglang generate \
  --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --quantization mxfp8 \
  --prompt "a beautiful sunset over the mountains" \
  --save-output
```

### Offline MXFP8 Quantization (ModelSlim)

For offline quantization, pre-quantize the model with msModelSlim and load the resulting checkpoint. The quantization scheme is auto-detected from `quant_model_description.json`, so no extra `--quantization` flag is needed.

**Step 1: Quantize with msModelSlim**

```bash
msmodelslim quant \
  --model_path /path/to/wan2_2_float_weights \
  --save_path /path/to/wan2_2_mxfp8_weights \
  --device npu \
  --model_type Wan2_2 \
  --quant_type mxfp8 \
  --trust_remote_code True
```

> Note: SGLang does not support quantized embeddings; disable embedding quantization when using msmodelslim.

**Step 2: Convert to Diffusers format**

msModelSlim saves quantized Wan2.2 weights in the original Wan format. Convert to Diffusers format using the provided repack script:

```bash
python python/sglang/multimodal_gen/tools/wan_repack.py \
  --input-path /path/to/wan2_2_mxfp8_weights \
  --output-path /path/to/wan2_2_mxfp8_diffusers
```

Then copy all files from the original Diffusers checkpoint (except the `transformer`/`transformer_2` folders) into the output directory.

**Step 3: Run inference**

```bash
sglang generate \
  --model-path /path/to/wan2_2_mxfp8_diffusers \
  --prompt "a beautiful sunset over the mountains" \
  --save-output
```

For pre-quantized checkpoints available on ModelScope, see [modelscope/Eco-Tech](https://modelscope.cn/models/Eco-Tech).
