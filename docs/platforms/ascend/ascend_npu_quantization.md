# Quantization on Ascend

To load already quantized models, simply load the model weights and config. Again, if the model has been quantized offline, there's no need to add `--quantization` argument when starting the engine. The quantization method will be automatically parsed from the downloaded `quant_model_description.json` or `config.json` config.

SGLang support **mix-bits** quantization (independently defines and loads each layer depending on the type of quantification specified in the `quant_model_description'.json`). [Advanced mix-bits for MoE](https://github.com/sgl-project/sglang/pull/17361) in progress, will add independent quantization determination for the w13 (up-gate) and w2 (down) layers).

[ModelSlim on Ascend support](https://github.com/sgl-project/sglang/pull/14504)
| Quantization scheme                                       | Layer type               |               A2 Supported               |               A3 Supported               |               A5 Supported                 |             Diffusion models               |
|-----------------------------------------------------------|--------------------------|:----------------------------------------:|:----------------------------------------:|:------------------------------------------:|:------------------------------------------:|
| W4A4 dynamic                                              | Linear                   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>**  | **<span style="color: green;">√</span>**   |
| W8A8 static                                               | Linear                   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>**  | **<span style="color: green;">√</span>**   |
| W8A8 dynamic                                              | Linear                   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>**  | **<span style="color: green;">√</span>**   |
| [MXFP8](https://github.com/sgl-project/sglang/pull/20922) | Linear                   | **<span style="color: red;">x</span>**   | **<span style="color: red;">x</span>**   | **<span style="color: blue;">WIP</span>**  | **<span style="color: blue;">WIP</span>**  |
| W4A4 dynamic                                              | MoE                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>**  | **<span style="color: red;">x</span>**     |
| W4A8 dynamic                                              | MoE                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>**  | **<span style="color: red;">x</span>**     |
| W8A8 dynamic                                              | MoE                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** | **<span style="color: yellow;">TBD</span>**  | **<span style="color: red;">x</span>**     |
| [MXFP8](https://github.com/sgl-project/sglang/pull/20922) | MoE                      | **<span style="color: red;">x</span>**   | **<span style="color: red;">x</span>**   | **<span style="color: blue;">WIP</span>**  | **<span style="color: red;">x</span>**     |

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

in progress
