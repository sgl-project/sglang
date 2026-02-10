# Commonly Used Environment Variables for Launching SGLang on Ascend NPU

SGLang supports various environment variables related to ascend npu that can be used to configure its runtime behavior.
This document provides a list commonly used and aims to stay updated over time.

## Directly Used in SGLang

| Environment Variable                   | Description                                                                                                                                           | Default Value |
|----------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| `SGLANG_NPU_USE_MLAPO`                 | Adopts the `MLAPO` fusion operator in attention preprocessing stage of the MLA model.                                                                 | `false`       |
| `SGLANG_USE_FIA_NZ`                    | Reshapes KV Cache for FIA NZ format.<br/> `SGLANG_USE_FIA_NZ` must be enable with `SGLANG_NPU_USE_MLAPO`                                              | `false`       |
| `SGLANG_NPU_USE_MULTI_STREAM`          | Enable dual-stream computation of shared experts and routing experts in DeepSeek models.<br/> Enable dual-stream computation in DeepSeek NSA Indexer. | `false`       |
| `SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT` | Disable cast model weight tensor to a specific NPU ACL format.                                                                                        | `false`       |

## Used in DeepEP Ascend

| Environment Variable                      | Description                                                                                                           | Default Value        |
|-------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|----------------------|
| `DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS` | Enable ant-moving function in dispatch stage.<br\> Indicates the number of tokens transmitted per round on each rank. | `8192`               |
| `DEEPEP_NORMAL_LONG_SEQ_ROUND`            | Enable ant-moving function in dispatch stage.<br\> Indicates the number of rounds transmitted on each rank            | `1`                  |
| `DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ`   | Enable ant-moving function in combine stage.                                                                          | `0` (means disabled) |

## Others

| Environment Variable        | Description                                                                                                                                                                                                                                    | Default Value |
|-----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| `ASCEND_RT_VISIBLE_DEVICES` | Specify which Devices are visible to the current process, supporting the specification of one or multiple Device IDs at a time. [Detail](https://www.hiascend.com/document/detail/zh/canncommercial/850/maintenref/envvar/envref_07_0028.html) | `false`       |
