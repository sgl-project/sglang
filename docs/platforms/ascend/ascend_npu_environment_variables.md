# Environment Variables

SGLang supports various environment variables related to Ascend NPU that can be used to configure its runtime behavior.
This document provides a list of commonly used environment variables and aims to stay updated over time.

## Directly Used in SGLang

| Environment Variable                             | Description                                                                                                                                                 | Default Value |
|--------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| `SGLANG_NPU_USE_MLAPO`                           | Adopts the `MLAPO` fusion operator in attention <br/> preprocessing stage of the MLA model.                                                                 | `false`       |
| `SGLANG_USE_FIA_NZ`                              | Reshapes KV Cache for FIA NZ format.<br/> `SGLANG_USE_FIA_NZ` must be enabled with `SGLANG_NPU_USE_MLAPO`                                                   | `false`       |
| `SGLANG_NPU_USE_MULTI_STREAM`                    | Enable dual-stream computation of shared experts <br/> and routing experts in DeepSeek models.<br/> Enable dual-stream computation in DeepSeek NSA Indexer. | `false`       |
| `SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT`           | Disable cast model weight tensor to a specific NPU <br/> ACL format.                                                                                        | `false`       |
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | The maximum number of dispatched tokens on each rank.                                                                                                       | `128`         |

## Used in DeepEP Ascend

| Environment Variable                      | Description                                                                                                            | Default Value |
|-------------------------------------------|------------------------------------------------------------------------------------------------------------------------|---------------|
| `DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS` | Enable ant-moving function in dispatch stage. Indicates <br/> the number of tokens transmitted per round on each rank. | `8192`        |
| `DEEPEP_NORMAL_LONG_SEQ_ROUND`            | Enable ant-moving function in dispatch stage. Indicates <br/> the number of rounds transmitted on each rank.           | `1`           |
| `DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ`   | Enable ant-moving function in combine stage. <br/> The value `0` means disabled.                                       | `0`           |
| `MOE_ENABLE_TOPK_NEG_ONE`                 | Needs to be enabled when the expert ID to be processed by <br/> DEEPEP contains -1.                                    | `0`           |
| `DEEP_NORMAL_MODE_USE_INT8_QUANT`         | Quantizes x to int8 and returns (tensor, scales) in dispatch operator.                                                 | `0`           |

## Others

| Environment Variable     | Description                                                                                                                                                                                                                                                                | Default Value |
|--------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| `TASK_QUEUE_ENABLE`      | Used to control the optimization level of the dispatch queue<br/> about the task_queue operator. [Detail](https://www.hiascend.com/document/detail/zh/Pytorch/730/comref/Envvariables/docs/zh/environment_variable_reference/TASK_QUEUE_ENABLE.md)                         | `1`           |
| `INF_NAN_MODE_ENABLE`    | Controls whether the chip uses saturation mode or INF_NAN mode. [Detail](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha001/apiref/envref/envref_07_0056.html)                                                                                   | `1`           |
| `STREAMS_PER_DEVICE`     | Configures the maximum number of streams for the stream pool. [Detail](https://www.hiascend.com/document/detail/zh/Pytorch/720/comref/Envvariables/Envir_041.html)                                                                                                         | `32`          |
| `PYTORCH_NPU_ALLOC_CONF` | Controls the behavior of the cache allocator. <br/>This variable changes memory usage and may cause performance fluctuations. [Detail](https://www.hiascend.com/document/detail/zh/Pytorch/700/comref/Envvariables/Envir_012.html)                                         |               |
| `ASCEND_MF_STORE_URL`    | The address of config store in MemFabric during PD separation, <br/>which is generally set to the IP address of the P primary node<br/> with an arbitrary port number.                                                                                                     |               |
| `ASCEND_LAUNCH_BLOCKING` | Controls whether synchronous mode is enabled during operator execution. [Detail](https://www.hiascend.com/document/detail/zh/Pytorch/710/comref/Envvariables/Envir_006.html)                                                                                               | `0`           |
| `HCCL_OP_EXPANSION_MODE` | Configures the expansion position for communication algorithm scheduling. [Detail](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha001/apiref/envref/envref_07_0094.html)                                                                         |               |
| `HCCL_BUFFSIZE`          | Controls the size of the buffer area for shared data between two NPUs. <br/>The unit is MB, and the value must be greater than or equal to 1. [Detail](https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/ptmoddevg/trainingmigrguide/performance_tuning_0047.html) | `200`         |
| `HCCL_SOCKET_IFNAME`     | Configures the name of the network card used by the Host <br/>during HCCL initialization. [Detail](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/apiref/envvar/envref_07_0075.html)                                                                     |               |
| `GLOO_SOCKET_IFNAME`     | Configures the network interface name for GLOO communication.                                                                                                                                                                                                              |               |
