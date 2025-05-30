# Pipeline Parallelism(PP)

## What is Pipeline Parallelism?

Pipeline Parallelism is a technique to distribute a large deep learning model across multiple GPUs (or other devices). Instead of trying to fit the entire model onto a single GPU, the model's layers are partitioned into sequential stages, with each stage being assigned to a different GPU.

During inference, a mini-batch of data is processed through these stages in a pipelined fashion.
1.  The first GPU processes the first part of the model (stage 1) for a micro-batch.
2.  The output (activations) is then passed to the second GPU.
3.  While the second GPU processes stage 2 for that micro-batch, the first GPU can start processing stage 1 for the *next* micro-batch.

## When to Use Pipeline Parallelism

You should consider using Pipeline Parallelism when:

1.  **Model Size Exceeds Single GPU Memory:** This is the primary use case. If your model is too large to fit onto one GPU, PP allows you to split it.
2.  **You Have Multiple GPUs Available:** PP inherently requires multiple processing units.
3.  **Seeking Higher Throughput:** For very large models, PP can increase throughput by parallelizing the execution across GPUs, although there's an overhead for communication between stages(less than TP).

## How to Use Pipeline Parallelism

### Basic Usage

For example, if you want to run a model across 2 GPUs using pipeline parallelism:

```bash
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --pp 2
```

SGLang will attempt to automatically partition the model's layers as evenly as possible across the specified number of pipeline stages. The embedding layer and the final language model head/normalization are usually placed on the first and last stages, respectively. The transformer blocks (or equivalent repeating units) are then distributed among the stages.

### Customizing Layer Partitioning (`SGLANG_PP_LAYER_PARTITION`)

While automatic partitioning works well for many cases, you might want to fine-tune how layers are distributed, perhaps due to specific layer characteristics or memory footprints. SGLang allows this through the `SGLANG_PP_LAYER_PARTITION` environment variable.

**Understanding `SGLANG_PP_LAYER_PARTITION`**

This environment variable takes a comma-separated list of integers. Each integer specifies the number of *transformer layers* (or the main repetitive blocks of the model) to be placed on the corresponding pipeline stage.

*   The number of integers in the list **must** match the `--pp` you specify.
*   The sum of the integers in the list **must** equal the total number of transformer layers in the model you are using.

**Example:**

Suppose you are using a model with 32 transformer layers (e.g., Llama-2 7B) and you want to use 3 GPUs for pipeline parallelism (`--pp 3`).
You could customize the partitioning like this:

*   Place the first 10 layers on GPU 0.
*   Place the next 10 layers on GPU 1.
*   Place the final 12 layers on GPU 2.

To achieve this, you would set the environment variable before running SGLang:

```bash
export SGLANG_PP_LAYER_PARTITION="10,10,12"
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --pp 3
```

**Important Notes:**
*   You need to know the total number of transformer layers in your specific model. This information is usually available in the model's configuration file or documentation.
*   The `SGLANG_PP_LAYER_PARTITION` variable only controls the distribution of the main transformer blocks. Embedding layers, final normalization, and LM heads are typically handled automatically and assigned to the first and last stages.

## Supported Models

SGLang officially supports Pipeline Parallelism for the following models:

*   Llama family (e.g., Llama-2 7B, 13B, 70B)
*   Mistral (e.g., Mistral-7B)
*   Qwen2/Qwen3/Qwen2-MoE/Qwen3-MoE

This list is actively growing. Please check the latest SGLang documentation or release notes for the most up-to-date list.


## Contributing: Enabling Pipeline Parallelism for New Models


### General Principles

1.  **Identify Sequential Blocks:** Most transformer-based models have a clear structure: an embedding layer, a series of identical (or structurally similar) transformer blocks, and a final normalization/output layer. PP primarily focuses on splitting these transformer blocks.
2.  **Device Placement:** The core modification involves ensuring that each part of the model (embeddings, specific transformer blocks, LM head) can be explicitly placed on the correct GPU corresponding to its pipeline stage.
3.  **Tensor Forwarding:** Hidden states (activations) must be correctly passed from one GPU (stage) to the next. SGLang's PP runtime will handle the actual communication, but the model code needs to ensure it sends its output to the next stage's device or receives input from the previous stage's device.
4.  **Configuration Awareness:** The model needs to be aware of its pipeline rank (which stage it is) and the total number of pipeline stages to correctly determine which layers it should execute.

### Steps to Modify a Model

*Study Existing Implementations:** The best way to start is by looking at how PP is implemented for already supported models in the SGLang codebase (e.g., the Qwen model implementation).

1.  **Modify the Casual Model's `__init__`:**
    *   Inherit `SupportsPP`, The model class will get `pipeline_rank` and `start_layer`, `end_layer` parameters from SGLang.

2.  **Modify the Casual Model's `load_weights`:**
    *   Call the `filter_weights_by_layers` before visit the weights

3. **Modify the Model's `__init__`:** This is where most changes occur.
    *   **Embedding Layer:** If `pipeline_rank == 0`, execute the embedding layer and ensure its output tensor is on the device for stage 0.
    *   **Transformer Blocks:**
        *   Determine the range of transformer blocks this `pipeline_rank` is responsible for (e.g., `layers_this_stage = self.layer_partitions[self.pipeline_rank]`).
        *   Iterate *only* through these assigned layers.
        *   Before executing a layer, ensure it (and its inputs) are on the correct device for the current `pipeline_rank`. SGLang might provide a utility like `get_pp_device(rank)`.
        *   `hidden_states = layer(hidden_states, ...)`
    *   **Final Layer (Normalization, LM Head):** If `pipeline_rank == world_size - 1`, execute the final normalization layer and the LM head. Ensure its output is on the device for the last stage.
4.  **Adapt the Model's `forward` Pass:**
    *   Add `pp_proxy_tensors`  parameter to connect different layers.
    *   **Inter-GPU Transfer:**
        *   After processing its assigned layers, if a stage is not the last one, its output `hidden_states` must be moved to the device of the *next* stage (`pipeline_rank + 1`).
        *   Similarly, a stage (except the first) must expect its input `hidden_states` to arrive from the *previous* stage (`pipeline_rank - 1`). SGLang's PP communication primitives will likely handle the send/recv operations. Your model code needs to ensure tensors are on the correct device before and after these implicit communications.



6.  **Testing:**
    take the following as example ,but change the model to your self.
    ```
    python3 -m unittest test_pp_single_node.TestQwenPPAccuracy.test_pp_consistency
    ```
