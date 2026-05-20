# MSProbe Debugging Guide

## Introduction to MSProbe

MSProbe is a debugging tool for AI models that diagnoses accuracy anomalies and
numerical errors during model training and inference. It captures and monitors intermediate data (feature maps, weights,
activations, layer outputs) and contextual metadata (prompts, tensor dtypes, hardware configuration), and supports
visual analysis to systematically trace the root cause of accuracy degradation or numerical errors (e.g., NaN/Inf,
output drift, mismatched predictions).

## Basic Details

### Background Concepts: MSProbe Dumping Levels

MSProbe supports three accuracy levels for data dumping, each for different debugging needs:

- **L0**: Dumps tensors/statistics at the **module level** and generates `construct.json` (for network structure
  reconstruction in visualization). Requires passing a model/submodule handle.
- **L1**: Dumps tensors/statistics at the **torch API level**, suitable for fine-grained API-level numerical checking.
- **mix**: Combines L0 + L1, ideal for scenarios that require both **graph reconstruction** and **numerical comparison**.

### Prerequisites: Install MSProbe

Install MSProbe with pip:

```shell
pip install mindstudio-probe --pre
```

### Key Configuration Parameters

MSProbe uses a JSON configuration file for customized data dumping. All core parameters are listed in the table below,
with the default JSON configuration provided for reference.

#### Configuration Parameter Table

|    Field     | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Required |
|:------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------:|
|    `task`    | Type of dump task. Common PyTorch values include `"statistics"` and `"tensor"`. A statistics task collects tensor statistics (mean, variance, max, min, etc.) while a tensor task captures arbitrary tensors.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |   Yes    |
| `dump_path`  | Directory where dump results are stored. When omitted, `MSProbe` uses its default path.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |    No    |
|    `rank`    | Ranks to sample. An empty list collects every rank. For single-card tasks you must set this field to `[]`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |    No    |
|    `step`    | Token iteration(s) to sample. An empty list means every iteration.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |    No    |
|   `level`    | Dump level string (`"L0"`, `"L1"`, or `"mix"`). `L0` targets `nn.Module`, `L1` targets `torch.api`, and `mix` collects both.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |   Yes    |
| `async_dump` | Whether to enable asynchronous dump (supported for PyTorch `statistics`/`tensor` tasks). Defaults to `false`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |    No    |
|   `scope`    | Customize the scope of dump. Provide two module or API names that follow the tool's naming convention to lock a range, only data between the two names will be dumped. An empty list dumps every module or torch API.<br/><br/>Examples:<br/>`"scope": ["Module.conv1.Conv2d.forward.0", "Module.fc2.Linear.forward.0"]`<br/>`"scope": ["Tensor.add.0.forward", "Functional.square.2.forward"]`<br/><br/>The `level` setting determines what can be provided—modules when `level=L0`, APIs when `level=L1`, and either modules or APIs when `level=mix`.                                                                                                                                                                                                                                 |    No    |
|    `list`    | Customize dump list, only dumps elements from the list. An empty list dumps every module or torch API. Options include:<br/><br/>&#738226;Supply the full names of specific APIs in PyTorch pynative scenarios to only dump those APIs. Example: `"list": ["Tensor.permute.1.forward", "Tensor.transpose.2.forward", "Torch.relu.3.backward"]`.<br/>&#738226;When `level=mix`, you can provide module names so that the dump expands to everything produced while the module is running. Example: `"list": ["Module.module.language_model.encoder.layers.0.mlp.ParallelMlp.forward.0"]`.<br/>&#738226;Provide a substring such as `"list": ["relu"]` to dump every API whose name contains the substring. When `level=mix`, modules whose names contain the substring are also expanded. |    No    |

#### Default configuration

```json
{
  "task": "statistics",
  "dump_path": "./dump_path",
  "rank": [],
  "step": [],
  "level": "L1",
  "async_dump": false,
  "statistics": {
    "scope": [],
    "list": [],
    "data_mode": [
      "all"
    ],
    "summary_mode": "statistics"
  },
  "tensor": {
    "scope": [],
    "list": [],
    "data_mode": [
      "all"
    ],
    "file_format": "npy"
  },
  "acc_check": {
    "white_list": [],
    "black_list": [],
    "error_data_path": "./"
  }
}
```

#### Outputs

Dump files are written into `dump_path` you defined. They usually contain:

- `dump.json`, which records metadata such as dtype, shape, min, max, mean, L2 norm, and `requires_grad`.
- `construct.json`, hierarchical structure description, when `level` is `L0` or `mix` (required for visualization), its
  content is not empty.
- `stack.json`, record the call stack information of API/Module.
- `dump_tensor_data`, generated when `task` is `tensor` and save the collected tensor data.

See [dump directory description](#dump-directory-description) for details.

> **Note**: When MSProbe is enabled, cuda graph is disabled(disable_cuda_graph=True) because MSProbe only supports dump
> in eager mode, warmup is disabled(skip_server_warmup=True) because there is no need to dump data for this stage.

## End-to-End Examples

MSProbe’s full debugging workflow follows **Enable → Collect Data → Visualize → Analyze Root Cause**. Below is a common
E2E example for SGLang-based model inference debugging.

### Example : Advanced Debugging with Custom Configuration

Suitable for targeted debugging (e.g., only collect statistics data for specific ranks/steps, enable mix level for graph
reconstruction + numerical comparison) and root cause analysis via **problem vs. benchmark comparison**.

#### Step 1: Enable
##### Prepare Custom Configuration JSON

Create `msprobe-config.json` (dump statistics data for rank0/1, step0/1, mix level):

```json
{
  "task": "statistics",
  "dump_path": "./problem_dump",
  "rank": [
    0,
    1
  ],
  "step": [
    0,
    1
  ],
  "level": "mix",
  "async_dump": false,
  "statistics": {
    "scope": [],
    "list": [],
    "data_mode": [
      "all"
    ],
    "summary_mode": "statistics"
  }
}
```

##### Enable MSProbe with Custom Configuration in SGLang

Launch the SGLang server and specify the configuration file path with `--msprobe-dump-config`:

```bash
python3 -m sglang.launch_server \
 --model-path Qwen/Qwen2.5-0.5B-Instruct \
 --host 127.0.0.1 \
 --port 1027 \
 --msprobe-dump-config /home/msprobe-config.json
```
#### Step 2: Collect Data
##### Collect Dump Data for Problem & Benchmark Sides

Send normal inference requests to trigger model running (MSProbe automatically collects data during request processing):

```bash
curl -H "Content-type: application/json" \
 -X POST \
 -d '{
     "model": "Qwen/Qwen2.5-0.5B-Instruct",
     "messages": [
         {
             "role": "user",
             "content": "Hello, my name is"
         }
     ],
     "max_tokens": 10
 }' \
 http://127.0.0.1:1027/v1/chat/completions
```

- **Problem side**: Run the above SGLang server (with the accuracy/numerical issue) and send inference request; dump
  data is saved to `./problem_dump`.
- **Benchmark side**: Launch a normal SGLang server (without the issue, e.g., stable framework version/operator) with
  the **same custom configuration** and send the **same inference request**; rename the dump directory
  to `./bench_dump`.

> **Key Requirement**: Problem and benchmark dumps must use the same inputs and sampling points (rank/step)
> for valid comparison.

##### Check Generated Dump Files

Dump files are saved to `./problem_dump` and `./bench_dump` you defined and include core files for subsequent analysis:

- `dump.json`: Records tensor metadata of APIs and modules (dtype, shape, min/max/mean, L2 norm, `requires_grad`, etc.).
- `stack.json`: Logs call stack information of APIs and modules.
- `construct.json`: hierarchical structure description, required for visualization, its content is not empty.

#### Step 3: Visualize
##### Visualize Problem vs. Benchmark Comparison (Multi-Rank)

Generate a multi-rank comparison visualization file (mix level generates `construct.json` for graph reconstruction):

```shell
msprobe graph_visualize -tp ./problem_dump/step0 -gp ./bench_dump/step0 -o ./graph_output
```

- `-tp`: Path to problem-side dump data
- `-gp`: Path to benchmark-side dump data
- `-o`: Output directory for visualization files

If you want overflow check (for NaN/Inf detection), please specify the parameter `-oc`

```shell
msprobe graph_visualize -tp ./problem_dump/step0 -gp ./bench_dump/step0 -o ./graph_output -oc
```

After the comparison or build task finishes, a `compare_{timestamp}.vis.db` file is created under `graph_output`.

##### Launch TensorBoard

Start TensorBoard:
```bash
tensorboard --logdir ./graph_output --bind_all --port 6006
```
#### Step 4: Analyze Root Cause
##### Locate Root Cause

Root Cause Analysis in TensorBoard:
- Divergent nodes (with accuracy/numerical differences) are highlighted in **red** (darker red = larger difference).
- Click on divergent nodes to view detailed tensor data (inputs/outputs, parameters) and API/module call stacks.
- Use the **search/filter** function to quickly locate key layers/APIs (e.g., "relu", "conv").
- Switch between ranks/steps via the UI to check cross-rank/cross-step divergence.
- Check the **overflow check** tab for NaN/Inf values in specific nodes (the direct cause of numerical instability).

##### Verify the Root Cause

After locating the divergent node (e.g., a specific Conv layer or torch API with abnormal tensor values), verify by:

- Narrowing the dump scope to this node (via `scope`/`list` in the configuration file) for fine-grained data collection.
- Modifying the problematic layer/API (e.g., replacing the operator, adjusting the dtype) and re-running the debugging
  workflow to confirm the issue is resolved.

## Troubleshooting

### No Dump Files Generated

1. To confirm if MSProbe is installed, use `pip show mindstudio_probe` to troubleshoot. If it is installed, the MSProbe
   version information will be printed. If it is confirmed that it has not been installed, please
   use `pip install mindstudio-probe --pre` for installation;
2. Confirm the `--msprobe-dump-config` parameter points to the **correct JSON file path**.

### Dump Files Are Too Large (Excessive Data)

1. Start with `task: "statistics"` instead of `"tensor"` to collect only tensor statistics (avoids raw tensor dump);
2. Narrow the dump range with the `scope` field (specify start/end module/API);
3. Filter dump targets with the `list` field (only dump specific modules/APIs or substrings);
4. Sample specific `rank` and `step` (avoid dumping all ranks/iterations).

### TensorBoard Visualization Fails

1. Confirm `construct.json` is not empty (requires `level: L0` or `mix` – L1 does not generate graph files);
2. Check that the `-tp` (problem dump) and `-gp` (benchmark dump) paths point to **valid rank/step subdirectories** (
   e.g., `srep0/rank0`);
3. Ensure the MSProbe version is up-to-date (reinstall with `pip install mindstudio-probe --pre --upgrade`);
4. Verify TensorBoard is installed and the `--logdir` parameter points to the directory containing `.vis.db` files (not
   the file itself).

### Numerical Comparison Shows No Divergence But Model Accuracy Is Low

1. Expand the dump `step` range (check more token iterations for late-stage divergence);
2. Switch to `task: "tensor"` (statistics may mask subtle numerical differences in raw tensor data);
3. Ensure the problem and benchmark dumps use **the same input data/hardware configuration** (different inputs lead to
   invalid comparisons);
4. Use the `manual mapping` feature in TensorBoard (automatic mapping may miss some nodes for custom models).

---

## Appendix

### Dump directory description

```text
├── problem_dump or bench_dump
│   ├── step0
│   │   ├── rank0
│   │   │   ├── dump_tensor_data
│   │   │   │    ├── Tensor.permute.1.forward.pt
│   │   │   │    ├── Functional.linear.5.backward.output.pt    # Format: {api_type}.{api_name}.{call_count}.{forward/backward}.{input/output}.{arg_index}.
│   │   │   │    │                                              # arg_index is the nth input or output of the API. If an input is a list, keep numbering with decimals (e.g., 1.1 is the first element of the first argument).
│   │   │   │    ├── Module.conv1.Conv2d.forward.0.input.0.pt          # Format: {Module}.{module_name}.{class_name}.{forward/backward}.{call_count}.{input/output}.{arg_index}.
│   │   │   │    ├── Module.conv1.Conv2d.forward.0.parameters.bias.pt  # Module parameter data: {Module}.{module_name}.{class_name}.forward.{call_count}.parameters.{parameter_name}.
│   │   │   │    └── Module.conv1.Conv2d.parameters_grad.weight.pt     # Module parameter gradients: {Module}.{module_name}.{class_name}.parameters_grad.{parameter_name}. Gradients do not include call_count because the same gradient updates all invocations.
│   │   │   │                                                          # When the `model` argument passed to dump is a List[torch.nn.Module] or Tuple[torch.nn.Module], module-level data names also include the index inside the list ({Module}.{index}.*), e.g., Module.0.conv1.Conv2d.forward.0.input.0.pt.
│   │   │   ├── dump.json
│   │   │   ├── stack.json
│   │   │   ├── dump_error_info.log
│   │   │   └── construct.json
│   │   ├── rank1
│   │   │   ├── dump_tensor_data
│   │   │   │   └── ...
│   │   │   ├── dump.json
│   │   │   ├── stack.json
│   │   │   ├── dump_error_info.log
│   │   │   └── construct.json
│   │   ├── ...
│   │   │
│   │   └── rank7
│   ├── step1
│   │   ├── ...
│   ├── step2
```

- `rank`: Device ID. Each card writes its data to the corresponding `rank{ID}` directory. In non-distributed scenarios
  the directory is simply named `rank`.
- `dump_tensor_data`: Save the collected tensor data.
- `dump.json`: Statistics for the forward data of each API or module, including names, dtype, shape, max, min, mean, L2
  norm (square root of the L2 variance), and CRC-32 when `summary_mode="md5"`.
  See [dump.json file description](#dumpjson-file-description) for details.
- `dump_error_info.log`: Present only when the dump tool encountered an error and records the failure log.
- `stack.json`: Call stacks for APIs/modules.
- `construct.json`: Hierarchical structure description. Empty when `level=L1`.

### dump.json file description

#### L0 level

An L0 `dump.json` contains forward/backward I/O for modules together with parameters and parameter gradients. Using
PyTorch's `Conv2d` as an example, the network code looks like:

`output = self.conv2(input)  # self.conv2 = torch.nn.Conv2d(64, 128, 5, padding=2, bias=True)`

`dump.json` contains the following entries:

- `Module.conv2.Conv2d.forward.0`: Forward data of the module. `input_args` represents positional inputs, `input_kwargs`
  represents keyword inputs, `output` stores forward outputs, and `parameters` stores weights/biases.
- `Module.conv2.Conv2d.parameters_grad`: Parameter gradients (weight and bias).
- `Module.conv2.Conv2d.backward.0`: Backward data of the module. `input` represents gradients that flow into the
  module (gradients of the forward outputs) and `output` represents gradients that flow out (gradients of the module
  inputs).

**Note**: When the `model` parameter passed to the dump API is `List[torch.nn.Module]` or `Tuple[torch.nn.Module]`,
module-level names include the index inside the list (`{Module}.{index}.*`). Example: `Module.0.conv1.Conv2d.forward.0`.

<details>

<summary>L0 dump.json</summary>

```json
{
  "task": "tensor",
  "level": "L0",
  "framework": "pytorch",
  "dump_data_dir": "/dump/path",
  "data": {
    "Module.conv2.Conv2d.forward.0": {
      "input_args": [
        {
          "type": "torch.Tensor",
          "dtype": "torch.float32",
          "shape": [
            8,
            16,
            14,
            14
          ],
          "Max": 1.638758659362793,
          "Min": 0.0,
          "Mean": 0.2544615864753723,
          "Norm": 70.50277709960938,
          "requires_grad": true,
          "data_name": "Module.conv2.Conv2d.forward.0.input.0.pt"
        }
      ],
      "input_kwargs": {},
      "output": [
        {
          "type": "torch.Tensor",
          "dtype": "torch.float32",
          "shape": [
            8,
            32,
            10,
            10
          ],
          "Max": 1.6815717220306396,
          "Min": -1.5120246410369873,
          "Mean": -0.025344856083393097,
          "Norm": 149.65576171875,
          "requires_grad": true,
          "data_name": "Module.conv2.Conv2d.forward.0.output.0.pt"
        }
      ],
      "parameters": {
        "weight": {
          "type": "torch.Tensor",
          "dtype": "torch.float32",
          "shape": [
            32,
            16,
            5,
            5
          ],
          "Max": 0.05992485210299492,
          "Min": -0.05999220535159111,
          "Mean": -0.0006165213999338448,
          "Norm": 3.421217441558838,
          "requires_grad": true,
          "data_name": "Module.conv2.Conv2d.forward.0.parameters.weight.pt"
        },
        "bias": {
          "type": "torch.Tensor",
          "dtype": "torch.float32",
          "shape": [
            32
          ],
          "Max": 0.05744686722755432,
          "Min": -0.04894155263900757,
          "Mean": 0.006410328671336174,
          "Norm": 0.17263513803482056,
          "requires_grad": true,
          "data_name": "Module.conv2.Conv2d.forward.0.parameters.bias.pt"
        }
      }
    },
    "Module.conv2.Conv2d.parameters_grad": {
      "weight": [
        {
          "type": "torch.Tensor",
          "dtype": "torch.float32",
          "shape": [
            32,
            16,
            5,
            5
          ],
          "Max": 0.018550323322415352,
          "Min": -0.008627401664853096,
          "Mean": 0.0006675920449197292,
          "Norm": 0.26084786653518677,
          "requires_grad": false,
          "data_name": "Module.conv2.Conv2d.parameters_grad.weight.pt"
        }
      ],
      "bias": [
        {
          "type": "torch.Tensor",
          "dtype": "torch.float32",
          "shape": [
            32
          ],
          "Max": 0.014914230443537235,
          "Min": -0.006656786892563105,
          "Mean": 0.002657240955159068,
          "Norm": 0.029451673850417137,
          "requires_grad": false,
          "data_name": "Module.conv2.Conv2d.parameters_grad.bias.pt"
        }
      ]
    },
    "Module.conv2.Conv2d.backward.0": {
      "input": [
        {
          "type": "torch.Tensor",
          "dtype": "torch.float32",
          "shape": [
            8,
            32,
            10,
            10
          ],
          "Max": 0.0015069986693561077,
          "Min": -0.001139344065450132,
          "Mean": 3.3215508210560074e-06,
          "Norm": 0.020567523315548897,
          "requires_grad": false,
          "data_name": "Module.conv2.Conv2d.backward.0.input.0.pt"
        }
      ],
      "output": [
        {
          "type": "torch.Tensor",
          "dtype": "torch.float32",
          "shape": [
            8,
            16,
            14,
            14
          ],
          "Max": 0.0007466732058674097,
          "Min": -0.00044813455315306783,
          "Mean": 6.814070275140693e-06,
          "Norm": 0.01474067009985447,
          "requires_grad": false,
          "data_name": "Module.conv2.Conv2d.backward.0.output.0.pt"
        }
      ]
    }
  }
}
```

</details>

#### L1 level

An L1 `dump.json` records forward/backward I/O for APIs. Using PyTorch's `relu` function as an
example (`output = torch.nn.functional.relu(input)`), the file contains:

- `Functional.relu.0.forward`: Forward data of the API. `input_args` are positional inputs, `input_kwargs` are keyword
  inputs, and `output` stores the forward outputs.
- `Functional.relu.0.backward`: Backward data of the API. `input` represents the gradients of the forward outputs,
  and `output` represents the gradients that flow back to the forward inputs.

<details>

<summary>L1 dump.json</summary>

```json
{
  "task": "tensor",
  "level": "L1",
  "framework": "pytorch",
  "dump_data_dir": "/dump/path",
  "data": {
    "Functional.relu.0.forward": {
      "input_args": [
        {
          "type": "torch.Tensor",
          "dtype": "torch.float32",
          "shape": [
            32,
            16,
            28,
            28
          ],
          "Max": 1.3864083290100098,
          "Min": -1.3364859819412231,
          "Mean": 0.03711778670549393,
          "Norm": 236.20692443847656,
          "requires_grad": true,
          "data_name": "Functional.relu.0.forward.input.0.pt"
        }
      ],
      "input_kwargs": {},
      "output": [
        {
          "type": "torch.Tensor",
          "dtype": "torch.float32",
          "shape": [
            32,
            16,
            28,
            28
          ],
          "Max": 1.3864083290100098,
          "Min": 0.0,
          "Mean": 0.16849493980407715,
          "Norm": 175.23345947265625,
          "requires_grad": true,
          "data_name": "Functional.relu.0.forward.output.0.pt"
        }
      ]
    },
    "Functional.relu.0.backward": {
      "input": [
        {
          "type": "torch.Tensor",
          "dtype": "torch.float32",
          "shape": [
            32,
            16,
            28,
            28
          ],
          "Max": 0.0001815402356442064,
          "Min": -0.00013352684618439525,
          "Mean": 0.00011915402356442064,
          "Norm": 0.007598237134516239,
          "requires_grad": false,
          "data_name": "Functional.relu.0.backward.input.0.pt"
        }
      ],
      "output": [
        {
          "type": "torch.Tensor",
          "dtype": "torch.float32",
          "shape": [
            32,
            16,
            28,
            28
          ],
          "Max": 0.0001815402356442064,
          "Min": -0.00012117840378778055,
          "Mean": 2.0098118724831693e-08,
          "Norm": 0.006532244384288788,
          "requires_grad": false,
          "data_name": "Functional.relu.0.backward.output.0.pt"
        }
      ]
    }
  }
}
```

</details>

#### mix level

A `mix` dump.json contains both L0 and L1 level data; the file format is the same as the examples above.
