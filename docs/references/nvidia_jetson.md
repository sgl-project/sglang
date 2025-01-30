# Step-by-Step Guide to Use SGLang on NVIDIA Jetson Orin platform

This is a replicate from https://github.com/shahizat/SGLang-Jetson, thanks to the support from [shahizat](https://github.com/shahizat).
## Prerequisites

Before starting, ensure the following:

- **NVIDIA Jetson AGX Orin Devkit** is set up with **JetPack 6.1** or later.
- **CUDA Toolkit** and **cuDNN** are installed.
- Verify that the Jetson AGX Orin is in **high-performance mode**:
  ```bash
  sudo nvpmodel -m 0
  ```
- A custom PyPI index hosted at https://pypi.jetson-ai-lab.dev/jp6/cu126, tailored for NVIDIA Jetson Orin platforms and CUDA 12.6.

To install torch from this index:
  ```bash
pip install torch --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126
 ```
* * * * *

Step 1: Build FlashInfer from Source
------------------------------------

1.  Clone the FlashInfer repository:
```bash
git clone -b v0.2.0 https://github.com/flashinfer-ai/flashinfer.git --recursive
cd flashinfer
```

2.  Build FlashInfer:
```bash
python3 setup.py --verbose bdist_wheel --dist-dir ./
```
3.  Install FlashInfer:
```bash
pip3 install flashinfer-0.2.0-py3-none-any.whl
```
Expected Output:
```
Successfully installed flashinfer-0.2.0
```

Step 2: Build SGLang from Source
--------------------------------
Clone the SGLang repository:
```bash
git clone https://github.com/sgl-project/sglang
cd sglang
```
Build from source:
```bash
cd python
pip install -e .
```
Build the SGLang kernel:
```bash
cd sgl-kernel
python3 setup.py --verbose bdist_wheel --dist-dir ./
pip3 install sgl_kernel-0.0.2.post14-cp310-cp310-linux_aarch64.whl
```
* * * * *

Running Inference with FlashInfer Backend
-----------------------------------------

Launch the server:
```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --device cuda \
  --dtype half \
  --attention-backend flashinfer \
  --mem-fraction-static 0.8 \
  --context-length 8192
```

Run a sample inference script:\
    Create a Python script (e.g., `inference.py`) with the following content:
```bash
import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    messages=[
        {"role": "user", "content": "A pizza is divided into 8 slices. If 3 people eat 2 slices each, how many slices are left?"},
    ],
    temperature=0,
    max_tokens=500,
    stream=True  # Enable streaming
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```



Output

```bash
<think>
Alright , so I 've got this pizza problem here . Let me read it again to make sure I understand what 's being asked . It says , " A pizza is divided into 8 slices . If 3 people eat 2 slices each , how many slices are left ?" Okay , so we have 8 slices in total , and three people are each eating 2 slices . The question is asking how many slices are left after these three people have eaten their share .

First , I need to figure out the total number of slices that the three people will eat . Each person eats 2 slices , and there are 3 people . So , I can calculate the total slices eaten by multiplying the number of people by the number of slices each person eats . That would be 3 people times 2 slices per person , which equals 6 slices in total .

Now , since the pizza has 8 slices and 6 slices have been eaten , I need to find out how many slices are left . To do this , I 'll subtract the total number of slices eaten from the total number of slices available . So , 8 slices minus 6 slices eaten equals 2 slices remaining .

Wait , let me double -check that . If each of the three people eats 2 slices , that 's 2 + 2 + 2 , which is indeed 6 slices . And since the pizza only has 8 slices to begin with , subtract ing the 6 slices eaten leaves us with 2 slices . Yeah , that makes sense .

I 'm also thinking about whether there 's any possibility that the slices could be uneven ly distributed or if there 's something tricky with the way the pizza is divided . But the problem doesn 't mention anything like that , so I assume that the slices are uniform and that each person eats exactly 2 whole slices each .

Another thing to consider is if everyone eats their slices simultaneously or one after another . If they eat simultaneously , then all 6 slices are eaten at once , leaving 2 slices . If they eat one after another , the result would still be the same because each person is eating 2 slices , regardless of the order in which they eat them . So , the total eaten remains 6 slices , leaving 2 slices .

I 'm also ponder ing if there 's any alternative method to approach this problem . Maybe instead of calculating the total eaten first , I could subtract the number of slices eaten by each person from the total . So , if I take the total slices ( 8 ) and subtract 2 slices for the first person , that leaves me with 6 slices . Then subtract another 2 slices for the second person , leaving 4 slices . Then subtract another 2 slices for the third person , leaving me with 2 slices . This method also confirms the same answer .

Both methods lead me to the same conclusion , which reinforces that the number of slices left is indeed 2 .

I 'm also thinking about real -life scenarios where this might apply . Maybe at a party , a pizza is cut into slices , and several people each take their share . Understanding how many slices are left helps in planning for the next course or ensuring that everyone gets a fair share .

Additionally , this problem could be extended by introducing variables and algebra ic expressions to represent the number of people , slices per person , and total slices . For example , if P is the number of people , S is the number of slices per person , and T is the total number of slices , then the equation would be :

Total slices eaten = P * S

Slices left = T - ( P * S )

Using this formula , pl ugging in the given values :

Slices left = 8 - ( 3 * 2 ) = 8 - 6 = 2

This algebra ic approach provides a systematic way to solve similar problems in the future .

I also wonder what would happen if the number of people or slices per person changes . For instance , if there were 4 people each eating 2 slices , the total eaten would be 8 slices , leaving 0 slices . Or if there were 2 people eating 4 slices each , the total eaten would be 8 slices , leaving 0 slices . These variations could be interesting to explore as they showcase how the number of people and slices per person affect the number of slices left .

Furthermore , this problem could be visual ized with diagrams or graphs to provide a more intuitive understanding . For example , a bar graph showing the number of slices eaten by each person or a pie chart representing the total slices eaten relative to the whole pizza . Visual aids can often make mathematical concepts more accessible , especially for younger learners or those new to problem -solving techniques .

In summary , by carefully analyzing the problem , using multiple methods to verify the solution , and considering extensions and visual representations , I 'm confident that the number of slices left after 3 people each eat 2 slices is 2 .

After carefully analyzing the problem and verifying through multiple methods and considerations , we can confidently determine the number of slices left after 3 people each eat 2
</think>

```
Performance metrics
```bash
[2025-01-26 21:32:18 TP0] Decode batch. #running-req: 1, #token: 1351, token usage: 0.01, gen throughput (token/s): 11.19, #queue-req: 0
```
* * * * *

Running Inference with Torch Native Backend
-------------------------------------------

Launch the server:
```bash
python -m sglang.launch_server \
      --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
      --device cuda \
      --attention-backend torch_native \
      --mem-fraction-static 0.8 \
      --context-length 8192
```
Output
```bash
[2025-01-26 23:26:39 TP0] Decode batch. #running-req: 1, #token: 221, token usage: 0.00, gen throughput (token/s): 8.04, #queue-req: 0
```

* * * * *

Running Inference with Triton Backend
-------------------------------------
Launch the server:
```bash
python -m sglang.launch_server \
      --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
      --device cuda \
      --attention-backend triton \
      --mem-fraction-static 0.8 \
      --context-length 8192
```
Output
```bash
[2025-01-26 23:31:24 TP0] Decode batch. #running-req: 1, #token: 101, token usage: 0.00, gen throughput (token/s): 11.34, #queue-req:
```

* * * * *
Running quantization with TorchAO
-------------------------------------
Launch the server with the best configuration:
```bash
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --device cuda \
    --dtype bfloat16 \
    --attention-backend flashinfer \
    --mem-fraction-static 0.8 \
    --context-length 8192 \
    --torchao-config int4wo-128
```
This enables TorchAO's int4 weight-only quantization with 128-group size.

```bash
[2025-01-27 00:06:47 TP0] Decode batch. #running-req: 1, #token: 115, token usage: 0.00, gen throughput (token/s): 30.84, #queue-req:
```

* * * * *
Structured output with XGrammar
-------------------------------

Install XGrammar:
```bash
git clone --recursive https://github.com/mlc-ai/xgrammar.git && cd xgrammar
pre-commit install
mkdir build && cd build
cmake .. -G Ninja
ninja
cd ../python
python3 -m pip install .
```
Launch the server with XGrammar:
```bash
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --device cuda \
    --dtype bfloat16 \
    --attention-backend triton \
    --mem-fraction-static 0.8 \
    --context-length 8192 \
    --torchao-config int4wo-128 \
    --grammar-backend xgrammar
```
Run a structured output script:
```bash
import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Introduce yourself in JSON briefly."},
    ],
    temperature=0,
    max_tokens=500,
    stream=True  # Enable streaming
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```
Output
```bash
{
    "name": "Knowledge Assistant",
    "description": "A helpful AI assistant",
    "function": "Provide information and answer questions",
    "language": "English"
}
```
* * * * *

References
----------

-   [FlashInfer GitHub Repository](https://github.com/flashinfer-ai/flashinfer)

-   [SGLang GitHub Repository](https://github.com/sgl-project/sglang)

-   [NVIDIA Jetson AGX Orin Documentation](https://developer.nvidia.com/embedded/jetson-agx-orin)
