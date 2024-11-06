from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER = "/home/ying/test_lora"
prompt = """
### Instruction:
Write a poem about the transformers Python library.
Mention the word "large language models" in that poem.
### Response:
The Transformers are large language models,
They're used to make predictions on text.
"""


llm = LLM(model=MODEL, enable_lora=True)

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=32,
)

prompts = [prompt]

outputs = llm.generate(
    prompts, sampling_params, lora_request=LoRARequest("test_lora", 1, ADAPTER)
)

print(outputs[0].prompt)
print(outputs[0].outputs[0].text)
