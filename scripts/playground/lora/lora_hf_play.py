import torch
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
# ADAPTER = "winddude/wizardLM-LlaMA-LoRA-7B"
ADAPTER = "/home/ying/test_lora"
HF_TOKEN = "..."


prompt = """
### Instruction:
Write a poem about the transformers Python library.
Mention the word "large language models" in that poem.
### Response:
The Transformers are large language models,
They're used to make predictions on text.
"""


tokenizer = LlamaTokenizer.from_pretrained(MODEL)

base_model = LlamaForCausalLM.from_pretrained(
    MODEL,
    device_map="auto",
    # load_in_8bit=True,
    torch_dtype=torch.float16,
    # use_auth_token=HF_TOKEN,
).cuda()


# base model generate
with torch.no_grad():
    output_tensors = base_model.generate(
        input_ids=tokenizer(prompt, return_tensors="pt").input_ids.cuda(),
        max_new_tokens=32,
        do_sample=False,
    )[0]

output = tokenizer.decode(output_tensors, skip_special_tokens=True)
print("======= base output ========")
print(output)


# peft model generate
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER,
    torch_dtype=torch.float16,
    is_trainable=False,
)

with torch.no_grad():
    output_tensors = model.generate(
        input_ids=tokenizer(prompt, return_tensors="pt").input_ids.cuda(),
        max_new_tokens=32,
        do_sample=False,
    )[0]

output = tokenizer.decode(output_tensors, skip_special_tokens=True)
print("======= peft output ========")
print(output)
