import json

import transformers

content = "\n".join(
    open("llama2.txt", 'r', encoding='utf-8', errors='ignore').readlines())
content = content.replace("\n\n", "\n")

# Count token
name = "meta-llama/Llama-2-7b-chat-hf"
t = transformers.AutoTokenizer.from_pretrained(name)
print(f"num tokens: {len(t.encode(content))}")

# Segment
SEP = "\n\n"
parts = content.split(SEP)
print(f"num segments: {len(parts)}")

segment_len = 1100

segments = []
tmp = []
tmp_len = 0
for i in range(len(parts)):
    tmp.append(parts[i])
    tmp_len += len(t.encode(parts[i]))

    if tmp_len > segment_len:
        segments.append(SEP.join(tmp))
        tmp = []
        tmp_len = 0

for i, s in enumerate(segments):
    print(i, len(t.encode(segments[i])))

# Dump
with open("questions.jsonl", "w") as fout:
    fout.write(json.dumps({
        "documents": segments[:30],
        "questions": [
            "What is the name of the fine-tuned LLMs?",
            "Which figure shows the helpfulness human evaluation results for Llama 2-Chat?",
            "What is the number of parameters in the largest Llama 2 model?",
            "What is the batch size of fine-tuning?",
            "Where can we find the details of potential data contamination?",
            "What is the full name of MPT?",
            "What is the power consumption of RSC in Watt?",
            "How many tokens of data do they train on?",
            "Which model's release is delayed due to a lack of time to sufficiently red team?",
            "Which activation function is used in Llama?"
        ],
        "answers": [
            "Llama 2 Chat",
            "1",
            "70 B",
            "64",
            "A 6",
            "MosaicML",
            "400",
            "2 trillion",
            "34 B",
            "SwiGLU",
        ],
    }) + "\n")
