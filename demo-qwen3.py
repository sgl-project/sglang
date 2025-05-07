from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = f"http://127.0.0.1:30000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

model = "Qwen/Qwen3-8B"  # Use the model loaded by the server
messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]

response = client.chat.completions.create(
    model=model,
    messages=messages,
    extra_body={
        "chat_template_kwargs": {"enable_thinking": True},
        "separate_reasoning": True,
        "thinking_budget": 20,
    },
)

print(
    "response.choices[0].message.reasoning_content: \n",
    response.choices[0].message.reasoning_content,
)
print("response.choices[0].message.content: \n", response.choices[0].message.content)

"""
$ python3 -m sglang.launch_server --model Qwen/Qwen3-8B --reasoning-parser qwen3

$ python3 demo-qwen3.py

server-side:
[2025-05-07 09:50:01] Prefill batch. #new-seq: 1, #new-token: 1, #cached-token: 21, token usage: 0.00, #running-req: 0, #queue-req: 0
[2025-05-07 09:50:01] Decode batch. #running-req: 1, #token: 59, token usage: 0.00, gen throughput (token/s): 0.96, #queue-req: 0
[2025-05-07 09:50:02] Decode batch. #running-req: 1, #token: 99, token usage: 0.00, gen throughput (token/s): 77.55, #queue-req: 0
[2025-05-07 09:50:02] Decode batch. #running-req: 1, #token: 139, token usage: 0.00, gen throughput (token/s): 77.55, #queue-req: 0
[2025-05-07 09:50:03] Decode batch. #running-req: 1, #token: 179, token usage: 0.00, gen throughput (token/s): 77.10, #queue-req: 0
[2025-05-07 09:50:03] Decode batch. #running-req: 1, #token: 219, token usage: 0.01, gen throughput (token/s): 77.24, #queue-req: 0
[2025-05-07 09:50:03] num_thinking_tokens: 199
[2025-05-07 09:50:04] Decode batch. #running-req: 1, #token: 259, token usage: 0.01, gen throughput (token/s): 78.60, #queue-req: 0
[2025-05-07 09:50:04] Decode batch. #running-req: 1, #token: 299, token usage: 0.01, gen throughput (token/s): 78.94, #queue-req: 0
[2025-05-07 09:50:05] Decode batch. #running-req: 1, #token: 339, token usage: 0.01, gen throughput (token/s): 79.01, #queue-req: 0
[2025-05-07 09:50:05] Decode batch. #running-req: 1, #token: 379, token usage: 0.01, gen throughput (token/s): 78.98, #queue-req: 0
[2025-05-07 09:50:06] Decode batch. #running-req: 1, #token: 419, token usage: 0.01, gen throughput (token/s): 78.98, #queue-req: 0
[2025-05-07 09:50:06] Decode batch. #running-req: 1, #token: 459, token usage: 0.01, gen throughput (token/s): 79.51, #queue-req: 0
[2025-05-07 09:50:06] INFO:     127.0.0.1:42554 - "POST /v1/chat/completions HTTP/1.1" 200 OK

client-side:
response.choices[0].message.reasoning_content: 
 Okay, so I need to figure out whether 9.11 is greater than or less than 9.8. Let me think. Both numbers are decimals, right? They both start with 9, so that's the same part. Then there's the decimal part. Let me write them down to compare them more clearly.

First, 9.11. That's 9 and 11 hundredths. Then 9.8 is 9 and 8 tenths. Hmm. Maybe I should convert them to the same decimal places to compare. Let me see. If I convert 9.8 to hundredths, that would be 9.80, right? Because 9.8 is the same as 9.80. So now, comparing 9.11 and 9.80.

Now, looking at the whole number part first. Both are 9, so that's equal. Then the decimal part
response.choices[0].message.content: 
 To determine which is greater between **9.11** and **9.8**, we can compare them step by step:

1. **Compare the whole number parts**:  
   Both numbers have **9** as the whole number part.  
   So, **9 = 9**.

2. **Compare the decimal parts**:  
   - **9.11** has a decimal part of **0.11** (11 hundredths).  
   - **9.8** has a decimal part of **0.8** (8 tenths).  
   To compare these, convert them to the same unit:  
   - **0.8** is equivalent to **0.80** (80 hundredths).  
   - **0.11** is **11 hundredths**.  

3. **Compare the decimal values**:  
   - **0.80** (from 9.8) is greater than **0.11** (from 9.11).  

### Final Answer:  
**9.8 is greater than 9.11.**  
$$
\boxed{9.8}
$$
"""