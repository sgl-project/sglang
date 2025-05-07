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
        "thinking_budget": 1000,
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
response.choices[0].message.reasoning_content:
 Okay, so I need to figure out which number is greater between 9.11 and 9.8. Let me start by recalling how to compare decimal numbers. From what I remember, when comparing decimals, you start by looking at the whole number part first. If the whole numbers are different, then the one with the larger whole number is the bigger number. If the whole numbers are the same, then you move on to the decimal parts.

So, in this case, both numbers have the same whole number part, which is 9. That means I need to look at the decimal parts to determine which one is larger. The decimal part of 9.11 is 0.11, and the decimal part of 9.8 is 0.8.

Now, comparing 0.11 and 0.8. Let me think. Since 0.8 is the same as 0.80, right? Because adding a zero at the end doesn't change the value. So, 0.80 versus 0.11. Let me write them down aligned by their decimal places:

0.80
0.11

Starting from the leftmost digit after the decimal, the first digit is 8 versus 1. Since 8 is greater than 1, that means 0.80 is greater than 0.11. Therefore, 9.8 is greater than 9.11.

Wait, let me double-check. Maybe I should convert them to fractions to verify. Let's see. 0.11 is 11/100, and 0.8 is 8/10, which is 80/100. Comparing 11/100 and 80/100, clearly 80/100 is larger. So, yes, 0.8 is larger than 0.11. Therefore, 9.8 is greater than 9.11.

But just to be thorough, maybe I can think of them on a number line. 9.11 is between 9 and 10, closer to 9, while 9.8 is also between 9 and 10 but closer to 10. Since 9.8 is 0.8 away from 9 and 0.2 away from 10, whereas 9.11 is 0.11 away from 9 and 0.89 away from 10. So, 9.8 is much closer to 10, which makes it larger than 9.11.

Another way is to subtract them. Let's compute 9.8 - 9.11. If the result is positive, then 9.8 is larger. Let me do the subtraction:

9.8 is the same as 9.80. Subtracting 9.11 from 9.80:

9.80
-9.11
------
0.69

So, the result is 0.69, which is positive. Therefore, 9.8 is greater than 9.11 by 0.69.

Alternatively, maybe I can think in terms of place value. The first decimal place is tenths, and the second is hundredths. For 9.11, the tenths place is 1, and the hundredths is 1. For 9.8, the tenths place is 8, and there's no hundredths digit, which is the same as 0. So, comparing the tenths place first, 8 is greater than 1. Therefore, even without looking at the hundredths place, since the tenths place of 9.8 is larger, the entire number is larger.

Wait, but sometimes people might think that because 9.11 has two decimal places and 9.8 has one, but that doesn't matter. The key is that when comparing decimals, you compare each digit from left to right. So first, the whole number part. Then tenths, then hundredths, etc. So, since the tenths place is 1 vs. 8, that's sufficient. So, even if 9.11 has more decimal places, the tenths place is already higher in 9.8, making it larger.

So all these methods lead me to the same conclusion: 9.8 is greater than 9.11. I don't see any contradictions here. All the different ways of comparing confirm that 9.8 is the larger number.

Just to make sure I haven't made any mistakes in my calculations, let me verify once again:

9.8 is equal to 9 + 0.8, and 9.11 is 9 + 0.11. Since 0.8 > 0.11, then 9.8 > 9.11. Yep, that's straightforward.

Alternatively, if I convert both numbers to fractions with the same denominator. Let me try that. 9.11 is 911/100, and 9.8 is 98/10. To compare them, I can convert them to have the same denominator. Let's find the least common denominator of 100 and 10. That would be 100. So, 98/10 is equal to 980/100. Comparing 911/100 and 980/100. Since 980 is more than 911, 980/100 is greater. Therefore, 9.8 is greater than 9.11.

Another check: if I add 0.11 to 9, I get 9.11. If I add 0.8 to 9, I get 9.8. Since 0.8 is a larger increment than 0.11, adding it to 9 gives a larger number. So that also makes sense.

I think I've verified this multiple ways. All methods consistently show that 9.8 is greater than 9.11. I can't see any error in my reasoning here. Therefore, the answer should be 9.8.

response.choices[0].message.content:
 To determine which number is greater between **9.11** and **9.8**, we can follow a systematic approach to compare decimal numbers:

---

### **Step 1: Compare the Whole Number Parts**

Both numbers have the same whole number part:

- 9.11 → whole number = **9**
- 9.8 → whole number = **9**

Since the whole numbers are equal, we move on to compare the **decimal parts**.

---

### **Step 2: Compare the Decimal Parts**

- **9.11** has a decimal part of **0.11**
- **9.8** has a decimal part of **0.8**

We can convert 0.8 to **0.80** to align the decimal places:

- **0.80** vs. **0.11**

Now compare digit by digit:

- First decimal place: **8** (from 0.80) vs. **1** (from 0.11) → **8 > 1**

Since the first decimal digit of 9.8 is larger, **0.80 > 0.11**, and therefore:

> **9.8 > 9.11**

---

### **Step 3: Additional Verification via Subtraction**

Let’s subtract 9.11 from 9.8 to confirm:

$$
9.80 - 9.11 = 0.69
$$

Since the result is **positive**, this confirms that **9.8 is greater than 9.11**.

---

### **Conclusion**

After comparing the whole number parts, analyzing the decimal places, and verifying through subtraction, we consistently find that:

$$
\boxed{9.8 > 9.11}
$$
"""


# TODO: Add doc
# TODO: Add test
# TODO: 模仿stop_token_ids进行实现

# Q: 如何强行插入一个token？
# A: 通过修改mask
