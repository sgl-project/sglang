import json

import requests

prompt = """
According to CNBC's Faber, the investors present on the call interpreted this statement as an indication of an upcoming funding round. While speculative, Faber believes the funding round could be as large as $25 billion, and bestow a valuation of between $150 billion and $200 billion on xAI.

For the benefit of those who might not be aware, xAI recently acquired the social media platform X in an all-stock deal that valued the former at $80 billion and the latter at $33 billion, inclusive of $12 billion in liabilities. This meant that the deal bestowed a gross valuation of $45 billion on X before factoring in its debt load of $12 billion.

Bear in mind that Elon Musk took X (then called Twitter) private back in 2022 in a $44 billion deal. Since then, Musk has managed to stem X's cash bleed, with the company reportedly generating $1.2 billion in adjusted EBITDA in 2024.

According to the investors present on the call, xAI is currently generating around $1 billion in annual revenue. This contrasts sharply with the erstwhile muted expectations of many investors, who did not expect the startup to generate any material revenue this year.

Elsewhere, Faber also alludes to the fact that xAI is already working on its next big training supercluster, officially dubbed the Colossus 2, which is expected to eventually house as many as 1 million NVIDIA GPUs at a cost of between $35 billion and $40 billion.


Even though xAI's Grok LLM is already largely comparable with OpenAI's cutting-edge models, the Colossus 2 would significantly up the ante, and could feasibly challenge OpenAI's apex position in the AI sphere.

Give your honest take on the above text:
"""

response = requests.post(
    "http://0.0.0.0:8000/generate",
    json={"text": prompt, "sampling_params": {"temperature": 0}},
)


response_json = response.json()
print(response_json["text"])
