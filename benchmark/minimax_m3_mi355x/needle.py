import glob
import json
import sys
import time

import requests

url = sys.argv[1]
lengths = [int(x) for x in sys.argv[2].split(",")]
files = sorted(
    glob.glob("/sgl-workspace/sglang/python/sglang/srt/managers/*.py")
) + sorted(glob.glob("/sgl-workspace/sglang/python/sglang/srt/layers/*.py"))
corpus = "\n".join(open(f).read() for f in files)
for L in lengths:
    chars = L * 3
    doc = corpus[:chars]
    # the needle sits at 40% of the document
    k = int(len(doc) * 0.4)
    doc = doc[:k] + "\n# NOTE: the secret access code is 7391-ALPHA.\n" + doc[k:]
    msgs = [
        {
            "role": "user",
            "content": "Below is a code dump. Read it and then answer.\n\n"
            + doc
            + "\n\nQuestion: What is the secret access code mentioned in a NOTE comment? Reply with just the code.",
        }
    ]
    t = time.time()
    r = requests.post(
        url + "/v1/chat/completions",
        json=dict(
            model="MiniMax-M3",
            messages=msgs,
            max_tokens=48,
            temperature=0,
            chat_template_kwargs={"enable_thinking": False},
        ),
        timeout=3600,
    ).json()
    m = r["choices"][0]["message"]
    print(
        f"target~{L}: prompt_tokens={r['usage']['prompt_tokens']} dt={time.time() - t:.1f}s content={json.dumps(m.get('content'))[:120]} reasoning={json.dumps(m.get('reasoning_content'))[:80]}",
        flush=True,
    )
