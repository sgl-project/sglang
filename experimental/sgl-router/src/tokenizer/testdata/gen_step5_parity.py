"""Refresh step5_parity.json's expected_ids from a live SGLang Step-5 engine.

    ENGINE_URL=http://<engine>:30000 MODEL=<served model id> python gen_step5_parity.py

Each case's request is sent to /v1/chat/completions with max_tokens=1 and
return_prompt_token_ids=true; the engine's ids are stored with its extra leading
BOS dropped (text-only requests of a multimodal model are re-tokenized with
add_special_tokens=True, prepending a second BOS the forwarded ids do not carry).
Point it at the engine directly, not a router: a router may forward its own ids.
"""

import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "step5_parity.json")
URL = os.environ["ENGINE_URL"].rstrip("/") + "/v1/chat/completions"
MODEL = os.environ["MODEL"]

fixture = json.load(open(PATH))
for case in fixture["cases"]:
    body = dict(case["request"], model=MODEL, max_tokens=1, return_prompt_token_ids=True)
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    ids = json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["prompt_token_ids"]
    case["expected_ids"] = ids[1:] if ids[:2] == [0, 0] else ids
    print(f"{case['name']}: {len(case['expected_ids'])} ids")
json.dump(fixture, open(PATH, "w"), ensure_ascii=False, indent=1)
