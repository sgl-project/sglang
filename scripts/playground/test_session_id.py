# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# FIXME: Make it a CI test

import requests

from sglang.srt.hf_transformers_utils import get_tokenizer

url = "http://localhost:30000"

# Open a session
response = requests.post(
    url + "/open_session",
    json={"capacity_of_str_len": 1000},
)
session_id = response.json()
print("session_id", session_id, "\n")

# Prefill only
prompt = "chunk 1"
response = requests.post(
    url + "/generate",
    json={
        "text": prompt,
        "session_id": session_id,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
        },
    },
)
print(response.json(), "\n")
session_id = response.json()["session_id"]

# Generate
prompt = "Chunk 2"
response = requests.post(
    url + "/generate",
    json={
        "text": prompt,
        "session_id": session_id,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 16,
        },
    },
)
print(response.json(), "\n")
session_id = response.json()["session_id"]
rid = response.json()["meta_info"]["id"]

# Generate
prompt = "Chunk 3"
response = requests.post(
    url + "/generate",
    json={
        "text": prompt,
        "session_id": session_id,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 2,
        },
    },
)
print(response.json(), "\n")
session_id = response.json()["session_id"]
rid_to_del = response.json()["meta_info"]["id"]

# Interrupt and re-generate
prompt = "Chunk 4"
response = requests.post(
    url + "/generate",
    json={
        "text": prompt,
        "session_id": session_id,
        "session_rid": rid,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 16,
        },
    },
)
print(response.json(), "\n")
session_id = response.json()["session_id"]

# Query a session based on a deleted request, should see finish reason abort
prompt = "Chunk 4"
response = requests.post(
    url + "/generate",
    json={
        "text": prompt,
        "session_id": session_id,
        "session_rid": rid_to_del,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 16,
        },
    },
)
print(response.json(), "\n")

# Close session
ret = requests.post(
    url + "/close_session",
    json={"session_id": session_id},
)
print(ret, "\n")

# Query a deleted session, should see finish reason abort
prompt = "chunk 1"
response = requests.post(
    url + "/generate",
    json={
        "text": prompt,
        "session_id": session_id,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
        },
    },
)
print(response.json(), "\n")
