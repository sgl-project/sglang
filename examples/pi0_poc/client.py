import json

import requests


BASE_URL = "http://127.0.0.1:30000"


def main():
    payload = {
        "input_ids": [1, 2, 3],
        "sampling_params": {
            "max_new_tokens": 1,
            "temperature": 0,
            "custom_params": {
                "state": [0.0, 0.1, 0.2, 0.3, 0.4],
                "num_inference_steps": 8,
                "seed": 7,
            },
        },
    }
    response = requests.post(f"{BASE_URL}/generate", json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    actions = data["meta_info"]["actions"][0]
    print(json.dumps({"shape": [len(actions), len(actions[0])], "actions": actions}))


if __name__ == "__main__":
    main()
