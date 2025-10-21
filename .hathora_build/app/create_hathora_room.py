#!/usr/bin/env python3
import os
import sys
import json
import requests
import argparse

def main():
    parser = argparse.ArgumentParser(description="Create a Hathora room with master IP")
    parser.add_argument("master_ip", help="Master node IP address")
    args = parser.parse_args()

    hathora_token = os.getenv("HATHORA_TOKEN")
    kimi_b_app_id = os.getenv("KIMI_B_APP_ID")
    if not hathora_token or not kimi_b_app_id:
        sys.exit("HATHORA_TOKEN and KIMI_B_APP_ID must be set")

    api_host = "hathora.io" if "hathora.io" in os.getenv("HATHORA_HOSTNAME", "") else "hathora.dev"
    room_config = json.dumps({"master_ip": args.master_ip})

    response = requests.post(
        f"https://api.{api_host}/rooms/v2/{kimi_b_app_id}/create",
        headers={"Authorization": f"Bearer {hathora_token}", "Content-Type": "application/json"},
        json={"roomConfig": room_config, "region": "Washington_DC"},
        timeout=10,
    )

    if response.status_code == 201:
        print(f"Created room: {response.json().get('roomId')}")
    else:
        sys.exit(f"Failed to create room: {response.status_code} - {response.text}")

if __name__ == "__main__":
    main()
