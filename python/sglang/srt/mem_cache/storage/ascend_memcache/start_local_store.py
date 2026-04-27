import json
import argparse
import os
from memcache_hybrid import DistributedObjectStore, LocalConfig

def launch_local_store(json_path):
    """
    Load configuration from a JSON file and start the DistributedObjectStore.
    """
    # 1. Load JSON file content
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    # 2. Instantiate the configuration object
    config = LocalConfig()

    # 3. Dynamically inject JSON key-value pairs into the config object
    # This ensures all necessary parameters are inside the config object
    for key, value in data.items():
        if hasattr(config, key):
            setattr(config, key, value)
            print(f"Set config.{key} = {value}")
        else:
            # If your LocalConfig doesn't have device_id as an attribute, 
            # it might be handled via environment variables or other config fields.
            print(f"Note: '{key}' is not a direct attribute of LocalConfig. Skipping attribute injection...")

    # 4. Initialize the store
    print(f"Successfully loaded {json_path}. Initializing DistributedObjectStore...")
    store = DistributedObjectStore()

    # 5. Correct Setup: Only pass the config object
    # Based on the TypeError, this is the ONLY supported signature:
    # (self: _pymmc.DistributedObjectStore, config: _pymmc.LocalConfig) -> int
    ret = store.setup(config)
    
    if ret == 0:
        print(f"DistributedObjectStore initialized successfully! (Return code: {ret})")
    else:
        print(f"DistributedObjectStore failed to start. Return code: {ret}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(script_dir, 'localservice_config.json')

    parser = argparse.ArgumentParser(description="Launch Local DistributedObjectStore via JSON.")
    parser.add_argument(
        '--config_path', 
        type=str, 
        default=default_path,
        help=f"Path to the local config JSON file (default: {default_path})"
    )

    args = parser.parse_args()
    launch_local_store(args.config_path)