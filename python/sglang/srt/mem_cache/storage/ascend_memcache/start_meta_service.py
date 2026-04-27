import json
import argparse
import os
from memcache_hybrid import MetaService, MetaConfig

def launch_with_json(json_path):
    """
    Load configuration from a JSON file and start the MetaService.
    """
    # 1. Load JSON file content
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{json_path}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{json_path}'.")
        return

    # 2. Instantiate the configuration object
    config = MetaConfig()

    # 3. Dynamically inject JSON key-value pairs into the config object
    for key, value in data.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Configuration key '{key}' is not a valid MetaConfig attribute. Skipping...")

    # 4. Launch the service
    print(f"Successfully loaded {json_path}. Starting service...")
    
    # Initialize service with the populated config object
    MetaService.setup(config)
    
    # Start the main service loop/process
    MetaService.main()

if __name__ == "__main__":
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Set default config path to the same directory as the script
    default_path = os.path.join(script_dir, 'metaservice_config.json')

    parser = argparse.ArgumentParser(description="Launch MetaService using a JSON configuration file.")
    
    # Added the config_path argument
    parser.add_argument(
        '--config_path', 
        type=str, 
        default=default_path,
        help=f"Path to the configuration JSON file (default: {default_path})"
    )

    args = parser.parse_args()

    # Launch with the provided or default path
    launch_with_json(args.config_path)