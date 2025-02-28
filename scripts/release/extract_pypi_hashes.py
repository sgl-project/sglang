import argparse
import requests
import json
import time
import sys

def wait_for_package_on_pypi(package_name, version, max_wait_time=300, poll_interval=5):
    """
    Wait for a specific package version to be available on PyPI.
    
    Args:
        package_name (str): Name of the package
        version (str): Version of the package to wait for
        max_wait_time (int): Maximum time to wait in seconds
        poll_interval (int): Time between polling attempts in seconds
        
    Returns:
        bool: True if the package is available, False otherwise
    """
    url = f'https://pypi.org/pypi/{package_name}/{version}/json'
    start_time = time.time()
    
    print(f"Waiting for {package_name} {version} to be available on PyPI...")
    
    while time.time() - start_time < max_wait_time:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print(f"Package {package_name} {version} is now available on PyPI")
                return True
                
            elapsed = time.time() - start_time
            remaining = max_wait_time - elapsed
            print(f"Package not yet available. Retrying in {poll_interval} seconds... (Timeout in {int(remaining)} seconds)")
            time.sleep(poll_interval)
            
        except requests.exceptions.RequestException as e:
            print(f"Error checking PyPI: {e}")
            time.sleep(poll_interval)
    
    print(f"Timed out waiting for {package_name} {version} to be available on PyPI")
    return False

def extract_hashes(package_name, version=None, max_retries=10, retry_delay=5, wait_for_version=False, max_wait_time=300):
    """
    Extract SHA-256 hashes for a package from PyPI.
    
    Args:
        package_name (str): Name of the package
        version (str, optional): Specific version to extract hashes for.
                                If None, the latest version will be used.
        max_retries (int): Maximum number of retries if the package is not found
        retry_delay (int): Delay between retries in seconds
        wait_for_version (bool): Whether to wait for the version to be available on PyPI
        max_wait_time (int): Maximum time to wait for the version in seconds
        
    Returns:
        dict: Dictionary containing the hashes for each file
    """
    # If we need to wait for a specific version to be available
    if version and wait_for_version:
        if not wait_for_package_on_pypi(package_name, version, max_wait_time, retry_delay):
            return {}
    
    # PyPI API URL
    url = f'https://pypi.org/pypi/{package_name}/json'
    if version:
        url = f'https://pypi.org/pypi/{package_name}/{version}/json'
    
    for attempt in range(max_retries):
        try:
            # Make the request to PyPI
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse the JSON response
            data = response.json()
            
            if version is None:
                # Get the latest version if no specific version was provided
                version = data.get('info', {}).get('version')
                print(f"Latest version: {version}")
            
            # Extract the URLs and hashes for the package files
            releases = data.get('urls', [])
            
            if not releases:
                print(f"No releases found for {package_name} {version}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                return {}
            
            # Create a dictionary to store the hashes
            hashes = {}
            
            for release in releases:
                filename = release.get('filename')
                packagetype = release.get('packagetype')
                
                # Extract the SHA-256 hash
                digests = release.get('digests', {})
                sha256 = digests.get('sha256')
                
                if sha256:
                    hashes[filename] = {
                        'type': packagetype,
                        'sha256': sha256
                    }
            
            return hashes
            
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                return {}
        except json.JSONDecodeError:
            print("Error: Invalid JSON response")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                return {}
    
    return {}

def write_hash_file(hashes, output_file):
    """
    Write hashes to a file.
    
    Args:
        hashes (dict): Dictionary containing the hashes for each file
        output_file (str): Path to the output file
    """
    with open(output_file, "w") as f:
        for filename, info in sorted(hashes.items()):
            f.write(f"{info['sha256']} {filename}\n")
    
    print(f"Hashes written to {output_file}")

def format_for_changelog(package_name, version, hashes):
    """
    Format hashes for inclusion in the changelog.
    
    Args:
        package_name (str): Name of the package
        version (str): Version of the package
        hashes (dict): Dictionary containing the hashes for each file
        
    Returns:
        str: Formatted string for the changelog
    """
    if not hashes:
        return f"No hashes found for {package_name} {version}"
    
    lines = [f"## SHA-256 Hashes for {package_name} {version}"]
    lines.append("```")
    for filename, info in sorted(hashes.items()):
        lines.append(f"{info['sha256']} {filename}")
    lines.append("```")
    
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Extract SHA-256 hashes for a package from PyPI")
    parser.add_argument("package_name", help="Name of the package")
    parser.add_argument("--version", help="Specific version to extract hashes for")
    parser.add_argument("--output", default="hash.txt", help="Output file for the hashes")
    parser.add_argument("--format", choices=["file", "changelog", "both"], default="both",
                       help="Output format (file, changelog, or both)")
    parser.add_argument("--retries", type=int, default=10, help="Maximum number of retries")
    parser.add_argument("--delay", type=int, default=5, help="Delay between retries in seconds")
    parser.add_argument("--wait", action="store_true", help="Wait for the package to be available on PyPI")
    parser.add_argument("--max-wait-time", type=int, default=300, 
                       help="Maximum time to wait for the package to be available on PyPI in seconds")
    
    args = parser.parse_args()
    
    hashes = extract_hashes(
        args.package_name, 
        args.version, 
        args.retries, 
        args.delay, 
        wait_for_version=args.wait,
        max_wait_time=args.max_wait_time
    )
    
    if not hashes:
        print(f"Failed to extract hashes for {args.package_name} {args.version or 'latest'}")
        sys.exit(1)
    
    if args.format in ["file", "both"]:
        write_hash_file(hashes, args.output)
    
    if args.format in ["changelog", "both"]:
        changelog_text = format_for_changelog(args.package_name, args.version or "latest", hashes)
        print("\n" + changelog_text)

if __name__ == "__main__":
    main()
