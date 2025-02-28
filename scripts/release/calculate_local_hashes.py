#!/usr/bin/env python3
"""
Calculate SHA-256 hashes for wheel files and other distribution files.
This script can be used as an alternative to extracting hashes from PyPI.
"""

import os
import sys
import hashlib
import glob
import argparse
import json
from typing import Dict, Optional

def calculate_file_hash(file_path: str) -> str:
    """
    Calculate SHA-256 hash for a file.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: SHA-256 hash of the file
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read and update hash in chunks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def hash_dist_files(dist_dir: str, package_name: Optional[str] = None, version: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """
    Hash all files in the dist directory.
    
    Args:
        dist_dir (str): Path to the dist directory
        package_name (str, optional): Name of the package
        version (str, optional): Version of the package
        
    Returns:
        Dict[str, Dict[str, str]]: Dictionary containing the hashes for each file
    """
    if not os.path.exists(dist_dir):
        print(f"Error: Directory {dist_dir} does not exist")
        return {}
    
    hashes = {}
    for file_path in glob.glob(os.path.join(dist_dir, "*")):
        if os.path.isfile(file_path):
            file_name = os.path.basename(file_path)
            file_hash = calculate_file_hash(file_path)
            
            # Determine package type based on file extension
            if file_name.endswith('.whl'):
                package_type = 'bdist_wheel'
            elif file_name.endswith('.tar.gz'):
                package_type = 'sdist'
            else:
                package_type = 'unknown'
            
            hashes[file_name] = {
                'type': package_type,
                'sha256': file_hash
            }
            
            print(f"{file_hash} {file_name}")
    
    return hashes

def write_hash_file(hashes: Dict[str, Dict[str, str]], output_file: str) -> None:
    """
    Write hashes to a file.
    
    Args:
        hashes (Dict[str, Dict[str, str]]): Dictionary containing the hashes for each file
        output_file (str): Path to the output file
    """
    with open(output_file, "w") as f:
        for filename, info in sorted(hashes.items()):
            f.write(f"{info['sha256']} {filename}\n")
    
    print(f"Hashes written to {output_file}")

def format_for_changelog(package_name: str, version: str, hashes: Dict[str, Dict[str, str]]) -> str:
    """
    Format hashes for inclusion in the changelog.
    
    Args:
        package_name (str): Name of the package
        version (str): Version of the package
        hashes (Dict[str, Dict[str, str]]): Dictionary containing the hashes for each file
        
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
    parser = argparse.ArgumentParser(description="Calculate SHA-256 hashes for files in the dist directory")
    parser.add_argument("package_name", help="Name of the package")
    parser.add_argument("--version", help="Version of the package")
    parser.add_argument("--dist-dir", default="dist", help="Path to the dist directory")
    parser.add_argument("--output", default="hash.txt", help="Output file for the hashes")
    parser.add_argument("--format", choices=["file", "changelog", "both"], default="both",
                       help="Output format (file, changelog, or both)")
    
    args = parser.parse_args()
    
    hashes = hash_dist_files(args.dist_dir, args.package_name, args.version)
    
    if not hashes:
        print(f"No files found in {args.dist_dir}")
        sys.exit(1)
    
    if args.format in ["file", "both"]:
        write_hash_file(hashes, args.output)
    
    if args.format in ["changelog", "both"]:
        changelog_text = format_for_changelog(args.package_name, args.version or "unknown", hashes)
        print("\n" + changelog_text)

if __name__ == "__main__":
    main()
