#!/bin/bash
# Example usage of SGLang CI Analyzer

# IMPORTANT: Get your GitHub token from https://github.com/settings/tokens
# Make sure to select 'repo' and 'workflow' permissions!

# Basic usage - analyze last 100 runs
python3 ci_analyzer.py --token YOUR_GITHUB_TOKEN

# Analyze last 1000 runs
python3 ci_analyzer.py --token YOUR_GITHUB_TOKEN --limit 1000

# Custom output file
python3 ci_analyzer.py --token YOUR_GITHUB_TOKEN --limit 500 --output my_analysis.json
