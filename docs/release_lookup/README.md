# SGLang Release Lookup Tool

This tool allows users to find the earliest release that contains a specific PR or commit.
It runs entirely in the browser using a static JSON index generated from the git history.

## Usage

1. **Generate the Index**:
   Run the Python script to generate the `release_index.json` file from your local git repository.

   ```bash
   python3 generate_index.py --output release_index.json
   ```

   This script:
   - Finds all tags matching `v*` and `gateway-v*`.
   - Sorts them by creation date.
   - Traverses the history to find which release first introduced each commit and PR.
   - Extracts PR numbers from commit messages.

2. **Open the Tool**:
   Open `index.html` in your browser.

   ```bash
   # You can open it directly if your browser supports local file fetch (Firefox usually does),
   # or serve it locally:
   python3 -m http.server
   # Then go to http://localhost:8000/index.html
   ```

## Files

- `index.html`: The UI for the lookup tool.
- `generate_index.py`: Script to build the index.
- `release_index.json`: The index file used by the UI.

## Logic

The tool determines the "earliest release" based on the tag creation date. It traverses tags from oldest to newest. Any commit reachable from a tag (that wasn't reachable from a previous tag) is assigned to that release.
