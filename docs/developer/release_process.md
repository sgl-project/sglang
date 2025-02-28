# PyPI Package Release Process

## Update the version in code
Update the package version in `python/pyproject.toml` and `python/sglang/__init__.py`.

## Upload the PyPI package

```
pip install build twine
```

```
cd python
bash upload_pypi.sh
```

## Make a release in GitHub
Make a new release https://github.com/sgl-project/sglang/releases/new.

## SHA-256 Hash Verification
SHA-256 hashes for package releases are automatically calculated locally during the build process and are available:

1. In the GitHub release artifacts as `hash.txt`
2. In the release notes (copy the hash information from the CI/CD output)

The hashes are calculated directly from the wheel files before they are uploaded to PyPI, ensuring that the hashes are available immediately after the release. If local hash calculation fails for any reason, the system will fall back to extracting hashes from PyPI.

Users can verify package integrity by comparing the SHA-256 hash of the downloaded package with the published hash:

```
sha256sum downloaded-package.whl
```

For manual calculation or extraction of hashes, you can use the provided scripts:

```
# Calculate hashes from local wheel files
./scripts/release/calculate_local_hashes.py sglang --dist-dir path/to/dist --output hash.txt

# Extract hashes from PyPI (fallback method)
./scripts/release/extract_pypi_hashes.py sglang --version 0.4.3.post2 --output hash.txt

# Convenience script that tries local calculation first, then falls back to PyPI extraction
./scripts/release/extract_local_hashes.sh sglang 0.4.3.post2
```
