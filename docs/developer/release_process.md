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
SHA-256 hashes for package releases are automatically extracted from PyPI as part of the CI/CD pipeline and are available:

1. In the GitHub release artifacts as `hash.txt`
2. In the release notes (copy the hash information from the CI/CD output)

Users can verify package integrity by comparing the SHA-256 hash of the downloaded package with the published hash:

```
sha256sum downloaded-package.whl
```

For manual extraction of hashes, you can use the provided script:

```
./scripts/release/extract_local_hashes.sh sglang 0.4.3.post2
```
