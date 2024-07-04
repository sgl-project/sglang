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
