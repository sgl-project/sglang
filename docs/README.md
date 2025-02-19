# SGLang Documentation
We recommend new contributors start from writing documentation, which helps you quickly understand SGLang codebase. Most documentation files are located under the `docs/` folder. We prefer **Jupyter Notebooks** over Markdown so that all examples can be executed and validated by our docs CI pipeline.

## Docs Workflow

### Install Dependency

```bash
pip install -r requirements.txt
```

### Update Documentation

Update your Jupyter notebooks in the appropriate subdirectories under `docs/`. If you add new files, remember to update `index.rst` (or relevant `.rst` files) accordingly.

- **`pre-commit run --all-files`** manually runs all configured checks, applying fixes if possible. If it fails the first time, re-run it to ensure lint errors are fully resolved. Make sure your code passes all checks **before** creating a Pull Request.
- **Do not commit** directly to the `main` branch. Always create a new branch (e.g., `feature/my-new-feature`), push your changes, and open a PR from that branch.

```bash
# 1) Compile all Jupyter notebooks
make compile

# 2) Generate static HTML
make html

# 3) Preview documentation locally
# Open your browser at the displayed port to view the docs
bash serve.sh

# 4) Clean notebook outputs
# nbstripout removes notebook outputs so your PR stays clean
pip install nbstripout
find . -name '*.ipynb' -exec nbstripout {} \;

# 5) Pre-commit checks and create a PR
# After these checks pass, push your changes and open a PR on your branch
pre-commit run --all-files
```
