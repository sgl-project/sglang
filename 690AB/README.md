# Development Flow

## Branches

- `test-*`: active development and quick GPU testing
- `development`: merge validated `test-*` branches here, ideally via PRs
- `main`: final/stable branch for official benchmarks and paper results

## Workflow

1. Develop locally on your Mac in your own `test-*` branch
2. Push changes to the same remote `test-*` branch
3. In Colab, change the first setup cell:

   ```python
   BRANCH = "test-v0.5.6"
   ```

4. Re-run setup cells to sync the latest branch state
5. Validate on GPU in Colab
6. Merge validated `test-*` branches into `development`
7. Merge `development` into `main` for final benchmarks

### Why
SGLang may not be practical to fully test on Mac / Apple Silicon, so:

- **Mac** = local development
- **Colab** = GPU validation
- **main** = final benchmark runs

### Quick Colab Edits
Small fixes can also be made directly in Colab.

**Flow:**
1. Edit files in Colab
2. Add a new bash cell
3. Commit and push back to your own `test-*` branch

**Example:**
```bash
%%bash
cd /content/sglang
git add .
git commit -m "small colab fix"
git push origin test-v0.5.6
```

> **Note:** Use this only for quick fixes.

### Rules
- Do active work only in `test-*`
- Push only to your own `test-*` branch
- Do not push experimental work to `development` or `main`
- Merge tested work into `development`
- Run official/final benchmarks only from `main`
