---
name: review-pr
description: Review a pull request against the cookbook contribution checklist. Run with /review-pr <PR number>.
---

# Review PR

Given a PR number, fetch the diff and run the full checklist. Report findings clearly.

## Usage

```
/review-pr <PR number>
```

## Checklist

### 1. File hygiene
- No stray files: `package-lock.json`, `.claude/settings.local.json`, unrelated `.gitignore` changes
- Only expected files changed: `.mdx` doc + `docs/src/snippets/.../deployment.jsx` + `docs/docs.json`

### 2. Launch command
- Must use `sglang serve` — flag any use of `python -m sglang.launch_server` (deprecated, see issue #33)

### 3. Hardware specs (AMD)
- MI300X = 192GB HBM3 per card
- MI325X = 256GB HBM3 per card
- MI355X = 288GB HBM3E per card
- TP degree must be consistent with model size / GPU memory per card

### 4. JSX deployment snippet
- Must be a functional component with a named export (`export const XxxDeployment`)
- Uses inline styles and dark mode detection (`useState` + `useEffect` pattern)
- Options and generated command match the doc
- AMD hardware options use correct card names and memory values

### 5. Port consistency
- All examples should use port 30000 (not 8000 or other non-standard ports)

### 6. Quantization consistency
- FP4 quantization should only appear for NVIDIA Blackwell GPUs (B200), never for AMD
- BF16 and FP8 are supported on both NVIDIA and AMD platforms

### 7. Duplicate PR check
- Check if another open PR targets the same model — flag conflicts to avoid wasted effort

### 8. Navigation entry
- If the PR adds a new model, verify `docs/docs.json` is updated with the new entry

### 9. Links
- HuggingFace model URLs are valid and point to the right model
- No references to `sgl-project-dev` org (use `sgl-project`)

### 10. Build check
Run after reviewing:
```bash
cd docs && npx mintlify build
```
Fix any import errors before approving.

## Output format

For each file changed, give a one-line verdict:
- PASS
- ISSUE: <what's wrong>
- BLOCK: <must fix before merge>

End with overall: **APPROVE** / **REQUEST CHANGES** / **BLOCKED**
