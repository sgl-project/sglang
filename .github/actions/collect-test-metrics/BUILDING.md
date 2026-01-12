# Building the Action

This action requires Node.js and must be bundled before use.

## Build Steps

```bash
cd .github/actions/collect-test-metrics
npm ci
npm run build
```

This generates `dist/index.js` which must be committed:

```bash
git add dist/ package-lock.json
git commit -m "Build action bundle"
```

## Why Bundle?

GitHub Actions runners don't have our dependencies installed. Bundling with `@vercel/ncc` creates a single self-contained file including all `@actions/*` packages.
