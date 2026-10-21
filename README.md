# CP BCG package-move proof for SGLang #43182

This proof reproduces commit `a3d9fac64feae6e8cde40a99b4835ccebe0a0d44` from `68c42ba6e389c1d2b6387615f227c5720ff635cb`: two file renames, an empty package initializer, import repaths, and repository formatting. The later strategy refactor is reviewed and tested separately.

From a SGLang checkout with Git, Python 3, and pre-commit installed:

```bash
git fetch origin pull/43182/head codex/pr43182-bcg-move-proof
mkdir -p /tmp/pr43182-move-proof
git archive origin/codex/pr43182-bcg-move-proof | tar -x -C /tmp/pr43182-move-proof
python3 /tmp/pr43182-move-proof/a3d9fac64feae6e8cde40a99b4835ccebe0a0d44.py
```

The verifier runs the repository formatters and compares the reproduced tree byte-for-byte against the target commit. See `output.log` for the successful run. It leaves a temporary worktree and prints cleanup commands.

## Package-level support predicate

Commit `b709e9b5ef7d38913653048f27960e65617da04b` moves `supports_prefill_cp_bcg` into the package initializer and restores its original import path. Its function body is unchanged in this move commit; removing the now-redundant self-import and the review's test changes are separate.

After downloading this branch as above, run from a SGLang checkout:

```bash
python3 /tmp/pr43182-move-proof/b709e9b5e.py
```

The recipe uses the relocation primitives, the package import path (rather than importing `.__init__`), and an explicit blank-line normalization. See `support-api-output.log` for its byte-for-byte PASS result.
