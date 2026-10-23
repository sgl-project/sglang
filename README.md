# CP attention token-bound extraction proof

This proof reproduces only the mechanical extraction commit
`9cbd39a5ca65fcf7d4211d66459c296b84a7fadd` from
`5f5529241c95258a3572773278c0e75983aa29fd`.

Keep the commit-named Python script and
`mechanical_refactor_reproduction_utils.py` in the same directory. From a
checkout of sgl-project/sglang containing both commits, with the repository's
pre-commit prerequisites installed, run:

```bash
python3 /path/to/proof/9cbd39a5ca65fcf7d4211d66459c296b84a7fadd.py
```

Observed result:

```text
PASS: reproduces the commit byte-for-byte.
```

The preceding preparation, subsequent review fixes, and subsequent lazy-import
fix are separate commits. This proof does not certify their behavior; those
changes are checked by focused tests and the end-to-end accuracy campaign.
