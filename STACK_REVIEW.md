# RuntimeContext configuration-namespace migration — full-stack review

**Review/CI vehicle — DO NOT MERGE.** The individual stacked PRs merge in order;
this PR carries the combined diff so full CI runs against the end state.

Introduces a structured `RuntimeContext` configuration API. Resolved
configuration is read through domain-scoped namespaces (`get_model()`,
`get_serving()`, `get_exec()`, `get_schedule()`, `get_memory()`, …) instead of
the flat `ServerArgs` global; `ServerArgs` becomes the read-only resolved-config
record, and a single audited post-resolution mutation entry writes the namespace
bags. Incremental and behavior-preserving.

Stacked members (merge in order):
1. annotate ServerArgs fields with their runtime-config namespace
2. resolved-config namespace bags + accessors
3. make ServerArgs read-only with a single audited mutation entry
4. route runtime config adjustments through the namespace bags
5. record the publishing process role
6. read resolved config via namespace accessors
7. load-time declarations write the config bags
8. read parallel config leaves via get_parallel()
9. publish resolved config in unit fixtures for the namespace API
