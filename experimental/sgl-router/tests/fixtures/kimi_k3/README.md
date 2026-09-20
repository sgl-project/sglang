Tiny synthetic Kimi vocabulary and golden token IDs from the Python reference
`moonshotai/Kimi-K3` revision `a590ce09`. These exercise native rendering and
segmented BPE without a model download. The `no_effort` case records a known
Dynamo 5.1.2 mismatch: the router must fall back to engine-side tokenization
instead of rewriting the native prompt.
