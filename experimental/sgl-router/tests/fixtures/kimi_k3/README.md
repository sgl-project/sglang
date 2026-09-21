Tiny synthetic Kimi vocabulary for testing Dynamo's native formatter and
segmented tokenizer without a model download. The added tokens include Kimi
protocol markers; ordinary text that spells those markers must not become
structural tokens. Adapter tests compare directly with the pinned Dynamo crates,
including null thinking effort, long text, tools, and response formats.
