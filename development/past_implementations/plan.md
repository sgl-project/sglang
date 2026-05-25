# As-Built Design Study: Three Double Sparsity Implementations

Act as a senior ML systems engineer reverse-engineering three existing implementations of double sparsity.

This is an **as-built design study** of implemented systems, not a proposal for a new design. Do not state inferred motivations or engineering decisions as facts. Clearly label claims as:

- **Observed**: directly supported by code, configuration, tests, docs, scripts, or benchmark artifacts.
- **Inferred**: strongly suggested by implementation structure or conventional design rationale, but not explicitly stated.
- **Unknown**: not establishable from the repositories or supplied references.

## Inputs

- Implementation A: `sglang/development/past_implementations/DoubleSparse`
- Implementation B: `sglang/development/past_implementations/sglang-last-with-double-sparsity`
- Implementation C: `sglang/development/past_implementations/Twilight`
- Reference paper or intended definition of double sparsity: `sglang/development/past_implementations/double_sparsity_paper_2408.07092.pdf`
- Output directory: `sglang/development/past_implementations/study`
- Intended audience: `ML engineer looking to a bring a minimal yet highly  performant implementation to sglang that is sglang-native, bringing it to modern models like deepseek v3.2 and GLM 5.1, with the goal of reusing sglang code where possible but performance is first class citizen.`

Treat the implementation repositories as read-only. Write only into the output directory. Every significant technical claim should cite evidence using repository-relative file paths and symbols or line numbers where practical.

Use the design-document principles from:

- https://www.industrialempathy.com/posts/design-docs-at-google/
- https://www.atlassian.com/work-management/knowledge-sharing/documentation/software-design-document

The documents are for understanding existing implementations. Optimize for technical clarity, traceability to code, comparability across implementations, and usefulness to an engineer who may need to modify or reproduce one of them later.

## Working Rules

1. Analyze each implementation using the same rubric so the comparison is fair.
2. Read source code, configurations, tests, scripts, examples, README files, benchmark results, and relevant build/dependency metadata.
3. Prefer entry-point-to-core-kernel tracing over superficial directory summaries.
4. For ML behavior, explicitly track tensors, sparse metadata, masks, indices, shapes, dtypes, devices, checkpoint state, configuration values, and lifecycle transitions whenever they can be established.
5. Do not claim performance, memory, numerical, or quality advantages without code-grounded reasoning or comparable benchmark evidence.
6. Do not conflate what the source paper intends with what each codebase actually implements.
7. When the repositories use different meanings of "double sparsity," state that plainly and define a common comparison vocabulary.
8. If subagents are available, use one parallel analysis subagent per implementation with this same rubric, then reconcile their findings yourself before writing cross-implementation conclusions.
9. Before producing final documents, identify gaps and contradictions in the evidence and either investigate them or label them as unresolved.

## Phase 1: Repository Survey

Inspect all three implementations consistently. Identify:

- Languages, frameworks, runtime dependencies, hardware assumptions, accelerators, sparse libraries, and custom kernels.
- Entry points for training, inference, evaluation, preprocessing, conversion, export, profiling, or benchmarking.
- Configuration surfaces and important hyperparameters.
- Code locations where sparsity is represented, initialized, computed, applied, updated, serialized, loaded, or measured.
- Existing tests, benchmarks, examples, documentation, execution commands, and reproducibility support.
- Whether each repository appears to implement the same meaning of "double sparsity."
- The most important files and symbols for understanding each implementation.

Create:

- `00-survey.md`
- `evidence-index.md`

### Required Content For `00-survey.md`

Include:

1. A concise explanation of the intended double sparsity concept based on the supplied reference, clearly separated from repository observations.
2. A repository map for each implementation, including entry points and core code paths.
3. A short side-by-side table comparing:
   - What is sparse: weights, activations, latent/code vectors, dictionary atoms, attention, gradients, routing choices, or other objects.
   - Sparsity structure: unstructured, structured, block, top-k, mask-based, indexed, learned, fixed, dynamic, or static.
   - When sparsity is applied: preprocessing, initialization, training, forward pass, inference, checkpoint conversion, export, or evaluation.
   - Whether sparse structure plausibly changes compute or storage behavior, and at what stage.
   - Evidence confidence and major open questions.
4. A proposed common vocabulary to use across the three documents.
5. Any initial finding that the implementations are not directly comparable without qualification.

### Required Content For `evidence-index.md`

For each implementation, provide an evidence table with:

- Topic or claim area.
- Key files and symbols.
- What can be directly concluded.
- What remains uncertain.
- Suggested next inspection step, where needed.

### Phase 1 Checkpoint

Stop after completing Phase 1. Summarize the preliminary interpretation of all three implementations and ask me to confirm or correct the meaning of double sparsity and the comparison scope before drafting the full design documents.

## Phase 2: Per-Implementation As-Built Design Documents

After I confirm the Phase 1 interpretation, create:

- `01-<implementation-a>.md`
- `02-<implementation-b>.md`
- `03-<implementation-c>.md`
- `04-comparison-and-recommendations.md`

Use the same structure for each implementation document.

### 1. Executive Summary

Explain:

- What this implementation does.
- How it realizes double sparsity.
- Its apparent use case and execution environment.
- What most distinguishes it from the other two implementations.
- The most important uncertainty still remaining after code inspection.

### 2. Context, Scope, Goals, And Non-Goals

Include:

- Context and scope supported by repository evidence.
- Problem addressed by the implementation.
- Supported workflows, such as training, inference, evaluation, conversion, or benchmarking.
- Apparent goals and non-goals, explicitly labeled **Inferred** unless stated in documentation or code comments.
- Out-of-scope functionality or unsupported scenarios visible in the codebase.

### 3. Algorithmic Interpretation

Explain:

- The mathematical or conceptual definition of double sparsity represented in this codebase.
- The mapping from the reference idea or paper to concrete code constructs.
- Which objects are sparse and how their sparse representation is stored or computed.
- Mask/index generation, selection rules, granularity, scheduling, updates, and lifecycle.
- Important invariants, assumptions, constraints, and edge cases.
- Any deviation from or ambiguity relative to the reference paper or intended idea.

Use equations or compact pseudocode where they substantially clarify the implementation, but ground each explanation in actual code references.

### 4. System Architecture

Describe:

- Major architectural layers and module boundaries.
- Runtime and library dependencies that materially shape the design.
- Model, data, checkpoint, configuration, kernel, evaluation, and benchmark interactions.
- External systems, datasets, model formats, or hardware assumptions where present.

Include a component inventory table with:

| Component | Responsibility | Inputs | Outputs / State | Important Interfaces | Evidence |
|---|---|---|---|---|---|

### 5. Data Flow

Trace the end-to-end path by which inputs become outputs through the double sparsity mechanism. Cover the applicable paths:

- Setup, conversion, or preprocessing.
- Training forward and backward flow.
- Inference forward flow.
- Evaluation or benchmarking flow.
- Serialization and checkpoint load/save flow.

For the core paths, identify where practical:

- Tensor shapes.
- Dtypes.
- Device placement.
- Dense versus sparse representations.
- Masks, indices, metadata, or auxiliary tensors.
- Materialization, gather/scatter, sparse kernel, pruning, top-k, routing, or reconstruction operations.
- Any difference between conceptual sparsity and actual computational sparsity.

### 6. Key Execution Sequences

Describe only sequences whose ordering materially clarifies the implementation, such as:

- A training step.
- An inference request or model forward call.
- A mask/index construction or update operation.
- A checkpoint conversion flow.
- An evaluation or performance benchmark flow.

For each chosen sequence, name the initiating entry point and trace through the major functions/modules.

### 7. Interfaces And Configuration

Document:

- Public or semi-public Python/C++/CUDA/API interfaces.
- CLI commands and scripts.
- Configuration files, flags, environment variables, hyperparameters, and defaults.
- Dataset or checkpoint formats.
- Build, installation, and hardware requirements.
- A minimal reproducible run path if it is discoverable from repository evidence.

Distinguish verified execution instructions from inferred execution instructions.

### 8. Engineering Decisions And Trade-Offs

Provide a decision/trade-off table:

| Observed Design Choice | Evidence | Likely Rationale | Benefits | Costs / Risks | Confidence |
|---|---|---|---|---|---|

Requirements:

- Treat the design choice as observed only if it is visible in the code.
- Label the rationale as **Inferred** unless stated directly by authors.
- Cover meaningful choices such as representation, kernel strategy, abstraction boundary, configuration style, mask lifecycle, checkpoint handling, batching, device movement, memory trade-offs, testing approach, and dependency choices.

### 9. Correctness, Testing, And Reproducibility

Analyze:

- Tests and validation mechanisms that exist.
- What correctness properties those tests validate.
- Numerical equivalence checks, reference implementations, golden outputs, assertions, logging, or debugging hooks.
- Reproducibility artifacts such as pinned dependencies, seeds, environment files, launch scripts, sample configs, checkpoints, and benchmark recipes.
- Missing tests or validation for the double sparsity mechanism.
- Failure modes that may be difficult to observe.
- Steps required to reproduce a representative run, if establishable.

### 10. Performance Characteristics

Explain, without making unsupported benchmark claims:

- Likely compute effects of the sparse representation and operations.
- Likely memory effects of masks, indices, sparse metadata, weights, activations, and checkpoints.
- GPU/CPU execution behavior and device transfers.
- Custom kernels, sparse operator support, fusion, caching, serialization, batching, compilation, quantization, or distributed execution behavior where applicable.
- What would need to be measured to fairly compare performance.
- Any benchmark evidence present in the repository, including its limitations.

### 11. Strengths, Weaknesses, And Improvements

Provide:

- Strengths of the current design.
- Weaknesses or constraints of the current design.
- Low-hanging improvements: localized changes with modest implementation risk and clear value.
- Larger redesign opportunities: material architectural changes that may provide greater benefit but need validation.
- Areas where additional documentation, tests, benchmarks, or observability would improve engineering confidence.

For each improvement, identify the motivating evidence and expected benefit.

### 12. Open Questions

List questions that code inspection cannot confidently answer, such as:

- Intended research hypothesis or target use case.
- Unrecorded benchmark methodology.
- Motivation for alternative implementation choices.
- Known limitations or failed experiments.
- Missing expected accuracy/performance targets.

Phrase these as questions for the implementation author or future experiment owner.

## Diagram Requirements

For each implementation, create:

1. A **system context or component architecture diagram** showing major modules, data/model/config/checkpoint inputs, kernels or runtimes, and produced artifacts.
2. A **data-flow diagram** showing the core double sparsity path, including sparse objects, masks/indices/metadata, relevant tensor transformations, and output.
3. A **sequence diagram** only when temporal order is informative, such as a training step, dynamic mask update, conversion pass, or inference execution path.

Diagram rules:

- Create Mermaid source files under `diagrams/*.mmd` as the editable source of truth.
- Render diagrams to `diagrams/*.svg` only if an SVG rendering tool; you can install dependencies solely for rendering.
- Embed rendered SVG diagrams into the Markdown design documents when rendering succeeds.
- If SVG rendering is unavailable, embed Mermaid code blocks in the Markdown documents and state how the `.mmd` source can later be rendered.
- Each diagram must have a short explanation beneath it identifying which elements are directly supported by code evidence and which connections are inferred.
- Do not create visually impressive but untraceable architecture. Every node should correspond to a real component, data object, external dependency, or explicitly marked conceptual boundary.

## Phase 3: Comparative Synthesis

In `04-comparison-and-recommendations.md`, include:

### 1. Comparison Summary

- Short description of all three designs.
- Whether they are direct implementations of the same algorithmic idea or variants under a broader label.
- Most important practical difference among them.

### 2. Common Vocabulary

Define consistent comparison terms, particularly:

- Sparse object.
- Sparse representation.
- Mask/index lifecycle.
- Sparse compute versus sparse storage.
- Static versus dynamic sparsity.
- Training-time versus inference-time sparsity.
- Algorithmic equivalence versus implementation equivalence.

### 3. Side-By-Side Comparison Table

Compare:

- Intended or apparent purpose.
- Sparse objects.
- Definition of double sparsity.
- Granularity and structure.
- Mask/index creation and update lifecycle.
- Training path.
- Inference path.
- Kernel/operator strategy.
- Device/hardware dependencies.
- Checkpoint and serialization approach.
- Configuration and extension points.
- Test and benchmark maturity.
- Reproducibility.
- Maintainability and readability.
- Expected compute/memory implications.
- Major strengths.
- Major risks.

### 4. Architectural Comparison

Explain:

- Where architectural approaches converge.
- Where they diverge.
- Which differences are algorithmically necessary versus implementation choices.
- Which parts could potentially be shared or standardized.

Include a comparative architecture or concept diagram if it helps clarify the relationship among the implementations.

### 5. Trade-Off Analysis

Discuss which implementation appears easiest to:

- Understand.
- Modify for new sparse variants.
- Validate for correctness.
- Benchmark fairly.
- Optimize for runtime performance.
- Port to different hardware or model architectures.
- Use as a research baseline.
- Maintain in production-oriented engineering work.

All rankings or judgments must identify evidence and uncertainty.

### 6. Fair Experimental Comparison Plan

Propose a benchmark and validation plan that could empirically compare the implementations, including:

- Preconditions required to ensure they are meaningfully comparable.
- Common model/dataset/input setup where possible.
- Correctness or equivalence checks.
- Accuracy or quality metrics.
- Wall-clock, throughput, latency, peak memory, checkpoint size, preprocessing time, compilation/warmup, and sparse-metadata overhead metrics where relevant.
- Hardware, dependency, seeds, and configuration controls.
- How to handle implementations that cannot run under exactly identical conditions.

Do not execute expensive experiments unless I separately approve them.

### 7. Recommendations

Provide:

- Immediate low-hanging improvements for each implementation.
- Cross-cutting documentation and testing improvements.
- Suggested next experiments.
- A recommendation for which implementation to study or extend first, based on my stated goal and the available evidence.

## Document Quality Requirements

- Be concrete and code-grounded, not generic.
- Separate observed behavior from inferred intent throughout.
- Prefer compact tables and explanatory diagrams over repetitive prose.
- Surface ambiguity rather than hiding it.
- Make each document useful independently, while reserving direct rankings and comparison for `04-comparison-and-recommendations.md`.
- Cite code evidence throughout, using file paths and functions/classes/symbols or line numbers where practical.
- Include a brief evidence and confidence summary at the end of every document.
- Keep diagrams readable and technically meaningful.
- Avoid recommending refactors merely for style; tie improvements to comprehension, correctness, reproducibility, runtime behavior, extensibility, or maintainability.

## Final Review

After producing the documents:

1. Re-read all four documents for consistency in terminology and claims.
2. Check that every important comparative assertion has evidence or is labeled inferred.
3. Check that each implementation has an architecture view and data-flow view.
4. Use a fresh subagent, if available, as a reader to answer:
   - How does each implementation realize double sparsity?
   - Are they implementing the same idea?
   - Where do masks/indices/sparse data enter the execution path?
   - What are the major trade-offs?
   - What should an engineer investigate or improve next?
5. Repair document gaps identified by that reader test.
6. Return a short completion summary listing generated files, diagrams rendered or left as Mermaid, key uncertainties, and the best order in which I should read the documents.
