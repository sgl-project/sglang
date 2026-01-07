# Circular Bias Detection in SGLang

SGLang includes built-in support for detecting circular reasoning bias in LLM evaluation workflows, based on the statistical framework from:

> Zhang et al. (2024). "Circular Reasoning Bias Detection in AI Algorithm Evaluation." *Journal of Open Source Software* (under review).

## What is Circular Reasoning Bias?

Circular reasoning occurs when evaluation constraints (e.g., `temperature`, `max_tokens`) are iteratively adjusted based on model performance, creating a feedback loop that inflates performance metrics.

### Example of Circular Bias

```
Round 1: temperature=0.5 → performance=0.7 → "too low, increase temp"
Round 2: temperature=0.7 → performance=0.8 → "better, increase more"
Round 3: temperature=0.9 → performance=0.85 → "best performance!"
```

**Problem**: The "best" performance might be an artifact of constraint optimization, not true model capability.

## Statistical Indicators

The framework uses three complementary indicators:

### 1. PSI (Parameter Stability Index)
- **Range**: 0 to ∞ (lower is better)
- **Measures**: How much performance varies across evaluation periods
- **Threshold**: Typically 0.1
- **Interpretation**: High PSI suggests unstable evaluation

### 2. CCS (Constraint Consistency Score)
- **Range**: 0 to 1 (higher is better)
- **Measures**: How consistent evaluation constraints are
- **Threshold**: Typically 0.8
- **Interpretation**: Low CCS suggests iterative tuning

### 3. ρ_PC (Performance-Constraint Correlation)
- **Range**: -1 to 1
- **Measures**: Correlation between performance and constraints
- **Threshold**: Typically 0.3 (absolute value)
- **Interpretation**: High |ρ_PC| suggests circular relationship

### Bias Detection

Bias is flagged when **2 or more** indicators exceed their thresholds (majority voting).

---

## Quick Start

### Installation

```bash
pip install circular-bias-detection
```

### Basic Usage

```python
from sglang.lang.bias_audit import BiasAuditor

# Initialize auditor
auditor = BiasAuditor()

# Record generations
for i in range(15):
    response = model.generate(prompt, temperature=0.7)
    auditor.record_generation(
        output=response,
        constraints={'temperature': 0.7, 'max_tokens': 100}
    )

# Perform audit
result = auditor.audit(time_periods=5)

# Check results
if result.overall_bias:
    print(f"⚠️  Bias detected! Confidence: {result.confidence:.0%}")
    print(result.summary())
```

---

## API Reference

### BiasAuditor

Main class for bias detection.

```python
BiasAuditor(
    psi_threshold: float = 0.1,
    ccs_threshold: float = 0.8,
    rho_pc_threshold: float = 0.3,
    auto_score: bool = True
)
```

**Parameters:**
- `psi_threshold`: Threshold for PSI (default: 0.1)
- `ccs_threshold`: Threshold for CCS (default: 0.8)
- `rho_pc_threshold`: Threshold for |ρ_PC| (default: 0.3)
- `auto_score`: Automatically compute performance scores (default: True)

**Methods:**

#### `record_generation(output, constraints, performance_score=None, metadata=None)`

Record a single generation for later auditing.

```python
auditor.record_generation(
    output="Generated text...",
    constraints={'temperature': 0.7, 'max_tokens': 100},
    performance_score=0.85,  # Optional
    metadata={'prompt_id': 123}  # Optional
)
```

#### `record_batch(outputs, constraints, performance_scores=None)`

Record multiple generations at once.

```python
auditor.record_batch(
    outputs=["Text 1", "Text 2", "Text 3"],
    constraints=[
        {'temperature': 0.7},
        {'temperature': 0.7},
        {'temperature': 0.7}
    ],
    performance_scores=[0.8, 0.85, 0.82]  # Optional
)
```

#### `audit(time_periods=None, min_generations=6)`

Perform bias audit on recorded generations.

```python
result = auditor.audit(
    time_periods=5,  # Divide history into 5 periods
    min_generations=6  # Minimum required generations
)
```

**Returns:** `BiasAuditResult` with all indicators and bias detection.

#### `clear_history()`

Clear all recorded generations.

```python
auditor.clear_history()
```

#### `get_statistics()`

Get auditor statistics.

```python
stats = auditor.get_statistics()
# Returns: {
#     'total_generations': 20,
#     'total_audits': 2,
#     'history_size': 20,
#     'thresholds': {...}
# }
```

---

### BiasAuditResult

Result object from an audit operation.

**Attributes:**
- `psi_score`: PSI value
- `ccs_score`: CCS value
- `rho_pc_score`: ρ_PC value
- `overall_bias`: True if bias detected
- `confidence`: Confidence level (0 to 1)
- `bias_votes`: Number of indicators flagged (0 to 3)
- `psi_bias`, `ccs_bias`, `rho_pc_bias`: Individual indicator flags
- `metadata`: Additional audit metadata

**Methods:**

```python
result.to_dict()     # Convert to dictionary
result.to_json()     # Convert to JSON string
result.summary()     # Human-readable summary
```

---

## Usage Patterns

### Pattern 1: Real-time Monitoring

Monitor bias during evaluation:

```python
auditor = BiasAuditor()

for batch in evaluation_batches:
    for prompt in batch:
        response = model.generate(prompt, **constraints)
        auditor.record_generation(response, constraints)
    
    # Check after each batch
    if len(auditor.history) >= 10:
        result = auditor.audit()
        if result.overall_bias:
            logging.warning(f"Bias detected: {result.summary()}")
```

### Pattern 2: Post-hoc Analysis

Analyze completed evaluations:

```python
# Load evaluation logs
evaluations = load_evaluation_logs()

auditor = BiasAuditor()
for eval in evaluations:
    auditor.record_generation(
        output=eval['response'],
        constraints=eval['constraints'],
        performance_score=eval['score']
    )

result = auditor.audit()
generate_report(result)
```

### Pattern 3: Comparative Analysis

Compare different evaluation strategies:

```python
# Strategy A: Fixed constraints
auditor_a = BiasAuditor()
run_evaluation(auditor_a, fixed_constraints=True)
result_a = auditor_a.audit()

# Strategy B: Adaptive constraints
auditor_b = BiasAuditor()
run_evaluation(auditor_b, fixed_constraints=False)
result_b = auditor_b.audit()

# Compare
if result_b.overall_bias and not result_a.overall_bias:
    print("Strategy A is less biased!")
```

### Pattern 4: Continuous Integration

Integrate into CI/CD:

```python
def test_evaluation_bias():
    """Test that evaluation doesn't have circular bias."""
    auditor = BiasAuditor()
    
    # Run standard evaluation
    run_standard_evaluation(auditor)
    
    result = auditor.audit()
    
    # Fail if bias detected with high confidence
    assert not (result.overall_bias and result.confidence > 0.7), \
        f"Evaluation shows circular bias: {result.summary()}"
```

---

## Integration with SGLang Runtime

### Option 1: Manual Integration

```python
import sglang as sgl
from sglang.lang.bias_audit import BiasAuditor

# Initialize
auditor = BiasAuditor()
runtime = sgl.Runtime(model_path="meta-llama/Llama-2-7b-hf")

@sgl.function
def evaluate_sentiment(s, text, temperature):
    s += sgl.user(f"Analyze: {text}")
    s += sgl.gen("response", temperature=temperature, max_tokens=100)

# Run with auditing
for temp in [0.5, 0.6, 0.7, 0.8, 0.9]:
    for prompt in prompts:
        state = evaluate_sentiment.run(
            text=prompt,
            temperature=temp,
            runtime=runtime
        )
        
        # Record for audit
        auditor.record_generation(
            output=state["response"],
            constraints={'temperature': temp}
        )

# Audit
result = auditor.audit()
```

### Option 2: Runtime Extension (Future)

```python
# Future API (when integrated into SGLang core)
runtime = sgl.Runtime(
    model_path="meta-llama/Llama-2-7b-hf",
    enable_bias_audit=True,
    bias_audit_config={
        'psi_threshold': 0.1,
        'ccs_threshold': 0.8
    }
)

# Audit happens automatically
result = runtime.get_bias_audit_result()
```

---

## Best Practices

### 1. Design Unbiased Evaluations

✅ **DO:**
- Use fixed constraints across all evaluations
- Separate hyperparameter tuning from evaluation
- Document all constraint choices
- Use cross-validation with fixed settings

❌ **DON'T:**
- Adjust constraints based on intermediate results
- Cherry-pick "best" constraint settings
- Iterate constraints without documenting
- Mix tuning and evaluation phases

### 2. Interpret Results Carefully

- **Low bias ≠ perfect evaluation**: Other biases may exist
- **High bias ≠ invalid results**: May be legitimate optimization
- **Consider context**: Some iterative refinement is acceptable
- **Use all three indicators**: Don't rely on just one

### 3. Set Appropriate Thresholds

Default thresholds are conservative. Adjust based on your domain:

```python
# More strict (less tolerance)
auditor = BiasAuditor(
    psi_threshold=0.05,   # Lower = stricter
    ccs_threshold=0.9,    # Higher = stricter
    rho_pc_threshold=0.2  # Lower = stricter
)

# More lenient (more tolerance)
auditor = BiasAuditor(
    psi_threshold=0.15,
    ccs_threshold=0.7,
    rho_pc_threshold=0.4
)
```

### 4. Record Sufficient Data

- **Minimum**: 6 generations
- **Recommended**: 15-30 generations
- **Ideal**: 50+ generations for robust statistics
- More data → more reliable detection

### 5. Choose Time Periods Wisely

```python
# Too few periods: Less sensitive
result = auditor.audit(time_periods=2)

# Balanced
result = auditor.audit(time_periods=5)

# Too many periods: May be overly sensitive
result = auditor.audit(time_periods=20)
```

**Rule of thumb**: `time_periods = len(history) // 3` (at least 2)

---

## Troubleshooting

### Issue: "Need at least 6 generations"

**Cause**: Insufficient data recorded.

**Solution:**
```python
# Check history size
print(f"Recorded: {len(auditor.history)} generations")

# Record more before auditing
for i in range(10):
    auditor.record_generation(...)
```

### Issue: False Positives

**Cause**: Natural variation flagged as bias.

**Solution:**
```python
# Use more lenient thresholds
auditor = BiasAuditor(
    psi_threshold=0.15,
    ccs_threshold=0.75,
    rho_pc_threshold=0.4
)

# Or require higher confidence
result = auditor.audit()
if result.overall_bias and result.confidence > 0.8:
    # Only act on high-confidence detections
    handle_bias()
```

### Issue: False Negatives

**Cause**: Subtle bias not detected.

**Solution:**
```python
# Use stricter thresholds
auditor = BiasAuditor(
    psi_threshold=0.05,
    ccs_threshold=0.9,
    rho_pc_threshold=0.2
)

# Record more data for better statistics
```

### Issue: Performance Scores Not Computed

**Cause**: `auto_score=False` but no scores provided.

**Solution:**
```python
# Enable auto-scoring
auditor = BiasAuditor(auto_score=True)

# Or provide scores manually
auditor.record_generation(
    output=response,
    constraints=constraints,
    performance_score=compute_score(response)
)
```

---

## Performance Considerations

### Computational Cost

- **Recording**: O(1) per generation (negligible overhead)
- **Audit**: O(T × K) where T = time periods, K = constraints
  - Typically <100ms for 50 generations
  - Can be run offline without blocking

### Memory Usage

- Each generation stores: output text + constraints + score + metadata
- Typical: ~1KB per generation
- For 1000 generations: ~1MB memory

### Optimization Tips

```python
# 1. Don't store full outputs if not needed
auditor.record_generation(
    output="",  # Empty string
    constraints=constraints,
    performance_score=score  # Provide score directly
)

# 2. Clear history periodically
result = auditor.audit()
auditor.clear_history()  # Free memory

# 3. Use batch recording for efficiency
auditor.record_batch(outputs, constraints, scores)
```

---

## Examples

See `examples/bias_detection_demo.py` for complete working examples:
- Scenario 1: Stable evaluation (no bias)
- Scenario 2: Iterative tuning (with bias)
- Scenario 3: Batch recording
- Scenario 4: JSON export

Run the demo:
```bash
python examples/bias_detection_demo.py
```

---

## Academic Reference

If you use this bias detection framework in your research, please cite:

```bibtex
@article{zhang2024circular,
  title={Circular Reasoning Bias Detection in AI Algorithm Evaluation},
  author={Zhang, Hongping and others},
  journal={Journal of Open Source Software},
  year={2024},
  note={Under review}
}
```

---

## Support

- **Issues**: [GitHub Issues](https://github.com/hongping-zh/circular-bias-detection/issues)
- **Discussions**: [SGLang Discussions](https://github.com/sgl-project/sglang/discussions)
- **Documentation**: [Full Docs](https://circular-bias-detection.readthedocs.io)

---

## License

This bias detection module is part of SGLang and follows the same Apache 2.0 license.

The underlying framework is from `circular-bias-detection` (MIT License).
