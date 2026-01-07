# Attention Visualization Validation Checklist

## Pre-Merge Validation for `feature/attention-token-visualization`

### 1. Off-by-One Alignment Check ✅

**Status:** PASS (Code Review)

**The Logic:** In Decoder-Only models (Llama, Qwen), attention computed at token T generates token T+1.

**Code Verification:**
- Server: `attention_tokens_decode_step` incremented AFTER token generation (`scheduler_output_processor_mixin.py:571`)
- Client: `tokenIndex = step - 1` correctly maps 1-indexed step to 0-indexed array (`useSessionStore.ts:110`)

**Visual Test:**
- Hover over "Paris" in "France is Paris"
- ✅ PASS: Line points to "France" (semantic source)
- ❌ FAIL: Line points to "is" (off-by-one error)

---

### 2. Sink Token Handling ✅

**Status:** PASS (Code Review)

**The Issue:** Token 0 (`<s>`) absorbs 50-90% of attention mass, making semantic connections invisible.

**Implementation:** (`TokenLensDrawer.tsx:52-65`)
```typescript
// Normalize excluding sink token
const nonSinkTotal = topK
  .filter((item) => item.position !== 0)
  .reduce((sum, item) => sum + item.score, 0);

const normalizedScore = isSinkToken
  ? item.score  // Keep raw score for sink
  : item.score / nonSinkTotal;  // Normalize others
```

**UI Features:**
- Warning shown when sink absorbs >30%: "🕳️ Sink token (pos 0) absorbs X% of attention"
- Semantic connections remain visible via local normalization

---

### 3. Needle Test ⏳

**Status:** PENDING (Requires Live Server)

**Prompt:**
```
The secret code is 4829. Please repeat the code. The secret code is
```

**Expected Output:** `4829`

**Validation Steps:**
1. Hover over generated **`4`**
2. ✅ PASS: Strong connection to `4` in "4829" (position ~7)
3. ❌ FAIL: Connection to "is" or ":" (alignment broken)

**Command to Test:**
```bash
# Start server
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --return-attention-tokens \
  --port 8000

# Run UI
cd examples/attention_explorer/ui
npm run dev

# Navigate to http://localhost:5173
# Enter the needle prompt and verify
```

---

### 4. Layer Selection Consistency

**Test:** Different layers should show different attention patterns.
- Early layers: More local/syntactic patterns
- Middle layers: Semantic retrieval
- Late layers: Output preparation

**Validation:**
- [ ] Layer selector works
- [ ] Different layers show visually distinct patterns
- [ ] No "all layers same" bug

---

### 5. Multi-Layer Capture

**Test:** When `attention_capture_layer_ids=[0, 15, 31]`:
- All specified layers should have data
- Non-specified layers should show "No data for this layer"

---

## Merge Criteria

| Check | Status | Required for Merge |
|-------|--------|-------------------|
| Off-by-one alignment | ✅ PASS | Yes |
| Sink token handling | ✅ PASS | Yes |
| Needle test | ⏳ Pending | Yes |
| Layer selection | ⏳ Pending | No |
| Multi-layer capture | ⏳ Pending | No |

**Decision:** Merge after Needle Test passes.
