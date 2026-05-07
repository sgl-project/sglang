# SemanticEmbedding Fuzzy Provider - Response Quality Samples

Captured by `scripts/run_quality_probe.py` against a local SGLang
server running Qwen2.5-7B-Instruct-AWQ with the SemanticEmbedding
fuzzy provider enabled. See `SEMANTIC_FUZZY_MATCH.md` ("Reproduce
locally") for the exact server-launch command and how to re-run this.

- Endpoint: `http://localhost:8000` (originally captured against an
  internal A10G + Qwen2.5-7B-Instruct-AWQ pod; reproduces locally
  on any GPU large enough to host the model)
- Model: `Qwen/Qwen2.5-7B-Instruct-AWQ`
- Sampling: temperature=0.0, max_tokens=512, stream=true
- SGLang: `feat(fuzzy_match): SemanticEmbedding provider + 4 pool-slot
  leak fixes` (this PR's branch tip)

## What this file demonstrates

This file demonstrates one thing: **with the SemanticEmbedding fuzzy
provider enabled, paraphrased instructions over the same context return
substantively the same answer as the seed**. That is a correctness
property - the cache layer is not corrupting outputs.

The probe is a response-quality smoke test, not a microbenchmark.
Cache-speedup evidence comes from the longeval and 7B-AWQ benches in
the "Validation" section of `SEMANTIC_FUZZY_MATCH.md` (42 fuzzy
realizations, max 22.26× warm-path speedup at 8K ctx on 1.5B; 18
realizations and 31% HIT rate at 8K on 7B-AWQ, both pool-leak-free).

## `article_summary`

Context: 1652 chars.

### Seed instruction
> Summarize the article above in three short paragraphs.

**Seed response**:

The James Webb Space Telescope (JWST) is the largest and most powerful space telescope ever constructed, launched on December 25, 2021. Orbiting the Sun at the second Lagrange point (L2), approximately 1.5 million kilometers from Earth, JWST's primary mirror, consisting of 18 hexagonal beryllium segments coated in gold, spans 6.5 meters in diameter-nearly three times that of the Hubble Space Telescope. Unlike Hubble, which primarily observes in the optical and near-infrared spectra, JWST focuses on the infrared spectrum, enabling it to observe through cosmic dust clouds and trace the light from the earliest galaxies formed after the Big Bang.

Among its groundbreaking achievements, JWST has produced the deepest infrared image of the universe, detailed analyses of exoplanet atmospheres, including the detection of carbon dioxide on WASP-39b, and unprecedented views of the Carina Nebula and Stephan's Quintet. Originally projected to cost $1 billion and launch in 2007, the mission faced significant delays and increased costs due to technical challenges and management issues, ultimately costing approximately $10 billion and launching 14 years later than planned. Despite these setbacks, JWST is performing exceptionally well and is funded for at least a 20-year mission, with scientists hoping it will help answer fundamental questions about star and planetary system formation, galaxy evolution, and the potential habitability of exoplanets.

### Variant instruction (paraphrased)
> Give me a three-paragraph summary of the article above.

**Variant response**:

The James Webb Space Telescope (JWST), launched on December 25, 2021, is a groundbreaking astronomical instrument designed to observe the infrared spectrum, enabling unprecedented views of the cosmos. With its primary mirror spanning 6.5 meters in diameter-nearly three times that of the Hubble Space Telescope-and equipped with a five-layer sunshield the size of a tennis court, JWST operates at the second Lagrange point (L2) approximately 1.5 million kilometers from Earth. Since its launch, JWST has produced remarkable scientific results, including the deepest infrared image of the universe ever captured, detailed analyses of exoplanet atmospheres such as detecting carbon dioxide on WASP-39b, and stunning visuals of celestial formations like the Carina Nebula and Stephan's Quintet.

Despite initial projections of a $1 billion budget and a 2007 launch date, JWST faced significant delays and cost overruns due to technical challenges and management issues, ultimately costing around $10 billion and launching 14 years later than planned. However, the telescope has surpassed expectations in performance and is currently funded for at least a 20-year mission. Scientists anticipate that JWST will provide crucial insights into the formation of stars and planetary systems, the evolution of galaxies, and the potential for life on exoplanets, thereby advancing our understanding of the universe's origins and the conditions necessary for habitability.

Both responses cover the same core facts (launch date, L2 orbit, mirror
size, infrared focus, key discoveries, cost overruns, mission length)
and reach equivalent summaries. The paraphrased variant elaborates on
slightly different aspects but does not contradict the seed.

---

## `factual_extraction`

Context: 1457 chars.

### Seed instruction
> List the years and the major events at each year mentioned in the passage above.

**Seed response**:

Here is the list of years and major events in Marie Curie's life based on the provided passage:

- **1867**: Born Maria Salomea Sklodowska in Warsaw, Poland.
- **1891**: Moves to Paris to study at the Sorbonne.
- **1894**: Meets her future husband and research partner, Pierre Curie.
- **July 1898**: Discovered polonium (named after Marie's native Poland).
- **December 1898**: Discovered radium.
- **1903**: Became the first woman to receive a Nobel Prize, sharing the Physics prize with Pierre and Henri Becquerel for their work on spontaneous radiation.
- **1906**: After Pierre's tragic death, continues their research.
- **1911**: Received a second Nobel Prize, this time in Chemistry, for her discovery of radium and polonium and her isolation of pure radium.
- **World War I (1914-1918)**: Developed mobile X-ray units, known as 'petites Curies', which were deployed near the front lines and helped diagnose injuries in over a million wounded soldiers.
- **July 4, 1934**: Died in Sancellemoz, France, of aplastic anemia, almost certainly caused by her prolonged exposure to radiation.

### Variant instruction (paraphrased)
> Extract a chronological list of years and what happened at each one from the passage above.

**Variant response**:

Here is a chronological list of years and what happened in each one based on the passage:

- 1867: Marie Curie, born Maria Salomea Sklodowska, is born in Warsaw.
- 1891: Marie Curie moves to Paris to study at the Sorbonne.
- 1894: Marie Curie meets her future husband and research partner, Pierre Curie.
- July 1898: Marie and Pierre Curie discover polonium.
- December 1898: Marie and Pierre Curie discover radium.
- 1903: Marie Curie becomes the first woman to receive a Nobel Prize, sharing the Physics prize with Pierre and Henri Becquerel for their work on spontaneous radiation.
- 1906: Pierre Curie dies in a street accident.
- 1911: Marie Curie receives a second Nobel Prize, this time in Chemistry, for her discovery of radium and polonium and her isolation of pure radium.
- During World War I (circa 1914-1918): Marie Curie develops mobile X-ray units, known as 'petites Curies', which are deployed near the front lines.
- 1934: Marie Curie dies on July 4 in Sancellemoz, France, of aplastic anemia, almost certainly caused by her prolonged exposure to radiation.

Identical fact set. Both list every year mentioned in the passage with
the correct event attached. Formatting differs (bullet style, bold
emphasis) but content is equivalent.

---

## `code_explanation`

Context: 756 chars.

### Seed instruction
> Explain how the merge_intervals function works, step by step.

**Seed response**:

Certainly! Let's break down the `merge_intervals` function step by step:

### Step 1: Check for Empty Input
```python
if not intervals:
    return []
```
- The function first checks if the input list `intervals` is empty.
- If it is, the function immediately returns an empty list.

### Step 2: Sort the Intervals
```python
intervals.sort(key=lambda x: x[0])
```
- The intervals are sorted based on their starting points (`x[0]`). This ensures that overlapping or adjacent intervals are next to each other in the list.

### Step 3: Initialize the Merged List
```python
merged = [intervals[0]]
```
- A new list `merged` is initialized with the first interval from the sorted list. This will be used to store the merged intervals.

### Step 4: Iterate Through the Remaining Intervals
```python
for current in intervals[1:]:
    last = merged[-1]
    if current[0] <= last[1]:
        # Overlapping or adjacent intervals
        merged[-1] = (last[0], max(last[1], current[1]))
    else:
        # No overlap, add the current interval as a new entry
        merged.append(current)
```
- The loop starts from the second interval and iterates through the rest of the intervals.
- For each `current` interval, we compare its start point (`current[0]`) with the end point of the last merged interval (`last[1]`).

#### Case 1: Overlapping or Adjacent Intervals
- If `current[0]` is less than or equal to `last[1]`, it means the intervals overlap or are adjacent.
  - We update the last merged interval to cover both intervals. Specifically, the start of the last merged interval remains the same (`last[0]`), but the end is updated to be the maximum of the two end points (`max(last[1], current[1])`).

#### Case 2: Non-Overlapping Intervals
- If `current[0]` is greater than `last[1]`, it means there is no overlap between the intervals.
  - We simply append the `current` interval to the `merged` list.

### Step 5: Return the Result
```python
return merged
```
- Finally, the function returns the `merged` list containing the non-overlapping,

### Variant instruction (paraphrased)
> Walk me through what the merge_intervals function does, in detail.

**Variant response**:

Certainly! The `merge_intervals` function is designed to take a list of intervals, where each interval is represented as a tuple `(start, end)`, and merge any overlapping intervals into a single interval. Here's a detailed step-by-step explanation of how the function works:

1. **Check for Empty Input:**
   ```python
   if not intervals:
       return []
   ```
   - If the input list `intervals` is empty, the function immediately returns an empty list.

2. **Sort the Intervals by Start Time:**
   ```python
   intervals.sort(key=lambda x: x[0])
   ```
   - The intervals are sorted based on their start times (`x[0]`). This ensures that we process the intervals in chronological order, which is necessary for detecting overlaps correctly.

3. **Initialize the Merged List with the First Interval:**
   ```python
   merged = [intervals[0]]
   ```
   - We start by initializing the `merged` list with the first interval from the sorted list. This will be our starting point for merging intervals.

4. **Iterate Through the Remaining Intervals:**
   ```python
   for current in intervals[1:]:
       last = merged[-1]
       if current[0] <= last[1]:
           merged[-1] = (last[0], max(last[1], current[1]))
       else:
           merged.append(current)
   ```
   - For each subsequent interval (`current`), we compare it with the last interval in the `merged` list (`last`).
     - If the start time of the current interval (`current[0]`) is less than or equal to the end time of the last interval (`last[1]`), it means they overlap.
       - In this case, we update the last interval in the `merged` list to cover the entire range from the start of the last interval to the maximum end time between the last interval and the current interval.
     - If the current interval does not overlap with the last interval, we simply append the current interval to the `merged` list.

5. **Return the Final List of Non-Overlapping Intervals:**
   ```python
   return merged
   ```
   - After processing all intervals, the `merged` list contains the final set of non-overlapping intervals, which are returned by the function.

Both walk through the same five logical steps (empty check → sort →
initialize merged → iterate and merge or append → return). Both
correctly identify the overlap condition (`current[0] <= last[1]`) and
the merge formula (`(last[0], max(last[1], current[1]))`). Different
narrative voice; same algorithm coverage.
