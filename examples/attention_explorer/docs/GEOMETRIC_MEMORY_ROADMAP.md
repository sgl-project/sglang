# Geometric Memory Research Roadmap

**Based on:** "Deep sequence models tend to memorize geometrically" (arXiv:2510.26745)
**Date:** January 2026
**Status:** Research Directions for SGLang Attention Explorer

---

## Theoretical Foundation

### The Core Discovery

Our empirical findings from the Qwen3 80B vs 4B exploration align with the paper's theoretical framework:

| Our Empirical Finding | Theoretical Explanation |
|----------------------|------------------------|
| Zone clustering (syntax_floor, semantic_bridge, structure_ripple) | Spectral modes of the graph Laplacian |
| Response length predicts zone better than task type | Geometric distance determines computation mode |
| MoE has more semantic_bridge (35% vs 15%) | Expert routing enables multi-modal geometry |
| 4B relies more on structure_ripple (84% vs 64%) | Dense models compress to single geometric mode |
| Attention sink (Sinq) filtering improves clustering | Sinq is the geometric origin/anchor point |

### The Paradigm Shift

**Before:** "Manifold discovery is a useful visualization heuristic"
**After:** "Manifold discovery captures the actual computational geometry of reasoning"

---

## Research Direction 1: Spectral Embedding Router

### Current Implementation (PCA)
```python
# discovery/discovery_job.py - Stage 3
pca = PCA(n_components=50)
embeddings_pca = pca.fit_transform(scaled_fingerprints)
```

### Upgraded Implementation (Laplacian Eigenmaps)

```python
# discovery/spectral_discovery.py (NEW)
from sklearn.manifold import SpectralEmbedding
from scipy.sparse.csgraph import laplacian
from scipy.sparse import kneighbors_graph

class SpectralManifoldDiscovery:
    """
    Replace PCA with Laplacian Eigenmaps to capture the true
    spectral structure of the model's geometric memory.
    """

    def __init__(self, n_components=50, n_neighbors=15):
        self.n_components = n_components
        self.n_neighbors = n_neighbors

    def fit_transform(self, fingerprints: np.ndarray) -> np.ndarray:
        # Build k-NN graph (the "subway map" structure)
        connectivity = kneighbors_graph(
            fingerprints,
            n_neighbors=self.n_neighbors,
            mode='connectivity',
            include_self=False
        )

        # Compute graph Laplacian
        L = laplacian(connectivity, normed=True)

        # Spectral embedding via eigenvectors
        embedding = SpectralEmbedding(
            n_components=self.n_components,
            affinity='precomputed_nearest_neighbors',
            n_neighbors=self.n_neighbors,
            eigen_solver='arpack'
        )

        return embedding.fit_transform(fingerprints)

    def compute_spectral_coherence(self, query_fingerprint: np.ndarray) -> float:
        """
        Measure how well a query projects onto the spectral skeleton.

        High coherence -> Route to small model, disable CoT
        Low coherence -> Route to large model, enable CoT
        """
        # Project query onto learned eigenvectors
        projection = self.eigenvectors_.T @ query_fingerprint

        # Coherence = how much energy in top-k eigenmodes
        total_energy = np.sum(projection ** 2)
        topk_energy = np.sum(projection[:self.n_components] ** 2)

        return topk_energy / total_energy if total_energy > 0 else 0.0
```

### Router Integration

```python
# discovery/spectral_router.py (NEW)
class SpectralRouter:
    """
    Route queries based on spectral coherence with the model's
    internal geometric memory.
    """

    def __init__(self, coherence_threshold=0.7):
        self.coherence_threshold = coherence_threshold

    def route(self, query_fingerprint: np.ndarray) -> RoutingDecision:
        coherence = self.spectral_discovery.compute_spectral_coherence(query_fingerprint)

        if coherence > self.coherence_threshold:
            return RoutingDecision(
                model_size="small",  # 8B
                use_cot=False,       # Geometric lookup, no reasoning needed
                reason="High spectral coherence - query on manifold skeleton"
            )
        else:
            return RoutingDecision(
                model_size="large",  # 70B
                use_cot=True,        # Need to compute, not navigate
                reason="Low spectral coherence - query in manifold void"
            )
```

---

## Research Direction 2: Sinq as Geometric Origin

### The Hypothesis

The attention sink (token 0) acts as the geometric anchor point (0,0,0) in the model's internal coordinate system. The **angle** between current token and sink reveals reasoning mode.

### Implementation

```python
# discovery/sinq_geometry.py (NEW)
class SinqGeometryAnalyzer:
    """
    Analyze the geometric relationship between tokens and the
    attention sink to detect reasoning vs retrieval modes.
    """

    def compute_sink_angle(
        self,
        token_fingerprint: np.ndarray,
        sink_fingerprint: np.ndarray
    ) -> float:
        """
        Compute angle between token and sink in fingerprint space.
        """
        dot = np.dot(token_fingerprint, sink_fingerprint)
        norm_t = np.linalg.norm(token_fingerprint)
        norm_s = np.linalg.norm(sink_fingerprint)

        if norm_t == 0 or norm_s == 0:
            return 0.0

        cos_angle = dot / (norm_t * norm_s)
        return np.arccos(np.clip(cos_angle, -1, 1))

    def compute_angle_trajectory(
        self,
        fingerprints: List[np.ndarray],
        sink_fingerprint: np.ndarray
    ) -> np.ndarray:
        """
        Compute sink angle for each token in sequence.

        Returns trajectory that reveals:
        - Stable angle: Model is grounded (retrieval mode)
        - Spinning angle: Model is reasoning (or hallucinating)
        """
        return np.array([
            self.compute_sink_angle(fp, sink_fingerprint)
            for fp in fingerprints
        ])

    def detect_topic_shift(
        self,
        angle_trajectory: np.ndarray,
        window_size: int = 10
    ) -> List[int]:
        """
        Detect points where the sink angle changes drastically,
        indicating topic/reasoning mode transitions.
        """
        # Compute rolling angle variance
        variances = []
        for i in range(len(angle_trajectory) - window_size):
            window = angle_trajectory[i:i+window_size]
            variances.append(np.var(window))

        # Find peaks in variance (topic shifts)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(variances, height=np.mean(variances) * 2)

        return peaks.tolist()
```

### Visualization Update

```typescript
// ui/src/components/SinqCompass.tsx (NEW)
interface SinqCompassProps {
  angleTrajectory: number[];
  topicShifts: number[];
  currentStep: number;
}

export function SinqCompass({ angleTrajectory, topicShifts, currentStep }: SinqCompassProps) {
  // Render a "compass" showing the token's angle to the sink
  // Highlight topic shifts with markers
  // Show if model is "grounded" vs "spinning"
}
```

---

## Research Direction 3: Geometric Hallucination Detection

### The Science

True facts form a rigid lattice in the model's geometry. Hallucinations are "loose nodes" that don't fit the learned structure.

### Implementation

```python
# discovery/hallucination_detector.py (NEW)
class GeometricHallucinationDetector:
    """
    Detect hallucinations by measuring manifold distance.

    A token is flagged as potentially hallucinated if it's "off-manifold"
    even when its logit probability is high.
    """

    def __init__(self, cluster_centroids: np.ndarray, distance_threshold: float = 2.0):
        self.centroids = cluster_centroids
        self.threshold = distance_threshold

    def compute_manifold_distance(self, fingerprint: np.ndarray) -> float:
        """
        Compute minimum distance to any cluster centroid.
        """
        distances = np.linalg.norm(self.centroids - fingerprint, axis=1)
        return np.min(distances)

    def is_off_manifold(self, fingerprint: np.ndarray) -> Tuple[bool, float]:
        """
        Check if a token's fingerprint is geometrically suspicious.

        Returns (is_suspicious, manifold_distance)
        """
        distance = self.compute_manifold_distance(fingerprint)
        return distance > self.threshold, distance

    def score_generation(
        self,
        fingerprints: List[np.ndarray],
        logit_probs: List[float]
    ) -> List[HallucinationScore]:
        """
        Score each generated token for hallucination risk.

        High risk = High logit probability BUT high manifold distance
        (The model is confident but geometrically wrong)
        """
        scores = []
        for fp, prob in zip(fingerprints, logit_probs):
            is_off, distance = self.is_off_manifold(fp)

            # Hallucination risk = confidence * off-manifold-ness
            risk = prob * (distance / self.threshold) if is_off else 0.0

            scores.append(HallucinationScore(
                manifold_distance=distance,
                logit_probability=prob,
                hallucination_risk=risk,
                is_flagged=risk > 0.5
            ))

        return scores
```

### API Integration

```python
# srt/entrypoints/openai/serving_chat.py (UPDATE)
async def generate_with_hallucination_detection(
    request: ChatCompletionRequest,
    detector: GeometricHallucinationDetector
) -> ChatCompletionResponse:
    """
    Generate response with real-time hallucination scoring.
    """
    response = await generate_normal(request)

    if request.extra_body.get("detect_hallucinations"):
        scores = detector.score_generation(
            response.fingerprints,
            response.logit_probs
        )

        # Add hallucination metadata to response
        response.hallucination_scores = scores
        response.flagged_tokens = [
            i for i, s in enumerate(scores) if s.is_flagged
        ]

    return response
```

---

## Research Direction 4: Spectral KV Cache Eviction

### The Science

Since memory is low-rank (geometric), we don't need every token in KV cache—only the tokens that define the "shape" of the context.

### Implementation

```python
# srt/mem_cache/spectral_eviction.py (NEW)
class SpectralKVCacheEviction:
    """
    Evict geometrically redundant tokens from KV cache.

    Keep tokens that define the "spectral skeleton" of the context.
    Evict tokens that lie on interpolated lines between skeleton points.
    """

    def __init__(self, retention_ratio: float = 0.3):
        self.retention_ratio = retention_ratio

    def identify_skeleton_tokens(
        self,
        kv_cache: KVCache,
        fingerprints: np.ndarray
    ) -> List[int]:
        """
        Identify tokens that define the geometric skeleton.

        Uses spectral clustering to find "landmark" tokens.
        """
        n_retain = int(len(fingerprints) * self.retention_ratio)

        # Compute spectral embedding
        from sklearn.manifold import SpectralEmbedding
        embedding = SpectralEmbedding(n_components=min(10, n_retain))
        coords = embedding.fit_transform(fingerprints)

        # Find tokens at extremes of each spectral dimension
        skeleton_indices = set()
        for dim in range(coords.shape[1]):
            # Keep min and max along each dimension
            skeleton_indices.add(np.argmin(coords[:, dim]))
            skeleton_indices.add(np.argmax(coords[:, dim]))

        # Fill remaining slots with diverse tokens
        from sklearn.cluster import KMeans
        remaining = n_retain - len(skeleton_indices)
        if remaining > 0:
            kmeans = KMeans(n_clusters=remaining)
            kmeans.fit(fingerprints)
            # Get token closest to each centroid
            for centroid in kmeans.cluster_centers_:
                distances = np.linalg.norm(fingerprints - centroid, axis=1)
                skeleton_indices.add(np.argmin(distances))

        return sorted(skeleton_indices)[:n_retain]

    def evict(self, kv_cache: KVCache, fingerprints: np.ndarray) -> KVCache:
        """
        Evict redundant tokens, keeping only skeleton.
        """
        skeleton_indices = self.identify_skeleton_tokens(kv_cache, fingerprints)

        return KVCache(
            keys=kv_cache.keys[:, skeleton_indices],
            values=kv_cache.values[:, skeleton_indices],
            skeleton_indices=skeleton_indices
        )
```

### Potential Impact

- **Current:** 100K context requires ~40GB VRAM
- **With Spectral Eviction (30% retention):** ~12GB VRAM
- **Reasoning preserved:** Skeleton tokens maintain geometric structure

---

## Research Direction 5: Isomorphic RAG

### The Problem

Standard RAG uses generic embeddings (OpenAI, Sentence-BERT) with different geometry than the LLM. This causes "distractor" retrievals—documents that match keywords but don't fit the model's reasoning path.

### Implementation

```python
# discovery/isomorphic_rag.py (NEW)
class IsomorphicRAGAdapter:
    """
    Project external embeddings into the LLM's internal Eigen-space.

    This ensures retrieved documents fit the model's geometric
    reasoning structure, not just keyword similarity.
    """

    def __init__(self, llm_eigenvectors: np.ndarray):
        self.llm_eigenvectors = llm_eigenvectors
        self.projection_matrix = None

    def train_adapter(
        self,
        external_embeddings: np.ndarray,  # From OpenAI/SBERT
        llm_fingerprints: np.ndarray       # From our manifold discovery
    ):
        """
        Learn a projection that aligns external embedding space
        with LLM's internal geometry.

        Uses Procrustes analysis for optimal alignment.
        """
        from scipy.linalg import orthogonal_procrustes

        # Find optimal rotation/scaling to align spaces
        R, scale = orthogonal_procrustes(external_embeddings, llm_fingerprints)

        self.projection_matrix = R * scale

    def project_to_llm_space(self, external_embedding: np.ndarray) -> np.ndarray:
        """
        Project an external embedding into LLM's geometry.
        """
        return external_embedding @ self.projection_matrix

    def retrieve_isomorphic(
        self,
        query_fingerprint: np.ndarray,
        document_embeddings: np.ndarray,
        top_k: int = 5
    ) -> List[int]:
        """
        Retrieve documents that fit the LLM's reasoning path,
        not just keyword similarity.
        """
        # Project documents into LLM space
        projected = np.array([
            self.project_to_llm_space(emb)
            for emb in document_embeddings
        ])

        # Find documents closest to query in LLM geometry
        distances = np.linalg.norm(projected - query_fingerprint, axis=1)

        return np.argsort(distances)[:top_k].tolist()
```

---

## Updated 8-Hour Discovery Workflow

### Current Workflow
1. Hours 0-2: Capture attention matrices
2. Hours 2-4: PCA + UMAP dimensionality reduction
3. Hours 4-6: HDBSCAN clustering
4. Hours 6-8: Zone classification + visualization

### Upgraded Spectral Workflow
1. **Hours 0-2:** Capture attention matrices + sink token fingerprints
2. **Hours 2-4:** Compute **Laplacian Eigenmaps** (spectral skeleton)
3. **Hours 4-6:** Train **Spectral Router** (coherence thresholds)
4. **Hours 6-8:** Validate with **Hallucination Detection** + **Sink Angle Analysis**

### New Metrics to Track

| Metric | Meaning | Use Case |
|--------|---------|----------|
| Spectral Coherence | How well query fits skeleton | Routing decision |
| Sink Angle Variance | Reasoning vs retrieval mode | CoT trigger |
| Manifold Distance | Off-manifold risk | Hallucination flag |
| Skeleton Coverage | KV cache efficiency | Memory optimization |

---

## Implementation Priority

### Phase 1: Spectral Foundation (2 weeks)
- [ ] Implement `SpectralManifoldDiscovery` class
- [ ] Add spectral coherence metric to discovery output
- [ ] Update UI to show spectral vs PCA embeddings

### Phase 2: Sinq Geometry (1 week)
- [ ] Implement `SinqGeometryAnalyzer`
- [ ] Add sink angle trajectory to fingerprint capture
- [ ] Create Sinq Compass visualization component

### Phase 3: Hallucination Detection (2 weeks)
- [ ] Implement `GeometricHallucinationDetector`
- [ ] Add hallucination scoring to API response
- [ ] Build flagged token highlighting in UI

### Phase 4: Spectral Router (2 weeks)
- [ ] Implement `SpectralRouter`
- [ ] Integrate with SGLang model routing
- [ ] A/B test spectral vs heuristic routing

### Phase 5: Advanced Optimizations (4 weeks)
- [ ] Implement spectral KV cache eviction
- [ ] Build isomorphic RAG adapter
- [ ] Benchmark memory/quality tradeoffs

---

## Validation Experiments

### Experiment 1: Spectral vs PCA Clustering Quality
- Run discovery with both methods on same dataset
- Compare cluster coherence (silhouette score)
- Measure zone classification accuracy

### Experiment 2: Sink Angle Predicts Hallucination
- Generate responses with known hallucinations
- Measure sink angle variance at hallucination points
- Train classifier: angle_variance -> hallucination_probability

### Experiment 3: Spectral Router Accuracy
- Compare routing decisions: spectral_coherence vs response_length
- Measure quality/latency tradeoffs for each routing strategy

### Experiment 4: KV Cache Compression
- Benchmark: full cache vs spectral eviction (30%, 50%, 70% retention)
- Measure perplexity degradation
- Test on long-context tasks (100K+ tokens)

---

## Connection to 80B vs 4B Findings

Our exploration revealed:
- **80B (MoE):** 35% semantic_bridge, 64% structure_ripple
- **4B (Dense):** 15% semantic_bridge, 84% structure_ripple

### Spectral Interpretation

| Finding | Spectral Explanation |
|---------|---------------------|
| 4B has more structure_ripple | Dense model has fewer spectral modes, collapses to single geometric pattern |
| 80B has more semantic_bridge | MoE experts enable multiple spectral modes, richer geometry |
| Both minimize syntax_floor | Local patterns are low-frequency, handled by position encoding |
| Response length predicts zone | Longer generation = more eigenmodes activated |

### Prediction

If we run spectral analysis on both models:
- **80B:** Higher spectral rank (more eigenmodes with significant energy)
- **4B:** Lower spectral rank (energy concentrated in fewer modes)

This would confirm that MoE enables **geometric diversity** while dense models **compress to simpler geometry**.

---

## References

1. "Deep sequence models tend to memorize geometrically" (arXiv:2510.26745)
2. Laplacian Eigenmaps: Belkin & Niyogi, 2003
3. Spectral Clustering: Von Luxburg, 2007
4. RoPE: Rotary Position Embedding (Su et al., 2021)

---

*Roadmap created: January 2026*
*Status: Research directions pending implementation*
