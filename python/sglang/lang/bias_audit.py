"""
Circular Bias Detection for SGLang

This module provides real-time circular reasoning bias detection for LLM
evaluation workflows in SGLang. It implements statistical indicators to
identify when evaluation constraints are being adjusted based on performance
in a circular manner.

The implementation is based on the framework described in:
    Zhang et al. (2024). "Circular Reasoning Bias Detection in AI Algorithm 
    Evaluation." Journal of Open Source Software (under review).

Example:
    >>> from sglang.lang.bias_audit import BiasAuditor
    >>> 
    >>> # Initialize auditor
    >>> auditor = BiasAuditor()
    >>> 
    >>> # Record generations
    >>> for i in range(10):
    >>>     auditor.record_generation(
    >>>         output="Model response...",
    >>>         constraints={'temperature': 0.7, 'max_new_tokens': 100}
    >>>     )
    >>> 
    >>> # Perform audit
    >>> result = auditor.audit(time_periods=3)
    >>> print(f"Bias detected: {result.overall_bias}")
    >>> print(f"PSI: {result.psi_score:.4f}")
    >>> print(f"CCS: {result.ccs_score:.4f}")
    >>> print(f"ρ_PC: {result.rho_pc_score:.4f}")
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
import warnings
import json

try:
    import numpy as np
except ImportError:
    raise ImportError(
        "NumPy is required for bias detection. Install with: pip install numpy"
    )

try:
    from circular_bias_detector.core.metrics import (
        compute_psi,
        compute_ccs,
        compute_rho_pc,
        detect_bias_threshold
    )
    _CBD_AVAILABLE = True
except ImportError:
    _CBD_AVAILABLE = False
    warnings.warn(
        "circular_bias_detector not found. "
        "Install with: pip install circular-bias-detection"
    )


@dataclass
class BiasAuditResult:
    """
    Result of a bias audit operation.
    
    Attributes
    ----------
    psi_score : float
        Parameter Stability Index (0 to ∞, lower is better)
        Measures how much performance varies across time periods
    ccs_score : float
        Constraint Consistency Score (0 to 1, higher is better)
        Measures how consistent evaluation constraints are
    rho_pc_score : float
        Performance-Constraint Correlation (-1 to 1)
        Measures correlation between performance and constraints
    overall_bias : bool
        True if circular bias is detected (majority vote)
    confidence : float
        Confidence level (0 to 1) based on indicator agreement
    bias_votes : int
        Number of indicators that flagged bias (out of 3)
    psi_bias : bool
        Whether PSI indicates bias
    ccs_bias : bool
        Whether CCS indicates bias
    rho_pc_bias : bool
        Whether ρ_PC indicates bias
    metadata : dict
        Additional metadata about the audit
    """
    psi_score: float
    ccs_score: float
    rho_pc_score: float
    overall_bias: bool
    confidence: float
    bias_votes: int = 0
    psi_bias: bool = False
    ccs_bias: bool = False
    rho_pc_bias: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "Circular Bias Detection Report",
            "=" * 60,
            f"PSI (Parameter Stability):    {self.psi_score:.4f} {'⚠ BIAS' if self.psi_bias else '✓ OK'}",
            f"CCS (Constraint Consistency): {self.ccs_score:.4f} {'⚠ BIAS' if self.ccs_bias else '✓ OK'}",
            f"ρ_PC (Perf-Const Correlation): {self.rho_pc_score:+.4f} {'⚠ BIAS' if self.rho_pc_bias else '✓ OK'}",
            "",
            f"Overall Assessment: {'⚠ BIAS DETECTED' if self.overall_bias else '✓ NO BIAS'}",
            f"Confidence: {self.confidence:.1%}",
            f"Indicators Flagged: {self.bias_votes}/3",
            "=" * 60
        ]
        return "\n".join(lines)


class BiasAuditor:
    """
    Auditor for detecting circular reasoning bias in LLM evaluations.
    
    This class collects generation history and computes statistical indicators
    to detect if evaluation constraints are being adjusted based on performance
    in a circular manner.
    
    Parameters
    ----------
    psi_threshold : float, default=0.1
        Threshold for PSI. Values above indicate instability.
    ccs_threshold : float, default=0.8
        Threshold for CCS. Values below indicate inconsistency.
    rho_pc_threshold : float, default=0.3
        Threshold for |ρ_PC|. Values above indicate correlation.
    auto_score : bool, default=True
        Automatically compute performance scores if not provided.
    
    Examples
    --------
    >>> auditor = BiasAuditor()
    >>> 
    >>> # Simulate iterative tuning
    >>> for temp in [0.5, 0.6, 0.7, 0.8, 0.9]:
    >>>     for _ in range(3):
    >>>         auditor.record_generation(
    >>>             output="Response...",
    >>>             constraints={'temperature': temp}
    >>>         )
    >>> 
    >>> result = auditor.audit()
    >>> print(result.summary())
    """
    
    def __init__(self,
                 psi_threshold: float = 0.1,
                 ccs_threshold: float = 0.8,
                 rho_pc_threshold: float = 0.3,
                 auto_score: bool = True):
        if not _CBD_AVAILABLE:
            raise ImportError(
                "circular_bias_detector package is required. "
                "Install with: pip install circular-bias-detection"
            )
        
        self.psi_threshold = psi_threshold
        self.ccs_threshold = ccs_threshold
        self.rho_pc_threshold = rho_pc_threshold
        self.auto_score = auto_score
        
        # Generation history
        self.history: List[Dict[str, Any]] = []
        
        # Statistics
        self.total_generations = 0
        self.total_audits = 0
    
    def record_generation(self,
                         output: str,
                         constraints: Dict[str, Any],
                         performance_score: Optional[float] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a single generation for later auditing.
        
        Parameters
        ----------
        output : str
            Generated text output
        constraints : dict
            Generation constraints (temperature, max_new_tokens, etc.)
        performance_score : float, optional
            Pre-computed performance score. If None and auto_score=True,
            will be computed automatically.
        metadata : dict, optional
            Additional metadata to store with this generation
        
        Examples
        --------
        >>> auditor.record_generation(
        ...     output="The sentiment is positive.",
        ...     constraints={'temperature': 0.7, 'max_new_tokens': 50},
        ...     performance_score=0.85
        ... )
        """
        # Compute performance score if needed
        if performance_score is None and self.auto_score:
            performance_score = self._compute_performance_score(output)
        
        # Store record
        record = {
            'output': output,
            'constraints': dict(constraints),  # Copy to avoid mutation
            'performance_score': performance_score,
            'metadata': metadata or {},
            'generation_id': self.total_generations
        }
        
        self.history.append(record)
        self.total_generations += 1
    
    def record_batch(self,
                    outputs: List[str],
                    constraints: List[Dict[str, Any]],
                    performance_scores: Optional[List[float]] = None) -> None:
        """
        Record multiple generations at once.
        
        Parameters
        ----------
        outputs : List[str]
            List of generated outputs
        constraints : List[dict]
            List of constraint dicts (one per output)
        performance_scores : List[float], optional
            Pre-computed performance scores
        """
        if performance_scores is None:
            performance_scores = [None] * len(outputs)
        
        if len(outputs) != len(constraints) or len(outputs) != len(performance_scores):
            raise ValueError("outputs, constraints, and performance_scores must have same length")
        
        for output, const, score in zip(outputs, constraints, performance_scores):
            self.record_generation(output, const, score)
    
    def audit(self,
              time_periods: Optional[int] = None,
              min_generations: int = 6) -> BiasAuditResult:
        """
        Perform bias audit on recorded generations.
        
        Parameters
        ----------
        time_periods : int, optional
            Number of time periods to divide history into.
            Default: len(history) // 3 (at least 2)
        min_generations : int, default=6
            Minimum number of generations required for audit
        
        Returns
        -------
        BiasAuditResult
            Audit result with all indicators and bias detection
        
        Raises
        ------
        ValueError
            If insufficient generations recorded
        
        Examples
        --------
        >>> result = auditor.audit(time_periods=5)
        >>> if result.overall_bias:
        ...     print(f"Warning: Bias detected with {result.confidence:.0%} confidence")
        """
        if len(self.history) < min_generations:
            raise ValueError(
                f"Need at least {min_generations} generations for audit, "
                f"but only {len(self.history)} recorded. "
                f"Call record_generation() more times."
            )
        
        # Determine time periods
        if time_periods is None:
            time_periods = max(len(self.history) // 3, 2)
        
        if time_periods < 2:
            raise ValueError("time_periods must be at least 2")
        
        # Build matrices
        perf_matrix, const_matrix = self._build_matrices(time_periods)
        
        # Compute indicators
        psi = compute_psi(perf_matrix)
        ccs = compute_ccs(const_matrix)
        rho_pc = compute_rho_pc(perf_matrix, const_matrix)
        
        # Detect bias
        bias_result = detect_bias_threshold(
            psi_score=psi,
            ccs_score=ccs,
            rho_pc_score=rho_pc,
            psi_threshold=self.psi_threshold,
            ccs_threshold=self.ccs_threshold,
            rho_pc_threshold=self.rho_pc_threshold
        )
        
        # Create result
        result = BiasAuditResult(
            psi_score=psi,
            ccs_score=ccs,
            rho_pc_score=rho_pc,
            overall_bias=bias_result['overall_bias'],
            confidence=bias_result['confidence'],
            bias_votes=bias_result['bias_votes'],
            psi_bias=bias_result['psi_bias'],
            ccs_bias=bias_result['ccs_bias'],
            rho_pc_bias=bias_result['rho_pc_bias'],
            metadata={
                'num_generations': len(self.history),
                'time_periods': time_periods,
                'generations_per_period': len(self.history) // time_periods,
                'thresholds': {
                    'psi': self.psi_threshold,
                    'ccs': self.ccs_threshold,
                    'rho_pc': self.rho_pc_threshold
                }
            }
        )
        
        self.total_audits += 1
        return result
    
    def clear_history(self) -> None:
        """Clear all recorded generations."""
        self.history.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get auditor statistics.
        
        Returns
        -------
        dict
            Statistics including total generations, audits, etc.
        """
        return {
            'total_generations': self.total_generations,
            'total_audits': self.total_audits,
            'history_size': len(self.history),
            'thresholds': {
                'psi': self.psi_threshold,
                'ccs': self.ccs_threshold,
                'rho_pc': self.rho_pc_threshold
            }
        }
    
    def _build_matrices(self, time_periods: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build performance and constraint matrices from history.
        
        Parameters
        ----------
        time_periods : int
            Number of time periods to divide history into
        
        Returns
        -------
        perf_matrix : ndarray, shape (T, 1)
            Performance matrix with T time periods
        const_matrix : ndarray, shape (T, M)
            Constraint matrix with M constraint types
        """
        period_size = len(self.history) // time_periods
        
        # First, collect all unique numerical constraint keys across entire history
        # This ensures consistent matrix shape even if keys vary across time periods
        all_constraint_keys = set()
        for record in self.history:
            for key, value in record['constraints'].items():
                if isinstance(value, (int, float)):
                    all_constraint_keys.add(key)
        
        # Sort for consistent ordering
        sorted_keys = sorted(all_constraint_keys)
        
        if not sorted_keys:
            warnings.warn(
                "No numerical constraints found in history. "
                "Using default constraint matrix."
            )
            sorted_keys = ['_default_']
        
        perf_matrix = []
        const_matrix = []
        
        for t in range(time_periods):
            start_idx = t * period_size
            end_idx = start_idx + period_size if t < time_periods - 1 else len(self.history)
            
            period_history = self.history[start_idx:end_idx]
            
            if not period_history:
                warnings.warn(f"Empty time period {t}, using default values")
                perf_matrix.append([0.5])
                const_matrix.append([0.0] * len(sorted_keys))
                continue
            
            # Aggregate performance for this period
            period_perfs = [
                h['performance_score'] for h in period_history 
                if h['performance_score'] is not None
            ]
            
            if not period_perfs:
                warnings.warn(f"No valid performance scores for time period {t}")
                period_perfs = [0.5]  # Default
            
            perf_matrix.append([np.mean(period_perfs)])
            
            # Extract constraint values using complete key set
            constraint_values = []
            for key in sorted_keys:
                # Average this constraint across the period
                values = []
                for h in period_history:
                    val = h['constraints'].get(key)
                    if val is not None and isinstance(val, (int, float)):
                        values.append(float(val))
                
                if values:
                    constraint_values.append(np.mean(values))
                else:
                    # Use NaN for missing constraints, then fill with 0.0
                    # This distinguishes between "not present" and "present but zero"
                    constraint_values.append(np.nan)
            
            const_matrix.append(constraint_values)
        
        # Convert to arrays and handle missing values
        perf_array = np.array(perf_matrix)
        const_array = np.array(const_matrix)
        
        # Replace NaN with 0.0 (or could use mean/median of that column)
        const_array = np.nan_to_num(const_array, nan=0.0)
        
        return perf_array, const_array
    
    def _compute_performance_score(self, output: str) -> float:
        """
        Compute performance score from output text.
        
        This is a simple heuristic. For production use, consider:
        - Perplexity
        - Task-specific metrics
        - Human evaluation scores
        
        Parameters
        ----------
        output : str
            Generated text
        
        Returns
        -------
        float
            Performance score in [0, 1]
        """
        # Simple heuristic: based on length and diversity
        # Longer outputs are generally better (up to a point)
        length_score = min(len(output) / 100.0, 1.0)
        
        # Vocabulary diversity (unique words / total words)
        words = output.lower().split()
        if len(words) > 0:
            diversity_score = len(set(words)) / len(words)
        else:
            diversity_score = 0.0
        
        # Combined score
        score = 0.7 * length_score + 0.3 * diversity_score
        
        return float(np.clip(score, 0.0, 1.0))
    
    def __repr__(self) -> str:
        return (
            f"BiasAuditor("
            f"history_size={len(self.history)}, "
            f"total_generations={self.total_generations}, "
            f"total_audits={self.total_audits}"
            f")"
        )


def create_auditor(thresholds: Optional[Dict[str, float]] = None,
                  **kwargs) -> BiasAuditor:
    """
    Convenience function to create a BiasAuditor.
    
    Parameters
    ----------
    thresholds : dict, optional
        Custom thresholds: {'psi': 0.1, 'ccs': 0.8, 'rho_pc': 0.3}
    **kwargs
        Additional arguments for BiasAuditor
    
    Returns
    -------
    BiasAuditor
        Configured auditor instance
    
    Examples
    --------
    >>> auditor = create_auditor(
    ...     thresholds={'psi': 0.05, 'ccs': 0.9},
    ...     auto_score=True
    ... )
    """
    if thresholds:
        kwargs.update({
            'psi_threshold': thresholds.get('psi', 0.1),
            'ccs_threshold': thresholds.get('ccs', 0.8),
            'rho_pc_threshold': thresholds.get('rho_pc', 0.3)
        })
    
    return BiasAuditor(**kwargs)
