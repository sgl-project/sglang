"""
Unit tests for SGLang BiasAuditor

Run with: pytest test_bias_audit.py -v
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.sglang.lang.bias_audit import (
    BiasAuditor,
    BiasAuditResult,
    create_auditor
)


class TestBiasAuditResult:
    """Tests for BiasAuditResult dataclass."""
    
    def test_creation(self):
        """Test creating a BiasAuditResult."""
        result = BiasAuditResult(
            psi_score=0.05,
            ccs_score=0.95,
            rho_pc_score=0.1,
            overall_bias=False,
            confidence=0.0
        )
        
        assert result.psi_score == 0.05
        assert result.ccs_score == 0.95
        assert result.rho_pc_score == 0.1
        assert result.overall_bias is False
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = BiasAuditResult(
            psi_score=0.05,
            ccs_score=0.95,
            rho_pc_score=0.1,
            overall_bias=False,
            confidence=0.0
        )
        
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d['psi_score'] == 0.05
        assert d['ccs_score'] == 0.95
    
    def test_to_json(self):
        """Test conversion to JSON."""
        result = BiasAuditResult(
            psi_score=0.05,
            ccs_score=0.95,
            rho_pc_score=0.1,
            overall_bias=False,
            confidence=0.0
        )
        
        json_str = result.to_json()
        assert isinstance(json_str, str)
        assert 'psi_score' in json_str
        assert '0.05' in json_str
    
    def test_summary(self):
        """Test human-readable summary."""
        result = BiasAuditResult(
            psi_score=0.15,
            ccs_score=0.7,
            rho_pc_score=0.5,
            overall_bias=True,
            confidence=0.67,
            bias_votes=2,
            psi_bias=True,
            ccs_bias=True,
            rho_pc_bias=False
        )
        
        summary = result.summary()
        assert isinstance(summary, str)
        assert 'PSI' in summary
        assert 'CCS' in summary
        assert 'BIAS DETECTED' in summary
        assert '67' in summary  # Confidence


class TestBiasAuditor:
    """Tests for BiasAuditor class."""
    
    def test_initialization(self):
        """Test BiasAuditor initialization."""
        auditor = BiasAuditor()
        
        assert auditor.psi_threshold == 0.1
        assert auditor.ccs_threshold == 0.8
        assert auditor.rho_pc_threshold == 0.3
        assert len(auditor.history) == 0
        assert auditor.total_generations == 0
    
    def test_initialization_custom_thresholds(self):
        """Test initialization with custom thresholds."""
        auditor = BiasAuditor(
            psi_threshold=0.05,
            ccs_threshold=0.9,
            rho_pc_threshold=0.4
        )
        
        assert auditor.psi_threshold == 0.05
        assert auditor.ccs_threshold == 0.9
        assert auditor.rho_pc_threshold == 0.4
    
    def test_record_generation(self):
        """Test recording a single generation."""
        auditor = BiasAuditor()
        
        auditor.record_generation(
            output="This is a test response.",
            constraints={'temperature': 0.7, 'max_new_tokens': 100},
            performance_score=0.85
        )
        
        assert len(auditor.history) == 1
        assert auditor.total_generations == 1
        assert auditor.history[0]['output'] == "This is a test response."
        assert auditor.history[0]['performance_score'] == 0.85
        assert auditor.history[0]['constraints']['temperature'] == 0.7
    
    def test_record_generation_auto_score(self):
        """Test automatic performance scoring."""
        auditor = BiasAuditor(auto_score=True)
        
        auditor.record_generation(
            output="This is a test response with some content.",
            constraints={'temperature': 0.7}
        )
        
        assert auditor.history[0]['performance_score'] is not None
        assert 0 <= auditor.history[0]['performance_score'] <= 1
    
    def test_record_batch(self):
        """Test recording multiple generations at once."""
        auditor = BiasAuditor()
        
        outputs = ["Response 1", "Response 2", "Response 3"]
        constraints = [
            {'temperature': 0.7},
            {'temperature': 0.8},
            {'temperature': 0.9}
        ]
        scores = [0.7, 0.8, 0.9]
        
        auditor.record_batch(outputs, constraints, scores)
        
        assert len(auditor.history) == 3
        assert auditor.total_generations == 3
        assert auditor.history[1]['performance_score'] == 0.8
    
    def test_audit_insufficient_data(self):
        """Test audit with insufficient generations."""
        auditor = BiasAuditor()
        
        # Record only 3 generations (need at least 6)
        for i in range(3):
            auditor.record_generation(
                output=f"Response {i}",
                constraints={'temperature': 0.7},
                performance_score=0.8
            )
        
        with pytest.raises(ValueError, match="at least 6 generations"):
            auditor.audit()
    
    def test_audit_no_bias(self):
        """Test audit when no bias is present."""
        auditor = BiasAuditor()
        
        # Record stable performance with consistent constraints
        for i in range(12):
            auditor.record_generation(
                output=f"Response {i}" * 5,
                constraints={'temperature': 0.7, 'max_new_tokens': 100},
                performance_score=0.8 + np.random.randn() * 0.01
            )
        
        result = auditor.audit(time_periods=4)
        
        assert isinstance(result, BiasAuditResult)
        assert 0 <= result.psi_score
        assert 0 <= result.ccs_score <= 1
        assert -1 <= result.rho_pc_score <= 1
        
        # Should detect no bias (consistent constraints)
        assert result.ccs_score == 1.0  # Perfect consistency
        assert result.overall_bias is False or result.confidence < 0.5
    
    def test_audit_with_bias(self):
        """Test audit when bias is present."""
        auditor = BiasAuditor()
        
        # Simulate iterative tuning: increasing temp as performance improves
        temperatures = [0.5, 0.6, 0.7, 0.8, 0.9]
        
        for temp in temperatures:
            for _ in range(3):
                # Performance increases with temperature (correlation)
                perf = 0.5 + (temp - 0.5) * 0.8
                auditor.record_generation(
                    output=f"Response at temp {temp}" * 10,
                    constraints={'temperature': temp, 'max_new_tokens': 100},
                    performance_score=perf
                )
        
        result = auditor.audit(time_periods=5)
        
        # Should detect correlation and constraint variation
        assert result.ccs_score < 1.0  # Constraints are not consistent
        assert abs(result.rho_pc_score) > 0.5  # High correlation
        
        # Likely to detect bias
        # (may not always due to statistical variance, but correlation should be high)
    
    def test_clear_history(self):
        """Test clearing generation history."""
        auditor = BiasAuditor()
        
        for i in range(5):
            auditor.record_generation(
                output=f"Response {i}",
                constraints={'temperature': 0.7},
                performance_score=0.8
            )
        
        assert len(auditor.history) == 5
        
        auditor.clear_history()
        
        assert len(auditor.history) == 0
        assert auditor.total_generations == 5  # Counter not reset
    
    def test_get_statistics(self):
        """Test getting auditor statistics."""
        auditor = BiasAuditor()
        
        for i in range(10):
            auditor.record_generation(
                output=f"Response {i}",
                constraints={'temperature': 0.7},
                performance_score=0.8
            )
        
        auditor.audit(time_periods=3)
        
        stats = auditor.get_statistics()
        
        assert stats['total_generations'] == 10
        assert stats['total_audits'] == 1
        assert stats['history_size'] == 10
        assert 'thresholds' in stats
    
    def test_repr(self):
        """Test string representation."""
        auditor = BiasAuditor()
        
        for i in range(5):
            auditor.record_generation(
                output=f"Response {i}",
                constraints={'temperature': 0.7},
                performance_score=0.8
            )
        
        repr_str = repr(auditor)
        
        assert 'BiasAuditor' in repr_str
        assert 'history_size=5' in repr_str


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_create_auditor_default(self):
        """Test create_auditor with defaults."""
        auditor = create_auditor()
        
        assert isinstance(auditor, BiasAuditor)
        assert auditor.psi_threshold == 0.1
    
    def test_create_auditor_custom_thresholds(self):
        """Test create_auditor with custom thresholds."""
        auditor = create_auditor(
            thresholds={'psi': 0.05, 'ccs': 0.9, 'rho_pc': 0.4}
        )
        
        assert auditor.psi_threshold == 0.05
        assert auditor.ccs_threshold == 0.9
        assert auditor.rho_pc_threshold == 0.4
    
    def test_create_auditor_mixed_args(self):
        """Test create_auditor with mixed arguments."""
        auditor = create_auditor(
            thresholds={'psi': 0.05},
            auto_score=False
        )
        
        assert auditor.psi_threshold == 0.05
        assert auditor.ccs_threshold == 0.8  # Default
        assert auditor.auto_score is False


class TestIntegrationScenarios:
    """Integration tests simulating real-world scenarios."""
    
    def test_llm_evaluation_workflow(self):
        """Test complete LLM evaluation workflow."""
        auditor = BiasAuditor()
        
        # Simulate 3 rounds of evaluation with different settings
        settings = [
            {'temperature': 0.5, 'max_new_tokens': 50},
            {'temperature': 0.7, 'max_new_tokens': 100},
            {'temperature': 0.9, 'max_new_tokens': 150}
        ]
        
        for setting in settings:
            for i in range(5):
                auditor.record_generation(
                    output=f"Evaluation response {i}" * 10,
                    constraints=setting,
                    performance_score=0.7 + np.random.rand() * 0.2
                )
        
        result = auditor.audit(time_periods=3)
        
        assert result is not None
        assert isinstance(result.overall_bias, bool)
        assert 0 <= result.confidence <= 1
    
    def test_multiple_audits(self):
        """Test performing multiple audits."""
        auditor = BiasAuditor()
        
        # First batch
        for i in range(10):
            auditor.record_generation(
                output=f"Batch 1 response {i}",
                constraints={'temperature': 0.7},
                performance_score=0.8
            )
        
        result1 = auditor.audit(time_periods=3)
        
        # Add more data
        for i in range(10):
            auditor.record_generation(
                output=f"Batch 2 response {i}",
                constraints={'temperature': 0.8},
                performance_score=0.85
            )
        
        result2 = auditor.audit(time_periods=5)
        
        assert result1.metadata['num_generations'] == 10
        assert result2.metadata['num_generations'] == 20
        assert auditor.total_audits == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
