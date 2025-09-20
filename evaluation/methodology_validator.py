"""
Methodology Validator

This module validates that the implementation correctly follows the mathematical
formulations and biological principles described in the methodology paper.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import math
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.memory_item import MemoryItem
from core.enhanced_memory_manager import EnhancedMemoryManager
import config


class MethodologyValidator:
    """
    Validates implementation against methodology paper specifications.
    
    This validator checks:
    - Mathematical accuracy of forgetting curves
    - Half-life calculations
    - Importance scoring formulations
    - Layer transition logic
    - Biological parameter adherence
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize the methodology validator.
        
        Args:
            tolerance (float): Numerical tolerance for comparisons
        """
        self.tolerance = tolerance
        self.validation_results = {}
        
    def validate_forgetting_curve(self, memory: MemoryItem, time_elapsed_days: float) -> Dict[str, Any]:
        """
        Validate forgetting curve implementation against methodology formula.
        
        v_i(t) = v_i(0) * exp(-λ_i * (t - τ_i)^β_i)
        
        Args:
            memory (MemoryItem): Memory to validate
            time_elapsed_days (float): Time elapsed in days
            
        Returns:
            Dict[str, Any]: Validation results
        """
        # Ensure decay parameters are set
        if memory.lambda_i is None or memory.beta_i is None:
            memory.update_decay_parameters()
            
        # Calculate expected value using methodology formula
        initial_strength = 1.0  # v_i(0)
        expected_strength = initial_strength * math.exp(
            -memory.lambda_i * (time_elapsed_days ** memory.beta_i)
        )
        
        # Calculate actual value using implementation
        # Store original timestamp to restore later
        original_timestamp = memory.creation_timestamp
        
        # Set timestamp to simulate elapsed time
        memory.creation_timestamp = time.time() - (time_elapsed_days * 86400)
        actual_strength = memory.apply_biological_decay()
        
        # Restore original timestamp
        memory.creation_timestamp = original_timestamp
        
        # Compare results
        difference = abs(expected_strength - actual_strength)
        is_valid = difference < self.tolerance
        
        return {
            'test_name': 'forgetting_curve',
            'valid': is_valid,
            'expected_strength': expected_strength,
            'actual_strength': actual_strength,
            'difference': difference,
            'tolerance': self.tolerance,
            'time_elapsed_days': time_elapsed_days,
            'lambda_i': memory.lambda_i,
            'beta_i': memory.beta_i,
            'formula': 'v_i(t) = v_i(0) * exp(-λ_i * (t - τ_i)^β_i)'
        }
        
    def validate_half_life_calculation(self, memory: MemoryItem) -> Dict[str, Any]:
        """
        Validate half-life calculation against methodology formula.
        
        t_1/2(i) = (ln(2)/λ_i)^(1/β_i)
        
        Args:
            memory (MemoryItem): Memory to validate
            
        Returns:
            Dict[str, Any]: Validation results
        """
        # Ensure decay parameters are set
        if memory.lambda_i is None or memory.beta_i is None:
            memory.update_decay_parameters()
            
        # Calculate expected half-life using methodology formula
        expected_half_life = (math.log(2) / memory.lambda_i) ** (1 / memory.beta_i)
        
        # Calculate actual half-life using implementation
        actual_half_life = memory.get_half_life()
        
        # Compare results
        difference = abs(expected_half_life - actual_half_life)
        is_valid = difference < self.tolerance
        
        # Check against reference values for I_i(t)=0
        layer_assignment = memory.layer_assignment
        if layer_assignment == 'LML':
            reference_half_life = config.EXPECTED_LML_HALF_LIFE_DAYS
        elif layer_assignment == 'SML':
            reference_half_life = config.EXPECTED_SML_HALF_LIFE_DAYS
        else:
            reference_half_life = None
            
        return {
            'test_name': 'half_life_calculation',
            'valid': is_valid,
            'expected_half_life': expected_half_life,
            'actual_half_life': actual_half_life,
            'difference': difference,
            'tolerance': self.tolerance,
            'lambda_i': memory.lambda_i,
            'beta_i': memory.beta_i,
            'layer_assignment': layer_assignment,
            'reference_half_life': reference_half_life,
            'formula': 't_1/2(i) = (ln(2)/λ_i)^(1/β_i)'
        }
        
    def validate_importance_scoring(self, memory: MemoryItem, query_context: List[float]) -> Dict[str, Any]:
        """
        Validate importance scoring against methodology formula.
        
        I_i(t) = α·rel(c_i, Q_t) + β·f_i/(1+f_i) + γ·recency(τ_i, t)
        
        Args:
            memory (MemoryItem): Memory to validate
            query_context (List[float]): Query context for relevance calculation
            
        Returns:
            Dict[str, Any]: Validation results
        """
        current_time = time.time()
        
        # Calculate components manually using methodology formulas
        # Semantic relevance component
        relevance = memory._cosine_similarity(memory.content_embedding, query_context)
        
        # Frequency component with saturation: f_i/(1+f_i)
        frequency_score = memory.time_decayed_access_rate / (1 + memory.time_decayed_access_rate)
        
        # Recency component: exp(-δ(t - τ_i))
        age_days = (current_time - memory.creation_timestamp) / 86400
        recency_score = math.exp(-config.DELTA_RECENCY * age_days)
        
        # Expected importance score
        expected_importance = (config.ALPHA * relevance + 
                             config.BETA * frequency_score + 
                             config.GAMMA * recency_score)
        expected_importance = max(0.0, min(1.0, expected_importance))
        
        # Actual importance score from implementation
        actual_importance = memory.calculate_importance(
            query_context=query_context,
            alpha=config.ALPHA,
            beta=config.BETA,
            gamma=config.GAMMA,
            current_time=current_time
        )
        
        # Compare results
        difference = abs(expected_importance - actual_importance)
        is_valid = difference < self.tolerance
        
        return {
            'test_name': 'importance_scoring',
            'valid': is_valid,
            'expected_importance': expected_importance,
            'actual_importance': actual_importance,
            'difference': difference,
            'tolerance': self.tolerance,
            'components': {
                'relevance': relevance,
                'frequency_score': frequency_score,
                'recency_score': recency_score,
                'age_days': age_days
            },
            'weights': {
                'alpha': config.ALPHA,
                'beta': config.BETA,
                'gamma': config.GAMMA
            },
            'formula': 'I_i(t) = α·rel(c_i, Q_t) + β·f_i/(1+f_i) + γ·recency(τ_i, t)'
        }
        
    def validate_consolidation_mechanics(self, memory: MemoryItem, 
                                       access_times: List[float],
                                       W_days: int = 7) -> Dict[str, Any]:
        """
        Validate memory consolidation mechanics.
        
        v_i(t^+) = v_i(t) + Δv * (1 - v_i(t)) * exp(-n_i/N)
        
        Args:
            memory (MemoryItem): Memory to validate
            access_times (List[float]): List of access timestamps
            W_days (int): Window for access counting
            
        Returns:
            Dict[str, Any]: Validation results
        """
        original_strength = memory.memory_strength
        current_time = time.time()
        
        # Count accesses within window
        window_start = current_time - (W_days * 86400)
        n_i = sum(1 for t in access_times if t >= window_start)
        
        # Calculate expected consolidation using methodology formula
        delta_v = config.DELTA_V
        N = config.N_SPACING
        
        expected_reinforcement = delta_v * (1 - original_strength) * math.exp(-n_i / N)
        expected_new_strength = min(1.0, original_strength + expected_reinforcement)
        
        # Apply consolidation using implementation
        # Set up memory state
        memory.access_timestamps = access_times.copy()
        memory.memory_strength = original_strength
        
        # Apply access with consolidation
        memory.access(
            current_time=current_time,
            delta_v=delta_v,
            N=N,
            W=W_days
        )
        
        actual_new_strength = memory.memory_strength
        
        # Compare results
        difference = abs(expected_new_strength - actual_new_strength)
        is_valid = difference < self.tolerance
        
        return {
            'test_name': 'consolidation_mechanics',
            'valid': is_valid,
            'original_strength': original_strength,
            'expected_new_strength': expected_new_strength,
            'actual_new_strength': actual_new_strength,
            'difference': difference,
            'tolerance': self.tolerance,
            'access_count_in_window': n_i,
            'window_days': W_days,
            'parameters': {
                'delta_v': delta_v,
                'N': N
            },
            'formula': 'v_i(t^+) = v_i(t) + Δv * (1 - v_i(t)) * exp(-n_i/N)'
        }
        
    def validate_decay_rate_adaptation(self, memory: MemoryItem) -> Dict[str, Any]:
        """
        Validate decay rate adaptation to importance.
        
        λ_i = λ_base * exp(-μ * I_i(t))
        
        Args:
            memory (MemoryItem): Memory to validate
            
        Returns:
            Dict[str, Any]: Validation results
        """
        # Calculate current importance
        importance = memory.calculate_importance()
        
        # Expected decay rate using methodology formula
        expected_lambda = config.LAMBDA_BASE * math.exp(-config.MU * importance)
        
        # Update decay parameters in implementation
        memory.update_decay_parameters(lambda_base=config.LAMBDA_BASE, mu=config.MU)
        actual_lambda = memory.lambda_i
        
        # Compare results
        difference = abs(expected_lambda - actual_lambda)
        is_valid = difference < self.tolerance
        
        return {
            'test_name': 'decay_rate_adaptation',
            'valid': is_valid,
            'importance_score': importance,
            'expected_lambda': expected_lambda,
            'actual_lambda': actual_lambda,
            'difference': difference,
            'tolerance': self.tolerance,
            'parameters': {
                'lambda_base': config.LAMBDA_BASE,
                'mu': config.MU
            },
            'formula': 'λ_i = λ_base * exp(-μ * I_i(t))'
        }
        
    def validate_layer_shape_parameters(self, memory: MemoryItem) -> Dict[str, Any]:
        """
        Validate layer-specific shape parameters.
        
        β_i = 0.8 for LML (sub-linear decay)
        β_i = 1.2 for SML (super-linear decay)
        
        Args:
            memory (MemoryItem): Memory to validate
            
        Returns:
            Dict[str, Any]: Validation results
        """
        # Update decay parameters to ensure beta_i is set
        memory.update_decay_parameters()
        
        layer = memory.layer_assignment
        actual_beta = memory.beta_i
        
        if layer == 'LML':
            expected_beta = config.BETA_LML
            description = "sub-linear decay"
        elif layer == 'SML':
            expected_beta = config.BETA_SML
            description = "super-linear decay"
        else:
            expected_beta = 1.0  # Default exponential
            description = "exponential decay"
            
        difference = abs(expected_beta - actual_beta)
        is_valid = difference < self.tolerance
        
        return {
            'test_name': 'layer_shape_parameters',
            'valid': is_valid,
            'layer_assignment': layer,
            'expected_beta': expected_beta,
            'actual_beta': actual_beta,
            'difference': difference,
            'tolerance': self.tolerance,
            'description': description,
            'specification': 'β_i = 0.8 for LML, β_i = 1.2 for SML'
        }
        
    def run_comprehensive_validation(self, memory_manager: EnhancedMemoryManager) -> Dict[str, Any]:
        """
        Run comprehensive validation of the entire system.
        
        Args:
            memory_manager (EnhancedMemoryManager): System to validate
            
        Returns:
            Dict[str, Any]: Complete validation results
        """
        validation_results = {
            'timestamp': time.time(),
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_results': [],
            'system_compliance': {
                'mathematical_accuracy': True,
                'biological_fidelity': True,
                'parameter_adherence': True
            },
            'summary': {}
        }
        
        # Get test memories from the system
        all_memories = memory_manager.dual_layer_memory.get_all_memories()
        
        if not all_memories:
            # Create test memories if none exist
            test_contents = [
                "Test memory for LML validation",
                "Test memory for SML validation", 
                "Test memory for importance scoring"
            ]
            
            for content in test_contents:
                memory_manager.add_memory(content, {"test": True})
                
            all_memories = memory_manager.dual_layer_memory.get_all_memories()
        
        # Test each validation function
        for memory in all_memories[:3]:  # Test first 3 memories
            # Generate test query context
            test_context = [np.random.random() for _ in range(len(memory.content_embedding))]
            
            # Run all validation tests
            tests = [
                self.validate_forgetting_curve(memory, 5.0),  # 5 days elapsed
                self.validate_half_life_calculation(memory),
                self.validate_importance_scoring(memory, test_context),
                self.validate_consolidation_mechanics(memory, [time.time() - i*3600 for i in range(5)]),
                self.validate_decay_rate_adaptation(memory),
                self.validate_layer_shape_parameters(memory)
            ]
            
            for test_result in tests:
                validation_results['test_results'].append(test_result)
                validation_results['total_tests'] += 1
                
                if test_result['valid']:
                    validation_results['passed_tests'] += 1
                else:
                    validation_results['failed_tests'] += 1
                    
                    # Update compliance flags
                    if 'curve' in test_result['test_name'] or 'half_life' in test_result['test_name']:
                        validation_results['system_compliance']['mathematical_accuracy'] = False
                    if 'consolidation' in test_result['test_name'] or 'layer_shape' in test_result['test_name']:
                        validation_results['system_compliance']['biological_fidelity'] = False
                    if 'decay_rate' in test_result['test_name'] or 'importance' in test_result['test_name']:
                        validation_results['system_compliance']['parameter_adherence'] = False
        
        # Calculate summary statistics
        if validation_results['total_tests'] > 0:
            pass_rate = validation_results['passed_tests'] / validation_results['total_tests']
            validation_results['summary'] = {
                'pass_rate': pass_rate,
                'overall_compliance': pass_rate >= 0.95,  # 95% pass rate required
                'critical_failures': validation_results['failed_tests'],
                'methodology_adherence': all(validation_results['system_compliance'].values())
            }
        
        return validation_results
        
    def generate_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """
        Generate a human-readable validation report.
        
        Args:
            validation_results (Dict[str, Any]): Validation results from run_comprehensive_validation
            
        Returns:
            str: Formatted validation report
        """
        report_lines = [
            "=" * 60,
            "METHODOLOGY VALIDATION REPORT",
            "=" * 60,
            f"Validation Time: {time.ctime(validation_results['timestamp'])}",
            f"Total Tests: {validation_results['total_tests']}",
            f"Passed: {validation_results['passed_tests']}",
            f"Failed: {validation_results['failed_tests']}",
            f"Pass Rate: {validation_results['summary']['pass_rate']:.1%}",
            "",
            "COMPLIANCE STATUS:",
            f"  Mathematical Accuracy: {'✓' if validation_results['system_compliance']['mathematical_accuracy'] else '✗'}",
            f"  Biological Fidelity: {'✓' if validation_results['system_compliance']['biological_fidelity'] else '✗'}",
            f"  Parameter Adherence: {'✓' if validation_results['system_compliance']['parameter_adherence'] else '✗'}",
            "",
            "DETAILED TEST RESULTS:",
            "-" * 40
        ]
        
        # Group results by test type
        test_groups = {}
        for result in validation_results['test_results']:
            test_name = result['test_name']
            if test_name not in test_groups:
                test_groups[test_name] = []
            test_groups[test_name].append(result)
        
        for test_name, results in test_groups.items():
            passed = sum(1 for r in results if r['valid'])
            total = len(results)
            
            report_lines.extend([
                f"\n{test_name.replace('_', ' ').title()}:",
                f"  Status: {passed}/{total} passed ({'✓' if passed == total else '✗'})"
            ])
            
            # Show details for failed tests
            for result in results:
                if not result['valid']:
                    report_lines.append(f"  FAILED: {result.get('formula', 'N/A')}")
                    report_lines.append(f"    Expected: {result.get('expected_strength', result.get('expected_importance', result.get('expected_lambda', 'N/A')))}")
                    report_lines.append(f"    Actual: {result.get('actual_strength', result.get('actual_importance', result.get('actual_lambda', 'N/A')))}")
                    report_lines.append(f"    Difference: {result['difference']:.2e}")
        
        report_lines.extend([
            "",
            "=" * 60,
            f"OVERALL RESULT: {'COMPLIANT' if validation_results['summary']['methodology_adherence'] else 'NON-COMPLIANT'}",
            "=" * 60
        ])
        
        return "\n".join(report_lines)
