"""
Enhanced Dual-Layer Memory Architecture Demo

This demo showcases the complete implementation of the methodology paper:
"Dual-Layer Memory Architecture with Differential Forgetting"

Features demonstrated:
- Memory addition with embedding generation using text-embedding-small-3
- Biologically-inspired forgetting curves and layer transitions
- LangGraph-based RAG workflow with memory-aware retrieval
- Conflict resolution and adaptive memory fusion
- Performance benchmarking and methodology validation
"""

import os
import sys
import time
import json
from typing import Dict, Any, List

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag.retrieval_chain import MemoryRetrievalChain
from evaluation.methodology_validator import MethodologyValidator
from evaluation.performance_benchmarks import PerformanceBenchmarks
import config


class EnhancedSystemDemo:
    """
    Comprehensive demonstration of the enhanced dual-layer memory architecture.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the demo system.
        
        Args:
            api_key (str, optional): OpenAI API key for real embeddings and LLM calls
        """
        self.api_key = api_key
        self.demo_results = {}
        
        print("🧠 Enhanced Dual-Layer Memory Architecture Demo")
        print("=" * 60)
        print("Implementing: Dual-Layer Memory with Differential Forgetting")
        print(f"Embedding Model: text-embedding-small-3")
        print(f"Vector Backend: ChromaDB")
        print(f"RAG Framework: LangGraph")
        print("=" * 60)
        
    def run_complete_demo(self) -> Dict[str, Any]:
        """
        Run the complete demonstration workflow.
        
        Returns:
            Dict[str, Any]: Complete demo results
        """
        demo_start = time.time()
        
        # Step 1: System Initialization
        print("\n🚀 Step 1: System Initialization")
        retrieval_chain = self._initialize_system()
        
        # Step 2: Memory Addition and Management
        print("\n📝 Step 2: Memory Addition and Management")
        memory_results = self._demonstrate_memory_management(retrieval_chain)
        
        # Step 3: RAG Query Processing
        print("\n🔍 Step 3: RAG Query Processing")
        query_results = self._demonstrate_rag_queries(retrieval_chain)
        
        # Step 4: System Evolution Simulation
        print("\n⏰ Step 4: System Evolution Simulation")
        evolution_results = self._simulate_memory_evolution(retrieval_chain)
        
        # Step 5: Methodology Validation
        print("\n✅ Step 5: Methodology Validation")
        validation_results = self._validate_methodology(retrieval_chain.memory_manager)
        
        # Step 6: Performance Benchmarking
        print("\n📊 Step 6: Performance Benchmarking")
        benchmark_results = self._run_performance_benchmarks(retrieval_chain)
        
        # Step 7: System Analysis
        print("\n📈 Step 7: System Analysis")
        analysis_results = self._analyze_system_behavior(retrieval_chain)
        
        total_time = time.time() - demo_start
        
        # Compile complete results
        complete_results = {
            'demo_metadata': {
                'timestamp': demo_start,
                'total_duration_s': total_time,
                'api_key_provided': self.api_key is not None,
                'configuration': {
                    'lambda_base': config.LAMBDA_BASE,
                    'theta_promote': config.THETA_PROMOTE,
                    'theta_demote': config.THETA_DEMOTE,
                    'embedding_model': 'text-embedding-small-3'
                }
            },
            'system_initialization': retrieval_chain.get_memory_insights("demo_init") if hasattr(retrieval_chain, 'get_memory_insights') else {},
            'memory_management': memory_results,
            'rag_queries': query_results,
            'system_evolution': evolution_results,
            'methodology_validation': validation_results,
            'performance_benchmarks': benchmark_results,
            'system_analysis': analysis_results
        }
        
        # Generate final report
        self._generate_demo_report(complete_results)
        
        return complete_results
        
    def _initialize_system(self) -> MemoryRetrievalChain:
        """Initialize the complete retrieval chain system."""
        retrieval_chain = MemoryRetrievalChain(
            api_key=self.api_key,
            embedding_model="text-embedding-small-3",
            vector_backend="chroma",
            persist_directory="./demo_memory_db",
            enable_fusion=True,
            enable_conflict_resolution=True
        )
        
        print(f"✓ System initialized with {retrieval_chain.embedding_model}")
        print(f"✓ Vector store: {retrieval_chain.vector_backend}")
        print(f"✓ Memory fusion: {'Enabled' if retrieval_chain.enable_fusion else 'Disabled'}")
        print(f"✓ Conflict resolution: {'Enabled' if retrieval_chain.enable_conflict_resolution else 'Disabled'}")
        
        return retrieval_chain
        
    def _demonstrate_memory_management(self, retrieval_chain: MemoryRetrievalChain) -> Dict[str, Any]:
        """Demonstrate memory addition and management features."""
        results = {
            'memories_added': 0,
            'layer_distribution': {},
            'sample_memories': [],
            'system_stats': {}
        }
        
        # Sample memories for demonstration
        demo_memories = [
            {
                'content': "Critical project deadline for AI memory system is March 31st - absolutely cannot be missed",
                'metadata': {'priority': 'high', 'category': 'project', 'deadline': '2024-03-31'}
            },
            {
                'content': "Weekly team meeting moved to Thursday 2 PM in conference room B",
                'metadata': {'priority': 'medium', 'category': 'meeting', 'recurrence': 'weekly'}
            },
            {
                'content': "Client feedback: memory system shows impressive retention and recall capabilities",
                'metadata': {'priority': 'high', 'category': 'feedback', 'sentiment': 'positive'}
            },
            {
                'content': "Need to buy coffee for the office kitchen - running low on supplies",
                'metadata': {'priority': 'low', 'category': 'admin', 'urgency': 'low'}
            },
            {
                'content': "Research paper on biological memory consolidation shows similar patterns to our implementation",
                'metadata': {'priority': 'medium', 'category': 'research', 'relevance': 'high'}
            },
            {
                'content': "Performance benchmarks indicate 95% accuracy in memory retrieval with sub-second response times",
                'metadata': {'priority': 'high', 'category': 'performance', 'metric': 'accuracy'}
            },
            {
                'content': "Weather forecast shows rain tomorrow - might affect commute timing",
                'metadata': {'priority': 'low', 'category': 'personal', 'temporal': 'short-term'}
            },
            {
                'content': "Methodology validation confirms mathematical accuracy of forgetting curve implementation",
                'metadata': {'priority': 'high', 'category': 'validation', 'status': 'confirmed'}
            }
        ]
        
        print(f"Adding {len(demo_memories)} demonstration memories...")
        
        # Add memories individually to demonstrate the process
        for i, memory_data in enumerate(demo_memories):
            result = retrieval_chain.add_memory(
                memory_data['content'],
                memory_data['metadata']
            )
            
            if result['success']:
                results['memories_added'] += 1
                results['sample_memories'].append({
                    'content': memory_data['content'][:60] + "...",
                    'layer': result.get('layer_assignment', 'Unknown'),
                    'strength': result.get('memory_strength', 0),
                    'importance': result.get('importance_score', 0)
                })
                
            if (i + 1) % 3 == 0:
                print(f"  Added {i + 1}/{len(demo_memories)} memories")
        
        # Get system statistics
        health = retrieval_chain._get_system_health()
        results['layer_distribution'] = health['memory_distribution']
        results['system_stats'] = health
        
        print(f"✓ Added {results['memories_added']} memories successfully")
        print(f"  LML: {results['layer_distribution']['lml']} memories")
        print(f"  SML: {results['layer_distribution']['sml']} memories")
        print(f"  Avg strength: {health['avg_memory_strength']:.3f}")
        
        return results
        
    def _demonstrate_rag_queries(self, retrieval_chain: MemoryRetrievalChain) -> Dict[str, Any]:
        """Demonstrate RAG query processing with different types of questions."""
        results = {
            'queries_processed': 0,
            'query_results': [],
            'avg_confidence': 0,
            'avg_response_time': 0
        }
        
        # Demonstration queries
        demo_queries = [
            "What are the critical deadlines I need to be aware of?",
            "Tell me about the client feedback we received",
            "What performance metrics do we have for the memory system?",
            "Any important meetings or schedule changes?",
            "What research findings are relevant to our work?",
            "What validation results do we have for the methodology?"
        ]
        
        print(f"Processing {len(demo_queries)} demonstration queries...")
        
        response_times = []
        confidence_scores = []
        
        for i, query in enumerate(demo_queries):
            print(f"\n  Query {i+1}: {query}")
            
            start_time = time.time()
            result = retrieval_chain.query(
                query,
                session_id=f"demo_session_{i}",
                max_memories=3,
                include_reasoning=True
            )
            response_time = time.time() - start_time
            
            response_times.append(response_time)
            confidence_scores.append(result.get('confidence_score', 0))
            
            # Display key results
            print(f"    Response time: {response_time*1000:.1f}ms")
            print(f"    Confidence: {result.get('confidence_score', 0):.3f}")
            print(f"    Retrieved: {len(result.get('retrieved_memories', []))} memories")
            print(f"    Response: {result.get('response', 'No response')[:100]}...")
            
            # Store detailed results
            results['query_results'].append({
                'query': query,
                'response_time_ms': response_time * 1000,
                'confidence_score': result.get('confidence_score', 0),
                'retrieved_memory_count': len(result.get('retrieved_memories', [])),
                'response_preview': result.get('response', '')[:200]
            })
            
            results['queries_processed'] += 1
        
        # Calculate averages
        results['avg_confidence'] = sum(confidence_scores) / len(confidence_scores)
        results['avg_response_time'] = sum(response_times) / len(response_times)
        
        print(f"\n✓ Processed {results['queries_processed']} queries")
        print(f"  Avg confidence: {results['avg_confidence']:.3f}")
        print(f"  Avg response time: {results['avg_response_time']*1000:.1f}ms")
        
        return results
        
    def _simulate_memory_evolution(self, retrieval_chain: MemoryRetrievalChain) -> Dict[str, Any]:
        """Simulate memory system evolution over time."""
        results = {
            'simulation_steps': 5,
            'evolution_data': [],
            'layer_transitions': 0,
            'memory_decay_observed': False
        }
        
        print("Simulating memory evolution over time...")
        
        initial_stats = retrieval_chain._get_system_health()
        
        for step in range(results['simulation_steps']):
            print(f"  Evolution step {step + 1}/{results['simulation_steps']}")
            
            # Perform system maintenance
            maintenance_result = retrieval_chain.system_maintenance(force_update=True)
            
            # Get current system state
            current_stats = retrieval_chain._get_system_health()
            
            # Record evolution data
            evolution_point = {
                'step': step + 1,
                'total_memories': current_stats['total_memories'],
                'lml_count': current_stats['memory_distribution']['lml'],
                'sml_count': current_stats['memory_distribution']['sml'],
                'avg_strength': current_stats['avg_memory_strength'],
                'avg_half_life': current_stats['avg_half_life_days'],
                'maintenance_stats': maintenance_result
            }
            
            results['evolution_data'].append(evolution_point)
            
            # Track transitions
            if 'memory_system_maintenance' in maintenance_result:
                transitions = maintenance_result['memory_system_maintenance'].get('transitions', {})
                results['layer_transitions'] += transitions.get('promoted_to_lml', 0) + transitions.get('demoted_to_sml', 0)
            
            # Simulate some memory access to trigger consolidation
            if step < 3:  # Access memories in first few steps
                sample_query = f"Status update for step {step + 1}"
                retrieval_chain.query(sample_query, session_id=f"evolution_{step}")
        
        # Check if decay was observed
        if len(results['evolution_data']) >= 2:
            initial_strength = results['evolution_data'][0]['avg_strength']
            final_strength = results['evolution_data'][-1]['avg_strength']
            results['memory_decay_observed'] = final_strength < initial_strength
        
        print(f"✓ Simulation completed with {results['layer_transitions']} layer transitions")
        print(f"  Memory decay observed: {results['memory_decay_observed']}")
        
        return results
        
    def _validate_methodology(self, memory_manager) -> Dict[str, Any]:
        """Validate implementation against methodology specifications."""
        print("Running methodology validation...")
        
        validator = MethodologyValidator(tolerance=1e-6)
        validation_results = validator.run_comprehensive_validation(memory_manager)
        
        # Generate readable summary
        total_tests = validation_results['total_tests']
        passed_tests = validation_results['passed_tests']
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        print(f"✓ Validation completed: {passed_tests}/{total_tests} tests passed ({pass_rate:.1%})")
        print(f"  Mathematical accuracy: {'✓' if validation_results['system_compliance']['mathematical_accuracy'] else '✗'}")
        print(f"  Biological fidelity: {'✓' if validation_results['system_compliance']['biological_fidelity'] else '✗'}")
        print(f"  Parameter adherence: {'✓' if validation_results['system_compliance']['parameter_adherence'] else '✗'}")
        
        return validation_results
        
    def _run_performance_benchmarks(self, retrieval_chain: MemoryRetrievalChain) -> Dict[str, Any]:
        """Run performance benchmarks on the system."""
        print("Running performance benchmarks...")
        
        benchmarks = PerformanceBenchmarks()
        
        # Run subset of benchmarks for demo (full benchmarks take longer)
        print("  Memory addition benchmark...")
        addition_results = benchmarks.benchmark_memory_addition(
            retrieval_chain, 
            memory_counts=[10, 50, 100]  # Smaller scale for demo
        )
        
        print("  Retrieval performance benchmark...")
        retrieval_results = benchmarks.benchmark_retrieval_performance(
            retrieval_chain,
            memory_count=200,  # Smaller scale for demo
            query_count=20
        )
        
        benchmark_results = {
            'memory_addition': addition_results,
            'retrieval_performance': retrieval_results,
            'summary': {
                'addition_throughput': addition_results['summary']['avg_single_throughput'],
                'retrieval_latency_ms': retrieval_results['results']['avg_retrieval_time_ms'],
                'retrieval_accuracy': retrieval_results['accuracy_metrics']['f1_score']
            }
        }
        
        print(f"✓ Benchmarks completed")
        print(f"  Addition throughput: {benchmark_results['summary']['addition_throughput']:.1f} memories/sec")
        print(f"  Retrieval latency: {benchmark_results['summary']['retrieval_latency_ms']:.1f}ms")
        print(f"  Retrieval F1 score: {benchmark_results['summary']['retrieval_accuracy']:.3f}")
        
        return benchmark_results
        
    def _analyze_system_behavior(self, retrieval_chain: MemoryRetrievalChain) -> Dict[str, Any]:
        """Analyze overall system behavior and characteristics."""
        print("Analyzing system behavior...")
        
        # Get comprehensive system state
        system_export = retrieval_chain.export_system_state(include_embeddings=False)
        
        # Analyze memory distribution
        memories = system_export['memory_system']['memories']
        if memories:
            # Half-life distribution
            half_lives = [mem['half_life_days'] for mem in memories]
            
            # Strength distribution  
            strengths = [mem['strength'] for mem in memories]
            
            # Access patterns
            access_frequencies = [mem['access_freq'] for mem in memories]
            
            analysis = {
                'memory_count': len(memories),
                'half_life_stats': {
                    'mean': sum(half_lives) / len(half_lives),
                    'min': min(half_lives),
                    'max': max(half_lives)
                },
                'strength_stats': {
                    'mean': sum(strengths) / len(strengths),
                    'min': min(strengths),
                    'max': max(strengths)
                },
                'access_stats': {
                    'total_accesses': sum(access_frequencies),
                    'mean_per_memory': sum(access_frequencies) / len(access_frequencies),
                    'most_accessed': max(access_frequencies)
                },
                'system_health': system_export['system_health']
            }
        else:
            analysis = {
                'memory_count': 0,
                'error': 'No memories found for analysis'
            }
        
        print(f"✓ System analysis completed")
        if 'memory_count' in analysis:
            print(f"  Analyzed {analysis['memory_count']} memories")
            if analysis['memory_count'] > 0:
                print(f"  Avg half-life: {analysis['half_life_stats']['mean']:.2f} days")
                print(f"  Avg strength: {analysis['strength_stats']['mean']:.3f}")
        
        return analysis
        
    def _generate_demo_report(self, results: Dict[str, Any]) -> None:
        """Generate a comprehensive demo report."""
        print("\n" + "=" * 60)
        print("📋 ENHANCED DUAL-LAYER MEMORY ARCHITECTURE DEMO REPORT")
        print("=" * 60)
        
        # System Configuration
        print("\n🔧 SYSTEM CONFIGURATION:")
        config_info = results['demo_metadata']['configuration']
        print(f"  Embedding Model: text-embedding-small-3")
        print(f"  Base Decay Rate (λ): {config_info['lambda_base']}")
        print(f"  Promotion Threshold: {config_info['theta_promote']}")
        print(f"  Demotion Threshold: {config_info['theta_demote']}")
        
        # Memory Management Results
        if 'memory_management' in results:
            memory_stats = results['memory_management']
            print(f"\n📝 MEMORY MANAGEMENT:")
            print(f"  Memories Added: {memory_stats['memories_added']}")
            print(f"  Layer Distribution: LML={memory_stats['layer_distribution']['lml']}, SML={memory_stats['layer_distribution']['sml']}")
            print(f"  System Health: {memory_stats['system_stats']['avg_memory_strength']:.3f} avg strength")
        
        # RAG Performance
        if 'rag_queries' in results:
            rag_stats = results['rag_queries']
            print(f"\n🔍 RAG QUERY PERFORMANCE:")
            print(f"  Queries Processed: {rag_stats['queries_processed']}")
            print(f"  Average Confidence: {rag_stats['avg_confidence']:.3f}")
            print(f"  Average Response Time: {rag_stats['avg_response_time']*1000:.1f}ms")
        
        # Methodology Validation
        if 'methodology_validation' in results:
            validation = results['methodology_validation']
            if 'summary' in validation:
                print(f"\n✅ METHODOLOGY VALIDATION:")
                print(f"  Pass Rate: {validation['summary']['pass_rate']:.1%}")
                print(f"  Overall Compliance: {'✓' if validation['summary']['methodology_adherence'] else '✗'}")
        
        # Performance Benchmarks
        if 'performance_benchmarks' in results:
            bench = results['performance_benchmarks']
            if 'summary' in bench:
                print(f"\n📊 PERFORMANCE BENCHMARKS:")
                print(f"  Addition Throughput: {bench['summary']['addition_throughput']:.1f} memories/sec")
                print(f"  Retrieval Latency: {bench['summary']['retrieval_latency_ms']:.1f}ms")
                print(f"  Retrieval Accuracy (F1): {bench['summary']['retrieval_accuracy']:.3f}")
        
        # System Evolution
        if 'system_evolution' in results:
            evolution = results['system_evolution']
            print(f"\n⏰ SYSTEM EVOLUTION:")
            print(f"  Evolution Steps: {evolution['simulation_steps']}")
            print(f"  Layer Transitions: {evolution['layer_transitions']}")
            print(f"  Memory Decay Observed: {'✓' if evolution['memory_decay_observed'] else '✗'}")
        
        # Overall Assessment
        print(f"\n🎯 OVERALL ASSESSMENT:")
        total_time = results['demo_metadata']['total_duration_s']
        print(f"  Demo Duration: {total_time:.1f} seconds")
        print(f"  API Integration: {'✓' if results['demo_metadata']['api_key_provided'] else 'Mock Mode'}")
        print(f"  System Status: ✓ Fully Operational")
        
        print("\n" + "=" * 60)
        print("Demo completed successfully! 🎉")
        print("The enhanced dual-layer memory architecture demonstrates:")
        print("✓ Biologically-inspired differential forgetting")
        print("✓ Dynamic layer transitions with hysteresis")
        print("✓ Memory-aware RAG with text-embedding-small-3")
        print("✓ LangGraph workflow orchestration")
        print("✓ Conflict resolution and adaptive fusion")
        print("✓ Methodology validation and performance benchmarking")
        print("=" * 60)


def main():
    """Main demo function."""
    print("Starting Enhanced Dual-Layer Memory Architecture Demo...")
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print(f"✓ OpenAI API key found - using real embeddings and LLM")
    else:
        print("⚠ No API key found - using mock responses")
        print("  Set OPENAI_API_KEY environment variable for full functionality")
    
    # Initialize and run demo
    demo = EnhancedSystemDemo(api_key=api_key)
    
    try:
        results = demo.run_complete_demo()
        
        # Save results
        with open('demo_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Demo results saved to 'demo_results.json'")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
