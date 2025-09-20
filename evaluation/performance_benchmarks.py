"""
Performance Benchmarks for Enhanced Dual-Layer Memory Architecture

This module provides comprehensive performance testing and benchmarking tools
to evaluate the efficiency, scalability, and quality of the memory system
under various workloads and conditions.
"""

from typing import List, Dict, Any, Optional, Callable
import time
import random
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.enhanced_memory_manager import EnhancedMemoryManager
from rag.retrieval_chain import MemoryRetrievalChain
import config


class PerformanceBenchmarks:
    """
    Comprehensive performance benchmarking suite for the memory system.
    
    Tests include:
    - Memory addition throughput
    - Retrieval latency and accuracy
    - System scalability under load
    - Memory decay performance
    - Conflict resolution efficiency
    - Fusion operation benchmarks
    """
    
    def __init__(self):
        """Initialize performance benchmarking suite."""
        self.benchmark_results = {}
        self.test_data_cache = {}
        
    def generate_test_memories(self, count: int, content_template: str = "Test memory {i}: {content}") -> List[Dict[str, Any]]:
        """
        Generate test memories for benchmarking.
        
        Args:
            count (int): Number of memories to generate
            content_template (str): Template for memory content
            
        Returns:
            List[Dict[str, Any]]: Generated test memories
        """
        cache_key = f"memories_{count}_{hash(content_template)}"
        if cache_key in self.test_data_cache:
            return self.test_data_cache[cache_key]
        
        # Sample content variations
        content_types = [
            "project deadline for {topic} due next week",
            "meeting with {person} scheduled at {time}",
            "important decision about {topic} needs consideration", 
            "team update on {topic} showing good progress",
            "client feedback on {topic} was very positive",
            "budget approval for {topic} still pending",
            "technical issue with {topic} has been resolved",
            "research findings on {topic} are promising"
        ]
        
        topics = ["AI development", "memory system", "user interface", "data analysis", 
                 "performance optimization", "testing framework", "documentation",
                 "security review", "integration testing", "deployment strategy"]
        
        people = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry"]
        times = ["9 AM", "2 PM", "10 AM", "3 PM", "11 AM", "4 PM", "1 PM", "5 PM"]
        
        memories = []
        for i in range(count):
            template = random.choice(content_types)
            content = template.format(
                topic=random.choice(topics),
                person=random.choice(people), 
                time=random.choice(times)
            )
            
            # Vary importance based on content keywords
            importance_keywords = {"deadline": 0.9, "important": 0.8, "urgent": 0.9, 
                                 "meeting": 0.7, "feedback": 0.6, "update": 0.4}
            importance = 0.5
            for keyword, score in importance_keywords.items():
                if keyword in content.lower():
                    importance = max(importance, score)
            
            memories.append({
                'content': content_template.format(i=i, content=content),
                'metadata': {
                    'test_id': i,
                    'importance_hint': importance,
                    'category': random.choice(['work', 'project', 'meeting', 'admin']),
                    'priority': random.choice(['high', 'medium', 'low'])
                }
            })
        
        self.test_data_cache[cache_key] = memories
        return memories
        
    def benchmark_memory_addition(self, retrieval_chain: MemoryRetrievalChain,
                                 memory_counts: List[int] = [10, 50, 100, 500, 1000]) -> Dict[str, Any]:
        """
        Benchmark memory addition performance across different scales.
        
        Args:
            retrieval_chain (MemoryRetrievalChain): System to benchmark
            memory_counts (List[int]): Different memory counts to test
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        results = {
            'test_name': 'memory_addition_benchmark',
            'timestamp': time.time(),
            'results_by_count': {},
            'summary': {}
        }
        
        for count in memory_counts:
            print(f"Benchmarking memory addition: {count} memories...")
            
            # Generate test memories
            test_memories = self.generate_test_memories(count)
            
            # Clear system for clean test
            retrieval_chain.clear_all_data(confirm=True)
            
            # Benchmark single additions
            single_times = []
            start_time = time.time()
            
            for memory_data in test_memories:
                add_start = time.time()
                result = retrieval_chain.add_memory(
                    memory_data['content'], 
                    memory_data['metadata']
                )
                add_end = time.time()
                
                if result['success']:
                    single_times.append(add_end - add_start)
                    
            total_time = time.time() - start_time
            
            # Benchmark batch addition
            retrieval_chain.clear_all_data(confirm=True)
            batch_start = time.time()
            batch_result = retrieval_chain.add_memory_batch(test_memories)
            batch_time = time.time() - batch_start
            
            # Calculate statistics
            if single_times:
                avg_single_time = np.mean(single_times)
                std_single_time = np.std(single_times)
                throughput_single = len(single_times) / total_time
            else:
                avg_single_time = std_single_time = throughput_single = 0
                
            throughput_batch = batch_result['successful_additions'] / batch_time if batch_time > 0 else 0
            
            results['results_by_count'][count] = {
                'single_addition': {
                    'avg_time_ms': avg_single_time * 1000,
                    'std_time_ms': std_single_time * 1000,
                    'total_time_s': total_time,
                    'throughput_per_sec': throughput_single,
                    'successful_additions': len(single_times)
                },
                'batch_addition': {
                    'total_time_s': batch_time,
                    'throughput_per_sec': throughput_batch,
                    'successful_additions': batch_result['successful_additions'],
                    'failed_additions': batch_result['failed_additions']
                },
                'efficiency_gain': throughput_batch / throughput_single if throughput_single > 0 else 0
            }
        
        # Calculate summary statistics
        avg_single_throughput = np.mean([r['single_addition']['throughput_per_sec'] 
                                       for r in results['results_by_count'].values()])
        avg_batch_throughput = np.mean([r['batch_addition']['throughput_per_sec'] 
                                      for r in results['results_by_count'].values()])
        
        results['summary'] = {
            'avg_single_throughput': avg_single_throughput,
            'avg_batch_throughput': avg_batch_throughput,
            'avg_efficiency_gain': avg_batch_throughput / avg_single_throughput if avg_single_throughput > 0 else 0,
            'scalability_assessment': self._assess_scalability(results['results_by_count'])
        }
        
        return results
        
    def benchmark_retrieval_performance(self, retrieval_chain: MemoryRetrievalChain,
                                      memory_count: int = 1000,
                                      query_count: int = 100) -> Dict[str, Any]:
        """
        Benchmark retrieval performance and accuracy.
        
        Args:
            retrieval_chain (MemoryRetrievalChain): System to benchmark
            memory_count (int): Number of memories to add for testing
            query_count (int): Number of queries to test
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        results = {
            'test_name': 'retrieval_performance_benchmark',
            'timestamp': time.time(),
            'setup': {
                'memory_count': memory_count,
                'query_count': query_count
            },
            'results': {},
            'accuracy_metrics': {}
        }
        
        print(f"Setting up retrieval benchmark with {memory_count} memories...")
        
        # Setup test data
        test_memories = self.generate_test_memories(memory_count)
        retrieval_chain.clear_all_data(confirm=True)
        
        # Add memories in batch for efficiency
        batch_result = retrieval_chain.add_memory_batch(test_memories)
        print(f"Added {batch_result['successful_additions']} memories successfully")
        
        # Generate test queries based on memory content
        test_queries = self._generate_test_queries(test_memories, query_count)
        
        # Benchmark retrieval performance
        retrieval_times = []
        confidence_scores = []
        retrieved_memory_counts = []
        
        for i, query_data in enumerate(test_queries):
            if i % 20 == 0:
                print(f"Processing query {i+1}/{query_count}")
                
            query_start = time.time()
            result = retrieval_chain.query(
                query_data['query'],
                session_id=f"bench_{i}",
                max_memories=5,
                include_reasoning=False
            )
            query_time = time.time() - query_start
            
            retrieval_times.append(query_time)
            confidence_scores.append(result.get('confidence_score', 0))
            retrieved_memory_counts.append(len(result.get('retrieved_memories', [])))
        
        # Calculate performance metrics
        results['results'] = {
            'avg_retrieval_time_ms': np.mean(retrieval_times) * 1000,
            'std_retrieval_time_ms': np.std(retrieval_times) * 1000,
            'min_retrieval_time_ms': np.min(retrieval_times) * 1000,
            'max_retrieval_time_ms': np.max(retrieval_times) * 1000,
            'throughput_queries_per_sec': len(retrieval_times) / np.sum(retrieval_times),
            'avg_confidence_score': np.mean(confidence_scores),
            'std_confidence_score': np.std(confidence_scores),
            'avg_retrieved_memories': np.mean(retrieved_memory_counts),
            'high_confidence_rate': np.mean([1 for c in confidence_scores if c >= 0.6])
        }
        
        # Assess retrieval accuracy (simplified - checks if relevant memories are retrieved)
        accuracy_results = self._assess_retrieval_accuracy(test_queries, retrieval_chain)
        results['accuracy_metrics'] = accuracy_results
        
        return results
        
    def benchmark_concurrent_performance(self, retrieval_chain: MemoryRetrievalChain,
                                       num_threads: List[int] = [1, 2, 4, 8],
                                       operations_per_thread: int = 50) -> Dict[str, Any]:
        """
        Benchmark concurrent access performance.
        
        Args:
            retrieval_chain (MemoryRetrievalChain): System to benchmark
            num_threads (List[int]): Different thread counts to test
            operations_per_thread (int): Operations per thread
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        results = {
            'test_name': 'concurrent_performance_benchmark',
            'timestamp': time.time(),
            'results_by_thread_count': {},
            'summary': {}
        }
        
        # Setup initial data
        setup_memories = self.generate_test_memories(500)
        retrieval_chain.clear_all_data(confirm=True)
        retrieval_chain.add_memory_batch(setup_memories)
        
        test_queries = self._generate_test_queries(setup_memories, operations_per_thread * max(num_threads))
        
        for thread_count in num_threads:
            print(f"Testing concurrent performance with {thread_count} threads...")
            
            completion_times = []
            errors = []
            
            def worker_function(thread_id: int):
                """Worker function for concurrent testing."""
                start_idx = thread_id * operations_per_thread
                end_idx = start_idx + operations_per_thread
                thread_queries = test_queries[start_idx:end_idx]
                
                thread_start = time.time()
                thread_errors = 0
                
                for query_data in thread_queries:
                    try:
                        # Mix of queries and memory additions
                        if random.random() < 0.7:  # 70% queries, 30% additions
                            result = retrieval_chain.query(
                                query_data['query'],
                                session_id=f"thread_{thread_id}",
                                include_reasoning=False
                            )
                        else:
                            # Add a new memory
                            new_content = f"Concurrent memory from thread {thread_id}: {random.choice(['task', 'note', 'reminder'])}"
                            result = retrieval_chain.add_memory(new_content, {'thread_id': thread_id})
                            
                        if not result.get('success', True):
                            thread_errors += 1
                            
                    except Exception as e:
                        thread_errors += 1
                        
                thread_time = time.time() - thread_start
                return {
                    'thread_id': thread_id,
                    'completion_time': thread_time,
                    'errors': thread_errors
                }
            
            # Run concurrent workers
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=thread_count) as executor:
                futures = [executor.submit(worker_function, i) for i in range(thread_count)]
                worker_results = [future.result() for future in futures]
                
            total_time = time.time() - start_time
            
            # Aggregate results
            total_operations = thread_count * operations_per_thread
            total_errors = sum(r['errors'] for r in worker_results)
            avg_thread_time = np.mean([r['completion_time'] for r in worker_results])
            
            results['results_by_thread_count'][thread_count] = {
                'total_time_s': total_time,
                'avg_thread_time_s': avg_thread_time,
                'total_operations': total_operations,
                'total_errors': total_errors,
                'error_rate': total_errors / total_operations,
                'throughput_ops_per_sec': total_operations / total_time,
                'efficiency': (operations_per_thread / avg_thread_time) / (operations_per_thread / results['results_by_thread_count'].get(1, {}).get('avg_thread_time_s', avg_thread_time)) if thread_count > 1 else 1.0
            }
        
        # Calculate summary
        throughputs = [r['throughput_ops_per_sec'] for r in results['results_by_thread_count'].values()]
        error_rates = [r['error_rate'] for r in results['results_by_thread_count'].values()]
        
        results['summary'] = {
            'peak_throughput': max(throughputs),
            'optimal_thread_count': num_threads[throughputs.index(max(throughputs))],
            'avg_error_rate': np.mean(error_rates),
            'scalability_factor': max(throughputs) / min(throughputs) if min(throughputs) > 0 else 1.0
        }
        
        return results
        
    def benchmark_system_maintenance(self, memory_manager: EnhancedMemoryManager,
                                   memory_count: int = 1000,
                                   maintenance_cycles: int = 10) -> Dict[str, Any]:
        """
        Benchmark system maintenance operations (decay, transitions, fusion).
        
        Args:
            memory_manager (EnhancedMemoryManager): Memory system to benchmark
            memory_count (int): Number of memories for testing
            maintenance_cycles (int): Number of maintenance cycles to run
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        results = {
            'test_name': 'system_maintenance_benchmark',
            'timestamp': time.time(),
            'setup': {
                'memory_count': memory_count,
                'maintenance_cycles': maintenance_cycles
            },
            'cycle_results': [],
            'summary': {}
        }
        
        # Setup test memories
        test_memories = self.generate_test_memories(memory_count)
        memory_manager.clear_all_memories()
        
        for memory_data in test_memories:
            memory_manager.add_memory(memory_data['content'], memory_data['metadata'])
            
        print(f"Running {maintenance_cycles} maintenance cycles...")
        
        maintenance_times = []
        decay_times = []
        transition_times = []
        fusion_times = []
        
        for cycle in range(maintenance_cycles):
            cycle_start = time.time()
            
            # Measure individual operations
            decay_start = time.time()
            memory_manager.dual_layer_memory.apply_biological_decay()
            decay_time = time.time() - decay_start
            
            transition_start = time.time()
            transition_stats = memory_manager.dual_layer_memory.manage_layer_transitions()
            transition_time = time.time() - transition_start
            
            fusion_start = time.time()
            fusion_stats = memory_manager._check_and_perform_fusion()
            fusion_time = time.time() - fusion_start
            
            total_cycle_time = time.time() - cycle_start
            
            maintenance_times.append(total_cycle_time)
            decay_times.append(decay_time)
            transition_times.append(transition_time)
            fusion_times.append(fusion_time)
            
            # Record cycle details
            system_stats = memory_manager.get_system_statistics()
            cycle_result = {
                'cycle': cycle,
                'total_time_ms': total_cycle_time * 1000,
                'decay_time_ms': decay_time * 1000,
                'transition_time_ms': transition_time * 1000,
                'fusion_time_ms': fusion_time * 1000,
                'memory_count': system_stats['total_memories'],
                'lml_count': system_stats['lml_count'],
                'sml_count': system_stats['sml_count'],
                'transitions_performed': transition_stats,
                'fusion_performed': fusion_stats
            }
            
            results['cycle_results'].append(cycle_result)
            
            if cycle % 5 == 0:
                print(f"Completed cycle {cycle+1}/{maintenance_cycles}")
        
        # Calculate summary statistics
        results['summary'] = {
            'avg_maintenance_time_ms': np.mean(maintenance_times) * 1000,
            'std_maintenance_time_ms': np.std(maintenance_times) * 1000,
            'avg_decay_time_ms': np.mean(decay_times) * 1000,
            'avg_transition_time_ms': np.mean(transition_times) * 1000,
            'avg_fusion_time_ms': np.mean(fusion_times) * 1000,
            'maintenance_throughput': len(maintenance_times) / np.sum(maintenance_times),
            'time_distribution': {
                'decay_percentage': np.mean(decay_times) / np.mean(maintenance_times) * 100,
                'transition_percentage': np.mean(transition_times) / np.mean(maintenance_times) * 100,
                'fusion_percentage': np.mean(fusion_times) / np.mean(maintenance_times) * 100
            }
        }
        
        return results
        
    def _generate_test_queries(self, memories: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
        """Generate test queries based on memory content."""
        queries = []
        
        query_templates = [
            "What do you know about {topic}?",
            "Tell me about {keyword}",
            "Any updates on {topic}?",
            "What's the status of {keyword}?",
            "Remind me about {topic}",
            "What happened with {keyword}?",
            "Find information about {topic}",
            "Show me details on {keyword}"
        ]
        
        # Extract keywords from memory content
        all_content = " ".join([mem['content'] for mem in memories])
        words = all_content.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Only consider longer words
                word_freq[word] = word_freq.get(word, 0) + 1
                
        # Get most common keywords
        common_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:50]
        keywords = [word for word, freq in common_keywords]
        
        for i in range(count):
            template = random.choice(query_templates)
            keyword = random.choice(keywords) if keywords else f"topic_{i}"
            
            query = template.format(topic=keyword, keyword=keyword)
            
            # Find potentially relevant memories
            relevant_memories = []
            for j, mem in enumerate(memories):
                if keyword.lower() in mem['content'].lower():
                    relevant_memories.append(j)
                    
            queries.append({
                'query': query,
                'keyword': keyword,
                'potentially_relevant': relevant_memories[:5]  # Top 5 potentially relevant
            })
        
        return queries
        
    def _assess_retrieval_accuracy(self, test_queries: List[Dict[str, Any]], 
                                 retrieval_chain: MemoryRetrievalChain) -> Dict[str, Any]:
        """Assess retrieval accuracy using test queries."""
        precision_scores = []
        recall_scores = []
        
        for query_data in test_queries[:20]:  # Test subset for efficiency
            result = retrieval_chain.query(query_data['query'], include_reasoning=False)
            retrieved_memories = result.get('retrieved_memories', [])
            
            # Simple accuracy assessment based on keyword matching
            keyword = query_data['keyword'].lower()
            relevant_retrieved = 0
            
            for mem in retrieved_memories:
                if keyword in mem.get('content', '').lower():
                    relevant_retrieved += 1
                    
            # Calculate precision and recall (simplified)
            precision = relevant_retrieved / len(retrieved_memories) if retrieved_memories else 0
            recall = relevant_retrieved / max(1, len(query_data['potentially_relevant']))
            
            precision_scores.append(precision)
            recall_scores.append(recall)
        
        return {
            'avg_precision': np.mean(precision_scores),
            'avg_recall': np.mean(recall_scores),
            'f1_score': 2 * np.mean(precision_scores) * np.mean(recall_scores) / (np.mean(precision_scores) + np.mean(recall_scores)) if (np.mean(precision_scores) + np.mean(recall_scores)) > 0 else 0
        }
        
    def _assess_scalability(self, results_by_count: Dict[int, Dict]) -> str:
        """Assess scalability based on throughput trends."""
        counts = sorted(results_by_count.keys())
        throughputs = [results_by_count[c]['single_addition']['throughput_per_sec'] for c in counts]
        
        # Check if throughput decreases significantly with scale
        if len(throughputs) >= 3:
            throughput_ratio = throughputs[-1] / throughputs[0]  # Last vs first
            if throughput_ratio > 0.8:
                return "EXCELLENT - Maintains throughput at scale"
            elif throughput_ratio > 0.6:
                return "GOOD - Moderate throughput degradation"
            elif throughput_ratio > 0.4:
                return "FAIR - Noticeable throughput degradation"
            else:
                return "POOR - Significant throughput degradation"
        else:
            return "INSUFFICIENT_DATA"
            
    def run_comprehensive_benchmarks(self, retrieval_chain: MemoryRetrievalChain) -> Dict[str, Any]:
        """
        Run all benchmark tests and generate comprehensive report.
        
        Args:
            retrieval_chain (MemoryRetrievalChain): System to benchmark
            
        Returns:
            Dict[str, Any]: Complete benchmark results
        """
        print("Starting comprehensive performance benchmarks...")
        start_time = time.time()
        
        comprehensive_results = {
            'benchmark_suite': 'Enhanced Dual-Layer Memory Architecture',
            'timestamp': start_time,
            'system_info': {
                'embedding_model': retrieval_chain.embedding_model,
                'vector_backend': retrieval_chain.vector_backend,
                'configuration': {
                    'lambda_base': config.LAMBDA_BASE,
                    'theta_promote': config.THETA_PROMOTE,
                    'theta_demote': config.THETA_DEMOTE
                }
            },
            'benchmark_results': {},
            'summary': {}
        }
        
        # Run individual benchmarks
        benchmarks = [
            ('memory_addition', lambda: self.benchmark_memory_addition(retrieval_chain)),
            ('retrieval_performance', lambda: self.benchmark_retrieval_performance(retrieval_chain)),
            ('concurrent_performance', lambda: self.benchmark_concurrent_performance(retrieval_chain)),
            ('system_maintenance', lambda: self.benchmark_system_maintenance(retrieval_chain.memory_manager))
        ]
        
        for benchmark_name, benchmark_func in benchmarks:
            print(f"\nRunning {benchmark_name} benchmark...")
            try:
                result = benchmark_func()
                comprehensive_results['benchmark_results'][benchmark_name] = result
                print(f"✓ {benchmark_name} completed")
            except Exception as e:
                print(f"✗ {benchmark_name} failed: {e}")
                comprehensive_results['benchmark_results'][benchmark_name] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        # Generate summary
        total_time = time.time() - start_time
        comprehensive_results['summary'] = {
            'total_benchmark_time_s': total_time,
            'benchmarks_completed': len([r for r in comprehensive_results['benchmark_results'].values() if 'error' not in r]),
            'benchmarks_failed': len([r for r in comprehensive_results['benchmark_results'].values() if 'error' in r]),
            'performance_assessment': self._generate_performance_assessment(comprehensive_results['benchmark_results'])
        }
        
        print(f"\nBenchmark suite completed in {total_time:.2f} seconds")
        return comprehensive_results
        
    def _generate_performance_assessment(self, benchmark_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate overall performance assessment."""
        assessment = {}
        
        # Memory addition assessment
        if 'memory_addition' in benchmark_results and 'error' not in benchmark_results['memory_addition']:
            avg_throughput = benchmark_results['memory_addition']['summary']['avg_single_throughput']
            if avg_throughput > 100:
                assessment['memory_addition'] = "EXCELLENT (>100 memories/sec)"
            elif avg_throughput > 50:
                assessment['memory_addition'] = "GOOD (50-100 memories/sec)"
            elif avg_throughput > 20:
                assessment['memory_addition'] = "FAIR (20-50 memories/sec)"
            else:
                assessment['memory_addition'] = "POOR (<20 memories/sec)"
        
        # Retrieval assessment
        if 'retrieval_performance' in benchmark_results and 'error' not in benchmark_results['retrieval_performance']:
            avg_time = benchmark_results['retrieval_performance']['results']['avg_retrieval_time_ms']
            if avg_time < 100:
                assessment['retrieval_performance'] = "EXCELLENT (<100ms)"
            elif avg_time < 500:
                assessment['retrieval_performance'] = "GOOD (100-500ms)"
            elif avg_time < 1000:
                assessment['retrieval_performance'] = "FAIR (500-1000ms)"
            else:
                assessment['retrieval_performance'] = "POOR (>1000ms)"
        
        # Concurrent performance assessment
        if 'concurrent_performance' in benchmark_results and 'error' not in benchmark_results['concurrent_performance']:
            scalability = benchmark_results['concurrent_performance']['summary']['scalability_factor']
            if scalability > 3:
                assessment['concurrent_performance'] = "EXCELLENT (High scalability)"
            elif scalability > 2:
                assessment['concurrent_performance'] = "GOOD (Moderate scalability)"
            elif scalability > 1.5:
                assessment['concurrent_performance'] = "FAIR (Limited scalability)"
            else:
                assessment['concurrent_performance'] = "POOR (No scalability)"
        
        return assessment
