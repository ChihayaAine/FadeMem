"""
Memory Retrieval Chain

This module provides a high-level interface that integrates the dual-layer memory
system with the RAG workflow, offering a complete retrieval-augmented generation
solution for memory-enhanced AI applications.
"""

from typing import List, Dict, Any, Optional, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.enhanced_memory_manager import EnhancedMemoryManager
from llm.llm_interface import LLMInterface
from .embedding_service import OpenAIEmbeddingService
from .vector_store import MemoryVectorStore
from .rag_workflow import RAGWorkflow
import config
import time
import json


class MemoryRetrievalChain:
    """
    Complete retrieval chain integrating memory system with RAG capabilities.
    
    This class provides a unified interface for:
    - Adding and managing memories with automatic embedding generation
    - Querying the system with intelligent retrieval and response generation
    - Maintaining consistency between memory system and vector store
    - Providing comprehensive analytics and monitoring
    """
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 embedding_model: str = "text-embedding-small-3",
                 vector_backend: str = "chroma",
                 persist_directory: str = "./memory_rag_db",
                 enable_fusion: bool = True,
                 enable_conflict_resolution: bool = True):
        """
        Initialize the complete memory retrieval chain.
        
        Args:
            api_key (str, optional): OpenAI API key for embeddings and LLM
            embedding_model (str): Embedding model to use (default: text-embedding-small-3)
            vector_backend (str): Vector database backend ('chroma', 'faiss', or 'simple')
            persist_directory (str): Directory for persistent storage
            enable_fusion (bool): Enable adaptive memory fusion
            enable_conflict_resolution (bool): Enable memory conflict resolution
        """
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.vector_backend = vector_backend
        self.persist_directory = persist_directory
        self.enable_fusion = enable_fusion
        self.enable_conflict_resolution = enable_conflict_resolution
        
        # Initialize core components
        self._initialize_components()
        
        # Statistics and monitoring
        self.session_stats = {
            'queries_processed': 0,
            'memories_added': 0,
            'embeddings_generated': 0,
            'start_time': time.time(),
            'last_activity': time.time()
        }
        
        print(f"Memory Retrieval Chain initialized:")
        print(f"  - Embedding Model: {embedding_model}")
        print(f"  - Vector Backend: {vector_backend}")
        print(f"  - Fusion Enabled: {enable_fusion}")
        print(f"  - Conflict Resolution: {enable_conflict_resolution}")
        
    def _initialize_components(self) -> None:
        """Initialize all system components."""
        # LLM Interface
        self.llm_interface = LLMInterface(
            model_name="gpt-3.5-turbo",
            api_key=self.api_key
        )
        
        # Embedding Service
        self.embedding_service = OpenAIEmbeddingService(
            api_key=self.api_key,
            model=self.embedding_model,
            max_retries=3,
            timeout=30.0
        )
        
        # Enhanced Memory Manager
        self.memory_manager = EnhancedMemoryManager(
            llm_interface=self.llm_interface,
            embedding_generator=None  # We'll use our embedding service
        )
        
        # Override the embedding generator in memory manager
        self.memory_manager.embedding_generator = self.embedding_service
        
        # Vector Store
        self.vector_store = MemoryVectorStore(
            embedding_dimension=1536,  # text-embedding-small-3 dimension
            collection_name="dual_layer_memories",
            persist_directory=self.persist_directory,
            backend=self.vector_backend
        )
        
        # RAG Workflow
        self.rag_workflow = RAGWorkflow(
            memory_manager=self.memory_manager,
            embedding_service=self.embedding_service,
            vector_store=self.vector_store,
            llm_interface=self.llm_interface,
            max_retrieved_memories=5,
            confidence_threshold=0.6
        )
        
    def add_memory(self, 
                   content: str, 
                   metadata: Optional[Dict[str, Any]] = None,
                   auto_embed: bool = True) -> Dict[str, Any]:
        """
        Add a new memory to the system with automatic embedding and vector storage.
        
        Args:
            content (str): Memory content
            metadata (Dict, optional): Additional metadata
            auto_embed (bool): Automatically generate and store embedding
            
        Returns:
            Dict[str, Any]: Result with memory details and statistics
        """
        start_time = time.time()
        
        try:
            # Generate embedding if requested
            if auto_embed:
                embedding = self.embedding_service.embed_text(content)
                self.session_stats['embeddings_generated'] += 1
            else:
                # Use simple embedding for testing
                embedding = [0.0] * 1536
                
            # Add to memory system (this handles conflict resolution and fusion)
            success = self.memory_manager.add_memory(content, metadata)
            
            if success:
                # Get the added memory to store in vector database
                # Find the most recently added memory with this content
                all_memories = self.memory_manager.dual_layer_memory.get_all_memories()
                added_memory = None
                for memory in reversed(all_memories):  # Check most recent first
                    if memory.content == content:
                        added_memory = memory
                        break
                
                if added_memory and auto_embed:
                    # Add to vector store
                    vector_id = self.vector_store.add_memory(added_memory, embedding)
                    
                    # Update statistics
                    self.session_stats['memories_added'] += 1
                    self.session_stats['last_activity'] = time.time()
                    
                    return {
                        'success': True,
                        'memory_id': vector_id,
                        'content': content,
                        'layer_assignment': added_memory.layer_assignment,
                        'memory_strength': added_memory.memory_strength,
                        'importance_score': added_memory.calculate_importance(),
                        'embedding_generated': auto_embed,
                        'processing_time': time.time() - start_time,
                        'vector_store_stats': self.vector_store.get_stats()
                    }
                else:
                    return {
                        'success': True,
                        'memory_added_to_system': True,
                        'vector_storage': False,
                        'reason': 'Memory not found after addition or embedding disabled',
                        'processing_time': time.time() - start_time
                    }
            else:
                return {
                    'success': False,
                    'reason': 'Memory system rejected the addition',
                    'processing_time': time.time() - start_time
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def query(self, 
              query: str, 
              session_id: str = "default",
              max_memories: int = 5,
              include_reasoning: bool = True) -> Dict[str, Any]:
        """
        Process a query through the complete RAG workflow.
        
        Args:
            query (str): User query
            session_id (str): Session identifier for conversation tracking
            max_memories (int): Maximum memories to retrieve
            include_reasoning (bool): Include reasoning steps in response
            
        Returns:
            Dict[str, Any]: Complete query result with response, context, and metadata
        """
        start_time = time.time()
        
        try:
            # Update RAG workflow parameters
            self.rag_workflow.max_retrieved_memories = max_memories
            
            # Process through RAG workflow
            result = self.rag_workflow.process_query(query, session_id)
            
            # Update session statistics
            self.session_stats['queries_processed'] += 1
            self.session_stats['last_activity'] = time.time()
            
            # Add system information to result
            result.update({
                'system_info': {
                    'embedding_model': self.embedding_model,
                    'vector_backend': self.vector_backend,
                    'memory_system_stats': self.memory_manager.get_system_statistics(),
                    'workflow_stats': self.rag_workflow.get_workflow_stats(),
                    'session_id': session_id
                }
            })
            
            # Filter out reasoning if not requested
            if not include_reasoning:
                result.pop('reasoning_steps', None)
                result.get('metadata', {}).pop('explicit_reasoning', None)
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'query': query,
                'error': str(e),
                'response': f"I encountered an error processing your query: {str(e)}",
                'confidence_score': 0.0,
                'processing_time': time.time() - start_time
            }
    
    def add_memory_batch(self, 
                        memories: List[Dict[str, Any]],
                        auto_embed: bool = True) -> Dict[str, Any]:
        """
        Add multiple memories in batch for efficiency.
        
        Args:
            memories (List[Dict]): List of memory dictionaries with 'content' and optional 'metadata'
            auto_embed (bool): Automatically generate embeddings
            
        Returns:
            Dict[str, Any]: Batch processing results
        """
        start_time = time.time()
        results = {
            'successful_additions': 0,
            'failed_additions': 0,
            'details': [],
            'processing_time': 0
        }
        
        try:
            # Extract content for batch embedding
            contents = [mem.get('content', '') for mem in memories]
            
            if auto_embed and contents:
                # Generate embeddings in batch for efficiency
                embeddings = self.embedding_service.embed_batch(contents)
                self.session_stats['embeddings_generated'] += len(embeddings)
            else:
                embeddings = [[0.0] * 1536] * len(contents)
            
            # Process each memory
            for i, (memory_data, embedding) in enumerate(zip(memories, embeddings)):
                content = memory_data.get('content', '')
                metadata = memory_data.get('metadata', {})
                
                if not content:
                    results['failed_additions'] += 1
                    results['details'].append({
                        'index': i,
                        'success': False,
                        'reason': 'Empty content'
                    })
                    continue
                
                # Add to memory system
                success = self.memory_manager.add_memory(content, metadata)
                
                if success:
                    # Find and add to vector store
                    all_memories = self.memory_manager.dual_layer_memory.get_all_memories()
                    added_memory = None
                    for memory in reversed(all_memories):
                        if memory.content == content:
                            added_memory = memory
                            break
                    
                    if added_memory and auto_embed:
                        vector_id = self.vector_store.add_memory(added_memory, embedding)
                        results['successful_additions'] += 1
                        results['details'].append({
                            'index': i,
                            'success': True,
                            'vector_id': vector_id,
                            'layer': added_memory.layer_assignment
                        })
                    else:
                        results['successful_additions'] += 1
                        results['details'].append({
                            'index': i,
                            'success': True,
                            'vector_storage': False
                        })
                else:
                    results['failed_additions'] += 1
                    results['details'].append({
                        'index': i,
                        'success': False,
                        'reason': 'Memory system rejected addition'
                    })
            
            # Update statistics
            self.session_stats['memories_added'] += results['successful_additions']
            self.session_stats['last_activity'] = time.time()
            
            results['processing_time'] = time.time() - start_time
            return results
            
        except Exception as e:
            results['error'] = str(e)
            results['processing_time'] = time.time() - start_time
            return results
    
    def get_memory_insights(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed insights about a specific memory.
        
        Args:
            memory_id (str): Memory ID from vector store
            
        Returns:
            Dict[str, Any]: Memory insights and analysis
        """
        memory_data = self.vector_store.get_memory_by_id(memory_id)
        if not memory_data:
            return None
        
        # Find corresponding memory in the memory system
        all_memories = self.memory_manager.dual_layer_memory.get_all_memories()
        system_memory = None
        for mem in all_memories:
            if mem.content == memory_data.get('content', ''):
                system_memory = mem
                break
        
        insights = {
            'memory_id': memory_id,
            'content': memory_data.get('content', ''),
            'vector_store_data': memory_data,
            'system_memory_found': system_memory is not None
        }
        
        if system_memory:
            insights.update({
                'current_strength': system_memory.memory_strength,
                'access_frequency': system_memory.access_frequency,
                'layer_assignment': system_memory.layer_assignment,
                'half_life_days': system_memory.get_half_life(),
                'age_days': system_memory.get_age_days(),
                'importance_score': system_memory.calculate_importance(),
                'decay_parameters': {
                    'lambda_i': system_memory.lambda_i,
                    'beta_i': system_memory.beta_i
                }
            })
        
        return insights
    
    def system_maintenance(self, force_update: bool = False) -> Dict[str, Any]:
        """
        Perform system maintenance including decay updates, transitions, and fusion.
        
        Args:
            force_update (bool): Force all maintenance operations
            
        Returns:
            Dict[str, Any]: Maintenance results and statistics
        """
        start_time = time.time()
        
        # Run memory system maintenance
        maintenance_stats = self.memory_manager.update_system(force_all=force_update)
        
        # Sync any changes with vector store (simplified - in practice would need more sophisticated sync)
        sync_stats = {
            'vectors_updated': 0,
            'vectors_removed': 0,
            'sync_errors': 0
        }
        
        # Note: Full synchronization would require tracking changes in the memory system
        # and updating corresponding vectors. This is simplified for demonstration.
        
        maintenance_result = {
            'maintenance_time': time.time() - start_time,
            'memory_system_maintenance': maintenance_stats,
            'vector_store_sync': sync_stats,
            'system_health': self._get_system_health()
        }
        
        return maintenance_result
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics."""
        memory_stats = self.memory_manager.get_system_statistics()
        vector_stats = self.vector_store.get_stats()
        workflow_stats = self.rag_workflow.get_workflow_stats()
        embedding_stats = self.embedding_service.get_stats()
        
        return {
            'total_memories': memory_stats['total_memories'],
            'memory_distribution': {
                'lml': memory_stats['lml_count'],
                'sml': memory_stats['sml_count']
            },
            'avg_memory_strength': memory_stats['avg_memory_strength'],
            'avg_half_life_days': memory_stats['avg_half_life_days'],
            'vector_store_health': {
                'backend': vector_stats['backend'],
                'total_vectors': vector_stats['total_memories'],
                'search_count': vector_stats['search_count']
            },
            'workflow_performance': {
                'avg_response_time': workflow_stats['avg_response_time'],
                'avg_confidence': workflow_stats['avg_confidence'],
                'query_count': workflow_stats['query_count']
            },
            'embedding_efficiency': {
                'cache_hit_rate': embedding_stats['cache_hit_rate'],
                'api_calls': embedding_stats['api_calls'],
                'mock_mode': embedding_stats['mock_mode']
            }
        }
    
    def export_system_state(self, include_embeddings: bool = False) -> Dict[str, Any]:
        """
        Export complete system state for backup or analysis.
        
        Args:
            include_embeddings (bool): Include embedding vectors in export
            
        Returns:
            Dict[str, Any]: Complete system state
        """
        export_data = {
            'timestamp': time.time(),
            'configuration': {
                'embedding_model': self.embedding_model,
                'vector_backend': self.vector_backend,
                'enable_fusion': self.enable_fusion,
                'enable_conflict_resolution': self.enable_conflict_resolution
            },
            'memory_system': {
                'memories': self.memory_manager.export_memories(),
                'statistics': self.memory_manager.get_system_statistics()
            },
            'vector_store': {
                'statistics': self.vector_store.get_stats(),
                'metadata': {mid: meta for mid, meta in self.vector_store.memory_metadata.items()}
            },
            'session_statistics': self.session_stats,
            'system_health': self._get_system_health()
        }
        
        if include_embeddings:
            export_data['vector_store']['embeddings'] = self.vector_store.memory_embeddings
        
        return export_data
    
    def clear_all_data(self, confirm: bool = False) -> Dict[str, Any]:
        """
        Clear all data from the system (use with caution).
        
        Args:
            confirm (bool): Confirmation flag to prevent accidental deletion
            
        Returns:
            Dict[str, Any]: Deletion results
        """
        if not confirm:
            return {
                'success': False,
                'message': 'Deletion not confirmed. Set confirm=True to proceed.'
            }
        
        try:
            # Clear memory system
            self.memory_manager.clear_all_memories()
            
            # Clear vector store
            self.vector_store.clear()
            
            # Clear embedding cache
            self.embedding_service.clear_cache()
            
            # Reset statistics
            self.session_stats = {
                'queries_processed': 0,
                'memories_added': 0,
                'embeddings_generated': 0,
                'start_time': time.time(),
                'last_activity': time.time()
            }
            
            return {
                'success': True,
                'message': 'All data cleared successfully',
                'timestamp': time.time()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Error occurred during data clearing'
            }
    
    def __str__(self) -> str:
        health = self._get_system_health()
        return (f"MemoryRetrievalChain("
                f"memories={health['total_memories']}, "
                f"LML={health['memory_distribution']['lml']}, "
                f"SML={health['memory_distribution']['sml']}, "
                f"model={self.embedding_model}, "
                f"backend={self.vector_backend})")
