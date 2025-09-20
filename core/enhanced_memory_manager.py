"""
Enhanced Memory Manager implementing the complete dual-layer memory architecture

Integrates all methodology components:
- Dual-layer memory with dynamic transitions
- Biologically-inspired forgetting curves  
- Memory conflict resolution
- Adaptive memory fusion
- Enhanced retrieval with semantic similarity

Follows the complete memory evolution:
M_{t+Δt} = Fusion(Resolution(Decay(M_t, Δt) ∪ {m_new}))
"""

from typing import List, Dict, Tuple, Optional
from .memory_item import MemoryItem
from .dual_layer_memory import DualLayerMemory
from .conflict_resolution import MemoryConflictResolver
from .adaptive_fusion import AdaptiveMemoryFusion
from llm.llm_interface import LLMInterface
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import time
import numpy as np


class EmbeddingGenerator:
    """Mock embedding generator for demonstration. In practice, use real embeddings."""
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        # Simple vocabulary for mock embeddings
        self.vocab_embeddings = self._initialize_vocab()
        
    def _initialize_vocab(self) -> Dict[str, List[float]]:
        """Initialize mock embeddings for common words."""
        vocab = [
            'meeting', 'client', 'project', 'deadline', 'team', 'work', 'task',
            'schedule', 'presentation', 'report', 'budget', 'goal', 'strategy',
            'development', 'analysis', 'review', 'feedback', 'proposal', 'decision',
            'planning', 'implementation', 'testing', 'deployment', 'maintenance'
        ]
        
        embeddings = {}
        np.random.seed(42)  # For reproducible mock embeddings
        
        for word in vocab:
            # Generate random but consistent embeddings
            embedding = np.random.normal(0, 1, self.dimension).tolist()
            # Normalize to unit length
            magnitude = sum(x*x for x in embedding) ** 0.5
            embeddings[word] = [x/magnitude for x in embedding]
            
        return embeddings
        
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using simple averaging of word embeddings."""
        words = text.lower().split()
        embeddings = []
        
        for word in words:
            if word in self.vocab_embeddings:
                embeddings.append(self.vocab_embeddings[word])
            else:
                # Generate random embedding for unknown words
                np.random.seed(hash(word) % 2**32)
                embedding = np.random.normal(0, 1, self.dimension).tolist()
                magnitude = sum(x*x for x in embedding) ** 0.5
                embeddings.append([x/magnitude for x in embedding])
                
        if not embeddings:
            # Default zero embedding
            return [0.0] * self.dimension
            
        # Average embeddings
        avg_embedding = [0.0] * self.dimension
        for embedding in embeddings:
            for i in range(self.dimension):
                avg_embedding[i] += embedding[i]
                
        # Normalize result
        for i in range(self.dimension):
            avg_embedding[i] /= len(embeddings)
            
        # Ensure unit length
        magnitude = sum(x*x for x in avg_embedding) ** 0.5
        if magnitude > 0:
            avg_embedding = [x/magnitude for x in avg_embedding]
            
        return avg_embedding


class EnhancedMemoryManager:
    def __init__(self, 
                 llm_interface: Optional[LLMInterface] = None,
                 embedding_generator: Optional[EmbeddingGenerator] = None):
        """
        Initialize enhanced memory manager with all methodology components.
        
        Args:
            llm_interface (LLMInterface, optional): LLM for conflict resolution and fusion
            embedding_generator (EmbeddingGenerator, optional): Embedding generator for semantic analysis
        """
        # Core components
        self.dual_layer_memory = DualLayerMemory(
            theta_promote=config.THETA_PROMOTE,
            theta_demote=config.THETA_DEMOTE,
            max_lml_capacity=config.LONG_TERM_MEMORY_CAPACITY,
            max_sml_capacity=config.SHORT_TERM_MEMORY_CAPACITY,
            epsilon_prune=config.EPSILON_PRUNE,
            T_max_days=config.T_MAX_DAYS
        )
        
        # LLM interface (with fallback to mock)
        self.llm = llm_interface or LLMInterface()
        
        # Embedding generator
        self.embedding_generator = embedding_generator or EmbeddingGenerator(config.EMBEDDING_DIMENSION)
        
        # Advanced components
        self.conflict_resolver = MemoryConflictResolver(
            llm_interface=self.llm,
            theta_sim=config.THETA_SIM,
            omega=config.OMEGA,
            rho=config.RHO,
            W_age_days=config.W_AGE_DAYS
        )
        
        self.adaptive_fusion = AdaptiveMemoryFusion(
            llm_interface=self.llm,
            theta_fusion=config.THETA_FUSION,
            T_window_days=config.T_WINDOW_DAYS,
            cluster_size_threshold=config.CLUSTER_SIZE_THRESHOLD,
            theta_preserve=config.THETA_PRESERVE
        )
        
        # System state
        self.last_decay_update = time.time()
        self.last_transition_check = time.time()
        self.last_fusion_check = time.time()
        self.current_context_embedding: Optional[List[float]] = None
        
    def add_memory(self, content: str, metadata: Optional[Dict] = None) -> bool:
        """
        Add a new memory following the complete methodology evolution.
        
        Implements: M_{t+Δt} = Fusion(Resolution(Decay(M_t, Δt) ∪ {m_new}))
        
        Args:
            content (str): Memory content
            metadata (Dict, optional): Additional metadata
            
        Returns:
            bool: True if successfully added
        """
        # Generate embedding for new memory
        content_embedding = self.embedding_generator.generate_embedding(content)
        
        # Create new memory item
        new_memory = MemoryItem(
            content=content,
            content_embedding=content_embedding,
            metadata=metadata or {}
        )
        
        # Step 1: Apply decay to existing memories
        self.dual_layer_memory.apply_biological_decay()
        
        # Step 2: Get existing memories for conflict resolution
        existing_memories = self.dual_layer_memory.get_all_memories()
        
        # Step 3: Resolve conflicts with new memory
        memories_to_remove, memories_to_add = self.conflict_resolver.resolve_conflicts(
            new_memory, existing_memories
        )
        
        # Apply conflict resolution results
        for memory in memories_to_remove:
            self.dual_layer_memory.remove_memory(memory)
            
        # Add resolved memories (including potentially merged ones)
        memories_added = []
        for memory in memories_to_add:
            if self.dual_layer_memory.add_memory(memory, self.current_context_embedding):
                memories_added.append(memory)
                
        # Add original new memory if it wasn't removed during conflict resolution
        if new_memory not in memories_to_remove:
            success = self.dual_layer_memory.add_memory(new_memory, self.current_context_embedding)
            if success:
                memories_added.append(new_memory)
        else:
            success = len(memories_added) > 0
            
        # Step 4: Perform adaptive fusion if needed
        self._check_and_perform_fusion()
        
        # Step 5: Manage layer transitions
        self.dual_layer_memory.manage_layer_transitions(self.current_context_embedding)
        
        return success
        
    def retrieve_memories(self, 
                         query: str, 
                         top_k: int = 5,
                         relevance_threshold: float = 0.3) -> List[MemoryItem]:
        """
        Enhanced memory retrieval using methodology's importance scoring.
        
        Args:
            query (str): Query text for retrieval
            top_k (int): Number of top memories to retrieve
            relevance_threshold (float): Minimum relevance threshold
            
        Returns:
            List[MemoryItem]: Retrieved memories sorted by relevance
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        self.current_context_embedding = query_embedding  # Update context for importance calculation
        
        # Get all memories
        all_memories = self.dual_layer_memory.get_all_memories()
        
        # Calculate relevance scores using methodology's importance function
        memory_scores = []
        for memory in all_memories:
            # Calculate importance score with query context
            importance = memory.calculate_importance(
                query_context=query_embedding,
                alpha=config.ALPHA,
                beta=config.BETA,
                gamma=config.GAMMA
            )
            
            # Apply relevance threshold
            if importance >= relevance_threshold:
                memory_scores.append((memory, importance))
                
        # Sort by relevance and take top_k
        memory_scores.sort(key=lambda x: x[1], reverse=True)
        retrieved_memories = [mem for mem, score in memory_scores[:top_k]]
        
        # Mark retrieved memories as accessed
        for memory in retrieved_memories:
            memory.access(
                kappa=config.KAPPA,
                delta_v=config.DELTA_V,
                N=config.N_SPACING,
                W=config.W_WINDOW_DAYS
            )
            
        return retrieved_memories
        
    def get_memory_context(self, query: str, top_k: int = 5) -> str:
        """
        Get formatted context string from relevant memories.
        
        Args:
            query (str): Query for memory retrieval
            top_k (int): Number of memories to include
            
        Returns:
            str: Formatted context string
        """
        memories = self.retrieve_memories(query, top_k)
        
        if not memories:
            return "No relevant memories found."
            
        context_lines = ["Relevant memories:"]
        for i, memory in enumerate(memories, 1):
            importance = memory.calculate_importance(self.current_context_embedding)
            age_days = memory.get_age_days()
            half_life = memory.get_half_life()
            
            context_lines.append(
                f"{i}. [{age_days:.1f}d old, t½={half_life:.1f}d, I={importance:.3f}] {memory.content}"
            )
            
        return "\n".join(context_lines)
        
    def update_system(self, force_all: bool = False) -> Dict[str, any]:
        """
        Perform periodic system maintenance following methodology.
        
        Args:
            force_all (bool): Force all updates regardless of timing
            
        Returns:
            Dict: Statistics about updates performed
        """
        current_time = time.time()
        stats = {
            'decay_applied': False,
            'transitions_checked': False,
            'fusion_performed': False,
            'memories_pruned': 0,
            'transitions': {},
            'fusion_stats': {}
        }
        
        # Apply biological decay
        if (force_all or 
            current_time - self.last_decay_update >= config.DECAY_UPDATE_INTERVAL):
            self.dual_layer_memory.apply_biological_decay(current_time)
            self.last_decay_update = current_time
            stats['decay_applied'] = True
            
        # Check layer transitions
        if (force_all or 
            current_time - self.last_transition_check >= config.TRANSITION_CHECK_INTERVAL):
            transition_stats = self.dual_layer_memory.manage_layer_transitions(
                self.current_context_embedding
            )
            self.last_transition_check = current_time
            stats['transitions_checked'] = True
            stats['transitions'] = transition_stats
            stats['memories_pruned'] = transition_stats.get('pruned', 0)
            
        # Perform adaptive fusion
        if (force_all or 
            current_time - self.last_fusion_check >= config.FUSION_CHECK_INTERVAL):
            fusion_stats = self._check_and_perform_fusion()
            self.last_fusion_check = current_time
            stats['fusion_performed'] = True
            stats['fusion_stats'] = fusion_stats
            
        return stats
        
    def _check_and_perform_fusion(self) -> Dict[str, any]:
        """
        Check for fusion opportunities and perform adaptive fusion.
        
        Returns:
            Dict: Fusion statistics
        """
        all_memories = self.dual_layer_memory.get_all_memories()
        
        # Perform adaptive fusion
        memories_to_remove, fused_memories = self.adaptive_fusion.perform_adaptive_fusion(all_memories)
        
        # Apply fusion results
        fusion_count = 0
        for memory in memories_to_remove:
            if self.dual_layer_memory.remove_memory(memory):
                fusion_count += 1
                
        added_count = 0
        for fused_memory in fused_memories:
            if self.dual_layer_memory.add_memory(fused_memory, self.current_context_embedding):
                added_count += 1
                
        return {
            'memories_fused': fusion_count,
            'fused_memories_created': added_count,
            'net_memory_change': added_count - fusion_count
        }
        
    def get_system_statistics(self) -> Dict[str, any]:
        """
        Get comprehensive system statistics.
        
        Returns:
            Dict: System statistics and metrics
        """
        layer_stats = self.dual_layer_memory.get_layer_statistics()
        all_memories = self.dual_layer_memory.get_all_memories()
        
        # Calculate additional statistics
        if all_memories:
            strengths = [m.memory_strength for m in all_memories]
            half_lives = [m.get_half_life() for m in all_memories]
            ages = [m.get_age_days() for m in all_memories]
            
            stats = {
                **layer_stats,
                'avg_memory_strength': sum(strengths) / len(strengths),
                'min_memory_strength': min(strengths),
                'max_memory_strength': max(strengths),
                'avg_half_life_days': sum(half_lives) / len(half_lives),
                'avg_age_days': sum(ages) / len(ages),
                'total_accesses': sum(m.access_frequency for m in all_memories),
                'system_uptime_hours': (time.time() - self.last_decay_update) / 3600
            }
        else:
            stats = {
                **layer_stats,
                'avg_memory_strength': 0,
                'min_memory_strength': 0,
                'max_memory_strength': 0,
                'avg_half_life_days': 0,
                'avg_age_days': 0,
                'total_accesses': 0,
                'system_uptime_hours': 0
            }
            
        return stats
        
    def export_memories(self) -> List[Dict]:
        """
        Export all memories for analysis or backup.
        
        Returns:
            List[Dict]: Serialized memory data
        """
        all_memories = self.dual_layer_memory.get_all_memories()
        exported = []
        
        for memory in all_memories:
            exported.append({
                'content': memory.content,
                'memory_strength': memory.memory_strength,
                'access_frequency': memory.access_frequency,
                'creation_timestamp': memory.creation_timestamp,
                'layer_assignment': memory.layer_assignment,
                'lambda_i': memory.lambda_i,
                'beta_i': memory.beta_i,
                'half_life_days': memory.get_half_life(),
                'age_days': memory.get_age_days(),
                'metadata': memory.metadata
            })
            
        return exported
        
    def clear_all_memories(self) -> None:
        """Clear all memories from both layers."""
        self.dual_layer_memory.long_term_memories.clear()
        self.dual_layer_memory.short_term_memories.clear()
        
    def __str__(self) -> str:
        stats = self.get_system_statistics()
        return (f"EnhancedMemoryManager:\n"
                f"  Total Memories: {stats['total_memories']}\n"
                f"  LML: {stats['lml_count']} (avg strength: {stats['lml_avg_strength']:.3f})\n"
                f"  SML: {stats['sml_count']} (avg strength: {stats['sml_avg_strength']:.3f})\n"
                f"  System Avg Half-life: {stats['avg_half_life_days']:.2f} days")
