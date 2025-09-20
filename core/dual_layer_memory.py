"""
Dual-Layer Memory Architecture Implementation

Implements the methodology's dual-layer system with:
- Long-term Memory Layer (LML): High-importance memories with slow decay
- Short-term Memory Layer (SML): Low-importance memories with rapid decay
- Dynamic layer transitions based on importance thresholds with hysteresis
"""

from typing import List, Dict, Optional, Tuple
from .memory_item import MemoryItem
import time


class DualLayerMemory:
    def __init__(self, 
                 theta_promote: float = 0.7, 
                 theta_demote: float = 0.5,
                 max_lml_capacity: int = 100,
                 max_sml_capacity: int = 50,
                 epsilon_prune: float = 0.05,
                 T_max_days: int = 30):
        """
        Initialize dual-layer memory system following the methodology.
        
        Layer transitions occur when:
        Layer(m_i) = LML if I_i(t) ≥ θ_promote
                   = SML if I_i(t) < θ_demote
        
        Args:
            theta_promote (float): Promotion threshold to LML
            theta_demote (float): Demotion threshold to SML (should be < theta_promote for hysteresis)
            max_lml_capacity (int): Maximum capacity for Long-term Memory Layer
            max_sml_capacity (int): Maximum capacity for Short-term Memory Layer
            epsilon_prune (float): Memory strength threshold for automatic pruning
            T_max_days (int): Maximum days before dormant memory pruning
        """
        # Layer assignment thresholds with hysteresis
        self.theta_promote = theta_promote
        self.theta_demote = theta_demote
        
        if theta_promote <= theta_demote:
            raise ValueError("theta_promote must be > theta_demote to provide hysteresis")
            
        # Memory storage
        self.long_term_memories: List[MemoryItem] = []  # LML
        self.short_term_memories: List[MemoryItem] = []  # SML
        
        # Capacity limits
        self.max_lml_capacity = max_lml_capacity
        self.max_sml_capacity = max_sml_capacity
        
        # Pruning parameters
        self.epsilon_prune = epsilon_prune
        self.T_max_days = T_max_days
        
    def add_memory(self, memory: MemoryItem, query_context: Optional[List[float]] = None) -> bool:
        """
        Add a new memory to the appropriate layer based on importance.
        
        Args:
            memory (MemoryItem): Memory item to add
            query_context (List[float], optional): Recent context for importance calculation
            
        Returns:
            bool: True if successfully added, False if capacity exceeded
        """
        # Calculate importance score
        importance = memory.calculate_importance(query_context)
        
        # Assign to appropriate layer based on importance
        if importance >= self.theta_promote:
            target_layer = 'LML'
            target_list = self.long_term_memories
            max_capacity = self.max_lml_capacity
        else:
            target_layer = 'SML'
            target_list = self.short_term_memories
            max_capacity = self.max_sml_capacity
            
        # Check capacity
        if len(target_list) >= max_capacity:
            # Try to make space by pruning weak memories
            self._prune_weak_memories()
            if len(target_list) >= max_capacity:
                return False
                
        # Set layer assignment and update decay parameters
        memory.layer_assignment = target_layer
        memory.update_decay_parameters()
        target_list.append(memory)
        
        return True
        
    def remove_memory(self, memory: MemoryItem) -> bool:
        """
        Remove a memory from its current layer.
        
        Args:
            memory (MemoryItem): Memory to remove
            
        Returns:
            bool: True if removed, False if not found
        """
        if memory in self.long_term_memories:
            self.long_term_memories.remove(memory)
            return True
        elif memory in self.short_term_memories:
            self.short_term_memories.remove(memory)
            return True
        return False
        
    def manage_layer_transitions(self, query_context: Optional[List[float]] = None) -> Dict[str, int]:
        """
        Check and execute layer transitions based on current importance scores.
        
        Args:
            query_context (List[float], optional): Recent context for importance calculation
            
        Returns:
            Dict[str, int]: Statistics about transitions performed
        """
        stats = {
            'promoted_to_lml': 0,
            'demoted_to_sml': 0,
            'pruned': 0
        }
        
        # Check SML memories for promotion to LML
        memories_to_promote = []
        for memory in self.short_term_memories:
            importance = memory.calculate_importance(query_context)
            if importance >= self.theta_promote:
                memories_to_promote.append(memory)
                
        # Execute promotions
        for memory in memories_to_promote:
            if len(self.long_term_memories) < self.max_lml_capacity:
                self.short_term_memories.remove(memory)
                memory.layer_assignment = 'LML'
                memory.update_decay_parameters()
                self.long_term_memories.append(memory)
                stats['promoted_to_lml'] += 1
                
        # Check LML memories for demotion to SML
        memories_to_demote = []
        for memory in self.long_term_memories:
            importance = memory.calculate_importance(query_context)
            if importance < self.theta_demote:
                memories_to_demote.append(memory)
                
        # Execute demotions
        for memory in memories_to_demote:
            if len(self.short_term_memories) < self.max_sml_capacity:
                self.long_term_memories.remove(memory)
                memory.layer_assignment = 'SML'
                memory.update_decay_parameters()
                self.short_term_memories.append(memory)
                stats['demoted_to_sml'] += 1
                
        # Prune weak and dormant memories
        stats['pruned'] = self._prune_weak_memories()
        
        return stats
        
    def apply_biological_decay(self, current_time: Optional[float] = None) -> None:
        """
        Apply biologically-inspired decay to all memories in both layers.
        
        Args:
            current_time (float, optional): Current timestamp
        """
        if current_time is None:
            current_time = time.time()
            
        # Apply decay to all memories
        for memory in self.long_term_memories + self.short_term_memories:
            memory.apply_biological_decay(current_time)
            
    def _prune_weak_memories(self, current_time: Optional[float] = None) -> int:
        """
        Remove memories that fall below pruning criteria.
        
        Criteria:
        1. Memory strength < epsilon_prune
        2. Dormant beyond T_max days
        
        Args:
            current_time (float, optional): Current timestamp
            
        Returns:
            int: Number of memories pruned
        """
        if current_time is None:
            current_time = time.time()
            
        pruned_count = 0
        
        # Prune from both layers
        for memory_list in [self.long_term_memories, self.short_term_memories]:
            memories_to_remove = []
            
            for memory in memory_list:
                # Check strength threshold
                if memory.memory_strength < self.epsilon_prune:
                    memories_to_remove.append(memory)
                    continue
                    
                # Check dormancy threshold
                days_since_access = memory.get_time_since_access_days(current_time)
                if days_since_access > self.T_max_days:
                    memories_to_remove.append(memory)
                    
            # Remove identified memories
            for memory in memories_to_remove:
                memory_list.remove(memory)
                pruned_count += 1
                
        return pruned_count
        
    def get_all_memories(self) -> List[MemoryItem]:
        """
        Get all memories from both layers.
        
        Returns:
            List[MemoryItem]: Combined list of all memories
        """
        return self.long_term_memories + self.short_term_memories
        
    def get_layer_statistics(self) -> Dict[str, any]:
        """
        Get statistics about both memory layers.
        
        Returns:
            Dict: Layer statistics and metrics
        """
        lml_strengths = [m.memory_strength for m in self.long_term_memories]
        sml_strengths = [m.memory_strength for m in self.short_term_memories]
        
        stats = {
            'lml_count': len(self.long_term_memories),
            'sml_count': len(self.short_term_memories),
            'lml_capacity_used': len(self.long_term_memories) / self.max_lml_capacity,
            'sml_capacity_used': len(self.short_term_memories) / self.max_sml_capacity,
            'lml_avg_strength': sum(lml_strengths) / len(lml_strengths) if lml_strengths else 0,
            'sml_avg_strength': sum(sml_strengths) / len(sml_strengths) if sml_strengths else 0,
            'total_memories': len(self.long_term_memories) + len(self.short_term_memories)
        }
        
        return stats
        
    def __str__(self) -> str:
        stats = self.get_layer_statistics()
        return (f"DualLayerMemory:\n"
                f"  LML: {stats['lml_count']}/{self.max_lml_capacity} "
                f"(avg_strength: {stats['lml_avg_strength']:.3f})\n"
                f"  SML: {stats['sml_count']}/{self.max_sml_capacity} "
                f"(avg_strength: {stats['sml_avg_strength']:.3f})")
