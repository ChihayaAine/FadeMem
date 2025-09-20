"""
Adaptive Memory Fusion System

Implements LLM-guided fusion for related memories using temporal-semantic clustering:
- Identifies fusion candidates through temporal-semantic clustering
- Performs intelligent fusion preserving unique information, temporal progression, and causal relationships
- Validates information preservation through LLM verification
"""

from typing import List, Dict, Tuple, Optional, Set
from .memory_item import MemoryItem
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.llm_interface import LLMInterface
import math
import time
from collections import defaultdict


class MemoryCluster:
    """Represents a cluster of related memories for potential fusion."""
    
    def __init__(self, central_memory: MemoryItem):
        """
        Initialize a memory cluster with a central memory.
        
        Args:
            central_memory (MemoryItem): Central memory around which cluster forms
        """
        self.central_memory = central_memory
        self.memories: List[MemoryItem] = [central_memory]
        self.cluster_id = id(central_memory)  # Use memory id as cluster identifier
        
    def add_memory(self, memory: MemoryItem) -> None:
        """Add a memory to this cluster."""
        if memory not in self.memories:
            self.memories.append(memory)
            
    def get_temporal_span_days(self) -> float:
        """Get the temporal span of memories in this cluster in days."""
        if len(self.memories) <= 1:
            return 0.0
            
        timestamps = [m.creation_timestamp for m in self.memories]
        return (max(timestamps) - min(timestamps)) / 86400
        
    def get_average_similarity(self) -> float:
        """Calculate average pairwise similarity within cluster."""
        if len(self.memories) <= 1:
            return 1.0
            
        similarities = []
        for i, mem1 in enumerate(self.memories):
            for mem2 in self.memories[i+1:]:
                sim = mem1._cosine_similarity(mem1.content_embedding, mem2.content_embedding)
                similarities.append(sim)
                
        return sum(similarities) / len(similarities) if similarities else 0.0


class AdaptiveMemoryFusion:
    def __init__(self, 
                 llm_interface: LLMInterface,
                 theta_fusion: float = 0.6,
                 T_window_days: int = 7,
                 cluster_size_threshold: int = 3,
                 theta_preserve: float = 0.8):
        """
        Initialize adaptive memory fusion system.
        
        Args:
            llm_interface (LLMInterface): LLM for intelligent fusion
            theta_fusion (float): Similarity threshold for fusion candidacy
            T_window_days (int): Temporal window for clustering (days)
            cluster_size_threshold (int): Minimum cluster size for fusion
            theta_preserve (float): Information preservation threshold
        """
        self.llm = llm_interface
        self.theta_fusion = theta_fusion
        self.T_window_days = T_window_days
        self.cluster_size_threshold = cluster_size_threshold
        self.theta_preserve = theta_preserve
        
    def identify_fusion_candidates(self, memories: List[MemoryItem]) -> List[MemoryCluster]:
        """
        Identify fusion candidates through temporal-semantic clustering.
        
        C_k = {m_i : sim(c_i, c_k) > θ_fusion ∧ |τ_i - τ_k| < T_window}
        
        Args:
            memories (List[MemoryItem]): All memories to cluster
            
        Returns:
            List[MemoryCluster]: Clusters of related memories
        """
        clusters = []
        processed_memories = set()
        
        for central_memory in memories:
            if central_memory in processed_memories:
                continue
                
            # Create new cluster
            cluster = MemoryCluster(central_memory)
            processed_memories.add(central_memory)
            
            # Find related memories
            for candidate_memory in memories:
                if candidate_memory in processed_memories:
                    continue
                    
                # Check semantic similarity
                similarity = central_memory._cosine_similarity(
                    central_memory.content_embedding,
                    candidate_memory.content_embedding
                )
                
                if similarity <= self.theta_fusion:
                    continue
                    
                # Check temporal locality
                time_diff_days = abs(
                    central_memory.creation_timestamp - candidate_memory.creation_timestamp
                ) / 86400
                
                if time_diff_days > self.T_window_days:
                    continue
                    
                # Add to cluster
                cluster.add_memory(candidate_memory)
                processed_memories.add(candidate_memory)
                
            # Only keep clusters above threshold size
            if len(cluster.memories) >= self.cluster_size_threshold:
                clusters.append(cluster)
            else:
                # Remove memories from processed set if cluster too small
                for memory in cluster.memories:
                    processed_memories.discard(memory)
                    
        return clusters
        
    def fuse_memory_cluster(self, cluster: MemoryCluster) -> Optional[MemoryItem]:
        """
        Perform intelligent fusion of a memory cluster via LLM guidance.
        
        Args:
            cluster (MemoryCluster): Cluster of memories to fuse
            
        Returns:
            Optional[MemoryItem]: Fused memory if successful, None if fusion fails
        """
        if len(cluster.memories) < 2:
            return cluster.memories[0] if cluster.memories else None
            
        # Sort memories by creation timestamp for temporal progression
        sorted_memories = sorted(cluster.memories, key=lambda m: m.creation_timestamp)
        
        # Create fusion prompt preserving temporal progression and causal relationships
        fusion_prompt = self._create_fusion_prompt(sorted_memories)
        
        try:
            fused_content = self.llm.generate_response(fusion_prompt).strip()
            
            # Create fused memory with aggregated properties
            fused_memory = self._create_fused_memory(sorted_memories, fused_content)
            
            # Validate information preservation
            if self._validate_fusion_quality(sorted_memories, fused_memory):
                return fused_memory
            else:
                print(f"Fusion rejected due to insufficient information preservation")
                return None
                
        except Exception as e:
            print(f"Error in memory fusion: {e}")
            return None
            
    def _create_fusion_prompt(self, sorted_memories: List[MemoryItem]) -> str:
        """
        Create a prompt for LLM-guided fusion preserving temporal and causal relationships.
        
        Args:
            sorted_memories (List[MemoryItem]): Memories sorted by timestamp
            
        Returns:
            str: Fusion prompt for LLM
        """
        memory_entries = []
        for i, memory in enumerate(sorted_memories, 1):
            age_days = memory.get_age_days()
            memory_entries.append(f"{i}. [{age_days:.1f} days ago] {memory.content}")
            
        prompt = f"""
Intelligently merge these related memories while preserving:
1. Unique information from each memory
2. Temporal progression and sequence
3. Causal relationships between events
4. Important details and context

Memories (chronologically ordered):
{chr(10).join(memory_entries)}

Create a single consolidated memory that:
- Combines all unique information
- Maintains the temporal flow of events
- Preserves causal relationships
- Eliminates redundancy
- Keeps important details

Return only the merged memory content without additional text.
"""
        return prompt
        
    def _create_fused_memory(self, original_memories: List[MemoryItem], fused_content: str) -> MemoryItem:
        """
        Create a fused memory with aggregated properties following the methodology.
        
        v_fused(0) = max(v_i) + ε * var({v_i})
        λ_fused = λ_base / (1 + log(|C_k|))
        
        Args:
            original_memories (List[MemoryItem]): Original memories being fused
            fused_content (str): LLM-generated fused content
            
        Returns:
            MemoryItem: Newly created fused memory
        """
        # Use the embedding from the memory with highest strength as base
        # In practice, this should be re-embedded, but we'll use best available
        best_memory = max(original_memories, key=lambda m: m.memory_strength)
        base_embedding = best_memory.content_embedding.copy()
        
        # Create fused memory
        fused_memory = MemoryItem(
            content=fused_content,
            content_embedding=base_embedding,
            metadata={
                'fused_from': [m.content[:50] + "..." for m in original_memories],
                'fusion_timestamp': time.time(),
                'original_count': len(original_memories),
                'temporal_span_days': (
                    max(m.creation_timestamp for m in original_memories) - 
                    min(m.creation_timestamp for m in original_memories)
                ) / 86400
            }
        )
        
        # Aggregate properties following methodology
        strengths = [m.memory_strength for m in original_memories]
        
        # v_fused(0) = max(v_i) + ε * var({v_i})
        max_strength = max(strengths)
        mean_strength = sum(strengths) / len(strengths)
        variance = sum((s - mean_strength) ** 2 for s in strengths) / len(strengths)
        epsilon = 0.1  # Variance bonus factor
        
        fused_memory.memory_strength = min(1.0, max_strength + epsilon * variance)
        
        # Set creation timestamp to earliest
        fused_memory.creation_timestamp = min(m.creation_timestamp for m in original_memories)
        
        # Aggregate access information
        fused_memory.access_frequency = sum(m.access_frequency for m in original_memories)
        
        # Combine all access timestamps
        all_access_timestamps = []
        for memory in original_memories:
            all_access_timestamps.extend(memory.access_timestamps)
        fused_memory.access_timestamps = sorted(all_access_timestamps)
        
        # Recalculate time-decayed access rate
        current_time = time.time()
        kappa = 0.1
        fused_memory.time_decayed_access_rate = sum(
            math.exp(-kappa * (current_time - timestamp) / 86400)
            for timestamp in fused_memory.access_timestamps
        )
        
        # Set layer assignment to highest priority
        layer_assignments = [m.layer_assignment for m in original_memories if m.layer_assignment]
        if 'LML' in layer_assignments:
            fused_memory.layer_assignment = 'LML'
        elif 'SML' in layer_assignments:
            fused_memory.layer_assignment = 'SML'
        else:
            fused_memory.layer_assignment = None
            
        # Update decay parameters with fusion factor
        # λ_fused = λ_base / (1 + log(|C_k|))
        cluster_size = len(original_memories)
        xi_fused = 1 / (1 + math.log(cluster_size))
        
        lambda_base = 0.1  # Should come from config
        mu = 1.0
        importance = fused_memory.calculate_importance()
        fused_memory.lambda_i = lambda_base * xi_fused * math.exp(-mu * importance)
        
        # Set appropriate beta based on layer
        fused_memory.update_decay_parameters()
        
        return fused_memory
        
    def _validate_fusion_quality(self, 
                                original_memories: List[MemoryItem], 
                                fused_memory: MemoryItem) -> bool:
        """
        Validate that fusion preserves essential information using LLM verification.
        
        Args:
            original_memories (List[MemoryItem]): Original memories
            fused_memory (MemoryItem): Fused memory result
            
        Returns:
            bool: True if fusion quality is acceptable
        """
        validation_prompt = f"""
Evaluate this memory fusion for information preservation and quality.

Original memories:
{chr(10).join(f'- {m.content}' for m in original_memories)}

Fused memory: "{fused_memory.content}"

Rate the fusion quality considering:
1. Information preservation (0-40 points): Are key facts preserved?
2. Temporal coherence (0-30 points): Is the sequence/timing clear?
3. Causal relationships (0-20 points): Are cause-effect links maintained?
4. Redundancy elimination (0-10 points): Is repetition minimized?

Provide a total score from 0-100. Respond with only the numerical score.
"""
        
        try:
            response = self.llm.generate_response(validation_prompt).strip()
            score = float(response)
            
            # Convert to 0-1 scale and check against threshold
            normalized_score = score / 100.0
            return normalized_score >= self.theta_preserve
            
        except (ValueError, Exception) as e:
            print(f"Error validating fusion quality: {e}")
            return False  # Reject fusion if validation fails
            
    def perform_adaptive_fusion(self, memories: List[MemoryItem]) -> Tuple[List[MemoryItem], List[MemoryItem]]:
        """
        Perform comprehensive adaptive fusion on a collection of memories.
        
        Args:
            memories (List[MemoryItem]): Memories to potentially fuse
            
        Returns:
            Tuple[List[MemoryItem], List[MemoryItem]]: (memories_to_remove, fused_memories_to_add)
        """
        # Identify fusion candidates
        clusters = self.identify_fusion_candidates(memories)
        
        memories_to_remove = []
        fused_memories_to_add = []
        
        for cluster in clusters:
            fused_memory = self.fuse_memory_cluster(cluster)
            
            if fused_memory is not None:
                # Mark original memories for removal
                memories_to_remove.extend(cluster.memories)
                # Add fused memory
                fused_memories_to_add.append(fused_memory)
                
        return memories_to_remove, fused_memories_to_add
        
    def get_fusion_statistics(self, clusters: List[MemoryCluster]) -> Dict[str, any]:
        """
        Get statistics about potential fusion candidates.
        
        Args:
            clusters (List[MemoryCluster]): Memory clusters
            
        Returns:
            Dict: Fusion statistics
        """
        if not clusters:
            return {
                'total_clusters': 0,
                'total_memories_in_clusters': 0,
                'avg_cluster_size': 0,
                'avg_temporal_span_days': 0,
                'avg_cluster_similarity': 0
            }
            
        stats = {
            'total_clusters': len(clusters),
            'total_memories_in_clusters': sum(len(c.memories) for c in clusters),
            'avg_cluster_size': sum(len(c.memories) for c in clusters) / len(clusters),
            'avg_temporal_span_days': sum(c.get_temporal_span_days() for c in clusters) / len(clusters),
            'avg_cluster_similarity': sum(c.get_average_similarity() for c in clusters) / len(clusters)
        }
        
        return stats
