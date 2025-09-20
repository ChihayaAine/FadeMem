"""
Memory Conflict Resolution System

Implements LLM-based semantic analysis and resolution strategies for memory conflicts:
- Compatible: memories coexist with reduced redundancy
- Contradictory: competitive dynamics with suppression
- Subsumes/Subsumed: intelligent merging via LLM guidance
"""

from typing import List, Dict, Tuple, Optional, Set
from .memory_item import MemoryItem
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.llm_interface import LLMInterface
import math
import time


class ConflictRelationship:
    """Enumeration of memory relationship types from methodology."""
    COMPATIBLE = "compatible"
    CONTRADICTORY = "contradictory"
    SUBSUMES = "subsumes"
    SUBSUMED = "subsumed"


class MemoryConflictResolver:
    def __init__(self, 
                 llm_interface: LLMInterface,
                 theta_sim: float = 0.7,
                 omega: float = 0.3,
                 rho: float = 0.5,
                 W_age_days: int = 30):
        """
        Initialize memory conflict resolution system.
        
        Args:
            llm_interface (LLMInterface): LLM for semantic analysis
            theta_sim (float): Similarity threshold for conflict detection
            omega (float): Redundancy penalty parameter
            rho (float): Suppression strength for contradictory memories
            W_age_days (int): Age difference window for normalization
        """
        self.llm = llm_interface
        self.theta_sim = theta_sim
        self.omega = omega  # Redundancy penalty
        self.rho = rho     # Suppression strength
        self.W_age_days = W_age_days
        
    def detect_conflicts(self, new_memory: MemoryItem, existing_memories: List[MemoryItem]) -> List[Tuple[MemoryItem, float]]:
        """
        Detect semantically similar memories that may conflict with new memory.
        
        S = {m_i : sim(c_new, c_i) > θ_sim}
        
        Args:
            new_memory (MemoryItem): Newly added memory
            existing_memories (List[MemoryItem]): Existing memories to check
            
        Returns:
            List[Tuple[MemoryItem, float]]: Similar memories with similarity scores
        """
        similar_memories = []
        
        for memory in existing_memories:
            # Calculate cosine similarity on L2-normalized embeddings
            similarity = new_memory._cosine_similarity(
                new_memory.content_embedding, 
                memory.content_embedding
            )
            
            if similarity > self.theta_sim:
                similar_memories.append((memory, similarity))
                
        return similar_memories
        
    def classify_relationship(self, memory1: MemoryItem, memory2: MemoryItem) -> ConflictRelationship:
        """
        Use LLM to classify the relationship between two memories.
        
        Args:
            memory1 (MemoryItem): First memory
            memory2 (MemoryItem): Second memory
            
        Returns:
            ConflictRelationship: Classified relationship type
        """
        prompt = f"""
Analyze the relationship between these two memory items and classify it as one of:
- compatible: The memories can coexist and complement each other
- contradictory: The memories contain conflicting information
- subsumes: Memory 1 contains all information in Memory 2 plus more
- subsumed: Memory 1 is contained within Memory 2's information

Memory 1: "{memory1.content}"
Memory 2: "{memory2.content}"

Respond with only one word: compatible, contradictory, subsumes, or subsumed.
"""
        
        try:
            response = self.llm.generate_response(prompt).strip().lower()
            if response in [ConflictRelationship.COMPATIBLE, 
                           ConflictRelationship.CONTRADICTORY, 
                           ConflictRelationship.SUBSUMES, 
                           ConflictRelationship.SUBSUMED]:
                return response
            else:
                # Default to compatible if LLM response is unclear
                return ConflictRelationship.COMPATIBLE
        except Exception as e:
            print(f"Error in LLM classification: {e}")
            return ConflictRelationship.COMPATIBLE
            
    def resolve_compatible_conflict(self, new_memory: MemoryItem, existing_memory: MemoryItem, similarity: float) -> None:
        """
        Handle compatible memories by reducing redundancy in existing memory.
        
        I_i = I_i * (1 - ω * sim(c_new, c_i))
        
        Args:
            new_memory (MemoryItem): New memory being added
            existing_memory (MemoryItem): Existing compatible memory
            similarity (float): Similarity score between memories
        """
        # Calculate redundancy penalty
        redundancy_penalty = self.omega * similarity
        
        # Reduce importance of existing memory based on redundancy
        current_importance = existing_memory.calculate_importance()
        new_importance = current_importance * (1 - redundancy_penalty)
        
        # Update the existing memory's importance indirectly by adjusting its access pattern
        # Since importance is calculated dynamically, we reduce its access frequency impact
        if existing_memory.time_decayed_access_rate > 0:
            existing_memory.time_decayed_access_rate *= (1 - redundancy_penalty)
            
    def resolve_contradictory_conflict(self, new_memory: MemoryItem, existing_memory: MemoryItem) -> None:
        """
        Handle contradictory memories through competitive dynamics.
        
        v_i(t) = v_i(t) * exp(-ρ * clip((τ_new - τ_i) / W_age, 0, 1))
        
        Args:
            new_memory (MemoryItem): New memory being added
            existing_memory (MemoryItem): Existing contradictory memory
        """
        # Calculate normalized age difference
        age_diff_days = (new_memory.creation_timestamp - existing_memory.creation_timestamp) / 86400
        normalized_age_diff = max(0, min(1, age_diff_days / self.W_age_days))
        
        # Apply suppression to existing memory
        suppression_factor = math.exp(-self.rho * normalized_age_diff)
        existing_memory.memory_strength *= suppression_factor
        
        # Ensure memory strength doesn't go below 0
        existing_memory.memory_strength = max(0.0, existing_memory.memory_strength)
        
    def resolve_subsumption_conflict(self, 
                                   memory1: MemoryItem, 
                                   memory2: MemoryItem,
                                   relationship: ConflictRelationship) -> Optional[MemoryItem]:
        """
        Handle subsumption by merging memories via LLM-guided consolidation.
        
        Args:
            memory1 (MemoryItem): First memory
            memory2 (MemoryItem): Second memory  
            relationship (ConflictRelationship): Either SUBSUMES or SUBSUMED
            
        Returns:
            Optional[MemoryItem]: Merged memory if successful, None if merger fails
        """
        # Determine which memory is more general
        if relationship == ConflictRelationship.SUBSUMES:
            general_memory, specific_memory = memory1, memory2
        else:  # SUBSUMED
            general_memory, specific_memory = memory2, memory1
            
        # Use LLM to merge the memories
        merge_prompt = f"""
Merge these two memory items, preserving all unique information while eliminating redundancy:

General memory: "{general_memory.content}"
Specific memory: "{specific_memory.content}"

Create a single consolidated memory that includes all relevant information from both.
Return only the merged content without any additional text.
"""
        
        try:
            merged_content = self.llm.generate_response(merge_prompt).strip()
            
            # Create new merged memory
            # Use the general memory's embedding as base, could be improved with re-embedding
            merged_memory = MemoryItem(
                content=merged_content,
                content_embedding=general_memory.content_embedding.copy(),
                metadata={
                    'merged_from': [general_memory.content[:50], specific_memory.content[:50]],
                    'merge_timestamp': time.time()
                }
            )
            
            # Inherit aggregated properties following methodology
            # v_fused(0) = max(v_i) + ε * var({v_i})
            strengths = [general_memory.memory_strength, specific_memory.memory_strength]
            max_strength = max(strengths)
            variance_bonus = 0.1 * (sum((s - sum(strengths)/len(strengths))**2 for s in strengths) / len(strengths))
            merged_memory.memory_strength = min(1.0, max_strength + variance_bonus)
            
            # Combine access information
            merged_memory.access_frequency = general_memory.access_frequency + specific_memory.access_frequency
            merged_memory.access_timestamps = general_memory.access_timestamps + specific_memory.access_timestamps
            merged_memory.creation_timestamp = min(general_memory.creation_timestamp, specific_memory.creation_timestamp)
            
            # Set layer assignment to the higher priority layer
            if general_memory.layer_assignment == 'LML' or specific_memory.layer_assignment == 'LML':
                merged_memory.layer_assignment = 'LML'
            else:
                merged_memory.layer_assignment = 'SML'
                
            return merged_memory
            
        except Exception as e:
            print(f"Error in LLM merging: {e}")
            return None
            
    def resolve_conflicts(self, 
                         new_memory: MemoryItem, 
                         existing_memories: List[MemoryItem]) -> Tuple[List[MemoryItem], List[MemoryItem]]:
        """
        Comprehensive conflict resolution for a new memory against existing memories.
        
        Args:
            new_memory (MemoryItem): New memory being added
            existing_memories (List[MemoryItem]): Existing memories to check
            
        Returns:
            Tuple[List[MemoryItem], List[MemoryItem]]: (memories_to_remove, memories_to_add)
        """
        # Detect potential conflicts
        similar_memories = self.detect_conflicts(new_memory, existing_memories)
        
        memories_to_remove = []
        memories_to_add = []
        
        for existing_memory, similarity in similar_memories:
            # Classify relationship
            relationship = self.classify_relationship(new_memory, existing_memory)
            
            if relationship == ConflictRelationship.COMPATIBLE:
                # Reduce redundancy in existing memory
                self.resolve_compatible_conflict(new_memory, existing_memory, similarity)
                
            elif relationship == ConflictRelationship.CONTRADICTORY:
                # Apply competitive suppression
                self.resolve_contradictory_conflict(new_memory, existing_memory)
                
            elif relationship in [ConflictRelationship.SUBSUMES, ConflictRelationship.SUBSUMED]:
                # Attempt intelligent merging
                merged_memory = self.resolve_subsumption_conflict(new_memory, existing_memory, relationship)
                
                if merged_memory is not None:
                    # Replace both memories with merged version
                    memories_to_remove.append(existing_memory)
                    memories_to_add.append(merged_memory)
                    
                    # Don't add the original new_memory since it's been merged
                    if new_memory not in memories_to_remove:
                        memories_to_remove.append(new_memory)
                        
        return memories_to_remove, memories_to_add
        
    def validate_information_preservation(self, 
                                        original_memories: List[MemoryItem], 
                                        merged_memory: MemoryItem,
                                        theta_preserve: float = 0.8) -> bool:
        """
        Validate that merged memory preserves essential information from originals.
        
        Args:
            original_memories (List[MemoryItem]): Original memories before merging
            merged_memory (MemoryItem): Resulting merged memory
            theta_preserve (float): Preservation threshold
            
        Returns:
            bool: True if preservation is adequate, False otherwise
        """
        preservation_prompt = f"""
Rate how well this merged memory preserves the essential information from the original memories.

Original memories:
{chr(10).join(f'- {mem.content}' for mem in original_memories)}

Merged memory: "{merged_memory.content}"

Rate the information preservation on a scale of 0.0 to 1.0, where:
- 1.0 = All essential information preserved perfectly
- 0.8 = Most important information preserved
- 0.5 = Some information lost but core meaning intact
- 0.0 = Major information loss

Respond with only a number between 0.0 and 1.0.
"""
        
        try:
            response = self.llm.generate_response(preservation_prompt).strip()
            preservation_score = float(response)
            return preservation_score >= theta_preserve
        except (ValueError, Exception) as e:
            print(f"Error validating preservation: {e}")
            return False  # Reject merger if validation fails
