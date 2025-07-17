from typing import Optional, List
from ..core.memory_item import MemoryItem
from ..core.working_memory import WorkingMemory
from ..core.short_term_memory import ShortTermMemory
from ..core.long_term_memory import LongTermMemory
from ..core.archive import Archive
from ..utils.decay_functions import exponential_decay
from ..utils.importance_scorer import calculate_importance
from ..rag.retriever import Retriever
from ..management.transition_rules import TransitionRules

class MemoryManager:
    def __init__(self, decay_threshold: float = 0.1):
        """
        Initialize the Memory Manager with different memory layers and a decay threshold.
        
        Args:
            decay_threshold (float, optional): Threshold below which memories are considered for transition or archival. Defaults to 0.1.
        """
        self.working_memory = WorkingMemory()
        self.short_term_memory = ShortTermMemory()
        self.long_term_memory = LongTermMemory()
        self.archive = Archive()
        self.decay_threshold = decay_threshold
        self.retriever = Retriever(self)  # Initialize retriever for RAG
        self.transition_rules = TransitionRules()  # Initialize transition rules
        
    def add_memory(self, content: str, metadata: dict = None) -> bool:
        """
        Add a new memory, starting in Working Memory.
        
        Args:
            content (str): Content of the memory.
            metadata (dict, optional): Additional data for importance scoring.
            
        Returns:
            bool: True if added successfully.
        """
        memory = MemoryItem(content, metadata)
        memory.update_importance(calculate_importance(memory))
        return self.working_memory.add_memory(memory)
        
    def retrieve_memory(self, content: str) -> Optional[MemoryItem]:
        """
        Retrieve a memory by content from any layer, prioritizing higher layers.
        
        Args:
            content (str): Content to search for.
            
        Returns:
            Optional[MemoryItem]: The memory if found, None otherwise.
        """
        memory = self.working_memory.get_memory(content)
        if memory:
            return memory
        memory = self.short_term_memory.get_memory(content)
        if memory:
            return memory
        memory = self.long_term_memory.get_memory(content)
        if memory:
            return memory
        archived = self.archive.retrieve_archived(content)
        return archived[0] if archived else None
        
    def get_context_for_query(self, query: str) -> str:
        """
        Retrieve and format relevant memories as context for a given query.
        
        Args:
            query (str): The query to find relevant memories for.
            
        Returns:
            str: Formatted context string from relevant memories.
        """
        relevant_memories = self.retriever.retrieve_relevant_memories(query)
        return self.retriever.format_context(relevant_memories)
        
    def update_decays(self) -> None:
        """
        Apply decay to all memories in each layer.
        """
        self.working_memory.apply_decay(exponential_decay)
        self.short_term_memory.apply_decay(exponential_decay)
        self.long_term_memory.apply_decay(exponential_decay)
        
    def manage_transitions(self) -> None:
        """
        Check decay strengths and move memories between layers or to archive based on transition rules.
        """
        # Working to Short-Term
        wm_memories = list(self.working_memory.memories)  # Create a copy to iterate safely
        for memory in wm_memories:
            if self.transition_rules.should_transition_to_short_term(memory.decay_strength, memory.importance_score):
                if self.short_term_memory.add_memory(memory):
                    self.working_memory.remove_memory(memory)
                    
        # Short-Term to Long-Term
        stm_memories = list(self.short_term_memory.memories)
        for memory in stm_memories:
            if self.transition_rules.should_transition_to_long_term(memory.decay_strength, memory.importance_score):
                if self.long_term_memory.add_memory(memory):
                    self.short_term_memory.remove_memory(memory)
                    
        # Long-Term to Archive
        ltm_memories = list(self.long_term_memory.memories)
        for memory in ltm_memories:
            if self.transition_rules.should_archive(memory.decay_strength, memory.importance_score):
                self.archive.archive_memory(memory)
                self.long_term_memory.remove_memory(memory)
        
    def __str__(self) -> str:
        return (f"Memory Manager:\n"
                f"{self.working_memory}\n"
                f"{self.short_term_memory}\n"
                f"{self.long_term_memory}\n"
                f"{self.archive}") 