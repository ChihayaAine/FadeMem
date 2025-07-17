from typing import List, Optional
from .memory_item import MemoryItem

class MemoryLayer:
    def __init__(self, name: str, capacity: int, decay_rate: float):
        """
        Initialize a memory layer with a name, capacity, and base decay rate.
        
        Args:
            name (str): Name of the memory layer.
            capacity (int): Maximum number of memory items this layer can hold.
            decay_rate (float): Base decay rate for memories in this layer.
        """
        self.name = name
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.memories: List[MemoryItem] = []
        
    def add_memory(self, memory: MemoryItem) -> bool:
        """
        Add a memory item to this layer if within capacity.
        
        Args:
            memory (MemoryItem): The memory item to add.
            
        Returns:
            bool: True if added, False if at capacity.
        """
        if len(self.memories) < self.capacity:
            self.memories.append(memory)
            return True
        return False
        
    def remove_memory(self, memory: MemoryItem) -> bool:
        """
        Remove a specific memory item from this layer.
        
        Args:
            memory (MemoryItem): The memory item to remove.
            
        Returns:
            bool: True if removed, False if not found.
        """
        if memory in self.memories:
            self.memories.remove(memory)
            return True
        return False
        
    def get_memory(self, content: str) -> Optional[MemoryItem]:
        """
        Retrieve a memory item by its content.
        
        Args:
            content (str): The content to search for.
            
        Returns:
            Optional[MemoryItem]: The memory item if found, None otherwise.
        """
        for memory in self.memories:
            if memory.content == content:
                memory.access()
                return memory
        return None
        
    def apply_decay(self, decay_function) -> List[MemoryItem]:
        """
        Apply decay to all memories in this layer using the provided function.
        
        Args:
            decay_function: Function to calculate new decay strength.
            
        Returns:
            List[MemoryItem]: List of memories with updated decay strengths.
        """
        for memory in self.memories:
            new_strength = decay_function(memory, self.decay_rate)
            memory.decay_strength = new_strength
        return self.memories
        
    def get_low_priority_memories(self, threshold: float) -> List[MemoryItem]:
        """
        Get memories with decay strength below a threshold.
        
        Args:
            threshold (float): Decay strength threshold.
            
        Returns:
            List[MemoryItem]: List of low-priority memories.
        """
        return [m for m in self.memories if m.decay_strength < threshold]
        
    def __str__(self) -> str:
        return f"{self.name} Layer (Items: {len(self.memories)}/{self.capacity}, Decay Rate: {self.decay_rate})" 