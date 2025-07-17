from .memory_layer import MemoryLayer

class LongTermMemory(MemoryLayer):
    def __init__(self, capacity: int = 100):
        """
        Initialize Long-Term Memory with a large capacity and slow decay rate.
        
        Args:
            capacity (int, optional): Maximum number of memory items. Defaults to 100.
        """
        super().__init__(name="Long-Term Memory", capacity=capacity, decay_rate=0.005)
        
    def compress_old_memories(self, age_threshold: float = 30 * 86400) -> int:
        """
        Compress memories older than a specified age to save space while retaining core information.
        This simulates how human long-term memory might summarize or abstract old information.
        
        Args:
            age_threshold (float, optional): Age in seconds beyond which memories are compressed. Defaults to 30 days.
            
        Returns:
            int: Number of memories compressed.
        """
        compressed_count = 0
        for memory in self.memories:
            if memory.get_age() > age_threshold and not memory.content.startswith("Compressed: "):
                memory.content = f"Compressed: {memory.content[:50]}... (original length: {len(memory.content)})"
                memory.decay_strength = max(0.1, memory.decay_strength - 0.05)  # Slightly reduce strength
                compressed_count += 1
        return compressed_count
        
    def reevaluate_importance(self) -> int:
        """
        Re-evaluate the importance of long-term memories based on access frequency.
        Memories that are frequently accessed are given a slight importance boost.
        
        Returns:
            int: Number of memories whose importance was adjusted.
        """
        adjusted_count = 0
        for memory in self.memories:
            if memory.access_count > 5:  # Arbitrary threshold for frequent access
                old_importance = memory.importance_score
                memory.importance_score = min(1.0, memory.importance_score + 0.1)
                if memory.importance_score != old_importance:
                    adjusted_count += 1
                memory.decay_strength = min(1.0, memory.decay_strength + 0.05)  # Slight reinforcement
        return adjusted_count
        
    def prioritize(self) -> None:
        """
        Override prioritize method to ensure long-term memory retains the most important and frequently accessed memories.
        If over capacity, remove the least important and least accessed memories.
        """
        if len(self.memories) > self.capacity:
            self.memories.sort(key=lambda m: (m.importance_score, m.access_count), reverse=True)
            self.memories = self.memories[:self.capacity] 