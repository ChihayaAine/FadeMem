from .memory_layer import MemoryLayer

class WorkingMemory(MemoryLayer):
    def __init__(self, capacity: int = 7):
        """
        Initialize Working Memory with a small capacity and fast decay rate.
        
        Args:
            capacity (int, optional): Maximum number of memory items. Defaults to 7 (based on human cognitive limits).
        """
        super().__init__(name="Working Memory", capacity=capacity, decay_rate=0.05)
        
    def prioritize(self) -> None:
        """
        Prioritize memories in Working Memory based on recent access and importance.
        If over capacity, remove the least important and least recently accessed memories.
        """
        if len(self.memories) > self.capacity:
            self.memories.sort(key=lambda m: (m.importance_score, -m.get_time_since_access()), reverse=True)
            self.memories = self.memories[:self.capacity] 