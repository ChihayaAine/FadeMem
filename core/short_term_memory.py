from .memory_layer import MemoryLayer

class ShortTermMemory(MemoryLayer):
    def __init__(self, capacity: int = 20):
        """
        Initialize Short-Term Memory with a moderate capacity and decay rate.
        
        Args:
            capacity (int, optional): Maximum number of memory items. Defaults to 20.
        """
        super().__init__(name="Short-Term Memory", capacity=capacity, decay_rate=0.02) 