import time
from typing import Dict, Any

class MemoryItem:
    def __init__(self, content: str, metadata: Dict[str, Any] = None):
        """
        Initialize a memory item with content and metadata.
        
        Args:
            content (str): The content of the memory.
            metadata (Dict[str, Any], optional): Additional information about the memory.
        """
        self.content = content
        self.metadata = metadata if metadata else {}
        self.decay_strength = 1.0  # Initial decay strength
        self.importance_score = 0.5  # Default importance score (0 to 1)
        self.access_count = 0
        self.last_accessed = time.time()
        self.created_at = time.time()
        
    def update_importance(self, score: float):
        """
        Update the importance score of the memory.
        
        Args:
            score (float): New importance score between 0 and 1.
        """
        self.importance_score = max(0.0, min(1.0, score))
        
    def access(self):
        """
        Record an access to this memory, reinforcing it.
        """
        self.access_count += 1
        self.last_accessed = time.time()
        self.decay_strength = min(1.0, self.decay_strength + 0.1)  # Reinforce on access
        self.importance_score = min(1.0, self.importance_score + 0.05)  # Slight boost
        
    def get_age(self) -> float:
        """
        Calculate the age of the memory in seconds.
        
        Returns:
            float: Age in seconds since creation.
        """
        return time.time() - self.created_at
        
    def get_time_since_access(self) -> float:
        """
        Calculate time since last access in seconds.
        
        Returns:
            float: Time in seconds since last access.
        """
        return time.time() - self.last_accessed
        
    def __str__(self) -> str:
        return f"Memory(content='{self.content[:50]}...', decay={self.decay_strength:.2f}, importance={self.importance_score:.2f})" 