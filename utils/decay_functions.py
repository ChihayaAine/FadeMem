import math
from ..core.memory_item import MemoryItem

def exponential_decay(memory: MemoryItem, base_decay_rate: float) -> float:
    """
    Calculate new decay strength using exponential decay, adjusted by importance and access frequency.
    
    Args:
        memory (MemoryItem): The memory item to calculate decay for.
        base_decay_rate (float): Base decay rate of the memory layer.
        
    Returns:
        float: New decay strength between 0 and 1.
    """
    time_elapsed = memory.get_time_since_access()
    # Adjust decay rate based on importance (higher importance = slower decay)
    adjusted_rate = base_decay_rate * (1 - memory.importance_score * 0.5)
    # Exponential decay formula: strength = initial * e^(-rate * time)
    new_strength = memory.decay_strength * math.exp(-adjusted_rate * time_elapsed / 3600)  # Time in hours
    # Further adjust based on access frequency (more accesses = slower decay)
    access_bonus = min(0.2, memory.access_count * 0.01)
    new_strength = min(1.0, new_strength + access_bonus)
    return max(0.0, new_strength) 