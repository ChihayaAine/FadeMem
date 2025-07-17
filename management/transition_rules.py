# Transition Rules for Memory Management

class TransitionRules:
    def __init__(self):
        """
        Initialize transition rules for memory management.
        """
        pass
    
    def should_transition_to_short_term(self, decay_strength: float, importance: float) -> bool:
        """
        Determine if a memory should move from Working to Short-Term Memory.
        
        Args:
            decay_strength (float): Current decay strength of the memory.
            importance (float): Importance score of the memory.
            
        Returns:
            bool: True if the memory should transition.
        """
        return decay_strength < 0.3 and importance > 0.4
        
    def should_transition_to_long_term(self, decay_strength: float, importance: float) -> bool:
        """
        Determine if a memory should move from Short-Term to Long-Term Memory.
        
        Args:
            decay_strength (float): Current decay strength of the memory.
            importance (float): Importance score of the memory.
            
        Returns:
            bool: True if the memory should transition.
        """
        return decay_strength < 0.2 and importance > 0.6
        
    def should_archive(self, decay_strength: float, importance: float) -> bool:
        """
        Determine if a memory should be archived.
        
        Args:
            decay_strength (float): Current decay strength of the memory.
            importance (float): Importance score of the memory.
            
        Returns:
            bool: True if the memory should be archived.
        """
        return decay_strength < 0.1 