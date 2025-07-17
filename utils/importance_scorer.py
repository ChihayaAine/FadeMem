from typing import Dict, Any
from ..core.memory_item import MemoryItem

def calculate_importance(memory: MemoryItem) -> float:
    """
    Calculate the importance score of a memory based on multiple dimensions.
    
    Args:
        memory (MemoryItem): The memory item to score.
        
    Returns:
        float: Importance score between 0 and 1.
    """
    # Default weights for different dimensions
    weights = {
        'semantic_relevance': 0.3,
        'emotional_intensity': 0.4,
        'user_feedback': 0.3
    }
    
    # Extract scores from metadata or use defaults
    metadata = memory.metadata
    emotional_intensity = metadata.get('emotional_intensity', 0.5)  # Emotional impact
    user_feedback = metadata.get('user_feedback', 0.5)  # User-rated importance
    
    # Calculate semantic relevance with a mock implementation
    # In a real system, this would use NLP models or embeddings for context analysis
    semantic_relevance = metadata.get('semantic_relevance', calculate_mock_semantic_relevance(memory.content))
    
    # Calculate weighted score
    score = (
        weights['semantic_relevance'] * semantic_relevance +
        weights['emotional_intensity'] * emotional_intensity +
        weights['user_feedback'] * user_feedback
    )
    return max(0.0, min(1.0, score))

def calculate_mock_semantic_relevance(content: str) -> float:
    """
    Mock implementation to calculate semantic relevance based on keywords in content.
    This simulates an NLP-based relevance score by checking for important keywords.
    
    Args:
        content (str): The content of the memory item.
        
    Returns:
        float: Mock semantic relevance score between 0 and 1.
    """
    important_keywords = {
        'project': 0.9,
        'client': 0.8,
        'meeting': 0.7,
        'deadline': 0.9,
        'feedback': 0.8,
        'scope': 0.7,
        'launch': 0.7,
        'team': 0.6
    }
    content_lower = content.lower()
    relevance = 0.3  # Base relevance
    for keyword, value in important_keywords.items():
        if keyword in content_lower:
            relevance = max(relevance, value)
    return relevance 