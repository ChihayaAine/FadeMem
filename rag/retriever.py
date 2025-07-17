from typing import List, Optional
from ..core.memory_item import MemoryItem
from ..management.memory_manager import MemoryManager

class Retriever:
    def __init__(self, memory_manager: MemoryManager, top_k: int = 3):
        """
        Initialize the Retriever for fetching relevant memories.
        
        Args:
            memory_manager (MemoryManager): The memory manager instance to retrieve memories from.
            top_k (int, optional): Number of top relevant memories to retrieve. Defaults to 3.
        """
        self.memory_manager = memory_manager
        self.top_k = top_k
        # Mock embedding dictionary for tokens (simulating a pre-trained embedding space)
        self.mock_embeddings = self._initialize_mock_embeddings()
        
    def _initialize_mock_embeddings(self) -> dict:
        """
        Initialize a mock embedding dictionary for common tokens.
        This simulates a pre-trained embedding space for semantic similarity.
        
        Returns:
            dict: Dictionary mapping tokens to mock embedding vectors (lists of floats).
        """
        # Simple mock embeddings for a small vocabulary (in a real system, this would be a large pre-trained model)
        common_words = {
            'meeting': [0.8, 0.1, 0.2],
            'client': [0.7, 0.2, 0.1],
            'pm': [0.6, 0.1, 0.3],
            'buy': [0.2, 0.8, 0.1],
            'groceries': [0.1, 0.9, 0.2],
            'work': [0.3, 0.7, 0.3],
            'project': [0.8, 0.3, 0.1],
            'deadline': [0.7, 0.4, 0.2],
            'week': [0.5, 0.2, 0.4],
            'random': [0.1, 0.1, 0.8],
            'thought': [0.2, 0.2, 0.7],
            'weather': [0.1, 0.3, 0.8],
            'today': [0.4, 0.2, 0.5],
            'schedule': [0.7, 0.1, 0.3]
        }
        # Add some generic filler words with neutral embeddings
        filler_words = ['with', 'at', 'after', 'next', 'about']
        for word in filler_words:
            common_words[word] = [0.3, 0.3, 0.3]
        return common_words
        
    def _compute_mock_embedding(self, text: str) -> List[float]:
        """
        Compute a mock embedding for a piece of text by averaging embeddings of its tokens.
        
        Args:
            text (str): Text to compute embedding for.
            
        Returns:
            List[float]: Mock embedding vector for the text.
        """
        tokens = text.lower().split()
        embeddings = [self.mock_embeddings.get(token, [0.0, 0.0, 0.0]) for token in tokens]
        if not embeddings:
            return [0.0, 0.0, 0.0]
        # Average the embeddings (simple mock approach)
        dimension = len(embeddings[0])
        result = [0.0] * dimension
        for emb in embeddings:
            for i in range(dimension):
                result[i] += emb[i]
        return [val / len(embeddings) for val in result]
        
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1 (List[float]): First vector.
            vec2 (List[float]): Second vector.
            
        Returns:
            float: Cosine similarity score between 0 and 1.
        """
        if len(vec1) != len(vec2) or not vec1 or not vec2:
            return 0.0
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        return dot_product / (magnitude1 * magnitude2)
        
    def retrieve_relevant_memories(self, query: str) -> List[MemoryItem]:
        """
        Retrieve the most relevant memories based on a query.
        This implementation uses a mock semantic similarity score based on simulated embeddings,
        importance, and recency. In a real system, embeddings or NLP models would be used.
        
        Args:
            query (str): The query or context to match memories against.
            
        Returns:
            List[MemoryItem]: List of relevant memory items, sorted by relevance score.
        """
        # Combine memories from all layers
        all_memories = (
            self.memory_manager.working_memory.memories +
            self.memory_manager.short_term_memory.memories +
            self.memory_manager.long_term_memory.memories +
            self.memory_manager.archive.archived_memories
        )
        
        # Compute query embedding
        query_embedding = self._compute_mock_embedding(query)
        
        # Calculate relevance scores using mock semantic similarity
        relevant_memories = []
        for memory in all_memories:
            memory_embedding = self._compute_mock_embedding(memory.content)
            # Compute similarity between query and memory content
            similarity = self._cosine_similarity(query_embedding, memory_embedding)
            # Combine similarity with importance and recency
            recency_factor = 1 - min(memory.get_time_since_access() / 86400, 1.0)  # Normalize to 0-1, recent = higher
            relevance_score = (similarity * 0.5) + (memory.importance_score * 0.3) + (recency_factor * 0.2)
            relevant_memories.append((memory, relevance_score))
        
        # Sort by relevance score and take top_k
        relevant_memories.sort(key=lambda x: x[1], reverse=True)
        result = [mem[0] for mem in relevant_memories[:self.top_k]]
        
        # Mark accessed memories
        for mem in result:
            mem.access()
        return result
        
    def format_context(self, memories: List[MemoryItem]) -> str:
        """
        Format retrieved memories into a context string for LLM input.
        
        Args:
            memories (List[MemoryItem]): List of memory items to format.
            
        Returns:
            str: Formatted context string.
        """
        if not memories:
            return "No relevant memories found."
        context_lines = ["Relevant Memories:"]
        for i, mem in enumerate(memories, 1):
            context_lines.append(f"{i}. {mem.content} (Importance: {mem.importance_score:.2f})")
        return "\n".join(context_lines) 