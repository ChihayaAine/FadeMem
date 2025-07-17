from typing import List
from .memory_item import MemoryItem

class Archive:
    def __init__(self):
        """
        Initialize the Archive to store compressed or low-priority memories.
        """
        self.archived_memories: List[MemoryItem] = []
        
    def archive_memory(self, memory: MemoryItem, compress: bool = True) -> str:
        """
        Archive a memory, optionally compressing its content.
        
        Args:
            memory (MemoryItem): The memory item to archive.
            compress (bool, optional): Whether to compress the content. Defaults to True.
            
        Returns:
            str: The compressed or summarized content.
        """
        if compress:
            memory.content = self._compress_content(memory.content)
        memory.decay_strength = 0.01  # Very low priority
        self.archived_memories.append(memory)
        return memory.content
        
    def _compress_content(self, content: str) -> str:
        """
        Compress the content of a memory into a summary (mock implementation).
        
        Args:
            content (str): Original content.
            
        Returns:
            str: Compressed or summarized content.
        """
        return f"Summary: {content[:50]}..."
        
    def retrieve_archived(self, content_snippet: str) -> List[MemoryItem]:
        """
        Retrieve archived memories containing a content snippet.
        
        Args:
            content_snippet (str): Snippet to search for in archived content.
            
        Returns:
            List[MemoryItem]: List of matching archived memories.
        """
        return [m for m in self.archived_memories if content_snippet in m.content]
        
    def __str__(self) -> str:
        return f"Archive (Items: {len(self.archived_memories)})" 