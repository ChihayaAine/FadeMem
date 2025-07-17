import unittest
from ..core.memory_item import MemoryItem
from ..core.working_memory import WorkingMemory
from ..core.short_term_memory import ShortTermMemory
from ..core.long_term_memory import LongTermMemory
from ..core.archive import Archive
from ..utils.decay_functions import exponential_decay

class TestMemoryLayers(unittest.TestCase):
    def setUp(self):
        """
        Set up memory layer instances and sample memory items for testing.
        """
        self.working_memory = WorkingMemory(capacity=3)
        self.short_term_memory = ShortTermMemory(capacity=5)
        self.long_term_memory = LongTermMemory(capacity=10)
        self.archive = Archive()
        self.memory1 = MemoryItem(content="Memory 1")
        self.memory2 = MemoryItem(content="Memory 2")
        self.memory3 = MemoryItem(content="Memory 3")
        
    def test_working_memory_capacity(self):
        """
        Test capacity limits and prioritization in WorkingMemory.
        """
        self.assertTrue(self.working_memory.add_memory(self.memory1))
        self.assertTrue(self.working_memory.add_memory(self.memory2))
        self.assertTrue(self.working_memory.add_memory(self.memory3))
        # Adding beyond capacity should return False
        memory4 = MemoryItem(content="Memory 4")
        self.assertFalse(self.working_memory.add_memory(memory4))
        self.assertEqual(len(self.working_memory.memories), 3)
        
        # Test prioritization (based on importance and recency)
        self.memory1.update_importance(0.9)
        self.memory2.update_importance(0.2)
        self.memory3.update_importance(0.5)
        self.working_memory.prioritize()  # Should keep top 3, no change needed
        self.assertEqual(len(self.working_memory.memories), 3)
        self.assertIn(self.memory1, self.working_memory.memories)
        
    def test_short_term_memory_capacity(self):
        """
        Test capacity limits in ShortTermMemory.
        """
        for i in range(5):
            memory = MemoryItem(content=f"Short-term {i}")
            self.assertTrue(self.short_term_memory.add_memory(memory))
        self.assertEqual(len(self.short_term_memory.memories), 5)
        # Beyond capacity
        memory6 = MemoryItem(content="Short-term 6")
        self.assertFalse(self.short_term_memory.add_memory(memory6))
        self.assertEqual(len(self.short_term_memory.memories), 5)
        
    def test_long_term_memory_capacity(self):
        """
        Test capacity limits in LongTermMemory.
        """
        for i in range(10):
            memory = MemoryItem(content=f"Long-term {i}")
            self.assertTrue(self.long_term_memory.add_memory(memory))
        self.assertEqual(len(self.long_term_memory.memories), 10)
        # Beyond capacity
        memory11 = MemoryItem(content="Long-term 11")
        self.assertFalse(self.long_term_memory.add_memory(memory11))
        self.assertEqual(len(self.long_term_memory.memories), 10)
        
    def test_archive_functionality(self):
        """
        Test archiving memories and retrieval from Archive.
        """
        memory = MemoryItem(content="Archive this memory")
        archived_content = self.archive.archive_memory(memory, compress=True)
        self.assertIn("Summary: Archive this memory", archived_content)
        self.assertEqual(len(self.archive.archived_memories), 1)
        self.assertEqual(self.archive.archived_memories[0].decay_strength, 0.01)
        
        # Test retrieval from archive
        retrieved = self.archive.retrieve_archived("Archive this memory")
        self.assertEqual(len(retrieved), 1)
        self.assertEqual(retrieved[0], memory)
        
        retrieved_empty = self.archive.retrieve_archived("Non-existent")
        self.assertEqual(len(retrieved_empty), 0)
        
    def test_memory_retrieval(self):
        """
        Test retrieving memories from layers.
        """
        self.working_memory.add_memory(self.memory1)
        retrieved = self.working_memory.get_memory("Memory 1")
        self.assertEqual(retrieved, self.memory1)
        self.assertEqual(retrieved.access_count, 1)  # Access count should increment
        
        not_found = self.working_memory.get_memory("Non-existent")
        self.assertIsNone(not_found)
        
    def test_decay_application(self):
        """
        Test applying decay to memories in a layer.
        """
        self.working_memory.add_memory(self.memory1)
        self.memory1.last_accessed -= 3600  # Simulate 1 hour ago
        initial_strength = self.memory1.decay_strength
        
        updated_memories = self.working_memory.apply_decay(exponential_decay)
        self.assertEqual(len(updated_memories), 1)
        self.assertLess(updated_memories[0].decay_strength, initial_strength)
        
    def test_low_priority_memories(self):
        """
        Test identifying low-priority memories based on decay strength.
        """
        self.short_term_memory.add_memory(self.memory1)
        self.short_term_memory.add_memory(self.memory2)
        self.memory1.decay_strength = 0.05  # Below threshold
        self.memory2.decay_strength = 0.5   # Above threshold
        
        low_priority = self.short_term_memory.get_low_priority_memories(threshold=0.1)
        self.assertEqual(len(low_priority), 1)
        self.assertEqual(low_priority[0], self.memory1)

if __name__ == '__main__':
    unittest.main() 