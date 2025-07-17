import time
import unittest
from ..core.memory_item import MemoryItem

class TestMemoryItem(unittest.TestCase):
    def setUp(self):
        """
        Set up a MemoryItem instance for testing.
        """
        self.memory = MemoryItem(content="Test memory content", metadata={"test": "data"})
        
    def test_initialization(self):
        """
        Test that a MemoryItem is initialized correctly.
        """
        self.assertEqual(self.memory.content, "Test memory content")
        self.assertEqual(self.memory.metadata, {"test": "data"})
        self.assertEqual(self.memory.decay_strength, 1.0)
        self.assertEqual(self.memory.importance_score, 0.5)
        self.assertEqual(self.memory.access_count, 0)
        self.assertTrue(isinstance(self.memory.created_at, float))
        self.assertTrue(isinstance(self.memory.last_accessed, float))
        
    def test_update_importance(self):
        """
        Test updating the importance score of a MemoryItem.
        """
        self.memory.update_importance(0.8)
        self.assertEqual(self.memory.importance_score, 0.8)
        # Test boundary conditions
        self.memory.update_importance(1.5)
        self.assertEqual(self.memory.importance_score, 1.0)
        self.memory.update_importance(-0.5)
        self.assertEqual(self.memory.importance_score, 0.0)
        
    def test_access(self):
        """
        Test accessing a MemoryItem, which should reinforce it.
        """
        initial_strength = self.memory.decay_strength
        initial_importance = self.memory.importance_score
        initial_access_count = self.memory.access_count
        initial_last_accessed = self.memory.last_accessed
        
        time.sleep(0.1)  # Small delay to ensure timestamp changes
        self.memory.access()
        
        self.assertEqual(self.memory.access_count, initial_access_count + 1)
        self.assertGreater(self.memory.last_accessed, initial_last_accessed)
        self.assertGreater(self.memory.decay_strength, initial_strength)
        self.assertGreater(self.memory.importance_score, initial_importance)
        self.assertLessEqual(self.memory.decay_strength, 1.0)
        self.assertLessEqual(self.memory.importance_score, 1.0)
        
    def test_get_age(self):
        """
        Test calculating the age of a MemoryItem.
        """
        time.sleep(0.1)  # Small delay to ensure age > 0
        age = self.memory.get_age()
        self.assertGreater(age, 0.0)
        
    def test_get_time_since_access(self):
        """
        Test calculating time since last access of a MemoryItem.
        """
        time.sleep(0.1)  # Small delay to ensure time since access > 0
        time_since_access = self.memory.get_time_since_access()
        self.assertGreater(time_since_access, 0.0)
        
    def test_str_representation(self):
        """
        Test the string representation of a MemoryItem.
        """
        str_rep = str(self.memory)
        self.assertIn("Test memory content", str_rep)
        self.assertIn("decay=1.00", str_rep)
        self.assertIn("importance=0.50", str_rep)

if __name__ == '__main__':
    unittest.main() 