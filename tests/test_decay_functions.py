import time
import unittest
from ..core.memory_item import MemoryItem
from ..utils.decay_functions import exponential_decay

class TestDecayFunctions(unittest.TestCase):
    def setUp(self):
        """
        Set up a MemoryItem instance for testing decay functions.
        """
        self.memory = MemoryItem(content="Test memory content")
        self.base_decay_rate = 0.05
        
    def test_exponential_decay_initial(self):
        """
        Test exponential decay immediately after creation (minimal time elapsed).
        """
        new_strength = exponential_decay(self.memory, self.base_decay_rate)
        self.assertAlmostEqual(new_strength, 1.0, places=2)
        
    def test_exponential_decay_over_time(self):
        """
        Test exponential decay after some time has passed.
        """
        # Simulate time passing (mock by adjusting last_accessed)
        self.memory.last_accessed -= 3600  # 1 hour ago
        new_strength = exponential_decay(self.memory, self.base_decay_rate)
        self.assertLess(new_strength, 1.0)
        self.assertGreater(new_strength, 0.0)
        
    def test_exponential_decay_importance_effect(self):
        """
        Test that higher importance slows down decay.
        """
        self.memory.last_accessed -= 3600  # 1 hour ago
        self.memory.update_importance(0.9)  # High importance
        high_importance_strength = exponential_decay(self.memory, self.base_decay_rate)
        
        self.memory.update_importance(0.1)  # Low importance
        low_importance_strength = exponential_decay(self.memory, self.base_decay_rate)
        
        self.assertGreater(high_importance_strength, low_importance_strength)
        
    def test_exponential_decay_access_bonus(self):
        """
        Test that access count provides a bonus to decay strength.
        """
        self.memory.last_accessed -= 3600  # 1 hour ago
        self.memory.access_count = 5  # Multiple accesses
        with_bonus = exponential_decay(self.memory, self.base_decay_rate)
        
        self.memory.access_count = 0  # No accesses
        without_bonus = exponential_decay(self.memory, self.base_decay_rate)
        
        self.assertGreater(with_bonus, without_bonus)
        
    def test_exponential_decay_bounds(self):
        """
        Test that decay strength stays within bounds (0 to 1).
        """
        self.memory.last_accessed -= 86400 * 10  # Very old memory (10 days)
        new_strength = exponential_decay(self.memory, self.base_decay_rate)
        self.assertGreaterEqual(new_strength, 0.0)
        self.assertLessEqual(new_strength, 1.0)

if __name__ == '__main__':
    unittest.main() 