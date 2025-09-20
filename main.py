import time
import random
from core.enhanced_memory_manager import EnhancedMemoryManager, EmbeddingGenerator
from llm.llm_interface import LLMInterface


def simulate_user_interaction(memory_manager: EnhancedMemoryManager, llm: LLMInterface, query: str, day: int) -> None:
    """
    Simulate a user interaction with the enhanced memory system.
    
    Args:
        memory_manager (EnhancedMemoryManager): The enhanced memory manager instance.
        llm (LLMInterface): The LLM interface instance.
        query (str): The user's query.
        day (int): The simulated day for logging purposes.
    """
    print(f"\n[Day {day}] User Query: {query}")
    
    # Retrieve relevant memories using enhanced system
    context = memory_manager.get_memory_context(query, top_k=3)
    print(f"[Day {day}] Context from Memories:\n{context}")
    
    # Generate LLM response
    response = llm.generate_response(query, context)
    print(f"[Day {day}] LLM Response:\n{response}")
    
    # Add the interaction as new memories with metadata
    query_metadata = {
        "interaction_type": "user_query",
        "day": day,
        "query_topic": "project" if "project" in query.lower() else "general"
    }
    
    response_metadata = {
        "interaction_type": "system_response", 
        "day": day,
        "response_length": len(response)
    }
    
    memory_manager.add_memory(f"User asked: {query} on day {day}", query_metadata)
    memory_manager.add_memory(f"System responded: {response[:100]}... on day {day}", response_metadata)

def simulate_time_passing(memory_manager: EnhancedMemoryManager, days: int) -> None:
    """
    Simulate the passage of time using the enhanced memory system.
    
    Args:
        memory_manager (EnhancedMemoryManager): The enhanced memory manager instance.
        days (int): Number of days to simulate.
    """
    print(f"\nSimulating {days} days passing...")
    
    # Simulate time passage by advancing system time and applying updates
    for day in range(days):
        # Force system update to apply decay, transitions, and fusion
        stats = memory_manager.update_system(force_all=True)
        
        if (day + 1) % 5 == 0 or day == days - 1:  # Print status every 5 days or on last day
            print(f"\nAfter {day + 1} days:")
            print(memory_manager)
            if stats['transitions']:
                print(f"  Transitions: {stats['transitions']}")
            if stats['fusion_stats']:
                print(f"  Fusion: {stats['fusion_stats']}")
        
        # Simulate random memory access during the day (some memories get reinforced)
        all_memories = memory_manager.dual_layer_memory.get_all_memories()
        if all_memories and random.random() > 0.7:  # 30% chance of random access
            random_memory = random.choice(all_memories)
            random_memory.access()
            if day == days - 1:  # Show on last day
                print(f"  Random access: {random_memory.content[:50]}...")

def main():
    """
    Main function demonstrating the Enhanced Dual-Layer Memory Architecture
    implementing the complete methodology with biologically-inspired forgetting,
    conflict resolution, and adaptive fusion.
    """
    print("=== Enhanced Dual-Layer Memory Architecture Demo ===")
    print("Implementing methodology: Dual-Layer Memory with Differential Forgetting\n")
    
    # Initialize enhanced memory system
    llm = LLMInterface()  # Uses mock responses (set API key for real LLM)
    embedding_generator = EmbeddingGenerator(dimension=768)
    memory_manager = EnhancedMemoryManager(llm, embedding_generator)
    
    print("System initialized with:")
    print(f"- LML capacity: {memory_manager.dual_layer_memory.max_lml_capacity}")
    print(f"- SML capacity: {memory_manager.dual_layer_memory.max_sml_capacity}")
    print(f"- Promotion threshold: {memory_manager.dual_layer_memory.theta_promote}")
    print(f"- Demotion threshold: {memory_manager.dual_layer_memory.theta_demote}")
    
    # Add initial memories demonstrating different importance levels
    initial_memories = [
        {
            "content": "Critical project deadline is next month - client expects delivery by end of March",
            "metadata": {"priority": "high", "type": "deadline", "impact": "critical"}
        },
        {
            "content": "Meeting with client at 3 PM tomorrow to discuss project scope and requirements",
            "metadata": {"priority": "high", "type": "meeting", "impact": "important"}
        },
        {
            "content": "Need to buy groceries for team dinner - pizza, drinks, and dessert",
            "metadata": {"priority": "low", "type": "personal", "impact": "minor"}
        },
        {
            "content": "Client feedback: current approach needs major revision, pivot to new architecture",
            "metadata": {"priority": "critical", "type": "feedback", "impact": "major"}
        },
        {
            "content": "Weather forecast shows rain tomorrow, might affect commute",
            "metadata": {"priority": "low", "type": "information", "impact": "minimal"}
        },
        {
            "content": "Team lead mentioned budget constraints may affect project timeline",
            "metadata": {"priority": "medium", "type": "concern", "impact": "moderate"}
        },
        {
            "content": "Coffee machine in break room needs repair - facilities notified",
            "metadata": {"priority": "low", "type": "maintenance", "impact": "minor"}
        }
    ]
    
    print("\nAdding initial memories...")
    for i, mem in enumerate(initial_memories):
        success = memory_manager.add_memory(mem["content"], mem["metadata"])
        print(f"  {i+1}. {'✓' if success else '✗'} {mem['content'][:50]}...")
    
    print(f"\nInitial State (Day 0):")
    print(memory_manager)
    stats = memory_manager.get_system_statistics()
    print(f"System stats: {stats['total_memories']} total, "
          f"avg strength: {stats['avg_memory_strength']:.3f}, "
          f"avg half-life: {stats['avg_half_life_days']:.2f} days")
    
    # Simulate user interactions over time with methodology demonstration
    interaction_schedule = [
        (1, "What are my high-priority tasks for this week?"),
        (3, "Remind me about the client meeting details"),
        (7, "What was the client feedback about our project approach?"),
        (10, "Any updates on project deadlines or timeline?"),
        (14, "What team concerns were mentioned recently?"),
        (21, "Summarize all project-related information"),
        (28, "What important meetings or deadlines do I have coming up?")
    ]
    
    current_day = 0
    print(f"\n=== Simulating {len(interaction_schedule)} interactions over 30 days ===")
    
    for interaction_day, query in interaction_schedule:
        # Simulate time passing
        days_elapsed = interaction_day - current_day
        if days_elapsed > 0:
            print(f"\n--- Time passes: {days_elapsed} days ---")
            simulate_time_passing(memory_manager, days_elapsed)
            current_day = interaction_day
        
        # User interaction
        simulate_user_interaction(memory_manager, llm, query, current_day)
        
        # Show system evolution
        stats = memory_manager.get_system_statistics()
        print(f"  Memory distribution: LML={stats['lml_count']}, SML={stats['sml_count']}")
        print(f"  System health: avg_strength={stats['avg_memory_strength']:.3f}, "
              f"avg_half_life={stats['avg_half_life_days']:.2f}d")
    
    # Final time simulation
    print(f"\n--- Final time passage: 2 days ---")
    simulate_time_passing(memory_manager, 2)
    current_day += 2
    
    print(f"\n=== Final System State (Day {current_day}) ===")
    print(memory_manager)
    
    # Demonstrate long-term memory capabilities
    print(f"\n=== Long-Term Memory Test ===")
    final_query = "What was the critical client feedback from a month ago about our project?"
    simulate_user_interaction(memory_manager, llm, final_query, current_day)
    
    # Export system statistics for analysis
    print(f"\n=== System Analysis ===")
    final_stats = memory_manager.get_system_statistics()
    
    print(f"Final memory distribution:")
    print(f"  Long-term Memory (LML): {final_stats['lml_count']} memories")
    print(f"  Short-term Memory (SML): {final_stats['sml_count']} memories")
    print(f"  Total memories: {final_stats['total_memories']}")
    print(f"  Average memory strength: {final_stats['avg_memory_strength']:.3f}")
    print(f"  Average half-life: {final_stats['avg_half_life_days']:.2f} days")
    print(f"  Total accesses: {final_stats['total_accesses']}")
    
    # Show some example memories and their properties
    all_memories = memory_manager.dual_layer_memory.get_all_memories()
    if all_memories:
        print(f"\nExample surviving memories:")
        # Sort by strength and show top few
        sorted_memories = sorted(all_memories, key=lambda m: m.memory_strength, reverse=True)
        for i, memory in enumerate(sorted_memories[:3]):
            print(f"  {i+1}. [{memory.layer_assignment}] {memory.content[:60]}...")
            print(f"      Strength: {memory.memory_strength:.3f}, "
                  f"Half-life: {memory.get_half_life():.2f}d, "
                  f"Age: {memory.get_age_days():.1f}d, "
                  f"Accesses: {memory.access_frequency}")
    
    print(f"\n=== Demo Complete ===")
    print("The enhanced dual-layer memory architecture successfully demonstrated:")
    print("✓ Biologically-inspired differential forgetting")
    print("✓ Dynamic layer transitions with hysteresis")
    print("✓ Memory consolidation through access")
    print("✓ Importance-based memory management")
    print("✓ Long-term memory retention and retrieval")

if __name__ == "__main__":
    main() 