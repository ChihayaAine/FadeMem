import time
import random
from management.memory_manager import MemoryManager
from llm.llm_interface import LLMInterface


def simulate_user_interaction(memory_manager: MemoryManager, llm: LLMInterface, query: str, day: int) -> None:
    """
    Simulate a user interaction with the system, retrieving context and generating a response.
    
    Args:
        memory_manager (MemoryManager): The memory manager instance.
        llm (LLMInterface): The LLM interface instance.
        query (str): The user's query.
        day (int): The simulated day for logging purposes.
    """
    print(f"\n[Day {day}] User Query: {query}")
    context = memory_manager.get_context_for_query(query)
    print(f"[Day {day}] Context from Memories:\n{context}")
    response = llm.generate_response(query, context)
    print(f"[Day {day}] LLM Response:\n{response}")
    # Add the query and response as new memories
    memory_manager.add_memory(f"User asked: {query} on day {day}", {"semantic_relevance": 0.7, "emotional_intensity": 0.5, "user_feedback": 0.6})
    memory_manager.add_memory(f"Response: {response[:50]}... on day {day}", {"semantic_relevance": 0.6, "emotional_intensity": 0.4, "user_feedback": 0.5})

def simulate_time_passing(memory_manager: MemoryManager, days: int, hours_per_day: int = 24) -> None:
    """
    Simulate the passage of time, applying decay and managing transitions.
    
    Args:
        memory_manager (MemoryManager): The memory manager instance.
        days (int): Number of days to simulate.
        hours_per_day (int, optional): Hours per day to simulate decay updates. Defaults to 24.
    """
    print(f"\nSimulating {days} days passing...")
    for day in range(days):
        for hour in range(hours_per_day):
            # Simulate time passing by updating decays every hour
            memory_manager.update_decays()
            if hour % 6 == 0:  # Manage transitions every 6 hours
                memory_manager.manage_transitions()
        if (day + 1) % 5 == 0:  # Print status every 5 days
            print(f"After {day + 1} days:")
            print(memory_manager)

def main():
    """
    Main function to demonstrate the Agent Memory system with RAG and LLM integration,
    simulating a long-term interaction scenario over multiple days with memory retention.
    """
    # Initialize memory manager and LLM interface
    memory_manager = MemoryManager(decay_threshold=0.1)
    # Optionally set an API key for real LLM calls (uncomment and replace with actual key)
    # api_key = "your-openai-api-key-here"
    # llm = LLMInterface(api_key=api_key)
    llm = LLMInterface()  # Without API key, uses mock responses
    
    # Add initial set of memories (simulating past events or knowledge)
    initial_memories = [
        {"content": "Meeting with client at 3 PM on project launch", "metadata": {"semantic_relevance": 0.8, "emotional_intensity": 0.6, "user_feedback": 0.7}},
        {"content": "Buy groceries for team dinner", "metadata": {"semantic_relevance": 0.4, "emotional_intensity": 0.2, "user_feedback": 0.3}},
        {"content": "Project deadline set for next month", "metadata": {"semantic_relevance": 0.9, "emotional_intensity": 0.8, "user_feedback": 0.9}},
        {"content": "Random thought about weather being nice", "metadata": {"semantic_relevance": 0.1, "emotional_intensity": 0.1, "user_feedback": 0.1}},
        {"content": "Client feedback: Need to revise project scope", "metadata": {"semantic_relevance": 0.85, "emotional_intensity": 0.7, "user_feedback": 0.8}}
    ]
    
    for mem in initial_memories:
        memory_manager.add_memory(mem["content"], mem["metadata"])
    
    print("Initial State (Day 0):")
    print(memory_manager)
    
    # Simulate a long-term interaction over 30 days with multiple user interactions
    interaction_days = [1, 3, 7, 14, 21, 28]  # Days when user interacts with the system
    queries = [
        "What do I have scheduled today regarding the project?",
        "Any updates on the client meeting?",
        "Remind me about the project deadline.",
        "What did the client say about the project scope?",
        "Do I have any tasks related to the team?",
        "Summarize my project-related memories."
    ]
    
    current_day = 0
    for interaction_day, query in zip(interaction_days, queries):
        # Simulate time passing until the interaction day
        days_to_pass = interaction_day - current_day
        if days_to_pass > 0:
            simulate_time_passing(memory_manager, days_to_pass)
            current_day = interaction_day
        
        # Simulate user interaction on the specified day
        simulate_user_interaction(memory_manager, llm, query, current_day)
        
        # Randomly access some memories to simulate reinforcement (user revisiting old info)
        if random.random() > 0.5 and memory_manager.working_memory.memories:
            memory_to_access = random.choice(memory_manager.working_memory.memories + memory_manager.short_term_memory.memories)
            print(f"[Day {current_day}] User revisited memory: {memory_to_access.content[:30]}...")
            memory_to_access.access()
    
    # Final state after all interactions
    simulate_time_passing(memory_manager, 2)  # Simulate 2 more days after last interaction
    print("\nFinal State (Day 30):")
    print(memory_manager)
    
    # Test long-term memory retrieval after a long time
    print("\nTesting Long-Term Memory Retrieval after 30 days:")
    final_query = "What was the client feedback on the project scope from a month ago?"
    simulate_user_interaction(memory_manager, llm, final_query, 30)

if __name__ == "__main__":
    main() 