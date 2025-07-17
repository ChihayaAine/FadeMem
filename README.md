# Agent Memory Project

## Overview
The Agent Memory project is inspired by human memory mechanisms, simulating the processes of remembering important information and forgetting less relevant details. It implements an active forgetting mechanism using decay strength, which decreases over time via an exponential function. The system is divided into three memory layers—Working Memory, Short-Term Memory, and Long-Term Memory—with periodic checks to manage memory decay and transitions between layers or to an archive. Additionally, it supports LLM-based agents with a Retrieval-Augmented Generation (RAG) module and an interface for calling large language models (LLMs).

### Key Features
- **Memory Layers**: Simulates Working Memory (short-term, high-access), Short-Term Memory (temporary storage), and Long-Term Memory (persistent storage).
- **Decay Mechanism**: Uses exponential decay adjusted by importance and access frequency; memories below a threshold (e.g., 0.1) may be archived or marked as low priority.
- **Importance Scoring**: Combines semantic relevance, emotional intensity, and user feedback to prioritize memories.
- **Reinforcement**: Accessing a memory reinforces it, resetting decay progress and boosting importance.
- **Archival**: Long-unused memories are compressed into summaries and moved to an archive layer.
- **RAG Module**: Supports Retrieval-Augmented Generation by retrieving relevant memories to provide context for LLM queries.
- **LLM Integration**: Provides an interface to call large language models for generating responses based on memory context.

## Project Structure
```
agent_memory/
├── core/                # Core memory mechanisms
│   ├── memory_item.py
│   ├── memory_layer.py
│   ├── working_memory.py
│   ├── short_term_memory.py
│   ├── long_term_memory.py
│   └── archive.py
├── utils/               # Helper functions
│   ├── decay_functions.py
│   └── importance_scorer.py
├── management/          # Memory management and transitions
│   ├── memory_manager.py
│   └── transition_rules.py
├── rag/                 # Retrieval-Augmented Generation module
│   └── retriever.py
├── llm/                 # Large Language Model integration
│   └── llm_interface.py
├── tests/               # Unit tests (placeholder)
├── main.py              # Entry point for demonstration
├── config.py            # Configuration settings
├── requirements.txt      # Dependencies
└── README.md            # Project documentation
```

## Installation
1. Clone or download this repository to your local machine.
2. Ensure you have Python 3.x installed.
3. Navigate to the project directory:
   ```bash
   cd agent_memory
   ```
4. (Optional) Install any future dependencies listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the main script to see a demonstration of the memory system with RAG and LLM integration:
```bash
python main.py
```
This will initialize a `MemoryManager`, add sample memories, simulate decay over time, demonstrate retrieval, and show how a query is answered using context from memories via an LLM.

## Customization
- Adjust decay rates, capacities, and thresholds in `config.py`.
- Extend `importance_scorer.py` with real semantic analysis (e.g., using NLP libraries).
- Add persistence by saving archived memories to disk.
- Implement actual LLM API calls in `llm_interface.py` with libraries like `openai` or `huggingface`.
- Enhance the `retriever.py` with semantic similarity using embeddings for better memory matching.

## Future Enhancements
- Implement real semantic relevance scoring using NLP models.
- Add a database or file storage for persistent memory.
- Introduce more sophisticated compression algorithms for archiving.
- Develop a user interface for interacting with the memory system.
- Integrate with actual LLM APIs for real response generation.

## License
This project is open-source and available under the MIT License (to be added). 