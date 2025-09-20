# FadeMem: Biologically-Inspired Agent Memory Architecture

## Overview
FadeMem is a biologically-inspired agent memory architecture that addresses critical limitations in current AI systems by incorporating active forgetting mechanisms that mirror human cognitive efficiency. Unlike existing agent memory systems that suffer from either catastrophic forgetting at context boundaries or information overload within them, FadeMem implements differential decay rates across a dual-layer memory hierarchy to achieve superior multi-hop reasoning and retrieval with 45% storage reduction.

The system addresses the fundamental flaw in current agent memory architectures: the lack of selective forgetting mechanisms. While human memory elegantly balances retention and forgetting through natural decay processes, current AI systems employ binary retention strategies that preserve everything or lose it entirely. FadeMem bridges this gap by implementing adaptive exponential decay functions modulated by semantic relevance, access frequency, and temporal patterns.

### Key Features
- **Dual-Layer Memory Hierarchy**: Long-term Memory Layer (LML) with slow decay (β=0.8) and Short-term Memory Layer (SML) with rapid decay (β=1.2)
- **Adaptive Forgetting Mechanisms**: Biologically-inspired exponential decay curves following Ebbinghaus's forgetting curve principles
- **Dynamic Layer Transitions**: Importance-based memory migration with hysteresis (θ_promote=0.7, θ_demote=0.3) to prevent oscillation
- **LLM-Guided Conflict Resolution**: Semantic analysis classifying memory relationships as compatible/contradictory/subsumes/subsumed
- **Intelligent Memory Fusion**: Temporal-semantic clustering with LLM-guided merging that preserves causal relationships
- **Memory Consolidation**: Access-based strengthening with spacing effects and diminishing returns
- **Storage Efficiency**: Achieves 45% storage reduction while maintaining 82.1% retention of critical facts
- **Superior Performance**: Outperforms baselines on Multi-Session Chat, LoCoMo, and LTI-Bench datasets

## Project Structure
```
agent_memory/
├── core/                           # Enhanced dual-layer memory architecture
│   ├── memory_item.py             # Memory representation with biological decay
│   ├── dual_layer_memory.py       # LML/SML architecture with transitions
│   ├── enhanced_memory_manager.py # Complete system integration
│   ├── conflict_resolution.py     # LLM-based memory conflict analysis
│   └── adaptive_fusion.py         # Temporal-semantic memory fusion
├── rag/                           # Advanced RAG system with LangGraph
│   ├── __init__.py
│   ├── embedding_service.py       # OpenAI text-embedding-small-3 integration
│   ├── vector_store.py            # ChromaDB/FAISS vector database
│   ├── rag_workflow.py            # LangGraph-based RAG workflow
│   └── retrieval_chain.py         # Complete RAG chain integration
├── evaluation/                    # System evaluation and validation
│   ├── __init__.py
│   ├── methodology_validator.py   # Mathematical accuracy validation
│   └── performance_benchmarks.py  # Comprehensive performance testing
├── llm/                          # LLM integration
│   └── llm_interface.py          # OpenAI GPT integration with fallback
├── main.py                       # Enhanced demo with multiple modes
├── config.py                     # Complete methodology parameters
├── requirements.txt              # Full dependency list
└── README.md                     # Project documentation
```

## Installation

### Prerequisites
- Python 3.10 or higher
- OpenAI API key (optional, for real embeddings and LLM calls)

### Setup
1. Clone or download this repository:
   ```bash
   git clone https://github.com/ChihayaAine/FadeMem.git
   cd FadeMem
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Set OpenAI API key for full functionality:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   
   Note: The system includes fallback mechanisms for local operation without API access.

## Usage

### Quick Start
Run the enhanced demo with multiple options:
```bash
python main.py
```

Choose from:
1. **Basic FadeMem demonstration** - Core biologically-inspired memory system with adaptive forgetting
2. **Complete system with conflict resolution** - Full system with LLM-guided memory fusion and conflict resolution
3. **Performance benchmarks and validation** - Evaluation on MSC, LoCoMo, and LTI-Bench datasets


## System Configuration

### Key Parameters (config.py)
- `LAMBDA_BASE = 0.1` - Base decay rate (days⁻¹) following human memory patterns
- `THETA_PROMOTE = 0.7` - LML promotion threshold for high-importance memories
- `THETA_DEMOTE = 0.3` - SML demotion threshold with hysteresis to prevent oscillation
- `BETA_LML = 0.8` - LML shape parameter (sub-linear decay for long-term memories)
- `BETA_SML = 1.2` - SML shape parameter (super-linear decay for short-term memories)
- `THETA_FUSION = 0.75` - Memory fusion threshold for temporal-semantic clustering
- `MU = 1.0` - Importance modulation parameter for adaptive decay rates

### Embedding Model
The system uses OpenAI's `text-embedding-3-small` model:
- Dimensions: 1536
- Context length: 8191 tokens
- Optimized for semantic similarity computation and memory fusion operations

### Vector Database Options
- **ChromaDB** (default) - Persistent vector storage
- **FAISS** - High-performance similarity search
- **Simple** - In-memory fallback for testing

## FadeMem Methodology

### Core Architecture
FadeMem implements a dual-layer memory hierarchy where each memory `m_i(t) = (c_i, s_i, v_i(t), τ_i, f_i)` contains:
- `c_i`: Content embedding vector
- `s_i`: Original text content  
- `v_i(t)`: Memory strength [0,1]
- `τ_i`: Creation timestamp
- `f_i`: Access frequency

### Mathematical Formulations

**Memory Importance Score:**
```
I_i(t) = α·rel(c_i, Q_t) + β·f_i/(1+f_i) + γ·recency(τ_i, t)
```

**Biologically-Inspired Forgetting Curve:**
```
v_i(t) = v_i(0) · exp(-λ_i · (t - τ_i)^β_i)
```

**Adaptive Decay Rate:**
```
λ_i = λ_base · exp(-μ · I_i(t))
```

**Memory Consolidation:**
```
v_i(t+) = v_i(t) + Δv · (1 - v_i(t)) · exp(-n_i/N)
```

### Memory Half-Life Analysis
- **LML memories**: ~11.25 days half-life at I_i(t)=0 (sub-linear decay)
- **SML memories**: ~5.02 days half-life at I_i(t)=0 (super-linear decay)
- Important memories exhibit 3-5× slower decay than baseline
- Spacing effects implemented through diminishing returns in consolidation

### Conflict Resolution Strategies
The system classifies memory relationships and applies corresponding strategies:
- **Compatible**: Coexistence with redundancy penalty
- **Contradictory**: Temporal suppression favoring recent information
- **Subsumes/Subsumed**: LLM-guided intelligent merging

### Performance Results
- **Storage Reduction**: 45% reduction while maintaining quality
- **Critical Fact Retention**: 82.1% retention of important information  
- **Multi-hop Reasoning**: Superior performance on LoCoMo (F1=29.43 vs Mem0's 28.37)
- **Retrieval Quality**: 77.2% RP@10 on Multi-Session Chat dataset
- **Temporal Consistency**: 0.82 TCS score across multi-session interactions

## Experimental Validation

### Evaluation Datasets
- **Multi-Session Chat (MSC)**: 5,000 multi-session dialogues spanning up to 5 sessions per user
- **LoCoMo**: Long-context evaluation focusing on multi-hop reasoning across extended contexts
- **LTI-Bench**: Synthetic 30-day agent-user interactions with 10,780 sequences and controlled temporal dependencies

### Baseline Comparisons
FadeMem consistently outperforms existing approaches:
- **Fixed-window methods** (4K, 8K, 16K tokens with FIFO eviction)
- **RAG-based systems** (LangChain Memory with default configurations)
- **Specialized agent memory** (Mem0 unified memory layers, MemGPT hierarchical management)

### Key Findings
1. **Selective Forgetting is Essential**: Demonstrates that forgetting irrelevant information prevents overload and maintains relevance
2. **Biological Inspiration Works**: Ebbinghaus-inspired decay curves effectively balance retention and efficiency
3. **Component Synergy**: Ablation studies show each component (dual-layer, fusion, conflict resolution) contributes significantly
4. **Scalability**: Maintains performance while reducing storage requirements by nearly half

## License
This project is open-source and available under the MIT License (to be added). 