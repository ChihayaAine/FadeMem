# Enhanced Dual-Layer Memory Architecture

This repository implements a comprehensive dual-layer memory architecture based on the methodology paper "Dual-Layer Memory Architecture with Differential Forgetting". The system mimics human memory patterns with biologically-inspired forgetting curves, conflict resolution, and adaptive fusion.

## Architecture Overview

### Core Components

#### 1. Memory Item (`core/memory_item.py`)
Represents individual memories following the paper's specification:
- **Memory Representation**: `m_i(t) = (c_i, s_i, v_i(t), τ_i, f_i)`
  - `c_i`: Content embedding vector
  - `s_i`: Original text content  
  - `v_i(t)`: Memory strength ∈ [0, 1]
  - `τ_i`: Creation timestamp
  - `f_i`: Access frequency with time-decayed rate

#### 2. Dual-Layer Memory (`core/dual_layer_memory.py`)
Implements the two-layer architecture:
- **Long-term Memory Layer (LML)**: High-importance memories with slow decay (β = 0.8)
- **Short-term Memory Layer (SML)**: Low-importance memories with rapid decay (β = 1.2)
- **Dynamic Transitions**: Based on importance thresholds with hysteresis

#### 3. Memory Conflict Resolution (`core/conflict_resolution.py`)
LLM-based semantic analysis for memory conflicts:
- **Compatible**: Coexistence with redundancy reduction
- **Contradictory**: Competitive dynamics with suppression
- **Subsumes/Subsumed**: Intelligent merging via LLM guidance

#### 4. Adaptive Memory Fusion (`core/adaptive_fusion.py`)
Temporal-semantic clustering and intelligent merging:
- **Clustering**: `C_k = {m_i : sim(c_i, c_k) > θ_fusion ∧ |τ_i - τ_k| < T_window}`
- **Fusion**: Preserves unique information, temporal progression, and causal relationships
- **Validation**: LLM verification of information preservation

## Key Features

### Biologically-Inspired Forgetting Curves
Memory decay follows differential exponential functions:
```
v_i(t) = v_i(0) · exp(-λ_i · (t - τ_i)^β_i)
```

Where:
- `λ_i = λ_base · exp(-μ · I_i(t))` (importance-adaptive decay rate)
- `β_i = 0.8` for LML (sub-linear decay)
- `β_i = 1.2` for SML (super-linear decay)

### Importance Scoring
Dynamic importance calculation:
```
I_i(t) = α·rel(c_i, Q_t) + β·f_i/(1+f_i) + γ·recency(τ_i, t)
```

Components:
- **Semantic relevance**: Cosine similarity with query context
- **Frequency term**: Saturating function preventing over-weighting
- **Recency**: Exponential decay `exp(-δ(t - τ_i))`

### Memory Consolidation
Access-based strengthening with spacing effects:
```
v_i(t^+) = v_i(t) + Δv · (1 - v_i(t)) · exp(-n_i/N)
```

### Half-Life Calculation
Memory half-life in days:
```
t_1/2(i) = (ln(2)/λ_i)^(1/β_i)
```

Reference values at I_i(t)=0:
- LML: ~11.25 days
- SML: ~5.02 days

## Configuration

All methodology parameters are defined in `config.py`:

### Layer Architecture
- `THETA_PROMOTE = 0.7`: Promotion threshold to LML
- `THETA_DEMOTE = 0.5`: Demotion threshold to SML
- `LONG_TERM_MEMORY_CAPACITY = 100`: LML capacity
- `SHORT_TERM_MEMORY_CAPACITY = 50`: SML capacity

### Biological Parameters
- `LAMBDA_BASE = 0.1`: Base decay rate (days⁻¹)
- `MU = 1.0`: Importance modulation parameter
- `BETA_LML = 0.8`: LML shape parameter (sub-linear decay)
- `BETA_SML = 1.2`: SML shape parameter (super-linear decay)

### Consolidation
- `DELTA_V = 0.2`: Base reinforcement strength
- `N_SPACING = 10`: Spacing effects parameter
- `W_WINDOW_DAYS = 7`: Access counting window

### Conflict Resolution
- `THETA_SIM = 0.7`: Similarity threshold for conflict detection
- `OMEGA = 0.3`: Redundancy penalty
- `RHO = 0.5`: Suppression strength

### Adaptive Fusion
- `THETA_FUSION = 0.6`: Fusion similarity threshold
- `T_WINDOW_DAYS = 7`: Temporal clustering window
- `THETA_PRESERVE = 0.8`: Information preservation threshold

## Usage

### Basic Example
```python
from core.enhanced_memory_manager import EnhancedMemoryManager
from llm.llm_interface import LLMInterface

# Initialize system
llm = LLMInterface()
memory_manager = EnhancedMemoryManager(llm)

# Add memories
memory_manager.add_memory("Important project deadline next month")
memory_manager.add_memory("Team meeting scheduled for tomorrow")

# Retrieve relevant memories
memories = memory_manager.retrieve_memories("What are my deadlines?")
context = memory_manager.get_memory_context("project status")

# System maintenance
stats = memory_manager.update_system()
```

### Running the Demo
```bash
cd memory/agent_memory
python main.py
```

The demo demonstrates:
- Memory addition with different importance levels
- Biologically-inspired decay over time
- Layer transitions with hysteresis
- Long-term memory retention
- Conflict resolution and fusion

## Memory Evolution Process

The complete memory evolution follows:
```
M_{t+Δt} = Fusion(Resolution(Decay(M_t, Δt) ∪ {m_new}))
```

### Step-by-Step Process:
1. **Decay Application**: Apply biological decay to existing memories
2. **Conflict Resolution**: Detect and resolve conflicts with new memory
3. **Layer Assignment**: Assign memories to appropriate layers
4. **Adaptive Fusion**: Cluster and merge related memories
5. **Transition Management**: Check and execute layer transitions
6. **Pruning**: Remove weak or dormant memories

## Performance Characteristics

### Expected Behavior
- **High-importance memories**: Migrate to LML, longer retention
- **Low-importance memories**: Remain in SML, faster decay
- **Accessed memories**: Strengthened through consolidation
- **Conflicting memories**: Resolved through LLM analysis
- **Related memories**: Fused to reduce redundancy

### System Metrics
- Memory distribution across layers
- Average memory strength and half-life
- Access frequency patterns
- Fusion and conflict resolution statistics

## Research Implementation

This implementation directly follows the methodology paper specifications:

### Mathematical Accuracy
- All equations implemented as specified
- Parameter values match paper recommendations
- Half-life calculations verified against expected values

### Biological Fidelity
- Differential decay rates between layers
- Consolidation through access patterns
- Spacing effects in memory strengthening
- Forgetting curves following Ebbinghaus patterns

### Computational Efficiency
- Batch processing for system operations
- Configurable update intervals
- Memory pruning to prevent unbounded growth
- LLM call optimization for conflict resolution

## Extensions and Customization

### Custom Embeddings
Replace `EmbeddingGenerator` with real embedding models:
```python
from sentence_transformers import SentenceTransformer

class RealEmbeddingGenerator:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def generate_embedding(self, text):
        return self.model.encode(text).tolist()
```

### Custom LLM Integration
Use actual LLM services:
```python
llm = LLMInterface(model_name="gpt-4", api_key="your-key")
memory_manager = EnhancedMemoryManager(llm)
```

### Parameter Tuning
Adjust parameters in `config.py` for specific use cases:
- Increase `LAMBDA_BASE` for faster forgetting
- Adjust layer thresholds for different retention patterns
- Modify fusion parameters for clustering behavior

## Testing and Validation

Run tests to verify implementation:
```bash
python -m pytest tests/
```

Tests cover:
- Memory item functionality
- Decay function accuracy
- Layer transition logic
- Conflict resolution scenarios
- Fusion algorithm correctness

## Future Enhancements

- **Multi-modal memories**: Support for images, audio
- **Hierarchical clustering**: Multi-level memory organization
- **Attention mechanisms**: Focus-based memory retrieval
- **Distributed storage**: Scalable memory persistence
- **Real-time learning**: Online parameter adaptation

## References

This implementation is based on the methodology paper:
"Dual-Layer Memory Architecture with Differential Forgetting"

The system provides a complete, research-grade implementation suitable for:
- Academic research
- Production AI systems
- Memory-augmented applications
- Long-term conversation agents
- Knowledge management systems
