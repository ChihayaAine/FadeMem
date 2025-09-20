import time
import math
from typing import Dict, Any, List, Optional

class MemoryItem:
    def __init__(self, content: str, content_embedding: List[float], metadata: Dict[str, Any] = None):
        """
        Initialize a memory item following the dual-layer memory architecture.
        
        Memory representation: m_i(t) = (c_i, s_i, v_i(t), τ_i, f_i)
        
        Args:
            content (str): The original text content (s_i)
            content_embedding (List[float]): The content embedding vector (c_i)
            metadata (Dict[str, Any], optional): Additional information about the memory.
        """
        # Core memory components from methodology
        self.content = content  # s_i: original text
        self.content_embedding = content_embedding  # c_i: content embedding
        self.memory_strength = 1.0  # v_i(t): memory strength ∈ [0, 1]
        self.creation_timestamp = time.time()  # τ_i: creation timestamp (in seconds, convert to days when needed)
        self.access_frequency = 0  # f_i: raw access count
        self.time_decayed_access_rate = 0.0  # ṽ_i: exponentially time-decayed access rate
        
        # Additional tracking variables
        self.metadata = metadata if metadata else {}
        self.access_timestamps = []  # Track all access times for time-decayed rate calculation
        self.last_accessed = time.time()
        self.layer_assignment = None  # Will be 'LML' or 'SML'
        
        # Decay parameters (will be set based on layer assignment)
        self.lambda_i = None  # Decay rate λ_i
        self.beta_i = None    # Shape parameter β_i
        
    def access(self, current_time: Optional[float] = None, kappa: float = 0.1, 
               delta_v: float = 0.2, N: int = 10, W: int = 7):
        """
        Record an access to this memory with consolidation effects.
        
        Implements: v_i(t^+) = v_i(t) + Δv * (1 - v_i(t)) * exp(-n_i/N)
        
        Args:
            current_time (float, optional): Current timestamp. Defaults to current time.
            kappa (float): Time decay parameter for access rate calculation
            delta_v (float): Base reinforcement strength
            N (int): Parameter for diminishing returns (spacing effects)
            W (int): Sliding window in days for access counting
        """
        if current_time is None:
            current_time = time.time()
            
        # Record access
        self.access_frequency += 1
        self.access_timestamps.append(current_time)
        self.last_accessed = current_time
        
        # Update time-decayed access rate: ṽ_i = Σ_j exp(-κ(t-t_j))
        self.time_decayed_access_rate = sum(
            math.exp(-kappa * (current_time - timestamp) / 86400)  # Convert to days
            for timestamp in self.access_timestamps
        )
        
        # Memory consolidation with spacing effects
        # Count accesses within sliding window W
        window_start = current_time - (W * 86400)  # W days in seconds
        n_i = sum(1 for t in self.access_timestamps if t >= window_start)
        
        # Apply consolidation: v_i(t^+) = v_i(t) + Δv * (1 - v_i(t)) * exp(-n_i/N)
        reinforcement = delta_v * (1 - self.memory_strength) * math.exp(-n_i / N)
        self.memory_strength = min(1.0, self.memory_strength + reinforcement)
        
    def calculate_importance(self, query_context: Optional[List[float]] = None, 
                           alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3,
                           current_time: Optional[float] = None) -> float:
        """
        Calculate memory importance score following the methodology.
        
        I_i(t) = α·rel(c_i, Q_t) + β·f_i/(1+f_i) + γ·recency(τ_i, t)
        
        Args:
            query_context (List[float], optional): Recent context embedding Q_t
            alpha (float): Weight for semantic relevance
            beta (float): Weight for frequency term
            gamma (float): Weight for recency
            current_time (float, optional): Current timestamp
            
        Returns:
            float: Importance score I_i(t)
        """
        if current_time is None:
            current_time = time.time()
            
        # Semantic relevance component
        if query_context is not None:
            relevance = self._cosine_similarity(self.content_embedding, query_context)
        else:
            relevance = 0.5  # Default when no context available
            
        # Frequency component with saturation: f_i/(1+f_i)
        # Use time-decayed access rate instead of raw count
        frequency_score = self.time_decayed_access_rate / (1 + self.time_decayed_access_rate)
        
        # Recency component: exp(-δ(t - τ_i))
        delta = 0.1  # Recency decay parameter
        age_days = (current_time - self.creation_timestamp) / 86400  # Convert to days
        recency_score = math.exp(-delta * age_days)
        
        # Combine components
        importance = alpha * relevance + beta * frequency_score + gamma * recency_score
        return max(0.0, min(1.0, importance))
        
    def update_decay_parameters(self, lambda_base: float = 0.1, mu: float = 1.0):
        """
        Update decay parameters based on current importance and layer assignment.
        
        λ_i = λ_base * exp(-μ * I_i(t))
        β_i depends on layer assignment
        
        Args:
            lambda_base (float): Base decay rate
            mu (float): Importance modulation parameter
        """
        # Calculate current importance (with default context)
        importance = self.calculate_importance()
        
        # Update decay rate: λ_i = λ_base * exp(-μ * I_i(t))
        self.lambda_i = lambda_base * math.exp(-mu * importance)
        
        # Set shape parameter based on layer assignment
        if self.layer_assignment == 'LML':
            self.beta_i = 0.8  # Sub-linear decay for long-term memories
        elif self.layer_assignment == 'SML':
            self.beta_i = 1.2  # Super-linear decay for short-term memories
        else:
            self.beta_i = 1.0  # Default exponential decay
            
    def apply_biological_decay(self, current_time: Optional[float] = None) -> float:
        """
        Apply biologically-inspired forgetting curve decay.
        
        v_i(t) = v_i(0) * exp(-λ_i * (t - τ_i)^β_i)
        
        Args:
            current_time (float, optional): Current timestamp
            
        Returns:
            float: New memory strength after decay
        """
        if current_time is None:
            current_time = time.time()
            
        if self.lambda_i is None or self.beta_i is None:
            self.update_decay_parameters()
            
        # Calculate time elapsed in days
        time_elapsed_days = (current_time - self.creation_timestamp) / 86400
        
        # Apply forgetting curve: v_i(t) = v_i(0) * exp(-λ_i * (t - τ_i)^β_i)
        initial_strength = 1.0  # v_i(0)
        decay_factor = math.exp(-self.lambda_i * (time_elapsed_days ** self.beta_i))
        new_strength = initial_strength * decay_factor
        
        self.memory_strength = max(0.0, min(1.0, new_strength))
        return self.memory_strength
        
    def get_half_life(self) -> float:
        """
        Calculate memory half-life in days.
        
        t_1/2(i) = (ln(2)/λ_i)^(1/β_i)
        
        Returns:
            float: Half-life in days
        """
        if self.lambda_i is None or self.beta_i is None:
            self.update_decay_parameters()
            
        return (math.log(2) / self.lambda_i) ** (1 / self.beta_i)
        
    def get_age_days(self, current_time: Optional[float] = None) -> float:
        """
        Calculate the age of the memory in days.
        
        Args:
            current_time (float, optional): Current timestamp
            
        Returns:
            float: Age in days since creation.
        """
        if current_time is None:
            current_time = time.time()
        return (current_time - self.creation_timestamp) / 86400
        
    def get_time_since_access_days(self, current_time: Optional[float] = None) -> float:
        """
        Calculate time since last access in days.
        
        Args:
            current_time (float, optional): Current timestamp
            
        Returns:
            float: Time in days since last access.
        """
        if current_time is None:
            current_time = time.time()
        return (current_time - self.last_accessed) / 86400
        
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two L2-normalized embeddings.
        
        Args:
            vec1 (List[float]): First embedding vector
            vec2 (List[float]): Second embedding vector
            
        Returns:
            float: Cosine similarity score
        """
        if len(vec1) != len(vec2) or not vec1 or not vec2:
            return 0.0
            
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
            
        return dot_product / (magnitude1 * magnitude2)
        
    def __str__(self) -> str:
        layer_str = f", Layer: {self.layer_assignment}" if self.layer_assignment else ""
        return (f"Memory(content='{self.content[:50]}...', "
                f"strength={self.memory_strength:.3f}, "
                f"access_freq={self.access_frequency}, "
                f"half_life={self.get_half_life():.2f}d{layer_str})") 