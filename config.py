# Configuration settings for Dual-Layer Memory Architecture
# Following the methodology paper parameters

# =============================================================================
# DUAL-LAYER ARCHITECTURE PARAMETERS
# =============================================================================

# Layer Capacities
LONG_TERM_MEMORY_CAPACITY = 100   # LML capacity
SHORT_TERM_MEMORY_CAPACITY = 50   # SML capacity

# Layer Transition Thresholds (with hysteresis to prevent oscillation)
THETA_PROMOTE = 0.7               # Promotion threshold to LML
THETA_DEMOTE = 0.5                # Demotion threshold to SML (θ_promote > θ_demote)

# Memory Pruning Parameters
EPSILON_PRUNE = 0.05              # Memory strength threshold for pruning
T_MAX_DAYS = 30                   # Maximum dormancy days before pruning

# =============================================================================
# BIOLOGICALLY-INSPIRED DECAY PARAMETERS
# =============================================================================

# Base Decay Parameters
LAMBDA_BASE = 0.1                 # λ_base: base decay rate (days^-1)
MU = 1.0                          # μ: importance modulation parameter
DELTA_RECENCY = 0.1               # δ: recency decay parameter

# Shape Parameters for Differential Decay
BETA_LML = 0.8                    # β_i for LML (sub-linear decay)
BETA_SML = 1.2                    # β_i for SML (super-linear decay)

# Memory Consolidation Parameters
DELTA_V = 0.2                     # Δv: base reinforcement strength
N_SPACING = 10                    # N: parameter for spacing effects
W_WINDOW_DAYS = 7                 # W: sliding window for access counting (days)
KAPPA = 0.1                       # κ: time decay parameter for access rate

# =============================================================================
# IMPORTANCE SCORING PARAMETERS
# =============================================================================

# Importance Score Weights: I_i(t) = α·rel(c_i, Q_t) + β·f_i/(1+f_i) + γ·recency(τ_i, t)
ALPHA = 0.4                       # α: weight for semantic relevance
BETA = 0.3                        # β: weight for frequency term
GAMMA = 0.3                       # γ: weight for recency

# =============================================================================
# CONFLICT RESOLUTION PARAMETERS
# =============================================================================

# Semantic Similarity Thresholds
THETA_SIM = 0.7                   # θ_sim: similarity threshold for conflict detection

# Conflict Resolution Parameters
OMEGA = 0.3                       # ω: redundancy penalty for compatible memories
RHO = 0.5                         # ρ: suppression strength for contradictory memories
W_AGE_DAYS = 30                   # W_age: age difference normalization window (days)

# =============================================================================
# ADAPTIVE FUSION PARAMETERS
# =============================================================================

# Fusion Clustering Parameters
THETA_FUSION = 0.6                # θ_fusion: similarity threshold for fusion candidates
T_WINDOW_DAYS = 7                 # T_window: temporal window for clustering (days)
CLUSTER_SIZE_THRESHOLD = 3        # Minimum cluster size for fusion
THETA_PRESERVE = 0.8              # θ_preserve: information preservation threshold

# Fusion Decay Modification
XI_FUSED_BASE = 1.0               # Base factor for fused memory decay rate

# =============================================================================
# SYSTEM PARAMETERS
# =============================================================================

# Time intervals for system operations (in seconds)
DECAY_UPDATE_INTERVAL = 3600      # 1 hour - decay updates
TRANSITION_CHECK_INTERVAL = 1800  # 30 minutes - layer transitions
FUSION_CHECK_INTERVAL = 7200      # 2 hours - memory fusion
CONFLICT_CHECK_INTERVAL = 1800    # 30 minutes - conflict resolution

# Embedding Dimensions (for mock embeddings)
EMBEDDING_DIMENSION = 768         # Standard transformer embedding size

# Batch Processing Limits
MAX_MEMORIES_PER_BATCH = 100      # Maximum memories to process in one batch
MAX_CLUSTERS_PER_FUSION = 10      # Maximum clusters to fuse in one operation

# =============================================================================
# PERFORMANCE THRESHOLDS
# =============================================================================

# Memory System Performance Limits
MAX_TOTAL_MEMORIES = 200          # Maximum total memories across both layers
MEMORY_CLEANUP_THRESHOLD = 0.9    # Cleanup when capacity reaches this ratio

# Processing Time Limits (seconds)
MAX_LLM_RESPONSE_TIME = 30        # Maximum time to wait for LLM response
MAX_FUSION_TIME = 60              # Maximum time for single fusion operation
MAX_CONFLICT_RESOLUTION_TIME = 45 # Maximum time for conflict resolution

# =============================================================================
# VALIDATION PARAMETERS
# =============================================================================

# Quality Assurance Thresholds
MIN_MEMORY_CONTENT_LENGTH = 10    # Minimum content length for valid memory
MAX_MEMORY_CONTENT_LENGTH = 2000  # Maximum content length for single memory
MIN_SIMILARITY_FOR_PROCESSING = 0.1  # Minimum similarity to consider processing

# Half-life Reference Values (for validation)
# At I_i(t)=0: t_1/2 ≈ 11.25 days for LML, t_1/2 ≈ 5.02 days for SML
EXPECTED_LML_HALF_LIFE_DAYS = 11.25
EXPECTED_SML_HALF_LIFE_DAYS = 5.02

# =============================================================================
# LEGACY COMPATIBILITY (deprecated, maintained for backward compatibility)
# =============================================================================

# Old parameters - use new dual-layer parameters instead
WORKING_MEMORY_CAPACITY = SHORT_TERM_MEMORY_CAPACITY  # Deprecated
WORKING_MEMORY_DECAY_RATE = LAMBDA_BASE               # Deprecated
SHORT_TERM_MEMORY_DECAY_RATE = LAMBDA_BASE            # Deprecated
LONG_TERM_MEMORY_DECAY_RATE = LAMBDA_BASE             # Deprecated
DECAY_THRESHOLD = EPSILON_PRUNE                       # Deprecated

# Old importance weights - use new ALPHA, BETA, GAMMA instead
IMPORTANCE_WEIGHTS = {                                 # Deprecated
    'semantic_relevance': ALPHA,
    'emotional_intensity': BETA, 
    'user_feedback': GAMMA
} 