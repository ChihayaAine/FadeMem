# Configuration settings for Agent Memory system

# Memory Layer Capacities
WORKING_MEMORY_CAPACITY = 7
SHORT_TERM_MEMORY_CAPACITY = 20
LONG_TERM_MEMORY_CAPACITY = 100

# Decay Rates for each layer (higher value = faster decay)
WORKING_MEMORY_DECAY_RATE = 0.05
SHORT_TERM_MEMORY_DECAY_RATE = 0.02
LONG_TERM_MEMORY_DECAY_RATE = 0.005

# Decay Threshold for transitions or archival
DECAY_THRESHOLD = 0.1

# Importance Scoring Weights
IMPORTANCE_WEIGHTS = {
    'semantic_relevance': 0.3,
    'emotional_intensity': 0.4,
    'user_feedback': 0.3
}

# Time intervals for decay updates (in seconds)
DECAY_UPDATE_INTERVAL = 300  # 5 minutes 