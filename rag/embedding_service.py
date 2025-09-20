"""
Embedding Service using OpenAI's text-embedding-small-3 model

This module provides embedding generation services integrated with the dual-layer
memory architecture, specifically using OpenAI's text-embedding-small-3 model
as specified in the methodology requirements.
"""

from typing import List, Optional, Dict, Any
import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None


class OpenAIEmbeddingService:
    """
    OpenAI Embedding Service using text-embedding-small-3 model.
    
    This service provides high-quality embeddings for the dual-layer memory system,
    enabling semantic similarity calculations and vector-based retrieval.
    
    Model Specifications:
    - Model: text-embedding-small-3
    - Dimensions: 1536
    - Context length: 8191 tokens
    - Performance: Optimized for balance of quality and efficiency
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "text-embedding-small-3",
                 max_retries: int = 3,
                 timeout: float = 30.0):
        """
        Initialize the OpenAI embedding service.
        
        Args:
            api_key (str, optional): OpenAI API key
            model (str): Embedding model name (default: text-embedding-small-3)
            max_retries (int): Maximum retry attempts for API calls
            timeout (float): Request timeout in seconds
        """
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.dimension = 1536  # text-embedding-small-3 output dimension
        
        # Initialize OpenAI client
        if OpenAI and api_key:
            self.client = OpenAI(api_key=api_key, timeout=timeout)
            self.mock_mode = False
        else:
            self.client = None
            self.mock_mode = True
            if not OpenAI:
                print("Warning: 'openai' library not found. Using mock embeddings.")
            else:
                print("Warning: No API key provided. Using mock embeddings.")
                
        # Cache for embeddings to reduce API calls
        self.embedding_cache: Dict[str, List[float]] = {}
        self.cache_hits = 0
        self.api_calls = 0
        
    def embed_text(self, text: str, use_cache: bool = True) -> List[float]:
        """
        Generate embedding for a single text using text-embedding-small-3.
        
        Args:
            text (str): Text to embed
            use_cache (bool): Whether to use embedding cache
            
        Returns:
            List[float]: Embedding vector of dimension 1536
        """
        if not text or not text.strip():
            return [0.0] * self.dimension
            
        # Normalize text for caching
        text_key = text.strip().lower()
        
        # Check cache first
        if use_cache and text_key in self.embedding_cache:
            self.cache_hits += 1
            return self.embedding_cache[text_key].copy()
            
        if self.mock_mode:
            embedding = self._generate_mock_embedding(text)
        else:
            embedding = self._call_openai_embedding(text)
            
        # Cache the result
        if use_cache:
            self.embedding_cache[text_key] = embedding.copy()
            
        return embedding
        
    def embed_batch(self, texts: List[str], use_cache: bool = True) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts (List[str]): List of texts to embed
            use_cache (bool): Whether to use embedding cache
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        if not texts:
            return []
            
        # Separate cached and non-cached texts
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            if not text or not text.strip():
                embeddings.append([0.0] * self.dimension)
                continue
                
            text_key = text.strip().lower()
            if use_cache and text_key in self.embedding_cache:
                embeddings.append(self.embedding_cache[text_key].copy())
                self.cache_hits += 1
            else:
                embeddings.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
                
        # Process uncached texts
        if uncached_texts:
            if self.mock_mode:
                uncached_embeddings = [self._generate_mock_embedding(text) for text in uncached_texts]
            else:
                uncached_embeddings = self._call_openai_embedding_batch(uncached_texts)
                
            # Fill in the placeholders
            for idx, embedding in zip(uncached_indices, uncached_embeddings):
                embeddings[idx] = embedding
                
                # Cache the result
                if use_cache:
                    text_key = texts[idx].strip().lower()
                    self.embedding_cache[text_key] = embedding.copy()
                    
        return embeddings
        
    def _call_openai_embedding(self, text: str) -> List[float]:
        """Call OpenAI API for single text embedding with retry logic."""
        for attempt in range(self.max_retries):
            try:
                self.api_calls += 1
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text,
                    encoding_format="float"
                )
                return response.data[0].embedding
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"OpenAI API error after {self.max_retries} attempts: {e}")
                    return self._generate_mock_embedding(text)
                else:
                    print(f"OpenAI API attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
        return self._generate_mock_embedding(text)
        
    def _call_openai_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """Call OpenAI API for batch text embeddings with retry logic."""
        # OpenAI API supports batch requests up to 2048 inputs
        batch_size = min(len(texts), 100)  # Conservative batch size
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            for attempt in range(self.max_retries):
                try:
                    self.api_calls += 1
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=batch_texts,
                        encoding_format="float"
                    )
                    batch_embeddings = [data.embedding for data in response.data]
                    all_embeddings.extend(batch_embeddings)
                    break
                    
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        print(f"OpenAI API batch error after {self.max_retries} attempts: {e}")
                        # Fallback to mock embeddings for this batch
                        mock_embeddings = [self._generate_mock_embedding(text) for text in batch_texts]
                        all_embeddings.extend(mock_embeddings)
                    else:
                        print(f"OpenAI API batch attempt {attempt + 1} failed: {e}. Retrying...")
                        time.sleep(2 ** attempt)
                        
        return all_embeddings
        
    def _generate_mock_embedding(self, text: str) -> List[float]:
        """
        Generate deterministic mock embedding for testing/fallback.
        
        Uses a simple but consistent method based on text content.
        """
        # Create a deterministic seed from text content
        seed = abs(hash(text)) % (2**32)
        np.random.seed(seed)
        
        # Generate random embedding with unit norm
        embedding = np.random.normal(0, 1, self.dimension)
        
        # Normalize to unit length (same as OpenAI embeddings)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding.tolist()
        
    def cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1 (List[float]): First embedding vector
            embedding2 (List[float]): Second embedding vector
            
        Returns:
            float: Cosine similarity score [-1, 1]
        """
        if len(embedding1) != len(embedding2) or not embedding1 or not embedding2:
            return 0.0
            
        # Convert to numpy arrays for efficient computation
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(dot_product / (norm1 * norm2))
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get embedding service statistics.
        
        Returns:
            Dict: Service usage statistics
        """
        return {
            'model': self.model,
            'dimension': self.dimension,
            'mock_mode': self.mock_mode,
            'cache_size': len(self.embedding_cache),
            'cache_hits': self.cache_hits,
            'api_calls': self.api_calls,
            'cache_hit_rate': self.cache_hits / max(1, self.cache_hits + self.api_calls)
        }
        
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.embedding_cache.clear()
        self.cache_hits = 0
        self.api_calls = 0
        
    def set_api_key(self, api_key: str) -> None:
        """
        Update the API key and switch from mock mode to real API calls.
        
        Args:
            api_key (str): OpenAI API key
        """
        if OpenAI:
            self.client = OpenAI(api_key=api_key, timeout=self.timeout)
            self.mock_mode = False
            print(f"OpenAI client initialized with {self.model} model")
        else:
            print("Warning: 'openai' library not installed. Cannot use real API.")
            
    def __str__(self) -> str:
        stats = self.get_stats()
        return (f"OpenAIEmbeddingService({self.model}, "
                f"cache_size={stats['cache_size']}, "
                f"hit_rate={stats['cache_hit_rate']:.2%}, "
                f"mock_mode={self.mock_mode})")


class EmbeddingCache:
    """
    Persistent embedding cache for reducing API costs and improving performance.
    """
    
    def __init__(self, cache_file: str = "embedding_cache.json"):
        """
        Initialize embedding cache.
        
        Args:
            cache_file (str): Path to cache file
        """
        self.cache_file = cache_file
        self.cache: Dict[str, List[float]] = {}
        self.load_cache()
        
    def load_cache(self) -> None:
        """Load cache from file."""
        try:
            import json
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                self.cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.cache = {}
            
    def save_cache(self) -> None:
        """Save cache to file."""
        try:
            import json
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Could not save embedding cache: {e}")
            
    def get(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache."""
        return self.cache.get(text.strip().lower())
        
    def put(self, text: str, embedding: List[float]) -> None:
        """Store embedding in cache."""
        self.cache[text.strip().lower()] = embedding
        
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        
    def size(self) -> int:
        """Get cache size."""
        return len(self.cache)
