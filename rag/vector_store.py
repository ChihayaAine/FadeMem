"""
Memory Vector Store for Enhanced RAG System

This module implements a vector store that integrates with the dual-layer memory
architecture, providing efficient similarity search and memory-aware retrieval
that considers both semantic similarity and memory importance scores.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import json
import os
import time
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.memory_item import MemoryItem
import config

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    chromadb = None
    CHROMADB_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False


class MemoryVectorStore:
    """
    Enhanced vector store for dual-layer memory system.
    
    This vector store combines traditional semantic similarity search with
    memory-specific features like importance scoring, recency weighting,
    and layer-aware retrieval strategies.
    """
    
    def __init__(self, 
                 embedding_dimension: int = 1536,
                 collection_name: str = "memory_embeddings",
                 persist_directory: str = "./vector_db",
                 backend: str = "chroma"):
        """
        Initialize the memory vector store.
        
        Args:
            embedding_dimension (int): Dimension of embeddings (1536 for text-embedding-small-3)
            collection_name (str): Name of the vector collection
            persist_directory (str): Directory for persistent storage
            backend (str): Vector database backend ('chroma', 'faiss', or 'simple')
        """
        self.embedding_dimension = embedding_dimension
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.backend = backend
        
        # Memory tracking
        self.memory_metadata: Dict[str, Dict] = {}  # memory_id -> metadata
        self.memory_embeddings: Dict[str, List[float]] = {}  # memory_id -> embedding
        
        # Initialize backend
        self._initialize_backend()
        
        # Statistics
        self.total_memories = 0
        self.search_count = 0
        self.last_update = time.time()
        
    def _initialize_backend(self) -> None:
        """Initialize the chosen vector database backend."""
        if self.backend == "chroma" and CHROMADB_AVAILABLE:
            self._init_chroma()
        elif self.backend == "faiss" and FAISS_AVAILABLE:
            self._init_faiss()
        else:
            self._init_simple()
            if self.backend != "simple":
                print(f"Warning: {self.backend} backend not available. Using simple backend.")
                self.backend = "simple"
                
    def _init_chroma(self) -> None:
        """Initialize ChromaDB backend."""
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(
                name=self.collection_name,
                embedding_function=None  # We'll provide embeddings manually
            )
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                embedding_function=None,
                metadata={"description": "Dual-layer memory embeddings"}
            )
            
    def _init_faiss(self) -> None:
        """Initialize FAISS backend."""
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Create FAISS index
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)  # Inner product for cosine similarity
        self.faiss_id_map = {}  # memory_id -> faiss_id mapping
        self.faiss_reverse_map = {}  # faiss_id -> memory_id mapping
        self.faiss_counter = 0
        
        # Try to load existing index
        index_path = os.path.join(self.persist_directory, "faiss_index.bin")
        metadata_path = os.path.join(self.persist_directory, "faiss_metadata.json")
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            try:
                self.faiss_index = faiss.read_index(index_path)
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                    self.faiss_id_map = data.get('id_map', {})
                    self.faiss_reverse_map = data.get('reverse_map', {})
                    self.faiss_counter = data.get('counter', 0)
            except Exception as e:
                print(f"Warning: Could not load FAISS index: {e}")
                
    def _init_simple(self) -> None:
        """Initialize simple in-memory backend."""
        self.simple_embeddings: List[List[float]] = []
        self.simple_memory_ids: List[str] = []
        
        # Try to load from disk
        self._load_simple_backend()
        
    def add_memory(self, memory: MemoryItem, embedding: List[float]) -> str:
        """
        Add a memory and its embedding to the vector store.
        
        Args:
            memory (MemoryItem): Memory item to add
            embedding (List[float]): Embedding vector for the memory
            
        Returns:
            str: Unique memory ID in the vector store
        """
        memory_id = f"mem_{int(time.time() * 1000000)}_{id(memory)}"
        
        # Store memory metadata
        self.memory_metadata[memory_id] = {
            'content': memory.content,
            'memory_strength': memory.memory_strength,
            'access_frequency': memory.access_frequency,
            'creation_timestamp': memory.creation_timestamp,
            'layer_assignment': memory.layer_assignment,
            'lambda_i': memory.lambda_i,
            'beta_i': memory.beta_i,
            'time_decayed_access_rate': memory.time_decayed_access_rate,
            'metadata': memory.metadata,
            'added_timestamp': time.time()
        }
        
        self.memory_embeddings[memory_id] = embedding.copy()
        
        # Add to backend
        if self.backend == "chroma":
            self._add_to_chroma(memory_id, embedding)
        elif self.backend == "faiss":
            self._add_to_faiss(memory_id, embedding)
        else:
            self._add_to_simple(memory_id, embedding)
            
        self.total_memories += 1
        self.last_update = time.time()
        
        return memory_id
        
    def _add_to_chroma(self, memory_id: str, embedding: List[float]) -> None:
        """Add memory to ChromaDB."""
        metadata = self.memory_metadata[memory_id].copy()
        # Convert non-string values to strings for ChromaDB
        for key, value in metadata.items():
            if not isinstance(value, (str, int, float, bool)):
                metadata[key] = str(value)
                
        self.collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[metadata['content']]
        )
        
    def _add_to_faiss(self, memory_id: str, embedding: List[float]) -> None:
        """Add memory to FAISS index."""
        # Normalize embedding for cosine similarity
        embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(embedding_array)
        
        # Add to index
        faiss_id = self.faiss_counter
        self.faiss_index.add(embedding_array)
        
        # Update mappings
        self.faiss_id_map[memory_id] = faiss_id
        self.faiss_reverse_map[str(faiss_id)] = memory_id
        self.faiss_counter += 1
        
        # Save index periodically
        if self.faiss_counter % 100 == 0:
            self._save_faiss_backend()
            
    def _add_to_simple(self, memory_id: str, embedding: List[float]) -> None:
        """Add memory to simple backend."""
        self.simple_embeddings.append(embedding.copy())
        self.simple_memory_ids.append(memory_id)
        
        # Save periodically
        if len(self.simple_embeddings) % 50 == 0:
            self._save_simple_backend()
            
    def search_similar_memories(self, 
                               query_embedding: List[float],
                               top_k: int = 5,
                               similarity_threshold: float = 0.3,
                               layer_filter: Optional[str] = None,
                               include_importance: bool = True) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar memories using both semantic similarity and memory importance.
        
        Args:
            query_embedding (List[float]): Query embedding vector
            top_k (int): Number of top results to return
            similarity_threshold (float): Minimum similarity threshold
            layer_filter (str, optional): Filter by memory layer ('LML' or 'SML')
            include_importance (bool): Whether to boost scores with memory importance
            
        Returns:
            List[Tuple[str, float, Dict]]: List of (memory_id, combined_score, metadata)
        """
        self.search_count += 1
        
        if self.backend == "chroma":
            results = self._search_chroma(query_embedding, top_k * 2)  # Get more for filtering
        elif self.backend == "faiss":
            results = self._search_faiss(query_embedding, top_k * 2)
        else:
            results = self._search_simple(query_embedding, top_k * 2)
            
        # Apply memory-aware scoring and filtering
        enhanced_results = []
        current_time = time.time()
        
        for memory_id, similarity_score in results:
            if memory_id not in self.memory_metadata:
                continue
                
            metadata = self.memory_metadata[memory_id]
            
            # Apply layer filter
            if layer_filter and metadata.get('layer_assignment') != layer_filter:
                continue
                
            # Calculate combined score
            combined_score = similarity_score
            
            if include_importance:
                # Calculate memory importance (recreate the calculation)
                memory_strength = metadata.get('memory_strength', 0.5)
                access_rate = metadata.get('time_decayed_access_rate', 0.0)
                creation_time = metadata.get('creation_timestamp', current_time)
                
                # Frequency component with saturation
                frequency_score = access_rate / (1 + access_rate)
                
                # Recency component
                age_days = (current_time - creation_time) / 86400
                recency_score = np.exp(-config.DELTA_RECENCY * age_days)
                
                # Importance score (without semantic relevance since we already have similarity)
                importance = (config.BETA * frequency_score + config.GAMMA * recency_score)
                
                # Combine similarity with importance
                combined_score = (similarity_score * 0.7 + 
                                importance * 0.2 + 
                                memory_strength * 0.1)
                
            # Apply similarity threshold
            if combined_score >= similarity_threshold:
                enhanced_results.append((memory_id, combined_score, metadata))
                
        # Sort by combined score and return top_k
        enhanced_results.sort(key=lambda x: x[1], reverse=True)
        return enhanced_results[:top_k]
        
    def _search_chroma(self, query_embedding: List[float], top_k: int) -> List[Tuple[str, float]]:
        """Search ChromaDB for similar vectors."""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.total_memories)
            )
            
            memory_ids = results['ids'][0] if results['ids'] else []
            distances = results['distances'][0] if results['distances'] else []
            
            # Convert distances to similarities (ChromaDB returns L2 distances)
            similarities = [(1 / (1 + dist)) for dist in distances]
            
            return list(zip(memory_ids, similarities))
            
        except Exception as e:
            print(f"ChromaDB search error: {e}")
            return []
            
    def _search_faiss(self, query_embedding: List[float], top_k: int) -> List[Tuple[str, float]]:
        """Search FAISS index for similar vectors."""
        try:
            # Normalize query embedding
            query_array = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(query_array)
            
            # Search
            k = min(top_k, self.faiss_index.ntotal)
            similarities, indices = self.faiss_index.search(query_array, k)
            
            results = []
            for sim, idx in zip(similarities[0], indices[0]):
                if idx >= 0:  # Valid index
                    memory_id = self.faiss_reverse_map.get(str(idx))
                    if memory_id:
                        results.append((memory_id, float(sim)))
                        
            return results
            
        except Exception as e:
            print(f"FAISS search error: {e}")
            return []
            
    def _search_simple(self, query_embedding: List[float], top_k: int) -> List[Tuple[str, float]]:
        """Search simple backend for similar vectors."""
        if not self.simple_embeddings:
            return []
            
        query_array = np.array(query_embedding)
        similarities = []
        
        for i, embedding in enumerate(self.simple_embeddings):
            embedding_array = np.array(embedding)
            
            # Calculate cosine similarity
            dot_product = np.dot(query_array, embedding_array)
            norm_query = np.linalg.norm(query_array)
            norm_embedding = np.linalg.norm(embedding_array)
            
            if norm_query > 0 and norm_embedding > 0:
                similarity = dot_product / (norm_query * norm_embedding)
                similarities.append((self.simple_memory_ids[i], float(similarity)))
                
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
        
    def update_memory(self, memory_id: str, memory: MemoryItem, embedding: List[float]) -> bool:
        """
        Update an existing memory in the vector store.
        
        Args:
            memory_id (str): Memory ID to update
            memory (MemoryItem): Updated memory item
            embedding (List[float]): Updated embedding
            
        Returns:
            bool: True if update successful
        """
        if memory_id not in self.memory_metadata:
            return False
            
        # Update metadata
        self.memory_metadata[memory_id].update({
            'content': memory.content,
            'memory_strength': memory.memory_strength,
            'access_frequency': memory.access_frequency,
            'layer_assignment': memory.layer_assignment,
            'lambda_i': memory.lambda_i,
            'beta_i': memory.beta_i,
            'time_decayed_access_rate': memory.time_decayed_access_rate,
            'metadata': memory.metadata,
            'updated_timestamp': time.time()
        })
        
        self.memory_embeddings[memory_id] = embedding.copy()
        
        # Note: Most vector databases don't support efficient updates
        # For now, we'll just update our metadata. Full re-indexing would be needed
        # for production use with frequent updates.
        
        self.last_update = time.time()
        return True
        
    def remove_memory(self, memory_id: str) -> bool:
        """
        Remove a memory from the vector store.
        
        Args:
            memory_id (str): Memory ID to remove
            
        Returns:
            bool: True if removal successful
        """
        if memory_id not in self.memory_metadata:
            return False
            
        # Remove from metadata
        del self.memory_metadata[memory_id]
        del self.memory_embeddings[memory_id]
        
        # Note: Similar to updates, most vector databases don't support efficient deletion
        # This would require re-indexing for production use
        
        self.total_memories -= 1
        self.last_update = time.time()
        return True
        
    def get_memory_by_id(self, memory_id: str) -> Optional[Dict]:
        """Get memory metadata by ID."""
        return self.memory_metadata.get(memory_id)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        layer_counts = {'LML': 0, 'SML': 0, 'None': 0}
        avg_strength = 0
        total_accesses = 0
        
        for metadata in self.memory_metadata.values():
            layer = metadata.get('layer_assignment', 'None')
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
            avg_strength += metadata.get('memory_strength', 0)
            total_accesses += metadata.get('access_frequency', 0)
            
        return {
            'backend': self.backend,
            'total_memories': self.total_memories,
            'layer_distribution': layer_counts,
            'avg_memory_strength': avg_strength / max(1, self.total_memories),
            'total_accesses': total_accesses,
            'search_count': self.search_count,
            'last_update': self.last_update,
            'embedding_dimension': self.embedding_dimension
        }
        
    def _save_faiss_backend(self) -> None:
        """Save FAISS index and metadata to disk."""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Save index
            index_path = os.path.join(self.persist_directory, "faiss_index.bin")
            faiss.write_index(self.faiss_index, index_path)
            
            # Save metadata
            metadata_path = os.path.join(self.persist_directory, "faiss_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump({
                    'id_map': self.faiss_id_map,
                    'reverse_map': self.faiss_reverse_map,
                    'counter': self.faiss_counter
                }, f)
                
        except Exception as e:
            print(f"Warning: Could not save FAISS backend: {e}")
            
    def _save_simple_backend(self) -> None:
        """Save simple backend to disk."""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            
            data = {
                'embeddings': self.simple_embeddings,
                'memory_ids': self.simple_memory_ids,
                'metadata': self.memory_metadata
            }
            
            backend_path = os.path.join(self.persist_directory, "simple_backend.json")
            with open(backend_path, 'w') as f:
                json.dump(data, f)
                
        except Exception as e:
            print(f"Warning: Could not save simple backend: {e}")
            
    def _load_simple_backend(self) -> None:
        """Load simple backend from disk."""
        try:
            backend_path = os.path.join(self.persist_directory, "simple_backend.json")
            if os.path.exists(backend_path):
                with open(backend_path, 'r') as f:
                    data = json.load(f)
                    
                self.simple_embeddings = data.get('embeddings', [])
                self.simple_memory_ids = data.get('memory_ids', [])
                loaded_metadata = data.get('metadata', {})
                self.memory_metadata.update(loaded_metadata)
                self.total_memories = len(self.simple_memory_ids)
                
        except Exception as e:
            print(f"Warning: Could not load simple backend: {e}")
            
    def clear(self) -> None:
        """Clear all memories from the vector store."""
        self.memory_metadata.clear()
        self.memory_embeddings.clear()
        self.total_memories = 0
        self.search_count = 0
        
        if self.backend == "chroma":
            try:
                self.chroma_client.delete_collection(self.collection_name)
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    embedding_function=None
                )
            except Exception as e:
                print(f"Warning: Could not clear ChromaDB: {e}")
                
        elif self.backend == "faiss":
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)
            self.faiss_id_map.clear()
            self.faiss_reverse_map.clear()
            self.faiss_counter = 0
            
        else:
            self.simple_embeddings.clear()
            self.simple_memory_ids.clear()
            
    def __str__(self) -> str:
        stats = self.get_stats()
        return (f"MemoryVectorStore({self.backend}, "
                f"memories={stats['total_memories']}, "
                f"LML={stats['layer_distribution']['LML']}, "
                f"SML={stats['layer_distribution']['SML']})")
