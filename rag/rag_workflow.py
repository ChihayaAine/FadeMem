
"""
LangGraph-based RAG Workflow for Enhanced Memory System

This module implements a sophisticated RAG workflow using LangGraph that integrates
with the dual-layer memory architecture to provide context-aware response generation
with memory-enhanced retrieval and reasoning capabilities.
"""

from typing import List, Dict, Any, Optional, TypedDict, Annotated
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.memory_item import MemoryItem
from core.enhanced_memory_manager import EnhancedMemoryManager
from llm.llm_interface import LLMInterface
from .embedding_service import OpenAIEmbeddingService
from .vector_store import MemoryVectorStore
import config

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None
    BaseMessage = None
    HumanMessage = None
    AIMessage = None
    SystemMessage = None

import json
import time
from dataclasses import dataclass


class WorkflowState(TypedDict):
    """State representation for the RAG workflow."""
    query: str
    query_embedding: List[float]
    retrieved_memories: List[Dict[str, Any]]
    memory_context: str
    reasoning_steps: List[str]
    response: str
    confidence_score: float
    metadata: Dict[str, Any]


@dataclass
class RetrievalResult:
    """Result from memory retrieval with enhanced scoring."""
    memory_id: str
    content: str
    similarity_score: float
    importance_score: float
    combined_score: float
    layer: str
    metadata: Dict[str, Any]


class RAGWorkflow:
    """
    LangGraph-based RAG workflow for memory-enhanced question answering.
    
    This workflow implements a sophisticated pipeline that:
    1. Processes user queries with embedding generation
    2. Retrieves relevant memories using hybrid similarity + importance scoring
    3. Analyzes memory conflicts and consistency
    4. Generates reasoning chains with memory context
    5. Produces final responses with confidence scoring
    """
    
    def __init__(self,
                 memory_manager: EnhancedMemoryManager,
                 embedding_service: OpenAIEmbeddingService,
                 vector_store: MemoryVectorStore,
                 llm_interface: LLMInterface,
                 max_retrieved_memories: int = 5,
                 confidence_threshold: float = 0.6):
        """
        Initialize the RAG workflow.
        
        Args:
            memory_manager (EnhancedMemoryManager): Memory system manager
            embedding_service (OpenAIEmbeddingService): Embedding generation service
            vector_store (MemoryVectorStore): Vector database for similarity search
            llm_interface (LLMInterface): LLM for response generation
            max_retrieved_memories (int): Maximum memories to retrieve
            confidence_threshold (float): Minimum confidence for responses
        """
        self.memory_manager = memory_manager
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_interface = llm_interface
        self.max_retrieved_memories = max_retrieved_memories
        self.confidence_threshold = confidence_threshold
        
        # Initialize workflow graph
        if LANGGRAPH_AVAILABLE:
            self.workflow = self._build_workflow()
            self.checkpointer = MemorySaver()
        else:
            print("Warning: LangGraph not available. Using simplified workflow.")
            self.workflow = None
            
        # Statistics
        self.query_count = 0
        self.avg_response_time = 0
        self.confidence_scores = []
        
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for RAG processing."""
        workflow = StateGraph(WorkflowState)
        
        # Add nodes for each step in the RAG pipeline
        workflow.add_node("embed_query", self._embed_query_node)
        workflow.add_node("retrieve_memories", self._retrieve_memories_node)
        workflow.add_node("analyze_consistency", self._analyze_consistency_node)
        workflow.add_node("build_context", self._build_context_node)
        workflow.add_node("generate_reasoning", self._generate_reasoning_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("evaluate_confidence", self._evaluate_confidence_node)
        
        # Define the workflow edges
        workflow.set_entry_point("embed_query")
        workflow.add_edge("embed_query", "retrieve_memories")
        workflow.add_edge("retrieve_memories", "analyze_consistency")
        workflow.add_edge("analyze_consistency", "build_context")
        workflow.add_edge("build_context", "generate_reasoning")
        workflow.add_edge("generate_reasoning", "generate_response")
        workflow.add_edge("generate_response", "evaluate_confidence")
        workflow.add_edge("evaluate_confidence", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    def process_query(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Process a user query through the complete RAG workflow.
        
        Args:
            query (str): User query
            session_id (str): Session identifier for conversation tracking
            
        Returns:
            Dict[str, Any]: Complete workflow result with response and metadata
        """
        start_time = time.time()
        self.query_count += 1
        
        if self.workflow and LANGGRAPH_AVAILABLE:
            # Use LangGraph workflow
            initial_state = WorkflowState(
                query=query,
                query_embedding=[],
                retrieved_memories=[],
                memory_context="",
                reasoning_steps=[],
                response="",
                confidence_score=0.0,
                metadata={}
            )
            
            try:
                # Run the workflow
                config_dict = {"configurable": {"thread_id": session_id}}
                result = self.workflow.invoke(initial_state, config=config_dict)
                
                # Update statistics
                response_time = time.time() - start_time
                self._update_statistics(response_time, result.get('confidence_score', 0))
                
                return result
                
            except Exception as e:
                print(f"LangGraph workflow error: {e}")
                # Fallback to simple processing
                return self._simple_process_query(query, start_time)
        else:
            # Use simplified workflow
            return self._simple_process_query(query, start_time)
    
    def _simple_process_query(self, query: str, start_time: float) -> Dict[str, Any]:
        """Simplified query processing without LangGraph."""
        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)
        
        # Retrieve memories
        retrieved_memories = self.vector_store.search_similar_memories(
            query_embedding=query_embedding,
            top_k=self.max_retrieved_memories,
            include_importance=True
        )
        
        # Build context
        context_parts = []
        for memory_id, score, metadata in retrieved_memories:
            layer = metadata.get('layer_assignment', 'Unknown')
            strength = metadata.get('memory_strength', 0)
            content = metadata.get('content', '')
            
            context_parts.append(
                f"[{layer}, Strength: {strength:.3f}, Score: {score:.3f}] {content}"
            )
        
        memory_context = "\n".join(context_parts) if context_parts else "No relevant memories found."
        
        # Generate response
        response = self.llm_interface.generate_response(query, memory_context)
        
        # Calculate simple confidence based on retrieval quality
        if retrieved_memories:
            avg_score = sum(score for _, score, _ in retrieved_memories) / len(retrieved_memories)
            confidence = min(1.0, avg_score * 1.2)  # Boost and cap at 1.0
        else:
            confidence = 0.3  # Low confidence without memory context
        
        # Update statistics
        response_time = time.time() - start_time
        self._update_statistics(response_time, confidence)
        
        return {
            'query': query,
            'response': response,
            'confidence_score': confidence,
            'retrieved_memories': [
                {
                    'memory_id': mid,
                    'score': score,
                    'content': meta.get('content', ''),
                    'layer': meta.get('layer_assignment', 'Unknown')
                }
                for mid, score, meta in retrieved_memories
            ],
            'memory_context': memory_context,
            'response_time': response_time,
            'reasoning_steps': ['Query processed', 'Memories retrieved', 'Context built', 'Response generated']
        }
    
    def _embed_query_node(self, state: WorkflowState) -> WorkflowState:
        """Generate embedding for the input query."""
        query_embedding = self.embedding_service.embed_text(state['query'])
        
        state['query_embedding'] = query_embedding
        state['metadata']['embedding_time'] = time.time()
        
        return state
    
    def _retrieve_memories_node(self, state: WorkflowState) -> WorkflowState:
        """Retrieve relevant memories using hybrid similarity + importance scoring."""
        retrieved_memories = self.vector_store.search_similar_memories(
            query_embedding=state['query_embedding'],
            top_k=self.max_retrieved_memories * 2,  # Get more for further processing
            include_importance=True
        )
        
        # Convert to structured format
        structured_memories = []
        for memory_id, combined_score, metadata in retrieved_memories:
            # Calculate individual scores
            query_emb = state['query_embedding']
            memory_emb = self.vector_store.memory_embeddings.get(memory_id, [])
            
            if memory_emb:
                similarity_score = self.embedding_service.cosine_similarity(query_emb, memory_emb)
            else:
                similarity_score = 0.0
            
            # Extract importance components
            strength = metadata.get('memory_strength', 0)
            access_rate = metadata.get('time_decayed_access_rate', 0)
            frequency_score = access_rate / (1 + access_rate)
            
            current_time = time.time()
            creation_time = metadata.get('creation_timestamp', current_time)
            age_days = (current_time - creation_time) / 86400
            recency_score = np.exp(-config.DELTA_RECENCY * age_days) if 'np' in globals() else 0.5
            
            importance_score = (config.BETA * frequency_score + 
                              config.GAMMA * recency_score + 
                              strength * 0.1)
            
            structured_memories.append({
                'memory_id': memory_id,
                'content': metadata.get('content', ''),
                'similarity_score': similarity_score,
                'importance_score': importance_score,
                'combined_score': combined_score,
                'layer': metadata.get('layer_assignment', 'Unknown'),
                'memory_strength': strength,
                'access_frequency': metadata.get('access_frequency', 0),
                'age_days': age_days,
                'metadata': metadata
            })
        
        state['retrieved_memories'] = structured_memories
        state['metadata']['retrieval_count'] = len(structured_memories)
        
        return state
    
    def _analyze_consistency_node(self, state: WorkflowState) -> WorkflowState:
        """Analyze consistency and conflicts among retrieved memories."""
        memories = state['retrieved_memories']
        
        if len(memories) < 2:
            state['reasoning_steps'].append("Single memory retrieved - no consistency analysis needed")
            return state
        
        # Simple consistency analysis - in practice, this could use LLM-based analysis
        consistency_prompt = f"""
        Analyze the consistency of these memory contents:
        
        {chr(10).join([f"{i+1}. {mem['content']}" for i, mem in enumerate(memories[:5])])}
        
        Are there any contradictions or inconsistencies? Respond with:
        - CONSISTENT: All memories are consistent
        - CONFLICTING: There are contradictions (explain briefly)
        - PARTIAL: Some memories conflict (explain which ones)
        """
        
        try:
            consistency_analysis = self.llm_interface.generate_response(consistency_prompt)
            state['reasoning_steps'].append(f"Consistency analysis: {consistency_analysis[:100]}...")
            state['metadata']['consistency_analysis'] = consistency_analysis
        except Exception as e:
            state['reasoning_steps'].append(f"Consistency analysis failed: {str(e)}")
            state['metadata']['consistency_analysis'] = "Analysis unavailable"
        
        return state
    
    def _build_context_node(self, state: WorkflowState) -> WorkflowState:
        """Build comprehensive context from retrieved memories."""
        memories = state['retrieved_memories']
        
        if not memories:
            state['memory_context'] = "No relevant memories found."
            return state
        
        # Group memories by layer for structured presentation
        lml_memories = [m for m in memories if m['layer'] == 'LML']
        sml_memories = [m for m in memories if m['layer'] == 'SML']
        other_memories = [m for m in memories if m['layer'] not in ['LML', 'SML']]
        
        context_parts = []
        
        if lml_memories:
            context_parts.append("=== Long-term Memories (High Importance) ===")
            for mem in lml_memories[:3]:  # Top 3 from LML
                context_parts.append(
                    f"• [{mem['combined_score']:.3f}] {mem['content']} "
                    f"(Strength: {mem['memory_strength']:.3f}, Age: {mem['age_days']:.1f}d)"
                )
        
        if sml_memories:
            context_parts.append("\n=== Short-term Memories (Recent/Contextual) ===")
            for mem in sml_memories[:3]:  # Top 3 from SML
                context_parts.append(
                    f"• [{mem['combined_score']:.3f}] {mem['content']} "
                    f"(Strength: {mem['memory_strength']:.3f}, Age: {mem['age_days']:.1f}d)"
                )
        
        if other_memories:
            context_parts.append("\n=== Other Relevant Information ===")
            for mem in other_memories[:2]:  # Top 2 from others
                context_parts.append(f"• [{mem['combined_score']:.3f}] {mem['content']}")
        
        state['memory_context'] = "\n".join(context_parts)
        state['reasoning_steps'].append(f"Built context from {len(memories)} memories across layers")
        
        return state
    
    def _generate_reasoning_node(self, state: WorkflowState) -> WorkflowState:
        """Generate explicit reasoning steps for the response."""
        reasoning_prompt = f"""
        Given this query: "{state['query']}"
        And this memory context: {state['memory_context']}
        
        Provide a step-by-step reasoning process for how you would answer this query:
        1. What key information is available in the memories?
        2. How do the memories relate to the query?
        3. What conclusions can be drawn?
        4. Are there any gaps or uncertainties?
        
        Keep reasoning concise but clear.
        """
        
        try:
            reasoning = self.llm_interface.generate_response(reasoning_prompt)
            # Extract steps from reasoning
            reasoning_lines = [line.strip() for line in reasoning.split('\n') if line.strip()]
            state['reasoning_steps'].extend(reasoning_lines[:4])  # Take first 4 steps
            state['metadata']['explicit_reasoning'] = reasoning
        except Exception as e:
            state['reasoning_steps'].append(f"Reasoning generation failed: {str(e)}")
        
        return state
    
    def _generate_response_node(self, state: WorkflowState) -> WorkflowState:
        """Generate the final response using memory context."""
        enhanced_prompt = f"""
        Query: {state['query']}
        
        Available Memory Context:
        {state['memory_context']}
        
        Reasoning Process:
        {chr(10).join(state['reasoning_steps'])}
        
        Based on the above context and reasoning, provide a comprehensive and accurate response to the query.
        If the memory context doesn't fully address the query, acknowledge what's missing.
        """
        
        try:
            response = self.llm_interface.generate_response(enhanced_prompt)
            state['response'] = response
        except Exception as e:
            state['response'] = f"I apologize, but I encountered an error generating a response: {str(e)}"
        
        return state
    
    def _evaluate_confidence_node(self, state: WorkflowState) -> WorkflowState:
        """Evaluate confidence in the generated response."""
        memories = state['retrieved_memories']
        
        if not memories:
            state['confidence_score'] = 0.2
            return state
        
        # Calculate confidence based on multiple factors
        factors = {
            'memory_relevance': 0.0,
            'memory_strength': 0.0,
            'coverage': 0.0,
            'consistency': 0.5  # Default neutral
        }
        
        if memories:
            # Memory relevance (average combined score)
            factors['memory_relevance'] = sum(m['combined_score'] for m in memories) / len(memories)
            
            # Memory strength (average strength of retrieved memories)
            factors['memory_strength'] = sum(m['memory_strength'] for m in memories) / len(memories)
            
            # Coverage (how well memories address the query - simplified)
            factors['coverage'] = min(1.0, len(memories) / 3)  # 3+ memories = good coverage
            
            # Consistency (based on analysis if available)
            consistency_analysis = state['metadata'].get('consistency_analysis', '')
            if 'CONSISTENT' in consistency_analysis.upper():
                factors['consistency'] = 0.9
            elif 'CONFLICTING' in consistency_analysis.upper():
                factors['consistency'] = 0.3
            elif 'PARTIAL' in consistency_analysis.upper():
                factors['consistency'] = 0.6
        
        # Weighted combination of factors
        weights = {
            'memory_relevance': 0.4,
            'memory_strength': 0.2,
            'coverage': 0.2,
            'consistency': 0.2
        }
        
        confidence = sum(factors[k] * weights[k] for k in factors)
        state['confidence_score'] = max(0.0, min(1.0, confidence))
        
        # Add confidence explanation
        state['metadata']['confidence_factors'] = factors
        state['metadata']['confidence_weights'] = weights
        
        return state
    
    def _update_statistics(self, response_time: float, confidence: float) -> None:
        """Update workflow statistics."""
        # Update average response time
        self.avg_response_time = (
            (self.avg_response_time * (self.query_count - 1) + response_time) / self.query_count
        )
        
        # Track confidence scores
        self.confidence_scores.append(confidence)
        if len(self.confidence_scores) > 100:  # Keep last 100 scores
            self.confidence_scores = self.confidence_scores[-100:]
    
    def get_workflow_stats(self) -> Dict[str, Any]:
        """Get workflow performance statistics."""
        if self.confidence_scores:
            avg_confidence = sum(self.confidence_scores) / len(self.confidence_scores)
            high_confidence_rate = sum(1 for c in self.confidence_scores if c >= self.confidence_threshold) / len(self.confidence_scores)
        else:
            avg_confidence = 0
            high_confidence_rate = 0
        
        return {
            'query_count': self.query_count,
            'avg_response_time': self.avg_response_time,
            'avg_confidence': avg_confidence,
            'high_confidence_rate': high_confidence_rate,
            'confidence_threshold': self.confidence_threshold,
            'max_retrieved_memories': self.max_retrieved_memories,
            'langgraph_available': LANGGRAPH_AVAILABLE,
            'embedding_stats': self.embedding_service.get_stats(),
            'vector_store_stats': self.vector_store.get_stats()
        }
    
    def __str__(self) -> str:
        stats = self.get_workflow_stats()
        return (f"RAGWorkflow(queries={stats['query_count']}, "
                f"avg_time={stats['avg_response_time']:.3f}s, "
                f"avg_confidence={stats['avg_confidence']:.3f}, "
                f"langgraph={LANGGRAPH_AVAILABLE})")
