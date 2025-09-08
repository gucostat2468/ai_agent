"""
Memory System for AI Agent
Implements multi-layered memory architecture with short-term, long-term,
episodic, and vector-based memory components.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import asyncio
import json
import uuid
import logging

from ..interfaces.base import MemoryInterface, MemoryEntry, VectorStoreInterface
from ..config.settings import MemoryConfig
from ..monitoring.logger import StructuredLogger
from ..utils.exceptions import MemoryException, MemoryStorageException, MemoryRetrievalException
from ..memory.short_term import ShortTermMemory
from ..memory.long_term import LongTermMemory
from ..memory.vector_store import VectorStore
from ..memory.episodic import EpisodicMemory


class MemoryType(Enum):
    """Types of memory entries"""
    CONVERSATION = "conversation"
    TASK = "task"
    LEARNING = "learning"
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    EPISODIC = "episodic"
    CONTEXT = "context"


class ImportanceLevel(Enum):
    """Importance levels for memory entries"""
    LOW = 0.2
    MEDIUM = 0.5
    HIGH = 0.8
    CRITICAL = 1.0


@dataclass
class MemoryQuery:
    """Memory query parameters"""
    text: Optional[str] = None
    memory_type: Optional[MemoryType] = None
    tags: List[str] = field(default_factory=list)
    time_range: Optional[Tuple[datetime, datetime]] = None
    importance_threshold: float = 0.0
    limit: int = 10
    similarity_threshold: float = 0.7
    include_metadata: bool = True


@dataclass
class MemorySearchResult:
    """Memory search result"""
    entry: MemoryEntry
    score: float
    source: str  # which memory component found this
    reasoning: str  # why this was selected


class MemorySystem(MemoryInterface):
    """
    Unified memory system that coordinates multiple memory components.
    """
    
    def __init__(self, config: MemoryConfig):
        super().__init__(config.dict() if hasattr(config, 'dict') else config.__dict__)
        self.config = config
        self.logger = StructuredLogger(__name__)
        
        # Memory components
        self.short_term: Optional[ShortTermMemory] = None
        self.long_term: Optional[LongTermMemory] = None
        self.vector_store: Optional[VectorStore] = None
        self.episodic: Optional[EpisodicMemory] = None
        
        # Memory statistics
        self.stats = {
            "total_entries": 0,
            "queries_processed": 0,
            "cache_hits": 0,
            "last_cleanup": None
        }
        
        # Internal state
        self._cleanup_task: Optional[asyncio.Task] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize all memory components"""
        try:
            self.logger.info("Initializing Memory System")
            
            # Initialize short-term memory
            self.short_term = ShortTermMemory(
                capacity=self.config.short_term_capacity,
                ttl=self.config.short_term_ttl
            )
            await self.short_term.initialize()
            
            # Initialize long-term memory
            self.long_term = LongTermMemory(
                storage_type=self.config.long_term_storage,
                retention_days=self.config.long_term_retention_days
            )
            await self.long_term.initialize()
            
            # Initialize vector store if enabled
            if self.config.enable_vector_memory:
                self.vector_store = VectorStore(
                    dimension=self.config.vector_dimension,
                    similarity_threshold=self.config.similarity_threshold
                )
                await self.vector_store.initialize()
            
            # Initialize episodic memory if enabled
            if self.config.enable_episodic_memory:
                self.episodic = EpisodicMemory(
                    max_episode_length=self.config.episode_max_length,
                    episode_overlap=self.config.episode_overlap
                )
                await self.episodic.initialize()
            
            # Start background cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self._initialized = True
            self.logger.info("Memory System initialized successfully")
            
        except Exception as e:
            self.logger.error("Failed to initialize Memory System", error=str(e))
            raise MemoryException(f"Memory system initialization failed: {e}") from e
    
    async def cleanup(self) -> None:
        """Cleanup memory system resources"""
        try:
            self.logger.info("Cleaning up Memory System")
            
            # Cancel cleanup task
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Cleanup components
            components = [self.short_term, self.long_term, self.vector_store, self.episodic]
            for component in components:
                if component:
                    try:
                        await component.cleanup()
                    except Exception as e:
                        self.logger.warning(f"Error cleaning up component {component.__class__.__name__}", error=str(e))
            
            self._initialized = False
            self.logger.info("Memory System cleaned up successfully")
            
        except Exception as e:
            self.logger.error("Error during Memory System cleanup", error=str(e))
    
    async def store_memory(self, entry: MemoryEntry) -> bool:
        """Store a memory entry in appropriate memory components"""
        if not self._initialized:
            raise MemoryException("Memory system not initialized")
        
        try:
            self.logger.debug("Storing memory entry", 
                            memory_id=entry.id,
                            content_type=entry.content_type,
                            importance=entry.importance)
            
            success = True
            
            # Store in short-term memory
            if self.short_term:
                await self.short_term.store_memory(entry)
            
            # Store in long-term memory if important enough
            if entry.importance >= self.config.importance_threshold and self.long_term:
                success &= await self.long_term.store_memory(entry)
            
            # Store in vector store for similarity search
            if self.vector_store and entry.content_type == "text":
                await self.vector_store.store_text_memory(entry)
            
            # Add to episodic memory if it's part of a sequence
            if self.episodic and entry.content_type in ["conversation", "task"]:
                await self.episodic.add_to_episode(entry)
            
            # Update statistics
            self.stats["total_entries"] += 1
            
            self.logger.debug("Memory entry stored successfully", memory_id=entry.id)
            return success
            
        except Exception as e:
            self.logger.error("Failed to store memory entry", 
                            memory_id=entry.id, error=str(e))
            raise MemoryStorageException(f"Failed to store memory: {e}", entry.id) from e
    
    async def retrieve_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific memory entry by ID"""
        if not self._initialized:
            raise MemoryException("Memory system not initialized")
        
        try:
            self.logger.debug("Retrieving memory entry", memory_id=memory_id)
            
            # Try short-term memory first (fastest)
            if self.short_term:
                entry = await self.short_term.retrieve_memory(memory_id)
                if entry:
                    self.stats["cache_hits"] += 1
                    return entry
            
            # Try long-term memory
            if self.long_term:
                entry = await self.long_term.retrieve_memory(memory_id)
                if entry:
                    # Cache in short-term for future access
                    if self.short_term:
                        await self.short_term.store_memory(entry)
                    return entry
            
            # Try vector store
            if self.vector_store:
                entry = await self.vector_store.retrieve_memory(memory_id)
                if entry:
                    return entry
            
            self.logger.debug("Memory entry not found", memory_id=memory_id)
            return None
            
        except Exception as e:
            self.logger.error("Failed to retrieve memory entry", 
                            memory_id=memory_id, error=str(e))
            raise MemoryRetrievalException(f"Failed to retrieve memory: {e}", memory_id) from e
    
    async def search_memories(
        self, 
        query: str, 
        memory_type: str = None, 
        limit: int = 10
    ) -> List[MemoryEntry]:
        """Search memories with simple interface"""
        memory_query = MemoryQuery(
            text=query,
            memory_type=MemoryType(memory_type) if memory_type else None,
            limit=limit
        )
        
        results = await self.advanced_search(memory_query)
        return [result.entry for result in results]
    
    async def advanced_search(self, query: MemoryQuery) -> List[MemorySearchResult]:
        """Advanced memory search with detailed results"""
        if not self._initialized:
            raise MemoryException("Memory system not initialized")
        
        try:
            self.logger.debug("Performing advanced memory search", 
                            query_text=query.text[:50] if query.text else None,
                            memory_type=query.memory_type.value if query.memory_type else None,
                            limit=query.limit)
            
            all_results = []
            
            # Search short-term memory
            if self.short_term:
                short_results = await self.short_term.search_memories(
                    query.text or "", query.memory_type.value if query.memory_type else None, query.limit
                )
                for entry in short_results:
                    all_results.append(MemorySearchResult(
                        entry=entry,
                        score=1.0,  # Short-term has highest relevance
                        source="short_term",
                        reasoning="Recent memory from short-term cache"
                    ))
            
            # Search vector store for semantic similarity
            if self.vector_store and query.text:
                vector_results = await self.vector_store.semantic_search(
                    query.text, query.limit, query.similarity_threshold
                )
                for entry, score in vector_results:
                    all_results.append(MemorySearchResult(
                        entry=entry,
                        score=score,
                        source="vector_store",
                        reasoning=f"Semantic similarity: {score:.3f}"
                    ))
            
            # Search long-term memory
            if self.long_term:
                long_results = await self.long_term.search_memories(
                    query.text or "", query.memory_type.value if query.memory_type else None, query.limit
                )
                for entry in long_results:
                    all_results.append(MemorySearchResult(
                        entry=entry,
                        score=entry.importance,
                        source="long_term",
                        reasoning=f"Long-term memory, importance: {entry.importance}"
                    ))
            
            # Search episodic memory
            if self.episodic and query.text:
                episodic_results = await self.episodic.search_episodes(query.text, query.limit)
                for entry in episodic_results:
                    all_results.append(MemorySearchResult(
                        entry=entry,
                        score=0.8,  # High relevance for episodic
                        source="episodic",
                        reasoning="Part of episodic sequence"
                    ))
            
            # Filter and rank results
            filtered_results = self._filter_and_rank_results(all_results, query)
            
            self.stats["queries_processed"] += 1
            
            self.logger.debug("Memory search completed", 
                            results_found=len(filtered_results))
            
            return filtered_results[:query.limit]
            
        except Exception as e:
            self.logger.error("Advanced memory search failed", error=str(e))
            raise MemoryRetrievalException(f"Memory search failed: {e}", query=query.text) from e
    
    async def get_recent_memories(self, limit: int = 10, memory_type: str = None) -> List[MemoryEntry]:
        """Get recent memories"""
        if not self._initialized:
            raise MemoryException("Memory system not initialized")
        
        try:
            # Get from short-term memory first
            recent_memories = []
            
            if self.short_term:
                short_memories = await self.short_term.get_recent_memories(limit, memory_type)
                recent_memories.extend(short_memories)
            
            # If not enough, get from long-term
            if len(recent_memories) < limit and self.long_term:
                remaining = limit - len(recent_memories)
                long_memories = await self.long_term.get_recent_memories(remaining, memory_type)
                recent_memories.extend(long_memories)
            
            # Sort by timestamp
            recent_memories.sort(key=lambda x: x.timestamp, reverse=True)
            
            return recent_memories[:limit]
            
        except Exception as e:
            self.logger.error("Failed to get recent memories", error=str(e))
            raise MemoryRetrievalException(f"Failed to get recent memories: {e}") from e
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory entry from all storage"""
        if not self._initialized:
            raise MemoryException("Memory system not initialized")
        
        try:
            self.logger.debug("Deleting memory entry", memory_id=memory_id)
            
            success = True
            
            # Delete from all components
            components = [self.short_term, self.long_term, self.vector_store, self.episodic]
            for component in components:
                if component:
                    try:
                        result = await component.delete_memory(memory_id)
                        success = success and result
                    except Exception as e:
                        self.logger.warning(f"Failed to delete from {component.__class__.__name__}", 
                                          error=str(e))
                        success = False
            
            if success:
                self.stats["total_entries"] = max(0, self.stats["total_entries"] - 1)
                self.logger.debug("Memory entry deleted successfully", memory_id=memory_id)
            
            return success
            
        except Exception as e:
            self.logger.error("Failed to delete memory entry", 
                            memory_id=memory_id, error=str(e))
            return False
    
    async def cleanup_expired(self) -> int:
        """Clean up expired memories"""
        if not self._initialized:
            return 0
        
        try:
            self.logger.info("Starting memory cleanup")
            
            total_cleaned = 0
            
            # Cleanup each component
            if self.short_term:
                cleaned = await self.short_term.cleanup_expired()
                total_cleaned += cleaned
                self.logger.debug(f"Short-term memory cleaned: {cleaned} entries")
            
            if self.long_term:
                cleaned = await self.long_term.cleanup_expired()
                total_cleaned += cleaned
                self.logger.debug(f"Long-term memory cleaned: {cleaned} entries")
            
            if self.vector_store:
                cleaned = await self.vector_store.cleanup_expired()
                total_cleaned += cleaned
                self.logger.debug(f"Vector store cleaned: {cleaned} entries")
            
            if self.episodic:
                cleaned = await self.episodic.cleanup_expired()
                total_cleaned += cleaned
                self.logger.debug(f"Episodic memory cleaned: {cleaned} entries")
            
            # Update statistics
            self.stats["total_entries"] = max(0, self.stats["total_entries"] - total_cleaned)
            self.stats["last_cleanup"] = datetime.now(timezone.utc)
            
            self.logger.info(f"Memory cleanup completed, removed {total_cleaned} entries")
            
            return total_cleaned
            
        except Exception as e:
            self.logger.error("Memory cleanup failed", error=str(e))
            return 0
    
    async def retrieve_relevant(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve memories relevant to a query (simplified interface for agent)"""
        try:
            memory_query = MemoryQuery(
                text=query,
                limit=limit,
                similarity_threshold=self.config.similarity_threshold
            )
            
            results = await self.advanced_search(memory_query)
            
            # Convert to simplified format
            relevant_memories = []
            for result in results:
                memory_data = {
                    "id": result.entry.id,
                    "content": result.entry.content,
                    "type": result.entry.content_type,
                    "importance": result.entry.importance,
                    "timestamp": result.entry.timestamp.isoformat(),
                    "score": result.score,
                    "source": result.source,
                    "reasoning": result.reasoning
                }
                relevant_memories.append(memory_data)
            
            return relevant_memories
            
        except Exception as e:
            self.logger.error("Failed to retrieve relevant memories", error=str(e))
            return []
    
    async def store_task_execution(self, task_data: Dict[str, Any]) -> bool:
        """Store task execution data in memory"""
        try:
            entry = MemoryEntry(
                content=task_data,
                content_type=MemoryType.TASK.value,
                tags=["task_execution", task_data.get("task", {}).get("type", "unknown")],
                importance=ImportanceLevel.MEDIUM.value
            )
            
            return await self.store_memory(entry)
            
        except Exception as e:
            self.logger.error("Failed to store task execution", error=str(e))
            return False
    
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        stats = self.stats.copy()
        
        # Add component-specific statistics
        if self.short_term:
            stats["short_term"] = await self.short_term.get_statistics()
        
        if self.long_term:
            stats["long_term"] = await self.long_term.get_statistics()
        
        if self.vector_store:
            stats["vector_store"] = await self.vector_store.get_statistics()
        
        if self.episodic:
            stats["episodic"] = await self.episodic.get_statistics()
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on memory system"""
        health = {
            "status": "healthy" if self._initialized else "unhealthy",
            "initialized": self._initialized,
            "components": {},
            "statistics": await self.get_memory_statistics()
        }
        
        # Check each component
        components = [
            ("short_term", self.short_term),
            ("long_term", self.long_term),
            ("vector_store", self.vector_store),
            ("episodic", self.episodic)
        ]
        
        for name, component in components:
            if component:
                try:
                    health["components"][name] = await component.health_check()
                except Exception as e:
                    health["components"][name] = {"status": "error", "error": str(e)}
            else:
                health["components"][name] = {"status": "disabled"}
        
        return health
    
    def _filter_and_rank_results(
        self, 
        results: List[MemorySearchResult], 
        query: MemoryQuery
    ) -> List[MemorySearchResult]:
        """Filter and rank search results"""
        filtered = []
        
        for result in results:
            entry = result.entry
            
            # Filter by memory type
            if query.memory_type and entry.content_type != query.memory_type.value:
                continue
            
            # Filter by tags
            if query.tags and not any(tag in entry.tags for tag in query.tags):
                continue
            
            # Filter by time range
            if query.time_range:
                start_time, end_time = query.time_range
                if not (start_time <= entry.timestamp <= end_time):
                    continue
            
            # Filter by importance
            if entry.importance < query.importance_threshold:
                continue
            
            filtered.append(result)
        
        # Remove duplicates (same memory ID)
        seen_ids = set()
        unique_results = []
        for result in filtered:
            if result.entry.id not in seen_ids:
                seen_ids.add(result.entry.id)
                unique_results.append(result)
        
        # Sort by score (descending)
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        return unique_results
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop"""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                await self.cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in cleanup loop", error=str(e))
                await asyncio.sleep(60)  # Wait longer on error


# Utility functions

def create_memory_entry(
    content: Any,
    content_type: Union[str, MemoryType] = MemoryType.FACTUAL,
    importance: Union[float, ImportanceLevel] = ImportanceLevel.MEDIUM,
    tags: List[str] = None,
    ttl_hours: int = None
) -> MemoryEntry:
    """Create a memory entry with convenience parameters"""
    
    if isinstance(content_type, MemoryType):
        content_type = content_type.value
    
    if isinstance(importance, ImportanceLevel):
        importance = importance.value
    
    entry = MemoryEntry(
        content=content,
        content_type=content_type,
        importance=importance,
        tags=tags or []
    )
    
    if ttl_hours:
        entry.expiry = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)
    
    return entry


async def store_conversation_turn(
    memory_system: MemorySystem,
    user_message: str,
    agent_response: str,
    session_id: str = None,
    importance: float = 0.6
) -> bool:
    """Store a conversation turn in memory"""
    
    conversation_data = {
        "user_message": user_message,
        "agent_response": agent_response,
        "session_id": session_id,
        "turn_timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    entry = MemoryEntry(
        content=conversation_data,
        content_type=MemoryType.CONVERSATION.value,
        importance=importance,
        tags=["conversation", "turn", session_id] if session_id else ["conversation", "turn"]
    )
    
    return await memory_system.store_memory(entry)


async def store_learning_experience(
    memory_system: MemorySystem,
    task_type: str,
    success: bool,
    details: Dict[str, Any],
    importance: float = 0.8
) -> bool:
    """Store a learning experience in memory"""
    
    learning_data = {
        "task_type": task_type,
        "success": success,
        "details": details,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    entry = MemoryEntry(
        content=learning_data,
        content_type=MemoryType.LEARNING.value,
        importance=importance,
        tags=["learning", task_type, "success" if success else "failure"]
    )
    
    return await memory_system.store_memory(entry)