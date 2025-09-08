"""
Base Interfaces for AI Agent Components
Defines abstract interfaces for all major components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import uuid


class ComponentStatus(Enum):
    """Status enumeration for components"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


@dataclass
class ToolResult:
    """Result of tool execution"""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    tool_name: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
            "execution_time": self.execution_time,
            "tool_name": self.tool_name,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ToolCapability:
    """Description of a tool's capabilities"""
    name: str
    description: str
    parameters: Dict[str, Any]
    return_type: str
    required_permissions: List[str] = field(default_factory=list)
    usage_examples: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    author: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "return_type": self.return_type,
            "required_permissions": self.required_permissions,
            "usage_examples": self.usage_examples,
            "tags": self.tags,
            "version": self.version,
            "author": self.author
        }


@dataclass
class Message:
    """Standard message format"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    sender: str = "user"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    parent_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "content": self.content,
            "sender": self.sender,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "session_id": self.session_id,
            "parent_id": self.parent_id
        }


@dataclass
class MemoryEntry:
    """Memory entry structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Any = None
    content_type: str = "text"
    tags: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expiry: Optional[datetime] = None
    importance: float = 0.5  # 0.0 to 1.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "content": self.content,
            "content_type": self.content_type,
            "tags": self.tags,
            "timestamp": self.timestamp.isoformat(),
            "expiry": self.expiry.isoformat() if self.expiry else None,
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "metadata": self.metadata
        }


class BaseComponent(ABC):
    """Base class for all agent components"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.status = ComponentStatus.UNINITIALIZED
        self.component_id = str(uuid.uuid4())
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the component"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup component resources"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        pass
    
    @property
    def is_initialized(self) -> bool:
        """Check if component is initialized"""
        return self._initialized
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get basic component information"""
        return {
            "id": self.component_id,
            "type": self.__class__.__name__,
            "status": self.status.value,
            "initialized": self.is_initialized,
            "config": self.config.copy()
        }


class LLMInterface(BaseComponent):
    """Interface for Large Language Model integrations"""
    
    @abstractmethod
    async def generate_response(self, prompt: str, context: Dict[str, Any] = None, **kwargs) -> str:
        """
        Generate a response based on the prompt.
        
        Args:
            prompt: Input prompt
            context: Additional context information
            **kwargs: Model-specific parameters
            
        Returns:
            Generated response text
        """
        pass
    
    @abstractmethod
    async def generate_streaming_response(self, prompt: str, context: Dict[str, Any] = None, **kwargs) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response.
        
        Args:
            prompt: Input prompt
            context: Additional context information
            **kwargs: Model-specific parameters
            
        Yields:
            Response chunks as they are generated
        """
        pass
    
    @abstractmethod
    async def analyze_text(self, text: str, analysis_type: str = "general", **kwargs) -> Dict[str, Any]:
        """
        Analyze text content.
        
        Args:
            text: Text to analyze
            analysis_type: Type of analysis (sentiment, entities, summary, etc.)
            **kwargs: Analysis-specific parameters
            
        Returns:
            Analysis results
        """
        pass
    
    @abstractmethod
    async def classify_intent(self, message: str, possible_intents: List[str] = None) -> Dict[str, Any]:
        """
        Classify the intent of a message.
        
        Args:
            message: Message to classify
            possible_intents: Optional list of possible intents
            
        Returns:
            Intent classification results
        """
        pass
    
    @abstractmethod
    async def extract_entities(self, text: str, entity_types: List[str] = None) -> List[Dict[str, Any]]:
        """
        Extract entities from text.
        
        Args:
            text: Text to process
            entity_types: Types of entities to extract
            
        Returns:
            List of extracted entities
        """
        pass
    
    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the underlying model.
        
        Returns:
            Model information
        """
        pass


class ToolInterface(BaseComponent):
    """Interface for agent tools"""
    
    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """
        Execute the tool with given parameters.
        
        Args:
            parameters: Tool execution parameters
            
        Returns:
            Tool execution result
        """
        pass
    
    @abstractmethod
    async def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate tool parameters before execution.
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            Validation result with errors if any
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[ToolCapability]:
        """
        Get tool capabilities description.
        
        Returns:
            List of tool capabilities
        """
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Get tool parameter schema.
        
        Returns:
            JSON schema for parameters
        """
        pass
    
    def get_description(self) -> str:
        """
        Get tool description.
        
        Returns:
            Tool description
        """
        capabilities = self.get_capabilities()
        if capabilities:
            return capabilities[0].description
        return f"Tool: {self.__class__.__name__}"
    
    async def dry_run(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a dry run to see what would happen without executing.
        
        Args:
            parameters: Tool parameters
            
        Returns:
            Dry run result
        """
        validation_result = await self.validate_parameters(parameters)
        if not validation_result.get("valid", False):
            return {
                "would_succeed": False,
                "errors": validation_result.get("errors", []),
                "estimated_time": 0
            }
        
        return {
            "would_succeed": True,
            "estimated_time": 5.0,  # Default estimate
            "side_effects": [],
            "resources_required": []
        }


class DataInterface(BaseComponent):
    """Interface for data storage and retrieval"""
    
    @abstractmethod
    async def store(self, key: str, data: Any, metadata: Dict[str, Any] = None, ttl: int = None) -> bool:
        """
        Store data with optional TTL.
        
        Args:
            key: Storage key
            data: Data to store
            metadata: Optional metadata
            ttl: Time to live in seconds
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    async def retrieve(self, key: str) -> Optional[Any]:
        """
        Retrieve data by key.
        
        Args:
            key: Storage key
            
        Returns:
            Retrieved data or None
        """
        pass
    
    @abstractmethod
    async def search(self, query: str, filters: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search for data.
        
        Args:
            query: Search query
            filters: Optional filters
            limit: Maximum results
            
        Returns:
            Search results
        """
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        Delete data by key.
        
        Args:
            key: Storage key
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    async def list_keys(self, pattern: str = "*", limit: int = 1000) -> List[str]:
        """
        List keys matching pattern.
        
        Args:
            pattern: Key pattern
            limit: Maximum results
            
        Returns:
            List of matching keys
        """
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        Check if key exists.
        
        Args:
            key: Storage key
            
        Returns:
            True if key exists
        """
        pass
    
    @abstractmethod
    async def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a key.
        
        Args:
            key: Storage key
            
        Returns:
            Metadata or None
        """
        pass
    
    async def bulk_store(self, items: Dict[str, Any], metadata: Dict[str, Any] = None) -> Dict[str, bool]:
        """
        Store multiple items.
        
        Args:
            items: Dictionary of key-value pairs
            metadata: Optional metadata for all items
            
        Returns:
            Dictionary of key-success pairs
        """
        results = {}
        for key, data in items.items():
            try:
                results[key] = await self.store(key, data, metadata)
            except Exception:
                results[key] = False
        return results
    
    async def bulk_retrieve(self, keys: List[str]) -> Dict[str, Any]:
        """
        Retrieve multiple items.
        
        Args:
            keys: List of keys to retrieve
            
        Returns:
            Dictionary of key-value pairs
        """
        results = {}
        for key in keys:
            try:
                results[key] = await self.retrieve(key)
            except Exception:
                results[key] = None
        return results


class VectorStoreInterface(BaseComponent):
    """Interface for vector storage and similarity search"""
    
    @abstractmethod
    async def add_vectors(self, vectors: List[List[float]], metadata: List[Dict[str, Any]], ids: List[str] = None) -> bool:
        """
        Add vectors to the store.
        
        Args:
            vectors: List of vector embeddings
            metadata: List of metadata for each vector
            ids: Optional IDs for vectors
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    async def search_similar(self, query_vector: List[float], top_k: int = 10, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector
            top_k: Number of results
            filters: Optional metadata filters
            
        Returns:
            List of similar vectors with metadata and scores
        """
        pass
    
    @abstractmethod
    async def delete_vectors(self, ids: List[str]) -> bool:
        """
        Delete vectors by IDs.
        
        Args:
            ids: Vector IDs to delete
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    async def update_vector(self, id: str, vector: List[float] = None, metadata: Dict[str, Any] = None) -> bool:
        """
        Update a vector or its metadata.
        
        Args:
            id: Vector ID
            vector: New vector (optional)
            metadata: New metadata (optional)
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    async def get_vector_count(self) -> int:
        """
        Get total number of vectors in store.
        
        Returns:
            Vector count
        """
        pass


class MemoryInterface(BaseComponent):
    """Interface for agent memory systems"""
    
    @abstractmethod
    async def store_memory(self, entry: MemoryEntry) -> bool:
        """
        Store a memory entry.
        
        Args:
            entry: Memory entry to store
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    async def retrieve_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """
        Retrieve a specific memory.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            Memory entry or None
        """
        pass
    
    @abstractmethod
    async def search_memories(self, query: str, memory_type: str = None, limit: int = 10) -> List[MemoryEntry]:
        """
        Search memories by content.
        
        Args:
            query: Search query
            memory_type: Optional memory type filter
            limit: Maximum results
            
        Returns:
            List of matching memories
        """
        pass
    
    @abstractmethod
    async def get_recent_memories(self, limit: int = 10, memory_type: str = None) -> List[MemoryEntry]:
        """
        Get recent memories.
        
        Args:
            limit: Maximum results
            memory_type: Optional memory type filter
            
        Returns:
            List of recent memories
        """
        pass
    
    @abstractmethod
    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    async def cleanup_expired(self) -> int:
        """
        Clean up expired memories.
        
        Returns:
            Number of memories cleaned up
        """
        pass
    
    async def store_interaction(self, user_message: str, agent_response: str, context: Dict[str, Any] = None) -> bool:
        """
        Store a user-agent interaction.
        
        Args:
            user_message: User's message
            agent_response: Agent's response
            context: Optional context
            
        Returns:
            Success status
        """
        interaction_data = {
            "user_message": user_message,
            "agent_response": agent_response,
            "context": context or {}
        }
        
        entry = MemoryEntry(
            content=interaction_data,
            content_type="interaction",
            tags=["interaction", "conversation"],
            importance=0.6
        )
        
        return await self.store_memory(entry)
    
    async def get_conversation_history(self, session_id: str = None, limit: int = 20) -> List[MemoryEntry]:
        """
        Get conversation history.
        
        Args:
            session_id: Optional session filter
            limit: Maximum results
            
        Returns:
            List of conversation memories
        """
        memories = await self.search_memories("interaction", "interaction", limit * 2)
        
        if session_id:
            memories = [m for m in memories if m.metadata.get("session_id") == session_id]
        
        return memories[:limit]


class PluginInterface(BaseComponent):
    """Interface for agent plugins"""
    
    @abstractmethod
    def get_name(self) -> str:
        """Get plugin name"""
        pass
    
    @abstractmethod
    def get_version(self) -> str:
        """Get plugin version"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get plugin description"""
        pass
    
    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """Get plugin dependencies"""
        pass
    
    @abstractmethod
    def get_author(self) -> str:
        """Get plugin author"""
        pass
    
    @abstractmethod
    async def on_load(self, agent_context: Any) -> None:
        """Called when plugin is loaded"""
        pass
    
    @abstractmethod
    async def on_unload(self) -> None:
        """Called when plugin is unloaded"""
        pass
    
    def get_plugin_info(self) -> Dict[str, Any]:
        """Get plugin information"""
        return {
            "name": self.get_name(),
            "version": self.get_version(),
            "description": self.get_description(),
            "dependencies": self.get_dependencies(),
            "author": self.get_author(),
            "component_info": self.get_component_info()
        }
    
    async def execute_action(self, action: str, parameters: Dict[str, Any]) -> Any:
        """
        Execute a plugin action.
        
        Args:
            action: Action name
            parameters: Action parameters
            
        Returns:
            Action result
        """
        method_name = f"action_{action}"
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            if callable(method):
                return await method(parameters)
        
        raise NotImplementedError(f"Action '{action}' not implemented in plugin '{self.get_name()}'")


class CommunicationInterface(BaseComponent):
    """Interface for communication channels"""
    
    @abstractmethod
    async def send_message(self, message: Message, target: str = None) -> bool:
        """
        Send a message.
        
        Args:
            message: Message to send
            target: Optional target identifier
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    async def receive_messages(self) -> AsyncGenerator[Message, None]:
        """
        Receive messages asynchronously.
        
        Yields:
            Received messages
        """
        pass
    
    @abstractmethod
    async def setup_channel(self, channel_config: Dict[str, Any]) -> bool:
        """
        Set up a communication channel.
        
        Args:
            channel_config: Channel configuration
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    async def close_channel(self, channel_id: str) -> bool:
        """
        Close a communication channel.
        
        Args:
            channel_id: Channel identifier
            
        Returns:
            Success status
        """
        pass
    
    def get_supported_formats(self) -> List[str]:
        """Get supported message formats"""
        return ["text", "json"]
    
    def get_channel_info(self) -> Dict[str, Any]:
        """Get channel information"""
        return {
            "type": self.__class__.__name__,
            "supported_formats": self.get_supported_formats(),
            "component_info": self.get_component_info()
        }


class SecurityInterface(BaseComponent):
    """Interface for security and authentication"""
    
    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """
        Authenticate user credentials.
        
        Args:
            credentials: User credentials
            
        Returns:
            Authentication result
        """
        pass
    
    @abstractmethod
    async def authorize(self, user_id: str, resource: str, action: str) -> bool:
        """
        Check if user is authorized for action.
        
        Args:
            user_id: User identifier
            resource: Resource identifier
            action: Action to perform
            
        Returns:
            Authorization status
        """
        pass
    
    @abstractmethod
    async def validate_input(self, input_data: Any, validation_type: str = "general") -> Dict[str, Any]:
        """
        Validate input data for security.
        
        Args:
            input_data: Data to validate
            validation_type: Type of validation
            
        Returns:
            Validation result
        """
        pass
    
    @abstractmethod
    async def sanitize_input(self, input_data: str) -> str:
        """
        Sanitize input string.
        
        Args:
            input_data: Input to sanitize
            
        Returns:
            Sanitized input
        """
        pass
    
    @abstractmethod
    async def encrypt_data(self, data: str, key: str = None) -> str:
        """
        Encrypt sensitive data.
        
        Args:
            data: Data to encrypt
            key: Optional encryption key
            
        Returns:
            Encrypted data
        """
        pass
    
    @abstractmethod
    async def decrypt_data(self, encrypted_data: str, key: str = None) -> str:
        """
        Decrypt encrypted data.
        
        Args:
            encrypted_data: Data to decrypt
            key: Optional decryption key
            
        Returns:
            Decrypted data
        """
        pass


class MonitoringInterface(BaseComponent):
    """Interface for monitoring and metrics"""
    
    @abstractmethod
    async def record_metric(self, name: str, value: Union[int, float], labels: Dict[str, str] = None, timestamp: datetime = None) -> None:
        """
        Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            labels: Optional labels
            timestamp: Optional timestamp
        """
        pass
    
    @abstractmethod
    async def record_event(self, event_type: str, data: Dict[str, Any], timestamp: datetime = None) -> None:
        """
        Record an event.
        
        Args:
            event_type: Event type
            data: Event data
            timestamp: Optional timestamp
        """
        pass
    
    @abstractmethod
    async def get_metrics(self, name_pattern: str = "*", time_range: tuple = None) -> List[Dict[str, Any]]:
        """
        Get metrics data.
        
        Args:
            name_pattern: Metric name pattern
            time_range: Optional time range (start, end)
            
        Returns:
            List of metric data points
        """
        pass
    
    @abstractmethod
    async def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health status.
        
        Returns:
            Health status information
        """
        pass
    
    async def start_timer(self, name: str) -> str:
        """
        Start a timer for measuring duration.
        
        Args:
            name: Timer name
            
        Returns:
            Timer ID
        """
        timer_id = str(uuid.uuid4())
        await self.record_event("timer_start", {
            "timer_id": timer_id,
            "name": name,
            "start_time": datetime.now(timezone.utc).isoformat()
        })
        return timer_id
    
    async def stop_timer(self, timer_id: str, labels: Dict[str, str] = None) -> float:
        """
        Stop a timer and record duration.
        
        Args:
            timer_id: Timer ID
            labels: Optional labels
            
        Returns:
            Duration in seconds
        """
        end_time = datetime.now(timezone.utc)
        # In real implementation, would calculate duration from start time
        duration = 0.0  # Placeholder
        
        await self.record_event("timer_stop", {
            "timer_id": timer_id,
            "end_time": end_time.isoformat(),
            "duration": duration
        })
        
        await self.record_metric("duration", duration, labels)
        return duration