"""
Core Agent Implementation
Handles the main agent lifecycle and orchestration.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import asyncio
import logging
import uuid
from contextlib import asynccontextmanager

from .brain import Brain
from .memory import MemorySystem
from .decision_engine import DecisionEngine
from .task_coordinator import TaskCoordinator
from ..interfaces.base import LLMInterface, ToolInterface
from ..communication.message_handler import MessageHandler
from ..communication.context_manager import ContextManager
from ..monitoring.metrics import MetricsCollector
from ..monitoring.logger import StructuredLogger
from ..config.settings import AgentConfig
from ..utils.exceptions import (
    AgentException,
    AgentNotInitializedException,
    AgentAlreadyRunningException,
    TaskExecutionException
)


class AgentStatus(Enum):
    """Agent status enumeration"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PROCESSING = "processing"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


@dataclass
class AgentState:
    """Current state of the agent"""
    status: AgentStatus = AgentStatus.IDLE
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    current_task: Optional[str] = None
    active_plugins: List[str] = field(default_factory=list)
    memory_usage: float = 0.0
    last_activity: Optional[datetime] = None
    session_count: int = 0
    error_count: int = 0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now(timezone.utc)
    
    def increment_session(self):
        """Increment session counter"""
        self.session_count += 1
    
    def increment_error(self):
        """Increment error counter"""
        self.error_count += 1


@dataclass
class ProcessingContext:
    """Context for processing a message or task"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for all AI agents.
    Defines the core interface and lifecycle management.
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.state = AgentState()
        self.logger = StructuredLogger(self.__class__.__name__)
        self.metrics = MetricsCollector()
        
        # Core components
        self.brain: Optional[Brain] = None
        self.memory: Optional[MemorySystem] = None
        self.decision_engine: Optional[DecisionEngine] = None
        self.task_coordinator: Optional[TaskCoordinator] = None
        self.message_handler: Optional[MessageHandler] = None
        self.context_manager: Optional[ContextManager] = None
        
        # Runtime state
        self._running = False
        self._tasks: Dict[str, asyncio.Task] = {}
        self._shutdown_event = asyncio.Event()
        self._initialization_lock = asyncio.Lock()
    
    @property
    def is_running(self) -> bool:
        """Check if agent is currently running"""
        return self._running and self.state.status in [AgentStatus.RUNNING, AgentStatus.PROCESSING]
    
    @property
    def is_initialized(self) -> bool:
        """Check if agent is properly initialized"""
        return all([
            self.brain,
            self.memory,
            self.decision_engine,
            self.task_coordinator,
            self.message_handler,
            self.context_manager
        ])
    
    async def start(self) -> None:
        """
        Start the agent and initialize all components.
        
        Raises:
            AgentAlreadyRunningException: If agent is already running
            AgentException: If initialization fails
        """
        if self._running:
            raise AgentAlreadyRunningException("Agent is already running")
        
        async with self._initialization_lock:
            try:
                self.state.status = AgentStatus.INITIALIZING
                self.logger.info("Starting AI Agent", agent_id=self.state.agent_id)
                
                # Initialize components
                await self._initialize_components()
                
                # Start background tasks
                await self._start_background_tasks()
                
                # Mark as running
                self._running = True
                self.state.status = AgentStatus.RUNNING
                self.state.update_activity()
                
                self.metrics.agent_started()
                self.logger.info("AI Agent started successfully", agent_id=self.state.agent_id)
                
            except Exception as e:
                self.state.status = AgentStatus.ERROR
                self.state.increment_error()
                self.logger.error("Failed to start agent", error=str(e), agent_id=self.state.agent_id)
                await self._cleanup_components()
                raise AgentException(f"Failed to start agent: {e}") from e
    
    async def stop(self, timeout: float = 30.0) -> None:
        """
        Stop the agent gracefully.
        
        Args:
            timeout: Maximum time to wait for graceful shutdown
        """
        if not self._running:
            return
        
        self.state.status = AgentStatus.SHUTTING_DOWN
        self.logger.info("Stopping AI Agent", agent_id=self.state.agent_id)
        
        try:
            # Signal shutdown
            self._shutdown_event.set()
            self._running = False
            
            # Cancel background tasks
            await self._stop_background_tasks(timeout)
            
            # Cleanup components
            await self._cleanup_components()
            
            self.state.status = AgentStatus.STOPPED
            self.metrics.agent_stopped()
            self.logger.info("AI Agent stopped successfully", agent_id=self.state.agent_id)
            
        except Exception as e:
            self.state.status = AgentStatus.ERROR
            self.logger.error("Error during agent shutdown", error=str(e), agent_id=self.state.agent_id)
            raise AgentException(f"Failed to stop agent gracefully: {e}") from e
    
    @abstractmethod
    async def process_message(self, message: str, context: ProcessingContext = None) -> str:
        """
        Process a message from user or system.
        
        Args:
            message: The message to process
            context: Processing context with session info
            
        Returns:
            Response string
        """
        pass
    
    @abstractmethod
    async def execute_task(self, task: Dict[str, Any], context: ProcessingContext = None) -> Any:
        """
        Execute a specific task.
        
        Args:
            task: Task definition and parameters
            context: Processing context
            
        Returns:
            Task execution result
        """
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on agent and all components.
        
        Returns:
            Health status information
        """
        health_status = {
            "agent_id": self.state.agent_id,
            "status": self.state.status.value,
            "is_running": self.is_running,
            "is_initialized": self.is_initialized,
            "last_activity": self.state.last_activity.isoformat() if self.state.last_activity else None,
            "session_count": self.state.session_count,
            "error_count": self.state.error_count,
            "memory_usage": self.state.memory_usage,
            "components": {}
        }
        
        if self.is_initialized:
            # Check component health
            try:
                health_status["components"]["brain"] = await self.brain.health_check()
                health_status["components"]["memory"] = await self.memory.health_check()
                health_status["components"]["decision_engine"] = await self.decision_engine.health_check()
                health_status["components"]["task_coordinator"] = await self.task_coordinator.health_check()
            except Exception as e:
                health_status["components_error"] = str(e)
        
        return health_status
    
    async def _initialize_components(self) -> None:
        """Initialize all agent components"""
        self.logger.debug("Initializing agent components")
        
        # Initialize core components
        self.brain = Brain(self.config.brain_config)
        await self.brain.initialize()
        
        self.memory = MemorySystem(self.config.memory_config)
        await self.memory.initialize()
        
        self.decision_engine = DecisionEngine(self.config.decision_config)
        await self.decision_engine.initialize()
        
        self.task_coordinator = TaskCoordinator(self.config.task_config)
        await self.task_coordinator.initialize()
        
        self.message_handler = MessageHandler(self.config.communication_config)
        await self.message_handler.initialize()
        
        self.context_manager = ContextManager(self.config.context_config)
        await self.context_manager.initialize()
        
        # Connect components
        await self._connect_components()
    
    async def _connect_components(self) -> None:
        """Connect components together"""
        # Brain connections
        self.brain.set_memory(self.memory)
        self.brain.set_decision_engine(self.decision_engine)
        
        # Decision engine connections
        self.decision_engine.set_memory(self.memory)
        self.decision_engine.set_task_coordinator(self.task_coordinator)
        
        # Task coordinator connections
        self.task_coordinator.set_memory(self.memory)
        self.task_coordinator.set_brain(self.brain)
    
    async def _start_background_tasks(self) -> None:
        """Start background monitoring and maintenance tasks"""
        self._tasks["health_monitor"] = asyncio.create_task(self._health_monitor_task())
        self._tasks["memory_cleanup"] = asyncio.create_task(self._memory_cleanup_task())
        self._tasks["metrics_collector"] = asyncio.create_task(self._metrics_collection_task())
    
    async def _stop_background_tasks(self, timeout: float) -> None:
        """Stop all background tasks"""
        if not self._tasks:
            return
        
        # Cancel all tasks
        for task_name, task in self._tasks.items():
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._tasks.values(), return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            self.logger.warning("Some background tasks did not stop within timeout")
        
        self._tasks.clear()
    
    async def _cleanup_components(self) -> None:
        """Cleanup all components"""
        components = [
            self.brain,
            self.memory,
            self.decision_engine,
            self.task_coordinator,
            self.message_handler,
            self.context_manager
        ]
        
        for component in components:
            if component:
                try:
                    await component.cleanup()
                except Exception as e:
                    self.logger.warning(f"Error cleaning up component {component.__class__.__name__}: {e}")
    
    async def _health_monitor_task(self) -> None:
        """Background task for health monitoring"""
        while not self._shutdown_event.is_set():
            try:
                # Update memory usage
                import psutil
                process = psutil.Process()
                self.state.memory_usage = process.memory_percent()
                
                # Check component health
                if self.is_initialized:
                    health_status = await self.health_check()
                    if any(comp.get("status") == "unhealthy" for comp in health_status.get("components", {}).values()):
                        self.logger.warning("Unhealthy components detected", health=health_status)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Health monitor task error", error=str(e))
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _memory_cleanup_task(self) -> None:
        """Background task for memory cleanup"""
        while not self._shutdown_event.is_set():
            try:
                if self.memory:
                    await self.memory.cleanup_expired()
                
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Memory cleanup task error", error=str(e))
                await asyncio.sleep(600)  # Wait longer on error
    
    async def _metrics_collection_task(self) -> None:
        """Background task for metrics collection"""
        while not self._shutdown_event.is_set():
            try:
                # Collect metrics
                await self.metrics.collect_system_metrics()
                if self.is_initialized:
                    await self.metrics.collect_component_metrics({
                        "brain": self.brain,
                        "memory": self.memory,
                        "decision_engine": self.decision_engine,
                        "task_coordinator": self.task_coordinator
                    })
                
                await asyncio.sleep(60)  # Collect every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Metrics collection task error", error=str(e))
                await asyncio.sleep(120)  # Wait longer on error


class AIAgent(BaseAgent):
    """
    Main AI Agent implementation with LLM integration.
    """
    
    def __init__(self, config: AgentConfig, llm_interface: LLMInterface):
        super().__init__(config)
        self.llm = llm_interface
        self.tools: Dict[str, ToolInterface] = {}
        self.plugins: Dict[str, Any] = {}
        self._conversation_history: List[Dict[str, Any]] = []
    
    async def process_message(self, message: str, context: ProcessingContext = None) -> str:
        """
        Process a message from user.
        
        Args:
            message: User message
            context: Processing context
            
        Returns:
            Agent response
        """
        if not self.is_initialized:
            raise AgentNotInitializedException("Agent is not initialized")
        
        if context is None:
            context = ProcessingContext()
        
        self.state.status = AgentStatus.PROCESSING
        self.state.update_activity()
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Enhance context with relevant information
            enhanced_context = await self._enhance_context(message, context)
            
            # Process through message handler
            processed_message = await self.message_handler.process_incoming(message, enhanced_context)
            
            # Analyze message with brain
            analysis = await self.brain.analyze_message(processed_message, enhanced_context)
            
            # Make decision about how to respond
            decision = await self.decision_engine.decide(analysis)
            
            # Execute the decision
            response = await self._execute_decision(decision, enhanced_context)
            
            # Store interaction in memory
            await self._store_interaction(message, response, context)
            
            # Update metrics
            processing_time = asyncio.get_event_loop().time() - start_time
            await self.metrics.record_message_processed(processing_time)
            
            self.state.status = AgentStatus.RUNNING
            return response
            
        except Exception as e:
            self.state.status = AgentStatus.ERROR
            self.state.increment_error()
            self.logger.error("Message processing failed", error=str(e), message=message[:100])
            return await self._handle_error(e, message, context)
        
        finally:
            if self.state.status == AgentStatus.PROCESSING:
                self.state.status = AgentStatus.RUNNING
    
    async def execute_task(self, task: Dict[str, Any], context: ProcessingContext = None) -> Any:
        """
        Execute a complex task.
        
        Args:
            task: Task definition
            context: Processing context
            
        Returns:
            Task result
        """
        if not self.is_initialized:
            raise AgentNotInitializedException("Agent is not initialized")
        
        task_id = task.get("id", str(uuid.uuid4()))
        
        if context is None:
            context = ProcessingContext()
        
        self.state.current_task = task_id
        
        try:
            self.logger.info("Starting task execution", task_id=task_id, task_type=task.get("type"))
            
            # Create execution plan
            plan = await self.brain.create_execution_plan(task)
            
            # Execute through coordinator
            result = await self.task_coordinator.execute_plan(plan, context)
            
            # Learn from execution
            await self._learn_from_execution(task, result, context)
            
            self.logger.info("Task execution completed", task_id=task_id)
            return result
            
        except Exception as e:
            self.logger.error("Task execution failed", task_id=task_id, error=str(e))
            raise TaskExecutionException(f"Task {task_id} failed: {e}") from e
        
        finally:
            self.state.current_task = None
    
    def register_tool(self, name: str, tool: ToolInterface) -> None:
        """Register a tool for use by the agent"""
        self.tools[name] = tool
        if self.brain:
            self.brain.register_capability(name, tool.get_description())
        self.logger.info("Tool registered", tool_name=name)
    
    def register_plugin(self, plugin) -> None:
        """Register a plugin"""
        plugin_name = plugin.get_name()
        self.plugins[plugin_name] = plugin
        plugin.initialize(self)
        self.state.active_plugins.append(plugin_name)
        self.logger.info("Plugin registered", plugin_name=plugin_name)
    
    def unregister_plugin(self, plugin_name: str) -> None:
        """Unregister a plugin"""
        if plugin_name in self.plugins:
            plugin = self.plugins[plugin_name]
            plugin.cleanup()
            del self.plugins[plugin_name]
            if plugin_name in self.state.active_plugins:
                self.state.active_plugins.remove(plugin_name)
            self.logger.info("Plugin unregistered", plugin_name=plugin_name)
    
    @asynccontextmanager
    async def conversation_session(self, session_id: str = None) -> AsyncGenerator[str, None]:
        """
        Context manager for conversation sessions.
        
        Args:
            session_id: Optional session ID
            
        Yields:
            Session ID
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        self.state.increment_session()
        self.logger.info("Starting conversation session", session_id=session_id)
        
        try:
            yield session_id
        finally:
            # Session cleanup if needed
            self.logger.info("Ending conversation session", session_id=session_id)
    
    async def _enhance_context(self, message: str, context: ProcessingContext) -> Dict[str, Any]:
        """Enhance processing context with relevant information"""
        # Retrieve relevant memories
        relevant_memories = await self.memory.retrieve_relevant(message, limit=10)
        
        # Get conversation history
        conversation_context = self._conversation_history[-5:] if self._conversation_history else []
        
        # Build enhanced context
        enhanced = {
            "original_message": message,
            "session_id": context.session_id,
            "user_id": context.user_id,
            "timestamp": context.timestamp,
            "metadata": context.metadata,
            "relevant_memories": relevant_memories,
            "conversation_history": conversation_context,
            "agent_state": {
                "status": self.state.status.value,
                "active_plugins": self.state.active_plugins,
                "available_tools": list(self.tools.keys()),
                "session_count": self.state.session_count
            }
        }
        
        return enhanced
    
    async def _execute_decision(self, decision: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Execute a decision made by the decision engine"""
        action_type = decision.get("action_type")
        
        if action_type == "respond":
            return await self._generate_response(decision, context)
        elif action_type == "use_tool":
            return await self._use_tool(decision, context)
        elif action_type == "delegate_task":
            return await self._delegate_task(decision, context)
        elif action_type == "request_clarification":
            return await self._request_clarification(decision, context)
        else:
            self.logger.warning("Unknown action type", action_type=action_type)
            return "I'm not sure how to handle that request. Could you please rephrase it?"
    
    async def _generate_response(self, decision: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate a response using the brain"""
        return await self.brain.generate_response(decision, context)
    
    async def _use_tool(self, decision: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Execute a tool and interpret the result"""
        tool_name = decision.get("tool_name")
        tool_args = decision.get("tool_args", {})
        
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' is not available."
        
        try:
            tool = self.tools[tool_name]
            result = await tool.execute(tool_args)
            
            if result.success:
                # Interpret result through brain
                return await self.brain.interpret_tool_result(result, context)
            else:
                return f"Tool execution failed: {result.error}"
                
        except Exception as e:
            self.logger.error("Tool execution error", tool_name=tool_name, error=str(e))
            return f"An error occurred while using {tool_name}: {str(e)}"
    
    async def _delegate_task(self, decision: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Delegate a task to the task coordinator"""
        task = decision.get("task")
        try:
            result = await self.task_coordinator.execute_task(task)
            return await self.brain.format_task_result(result, context)
        except Exception as e:
            self.logger.error("Task delegation error", task=task, error=str(e))
            return f"Task execution failed: {str(e)}"
    
    async def _request_clarification(self, decision: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Request clarification from user"""
        clarification_prompt = decision.get("clarification_prompt", "Could you please provide more details?")
        return await self.brain.generate_clarification_request(clarification_prompt, context)
    
    async def _store_interaction(self, message: str, response: str, context: ProcessingContext) -> None:
        """Store the interaction in memory and conversation history"""
        # Store in memory system
        interaction = {
            "user_message": message,
            "agent_response": response,
            "timestamp": context.timestamp,
            "session_id": context.session_id,
            "user_id": context.user_id
        }
        
        await self.memory.store_interaction(interaction)
        
        # Add to conversation history
        self._conversation_history.append(interaction)
        
        # Keep only last 50 interactions in memory
        if len(self._conversation_history) > 50:
            self._conversation_history = self._conversation_history[-50:]
    
    async def _learn_from_execution(self, task: Dict[str, Any], result: Any, context: ProcessingContext) -> None:
        """Learn from task execution results"""
        learning_data = {
            "task": task,
            "result": result,
            "context": context,
            "success": result.get("success", True) if isinstance(result, dict) else True
        }
        
        # Store execution experience
        await self.memory.store_task_execution(learning_data)
        
        # Update brain knowledge
        await self.brain.learn_from_experience(learning_data)
    
    async def _handle_error(self, error: Exception, message: str, context: ProcessingContext) -> str:
        """Handle errors gracefully"""
        error_context = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "original_message": message,
            "context": context
        }
        
        # Try to generate a helpful error response
        try:
            return await self.brain.generate_error_response(error_context)
        except Exception:
            # Fallback response if brain fails
            return "I apologize, but I encountered an error while processing your request. Please try again or rephrase your question."