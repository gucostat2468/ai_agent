"""
Task Coordinator - Complex Task Execution Management
Handles task planning, execution, monitoring, and coordination of multi-step operations.
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import json
import traceback

from ..interfaces.base import BaseComponent
from ..config.settings import TaskConfig
from ..monitoring.logger import StructuredLogger, log_performance
from ..utils.exceptions import TaskException, TaskExecutionException, TaskTimeoutException, TaskValidationException


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class ExecutionStrategy(Enum):
    """Task execution strategies"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    CONDITIONAL = "conditional"


@dataclass
class TaskStep:
    """Individual step in a task execution plan"""
    step_id: str
    name: str
    description: str
    action_type: str  # "llm_call", "tool_execution", "data_processing", etc.
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "step_id": self.step_id,
            "name": self.name,
            "description": self.description,
            "action_type": self.action_type,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "status": self.status.value,
            "result": str(self.result) if self.result is not None else None,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


@dataclass
class Task:
    """Task definition with execution plan"""
    task_id: str
    name: str
    description: str
    task_type: str
    priority: TaskPriority = TaskPriority.NORMAL
    execution_strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL
    steps: List[TaskStep] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[float] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    result: Any = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "task_type": self.task_type,
            "priority": self.priority.value,
            "execution_strategy": self.execution_strategy.value,
            "steps": [step.to_dict() for step in self.steps],
            "context": self.context,
            "metadata": self.metadata,
            "timeout": self.timeout,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status.value,
            "progress": self.progress,
            "result": str(self.result) if self.result is not None else None,
            "error": self.error
        }


class TaskExecutor(ABC):
    """Abstract base class for task step executors"""
    
    @abstractmethod
    async def execute(self, step: TaskStep, context: Dict[str, Any]) -> Any:
        """Execute a task step"""
        pass
    
    @abstractmethod
    def can_handle(self, action_type: str) -> bool:
        """Check if executor can handle a specific action type"""
        pass
    
    @abstractmethod
    def get_supported_actions(self) -> List[str]:
        """Get list of supported action types"""
        pass


class LLMCallExecutor(TaskExecutor):
    """Executor for LLM API calls"""
    
    def __init__(self, llm_interface):
        self.llm = llm_interface
        self.logger = StructuredLogger(__name__)
    
    async def execute(self, step: TaskStep, context: Dict[str, Any]) -> Any:
        """Execute LLM call step"""
        try:
            prompt = step.parameters.get("prompt", "")
            model_params = step.parameters.get("model_params", {})
            
            if not prompt:
                raise TaskExecutionException("No prompt provided for LLM call", step.step_id)
            
            # Substitute context variables in prompt
            prompt = self._substitute_variables(prompt, context)
            
            # Make LLM call
            response = await self.llm.generate_response(prompt, context, **model_params)
            
            return {
                "response": response,
                "prompt": prompt,
                "model_params": model_params
            }
            
        except Exception as e:
            self.logger.error("LLM call execution failed", step_id=step.step_id, error=str(e))
            raise TaskExecutionException(f"LLM call failed: {e}", step.step_id) from e
    
    def can_handle(self, action_type: str) -> bool:
        return action_type == "llm_call"
    
    def get_supported_actions(self) -> List[str]:
        return ["llm_call"]
    
    def _substitute_variables(self, prompt: str, context: Dict[str, Any]) -> str:
        """Substitute variables in prompt template"""
        import string
        template = string.Template(prompt)
        return template.safe_substitute(context)


class ToolExecutor(TaskExecutor):
    """Executor for tool operations"""
    
    def __init__(self, tool_registry):
        self.tool_registry = tool_registry
        self.logger = StructuredLogger(__name__)
    
    async def execute(self, step: TaskStep, context: Dict[str, Any]) -> Any:
        """Execute tool operation step"""
        try:
            tool_name = step.parameters.get("tool_name")
            tool_params = step.parameters.get("tool_params", {})
            
            if not tool_name:
                raise TaskExecutionException("No tool name provided", step.step_id)
            
            # Get tool from registry
            tool = self.tool_registry.get_tool(tool_name)
            if not tool:
                raise TaskExecutionException(f"Tool '{tool_name}' not found", step.step_id)
            
            # Substitute context variables in parameters
            tool_params = self._substitute_variables(tool_params, context)
            
            # Execute tool
            result = await tool.execute(tool_params)
            
            return {
                "tool_result": result,
                "tool_name": tool_name,
                "parameters": tool_params
            }
            
        except Exception as e:
            self.logger.error("Tool execution failed", step_id=step.step_id, error=str(e))
            raise TaskExecutionException(f"Tool execution failed: {e}", step.step_id) from e
    
    def can_handle(self, action_type: str) -> bool:
        return action_type == "tool_execution"
    
    def get_supported_actions(self) -> List[str]:
        return ["tool_execution"]
    
    def _substitute_variables(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute variables in parameters"""
        substituted = {}
        for key, value in params.items():
            if isinstance(value, str) and "${" in value:
                import string
                template = string.Template(value)
                substituted[key] = template.safe_substitute(context)
            else:
                substituted[key] = value
        return substituted


class DataProcessingExecutor(TaskExecutor):
    """Executor for data processing operations"""
    
    def __init__(self):
        self.logger = StructuredLogger(__name__)
    
    async def execute(self, step: TaskStep, context: Dict[str, Any]) -> Any:
        """Execute data processing step"""
        try:
            operation = step.parameters.get("operation")
            data_source = step.parameters.get("data_source")
            
            if not operation:
                raise TaskExecutionException("No operation specified", step.step_id)
            
            # Get data from context or parameters
            if data_source:
                data = context.get(data_source)
            else:
                data = step.parameters.get("data")
            
            if data is None:
                raise TaskExecutionException("No data provided for processing", step.step_id)
            
            # Process data based on operation
            result = await self._process_data(operation, data, step.parameters)
            
            return {
                "processed_data": result,
                "operation": operation,
                "original_data_type": type(data).__name__
            }
            
        except Exception as e:
            self.logger.error("Data processing failed", step_id=step.step_id, error=str(e))
            raise TaskExecutionException(f"Data processing failed: {e}", step.step_id) from e
    
    def can_handle(self, action_type: str) -> bool:
        return action_type == "data_processing"
    
    def get_supported_actions(self) -> List[str]:
        return ["data_processing"]
    
    async def _process_data(self, operation: str, data: Any, params: Dict[str, Any]) -> Any:
        """Process data based on operation type"""
        
        if operation == "filter":
            condition = params.get("condition", {})
            if isinstance(data, list):
                return [item for item in data if self._matches_condition(item, condition)]
            
        elif operation == "transform":
            transformation = params.get("transformation", {})
            if isinstance(data, list):
                return [self._transform_item(item, transformation) for item in data]
            else:
                return self._transform_item(data, transformation)
        
        elif operation == "aggregate":
            agg_func = params.get("function", "count")
            if isinstance(data, list):
                if agg_func == "count":
                    return len(data)
                elif agg_func == "sum" and all(isinstance(x, (int, float)) for x in data):
                    return sum(data)
                elif agg_func == "average" and all(isinstance(x, (int, float)) for x in data):
                    return sum(data) / len(data) if data else 0
        
        elif operation == "sort":
            key_field = params.get("key_field", "")
            reverse = params.get("reverse", False)
            if isinstance(data, list):
                if key_field and isinstance(data[0], dict):
                    return sorted(data, key=lambda x: x.get(key_field, 0), reverse=reverse)
                else:
                    return sorted(data, reverse=reverse)
        
        return data  # Return unchanged if operation not recognized
    
    def _matches_condition(self, item: Any, condition: Dict[str, Any]) -> bool:
        """Check if item matches filter condition"""
        if not isinstance(item, dict):
            return True
        
        for key, expected_value in condition.items():
            if key not in item or item[key] != expected_value:
                return False
        
        return True
    
    def _transform_item(self, item: Any, transformation: Dict[str, Any]) -> Any:
        """Transform an item based on transformation rules"""
        if not isinstance(item, dict) or not transformation:
            return item
        
        transformed = item.copy()
        
        # Apply transformations
        for field, transform_rule in transformation.items():
            if field in transformed:
                if isinstance(transform_rule, str) and transform_rule.startswith("upper"):
                    transformed[field] = str(transformed[field]).upper()
                elif isinstance(transform_rule, str) and transform_rule.startswith("lower"):
                    transformed[field] = str(transformed[field]).lower()
                # Add more transformation rules as needed
        
        return transformed


class TaskCoordinator(BaseComponent):
    """
    Main task coordinator that manages complex task execution
    """
    
    def __init__(self, config: TaskConfig):
        super().__init__(config.dict() if hasattr(config, 'dict') else config.__dict__)
        self.config = config
        self.logger = StructuredLogger(__name__)
        
        # Task storage and tracking
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.task_queue: List[Task] = []
        
        # Executors
        self.executors: Dict[str, TaskExecutor] = {}
        
        # Dependencies (set by agent)
        self.memory = None
        self.brain = None
        self.tool_registry = None
        self.llm_interface = None
        
        # Performance metrics
        self.metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_execution_time": 0.0,
            "tasks_per_type": {}
        }
        
        # Task execution control
        self._max_concurrent = config.max_concurrent_tasks
        self._running_tasks = 0
        self._shutdown_event = asyncio.Event()
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the task coordinator"""
        try:
            self.logger.info("Initializing Task Coordinator")
            
            # Initialize built-in executors
            self.executors["data_processing"] = DataProcessingExecutor()
            
            # Start task processing loop
            asyncio.create_task(self._task_processing_loop())
            
            self._initialized = True
            self.logger.info("Task Coordinator initialized successfully")
            
        except Exception as e:
            self.logger.error("Failed to initialize Task Coordinator", error=str(e))
            raise TaskException(f"Task coordinator initialization failed: {e}") from e
    
    async def cleanup(self) -> None:
        """Cleanup task coordinator resources"""
        try:
            self.logger.info("Cleaning up Task Coordinator")
            
            # Signal shutdown
            self._shutdown_event.set()
            
            # Cancel running tasks
            await self._cancel_all_tasks()
            
            # Save task history if needed
            await self._save_task_history()
            
            self._initialized = False
            self.logger.info("Task Coordinator cleaned up successfully")
            
        except Exception as e:
            self.logger.error("Error during Task Coordinator cleanup", error=str(e))
    
    def set_memory(self, memory) -> None:
        """Set memory system reference"""
        self.memory = memory
    
    def set_brain(self, brain) -> None:
        """Set brain reference"""
        self.brain = brain
    
    def set_tool_registry(self, tool_registry) -> None:
        """Set tool registry and create tool executor"""
        self.tool_registry = tool_registry
        if tool_registry:
            self.executors["tool_execution"] = ToolExecutor(tool_registry)
    
    def set_llm_interface(self, llm_interface) -> None:
        """Set LLM interface and create LLM executor"""
        self.llm_interface = llm_interface
        if llm_interface:
            self.executors["llm_call"] = LLMCallExecutor(llm_interface)
    
    @log_performance("task_execution")
    async def execute_task(self, task_definition: Dict[str, Any]) -> Any:
        """Execute a single task"""
        
        # Create task from definition
        task = self._create_task_from_definition(task_definition)
        
        # Add to active tasks
        self.active_tasks[task.task_id] = task
        
        try:
            # Execute the task
            result = await self._execute_task(task)
            
            # Mark as completed
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(timezone.utc)
            task.result = result
            task.progress = 1.0
            
            # Move to completed tasks
            self.completed_tasks[task.task_id] = task
            del self.active_tasks[task.task_id]
            
            # Update metrics
            self._update_metrics(task)
            
            return result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now(timezone.utc)
            
            self.logger.error("Task execution failed", task_id=task.task_id, error=str(e))
            raise TaskExecutionException(f"Task {task.task_id} failed: {e}", task.task_id) from e
    
    async def execute_plan(self, execution_plan: Dict[str, Any], context: Dict[str, Any] = None) -> Any:
        """Execute a complex execution plan from the brain"""
        
        task_definition = {
            "name": execution_plan.get("name", "Brain Generated Task"),
            "description": execution_plan.get("description", ""),
            "task_type": "brain_generated",
            "steps": execution_plan.get("steps", []),
            "execution_strategy": execution_plan.get("execution_strategy", "sequential"),
            "context": context or {}
        }
        
        return await self.execute_task(task_definition)
    
    async def queue_task(self, task_definition: Dict[str, Any], priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """Queue a task for later execution"""
        
        task = self._create_task_from_definition(task_definition)
        task.priority = priority
        
        # Insert in queue based on priority
        self._insert_by_priority(task)
        
        self.logger.info("Task queued", task_id=task.task_id, priority=priority.name)
        
        return task.task_id
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running or queued task"""
        
        # Check active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now(timezone.utc)
            
            self.completed_tasks[task_id] = task
            del self.active_tasks[task_id]
            
            self.logger.info("Task cancelled", task_id=task_id)
            return True
        
        # Check queued tasks
        for i, task in enumerate(self.task_queue):
            if task.task_id == task_id:
                task.status = TaskStatus.CANCELLED
                self.task_queue.pop(i)
                self.completed_tasks[task_id] = task
                
                self.logger.info("Queued task cancelled", task_id=task_id)
                return True
        
        return False
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a task"""
        
        # Check active tasks
        if task_id in self.active_tasks:
            return self.active_tasks[task_id].to_dict()
        
        # Check completed tasks
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id].to_dict()
        
        # Check queued tasks
        for task in self.task_queue:
            if task.task_id == task_id:
                return task.to_dict()
        
        return None
    
    def list_active_tasks(self) -> List[Dict[str, Any]]:
        """List all active tasks"""
        return [task.to_dict() for task in self.active_tasks.values()]
    
    def list_queued_tasks(self) -> List[Dict[str, Any]]:
        """List all queued tasks"""
        return [task.to_dict() for task in self.task_queue]
    
    def _create_task_from_definition(self, definition: Dict[str, Any]) -> Task:
        """Create a Task object from definition"""
        
        task_id = definition.get("id", str(uuid.uuid4()))
        
        # Create task steps
        steps = []
        for i, step_def in enumerate(definition.get("steps", [])):
            step = TaskStep(
                step_id=step_def.get("id", f"{task_id}_step_{i}"),
                name=step_def.get("name", f"Step {i+1}"),
                description=step_def.get("description", ""),
                action_type=step_def.get("type", "unknown"),
                parameters=step_def.get("parameters", {}),
                dependencies=step_def.get("dependencies", []),
                timeout=step_def.get("timeout"),
                max_retries=step_def.get("max_retries", self.config.max_retry_attempts)
            )
            steps.append(step)
        
        # Determine execution strategy
        strategy_str = definition.get("execution_strategy", "sequential")
        try:
            execution_strategy = ExecutionStrategy(strategy_str)
        except ValueError:
            execution_strategy = ExecutionStrategy.SEQUENTIAL
        
        # Create task
        task = Task(
            task_id=task_id,
            name=definition.get("name", "Unnamed Task"),
            description=definition.get("description", ""),
            task_type=definition.get("task_type", "generic"),
            steps=steps,
            context=definition.get("context", {}),
            metadata=definition.get("metadata", {}),
            timeout=definition.get("timeout", self.config.task_timeout),
            execution_strategy=execution_strategy
        )
        
        return task
    
    async def _execute_task(self, task: Task) -> Any:
        """Execute a task with its steps"""
        
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now(timezone.utc)
        
        self.logger.info("Starting task execution", 
                        task_id=task.task_id,
                        task_type=task.task_type,
                        steps_count=len(task.steps))
        
        try:
            # Execute based on strategy
            if task.execution_strategy == ExecutionStrategy.SEQUENTIAL:
                result = await self._execute_sequential(task)
            elif task.execution_strategy == ExecutionStrategy.PARALLEL:
                result = await self._execute_parallel(task)
            elif task.execution_strategy == ExecutionStrategy.PIPELINE:
                result = await self._execute_pipeline(task)
            elif task.execution_strategy == ExecutionStrategy.CONDITIONAL:
                result = await self._execute_conditional(task)
            else:
                raise TaskExecutionException(f"Unsupported execution strategy: {task.execution_strategy}")
            
            self.logger.info("Task execution completed", 
                           task_id=task.task_id,
                           execution_time=(datetime.now(timezone.utc) - task.started_at).total_seconds())
            
            return result
            
        except Exception as e:
            self.logger.error("Task execution failed", 
                            task_id=task.task_id, 
                            error=str(e),
                            execution_time=(datetime.now(timezone.utc) - task.started_at).total_seconds())
            raise
    
    async def _execute_sequential(self, task: Task) -> Any:
        """Execute task steps sequentially"""
        
        results = {}
        total_steps = len(task.steps)
        
        for i, step in enumerate(task.steps):
            # Check for cancellation
            if task.status == TaskStatus.CANCELLED:
                break
            
            # Update progress
            task.progress = i / total_steps
            
            # Execute step
            step_result = await self._execute_step(step, task.context, results)
            results[step.step_id] = step_result
            
            # Add step result to context for next steps
            task.context[f"step_{step.step_id}_result"] = step_result
        
        task.progress = 1.0
        return results
    
    async def _execute_parallel(self, task: Task) -> Any:
        """Execute task steps in parallel where possible"""
        
        results = {}
        
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(task.steps)
        
        # Execute in waves based on dependencies
        executed_steps = set()
        
        while len(executed_steps) < len(task.steps):
            # Find steps ready to execute
            ready_steps = []
            for step in task.steps:
                if (step.step_id not in executed_steps and 
                    all(dep in executed_steps for dep in step.dependencies)):
                    ready_steps.append(step)
            
            if not ready_steps:
                break  # No more steps can be executed
            
            # Execute ready steps in parallel
            tasks_to_execute = []
            for step in ready_steps:
                task_coro = self._execute_step(step, task.context, results)
                tasks_to_execute.append((step.step_id, task_coro))
            
            # Execute all ready steps concurrently
            step_results = await asyncio.gather(
                *[task_coro for _, task_coro in tasks_to_execute],
                return_exceptions=True
            )
            
            # Process results
            for (step_id, _), result in zip(tasks_to_execute, step_results):
                if isinstance(result, Exception):
                    raise TaskExecutionException(f"Step {step_id} failed: {result}", task.task_id)
                
                results[step_id] = result
                executed_steps.add(step_id)
                task.context[f"step_{step_id}_result"] = result
            
            # Update progress
            task.progress = len(executed_steps) / len(task.steps)
        
        return results
    
    async def _execute_pipeline(self, task: Task) -> Any:
        """Execute task steps as a pipeline"""
        
        results = {}
        pipeline_data = task.context.get("pipeline_input")
        
        for i, step in enumerate(task.steps):
            # Update progress
            task.progress = i / len(task.steps)
            
            # For pipeline, pass output of previous step as input to next
            if i > 0:
                # Use result from previous step
                prev_step_id = task.steps[i-1].step_id
                pipeline_data = results.get(prev_step_id)
                
                # Add to step parameters
                step.parameters["pipeline_input"] = pipeline_data
            
            # Execute step
            step_result = await self._execute_step(step, task.context, results)
            results[step.step_id] = step_result
            
            # Update pipeline data for next step
            pipeline_data = step_result
        
        task.progress = 1.0
        return pipeline_data  # Return final pipeline output
    
    async def _execute_conditional(self, task: Task) -> Any:
        """Execute task steps based on conditions"""
        
        results = {}
        total_steps = len(task.steps)
        executed_count = 0
        
        for step in task.steps:
            # Check step condition
            if not self._evaluate_step_condition(step, task.context, results):
                self.logger.debug("Skipping step due to condition", 
                                step_id=step.step_id,
                                task_id=task.task_id)
                continue
            
            # Execute step
            step_result = await self._execute_step(step, task.context, results)
            results[step.step_id] = step_result
            
            # Add to context
            task.context[f"step_{step.step_id}_result"] = step_result
            
            executed_count += 1
            task.progress = executed_count / total_steps
        
        return results
    
    async def _execute_step(self, step: TaskStep, context: Dict[str, Any], previous_results: Dict[str, Any]) -> Any:
        """Execute a single task step"""
        
        step.status = TaskStatus.RUNNING
        step.started_at = datetime.now(timezone.utc)
        
        self.logger.debug("Executing step", 
                        step_id=step.step_id,
                        action_type=step.action_type)
        
        # Find appropriate executor
        executor = self._find_executor(step.action_type)
        if not executor:
            raise TaskExecutionException(f"No executor found for action type: {step.action_type}", step.step_id)
        
        # Prepare enhanced context
        enhanced_context = {
            **context,
            **previous_results,
            "current_step": step.step_id,
            "step_parameters": step.parameters
        }
        
        # Execute with timeout and retries
        max_retries = step.max_retries
        retry_delay = self.config.retry_delay
        
        for attempt in range(max_retries + 1):
            try:
                # Apply timeout if specified
                if step.timeout:
                    result = await asyncio.wait_for(
                        executor.execute(step, enhanced_context),
                        timeout=step.timeout
                    )
                else:
                    result = await executor.execute(step, enhanced_context)
                
                # Step completed successfully
                step.status = TaskStatus.COMPLETED
                step.completed_at = datetime.now(timezone.utc)
                step.result = result
                
                self.logger.debug("Step completed", 
                                step_id=step.step_id,
                                execution_time=(step.completed_at - step.started_at).total_seconds())
                
                return result
                
            except asyncio.TimeoutError:
                step.retry_count = attempt
                if attempt < max_retries:
                    self.logger.warning("Step timeout, retrying", 
                                      step_id=step.step_id,
                                      attempt=attempt + 1,
                                      timeout=step.timeout)
                    await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    step.status = TaskStatus.FAILED
                    step.error = f"Step timed out after {step.timeout}s"
                    step.completed_at = datetime.now(timezone.utc)
                    raise TaskTimeoutException(f"Step {step.step_id} timed out", step.timeout)
            
            except Exception as e:
                step.retry_count = attempt
                if attempt < max_retries:
                    self.logger.warning("Step execution failed, retrying", 
                                      step_id=step.step_id,
                                      attempt=attempt + 1,
                                      error=str(e))
                    await asyncio.sleep(retry_delay * (attempt + 1))
                else:
                    step.status = TaskStatus.FAILED
                    step.error = str(e)
                    step.completed_at = datetime.now(timezone.utc)
                    raise TaskExecutionException(f"Step {step.step_id} failed after {max_retries} retries: {e}", step.step_id) from e
        
        # Should never reach here
        raise TaskExecutionException(f"Step {step.step_id} execution failed", step.step_id)
    
    def _find_executor(self, action_type: str) -> Optional[TaskExecutor]:
        """Find appropriate executor for action type"""
        for executor in self.executors.values():
            if executor.can_handle(action_type):
                return executor
        return None
    
    def _build_dependency_graph(self, steps: List[TaskStep]) -> Dict[str, List[str]]:
        """Build dependency graph for parallel execution"""
        graph = {}
        for step in steps:
            graph[step.step_id] = step.dependencies[:]
        return graph
    
    def _evaluate_step_condition(self, step: TaskStep, context: Dict[str, Any], results: Dict[str, Any]) -> bool:
        """Evaluate whether a step should be executed based on conditions"""
        
        condition = step.parameters.get("condition")
        if not condition:
            return True  # No condition means always execute
        
        # Simple condition evaluation (can be extended)
        if isinstance(condition, dict):
            condition_type = condition.get("type", "always")
            
            if condition_type == "always":
                return True
            elif condition_type == "never":
                return False
            elif condition_type == "if_previous_success":
                prev_step_id = condition.get("step_id")
                if prev_step_id and prev_step_id in results:
                    # Check if previous step was successful
                    return results[prev_step_id] is not None
            elif condition_type == "if_context_value":
                key = condition.get("key")
                expected_value = condition.get("value")
                return context.get(key) == expected_value
        
        return True  # Default to execute if condition is unclear
    
    def _insert_by_priority(self, task: Task) -> None:
        """Insert task in queue based on priority"""
        
        # Find insertion point based on priority
        insert_index = 0
        for i, queued_task in enumerate(self.task_queue):
            if task.priority.value > queued_task.priority.value:
                insert_index = i
                break
            insert_index = i + 1
        
        self.task_queue.insert(insert_index, task)
    
    async def _task_processing_loop(self) -> None:
        """Background loop for processing queued tasks"""
        
        while not self._shutdown_event.is_set():
            try:
                # Check if we can process more tasks
                if self._running_tasks >= self._max_concurrent or not self.task_queue:
                    await asyncio.sleep(1)
                    continue
                
                # Get next task from queue
                task = self.task_queue.pop(0)
                
                # Start task execution
                self._running_tasks += 1
                asyncio.create_task(self._execute_queued_task(task))
                
            except Exception as e:
                self.logger.error("Error in task processing loop", error=str(e))
                await asyncio.sleep(5)  # Wait before continuing
    
    async def _execute_queued_task(self, task: Task) -> None:
        """Execute a queued task"""
        
        try:
            self.active_tasks[task.task_id] = task
            result = await self._execute_task(task)
            
            # Mark as completed
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(timezone.utc)
            task.result = result
            task.progress = 1.0
            
            # Move to completed tasks
            self.completed_tasks[task.task_id] = task
            del self.active_tasks[task.task_id]
            
            # Update metrics
            self._update_metrics(task)
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now(timezone.utc)
            
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            self.completed_tasks[task.task_id] = task
            
            self.logger.error("Queued task execution failed", task_id=task.task_id, error=str(e))
        
        finally:
            self._running_tasks = max(0, self._running_tasks - 1)
    
    async def _cancel_all_tasks(self) -> None:
        """Cancel all running and queued tasks"""
        
        # Cancel active tasks
        for task_id in list(self.active_tasks.keys()):
            await self.cancel_task(task_id)
        
        # Cancel queued tasks
        while self.task_queue:
            task = self.task_queue.pop()
            task.status = TaskStatus.CANCELLED
            self.completed_tasks[task.task_id] = task
    
    async def _save_task_history(self) -> None:
        """Save task execution history"""
        try:
            if self.memory:
                # Save recent completed tasks
                recent_tasks = list(self.completed_tasks.values())[-50:]  # Last 50 tasks
                task_history = [task.to_dict() for task in recent_tasks]
                
                await self.memory.store("task_execution_history", {
                    "tasks": task_history,
                    "metrics": self.metrics,
                    "saved_at": datetime.now(timezone.utc).isoformat()
                })
                
                self.logger.debug("Task history saved")
        
        except Exception as e:
            self.logger.warning("Failed to save task history", error=str(e))
    
    def _update_metrics(self, task: Task) -> None:
        """Update performance metrics"""
        
        self.metrics["total_tasks"] += 1
        
        if task.status == TaskStatus.COMPLETED:
            self.metrics["completed_tasks"] += 1
            
            # Update execution time
            if task.started_at and task.completed_at:
                execution_time = (task.completed_at - task.started_at).total_seconds()
                
                # Update average execution time
                total_completed = self.metrics["completed_tasks"]
                if total_completed > 1:
                    old_avg = self.metrics["average_execution_time"]
                    new_avg = (old_avg * (total_completed - 1) + execution_time) / total_completed
                    self.metrics["average_execution_time"] = new_avg
                else:
                    self.metrics["average_execution_time"] = execution_time
        
        elif task.status == TaskStatus.FAILED:
            self.metrics["failed_tasks"] += 1
        
        # Update task type metrics
        task_type = task.task_type
        if task_type not in self.metrics["tasks_per_type"]:
            self.metrics["tasks_per_type"][task_type] = {"total": 0, "completed": 0, "failed": 0}
        
        self.metrics["tasks_per_type"][task_type]["total"] += 1
        if task.status == TaskStatus.COMPLETED:
            self.metrics["tasks_per_type"][task_type]["completed"] += 1
        elif task.status == TaskStatus.FAILED:
            self.metrics["tasks_per_type"][task_type]["failed"] += 1
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on task coordinator"""
        
        health = {
            "status": "healthy" if self._initialized else "unhealthy",
            "initialized": self._initialized,
            "active_tasks": len(self.active_tasks),
            "queued_tasks": len(self.task_queue),
            "running_tasks": self._running_tasks,
            "max_concurrent": self._max_concurrent,
            "completed_tasks": len(self.completed_tasks),
            "metrics": self.metrics.copy(),
            "executors": list(self.executors.keys())
        }
        
        # Check executor health
        executor_health = {}
        for name, executor in self.executors.items():
            executor_health[name] = {
                "supported_actions": executor.get_supported_actions(),
                "active": True
            }
        
        health["executor_status"] = executor_health
        
        return health
    
    def get_task_metrics(self) -> Dict[str, Any]:
        """Get detailed task execution metrics"""
        
        success_rate = 0.0
        if self.metrics["total_tasks"] > 0:
            success_rate = self.metrics["completed_tasks"] / self.metrics["total_tasks"]
        
        return {
            "total_tasks": self.metrics["total_tasks"],
            "completed_tasks": self.metrics["completed_tasks"],
            "failed_tasks": self.metrics["failed_tasks"],
            "success_rate": success_rate,
            "average_execution_time": self.metrics["average_execution_time"],
            "tasks_per_type": self.metrics["tasks_per_type"],
            "current_load": {
                "active_tasks": len(self.active_tasks),
                "queued_tasks": len(self.task_queue),
                "running_tasks": self._running_tasks,
                "capacity_used": self._running_tasks / self._max_concurrent
            }
        }