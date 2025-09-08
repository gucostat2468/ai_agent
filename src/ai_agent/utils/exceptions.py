"""
Custom Exceptions for AI Agent
Comprehensive exception hierarchy for better error handling.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone


class AIAgentException(Exception):
    """Base exception for all AI Agent errors"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "AGENT_ERROR",
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.cause = cause
        self.timestamp = datetime.now(timezone.utc)
        
        # Add cause to details if provided
        if cause:
            self.details["cause"] = {
                "type": type(cause).__name__,
                "message": str(cause)
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary"""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }
    
    def __str__(self) -> str:
        """String representation"""
        return f"[{self.error_code}] {self.message}"


# Core Agent Exceptions

class AgentException(AIAgentException):
    """General agent-related exceptions"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="AGENT_ERROR", **kwargs)


class AgentNotInitializedException(AgentException):
    """Agent is not properly initialized"""
    
    def __init__(self, message: str = "Agent is not initialized", **kwargs):
        super().__init__(message, error_code="AGENT_NOT_INITIALIZED", **kwargs)


class AgentAlreadyRunningException(AgentException):
    """Agent is already running"""
    
    def __init__(self, message: str = "Agent is already running", **kwargs):
        super().__init__(message, error_code="AGENT_ALREADY_RUNNING", **kwargs)


class AgentShutdownException(AgentException):
    """Agent shutdown related errors"""
    
    def __init__(self, message: str = "Agent shutdown error", **kwargs):
        super().__init__(message, error_code="AGENT_SHUTDOWN_ERROR", **kwargs)


# Brain Component Exceptions

class BrainException(AIAgentException):
    """Brain component exceptions"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="BRAIN_ERROR", **kwargs)


class ReasoningException(BrainException):
    """Reasoning process exceptions"""
    
    def __init__(self, message: str, reasoning_type: str = "unknown", **kwargs):
        details = kwargs.get("details", {})
        details["reasoning_type"] = reasoning_type
        kwargs["details"] = details
        super().__init__(message, error_code="REASONING_ERROR", **kwargs)


class PlanningException(BrainException):
    """Planning process exceptions"""
    
    def __init__(self, message: str, task_id: str = None, **kwargs):
        details = kwargs.get("details", {})
        if task_id:
            details["task_id"] = task_id
        kwargs["details"] = details
        super().__init__(message, error_code="PLANNING_ERROR", **kwargs)


class ReflectionException(BrainException):
    """Reflection process exceptions"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="REFLECTION_ERROR", **kwargs)


# Memory System Exceptions

class MemoryException(AIAgentException):
    """Memory system exceptions"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="MEMORY_ERROR", **kwargs)


class MemoryStorageException(MemoryException):
    """Memory storage exceptions"""
    
    def __init__(self, message: str, memory_id: str = None, **kwargs):
        details = kwargs.get("details", {})
        if memory_id:
            details["memory_id"] = memory_id
        kwargs["details"] = details
        super().__init__(message, error_code="MEMORY_STORAGE_ERROR", **kwargs)


class MemoryRetrievalException(MemoryException):
    """Memory retrieval exceptions"""
    
    def __init__(self, message: str, memory_id: str = None, query: str = None, **kwargs):
        details = kwargs.get("details", {})
        if memory_id:
            details["memory_id"] = memory_id
        if query:
            details["query"] = query
        kwargs["details"] = details
        super().__init__(message, error_code="MEMORY_RETRIEVAL_ERROR", **kwargs)


class VectorStoreException(MemoryException):
    """Vector store exceptions"""
    
    def __init__(self, message: str, operation: str = None, **kwargs):
        details = kwargs.get("details", {})
        if operation:
            details["operation"] = operation
        kwargs["details"] = details
        super().__init__(message, error_code="VECTOR_STORE_ERROR", **kwargs)


# Decision Engine Exceptions

class DecisionException(AIAgentException):
    """Decision engine exceptions"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="DECISION_ERROR", **kwargs)


class DecisionTimeoutException(DecisionException):
    """Decision process timeout"""
    
    def __init__(self, message: str = "Decision process timed out", timeout: float = None, **kwargs):
        details = kwargs.get("details", {})
        if timeout:
            details["timeout"] = timeout
        kwargs["details"] = details
        super().__init__(message, error_code="DECISION_TIMEOUT", **kwargs)


class InsufficientInformationException(DecisionException):
    """Insufficient information for decision making"""
    
    def __init__(self, message: str, missing_info: List[str] = None, **kwargs):
        details = kwargs.get("details", {})
        if missing_info:
            details["missing_info"] = missing_info
        kwargs["details"] = details
        super().__init__(message, error_code="INSUFFICIENT_INFORMATION", **kwargs)


# Task Execution Exceptions

class TaskException(AIAgentException):
    """Task execution exceptions"""
    
    def __init__(self, message: str, task_id: str = None, **kwargs):
        details = kwargs.get("details", {})
        if task_id:
            details["task_id"] = task_id
        kwargs["details"] = details
        super().__init__(message, error_code="TASK_ERROR", **kwargs)


class TaskExecutionException(TaskException):
    """Task execution failures"""
    
    def __init__(self, message: str, task_id: str = None, step: str = None, **kwargs):
        details = kwargs.get("details", {})
        if step:
            details["failed_step"] = step
        super().__init__(message, task_id, error_code="TASK_EXECUTION_ERROR", **kwargs)


class TaskTimeoutException(TaskException):
    """Task execution timeout"""
    
    def __init__(self, message: str, task_id: str = None, timeout: float = None, **kwargs):
        details = kwargs.get("details", {})
        if timeout:
            details["timeout"] = timeout
        super().__init__(message, task_id, error_code="TASK_TIMEOUT", **kwargs)


class TaskValidationException(TaskException):
    """Task validation failures"""
    
    def __init__(self, message: str, task_id: str = None, validation_errors: List[str] = None, **kwargs):
        details = kwargs.get("details", {})
        if validation_errors:
            details["validation_errors"] = validation_errors
        super().__init__(message, task_id, error_code="TASK_VALIDATION_ERROR", **kwargs)


# Tool Exceptions

class ToolException(AIAgentException):
    """Tool-related exceptions"""
    
    def __init__(self, message: str, tool_name: str = None, **kwargs):
        details = kwargs.get("details", {})
        if tool_name:
            details["tool_name"] = tool_name
        kwargs["details"] = details
        super().__init__(message, error_code="TOOL_ERROR", **kwargs)


class ToolExecutionException(ToolException):
    """Tool execution failures"""
    
    def __init__(self, message: str, tool_name: str = None, parameters: Dict[str, Any] = None, **kwargs):
        details = kwargs.get("details", {})
        if parameters:
            details["parameters"] = parameters
        super().__init__(message, tool_name, error_code="TOOL_EXECUTION_ERROR", **kwargs)


class ToolValidationException(ToolException):
    """Tool parameter validation failures"""
    
    def __init__(self, message: str, tool_name: str = None, invalid_params: List[str] = None, **kwargs):
        details = kwargs.get("details", {})
        if invalid_params:
            details["invalid_params"] = invalid_params
        super().__init__(message, tool_name, error_code="TOOL_VALIDATION_ERROR", **kwargs)


class ToolNotFoundException(ToolException):
    """Tool not found"""
    
    def __init__(self, message: str, tool_name: str = None, **kwargs):
        super().__init__(message, tool_name, error_code="TOOL_NOT_FOUND", **kwargs)


# Plugin Exceptions

class PluginException(AIAgentException):
    """Plugin-related exceptions"""
    
    def __init__(self, message: str, plugin_name: str = None, **kwargs):
        details = kwargs.get("details", {})
        if plugin_name:
            details["plugin_name"] = plugin_name
        kwargs["details"] = details
        super().__init__(message, error_code="PLUGIN_ERROR", **kwargs)


class PluginLoadException(PluginException):
    """Plugin loading failures"""
    
    def __init__(self, message: str, plugin_name: str = None, **kwargs):
        super().__init__(message, plugin_name, error_code="PLUGIN_LOAD_ERROR", **kwargs)


class PluginValidationException(PluginException):
    """Plugin validation failures"""
    
    def __init__(self, message: str, plugin_name: str = None, validation_errors: List[str] = None, **kwargs):
        details = kwargs.get("details", {})
        if validation_errors:
            details["validation_errors"] = validation_errors
        super().__init__(message, plugin_name, error_code="PLUGIN_VALIDATION_ERROR", **kwargs)


class PluginSecurityException(PluginException):
    """Plugin security violations"""
    
    def __init__(self, message: str, plugin_name: str = None, violation_type: str = None, **kwargs):
        details = kwargs.get("details", {})
        if violation_type:
            details["violation_type"] = violation_type
        super().__init__(message, plugin_name, error_code="PLUGIN_SECURITY_ERROR", **kwargs)


# Communication Exceptions

class CommunicationException(AIAgentException):
    """Communication-related exceptions"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="COMMUNICATION_ERROR", **kwargs)


class MessageProcessingException(CommunicationException):
    """Message processing failures"""
    
    def __init__(self, message: str, message_id: str = None, **kwargs):
        details = kwargs.get("details", {})
        if message_id:
            details["message_id"] = message_id
        kwargs["details"] = details
        super().__init__(message, error_code="MESSAGE_PROCESSING_ERROR", **kwargs)


class ChannelException(CommunicationException):
    """Communication channel exceptions"""
    
    def __init__(self, message: str, channel_type: str = None, **kwargs):
        details = kwargs.get("details", {})
        if channel_type:
            details["channel_type"] = channel_type
        kwargs["details"] = details
        super().__init__(message, error_code="CHANNEL_ERROR", **kwargs)


# LLM and External Service Exceptions

class LLMException(AIAgentException):
    """Large Language Model exceptions"""
    
    def __init__(self, message: str, provider: str = None, model: str = None, **kwargs):
        details = kwargs.get("details", {})
        if provider:
            details["provider"] = provider
        if model:
            details["model"] = model
        kwargs["details"] = details
        super().__init__(message, error_code="LLM_ERROR", **kwargs)


class LLMTimeoutException(LLMException):
    """LLM request timeout"""
    
    def __init__(self, message: str = "LLM request timed out", timeout: float = None, **kwargs):
        details = kwargs.get("details", {})
        if timeout:
            details["timeout"] = timeout
        kwargs["details"] = details
        super().__init__(message, error_code="LLM_TIMEOUT", **kwargs)


class LLMRateLimitException(LLMException):
    """LLM rate limit exceeded"""
    
    def __init__(self, message: str = "LLM rate limit exceeded", retry_after: int = None, **kwargs):
        details = kwargs.get("details", {})
        if retry_after:
            details["retry_after"] = retry_after
        kwargs["details"] = details
        super().__init__(message, error_code="LLM_RATE_LIMIT", **kwargs)


class LLMQuotaException(LLMException):
    """LLM quota exceeded"""
    
    def __init__(self, message: str = "LLM quota exceeded", **kwargs):
        super().__init__(message, error_code="LLM_QUOTA_EXCEEDED", **kwargs)


class ExternalServiceException(AIAgentException):
    """External service exceptions"""
    
    def __init__(self, message: str, service_name: str = None, **kwargs):
        details = kwargs.get("details", {})
        if service_name:
            details["service_name"] = service_name
        kwargs["details"] = details
        super().__init__(message, error_code="EXTERNAL_SERVICE_ERROR", **kwargs)


# Configuration and Validation Exceptions

class ConfigurationException(AIAgentException):
    """Configuration-related exceptions"""
    
    def __init__(self, message: str, config_key: str = None, **kwargs):
        details = kwargs.get("details", {})
        if config_key:
            details["config_key"] = config_key
        kwargs["details"] = details
        super().__init__(message, error_code="CONFIGURATION_ERROR", **kwargs)


class ValidationException(AIAgentException):
    """Data validation exceptions"""
    
    def __init__(self, message: str, field: str = None, value: Any = None, **kwargs):
        details = kwargs.get("details", {})
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        kwargs["details"] = details
        super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)


# Security Exceptions

class SecurityException(AIAgentException):
    """Security-related exceptions"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="SECURITY_ERROR", **kwargs)


class AuthenticationException(SecurityException):
    """Authentication failures"""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(message, error_code="AUTHENTICATION_FAILED", **kwargs)


class AuthorizationException(SecurityException):
    """Authorization failures"""
    
    def __init__(self, message: str = "Authorization denied", user_id: str = None, resource: str = None, action: str = None, **kwargs):
        details = kwargs.get("details", {})
        if user_id:
            details["user_id"] = user_id
        if resource:
            details["resource"] = resource
        if action:
            details["action"] = action
        kwargs["details"] = details
        super().__init__(message, error_code="AUTHORIZATION_DENIED", **kwargs)


class InputValidationException(SecurityException):
    """Input validation security failures"""
    
    def __init__(self, message: str, validation_type: str = None, **kwargs):
        details = kwargs.get("details", {})
        if validation_type:
            details["validation_type"] = validation_type
        kwargs["details"] = details
        super().__init__(message, error_code="INPUT_VALIDATION_ERROR", **kwargs)


class RateLimitException(SecurityException):
    """Rate limiting exceptions"""
    
    def __init__(self, message: str = "Rate limit exceeded", limit: int = None, window: int = None, **kwargs):
        details = kwargs.get("details", {})
        if limit:
            details["limit"] = limit
        if window:
            details["window"] = window
        kwargs["details"] = details
        super().__init__(message, error_code="RATE_LIMIT_EXCEEDED", **kwargs)


# Database and Storage Exceptions

class DatabaseException(AIAgentException):
    """Database-related exceptions"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="DATABASE_ERROR", **kwargs)


class ConnectionException(DatabaseException):
    """Database connection exceptions"""
    
    def __init__(self, message: str = "Database connection failed", **kwargs):
        super().__init__(message, error_code="DATABASE_CONNECTION_ERROR", **kwargs)


class QueryException(DatabaseException):
    """Database query exceptions"""
    
    def __init__(self, message: str, query: str = None, **kwargs):
        details = kwargs.get("details", {})
        if query:
            details["query"] = query[:200]  # Truncate long queries
        kwargs["details"] = details
        super().__init__(message, error_code="DATABASE_QUERY_ERROR", **kwargs)


class TransactionException(DatabaseException):
    """Database transaction exceptions"""
    
    def __init__(self, message: str = "Transaction failed", **kwargs):
        super().__init__(message, error_code="DATABASE_TRANSACTION_ERROR", **kwargs)


class StorageException(AIAgentException):
    """Storage system exceptions"""
    
    def __init__(self, message: str, storage_type: str = None, **kwargs):
        details = kwargs.get("details", {})
        if storage_type:
            details["storage_type"] = storage_type
        kwargs["details"] = details
        super().__init__(message, error_code="STORAGE_ERROR", **kwargs)


# Network and API Exceptions

class NetworkException(AIAgentException):
    """Network-related exceptions"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="NETWORK_ERROR", **kwargs)


class APIException(AIAgentException):
    """API-related exceptions"""
    
    def __init__(self, message: str, api_name: str = None, status_code: int = None, **kwargs):
        details = kwargs.get("details", {})
        if api_name:
            details["api_name"] = api_name
        if status_code:
            details["status_code"] = status_code
        kwargs["details"] = details
        super().__init__(message, error_code="API_ERROR", **kwargs)


class HTTPException(APIException):
    """HTTP-specific exceptions"""
    
    def __init__(self, message: str, status_code: int, url: str = None, **kwargs):
        details = kwargs.get("details", {})
        details["status_code"] = status_code
        if url:
            details["url"] = url
        kwargs["details"] = details
        super().__init__(message, error_code="HTTP_ERROR", **kwargs)


# Resource and System Exceptions

class ResourceException(AIAgentException):
    """Resource-related exceptions"""
    
    def __init__(self, message: str, resource_type: str = None, **kwargs):
        details = kwargs.get("details", {})
        if resource_type:
            details["resource_type"] = resource_type
        kwargs["details"] = details
        super().__init__(message, error_code="RESOURCE_ERROR", **kwargs)


class ResourceExhaustedException(ResourceException):
    """Resource exhaustion exceptions"""
    
    def __init__(self, message: str, resource_type: str = None, limit: Any = None, current: Any = None, **kwargs):
        details = kwargs.get("details", {})
        if limit is not None:
            details["limit"] = str(limit)
        if current is not None:
            details["current"] = str(current)
        super().__init__(message, resource_type, error_code="RESOURCE_EXHAUSTED", **kwargs)


class MemoryException(ResourceException):
    """Memory-related exceptions"""
    
    def __init__(self, message: str = "Memory error", **kwargs):
        super().__init__(message, "memory", error_code="MEMORY_ERROR", **kwargs)


class DiskSpaceException(ResourceException):
    """Disk space exceptions"""
    
    def __init__(self, message: str = "Insufficient disk space", **kwargs):
        super().__init__(message, "disk_space", error_code="DISK_SPACE_ERROR", **kwargs)


# Business Logic Exceptions

class BusinessLogicException(AIAgentException):
    """Business logic exceptions"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="BUSINESS_LOGIC_ERROR", **kwargs)


class InvalidStateException(BusinessLogicException):
    """Invalid state exceptions"""
    
    def __init__(self, message: str, current_state: str = None, expected_state: str = None, **kwargs):
        details = kwargs.get("details", {})
        if current_state:
            details["current_state"] = current_state
        if expected_state:
            details["expected_state"] = expected_state
        kwargs["details"] = details
        super().__init__(message, error_code="INVALID_STATE", **kwargs)


class ConcurrencyException(BusinessLogicException):
    """Concurrency-related exceptions"""
    
    def __init__(self, message: str = "Concurrency error", **kwargs):
        super().__init__(message, error_code="CONCURRENCY_ERROR", **kwargs)


# Utility Functions

def create_error_response(exception: AIAgentException) -> Dict[str, Any]:
    """
    Create a standardized error response from an exception.
    
    Args:
        exception: The AI Agent exception
        
    Returns:
        Standardized error response dictionary
    """
    return {
        "error": True,
        "error_type": exception.__class__.__name__,
        "error_code": exception.error_code,
        "message": exception.message,
        "details": exception.details,
        "timestamp": exception.timestamp.isoformat()
    }


def handle_exception(func):
    """
    Decorator to handle exceptions and convert them to AI Agent exceptions.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AIAgentException:
            # Re-raise AI Agent exceptions
            raise
        except Exception as e:
            # Convert other exceptions to AI Agent exceptions
            raise AIAgentException(
                message=f"Unexpected error in {func.__name__}: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                cause=e
            ) from e
    
    return wrapper


async def ahandle_exception(func):
    """
    Async version of exception handling decorator.
    
    Args:
        func: Async function to decorate
        
    Returns:
        Decorated async function
    """
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except AIAgentException:
            # Re-raise AI Agent exceptions
            raise
        except Exception as e:
            # Convert other exceptions to AI Agent exceptions
            raise AIAgentException(
                message=f"Unexpected error in {func.__name__}: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                cause=e
            ) from e
    
    return wrapper


def get_exception_hierarchy() -> Dict[str, List[str]]:
    """
    Get the exception hierarchy for documentation purposes.
    
    Returns:
        Dictionary mapping base exceptions to their subclasses
    """
    import inspect
    
    hierarchy = {}
    
    # Get all exception classes defined in this module
    current_module = inspect.getmodule(inspect.currentframe())
    exception_classes = []
    
    for name, obj in inspect.getmembers(current_module):
        if (inspect.isclass(obj) and 
            issubclass(obj, Exception) and 
            obj != Exception and 
            obj != AIAgentException):
            exception_classes.append((name, obj))
    
    # Build hierarchy
    for name, cls in exception_classes:
        if hasattr(cls, '__bases__'):
            for base in cls.__bases__:
                if base != Exception:
                    base_name = base.__name__
                    if base_name not in hierarchy:
                        hierarchy[base_name] = []
                    hierarchy[base_name].append(name)
    
    return hierarchy


def format_exception_for_user(exception: Exception) -> str:
    """
    Format an exception for user-friendly display.
    
    Args:
        exception: The exception to format
        
    Returns:
        User-friendly error message
    """
    if isinstance(exception, AIAgentException):
        # Use the custom message for AI Agent exceptions
        return exception.message
    elif isinstance(exception, ConnectionError):
        return "Unable to connect to the service. Please check your connection and try again."
    elif isinstance(exception, TimeoutError):
        return "The operation timed out. Please try again."
    elif isinstance(exception, PermissionError):
        return "Permission denied. Please check your access rights."
    elif isinstance(exception, FileNotFoundError):
        return "The requested file or resource was not found."
    elif isinstance(exception, ValueError):
        return "Invalid input provided. Please check your request and try again."
    else:
        # Generic message for unknown exceptions
        return "An unexpected error occurred. Please try again or contact support if the problem persists."


def log_exception(exception: Exception, logger, context: Dict[str, Any] = None):
    """
    Log an exception with appropriate level and context.
    
    Args:
        exception: The exception to log
        logger: Logger instance
        context: Additional context for logging
    """
    context = context or {}
    
    if isinstance(exception, AIAgentException):
        # Log AI Agent exceptions with full details
        logger.error(
            "AI Agent Exception",
            error_type=exception.__class__.__name__,
            error_code=exception.error_code,
            message=exception.message,
            details=exception.details,
            **context
        )
    else:
        # Log other exceptions
        logger.error(
            "Unexpected Exception",
            error_type=exception.__class__.__name__,
            message=str(exception),
            **context,
            exc_info=True
        )


class ExceptionContext:
    """Context manager for exception handling with automatic logging"""
    
    def __init__(self, logger, operation: str, **context):
        self.logger = logger
        self.operation = operation
        self.context = context
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self.context["operation"] = self.operation
            log_exception(exc_value, self.logger, self.context)
        return False  # Don't suppress the exception


# Exception Registry for error code mapping

_EXCEPTION_REGISTRY: Dict[str, type] = {}


def register_exception(error_code: str, exception_class: type):
    """
    Register an exception class with an error code.
    
    Args:
        error_code: Error code string
        exception_class: Exception class
    """
    _EXCEPTION_REGISTRY[error_code] = exception_class


def get_exception_by_code(error_code: str) -> Optional[type]:
    """
    Get exception class by error code.
    
    Args:
        error_code: Error code string
        
    Returns:
        Exception class or None
    """
    return _EXCEPTION_REGISTRY.get(error_code)


def create_exception_from_code(
    error_code: str, 
    message: str, 
    **kwargs
) -> AIAgentException:
    """
    Create exception instance from error code.
    
    Args:
        error_code: Error code string
        message: Error message
        **kwargs: Additional arguments
        
    Returns:
        Exception instance
    """
    exception_class = get_exception_by_code(error_code)
    if exception_class:
        return exception_class(message, **kwargs)
    else:
        return AIAgentException(message, error_code=error_code, **kwargs)


# Auto-register common exceptions
def _auto_register_exceptions():
    """Automatically register common exceptions with their error codes"""
    import inspect
    
    current_module = inspect.getmodule(inspect.currentframe())
    
    for name, obj in inspect.getmembers(current_module):
        if (inspect.isclass(obj) and 
            issubclass(obj, AIAgentException) and 
            obj != AIAgentException):
            
            # Create instance to get default error code
            try:
                instance = obj("test")
                register_exception(instance.error_code, obj)
            except TypeError:
                # Skip classes that require additional parameters
                pass


# Register exceptions on module import
_auto_register_exceptions()