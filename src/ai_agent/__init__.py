# src/ai_agent/__init__.py
"""
AI Agent - Robust and Scalable Intelligent Assistant Framework

A professional-grade AI agent framework with plugin system, advanced reasoning,
and enterprise-ready features.
"""

__version__ = "1.0.0"
__author__ = "AI Agent Team"
__email__ = "team@aiagent.com"
__description__ = "Robust and scalable AI Agent framework"

from .core.agent import AIAgent, AgentState, ProcessingContext
from .config.settings import AgentSettings, load_settings
from .main import AIAgentApplication

__all__ = [
    "AIAgent",
    "AgentState", 
    "ProcessingContext",
    "AgentSettings",
    "load_settings",
    "AIAgentApplication",
    "__version__"
]

# ================================================================
# src/ai_agent/core/__init__.py
"""
Core components of the AI Agent system.
"""

from .agent import AIAgent, BaseAgent, AgentState, ProcessingContext
from .brain import Brain
from .memory import MemorySystem
from .decision_engine import DecisionEngine, Decision, DecisionType
from .task_coordinator import TaskCoordinator, Task, TaskStep

__all__ = [
    "AIAgent",
    "BaseAgent", 
    "AgentState",
    "ProcessingContext",
    "Brain",
    "MemorySystem",
    "DecisionEngine",
    "Decision",
    "DecisionType", 
    "TaskCoordinator",
    "Task",
    "TaskStep"
]

# ================================================================
# src/ai_agent/interfaces/__init__.py
"""
Interfaces and abstract base classes for AI Agent components.
"""

from .base import (
    LLMInterface,
    ToolInterface, 
    DataInterface,
    VectorStoreInterface,
    MemoryInterface,
    PluginInterface,
    CommunicationInterface,
    SecurityInterface,
    MonitoringInterface,
    ToolResult,
    ToolCapability,
    Message,
    MemoryEntry
)

__all__ = [
    "LLMInterface",
    "ToolInterface",
    "DataInterface", 
    "VectorStoreInterface",
    "MemoryInterface",
    "PluginInterface",
    "CommunicationInterface",
    "SecurityInterface",
    "MonitoringInterface",
    "ToolResult",
    "ToolCapability",
    "Message", 
    "MemoryEntry"
]

# ================================================================
# src/ai_agent/plugins/__init__.py
"""
Plugin system for AI Agent extensibility.
"""

from .base_plugin import BasePlugin, TaskPlugin, CommunicationPlugin, DataPlugin
from .registry import PluginRegistry, PluginMetadata

__all__ = [
    "BasePlugin",
    "TaskPlugin", 
    "CommunicationPlugin",
    "DataPlugin",
    "PluginRegistry",
    "PluginMetadata"
]

# ================================================================
# src/ai_agent/tools/__init__.py
"""
Tool system for AI Agent capabilities.
"""

from .tool_registry import ToolRegistry, ToolMetadata, get_tool_registry
from .built_in import *

__all__ = [
    "ToolRegistry",
    "ToolMetadata", 
    "get_tool_registry"
]

# ================================================================
# src/ai_agent/memory/__init__.py
"""
Memory system components.
"""

from .short_term import ShortTermMemory
from .long_term import LongTermMemory  
from .vector_store import VectorStore
from .episodic import EpisodicMemory

__all__ = [
    "ShortTermMemory",
    "LongTermMemory",
    "VectorStore", 
    "EpisodicMemory"
]

# ================================================================
# src/ai_agent/reasoning/__init__.py
"""
Reasoning engines and strategies.
"""

from .chain_of_thought import ChainOfThoughtReasoner
from .tree_of_thoughts import TreeOfThoughtsReasoner
from .reflection import ReflectionEngine
from .planning import PlanningEngine

__all__ = [
    "ChainOfThoughtReasoner",
    "TreeOfThoughtsReasoner", 
    "ReflectionEngine",
    "PlanningEngine"
]

# ================================================================
# src/ai_agent/communication/__init__.py
"""
Communication components.
"""

from .message_handler import MessageHandler
from .context_manager import ContextManager
from .response_formatter import ResponseFormatter

__all__ = [
    "MessageHandler",
    "ContextManager",
    "ResponseFormatter" 
]

# ================================================================
# src/ai_agent/monitoring/__init__.py
"""
Monitoring, logging, and metrics components.
"""

from .logger import (
    StructuredLogger, 
    LoggingContext,
    setup_logging,
    get_logger,
    get_audit_logger,
    get_performance_logger
)
from .metrics import MetricsCollector, setup_metrics

__all__ = [
    "StructuredLogger",
    "LoggingContext", 
    "setup_logging",
    "get_logger",
    "get_audit_logger",
    "get_performance_logger",
    "MetricsCollector",
    "setup_metrics"
]

# ================================================================
# src/ai_agent/security/__init__.py
"""
Security components for authentication, authorization, and input validation.
"""

from .authentication import authenticate_request
from .authorization import authorize_request  
from .input_validation import validate_input, sanitize_input

__all__ = [
    "authenticate_request",
    "authorize_request",
    "validate_input", 
    "sanitize_input"
]

# ================================================================
# src/ai_agent/config/__init__.py
"""
Configuration management.
"""

from .settings import (
    AgentSettings,
    AgentConfig, 
    load_settings,
    create_default_config_file,
    get_settings
)

__all__ = [
    "AgentSettings",
    "AgentConfig",
    "load_settings", 
    "create_default_config_file",
    "get_settings"
]

# ================================================================
# src/ai_agent/utils/__init__.py
"""
Utility functions and helpers.
"""

from .exceptions import *
from .decorators import retry, timeout, cached
from .helpers import (
    generate_id,
    timestamp,
    deep_merge,
    safe_json_loads
)

__all__ = [
    "retry",
    "timeout", 
    "cached",
    "generate_id",
    "timestamp",
    "deep_merge",
    "safe_json_loads"
]