"""
Tool Registry - Centralized Tool Management System
Handles tool discovery, registration, validation, and execution coordination.
"""

import asyncio
import importlib
import inspect
from typing import Dict, List, Any, Optional, Callable, Type
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json

from ..interfaces.base import ToolInterface, ToolResult, ToolCapability, BaseComponent
from ..monitoring.logger import StructuredLogger
from ..utils.exceptions import ToolException, ToolNotFoundException, ToolValidationException, ToolExecutionException
from .built_in import *  # Import built-in tools


@dataclass
class ToolMetadata:
    """Metadata for registered tools"""
    name: str
    description: str
    version: str
    author: str
    category: str
    tags: List[str] = field(default_factory=list)
    capabilities: List[ToolCapability] = field(default_factory=list)
    usage_count: int = 0
    success_count: int = 0
    last_used: Optional[datetime] = None
    average_execution_time: float = 0.0
    enabled: bool = True
    tool_class: Type[ToolInterface] = None
    instance: Optional[ToolInterface] = None


class ToolRegistry(BaseComponent):
    """
    Central registry for managing tools in the AI Agent system
    """
    
    def __init__(self):
        super().__init__()
        self.logger = StructuredLogger(__name__)
        
        # Tool storage
        self.tools: Dict[str, ToolInterface] = {}
        self.metadata: Dict[str, ToolMetadata] = {}
        self.categories: Dict[str, List[str]] = {}
        self.tags: Dict[str, List[str]] = {}
        
        # Performance metrics
        self.metrics = {
            "total_tools": 0,
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "most_used_tools": {},
            "categories_distribution": {}
        }
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the tool registry and load built-in tools"""
        try:
            self.logger.info("Initializing Tool Registry")
            
            # Load built-in tools
            await self._load_built_in_tools()
            
            # Discover and load custom tools
            await self._discover_custom_tools()
            
            self._initialized = True
            self.logger.info("Tool Registry initialized successfully", 
                           tools_count=len(self.tools))
            
        except Exception as e:
            self.logger.error("Failed to initialize Tool Registry", error=str(e))
            raise ToolException(f"Tool registry initialization failed: {e}") from e
    
    async def cleanup(self) -> None:
        """Cleanup tool registry resources"""
        try:
            self.logger.info("Cleaning up Tool Registry")
            
            # Cleanup all tool instances
            for tool_name, tool in self.tools.items():
                try:
                    await tool.cleanup()
                except Exception as e:
                    self.logger.warning(f"Error cleaning up tool {tool_name}", error=str(e))
            
            self.tools.clear()
            self.metadata.clear()
            self.categories.clear()
            self.tags.clear()
            
            self._initialized = False
            self.logger.info("Tool Registry cleaned up successfully")
            
        except Exception as e:
            self.logger.error("Error during Tool Registry cleanup", error=str(e))
    
    def register_tool(self, tool: ToolInterface, category: str = "custom", force: bool = False) -> bool:
        """
        Register a tool in the registry
        
        Args:
            tool: Tool instance to register
            category: Tool category
            force: Force registration even if tool exists
            
        Returns:
            Success status
        """
        try:
            capabilities = tool.get_capabilities()
            if not capabilities:
                raise ToolValidationException("Tool must have at least one capability")
            
            primary_capability = capabilities[0]
            tool_name = primary_capability.name
            
            # Check if tool already exists
            if tool_name in self.tools and not force:
                self.logger.warning("Tool already registered", tool_name=tool_name)
                return False
            
            # Validate tool
            validation_errors = self._validate_tool(tool)
            if validation_errors:
                raise ToolValidationException(f"Tool validation failed: {validation_errors}", tool_name)
            
            # Register tool
            self.tools[tool_name] = tool
            
            # Create metadata
            metadata = ToolMetadata(
                name=tool_name,
                description=primary_capability.description,
                version=primary_capability.version,
                author=primary_capability.author,
                category=category,
                tags=primary_capability.tags,
                capabilities=capabilities,
                tool_class=tool.__class__,
                instance=tool
            )
            
            self.metadata[tool_name] = metadata
            
            # Update category index
            if category not in self.categories:
                self.categories[category] = []
            self.categories[category].append(tool_name)
            
            # Update tag index
            for tag in primary_capability.tags:
                if tag not in self.tags:
                    self.tags[tag] = []
                self.tags[tag].append(tool_name)
            
            # Update metrics
            self.metrics["total_tools"] += 1
            self.metrics["categories_distribution"][category] = self.metrics["categories_distribution"].get(category, 0) + 1
            
            self.logger.info("Tool registered successfully", 
                           tool_name=tool_name,
                           category=category,
                           capabilities_count=len(capabilities))
            
            return True
            
        except Exception as e:
            self.logger.error("Tool registration failed", error=str(e))
            if isinstance(e, ToolException):
                raise
            else:
                raise ToolException(f"Tool registration failed: {e}") from e
    
    def unregister_tool(self, tool_name: str) -> bool:
        """
        Unregister a tool from the registry
        
        Args:
            tool_name: Name of tool to unregister
            
        Returns:
            Success status
        """
        try:
            if tool_name not in self.tools:
                self.logger.warning("Tool not found for unregistration", tool_name=tool_name)
                return False
            
            # Get metadata
            metadata = self.metadata.get(tool_name)
            
            # Remove from tools
            tool = self.tools.pop(tool_name)
            
            # Cleanup tool instance
            asyncio.create_task(tool.cleanup())
            
            # Remove metadata
            if tool_name in self.metadata:
                del self.metadata[tool_name]
            
            # Update category index
            if metadata and metadata.category in self.categories:
                self.categories[metadata.category] = [
                    t for t in self.categories[metadata.category] if t != tool_name
                ]
                if not self.categories[metadata.category]:
                    del self.categories[metadata.category]
            
            # Update tag index
            if metadata:
                for tag in metadata.tags:
                    if tag in self.tags:
                        self.tags[tag] = [t for t in self.tags[tag] if t != tool_name]
                        if not self.tags[tag]:
                            del self.tags[tag]