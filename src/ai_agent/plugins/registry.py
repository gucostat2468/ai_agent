"""
Plugin Registry - Plugin Discovery and Management System
Handles plugin loading, validation, and lifecycle management.
"""

import os
import sys
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Set
import asyncio
import json
import yaml
from dataclasses import dataclass, field

from ..interfaces.base import PluginInterface
from ..config.settings import PluginConfig
from ..monitoring.logger import StructuredLogger
from ..utils.exceptions import PluginException, PluginLoadException, PluginValidationException, PluginSecurityException


@dataclass
class PluginMetadata:
    """Plugin metadata information"""
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    min_python_version: str = "3.9"
    api_version: str = "1.0"
    permissions: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    entry_point: str = ""
    file_path: str = ""
    loaded: bool = False
    enabled: bool = True
    load_error: Optional[str] = None


class PluginSandbox:
    """Security sandbox for plugin execution"""
    
    def __init__(self, config: PluginConfig):
        self.config = config
        self.allowed_imports = set(config.allowed_imports)
        self.forbidden_imports = set(config.forbidden_imports)
        self.logger = StructuredLogger(__name__)
    
    def validate_imports(self, plugin_code: str) -> List[str]:
        """Validate plugin imports for security"""
        violations = []
        
        # Parse import statements (simplified)
        lines = plugin_code.split('\n')
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Check import statements
            if line.startswith('import ') or line.startswith('from '):
                # Extract module name
                if line.startswith('import '):
                    module = line.split('import ')[1].split()[0].split('.')[0]
                else:  # from ... import
                    module = line.split('from ')[1].split(' import')[0].split('.')[0]
                
                # Check against forbidden imports
                if module in self.forbidden_imports:
                    violations.append(f"Line {line_num}: Forbidden import '{module}'")
                
                # Check against allowed imports (if whitelist mode)
                if self.allowed_imports and module not in self.allowed_imports:
                    # Allow built-in modules
                    if module not in sys.builtin_module_names:
                        violations.append(f"Line {line_num}: Import '{module}' not in allowed list")
            
            # Check for dangerous function calls
            dangerous_calls = ['eval(', 'exec(', 'compile(', '__import__(', 'subprocess.']
            for call in dangerous_calls:
                if call in line:
                    violations.append(f"Line {line_num}: Dangerous function call '{call}'")
        
        return violations
    
    def check_permissions(self, plugin: PluginInterface, required_permissions: List[str]) -> List[str]:
        """Check if plugin has required permissions"""
        plugin_permissions = getattr(plugin, 'permissions', [])
        missing = [perm for perm in required_permissions if perm not in plugin_permissions]
        return missing


class PluginRegistry:
    """
    Registry for managing plugins - discovery, loading, validation, and lifecycle.
    """
    
    def __init__(self, config: PluginConfig):
        self.config = config
        self.logger = StructuredLogger(__name__)
        self.sandbox = PluginSandbox(config) if config.enable_sandboxing else None
        
        # Registry state
        self.plugins: Dict[str, PluginInterface] = {}
        self.metadata: Dict[str, PluginMetadata] = {}
        self.plugin_directories: List[Path] = []
        self.dependency_graph: Dict[str, Set[str]] = {}
        
        # Statistics
        self.stats = {
            "total_discovered": 0,
            "total_loaded": 0,
            "total_errors": 0,
            "last_scan": None
        }
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the plugin registry"""
        try:
            self.logger.info("Initializing Plugin Registry")
            
            # Setup plugin directories
            self._setup_plugin_directories()
            
            # Discover plugins
            if self.config.auto_discovery:
                await self.discover_plugins()
            
            self._initialized = True
            self.logger.info("Plugin Registry initialized successfully")
            
        except Exception as e:
            self.logger.error("Failed to initialize Plugin Registry", error=str(e))
            raise PluginException(f"Plugin registry initialization failed: {e}") from e
    
    async def cleanup(self) -> None:
        """Cleanup plugin registry resources"""
        try:
            self.logger.info("Cleaning up Plugin Registry")
            
            # Unload all plugins
            for plugin_name in list(self.plugins.keys()):
                await self.unload_plugin(plugin_name)
            
            self._initialized = False
            self.logger.info("Plugin Registry cleaned up successfully")
            
        except Exception as e:
            self.logger.error("Error during Plugin Registry cleanup", error=str(e))
    
    def _setup_plugin_directories(self) -> None:
        """Setup plugin directories from configuration"""
        for directory in self.config.plugin_directories:
            # Expand user home directory
            if directory.startswith('~'):
                directory = os.path.expanduser(directory)
            
            path = Path(directory)
            if path.exists() and path.is_dir():
                self.plugin_directories.append(path)
                self.logger.debug(f"Added plugin directory: {path}")
            else:
                # Create directory if it doesn't exist
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    self.plugin_directories.append(path)
                    self.logger.info(f"Created plugin directory: {path}")
                except