#!/usr/bin/env python3
"""
AI Agent Main Entry Point
Handles application startup, shutdown, and command-line interface.
"""

import asyncio
import signal
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import uvloop

# Set up async event loop policy
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    # uvloop not available, use default policy
    pass

from .config.settings import AgentSettings, load_settings, Environment
from .core.agent import AIAgent
from .interfaces.llm_interface import create_llm_interface
from .monitoring.logger import setup_logging
from .monitoring.metrics import setup_metrics
from .utils.exceptions import AIAgentException, ConfigurationException
from .plugins.registry import PluginRegistry
from .tools.tool_registry import ToolRegistry
from .server import create_server


class AIAgentApplication:
    """Main application class for AI Agent"""
    
    def __init__(self, settings: Optional[AgentSettings] = None):
        self.settings = settings or load_settings()
        self.agent: Optional[AIAgent] = None
        self.server = None
        self.plugin_registry: Optional[PluginRegistry] = None
        self.tool_registry: Optional[ToolRegistry] = None
        self.logger = None
        self._shutdown_event = asyncio.Event()
        self._running = False
    
    async def initialize(self) -> None:
        """Initialize the application and all components"""
        try:
            # Setup logging first
            self.logger = setup_logging(self.settings.monitoring)
            self.logger.info("Starting AI Agent initialization", 
                           environment=self.settings.environment.value,
                           debug=self.settings.debug)
            
            # Setup metrics
            if self.settings.monitoring.enable_metrics:
                setup_metrics(self.settings.monitoring)
            
            # Validate configuration
            config_errors = self.settings.validate_config()
            if config_errors:
                raise ConfigurationException(
                    f"Configuration validation failed: {'; '.join(config_errors)}"
                )
            
            # Initialize LLM interface
            llm_interface = await create_llm_interface(self.settings.llm)
            
            # Initialize plugin registry
            self.plugin_registry = PluginRegistry(self.settings.plugin)
            await self.plugin_registry.initialize()
            
            # Initialize tool registry
            self.tool_registry = ToolRegistry()
            await self.tool_registry.initialize()
            
            # Create and initialize agent
            from .config.settings import AgentConfig
            agent_config = AgentConfig(self.settings)
            self.agent = AIAgent(agent_config, llm_interface)
            
            # Register tools with agent
            for tool_name, tool in self.tool_registry.get_all_tools().items():
                self.agent.register_tool(tool_name, tool)
            
            # Load and register plugins
            plugins = await self.plugin_registry.load_plugins()
            for plugin in plugins:
                self.agent.register_plugin(plugin)
            
            # Initialize agent
            await self.agent.initialize()
            
            self.logger.info("AI Agent initialized successfully")
            
        except Exception as e:
            if self.logger:
                self.logger.error("Failed to initialize AI Agent", error=str(e))
            else:
                print(f"Failed to initialize AI Agent: {e}")
            raise
    
    async def start(self) -> None:
        """Start the AI Agent application"""
        if self._running:
            return
        
        try:
            # Initialize if not already done
            if not self.agent:
                await self.initialize()
            
            # Start agent
            await self.agent.start()
            
            # Start server if enabled
            if self.settings.environment != Environment.DEVELOPMENT or not self.settings.debug:
                self.server = create_server(self.agent, self.settings)
                await self.server.start()
            
            self._running = True
            self.logger.info("AI Agent application started successfully",
                           host=self.settings.host,
                           port=self.settings.port)
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
        except Exception as e:
            self.logger.error("Failed to start AI Agent application", error=str(e))
            await self.cleanup()
            raise
    
    async def run(self) -> None:
        """Run the application (blocking)"""
        await self.start()
        
        try:
            # Wait for shutdown signal
            await self._shutdown_event.wait()
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        finally:
            await self.cleanup()
    
    async def cleanup(self) -> None:
        """Cleanup application resources"""
        if not self._running:
            return
        
        self.logger.info("Shutting down AI Agent application")
        
        try:
            # Stop server
            if self.server:
                await self.server.stop()
                self.server = None
            
            # Stop agent
            if self.agent:
                await self.agent.stop()
                self.agent = None
            
            # Cleanup registries
            if self.plugin_registry:
                await self.plugin_registry.cleanup()
                self.plugin_registry = None
            
            if self.tool_registry:
                await self.tool_registry.cleanup()
                self.tool_registry = None
            
            self._running = False
            self.logger.info("AI Agent application shut down successfully")
            
        except Exception as e:
            self.logger.error("Error during cleanup", error=str(e))
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown")
            self._shutdown_event.set()
        
        # Setup signal handlers
        for sig in [signal.SIGTERM, signal.SIGINT]:
            if sys.platform != "win32":
                signal.signal(sig, signal_handler)
    
    async def health_check(self) -> Dict[str, Any]:
        """Get application health status"""
        health = {
            "status": "healthy" if self._running else "unhealthy",
            "running": self._running,
            "environment": self.settings.environment.value,
            "components": {}
        }
        
        if self.agent:
            health["components"]["agent"] = await self.agent.health_check()
        
        if self.plugin_registry:
            health["components"]["plugins"] = await self.plugin_registry.health_check()
        
        if self.tool_registry:
            health["components"]["tools"] = await self.tool_registry.health_check()
        
        if self.server:
            health["components"]["server"] = await self.server.health_check()
        
        return health


# CLI Functions

def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="AI Agent - Intelligent Assistant Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s run                     # Run with default settings
  %(prog)s run --dev               # Run in development mode
  %(prog)s run --config config.yaml  # Run with custom config
  %(prog)s chat                    # Start interactive chat
  %(prog)s health                  # Check agent health
  %(prog)s plugins list            # List available plugins
        """
    )
    
    # Global options
    parser.add_argument("--version", action="version", version="AI Agent 1.0.0")
    parser.add_argument("--config", "-c", type=str, help="Configuration file path")
    parser.add_argument("--env", type=str, help="Environment (.env file path)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run the AI Agent")
    run_parser.add_argument("--dev", action="store_true", help="Development mode")
    run_parser.add_argument("--host", default="localhost", help="Host to bind to")
    run_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    run_parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat")
    chat_parser.add_argument("--session-id", help="Chat session ID")
    
    # Health command
    subparsers.add_parser("health", help="Check agent health")
    
    # Plugin commands
    plugin_parser = subparsers.add_parser("plugins", help="Plugin management")
    plugin_subparsers = plugin_parser.add_subparsers(dest="plugin_command")
    
    plugin_subparsers.add_parser("list", help="List plugins")
    
    install_parser = plugin_subparsers.add_parser("install", help="Install plugin")
    install_parser.add_argument("plugin", help="Plugin name or path")
    
    remove_parser = plugin_subparsers.add_parser("remove", help="Remove plugin")
    remove_parser.add_argument("plugin", help="Plugin name")
    
    # Tool commands
    tool_parser = subparsers.add_parser("tools", help="Tool management")
    tool_subparsers = tool_parser.add_subparsers(dest="tool_command")
    
    tool_subparsers.add_parser("list", help="List tools")
    
    test_parser = tool_subparsers.add_parser("test", help="Test tool")
    test_parser.add_argument("tool", help="Tool name")
    test_parser.add_argument("--params", help="Tool parameters (JSON)")
    
    # Config commands
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_subparsers = config_parser.add_subparsers(dest="config_command")
    
    config_subparsers.add_parser("show", help="Show current configuration")
    config_subparsers.add_parser("validate", help="Validate configuration")
    
    init_parser = config_subparsers.add_parser("init", help="Initialize configuration")
    init_parser.add_argument("--format", choices=["yaml", "json"], default="yaml", help="Config format")
    init_parser.add_argument("--output", help="Output file path")
    
    return parser


async def run_command(args, settings: AgentSettings) -> int:
    """Run the AI Agent application"""
    try:
        # Override settings with command line arguments
        if args.host:
            settings.host = args.host
        if args.port:
            settings.port = args.port
        if args.workers:
            settings.workers = args.workers
        if args.dev:
            settings.environment = Environment.DEVELOPMENT
            settings.debug = True
            settings.development.debug = True
        
        # Create and run application
        app = AIAgentApplication(settings)
        await app.run()
        return 0
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


async def chat_command(args, settings: AgentSettings) -> int:
    """Start interactive chat session"""
    try:
        app = AIAgentApplication(settings)
        await app.initialize()
        await app.agent.start()
        
        print("AI Agent Chat Interface")
        print("Type 'exit' or 'quit' to end the session")
        print("Type 'help' for available commands")
        print("-" * 50)
        
        session_id = args.session_id or "interactive_session"
        
        async with app.agent.conversation_session(session_id) as session:
            while True:
                try:
                    user_input = input("\nYou: ").strip()
                    
                    if user_input.lower() in ["exit", "quit", "bye"]:
                        print("Goodbye!")
                        break
                    
                    if user_input.lower() == "help":
                        print("""
Available commands:
- exit, quit, bye: End the session
- help: Show this help message
- health: Check agent health
- clear: Clear conversation history
                        """)
                        continue
                    
                    if user_input.lower() == "health":
                        health = await app.agent.health_check()
                        print(f"Agent Health: {health['status']}")
                        continue
                    
                    if user_input.lower() == "clear":
                        print("Conversation history cleared")
                        continue
                    
                    if not user_input:
                        continue
                    
                    # Process message
                    from .core.agent import ProcessingContext
                    context = ProcessingContext(session_id=session)
                    response = await app.agent.process_message(user_input, context)
                    
                    print(f"\nAgent: {response}")
                    
                except KeyboardInterrupt:
                    print("\n\nGoodbye!")
                    break
                except Exception as e:
                    print(f"Error: {e}")
        
        await app.cleanup()
        return 0
        
    except Exception as e:
        print(f"Chat session error: {e}")
        return 1


async def health_command(args, settings: AgentSettings) -> int:
    """Check agent health"""
    try:
        app = AIAgentApplication(settings)
        await app.initialize()
        
        health = await app.health_check()
        
        print("AI Agent Health Status")
        print("=" * 30)
        print(f"Overall Status: {health['status']}")
        print(f"Running: {health['running']}")
        print(f"Environment: {health['environment']}")
        
        if "components" in health:
            print("\nComponent Status:")
            for component, status in health["components"].items():
                if isinstance(status, dict):
                    comp_status = status.get("status", "unknown")
                    print(f"  {component}: {comp_status}")
                else:
                    print(f"  {component}: {status}")
        
        await app.cleanup()
        return 0 if health["status"] == "healthy" else 1
        
    except Exception as e:
        print(f"Health check error: {e}")
        return 1


async def main_async() -> int:
    """Main async function"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Load settings
        settings = load_settings(args.config, args.env)
        
        # Override with command line arguments
        if args.debug:
            settings.debug = True
            settings.development.debug = True
        if args.log_level:
            settings.monitoring.log_level = args.log_level
        
        # Route to appropriate command
        if args.command == "run" or args.command is None:
            return await run_command(args, settings)
        elif args.command == "chat":
            return await chat_command(args, settings)
        elif args.command == "health":
            return await health_command(args, settings)
        elif args.command == "plugins":
            return await handle_plugin_commands(args, settings)
        elif args.command == "tools":
            return await handle_tool_commands(args, settings)
        elif args.command == "config":
            return await handle_config_commands(args, settings)
        else:
            parser.print_help()
            return 1
            
    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


async def handle_plugin_commands(args, settings: AgentSettings) -> int:
    """Handle plugin-related commands"""
    if args.plugin_command == "list":
        try:
            registry = PluginRegistry(settings.plugin)
            await registry.initialize()
            plugins = await registry.discover_plugins()
            
            print("Available Plugins:")
            print("-" * 20)
            for plugin_info in plugins:
                print(f"  {plugin_info['name']} v{plugin_info.get('version', 'unknown')}")
                if plugin_info.get('description'):
                    print(f"    {plugin_info['description']}")
                print()