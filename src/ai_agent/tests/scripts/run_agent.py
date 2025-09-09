#!/usr/bin/env python3
"""
AI Agent Startup Script
Production-ready script for running the AI Agent with proper error handling,
monitoring, and graceful shutdown capabilities.
"""

import os
import sys
import signal
import asyncio
import argparse
from pathlib import Path
from typing import Optional

# Add the src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from ai_agent.main import AIAgentApplication
from ai_agent.config.settings import load_settings, Environment
from ai_agent.monitoring.logger import setup_logging, LoggingContext
from ai_agent.utils.exceptions import AIAgentException


class ProductionRunner:
    """Production runner for AI Agent with enhanced error handling and monitoring"""
    
    def __init__(self, config_file: Optional[str] = None, env_file: Optional[str] = None):
        self.config_file = config_file
        self.env_file = env_file
        self.app: Optional[AIAgentApplication] = None
        self.shutdown_requested = False
        
    async def run(self) -> int:
        """Run the AI Agent application"""
        try:
            # Load configuration
            print("Loading configuration...")
            settings = load_settings(self.config_file, self.env_file)
            
            # Setup logging
            logger = setup_logging(settings.monitoring)
            
            with LoggingContext(operation="ai_agent_startup"):
                logger.info("Starting AI Agent", 
                           environment=settings.environment.value,
                           version="1.0.0")
                
                # Create and initialize application
                self.app = AIAgentApplication(settings)
                
                # Setup signal handlers
                self._setup_signal_handlers()
                
                # Run the application
                await self.app.run()
                
                logger.info("AI Agent stopped gracefully")
                return 0
                
        except KeyboardInterrupt:
            print("\nShutdown requested by user")
            return 0
            
        except AIAgentException as e:
            print(f"AI Agent Error: {e}")
            if hasattr(e, 'details') and e.details:
                print(f"Details: {e.details}")
            return 1
            
        except Exception as e:
            print(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return 1
            
        finally:
            if self.app:
                try:
                    await self.app.cleanup()
                except Exception as e:
                    print(f"Cleanup error: {e}")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            if not self.shutdown_requested:
                print(f"\nReceived signal {signum}, initiating graceful shutdown...")
                self.shutdown_requested = True
                if self.app:
                    asyncio.create_task(self.app.cleanup())
            else:
                print("Force shutdown requested, exiting...")
                sys.exit(1)
        
        # Setup handlers for common signals
        for sig in [signal.SIGTERM, signal.SIGINT]:
            if hasattr(signal, sig.name):
                signal.signal(sig, signal_handler)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="AI Agent Production Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Run with default configuration
  %(prog)s --config config.yaml     # Run with custom config file
  %(prog)s --env production.env     # Run with custom environment file
  %(prog)s --env-check              # Check environment and exit
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Configuration file path (YAML or JSON)"
    )
    
    parser.add_argument(
        "--env",
        type=str,
        help="Environment file path (.env)"
    )
    
    parser.add_argument(
        "--env-check",
        action="store_true",
        help="Check environment configuration and exit"
    )
    
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration and exit"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform dry run (initialize but don't start)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="AI Agent 1.0.0"
    )
    
    return parser


async def check_environment(settings) -> bool:
    """Check if environment is properly configured"""
    print("Checking environment configuration...")
    
    checks = [
        ("Database connection", check_database_connection),
        ("Redis connection", check_redis_connection),
        ("LLM API access", check_llm_access),
        ("Required directories", check_directories),
        ("Required permissions", check_permissions),
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            print(f"  Checking {check_name}...", end=" ")
            result = await check_func(settings)
            if result:
                print("✓ PASS")
            else:
                print("✗ FAIL")
                all_passed = False
        except Exception as e:
            print(f"✗ ERROR: {e}")
            all_passed = False
    
    return all_passed


async def check_database_connection(settings) -> bool:
    """Check database connection"""
    try:
        # Simple connection test
        import asyncpg
        conn = await asyncpg.connect(settings.get_database_url())
        await conn.close()
        return True
    except ImportError:
        print("asyncpg not installed")
        return False
    except Exception:
        return False


async def check_redis_connection(settings) -> bool:
    """Check Redis connection"""
    try:
        import aioredis
        redis = aioredis.from_url(settings.get_redis_url())
        await redis.ping()
        await redis.close()
        return True
    except ImportError:
        print("aioredis not installed")
        return False
    except Exception:
        return False


async def check_llm_access(settings) -> bool:
    """Check LLM API access"""
    try:
        if settings.llm.provider == "openai" and settings.llm.api_key:
            import openai
            client = openai.AsyncOpenAI(api_key=settings.llm.api_key)
            # Simple test call
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            return True
    except ImportError:
        print("OpenAI library not installed")
        return False
    except Exception:
        return False
    
    return True  # Skip check if not OpenAI or no API key


async def check_directories(settings) -> bool:
    """Check required directories exist and are writable"""
    directories = [
        "logs",
        "data",
        "uploads",
        "plugins"
    ]
    
    for directory in directories:
        path = Path(directory)
        try:
            path.mkdir(parents=True, exist_ok=True)
            # Test write access
            test_file = path / ".test_write"
            test_file.write_text("test")
            test_file.unlink()
        except Exception:
            return False
    
    return True


async def check_permissions(settings) -> bool:
    """Check file permissions"""
    # Check if we can create files in working directory
    try:
        test_file = Path(".permission_test")
        test_file.write_text("test")
        test_file.unlink()
        return True
    except Exception:
        return False


def validate_configuration(settings) -> bool:
    """Validate configuration settings"""
    print("Validating configuration...")
    
    from ai_agent.config.settings import AgentConfig
    
    try:
        agent_config = AgentConfig(settings)
        errors = agent_config.validate_config()
        
        if not errors:
            print("✓ Configuration is valid")
            return True
        else:
            print("✗ Configuration validation failed:")
            for error in errors:
                print(f"  - {error}")
            return False
            
    except Exception as e:
        print(f"✗ Configuration validation error: {e}")
        return False


async def main() -> int:
    """Main entry point"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Load settings
        settings = load_settings(args.config, args.env)
        
        # Override debug mode if specified
        if args.debug:
            settings.debug = True
            settings.environment = Environment.DEVELOPMENT
        
        # Environment check
        if args.env_check:
            success = await check_environment(settings)
            return 0 if success else 1
        
        # Configuration validation
        if args.validate_config:
            success = validate_configuration(settings)
            return 0 if success else 1
        
        # Dry run
        if args.dry_run:
            print("Performing dry run...")
            app = AIAgentApplication(settings)
            await app.initialize()
            print("✓ Dry run completed successfully")
            await app.cleanup()
            return 0
        
        # Normal run
        runner = ProductionRunner(args.config, args.env)
        return await runner.run()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"Fatal error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Ensure we're using the right event loop policy
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        # Try to use uvloop for better performance on Unix systems
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass
    
    # Run the main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)