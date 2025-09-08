"""
Configuration Settings for AI Agent
Centralized configuration management with validation.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import os
import json
import yaml
from pydantic import BaseSettings, Field, validator, root_validator


class Environment(str, Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str = "postgresql+asyncpg://localhost:5432/ai_agent"
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False
    connect_timeout: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    ssl_mode: Optional[str] = None
    
    def __post_init__(self):
        if not self.url:
            raise ValueError("Database URL is required")


@dataclass
class RedisConfig:
    """Redis configuration"""
    url: str = "redis://localhost:6379/0"
    password: Optional[str] = None
    max_connections: int = 50
    retry_on_timeout: bool = True
    socket_timeout: int = 30
    socket_connect_timeout: int = 30
    connection_pool_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMConfig:
    """LLM provider configuration"""
    provider: str = "openai"  # openai, anthropic, azure, google, etc.
    api_key: str = ""
    model: str = "gpt-4"
    max_tokens: int = 4000
    temperature: float = 0.7
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    endpoint: Optional[str] = None
    api_version: Optional[str] = None
    
    # Provider-specific settings
    openai_config: Dict[str, Any] = field(default_factory=dict)
    anthropic_config: Dict[str, Any] = field(default_factory=dict)
    azure_config: Dict[str, Any] = field(default_factory=dict)
    google_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorStoreConfig:
    """Vector store configuration"""
    provider: str = "chroma"  # chroma, pinecone, weaviate, qdrant, faiss
    host: str = "localhost"
    port: int = 8000
    api_key: Optional[str] = None
    collection_name: str = "ai_agent_vectors"
    dimension: int = 1536
    metric: str = "cosine"
    
    # Provider-specific settings
    chroma_config: Dict[str, Any] = field(default_factory=dict)
    pinecone_config: Dict[str, Any] = field(default_factory=dict)
    weaviate_config: Dict[str, Any] = field(default_factory=dict)
    qdrant_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityConfig:
    """Security configuration"""
    jwt_secret_key: str = "your-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    api_key_header: str = "X-API-Key"
    rate_limit: int = 100  # requests per minute
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    encryption_key: Optional[str] = None
    enable_input_validation: bool = True
    enable_output_sanitization: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration"""
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "json"
    log_file_path: Optional[str] = None
    log_max_size: str = "10MB"
    log_backup_count: int = 5
    
    # Metrics
    enable_metrics: bool = True
    metrics_port: int = 9090
    metrics_path: str = "/metrics"
    
    # Tracing
    enable_tracing: bool = False
    jaeger_host: str = "localhost"
    jaeger_port: int = 6831
    service_name: str = "ai-agent"
    
    # Error tracking
    sentry_dsn: Optional[str] = None
    sentry_environment: str = "development"
    
    # Health checks
    health_check_interval: int = 30
    health_check_timeout: int = 10


@dataclass
class BrainConfig:
    """Brain component configuration"""
    enable_reflection: bool = True
    enable_planning: bool = True
    max_reasoning_depth: int = 5
    reasoning_timeout: int = 30
    
    # Chain of Thought settings
    cot_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_steps": 10,
        "step_timeout": 5,
        "enable_verification": True
    })
    
    # Tree of Thoughts settings
    tot_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_branches": 5,
        "max_depth": 3,
        "pruning_threshold": 0.3
    })
    
    # Reflection settings
    reflection_config: Dict[str, Any] = field(default_factory=lambda: {
        "enable_self_critique": True,
        "improvement_threshold": 0.7,
        "max_iterations": 3
    })
    
    # Planning settings
    planning_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_plan_steps": 20,
        "enable_contingency": True,
        "risk_assessment": True
    })


@dataclass
class MemoryConfig:
    """Memory system configuration"""
    # Short-term memory
    short_term_capacity: int = 100
    short_term_ttl: int = 3600  # seconds
    
    # Long-term memory
    long_term_storage: str = "database"  # database, file, vector_store
    long_term_retention_days: int = 365
    
    # Vector memory
    enable_vector_memory: bool = True
    vector_dimension: int = 1536
    similarity_threshold: float = 0.7
    
    # Episodic memory
    enable_episodic_memory: bool = True
    episode_max_length: int = 50
    episode_overlap: int = 5
    
    # Memory management
    cleanup_interval: int = 300  # seconds
    importance_threshold: float = 0.3
    auto_summarize: bool = True
    
    # Memory types
    memory_types: List[str] = field(default_factory=lambda: [
        "conversation", "task", "learning", "factual", "procedural"
    ])


@dataclass
class DecisionConfig:
    """Decision engine configuration"""
    decision_timeout: int = 10
    confidence_threshold: float = 0.6
    enable_uncertainty_handling: bool = True
    fallback_strategy: str = "request_clarification"  # conservative, aggressive, request_clarification
    
    # Decision strategies
    strategies: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "simple": {"weight": 0.3},
        "contextual": {"weight": 0.4},
        "learning_based": {"weight": 0.3}
    })
    
    # Risk assessment
    enable_risk_assessment: bool = True
    risk_tolerance: float = 0.5  # 0.0 (very conservative) to 1.0 (very aggressive)
    
    # Learning settings
    enable_decision_learning: bool = True
    learning_rate: float = 0.1
    feedback_window: int = 3600  # seconds


@dataclass
class TaskConfig:
    """Task coordinator configuration"""
    max_concurrent_tasks: int = 10
    task_timeout: int = 300  # seconds
    max_retry_attempts: int = 3
    retry_delay: float = 2.0
    
    # Task prioritization
    enable_prioritization: bool = True
    priority_factors: List[str] = field(default_factory=lambda: [
        "urgency", "importance", "complexity", "resources"
    ])
    
    # Task execution
    execution_strategies: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "sequential": {"enabled": True},
        "parallel": {"enabled": True, "max_workers": 5},
        "pipeline": {"enabled": True, "buffer_size": 10}
    })
    
    # Task monitoring
    enable_progress_tracking: bool = True
    progress_update_interval: int = 5  # seconds
    enable_performance_metrics: bool = True


@dataclass
class CommunicationConfig:
    """Communication configuration"""
    default_timeout: int = 30
    max_message_size: int = 1024 * 1024  # 1MB
    enable_message_history: bool = True
    history_retention_hours: int = 24
    
    # Message processing
    enable_preprocessing: bool = True
    enable_postprocessing: bool = True
    max_processing_time: int = 10
    
    # Channels
    enabled_channels: List[str] = field(default_factory=lambda: ["http", "websocket"])
    channel_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class PluginConfig:
    """Plugin system configuration"""
    plugin_directories: List[str] = field(default_factory=lambda: ["plugins", "~/.ai_agent/plugins"])
    auto_discovery: bool = True
    auto_reload: bool = False
    enable_hot_reload: bool = False
    
    # Plugin security
    enable_sandboxing: bool = True
    allowed_imports: List[str] = field(default_factory=lambda: [
        "os", "sys", "json", "yaml", "requests", "asyncio", "datetime"
    ])
    forbidden_imports: List[str] = field(default_factory=lambda: [
        "subprocess", "eval", "exec", "compile", "__import__"
    ])
    
    # Plugin management
    max_plugins: int = 50
    plugin_timeout: int = 30
    enable_plugin_metrics: bool = True
    plugin_isolation: bool = True


@dataclass
class PerformanceConfig:
    """Performance optimization configuration"""
    # Memory management
    memory_limit_mb: int = 1024
    memory_warning_threshold: float = 0.8
    garbage_collection_interval: int = 300
    
    # Concurrency
    max_workers: int = 10
    worker_timeout: int = 30
    enable_async_processing: bool = True
    
    # Caching
    enable_response_cache: bool = True
    cache_ttl: int = 300
    cache_size_limit: int = 1000
    
    # Request optimization
    request_timeout: int = 30
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    enable_request_compression: bool = True
    
    # Background tasks
    background_task_interval: int = 60
    cleanup_interval: int = 300
    maintenance_hour: int = 2  # 2 AM


@dataclass
class DevelopmentConfig:
    """Development-specific configuration"""
    debug: bool = False
    hot_reload: bool = False
    auto_reload: bool = False
    profiling_enabled: bool = False
    mock_external_services: bool = False
    
    # Testing
    test_mode: bool = False
    test_data_path: str = "tests/data"
    enable_test_fixtures: bool = True
    
    # Development tools
    enable_dev_endpoints: bool = False
    dev_token: str = "dev-token-change-me"
    allow_dangerous_operations: bool = False


class AgentSettings(BaseSettings):
    """Main settings class using Pydantic for validation"""
    
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    
    # Server
    host: str = "localhost"
    port: int = 8000
    workers: int = 1
    
    # Core configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    # Component configurations
    brain: BrainConfig = Field(default_factory=BrainConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    task: TaskConfig = Field(default_factory=TaskConfig)
    communication: CommunicationConfig = Field(default_factory=CommunicationConfig)
    plugin: PluginConfig = Field(default_factory=PluginConfig)
    
    # System configurations
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    development: DevelopmentConfig = Field(default_factory=DevelopmentConfig)
    
    # Additional settings
    timezone: str = "UTC"
    locale: str = "en_US"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        case_sensitive = False
        
        # Custom env var names
        fields = {
            "database": {"env": "DATABASE"},
            "redis": {"env": "REDIS"},
            "llm": {"env": "LLM"},
            "vector_store": {"env": "VECTOR_STORE"},
        }
    
    @validator("environment", pre=True)
    def validate_environment(cls, v):
        """Validate environment setting"""
        if isinstance(v, str):
            try:
                return Environment(v.lower())
            except ValueError:
                valid_envs = [e.value for e in Environment]
                raise ValueError(f"Invalid environment. Must be one of: {valid_envs}")
        return v
    
    @validator("port")
    def validate_port(cls, v):
        """Validate port number"""
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v
    
    @root_validator
    def validate_production_settings(cls, values):
        """Validate production-specific settings"""
        env = values.get("environment")
        if env == Environment.PRODUCTION:
            security = values.get("security", SecurityConfig())
            
            # Ensure secure settings in production
            if security.jwt_secret_key == "your-secret-key-change-in-production":
                raise ValueError("JWT secret key must be changed in production")
            
            if values.get("debug", False):
                raise ValueError("Debug mode should not be enabled in production")
                
            if security.cors_origins == ["*"]:
                raise ValueError("CORS should be configured properly in production")
        
        return values
    
    def get_database_url(self) -> str:
        """Get database URL with environment variable substitution"""
        url = self.database.url
        # Replace environment variables in URL
        if "${" in url:
            import string
            template = string.Template(url)
            url = template.safe_substitute(os.environ)
        return url
    
    def get_redis_url(self) -> str:
        """Get Redis URL with environment variable substitution"""
        url = self.redis.url
        if "${" in url:
            import string
            template = string.Template(url)
            url = template.safe_substitute(os.environ)
        return url
    
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.environment == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.environment == Environment.PRODUCTION
    
    def is_testing(self) -> bool:
        """Check if running in testing mode"""
        return self.environment == Environment.TESTING
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return self.dict()
    
    def save_to_file(self, file_path: Union[str, Path], format: str = "yaml") -> None:
        """Save settings to file"""
        file_path = Path(file_path)
        data = self.to_dict()
        
        if format.lower() == "json":
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        elif format.lower() == "yaml":
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            raise ValueError("Format must be 'json' or 'yaml'")
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> "AgentSettings":
        """Load settings from file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Settings file not found: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            if file_path.suffix.lower() == ".json":
                data = json.load(f)
            elif file_path.suffix.lower() in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            else:
                raise ValueError("File must be .json, .yaml, or .yml")
        
        return cls(**data)


@dataclass
class AgentConfig:
    """Consolidated agent configuration"""
    
    def __init__(self, settings: AgentSettings = None):
        if settings is None:
            settings = AgentSettings()
        
        self.settings = settings
        
        # Extract component configs
        self.brain_config = settings.brain
        self.memory_config = settings.memory
        self.decision_config = settings.decision
        self.task_config = settings.task
        self.communication_config = settings.communication
        self.context_config = {}  # Additional context configuration
        
        # Database configs
        self.database_config = settings.database
        self.redis_config = settings.redis
        self.vector_store_config = settings.vector_store
        
        # LLM config
        self.llm_config = settings.llm
        
        # System configs
        self.security_config = settings.security
        self.monitoring_config = settings.monitoring
        self.plugin_config = settings.plugin
        self.performance_config = settings.performance
    
    def get_component_config(self, component_name: str) -> Dict[str, Any]:
        """Get configuration for a specific component"""
        config_map = {
            "brain": self.brain_config,
            "memory": self.memory_config,
            "decision": self.decision_config,
            "task": self.task_config,
            "communication": self.communication_config,
            "database": self.database_config,
            "redis": self.redis_config,
            "vector_store": self.vector_store_config,
            "llm": self.llm_config,
            "security": self.security_config,
            "monitoring": self.monitoring_config,
            "plugin": self.plugin_config,
            "performance": self.performance_config,
        }
        
        return config_map.get(component_name, {})
    
    def update_config(self, component_name: str, config_updates: Dict[str, Any]) -> None:
        """Update configuration for a component"""
        if hasattr(self, f"{component_name}_config"):
            current_config = getattr(self, f"{component_name}_config")
            if hasattr(current_config, '__dict__'):
                for key, value in config_updates.items():
                    if hasattr(current_config, key):
                        setattr(current_config, key, value)
    
    def validate_config(self) -> List[str]:
        """Validate all configuration settings"""
        errors = []
        
        # Validate database config
        if not self.database_config.url:
            errors.append("Database URL is required")
        
        # Validate LLM config
        if not self.llm_config.api_key and not self.settings.development.mock_external_services:
            errors.append("LLM API key is required")
        
        # Validate security in production
        if self.settings.is_production():
            if self.security_config.jwt_secret_key == "your-secret-key-change-in-production":
                errors.append("JWT secret key must be changed in production")
            
            if not self.security_config.encryption_key:
                errors.append("Encryption key is required in production")
        
        # Validate memory limits
        if self.performance_config.memory_limit_mb < 512:
            errors.append("Memory limit should be at least 512MB")
        
        # Validate task limits
        if self.task_config.max_concurrent_tasks > 100:
            errors.append("Max concurrent tasks should not exceed 100")
        
        return errors


def load_settings(
    config_file: Optional[Union[str, Path]] = None,
    env_file: Optional[str] = None
) -> AgentSettings:
    """
    Load settings from various sources.
    
    Args:
        config_file: Optional path to configuration file
        env_file: Optional path to environment file
    
    Returns:
        Loaded settings
    """
    # Load from file if provided
    if config_file:
        settings = AgentSettings.load_from_file(config_file)
    else:
        # Load from environment variables
        if env_file:
            settings = AgentSettings(_env_file=env_file)
        else:
            settings = AgentSettings()
    
    return settings


def create_default_config_file(file_path: Union[str, Path], format: str = "yaml") -> None:
    """
    Create a default configuration file.
    
    Args:
        file_path: Path where to create the file
        format: File format ('yaml' or 'json')
    """
    default_settings = AgentSettings()
    default_settings.save_to_file(file_path, format)


def get_config_template() -> Dict[str, Any]:
    """Get a configuration template with all available options"""
    return {
        "environment": "development",
        "debug": False,
        "host": "localhost",
        "port": 8000,
        "database": {
            "url": "postgresql+asyncpg://user:password@localhost:5432/ai_agent",
            "pool_size": 10,
            "max_overflow": 20
        },
        "redis": {
            "url": "redis://localhost:6379/0",
            "max_connections": 50
        },
        "llm": {
            "provider": "openai",
            "api_key": "your-api-key-here",
            "model": "gpt-4",
            "max_tokens": 4000,
            "temperature": 0.7
        },
        "vector_store": {
            "provider": "chroma",
            "host": "localhost",
            "port": 8000,
            "collection_name": "ai_agent_vectors"
        },
        "security": {
            "jwt_secret_key": "your-secret-key-change-in-production",
            "jwt_algorithm": "HS256",
            "rate_limit": 100
        },
        "monitoring": {
            "log_level": "INFO",
            "enable_metrics": True,
            "metrics_port": 9090
        },
        "brain": {
            "enable_reflection": True,
            "enable_planning": True,
            "max_reasoning_depth": 5
        },
        "memory": {
            "short_term_capacity": 100,
            "enable_vector_memory": True,
            "cleanup_interval": 300
        },
        "decision": {
            "confidence_threshold": 0.6,
            "enable_risk_assessment": True,
            "fallback_strategy": "request_clarification"
        },
        "task": {
            "max_concurrent_tasks": 10,
            "task_timeout": 300,
            "enable_prioritization": True
        },
        "plugin": {
            "plugin_directories": ["plugins"],
            "auto_discovery": True,
            "enable_sandboxing": True
        },
        "performance": {
            "memory_limit_mb": 1024,
            "max_workers": 10,
            "enable_response_cache": True
        }
    }


# Global settings instance
_settings: Optional[AgentSettings] = None


def get_settings() -> AgentSettings:
    """Get global settings instance"""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def set_settings(settings: AgentSettings) -> None:
    """Set global settings instance"""
    global _settings
    _settings = settings


def reload_settings() -> AgentSettings:
    """Reload settings from environment"""
    global _settings
    _settings = load_settings()
    return _settings