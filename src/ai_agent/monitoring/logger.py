"""
Structured Logging System for AI Agent
Provides comprehensive logging with structured data, correlation IDs, and multiple output formats.
"""

import logging
import logging.handlers
import json
import sys
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Union
from pathlib import Path
import structlog
import traceback
from contextvars import ContextVar
import uuid

from ..config.settings import MonitoringConfig, LogLevel


# Context variables for request correlation
correlation_id: ContextVar[str] = ContextVar('correlation_id', default='')
user_id: ContextVar[str] = ContextVar('user_id', default='')
session_id: ContextVar[str] = ContextVar('session_id', default='')


class CorrelationFilter(logging.Filter):
    """Add correlation ID to log records"""
    
    def filter(self, record):
        record.correlation_id = correlation_id.get('')
        record.user_id = user_id.get('')
        record.session_id = session_id.get('')
        return True


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def __init__(self, include_extra=True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record):
        # Base log data
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add correlation data if available
        if hasattr(record, 'correlation_id') and record.correlation_id:
            log_data['correlation_id'] = record.correlation_id
        if hasattr(record, 'user_id') and record.user_id:
            log_data['user_id'] = record.user_id
        if hasattr(record, 'session_id') and record.session_id:
            log_data['session_id'] = record.session_id
        
        # Add process/thread info
        log_data['process'] = record.process
        log_data['thread'] = record.thread
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info)
            }
        
        # Add extra fields
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                             'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                             'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                             'thread', 'threadName', 'processName', 'process', 'getMessage',
                             'correlation_id', 'user_id', 'session_id']:
                    try:
                        # Ensure value is JSON serializable
                        json.dumps(value)
                        log_data[key] = value
                    except (TypeError, ValueError):
                        log_data[key] = str(value)
        
        return json.dumps(log_data, default=str)


class HumanReadableFormatter(logging.Formatter):
    """Human-readable formatter for console output"""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s [%(levelname)8s] %(name)s:%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def format(self, record):
        # Add correlation info to message if available
        base_msg = super().format(record)
        
        correlation_info = []
        if hasattr(record, 'correlation_id') and record.correlation_id:
            correlation_info.append(f"correlation={record.correlation_id[:8]}")
        if hasattr(record, 'session_id') and record.session_id:
            correlation_info.append(f"session={record.session_id[:8]}")
        if hasattr(record, 'user_id') and record.user_id:
            correlation_info.append(f"user={record.user_id}")
        
        if correlation_info:
            base_msg += f" [{', '.join(correlation_info)}]"
        
        # Add extra fields if any
        extra_fields = []
        for key, value in record.__dict__.items():
            if key.startswith('extra_'):
                extra_fields.append(f"{key[6:]}={value}")
        
        if extra_fields:
            base_msg += f" | {', '.join(extra_fields)}"
        
        return base_msg


class StructuredLogger:
    """
    Structured logger wrapper that provides additional functionality
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)
        self._context_data = {}
    
    def with_context(self, **kwargs) -> 'StructuredLogger':
        """Create a new logger instance with additional context"""
        new_logger = StructuredLogger(self.name)
        new_logger._context_data = {**self._context_data, **kwargs}
        return new_logger
    
    def _log(self, level: int, message: str, **kwargs):
        """Internal logging method with context"""
        # Merge context data with kwargs
        log_data = {**self._context_data, **kwargs}
        
        # Create extra dict for structured data
        extra = {}
        for key, value in log_data.items():
            extra[f"extra_{key}"] = value
        
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self._log(logging.CRITICAL, message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        kwargs['exc_info'] = True
        self._log(logging.ERROR, message, **kwargs)


class LoggerManager:
    """Manages logger configuration and provides utilities"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self._configured = False
    
    def setup_logging(self):
        """Setup logging configuration"""
        if self._configured:
            return
        
        # Clear existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        
        # Set root log level
        root_logger.setLevel(getattr(logging, self.config.log_level.value))
        
        # Add correlation filter to root logger
        root_logger.addFilter(CorrelationFilter())
        
        # Configure console handler
        console_handler = logging.StreamHandler(sys.stdout)
        
        if self.config.log_format == 'json':
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_handler.setFormatter(HumanReadableFormatter())
        
        root_logger.addHandler(console_handler)
        
        # Configure file handler if specified
        if self.config.log_file_path:
            self._setup_file_logging()
        
        # Configure external logging integrations
        if self.config.sentry_dsn:
            self._setup_sentry_logging()
        
        self._configured = True
    
    def _setup_file_logging(self):
        """Setup file logging with rotation"""
        log_file = Path(self.config.log_file_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Parse max size (e.g., "10MB" -> 10 * 1024 * 1024)
        max_size_str = self.config.log_max_size.upper()
        multipliers = {'KB': 1024, 'MB': 1024*1024, 'GB': 1024*1024*1024}
        
        max_bytes = 10 * 1024 * 1024  # Default 10MB
        for suffix, multiplier in multipliers.items():
            if max_size_str.endswith(suffix):
                size_value = int(max_size_str[:-len(suffix)])
                max_bytes = size_value * multiplier
                break
        
        # Create