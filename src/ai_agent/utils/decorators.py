"""
Utility Decorators for AI Agent
Decorators for retry, timeout, caching, and other common functionality.
"""

import asyncio
import functools
import time
from typing import Any, Callable, Dict, Optional, Union
from datetime import datetime, timezone, timedelta

from ..monitoring.logger import StructuredLogger


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0, exceptions: tuple = (Exception,)):
    """
    Retry decorator for functions
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier for delay
        exceptions: Tuple of exceptions to catch and retry on
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = StructuredLogger(func.__module__)
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        logger.error(f"Function {func.__name__} failed after {max_attempts} attempts", 
                                   error=str(e))
                        break
                    
                    logger.warning(f"Function {func.__name__} failed on attempt {attempt + 1}, retrying in {current_delay}s", 
                                 error=str(e))
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            raise last_exception
        
        return wrapper
    return decorator


def async_retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0, exceptions: tuple = (Exception,)):
    """
    Async retry decorator for coroutine functions
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier for delay
        exceptions: Tuple of exceptions to catch and retry on
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            logger = StructuredLogger(func.__module__)
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        logger.error(f"Async function {func.__name__} failed after {max_attempts} attempts", 
                                   error=str(e))
                        break
                    
                    logger.warning(f"Async function {func.__name__} failed on attempt {attempt + 1}, retrying in {current_delay}s", 
                                 error=str(e))
                    
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
            raise last_exception
        
        return wrapper
    return decorator


def timeout(seconds: float):
    """
    Timeout decorator for functions
    
    Args:
        seconds: Timeout in seconds
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            # Set up signal handler
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(seconds))
            
            try:
                result = func(*args, **kwargs)
                signal.alarm(0)  # Cancel alarm
                return result
            except Exception as e:
                signal.alarm(0)  # Cancel alarm
                raise
            finally:
                signal.signal(signal.SIGALRM, old_handler)
        
        return wrapper
    return decorator


def async_timeout(seconds: float):
    """
    Async timeout decorator for coroutine functions
    
    Args:
        seconds: Timeout in seconds
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                raise TimeoutError(f"Async function {func.__name__} timed out after {seconds} seconds")
        
        return wrapper
    return decorator


def cached(ttl: Optional[float] = None, maxsize: int = 128):
    """
    Caching decorator with TTL support
    
    Args:
        ttl: Time to live in seconds (None for no expiration)
        maxsize: Maximum cache size
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_times = {}
        access_order = []
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            current_time = time.time()
            
            # Check if cached result exists and is still valid
            if key in cache:
                if ttl is None or (current_time - cache_times[key]) < ttl:
                    # Move to end of access order
                    if key in access_order:
                        access_order.remove(key)
                    access_order.append(key)
                    return cache[key]
                else:
                    # Expired, remove from cache
                    del cache[key]
                    del cache_times[key]
                    if key in access_order:
                        access_order.remove(key)
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            # Manage cache size
            if len(cache) >= maxsize:
                # Remove least recently used item
                lru_key = access_order.pop(0)
                del cache[lru_key]
                del cache_times[lru_key]
            
            # Add to cache
            cache[key] = result
            cache_times[key] = current_time
            access_order.append(key)
            
            return result
        
        # Add cache management methods
        wrapper.cache_clear = lambda: (cache.clear(), cache_times.clear(), access_order.clear())
        wrapper.cache_info = lambda: {
            'hits': len([k for k in cache if k in access_order]),
            'misses': 0,  # This would require more complex tracking
            'maxsize': maxsize,
            'currsize': len(cache),
            'ttl': ttl
        }
        
        return wrapper
    return decorator


def rate_limit(calls: int, period: float):
    """
    Rate limiting decorator
    
    Args:
        calls: Number of calls allowed
        period: Time period in seconds
    """
    def decorator(func: Callable) -> Callable:
        call_times = []
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            
            # Remove old call times outside the period
            call_times[:] = [t for t in call_times if current_time - t <= period]
            
            # Check if rate limit exceeded
            if len(call_times) >= calls:
                sleep_time = period - (current_time - call_times[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    current_time = time.time()
                    # Remove old call times again
                    call_times[:] = [t for t in call_times if current_time - t <= period]
            
            # Record this call
            call_times.append(current_time)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def async_rate_limit(calls: int, period: float):
    """
    Async rate limiting decorator
    
    Args:
        calls: Number of calls allowed
        period: Time period in seconds
    """
    def decorator(func: Callable) -> Callable:
        call_times = []
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            current_time = time.time()
            
            # Remove old call times outside the period
            call_times[:] = [t for t in call_times if current_time - t <= period]
            
            # Check if rate limit exceeded
            if len(call_times) >= calls:
                sleep_time = period - (current_time - call_times[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    current_time = time.time()
                    # Remove old call times again
                    call_times[:] = [t for t in call_times if current_time - t <= period]
            
            # Record this call
            call_times.append(current_time)
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def measure_time(log_result: bool = True):
    """
    Decorator to measure execution time
    
    Args:
        log_result: Whether to log the execution time
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                if log_result:
                    logger = StructuredLogger(func.__module__)
                    logger.debug(f"Function {func.__name__} executed in {execution_time:.3f}s")
                
                # Attach execution time to result if it's a dict
                if isinstance(result, dict):
                    result['_execution_time'] = execution_time
                
                return result
            
            except Exception as e:
                execution_time = time.time() - start_time
                if log_result:
                    logger = StructuredLogger(func.__module__)
                    logger.error(f"Function {func.__name__} failed after {execution_time:.3f}s", error=str(e))
                raise
        
        return wrapper
    return decorator


def async_measure_time(log_result: bool = True):
    """
    Async decorator to measure execution time
    
    Args:
        log_result: Whether to log the execution time
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                if log_result:
                    logger = StructuredLogger(func.__module__)
                    logger.debug(f"Async function {func.__name__} executed in {execution_time:.3f}s")
                
                # Attach execution time to result if it's a dict
                if isinstance(result, dict):
                    result['_execution_time'] = execution_time
                
                return result
            
            except Exception as e:
                execution_time = time.time() - start_time
                if log_result:
                    logger = StructuredLogger(func.__module__)
                    logger.error(f"Async function {func.__name__} failed after {execution_time:.3f}s", error=str(e))
                raise
        
        return wrapper
    return decorator


def validate_types(**type_checks):
    """
    Decorator to validate argument types
    
    Usage:
        @validate_types(x=int, y=str, z=(int, float))
        def my_func(x, y, z):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import inspect
            
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate types
            for param_name, expected_type in type_checks.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    
                    if value is not None and not isinstance(value, expected_type):
                        raise TypeError(f"Argument '{param_name}' must be of type {expected_type}, got {type(value)}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def deprecated(reason: str = "This function is deprecated"):
    """
    Decorator to mark functions as deprecated
    
    Args:
        reason: Reason for deprecation
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import warnings
            warnings.warn(
                f"{func.__name__} is deprecated: {reason}",
                DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def singleton(cls):
    """
    Decorator to make a class a singleton
    """
    instances = {}
    
    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance


def memoize_property(func):
    """
    Decorator for memoizing property methods
    """
    attr_name = f'_memoized_{func.__name__}'
    
    @property
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)
    
    return wrapper


def require_auth(permission: Optional[str] = None):
    """
    Decorator to require authentication for functions
    Used mainly for API endpoints
    
    Args:
        permission: Required permission string
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # This would integrate with the authentication system
            # For now, it's a placeholder
            
            # Check if user context exists in kwargs
            user = kwargs.get('user') or (args[0] if args and hasattr(args[0], 'user') else None)
            
            if not user:
                raise PermissionError("Authentication required")
            
            if permission and not user.get('permissions', {}).get(permission, False):
                raise PermissionError(f"Permission '{permission}' required")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def circuit_breaker(failure_threshold: int = 5, recovery_timeout: float = 60.0):
    """
    Circuit breaker decorator to prevent cascading failures
    
    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Time to wait before trying again
    """
    def decorator(func: Callable) -> Callable:
        failure_count = 0
        last_failure_time = None
        state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        
        @functools.wr