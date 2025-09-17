# src/ai_agent/security/authentication.py
"""
Authentication system for AI Agent
Handles JWT tokens, API keys, and user authentication.
"""

import jwt
import bcrypt
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional

from ..config.settings import SecurityConfig
from ..monitoring.logger import StructuredLogger
from ..utils.exceptions import AuthenticationException


async def authenticate_request(token: str, config: SecurityConfig) -> Dict[str, Any]:
    """Authenticate request using JWT token or API key"""
    logger = StructuredLogger(__name__)
    
    try:
        # Try JWT authentication first
        if token.startswith("Bearer "):
            jwt_token = token.replace("Bearer ", "")
            return await authenticate_jwt(jwt_token, config)
        
        # Try API key authentication
        return await authenticate_api_key(token, config)
        
    except Exception as e:
        logger.warning("Authentication failed", token=token[:20], error=str(e))
        raise AuthenticationException(f"Authentication failed: {e}")


async def authenticate_jwt(token: str, config: SecurityConfig) -> Dict[str, Any]:
    """Authenticate JWT token"""
    try:
        payload = jwt.decode(
            token,
            config.jwt_secret_key,
            algorithms=[config.jwt_algorithm]
        )
        
        # Check expiration
        if datetime.fromtimestamp(payload['exp'], timezone.utc) < datetime.now(timezone.utc):
            raise AuthenticationException("Token expired")
        
        return {
            "user_id": payload.get("user_id"),
            "username": payload.get("username"),
            "roles": payload.get("roles", []),
            "is_admin": "admin" in payload.get("roles", []),
            "auth_type": "jwt"
        }
        
    except jwt.InvalidTokenError as e:
        raise AuthenticationException(f"Invalid JWT token: {e}")


async def authenticate_api_key(api_key: str, config: SecurityConfig) -> Dict[str, Any]:
    """Authenticate API key"""
    # In production, this would check against a database
    # For now, simple hardcoded check
    if api_key == "dev-api-key-12345":
        return {
            "user_id": "api_user",
            "username": "api_user", 
            "roles": ["api_user"],
            "is_admin": False,
            "auth_type": "api_key"
        }
    
    raise AuthenticationException("Invalid API key")


# src/ai_agent/security/authorization.py
"""
Authorization system for AI Agent
Role-based access control and permission checking.
"""

from typing import Dict, Any, List
from ..monitoring.logger import StructuredLogger
from ..utils.exceptions import AuthorizationException


async def authorize_request(user: Dict[str, Any], resource: str, action: str) -> bool:
    """Check if user is authorized for action on resource"""
    logger = StructuredLogger(__name__)
    
    try:
        if not user:
            return False
        
        # Admin users have access to everything
        if user.get("is_admin", False):
            return True
        
        # Check specific permissions
        roles = user.get("roles", [])
        
        # Define role permissions
        permissions = {
            "user": ["chat", "read_own_data"],
            "api_user": ["chat", "tasks", "read_tools"],
            "premium_user": ["chat", "tasks", "read_tools", "advanced_features"],
            "admin": ["*"]  # All permissions
        }
        
        # Check permissions
        for role in roles:
            if role in permissions:
                role_perms = permissions[role]
                if "*" in role_perms or action in role_perms:
                    return True
        
        logger.warning("Authorization denied", 
                      user_id=user.get("user_id"),
                      resource=resource,
                      action=action)
        
        return False
        
    except Exception as e:
        logger.error("Authorization check failed", error=str(e))
        return False


# src/ai_agent/security/input_validation.py
"""
Input validation and sanitization for AI Agent
Protects against malicious inputs and data corruption.
"""

import re
import html
from typing import Dict, Any, List
from ..monitoring.logger import StructuredLogger


async def validate_input(data: Any, validation_type: str = "general") -> Dict[str, Any]:
    """Validate input data based on type"""
    logger = StructuredLogger(__name__)
    errors = []
    
    try:
        if validation_type == "text":
            errors.extend(_validate_text_input(data))
        elif validation_type == "email":
            errors.extend(_validate_email(data))
        elif validation_type == "url":
            errors.extend(_validate_url(data))
        elif validation_type == "json":
            errors.extend(_validate_json_input(data))
        
        # Common security checks
        errors.extend(_check_security_threats(str(data)))
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
        
    except Exception as e:
        logger.error("Input validation failed", error=str(e))
        return {"valid": False, "errors": [f"Validation error: {e}"]}


async def sanitize_input(data: str) -> str:
    """Sanitize input string"""
    if not isinstance(data, str):
        data = str(data)
    
    # HTML escape
    data = html.escape(data)
    
    # Remove or escape potentially dangerous characters
    data = re.sub(r'[<>"\']', '', data)  # Remove HTML/JS chars
    data = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', data)  # Remove control chars
    
    # Limit length
    if len(data) > 10000:
        data = data[:10000]
    
    return data.strip()


def _validate_text_input(text: str) -> List[str]:
    """Validate text input"""
    errors = []
    
    if not isinstance(text, str):
        errors.append("Input must be a string")
        return errors
    
    if len(text) > 10000:
        errors.append("Text too long (max 10000 characters)")
    
    if len(text.strip()) == 0:
        errors.append("Text cannot be empty")
    
    return errors


def _validate_email(email: str) -> List[str]:
    """Validate email format"""
    errors = []
    
    if not isinstance(email, str):
        errors.append("Email must be a string")
        return errors
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        errors.append("Invalid email format")
    
    return errors


def _validate_url(url: str) -> List[str]:
    """Validate URL format"""
    errors = []
    
    if not isinstance(url, str):
        errors.append("URL must be a string")
        return errors
    
    url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    if not re.match(url_pattern, url):
        errors.append("Invalid URL format")
    
    return errors


def _validate_json_input(data: Any) -> List[str]:
    """Validate JSON input"""
    errors = []
    
    try:
        import json
        if isinstance(data, str):
            json.loads(data)  # Test parsing
        else:
            json.dumps(data)  # Test serialization
    except json.JSONDecodeError:
        errors.append("Invalid JSON format")
    except TypeError:
        errors.append("Data not JSON serializable")
    
    return errors


def _check_security_threats(data: str) -> List[str]:
    """Check for common security threats"""
    errors = []
    
    # SQL injection patterns
    sql_patterns = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b)",
        r"(\bUNION\b.*\bSELECT\b)",
        r"(\bOR\b.*=.*)",
        r"(';|\")",
    ]
    
    for pattern in sql_patterns:
        if re.search(pattern, data, re.IGNORECASE):
            errors.append("Potentially malicious SQL detected")
            break
    
    # Script injection patterns
    script_patterns = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"vbscript:",
        r"onload=",
        r"onerror=",
    ]
    
    for pattern in script_patterns:
        if re.search(pattern, data, re.IGNORECASE):
            errors.append("Potentially malicious script detected")
            break
    
    # Command injection patterns
    command_patterns = [
        r"(\b(rm|del|format|shutdown|reboot)\b)",
        r"(;|\||\&\&|\|\|)",
        r"(\$\(|\`)",
    ]
    
    for pattern in command_patterns:
        if re.search(pattern, data, re.IGNORECASE):
            errors.append("Potentially malicious command detected")
            break
    
    return errors