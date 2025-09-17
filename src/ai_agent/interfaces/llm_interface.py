"""
LLM Interface Implementations
Concrete implementations for various Large Language Model providers.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, AsyncGenerator, Union
from dataclasses import dataclass
import time
from datetime import datetime, timezone

from .base import LLMInterface
from ..config.settings import LLMConfig
from ..monitoring.logger import StructuredLogger, get_performance_logger
from ..utils.exceptions import LLMException, LLMTimeoutException, LLMRateLimitException, LLMQuotaException


@dataclass
class LLMResponse:
    """Standardized LLM response"""
    content: str
    model: str
    provider: str
    tokens_used: int = 0
    cost: Optional[float] = None
    metadata: Dict[str, Any] = None
    response_time: float = 0.0


class OpenAIInterface(LLMInterface):
    """OpenAI API implementation"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config.dict() if hasattr(config, 'dict') else config.__dict__)
        self.config = config
        self.logger = StructuredLogger(__name__)
        self.performance_logger = get_performance_logger()
        
        # Initialize OpenAI client
        try:
            import openai
            self.client = openai.AsyncOpenAI(
                api_key=config.api_key,
                base_url=config.endpoint,
                timeout=config.timeout
            )
            self.openai = openai
        except ImportError:
            raise LLMException("OpenAI library not installed. Install with: pip install openai")
        
        # Rate limiting
        self._request_times = []
        self._rate_limit_lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize the LLM interface"""
        try:
            # Test connection with a simple request
            await self.client.models.list()
            self.status = "ready"
            self.logger.info("OpenAI interface initialized successfully", model=self.config.model)
        except Exception as e:
            self.status = "error"
            self.logger.error("Failed to initialize OpenAI interface", error=str(e))
            raise LLMException(f"OpenAI initialization failed: {e}") from e
    
    async def generate_response(self, prompt: str, context: Dict[str, Any] = None, **kwargs) -> str:
        """Generate response using OpenAI API"""
        
        start_time = time.time()
        
        try:
            # Apply rate limiting
            await self._apply_rate_limiting()
            
            # Prepare messages
            messages = self._prepare_messages(prompt, context)
            
            # Merge configuration with kwargs
            request_params = {
                "model": kwargs.get("model", self.config.model),
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
                **self.config.openai_config,
                **{k: v for k, v in kwargs.items() if k not in ["model", "messages", "max_tokens", "temperature"]}
            }
            
            # Make API call
            response = await self.client.chat.completions.create(**request_params)
            
            # Extract response data
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            # Calculate cost (approximate)
            cost = self._calculate_cost(tokens_used, request_params["model"])
            
            # Log performance
            response_time = time.time() - start_time
            self.performance_logger.log_llm_call(
                provider="openai",
                model=request_params["model"],
                tokens_used=tokens_used,
                duration=response_time,
                cost=cost
            )
            
            return content
            
        except self.openai.RateLimitError as e:
            self.logger.warning("OpenAI rate limit exceeded", error=str(e))
            raise LLMRateLimitException("OpenAI rate limit exceeded", retry_after=60) from e
            
        except self.openai.APITimeoutError as e:
            self.logger.error("OpenAI request timeout", error=str(e))
            raise LLMTimeoutException("OpenAI request timed out") from e
            
        except Exception as e:
            self.logger.error("OpenAI API call failed", error=str(e), prompt=prompt[:100])
            raise LLMException(f"OpenAI API call failed: {e}") from e
    
    async def generate_streaming_response(self, prompt: str, context: Dict[str, Any] = None, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        
        try:
            await self._apply_rate_limiting()
            
            messages = self._prepare_messages(prompt, context)
            
            request_params = {
                "model": kwargs.get("model", self.config.model),
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "stream": True,
                **self.config.openai_config,
                **{k: v for k, v in kwargs.items() if k not in ["model", "messages", "max_tokens", "temperature", "stream"]}
            }
            
            stream = await self.client.chat.completions.create(**request_params)
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            self.logger.error("OpenAI streaming failed", error=str(e))
            raise LLMException(f"OpenAI streaming failed: {e}") from e
    
    async def analyze_text(self, text: str, analysis_type: str = "general", **kwargs) -> Dict[str, Any]:
        """Analyze text using OpenAI"""
        
        analysis_prompts = {
            "sentiment": f"Analyze the sentiment of this text (positive/negative/neutral): {text}",
            "entities": f"Extract named entities from this text: {text}",
            "summary": f"Provide a concise summary of this text: {text}",
            "keywords": f"Extract key topics and keywords from this text: {text}",
            "general": f"Analyze this text and provide insights about its content, tone, and key points: {text}"
        }
        
        prompt = analysis_prompts.get(analysis_type, analysis_prompts["general"])
        response = await self.generate_response(prompt, **kwargs)
        
        return {
            "analysis_type": analysis_type,
            "result": response,
            "text_length": len(text)
        }
    
    async def classify_intent(self, message: str, possible_intents: List[str] = None) -> Dict[str, Any]:
        """Classify intent of a message"""
        
        if possible_intents:
            intent_list = ", ".join(possible_intents)
            prompt = f"Classify the intent of this message. Possible intents: {intent_list}\n\nMessage: {message}\n\nRespond with just the intent name."
        else:
            prompt = f"What is the intent behind this message? Classify it into one of these categories: question, request, command, complaint, compliment, information, other.\n\nMessage: {message}\n\nRespond with just the category."
        
        response = await self.generate_response(prompt, temperature=0.1)  # Low temperature for classification
        
        return {
            "intent": response.strip().lower(),
            "confidence": 0.8,  # OpenAI doesn't provide confidence scores directly
            "message": message
        }
    
    async def extract_entities(self, text: str, entity_types: List[str] = None) -> List[Dict[str, Any]]:
        """Extract entities from text"""
        
        if entity_types:
            types_str = ", ".join(entity_types)
            prompt = f"Extract entities of these types: {types_str} from this text: {text}\n\nFormat as JSON list with 'entity', 'type', 'start', 'end' fields."
        else:
            prompt = f"Extract all named entities (person, organization, location, date, etc.) from this text: {text}\n\nFormat as JSON list with 'entity', 'type', 'start', 'end' fields."
        
        response = await self.generate_response(prompt, temperature=0.1)
        
        try:
            entities = json.loads(response)
            return entities if isinstance(entities, list) else []
        except json.JSONDecodeError:
            # Fallback parsing if JSON is malformed
            return [{"entity": response, "type": "unknown", "start": 0, "end": len(response)}]
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        try:
            response = await self.client.embeddings.create(
                input=texts,
                model="text-embedding-ada-002"
            )
            
            return [embedding.embedding for embedding in response.data]
            
        except Exception as e:
            self.logger.error("OpenAI embeddings failed", error=str(e))
            raise LLMException(f"OpenAI embeddings failed: {e}") from e
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "provider": "openai",
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "supports_streaming": True,
            "supports_function_calling": True,
            "supports_embeddings": True
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for OpenAI interface"""
        try:
            # Simple test call
            test_response = await self.generate_response("Test", max_tokens=1)
            
            return {
                "status": "healthy",
                "provider": "openai",
                "model": self.config.model,
                "last_test": datetime.now(timezone.utc).isoformat(),
                "test_successful": True
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "openai",
                "model": self.config.model,
                "error": str(e),
                "last_test": datetime.now(timezone.utc).isoformat(),
                "test_successful": False
            }
    
    def _prepare_messages(self, prompt: str, context: Dict[str, Any] = None) -> List[Dict[str, str]]:
        """Prepare messages for OpenAI format"""
        messages = []
        
        # Add system message if available in context
        if context and "system_prompt" in context:
            messages.append({"role": "system", "content": context["system_prompt"]})
        elif self.config.openai_config.get("system_prompt"):
            messages.append({"role": "system", "content": self.config.openai_config["system_prompt"]})
        
        # Add conversation history if available
        if context and "conversation_history" in context:
            for interaction in context["conversation_history"]:
                if isinstance(interaction, dict):
                    if "user_message" in interaction:
                        messages.append({"role": "user", "content": interaction["user_message"]})
                    if "agent_response" in interaction:
                        messages.append({"role": "assistant", "content": interaction["agent_response"]})
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        return messages
    
    async def _apply_rate_limiting(self) -> None:
        """Apply rate limiting to API calls"""
        async with self._rate_limit_lock:
            current_time = time.time()
            
            # Remove old requests (older than 1 minute)
            self._request_times = [t for t in self._request_times if current_time - t < 60]
            
            # Check if we're hitting rate limits (rough estimate)
            if len(self._request_times) >= 50:  # Conservative limit
                sleep_time = 60 - (current_time - self._request_times[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            self._request_times.append(current_time)
    
    def _calculate_cost(self, tokens: int, model: str) -> float:
        """Calculate approximate cost for API call"""
        # Rough cost estimates (as of 2024)
        cost_per_1k_tokens = {
            "gpt-4": 0.03,
            "gpt-4-32k": 0.06,
            "gpt-3.5-turbo": 0.002,
            "gpt-3.5-turbo-16k": 0.004
        }
        
        rate = cost_per_1k_tokens.get(model, 0.002)  # Default to GPT-3.5 rate
        return (tokens / 1000) * rate


class AnthropicInterface(LLMInterface):
    """Anthropic Claude API implementation"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config.dict() if hasattr(config, 'dict') else config.__dict__)
        self.config = config
        self.logger = StructuredLogger(__name__)
        self.performance_logger = get_performance_logger()
        
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(
                api_key=config.api_key,
                timeout=config.timeout
            )
            self.anthropic = anthropic
        except ImportError:
            raise LLMException("Anthropic library not installed. Install with: pip install anthropic")
    
    async def initialize(self) -> None:
        """Initialize Anthropic interface"""
        try:
            # Test with a simple message
            await self.client.messages.create(
                model=self.config.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "test"}]
            )
            self.status = "ready"
            self.logger.info("Anthropic interface initialized successfully", model=self.config.model)
        except Exception as e:
            self.status = "error"
            self.logger.error("Failed to initialize Anthropic interface", error=str(e))
            raise LLMException(f"Anthropic initialization failed: {e}") from e
    
    async def generate_response(self, prompt: str, context: Dict[str, Any] = None, **kwargs) -> str:
        """Generate response using Anthropic API"""
        
        start_time = time.time()
        
        try:
            messages = self._prepare_messages(prompt, context)
            
            request_params = {
                "model": kwargs.get("model", self.config.model),
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "messages": messages,
                **self.config.anthropic_config,
                **{k: v for k, v in kwargs.items() if k not in ["model", "max_tokens", "temperature", "messages"]}
            }
            
            # Add system message if available
            if context and "system_prompt" in context:
                request_params["system"] = context["system_prompt"]
            elif self.config.anthropic_config.get("system_prompt"):
                request_params["system"] = self.config.anthropic_config["system_prompt"]
            
            response = await self.client.messages.create(**request_params)
            
            content = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            
            # Log performance
            response_time = time.time() - start_time
            self.performance_logger.log_llm_call(
                provider="anthropic",
                model=request_params["model"],
                tokens_used=tokens_used,
                duration=response_time
            )
            
            return content
            
        except self.anthropic.RateLimitError as e:
            self.logger.warning("Anthropic rate limit exceeded", error=str(e))
            raise LLMRateLimitException("Anthropic rate limit exceeded") from e
            
        except Exception as e:
            self.logger.error("Anthropic API call failed", error=str(e))
            raise LLMException(f"Anthropic API call failed: {e}") from e
    
    async def generate_streaming_response(self, prompt: str, context: Dict[str, Any] = None, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        try:
            messages = self._prepare_messages(prompt, context)
            
            request_params = {
                "model": kwargs.get("model", self.config.model),
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
                "messages": messages,
                "stream": True,
                **self.config.anthropic_config
            }
            
            if context and "system_prompt" in context:
                request_params["system"] = context["system_prompt"]
            
            async with self.client.messages.stream(**request_params) as stream:
                async for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            self.logger.error("Anthropic streaming failed", error=str(e))
            raise LLMException(f"Anthropic streaming failed: {e}") from e
    
    async def analyze_text(self, text: str, analysis_type: str = "general", **kwargs) -> Dict[str, Any]:
        """Analyze text using Anthropic"""
        
        analysis_prompts = {
            "sentiment": f"Analyze the sentiment of this text and respond with just: positive, negative, or neutral.\n\nText: {text}",
            "entities": f"Extract named entities from this text. Format as a simple list.\n\nText: {text}",
            "summary": f"Provide a concise summary of this text:\n\n{text}",
            "keywords": f"Extract the main keywords and topics from this text:\n\n{text}",
            "general": f"Analyze this text and provide insights about its content, tone, and key points:\n\n{text}"
        }
        
        prompt = analysis_prompts.get(analysis_type, analysis_prompts["general"])
        response = await self.generate_response(prompt, **kwargs)
        
        return {
            "analysis_type": analysis_type,
            "result": response,
            "text_length": len(text)
        }
    
    async def classify_intent(self, message: str, possible_intents: List[str] = None) -> Dict[str, Any]:
        """Classify intent of a message"""
        
        if possible_intents:
            intent_list = ", ".join(possible_intents)
            prompt = f"Classify this message into one of these intents: {intent_list}\n\nMessage: {message}\n\nRespond with just the intent name."
        else:
            prompt = f"Classify this message into one of these categories: question, request, command, complaint, compliment, information, other.\n\nMessage: {message}\n\nRespond with just the category."
        
        response = await self.generate_response(prompt, temperature=0.1)
        
        return {
            "intent": response.strip().lower(),
            "confidence": 0.8,
            "message": message
        }
    
    async def extract_entities(self, text: str, entity_types: List[str] = None) -> List[Dict[str, Any]]:
        """Extract entities from text"""
        
        if entity_types:
            types_str = ", ".join(entity_types)
            prompt = f"Extract entities of these types: {types_str} from this text.\n\nText: {text}\n\nList each entity with its type."
        else:
            prompt = f"Extract all named entities (people, organizations, locations, dates, etc.) from this text.\n\nText: {text}\n\nList each entity with its type."
        
        response = await self.generate_response(prompt, temperature=0.1)
        
        # Parse the response into structured format
        entities = []
        for line in response.split('\n'):
            line = line.strip()
            if line and ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    entity_type, entity_name = parts[0].strip(), parts[1].strip()
                    entities.append({
                        "entity": entity_name,
                        "type": entity_type.lower(),
                        "start": 0,
                        "end": len(entity_name)
                    })
        
        return entities
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings - Anthropic doesn't have embeddings API, so raise exception"""
        raise LLMException("Anthropic doesn't provide embeddings API. Use OpenAI or another provider.")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "provider": "anthropic",
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "supports_streaming": True,
            "supports_function_calling": False,
            "supports_embeddings": False
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for Anthropic interface"""
        try:
            test_response = await self.generate_response("Test", max_tokens=1)
            
            return {
                "status": "healthy",
                "provider": "anthropic",
                "model": self.config.model,
                "last_test": datetime.now(timezone.utc).isoformat(),
                "test_successful": True
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "anthropic",
                "model": self.config.model,
                "error": str(e),
                "last_test": datetime.now(timezone.utc).isoformat(),
                "test_successful": False
            }
    
    def _prepare_messages(self, prompt: str, context: Dict[str, Any] = None) -> List[Dict[str, str]]:
        """Prepare messages for Anthropic format"""
        messages = []
        
        # Add conversation history if available
        if context and "conversation_history" in context:
            for interaction in context["conversation_history"]:
                if isinstance(interaction, dict):
                    if "user_message" in interaction:
                        messages.append({"role": "user", "content": interaction["user_message"]})
                    if "agent_response" in interaction:
                        messages.append({"role": "assistant", "content": interaction["agent_response"]})
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        return messages


class MockLLMInterface(LLMInterface):
    """Mock LLM interface for testing"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config.dict() if hasattr(config, 'dict') else config.__dict__)
        self.config = config
        self.logger = StructuredLogger(__name__)
        self.call_count = 0
    
    async def initialize(self) -> None:
        """Initialize mock interface"""
        self.status = "ready"
        self.logger.info("Mock LLM interface initialized")
    
    async def generate_response(self, prompt: str, context: Dict[str, Any] = None, **kwargs) -> str:
        """Generate mock response"""
        self.call_count += 1
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Generate response based on prompt keywords
        prompt_lower = prompt.lower()
        
        if "hello" in prompt_lower or "hi" in prompt_lower:
            return "Hello! How can I help you today?"
        elif "weather" in prompt_lower:
            return "I'm sorry, I don't have access to current weather data. You might want to check a weather service."
        elif "calculate" in prompt_lower or "math" in prompt_lower:
            return "I can help with calculations. Please provide the specific mathematical problem you'd like me to solve."
        elif "error" in prompt_lower or "test error" in prompt_lower:
            raise LLMException("Mock error for testing")
        else:
            return f"This is a mock response to your prompt: '{prompt[:50]}...'. Call count: {self.call_count}"
    
    async def generate_streaming_response(self, prompt: str, context: Dict[str, Any] = None, **kwargs) -> AsyncGenerator[str, None]:
        """Generate mock streaming response"""
        response = await self.generate_response(prompt, context, **kwargs)
        
        # Yield response in chunks
        words = response.split()
        for word in words:
            await asyncio.sleep(0.05)  # Simulate streaming delay
            yield word + " "
    
    async def analyze_text(self, text: str, analysis_type: str = "general", **kwargs) -> Dict[str, Any]:
        """Mock text analysis"""
        return {
            "analysis_type": analysis_type,
            "result": f"Mock {analysis_type} analysis of text (length: {len(text)})",
            "text_length": len(text),
            "mock": True
        }
    
    async def classify_intent(self, message: str, possible_intents: List[str] = None) -> Dict[str, Any]:
        """Mock intent classification"""
        # Simple keyword-based classification for testing
        message_lower = message.lower()
        
        if "?" in message:
            intent = "question"
        elif any(word in message_lower for word in ["please", "can you", "help"]):
            intent = "request"
        elif any(word in message_lower for word in ["do", "make", "create"]):
            intent = "command"
        else:
            intent = "other"
        
        return {
            "intent": intent,
            "confidence": 0.9,
            "message": message,
            "mock": True
        }
    
    async def extract_entities(self, text: str, entity_types: List[str] = None) -> List[Dict[str, Any]]:
        """Mock entity extraction"""
        # Simple mock entities
        mock_entities = [
            {"entity": "Mock Entity", "type": "organization", "start": 0, "end": 11},
            {"entity": "Test Person", "type": "person", "start": 20, "end": 31}
        ]
        
        return mock_entities[:2] if len(text) > 20 else mock_entities[:1]
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings"""
        # Return random-ish embeddings for testing
        import random
        
        embeddings = []
        for text in texts:
            # Generate deterministic "embeddings" based on text hash
            random.seed(hash(text))
            embedding = [random.random() for _ in range(384)]  # Mock 384-dimensional embeddings
            embeddings.append(embedding)
        
        return embeddings
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get mock model information"""
        return {
            "provider": "mock",
            "model": "mock-model",
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "supports_streaming": True,
            "supports_function_calling": False,
            "supports_embeddings": True,
            "mock": True,
            "call_count": self.call_count
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Mock health check"""
        return {
            "status": "healthy",
            "provider": "mock",
            "model": "mock-model",
            "last_test": datetime.now(timezone.utc).isoformat(),
            "test_successful": True,
            "mock": True,
            "call_count": self.call_count
        }


# Factory function to create appropriate LLM interface
async def create_llm_interface(config: LLMConfig) -> LLMInterface:
    """Create and initialize appropriate LLM interface based on configuration"""
    
    provider = config.provider.lower()
    
    if provider == "openai":
        interface = OpenAIInterface(config)
    elif provider == "anthropic":
        interface = AnthropicInterface(config)
    elif provider == "mock":
        interface = MockLLMInterface(config)
    else:
        raise LLMException(f"Unsupported LLM provider: {provider}")
    
    # Initialize the interface
    await interface.initialize()
    
    return interface


# Utility functions for LLM operations

def estimate_tokens(text: str) -> int:
    """Rough estimate of token count for text"""
    # Very rough estimate: ~4 characters per token on average
    return len(text) // 4


def truncate_to_token_limit(text: str, max_tokens: int, preserve_end: bool = False) -> str:
    """Truncate text to fit within token limit"""
    
    estimated_tokens = estimate_tokens(text)
    
    if estimated_tokens <= max_tokens:
        return text
    
    # Calculate target character count
    target_chars = max_tokens * 4
    
    if preserve_end:
        # Keep the end of the text
        return "..." + text[-target_chars:]
    else:
        # Keep the beginning of the text
        return text[:target_chars] + "..."


def prepare_context_prompt(base_prompt: str, context: Dict[str, Any], max_tokens: int = 4000) -> str:
    """Prepare a prompt with context information, respecting token limits"""
    
    context_parts = []
    
    # Add relevant context information
    if context.get("conversation_history"):
        history_text = "Previous conversation:\n"
        for interaction in context["conversation_history"][-3:]:  # Last 3 interactions
            if isinstance(interaction, dict):
                if "user_message" in interaction:
                    history_text += f"User: {interaction['user_message']}\n"
                if "agent_response" in interaction:
                    history_text += f"Assistant: {interaction['agent_response']}\n"
        context_parts.append(history_text)
    
    if context.get("relevant_memories"):
        memory_text = "Relevant information:\n"
        for memory in context["relevant_memories"][:3]:  # Top 3 memories
            if isinstance(memory, dict) and "content" in memory:
                memory_text += f"- {memory['content']}\n"
        context_parts.append(memory_text)
    
    if context.get("system_state"):
        state = context["system_state"]
        if isinstance(state, dict):
            state_text = f"System status: {state.get('status', 'unknown')}\n"
            context_parts.append(state_text)
    
    # Combine context and prompt
    context_str = "\n".join(context_parts)
    full_prompt = f"{context_str}\n\nCurrent request: {base_prompt}"
    
    # Truncate if too long
    return truncate_to_token_limit(full_prompt, max_tokens, preserve_end=True)