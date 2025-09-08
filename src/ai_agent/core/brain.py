"""
Brain Component - Central Intelligence System
Handles reasoning, analysis, and response generation.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from datetime import datetime, timezone
import asyncio
import json
import logging

from ..interfaces.base import LLMInterface, ToolResult
from ..config.settings import BrainConfig
from ..monitoring.logger import StructuredLogger
from ..utils.exceptions import BrainException, LLMException
from ..reasoning.chain_of_thought import ChainOfThoughtReasoner
from ..reasoning.tree_of_thoughts import TreeOfThoughtsReasoner
from ..reasoning.reflection import ReflectionEngine
from ..reasoning.planning import PlanningEngine


@dataclass
class MessageAnalysis:
    """Result of message analysis"""
    intent: str
    entities: List[Dict[str, Any]]
    sentiment: float
    confidence: float
    complexity: str
    requires_tools: bool
    suggested_tools: List[str]
    context_dependencies: List[str]
    reasoning_type: str


@dataclass
class ExecutionPlan:
    """Plan for executing a task"""
    task_id: str
    steps: List[Dict[str, Any]]
    estimated_time: float
    required_tools: List[str]
    dependencies: List[str]
    risk_level: str
    contingency_plans: List[Dict[str, Any]]


@dataclass
class Capability:
    """Represents an agent capability"""
    name: str
    description: str
    parameters: Dict[str, Any]
    usage_count: int = 0
    success_rate: float = 1.0
    last_used: Optional[datetime] = None


class ReasoningStrategy(ABC):
    """Abstract base for reasoning strategies"""
    
    @abstractmethod
    async def reason(self, problem: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply reasoning strategy to a problem"""
        pass


class Brain:
    """
    Central intelligence system of the AI agent.
    Coordinates reasoning, analysis, and response generation.
    """
    
    def __init__(self, config: BrainConfig):
        self.config = config
        self.logger = StructuredLogger(__name__)
        
        # Core components
        self.llm: Optional[LLMInterface] = None
        self.memory = None  # Set by agent
        self.decision_engine = None  # Set by agent
        
        # Reasoning engines
        self.cot_reasoner = ChainOfThoughtReasoner(config.cot_config)
        self.tot_reasoner = TreeOfThoughtsReasoner(config.tot_config)
        self.reflection_engine = ReflectionEngine(config.reflection_config)
        self.planning_engine = PlanningEngine(config.planning_config)
        
        # Capabilities registry
        self.capabilities: Dict[str, Capability] = {}
        
        # Knowledge base
        self.knowledge_base: Dict[str, Any] = {}
        
        # Performance metrics
        self.metrics = {
            "total_analyses": 0,
            "total_responses": 0,
            "average_response_time": 0.0,
            "success_rate": 1.0
        }
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the brain component"""
        try:
            self.logger.info("Initializing Brain component")
            
            # Initialize reasoning engines
            await self.cot_reasoner.initialize()
            await self.tot_reasoner.initialize()
            await self.reflection_engine.initialize()
            await self.planning_engine.initialize()
            
            # Load knowledge base
            await self._load_knowledge_base()
            
            # Initialize built-in capabilities
            await self._initialize_capabilities()
            
            self._initialized = True
            self.logger.info("Brain component initialized successfully")
            
        except Exception as e:
            self.logger.error("Failed to initialize Brain", error=str(e))
            raise BrainException(f"Brain initialization failed: {e}") from e
    
    async def cleanup(self) -> None:
        """Cleanup brain resources"""
        try:
            # Cleanup reasoning engines
            await self.cot_reasoner.cleanup()
            await self.tot_reasoner.cleanup()
            await self.reflection_engine.cleanup()
            await self.planning_engine.cleanup()
            
            # Save knowledge base
            await self._save_knowledge_base()
            
            self._initialized = False
            self.logger.info("Brain component cleaned up")
            
        except Exception as e:
            self.logger.error("Error during brain cleanup", error=str(e))
    
    def set_llm(self, llm: LLMInterface) -> None:
        """Set the LLM interface"""
        self.llm = llm
    
    def set_memory(self, memory) -> None:
        """Set the memory system"""
        self.memory = memory
    
    def set_decision_engine(self, decision_engine) -> None:
        """Set the decision engine"""
        self.decision_engine = decision_engine
    
    async def analyze_message(self, message: str, context: Dict[str, Any]) -> MessageAnalysis:
        """
        Analyze an incoming message to understand intent and requirements.
        
        Args:
            message: The message to analyze
            context: Context information
            
        Returns:
            MessageAnalysis with detailed analysis results
        """
        if not self._initialized:
            raise BrainException("Brain not initialized")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            self.logger.debug("Analyzing message", message_length=len(message))
            
            # Extract basic features
            basic_analysis = await self._basic_message_analysis(message)
            
            # Determine reasoning approach
            reasoning_type = await self._determine_reasoning_type(message, basic_analysis)
            
            # Perform detailed analysis based on reasoning type
            if reasoning_type == "chain_of_thought":
                detailed_analysis = await self.cot_reasoner.analyze_message(message, context)
            elif reasoning_type == "tree_of_thoughts":
                detailed_analysis = await self.tot_reasoner.analyze_message(message, context)
            else:
                detailed_analysis = await self._standard_analysis(message, context)
            
            # Combine analyses
            analysis = MessageAnalysis(
                intent=detailed_analysis.get("intent", "unknown"),
                entities=detailed_analysis.get("entities", []),
                sentiment=detailed_analysis.get("sentiment", 0.0),
                confidence=detailed_analysis.get("confidence", 0.5),
                complexity=detailed_analysis.get("complexity", "medium"),
                requires_tools=detailed_analysis.get("requires_tools", False),
                suggested_tools=detailed_analysis.get("suggested_tools", []),
                context_dependencies=detailed_analysis.get("context_dependencies", []),
                reasoning_type=reasoning_type
            )
            
            # Update metrics
            processing_time = asyncio.get_event_loop().time() - start_time
            self.metrics["total_analyses"] += 1
            self._update_average_response_time(processing_time)
            
            self.logger.debug("Message analysis completed", 
                            intent=analysis.intent, 
                            complexity=analysis.complexity,
                            processing_time=processing_time)
            
            return analysis
            
        except Exception as e:
            self.logger.error("Message analysis failed", error=str(e), message=message[:100])
            raise BrainException(f"Message analysis failed: {e}") from e
    
    async def generate_response(self, decision: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        Generate a response based on decision and context.
        
        Args:
            decision: Decision from decision engine
            context: Full context information
            
        Returns:
            Generated response string
        """
        if not self.llm:
            raise BrainException("No LLM interface configured")
        
        try:
            self.logger.debug("Generating response", decision_type=decision.get("action_type"))
            
            # Prepare prompt based on decision type
            prompt = await self._prepare_response_prompt(decision, context)
            
            # Generate response using LLM
            response = await self.llm.generate_response(prompt, context)
            
            # Post-process response
            processed_response = await self._post_process_response(response, decision, context)
            
            # Reflect on response quality
            if self.config.enable_reflection:
                reflection = await self.reflection_engine.reflect_on_response(
                    processed_response, decision, context
                )
                if reflection.get("needs_improvement", False):
                    processed_response = await self._improve_response(
                        processed_response, reflection, context
                    )
            
            self.metrics["total_responses"] += 1
            return processed_response
            
        except Exception as e:
            self.logger.error("Response generation failed", error=str(e))
            raise BrainException(f"Response generation failed: {e}") from e
    
    async def create_execution_plan(self, task: Dict[str, Any]) -> ExecutionPlan:
        """
        Create a detailed execution plan for a task.
        
        Args:
            task: Task definition and requirements
            
        Returns:
            ExecutionPlan with detailed steps and requirements
        """
        try:
            self.logger.debug("Creating execution plan", task_type=task.get("type"))
            
            # Use planning engine to create plan
            plan_data = await self.planning_engine.create_plan(task)
            
            # Estimate execution time
            estimated_time = await self._estimate_execution_time(plan_data)
            
            # Identify required tools
            required_tools = await self._identify_required_tools(plan_data)
            
            # Assess risk level
            risk_level = await self._assess_risk_level(plan_data)
            
            # Create contingency plans
            contingency_plans = await self._create_contingency_plans(plan_data)
            
            execution_plan = ExecutionPlan(
                task_id=task.get("id", "unknown"),
                steps=plan_data.get("steps", []),
                estimated_time=estimated_time,
                required_tools=required_tools,
                dependencies=plan_data.get("dependencies", []),
                risk_level=risk_level,
                contingency_plans=contingency_plans
            )
            
            self.logger.debug("Execution plan created", 
                            steps_count=len(execution_plan.steps),
                            estimated_time=estimated_time,
                            risk_level=risk_level)
            
            return execution_plan
            
        except Exception as e:
            self.logger.error("Execution plan creation failed", error=str(e))
            raise BrainException(f"Execution plan creation failed: {e}") from e
    
    async def interpret_tool_result(self, result: ToolResult, context: Dict[str, Any]) -> str:
        """
        Interpret the result of a tool execution.
        
        Args:
            result: Tool execution result
            context: Context information
            
        Returns:
            Human-readable interpretation of the result
        """
        try:
            if not result.success:
                return f"Tool execution failed: {result.error}"
            
            # Prepare interpretation prompt
            interpretation_prompt = f"""
            Interpret the following tool execution result in a human-friendly way:
            
            Tool Result: {json.dumps(result.data, indent=2)}
            Context: {json.dumps(context.get('original_message', ''), indent=2)}
            
            Provide a clear, concise interpretation that:
            1. Explains what the tool accomplished
            2. Highlights key findings or results
            3. Suggests next steps if applicable
            4. Uses natural language appropriate for the user
            """
            
            if self.llm:
                interpretation = await self.llm.generate_response(interpretation_prompt, context)
                return interpretation
            else:
                # Fallback interpretation without LLM
                return self._basic_tool_result_interpretation(result)
                
        except Exception as e:
            self.logger.error("Tool result interpretation failed", error=str(e))
            return f"Tool executed successfully, but I had trouble interpreting the result: {str(e)}"
    
    async def format_task_result(self, result: Any, context: Dict[str, Any]) -> str:
        """
        Format task execution result for user presentation.
        
        Args:
            result: Task execution result
            context: Context information
            
        Returns:
            Formatted result string
        """
        try:
            formatting_prompt = f"""
            Format the following task execution result for user presentation:
            
            Result: {json.dumps(result, indent=2)}
            Context: {json.dumps(context.get('original_message', ''), indent=2)}
            
            Create a well-structured, user-friendly summary that:
            1. Clearly states what was accomplished
            2. Presents key results and findings
            3. Uses appropriate formatting (lists, sections, etc.)
            4. Maintains a helpful and professional tone
            """
            
            if self.llm:
                formatted_result = await self.llm.generate_response(formatting_prompt, context)
                return formatted_result
            else:
                return self._basic_result_formatting(result)
                
        except Exception as e:
            self.logger.error("Task result formatting failed", error=str(e))
            return f"Task completed. Result: {str(result)}"
    
    async def generate_error_response(self, error_context: Dict[str, Any]) -> str:
        """
        Generate a helpful error response for users.
        
        Args:
            error_context: Context about the error
            
        Returns:
            User-friendly error message
        """
        try:
            error_prompt = f"""
            Generate a helpful error response for the user based on this error context:
            
            Error Type: {error_context.get('error_type', 'Unknown')}
            Error Message: {error_context.get('error_message', 'No details available')}
            User Message: {error_context.get('original_message', 'No message')}
            
            Create a response that:
            1. Acknowledges the error apologetically
            2. Explains what went wrong in simple terms
            3. Suggests how the user might rephrase or try again
            4. Offers alternative approaches if applicable
            5. Maintains a helpful and professional tone
            """
            
            if self.llm:
                error_response = await self.llm.generate_response(error_prompt)
                return error_response
            else:
                return self._basic_error_response(error_context)
                
        except Exception as e:
            self.logger.error("Error response generation failed", error=str(e))
            return "I apologize, but I encountered an error. Please try rephrasing your request."
    
    async def generate_clarification_request(self, prompt: str, context: Dict[str, Any]) -> str:
        """
        Generate a clarification request for ambiguous user input.
        
        Args:
            prompt: Base clarification prompt
            context: Context information
            
        Returns:
            Clarification request message
        """
        try:
            clarification_prompt = f"""
            Generate a helpful clarification request based on:
            
            Base Prompt: {prompt}
            User Message: {context.get('original_message', '')}
            Context: {json.dumps(context.get('metadata', {}), indent=2)}
            
            Create a clarification request that:
            1. Politely asks for more information
            2. Specifies what details are needed
            3. Provides examples if helpful
            4. Maintains a conversational tone
            """
            
            if self.llm:
                clarification = await self.llm.generate_response(clarification_prompt, context)
                return clarification
            else:
                return f"{prompt} Could you provide more details about what you'd like me to help with?"
                
        except Exception as e:
            self.logger.error("Clarification request generation failed", error=str(e))
            return "Could you please provide more details about what you'd like me to help with?"
    
    async def learn_from_experience(self, experience: Dict[str, Any]) -> None:
        """
        Learn from task execution experience to improve future performance.
        
        Args:
            experience: Experience data including task, result, and success metrics
        """
        try:
            task = experience.get("task", {})
            result = experience.get("result", {})
            success = experience.get("success", True)
            
            # Update capability success rates
            used_tools = task.get("tools_used", [])
            for tool_name in used_tools:
                if tool_name in self.capabilities:
                    capability = self.capabilities[tool_name]
                    capability.usage_count += 1
                    capability.last_used = datetime.now(timezone.utc)
                    
                    # Update success rate
                    old_rate = capability.success_rate
                    old_count = capability.usage_count - 1
                    new_rate = (old_rate * old_count + (1.0 if success else 0.0)) / capability.usage_count
                    capability.success_rate = new_rate
            
            # Store learning in knowledge base
            learning_key = f"experience_{datetime.now(timezone.utc).isoformat()}"
            self.knowledge_base[learning_key] = {
                "task_type": task.get("type"),
                "success": success,
                "tools_used": used_tools,
                "execution_time": result.get("execution_time", 0),
                "lessons_learned": await self._extract_lessons(experience)
            }
            
            # Update overall metrics
            if success:
                self.metrics["success_rate"] = (
                    self.metrics["success_rate"] * 0.95 + 0.05
                )
            else:
                self.metrics["success_rate"] = (
                    self.metrics["success_rate"] * 0.95
                )
            
            self.logger.debug("Learning from experience completed", success=success)
            
        except Exception as e:
            self.logger.error("Learning from experience failed", error=str(e))
    
    def register_capability(self, name: str, description: str, parameters: Dict[str, Any] = None) -> None:
        """
        Register a new capability (tool or skill).
        
        Args:
            name: Capability name
            description: Capability description
            parameters: Optional parameters schema
        """
        self.capabilities[name] = Capability(
            name=name,
            description=description,
            parameters=parameters or {}
        )
        self.logger.debug("Capability registered", name=name)
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on brain component.
        
        Returns:
            Health status information
        """
        health_status = {
            "status": "healthy" if self._initialized else "unhealthy",
            "initialized": self._initialized,
            "capabilities_count": len(self.capabilities),
            "knowledge_base_size": len(self.knowledge_base),
            "metrics": self.metrics.copy()
        }
        
        # Check reasoning engines
        try:
            health_status["reasoning_engines"] = {
                "cot": await self.cot_reasoner.health_check(),
                "tot": await self.tot_reasoner.health_check(),
                "reflection": await self.reflection_engine.health_check(),
                "planning": await self.planning_engine.health_check()
            }
        except Exception as e:
            health_status["reasoning_engines_error"] = str(e)
        
        # Check LLM connectivity
        if self.llm:
            try:
                test_response = await self.llm.generate_response("Test", {})
                health_status["llm_status"] = "connected"
            except Exception as e:
                health_status["llm_status"] = f"error: {str(e)}"
        else:
            health_status["llm_status"] = "not_configured"
        
        return health_status
    
    # Private methods
    
    async def _load_knowledge_base(self) -> None:
        """Load knowledge base from persistent storage"""
        try:
            if self.memory:
                stored_knowledge = await self.memory.retrieve("brain_knowledge_base")
                if stored_knowledge:
                    self.knowledge_base.update(stored_knowledge)
                    self.logger.debug("Knowledge base loaded", entries=len(self.knowledge_base))
        except Exception as e:
            self.logger.warning("Failed to load knowledge base", error=str(e))
    
    async def _save_knowledge_base(self) -> None:
        """Save knowledge base to persistent storage"""
        try:
            if self.memory:
                await self.memory.store("brain_knowledge_base", self.knowledge_base)
                self.logger.debug("Knowledge base saved", entries=len(self.knowledge_base))
        except Exception as e:
            self.logger.warning("Failed to save knowledge base", error=str(e))
    
    async def _initialize_capabilities(self) -> None:
        """Initialize built-in capabilities"""
        built_in_capabilities = {
            "text_analysis": {
                "description": "Analyze and understand text content",
                "parameters": {"text": "string", "analysis_type": "string"}
            },
            "reasoning": {
                "description": "Apply logical reasoning to problems",
                "parameters": {"problem": "string", "reasoning_type": "string"}
            },
            "planning": {
                "description": "Create execution plans for complex tasks",
                "parameters": {"task": "object", "constraints": "array"}
            }
        }
        
        for name, config in built_in_capabilities.items():
            self.register_capability(name, config["description"], config["parameters"])
    
    async def _basic_message_analysis(self, message: str) -> Dict[str, Any]:
        """Perform basic message analysis without LLM"""
        analysis = {
            "length": len(message),
            "word_count": len(message.split()),
            "has_questions": "?" in message,
            "has_commands": any(word in message.lower() for word in ["please", "can you", "help", "do"]),
            "urgency_indicators": any(word in message.lower() for word in ["urgent", "asap", "immediately", "quickly"])
        }
        return analysis
    
    async def _determine_reasoning_type(self, message: str, basic_analysis: Dict[str, Any]) -> str:
        """Determine the appropriate reasoning type for a message"""
        # Complex multi-step problems
        if any(indicator in message.lower() for indicator in ["step by step", "analyze", "compare", "evaluate"]):
            return "chain_of_thought"
        
        # Problems with multiple possible solutions
        if any(indicator in message.lower() for indicator in ["options", "alternatives", "different ways", "explore"]):
            return "tree_of_thoughts"
        
        # Default to standard reasoning
        return "standard"
    
    async def _standard_analysis(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform standard message analysis"""
        # Basic intent classification
        intent = "unknown"
        if any(word in message.lower() for word in ["hello", "hi", "hey"]):
            intent = "greeting"
        elif any(word in message.lower() for word in ["help", "assist", "support"]):
            intent = "request_help"
        elif "?" in message:
            intent = "question"
        elif any(word in message.lower() for word in ["create", "make", "generate", "write"]):
            intent = "creation"
        
        # Tool requirement analysis
        requires_tools = any(indicator in message.lower() for indicator in [
            "search", "calculate", "file", "email", "schedule", "remind"
        ])
        
        # Suggested tools
        suggested_tools = []
        if "search" in message.lower() or "find" in message.lower():
            suggested_tools.append("web_search")
        if any(word in message.lower() for word in ["calculate", "compute", "math"]):
            suggested_tools.append("calculator")
        if "file" in message.lower():
            suggested_tools.append("file_manager")
        
        return {
            "intent": intent,
            "entities": [],
            "sentiment": 0.0,  # Neutral
            "confidence": 0.7,
            "complexity": "medium",
            "requires_tools": requires_tools,
            "suggested_tools": suggested_tools,
            "context_dependencies": []
        }
    
    async def _prepare_response_prompt(self, decision: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Prepare prompt for response generation"""
        base_prompt = f"""
        You are an intelligent AI assistant. Generate a helpful and accurate response.
        
        User Message: {context.get('original_message', '')}
        Intent: {decision.get('intent', 'unknown')}
        Context: {json.dumps(context.get('metadata', {}), indent=2)}
        
        Guidelines:
        - Be helpful, accurate, and concise
        - Use natural, conversational language
        - Provide specific information when possible
        - Ask for clarification if needed
        
        Response:
        """
        return base_prompt
    
    async def _post_process_response(self, response: str, decision: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Post-process generated response"""
        # Remove excessive whitespace
        processed = " ".join(response.split())
        
        # Ensure response ends appropriately
        if not processed.endswith(('.', '!', '?', ':')):
            processed += "."
        
        return processed
    
    async def _improve_response(self, response: str, reflection: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Improve response based on reflection feedback"""
        improvement_areas = reflection.get("improvement_areas", [])
        
        if "clarity" in improvement_areas:
            # Try to make response clearer
            improvement_prompt = f"""
            Improve the clarity of this response:
            Original: {response}
            Make it clearer and more understandable while maintaining accuracy.
            """
        elif "completeness" in improvement_areas:
            # Try to make response more complete
            improvement_prompt = f"""
            Make this response more complete:
            Original: {response}
            Add missing information or details that would be helpful.
            """
        else:
            # General improvement
            improvement_prompt = f"""
            Improve this response:
            Original: {response}
            Issues: {json.dumps(improvement_areas)}
            """
        
        try:
            if self.llm:
                improved = await self.llm.generate_response(improvement_prompt, context)
                return improved
        except Exception as e:
            self.logger.warning("Response improvement failed", error=str(e))
        
        return response  # Return original if improvement fails
    
    async def _estimate_execution_time(self, plan_data: Dict[str, Any]) -> float:
        """Estimate execution time for a plan"""
        steps = plan_data.get("steps", [])
        total_time = 0.0
        
        for step in steps:
            step_type = step.get("type", "unknown")
            # Base time estimates (in seconds)
            time_estimates = {
                "llm_call": 5.0,
                "tool_execution": 10.0,
                "data_processing": 15.0,
                "file_operation": 3.0,
                "web_request": 8.0,
                "unknown": 5.0
            }
            total_time += time_estimates.get(step_type, 5.0)
        
        # Add buffer time
        return total_time * 1.2
    
    async def _identify_required_tools(self, plan_data: Dict[str, Any]) -> List[str]:
        """Identify tools required for plan execution"""
        required_tools = set()
        steps = plan_data.get("steps", [])
        
        for step in steps:
            if step.get("requires_tool"):
                tool_name = step.get("tool_name")
                if tool_name:
                    required_tools.add(tool_name)
        
        return list(required_tools)
    
    async def _assess_risk_level(self, plan_data: Dict[str, Any]) -> str:
        """Assess risk level of plan execution"""
        risk_factors = 0
        steps = plan_data.get("steps", [])
        
        # Count risk factors
        for step in steps:
            if step.get("modifies_data", False):
                risk_factors += 2
            if step.get("external_dependency", False):
                risk_factors += 1
            if step.get("requires_permissions", False):
                risk_factors += 1
        
        # Determine risk level
        if risk_factors == 0:
            return "low"
        elif risk_factors <= 3:
            return "medium"
        else:
            return "high"
    
    async def _create_contingency_plans(self, plan_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create contingency plans for potential failures"""
        contingencies = []
        steps = plan_data.get("steps", [])
        
        for i, step in enumerate(steps):
            if step.get("critical", False):
                contingency = {
                    "trigger": f"step_{i}_failure",
                    "description": f"Fallback for step {i}: {step.get('description', 'Unknown')}",
                    "actions": [
                        {"type": "retry", "max_attempts": 3},
                        {"type": "alternative_approach", "description": "Try alternative method"},
                        {"type": "user_intervention", "description": "Request user assistance"}
                    ]
                }
                contingencies.append(contingency)
        
        return contingencies
    
    def _basic_tool_result_interpretation(self, result: ToolResult) -> str:
        """Basic tool result interpretation without LLM"""
        if isinstance(result.data, dict):
            if "count" in result.data:
                return f"Found {result.data['count']} results"
            elif "status" in result.data:
                return f"Operation completed with status: {result.data['status']}"
            elif "result" in result.data:
                return f"Result: {result.data['result']}"
        
        return f"Tool executed successfully. Result: {str(result.data)[:200]}"
    
    def _basic_result_formatting(self, result: Any) -> str:
        """Basic result formatting without LLM"""
        if isinstance(result, dict):
            formatted_lines = []
            for key, value in result.items():
                formatted_lines.append(f"{key}: {value}")
            return "\n".join(formatted_lines)
        elif isinstance(result, list):
            return "\n".join([f"- {item}" for item in result])
        else:
            return str(result)
    
    def _basic_error_response(self, error_context: Dict[str, Any]) -> str:
        """Basic error response without LLM"""
        error_type = error_context.get("error_type", "Error")
        
        if "timeout" in error_type.lower():
            return "I apologize, but the request timed out. Please try again with a simpler request."
        elif "permission" in error_type.lower():
            return "I don't have permission to perform that action. Please check if you have the necessary access."
        elif "not found" in error_type.lower():
            return "I couldn't find what you're looking for. Please check your request and try again."
        else:
            return "I encountered an error while processing your request. Please try rephrasing or contact support if the issue persists."
    
    async def _extract_lessons(self, experience: Dict[str, Any]) -> List[str]:
        """Extract lessons learned from experience"""
        lessons = []
        
        task = experience.get("task", {})
        result = experience.get("result", {})
        success = experience.get("success", True)
        
        if not success:
            error = result.get("error", "Unknown error")
            lessons.append(f"Task type '{task.get('type')}' failed due to: {error}")
        
        execution_time = result.get("execution_time", 0)
        if execution_time > 30:  # Long execution
            lessons.append(f"Task type '{task.get('type')}' takes longer than expected")
        
        return lessons
    
    def _update_average_response_time(self, new_time: float) -> None:
        """Update average response time with exponential moving average"""
        alpha = 0.1  # Smoothing factor
        if self.metrics["average_response_time"] == 0:
            self.metrics["average_response_time"] = new_time
        else:
            self.metrics["average_response_time"] = (
                alpha * new_time + (1 - alpha) * self.metrics["average_response_time"]
            )