"""
Decision Engine - Intelligent Decision Making System
Handles decision making with confidence scoring, risk assessment, and learning.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import asyncio
import logging
import json
import math

from ..interfaces.base import BaseComponent
from ..config.settings import DecisionConfig
from ..monitoring.logger import StructuredLogger
from ..utils.exceptions import DecisionException, DecisionTimeoutException, InsufficientInformationException


class DecisionType(Enum):
    """Types of decisions the engine can make"""
    RESPOND = "respond"
    USE_TOOL = "use_tool"
    DELEGATE_TASK = "delegate_task"
    REQUEST_CLARIFICATION = "request_clarification"
    ESCALATE = "escalate"
    DEFER = "defer"


class ConfidenceLevel(Enum):
    """Confidence levels for decisions"""
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9


class RiskLevel(Enum):
    """Risk levels for decisions"""
    MINIMAL = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    CRITICAL = 0.9


@dataclass
class DecisionContext:
    """Context information for decision making"""
    user_message: str = ""
    intent: str = "unknown"
    entities: List[Dict[str, Any]] = field(default_factory=list)
    sentiment: float = 0.0
    complexity: str = "medium"
    available_tools: List[str] = field(default_factory=list)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    system_state: Dict[str, Any] = field(default_factory=dict)
    time_constraints: Optional[float] = None
    resource_constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    """Represents a decision made by the engine"""
    decision_id: str
    action_type: DecisionType
    confidence: float
    risk_level: float
    reasoning: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    execution_plan: List[Dict[str, Any]] = field(default_factory=list)
    expected_outcome: str = ""
    fallback_options: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to dictionary"""
        return {
            "decision_id": self.decision_id,
            "action_type": self.action_type.value,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "reasoning": self.reasoning,
            "parameters": self.parameters,
            "alternatives": self.alternatives,
            "execution_plan": self.execution_plan,
            "expected_outcome": self.expected_outcome,
            "fallback_options": self.fallback_options,
            "timestamp": self.timestamp.isoformat()
        }


class DecisionStrategy(ABC):
    """Abstract base class for decision strategies"""
    
    @abstractmethod
    async def evaluate(self, context: DecisionContext) -> Tuple[DecisionType, float, Dict[str, Any]]:
        """
        Evaluate context and return decision type, confidence, and parameters
        
        Returns:
            Tuple of (decision_type, confidence, parameters)
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get strategy name"""
        pass
    
    @abstractmethod
    def get_weight(self) -> float:
        """Get strategy weight in ensemble"""
        pass


class SimpleDecisionStrategy(DecisionStrategy):
    """Simple rule-based decision strategy"""
    
    def __init__(self, weight: float = 0.3):
        self.weight = weight
        self.logger = StructuredLogger(__name__)
    
    async def evaluate(self, context: DecisionContext) -> Tuple[DecisionType, float, Dict[str, Any]]:
        """Simple rule-based evaluation"""
        
        # Check for obvious tool usage patterns
        if any(tool_word in context.user_message.lower() for tool_word in 
               ["calculate", "search", "file", "email", "schedule"]):
            
            # Determine which tool to use
            tool_name = self._identify_tool(context.user_message, context.available_tools)
            if tool_name:
                return DecisionType.USE_TOOL, 0.8, {"tool_name": tool_name}
        
        # Check for clarification needs
        if context.intent == "unclear" or context.complexity == "high":
            return DecisionType.REQUEST_CLARIFICATION, 0.7, {
                "clarification_prompt": "I need more information to help you properly."
            }
        
        # Check for task delegation
        if "complex" in context.user_message.lower() or "multiple steps" in context.user_message.lower():
            return DecisionType.DELEGATE_TASK, 0.6, {
                "task": {"type": "complex", "description": context.user_message}
            }
        
        # Default to respond
        return DecisionType.RESPOND, 0.5, {"response_type": "conversational"}
    
    def get_name(self) -> str:
        return "simple"
    
    def get_weight(self) -> float:
        return self.weight
    
    def _identify_tool(self, message: str, available_tools: List[str]) -> Optional[str]:
        """Identify which tool to use based on message content"""
        message_lower = message.lower()
        
        tool_keywords = {
            "calculator": ["calculate", "compute", "math", "arithmetic"],
            "web_search": ["search", "find", "look up", "google"],
            "file_manager": ["file", "document", "save", "load"],
            "email": ["email", "send", "mail"],
            "scheduler": ["schedule", "calendar", "appointment", "meeting"]
        }
        
        for tool, keywords in tool_keywords.items():
            if tool in available_tools and any(keyword in message_lower for keyword in keywords):
                return tool
        
        return None


class ContextualDecisionStrategy(DecisionStrategy):
    """Context-aware decision strategy using conversation history"""
    
    def __init__(self, weight: float = 0.4):
        self.weight = weight
        self.logger = StructuredLogger(__name__)
    
    async def evaluate(self, context: DecisionContext) -> Tuple[DecisionType, float, Dict[str, Any]]:
        """Context-aware evaluation"""
        
        confidence = 0.5
        decision_type = DecisionType.RESPOND
        parameters = {}
        
        # Analyze conversation flow
        if context.conversation_history:
            last_interactions = context.conversation_history[-3:]  # Last 3 interactions
            
            # Check if user is asking follow-up questions
            if self._is_followup_question(context.user_message, last_interactions):
                confidence += 0.2
                parameters["context_aware"] = True
                parameters["reference_previous"] = True
            
            # Check for repeated requests (might need escalation)
            if self._is_repeated_request(context.user_message, last_interactions):
                decision_type = DecisionType.ESCALATE
                confidence = 0.7
                parameters["escalation_reason"] = "repeated_request"
        
        # Consider user preferences
        if context.user_preferences:
            # Adjust based on user's preferred interaction style
            preferred_style = context.user_preferences.get("interaction_style", "balanced")
            
            if preferred_style == "detailed":
                parameters["response_length"] = "detailed"
                confidence += 0.1
            elif preferred_style == "concise":
                parameters["response_length"] = "brief"
                confidence += 0.1
        
        # Consider sentiment
        if context.sentiment < -0.5:  # Negative sentiment
            parameters["empathetic_response"] = True
            confidence += 0.1
        elif context.sentiment > 0.5:  # Positive sentiment
            parameters["enthusiastic_response"] = True
            confidence += 0.1
        
        return decision_type, confidence, parameters
    
    def get_name(self) -> str:
        return "contextual"
    
    def get_weight(self) -> float:
        return self.weight
    
    def _is_followup_question(self, message: str, history: List[Dict[str, Any]]) -> bool:
        """Check if message is a follow-up to previous interaction"""
        followup_indicators = ["also", "additionally", "furthermore", "what about", "how about"]
        message_lower = message.lower()
        
        return any(indicator in message_lower for indicator in followup_indicators)
    
    def _is_repeated_request(self, message: str, history: List[Dict[str, Any]]) -> bool:
        """Check if user is repeating a similar request"""
        if not history:
            return False
        
        # Simple similarity check (in production, use more sophisticated methods)
        for interaction in history:
            if isinstance(interaction, dict) and "user_message" in interaction:
                previous_message = interaction["user_message"].lower()
                current_message = message.lower()
                
                # Basic similarity check
                common_words = set(previous_message.split()) & set(current_message.split())
                if len(common_words) > 3:  # Arbitrary threshold
                    return True
        
        return False


class LearningBasedDecisionStrategy(DecisionStrategy):
    """Learning-based decision strategy that improves over time"""
    
    def __init__(self, weight: float = 0.3):
        self.weight = weight
        self.logger = StructuredLogger(__name__)
        self.decision_history: List[Dict[str, Any]] = []
        self.success_rates: Dict[str, float] = {}
    
    async def evaluate(self, context: DecisionContext) -> Tuple[DecisionType, float, Dict[str, Any]]:
        """Learning-based evaluation using historical success rates"""
        
        # Get success rates for different decision types in similar contexts
        context_key = self._get_context_key(context)
        
        best_decision = DecisionType.RESPOND
        best_confidence = 0.5
        parameters = {}
        
        # Check success rates for each decision type
        for decision_type in DecisionType:
            success_rate = self.success_rates.get(f"{context_key}:{decision_type.value}", 0.5)
            
            # Adjust confidence based on historical success
            adjusted_confidence = min(success_rate * 1.2, 0.95)  # Cap at 95%
            
            if adjusted_confidence > best_confidence:
                best_confidence = adjusted_confidence
                best_decision = decision_type
                parameters["learning_based"] = True
                parameters["historical_success_rate"] = success_rate
        
        return best_decision, best_confidence, parameters
    
    def get_name(self) -> str:
        return "learning_based"
    
    def get_weight(self) -> float:
        return self.weight
    
    def _get_context_key(self, context: DecisionContext) -> str:
        """Generate a key representing the context for learning purposes"""
        # Simplified context key generation
        key_components = [
            context.intent,
            context.complexity,
            "positive" if context.sentiment > 0 else "negative" if context.sentiment < 0 else "neutral"
        ]
        return ":".join(key_components)
    
    def record_outcome(self, decision: Decision, success: bool) -> None:
        """Record the outcome of a decision for learning"""
        context_key = f"{decision.decision_id}:{decision.action_type.value}"
        
        # Update success rate using exponential moving average
        current_rate = self.success_rates.get(context_key, 0.5)
        learning_rate = 0.1
        
        new_rate = current_rate * (1 - learning_rate) + (1.0 if success else 0.0) * learning_rate
        self.success_rates[context_key] = new_rate


class DecisionEngine(BaseComponent):
    """
    Main decision engine that coordinates multiple strategies to make intelligent decisions
    """
    
    def __init__(self, config: DecisionConfig):
        super().__init__(config.dict() if hasattr(config, 'dict') else config.__dict__)
        self.config = config
        self.logger = StructuredLogger(__name__)
        
        # Decision strategies
        self.strategies: List[DecisionStrategy] = []
        
        # Dependencies (set by agent)
        self.memory = None
        self.task_coordinator = None
        
        # Decision history for learning
        self.decision_history: List[Decision] = []
        self.performance_metrics = {
            "total_decisions": 0,
            "successful_decisions": 0,
            "average_confidence": 0.0,
            "decision_types_count": {}
        }
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the decision engine"""
        try:
            self.logger.info("Initializing Decision Engine")
            
            # Initialize strategies based on configuration
            strategy_configs = self.config.strategies
            
            if "simple" in strategy_configs:
                weight = strategy_configs["simple"].get("weight", 0.3)
                self.strategies.append(SimpleDecisionStrategy(weight))
            
            if "contextual" in strategy_configs:
                weight = strategy_configs["contextual"].get("weight", 0.4)
                self.strategies.append(ContextualDecisionStrategy(weight))
            
            if "learning_based" in strategy_configs:
                weight = strategy_configs["learning_based"].get("weight", 0.3)
                self.strategies.append(LearningBasedDecisionStrategy(weight))
            
            self._initialized = True
            self.logger.info("Decision Engine initialized successfully", 
                           strategies_count=len(self.strategies))
            
        except Exception as e:
            self.logger.error("Failed to initialize Decision Engine", error=str(e))
            raise DecisionException(f"Decision engine initialization failed: {e}") from e
    
    async def cleanup(self) -> None:
        """Cleanup decision engine resources"""
        try:
            self.logger.info("Cleaning up Decision Engine")
            
            # Save learning data if needed
            await self._save_learning_data()
            
            self._initialized = False
            self.logger.info("Decision Engine cleaned up successfully")
            
        except Exception as e:
            self.logger.error("Error during Decision Engine cleanup", error=str(e))
    
    def set_memory(self, memory) -> None:
        """Set memory system reference"""
        self.memory = memory
    
    def set_task_coordinator(self, task_coordinator) -> None:
        """Set task coordinator reference"""
        self.task_coordinator = task_coordinator
    
    async def decide(self, analysis: Dict[str, Any]) -> Decision:
        """
        Make a decision based on message analysis
        
        Args:
            analysis: Message analysis results from brain
            
        Returns:
            Decision object with action type and parameters
        """
        if not self._initialized:
            raise DecisionException("Decision engine not initialized")
        
        try:
            decision_id = f"decision_{len(self.decision_history)}"
            
            self.logger.debug("Making decision", 
                            decision_id=decision_id,
                            intent=analysis.get("intent"))
            
            # Create decision context
            context = self._create_decision_context(analysis)
            
            # Apply timeout if configured
            timeout = self.config.decision_timeout
            
            if timeout > 0:
                decision = await asyncio.wait_for(
                    self._make_decision(decision_id, context),
                    timeout=timeout
                )
            else:
                decision = await self._make_decision(decision_id, context)
            
            # Record decision
            self.decision_history.append(decision)
            self._update_metrics(decision)
            
            self.logger.info("Decision made", 
                           decision_id=decision_id,
                           action_type=decision.action_type.value,
                           confidence=decision.confidence,
                           risk_level=decision.risk_level)
            
            return decision
            
        except asyncio.TimeoutError:
            self.logger.error("Decision timeout", timeout=timeout)
            raise DecisionTimeoutException("Decision process timed out", timeout)
        except Exception as e:
            self.logger.error("Decision making failed", error=str(e))
            raise DecisionException(f"Decision making failed: {e}") from e
    
    async def _make_decision(self, decision_id: str, context: DecisionContext) -> Decision:
        """Internal decision making process"""
        
        # Check for insufficient information
        if not self._has_sufficient_information(context):
            missing_info = self._identify_missing_information(context)
            raise InsufficientInformationException(
                "Insufficient information for decision making",
                missing_info
            )
        
        # Evaluate using all strategies
        strategy_results = []
        
        for strategy in self.strategies:
            try:
                decision_type, confidence, parameters = await strategy.evaluate(context)
                strategy_results.append({
                    "strategy": strategy.get_name(),
                    "weight": strategy.get_weight(),
                    "decision_type": decision_type,
                    "confidence": confidence,
                    "parameters": parameters
                })
            except Exception as e:
                self.logger.warning(f"Strategy {strategy.get_name()} failed", error=str(e))
        
        if not strategy_results:
            # Fallback decision
            return self._create_fallback_decision(decision_id, context)
        
        # Combine strategy results
        final_decision_type, final_confidence, final_parameters = self._combine_strategies(strategy_results)
        
        # Assess risk
        risk_level = self._assess_risk(final_decision_type, final_parameters, context)
        
        # Apply risk mitigation if necessary
        if risk_level > self.config.risk_tolerance:
            final_decision_type, final_parameters = self._apply_risk_mitigation(
                final_decision_type, final_parameters, risk_level
            )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(strategy_results, final_decision_type, final_confidence)
        
        # Create decision
        decision = Decision(
            decision_id=decision_id,
            action_type=final_decision_type,
            confidence=final_confidence,
            risk_level=risk_level,
            reasoning=reasoning,
            parameters=final_parameters,
            alternatives=self._generate_alternatives(strategy_results, final_decision_type),
            execution_plan=self._create_execution_plan(final_decision_type, final_parameters),
            expected_outcome=self._predict_outcome(final_decision_type, final_parameters),
            fallback_options=self._identify_fallback_options(final_decision_type)
        )
        
        return decision
    
    def _create_decision_context(self, analysis: Dict[str, Any]) -> DecisionContext:
        """Create decision context from analysis"""
        return DecisionContext(
            user_message=analysis.get("original_message", ""),
            intent=analysis.get("intent", "unknown"),
            entities=analysis.get("entities", []),
            sentiment=analysis.get("sentiment", 0.0),
            complexity=analysis.get("complexity", "medium"),
            available_tools=analysis.get("suggested_tools", []),
            conversation_history=analysis.get("conversation_history", []),
            user_preferences=analysis.get("user_preferences", {}),
            system_state=analysis.get("agent_state", {})
        )
    
    def _has_sufficient_information(self, context: DecisionContext) -> bool:
        """Check if we have sufficient information to make a decision"""
        # Basic checks
        if not context.user_message.strip():
            return False
        
        if context.intent == "unknown" and context.complexity == "high":
            return False
        
        return True
    
    def _identify_missing_information(self, context: DecisionContext) -> List[str]:
        """Identify what information is missing"""
        missing = []
        
        if not context.user_message.strip():
            missing.append("user_message")
        
        if context.intent == "unknown":
            missing.append("intent")
        
        if not context.available_tools:
            missing.append("available_tools")
        
        return missing
    
    def _combine_strategies(self, results: List[Dict[str, Any]]) -> Tuple[DecisionType, float, Dict[str, Any]]:
        """Combine results from multiple strategies"""
        
        # Weighted voting for decision type
        decision_votes = {}
        total_weight = 0
        
        for result in results:
            decision_type = result["decision_type"]
            weight = result["weight"]
            confidence = result["confidence"]
            
            # Vote strength = weight * confidence
            vote_strength = weight * confidence
            
            if decision_type not in decision_votes:
                decision_votes[decision_type] = {
                    "vote_strength": 0,
                    "parameters": {},
                    "confidence_sum": 0,
                    "weight_sum": 0
                }
            
            decision_votes[decision_type]["vote_strength"] += vote_strength
            decision_votes[decision_type]["confidence_sum"] += confidence * weight
            decision_votes[decision_type]["weight_sum"] += weight
            
            # Merge parameters
            decision_votes[decision_type]["parameters"].update(result["parameters"])
            
            total_weight += weight
        
        # Select decision type with highest vote strength
        best_decision = max(decision_votes.items(), key=lambda x: x[1]["vote_strength"])
        decision_type = best_decision[0]
        decision_data = best_decision[1]
        
        # Calculate weighted average confidence
        final_confidence = decision_data["confidence_sum"] / decision_data["weight_sum"] if decision_data["weight_sum"] > 0 else 0.5
        
        # Ensure confidence is within bounds
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        return decision_type, final_confidence, decision_data["parameters"]
    
    def _assess_risk(self, decision_type: DecisionType, parameters: Dict[str, Any], context: DecisionContext) -> float:
        """Assess risk level of a decision"""
        base_risk = {
            DecisionType.RESPOND: 0.1,
            DecisionType.USE_TOOL: 0.3,
            DecisionType.DELEGATE_TASK: 0.4,
            DecisionType.REQUEST_CLARIFICATION: 0.1,
            DecisionType.ESCALATE: 0.6,
            DecisionType.DEFER: 0.2
        }
        
        risk = base_risk.get(decision_type, 0.3)
        
        # Risk modifiers
        if parameters.get("modifies_data", False):
            risk += 0.2
        
        if parameters.get("external_dependency", False):
            risk += 0.1
        
        if context.complexity == "high":
            risk += 0.1
        
        if context.sentiment < -0.5:  # User is frustrated
            risk += 0.1
        
        return min(risk, 1.0)
    
    def _apply_risk_mitigation(self, decision_type: DecisionType, parameters: Dict[str, Any], risk_level: float) -> Tuple[DecisionType, Dict[str, Any]]:
        """Apply risk mitigation strategies"""
        
        if self.config.fallback_strategy == "conservative":
            # Conservative approach - prefer safer options
            if decision_type == DecisionType.USE_TOOL:
                return DecisionType.REQUEST_CLARIFICATION, {
                    "clarification_prompt": "I want to make sure I understand correctly before proceeding."
                }
            elif decision_type == DecisionType.DELEGATE_TASK:
                return DecisionType.RESPOND, {"response_type": "explanation"}
        
        elif self.config.fallback_strategy == "request_clarification":
            # Always request clarification for high-risk decisions
            return DecisionType.REQUEST_CLARIFICATION, {
                "clarification_prompt": "This seems important. Can you provide more details to ensure I help you correctly?"
            }
        
        # Add safety parameters
        parameters["safety_check"] = True
        parameters["risk_level"] = risk_level
        
        return decision_type, parameters
    
    def _generate_reasoning(self, strategy_results: List[Dict[str, Any]], final_decision: DecisionType, confidence: float) -> str:
        """Generate human-readable reasoning for the decision"""
        
        reasoning_parts = [
            f"Decision: {final_decision.value} (confidence: {confidence:.2f})"
        ]
        
        # Add strategy contributions
        strategy_summaries = []
        for result in strategy_results:
            if result["decision_type"] == final_decision:
                strategy_summaries.append(f"{result['strategy']} strategy supported this decision")
        
        if strategy_summaries:
            reasoning_parts.append("Supporting strategies: " + ", ".join(strategy_summaries))
        
        # Add key factors
        key_factors = []
        for result in strategy_results:
            if "learning_based" in result.get("parameters", {}):
                key_factors.append("historical success rate")
            if "context_aware" in result.get("parameters", {}):
                key_factors.append("conversation context")
        
        if key_factors:
            reasoning_parts.append("Key factors: " + ", ".join(key_factors))
        
        return ". ".join(reasoning_parts)
    
    def _generate_alternatives(self, strategy_results: List[Dict[str, Any]], chosen_decision: DecisionType) -> List[Dict[str, Any]]:
        """Generate alternative decisions that were considered"""
        alternatives = []
        
        for result in strategy_results:
            if result["decision_type"] != chosen_decision:
                alternatives.append({
                    "decision_type": result["decision_type"].value,
                    "confidence": result["confidence"],
                    "strategy": result["strategy"],
                    "parameters": result["parameters"]
                })
        
        # Sort by confidence
        alternatives.sort(key=lambda x: x["confidence"], reverse=True)
        
        return alternatives[:3]  # Return top 3 alternatives
    
    def _create_execution_plan(self, decision_type: DecisionType, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create execution plan for the decision"""
        
        plan = []
        
        if decision_type == DecisionType.USE_TOOL:
            tool_name = parameters.get("tool_name", "unknown")
            plan = [
                {"step": "validate_tool", "tool": tool_name},
                {"step": "prepare_parameters", "tool": tool_name},
                {"step": "execute_tool", "tool": tool_name},
                {"step": "interpret_results", "tool": tool_name}
            ]
        
        elif decision_type == DecisionType.DELEGATE_TASK:
            plan = [
                {"step": "analyze_task", "task": parameters.get("task", {})},
                {"step": "create_subtasks"},
                {"step": "execute_subtasks"},
                {"step": "combine_results"}
            ]
        
        elif decision_type == DecisionType.RESPOND:
            plan = [
                {"step": "prepare_context"},
                {"step": "generate_response"},
                {"step": "validate_response"},
                {"step": "deliver_response"}
            ]
        
        else:
            plan = [{"step": "execute_decision", "type": decision_type.value}]
        
        return plan
    
    def _predict_outcome(self, decision_type: DecisionType, parameters: Dict[str, Any]) -> str:
        """Predict the expected outcome of the decision"""
        
        outcomes = {
            DecisionType.RESPOND: "User receives a helpful response",
            DecisionType.USE_TOOL: f"Tool {parameters.get('tool_name', 'unknown')} will be executed to help the user",
            DecisionType.DELEGATE_TASK: "Complex task will be broken down and executed systematically",
            DecisionType.REQUEST_CLARIFICATION: "User will provide additional information for better assistance",
            DecisionType.ESCALATE: "Issue will be escalated for specialized handling",
            DecisionType.DEFER: "Request will be deferred until more suitable conditions"
        }
        
        return outcomes.get(decision_type, "Action will be taken as determined")
    
    def _identify_fallback_options(self, decision_type: DecisionType) -> List[str]:
        """Identify fallback options if the decision fails"""
        
        fallbacks = {
            DecisionType.USE_TOOL: ["respond_with_explanation", "request_clarification"],
            DecisionType.DELEGATE_TASK: ["respond_with_guidance", "break_down_manually"],
            DecisionType.RESPOND: ["request_clarification", "provide_general_help"],
            DecisionType.REQUEST_CLARIFICATION: ["provide_general_guidance"],
            DecisionType.ESCALATE: ["respond_with_limitation_explanation"],
            DecisionType.DEFER: ["respond_with_explanation", "schedule_retry"]
        }
        
        return fallbacks.get(decision_type, ["apologize_and_retry"])
    
    def _create_fallback_decision(self, decision_id: str, context: DecisionContext) -> Decision:
        """Create a safe fallback decision when strategies fail"""
        
        return Decision(
            decision_id=decision_id,
            action_type=DecisionType.RESPOND,
            confidence=0.3,
            risk_level=0.1,
            reasoning="Fallback decision due to strategy evaluation failures",
            parameters={"response_type": "safe_fallback"},
            expected_outcome="Provide safe, general assistance"
        )
    
    def record_decision_outcome(self, decision_id: str, success: bool, feedback: Dict[str, Any] = None) -> None:
        """Record the outcome of a decision for learning"""
        
        # Find the decision
        decision = next((d for d in self.decision_history if d.decision_id == decision_id), None)
        
        if decision:
            # Update performance metrics
            self.performance_metrics["total_decisions"] += 1
            if success:
                self.performance_metrics["successful_decisions"] += 1
            
            # Update decision type counts
            decision_type_str = decision.action_type.value
            if decision_type_str not in self.performance_metrics["decision_types_count"]:
                self.performance_metrics["decision_types_count"][decision_type_str] = {"total": 0, "successful": 0}
            
            self.performance_metrics["decision_types_count"][decision_type_str]["total"] += 1
            if success:
                self.performance_metrics["decision_types_count"][decision_type_str]["successful"] += 1