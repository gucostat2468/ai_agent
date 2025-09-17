"""
Calculator Tool - Mathematical Operations
Provides safe mathematical calculations and expression evaluation.
"""

import ast
import operator
import math
from typing import Dict, List, Any, Optional, Union
import re

from ...interfaces.base import ToolInterface, ToolResult, ToolCapability
from ...monitoring.logger import StructuredLogger
from ...utils.exceptions import ToolExecutionException, ToolValidationException


class CalculatorTool(ToolInterface):
    """
    Safe mathematical calculator tool with support for basic arithmetic,
    trigonometric functions, and mathematical constants.
    """
    
    def __init__(self):
        super().__init__()
        self.logger = StructuredLogger(__name__)
        
        # Safe operators
        self.operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub, 
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.Mod: operator.mod,
            ast.FloorDiv: operator.floordiv,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }
        
        # Safe functions
        self.functions = {
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'pow': pow,
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'asin': math.asin,
            'acos': math.acos,
            'atan': math.atan,
            'sinh': math.sinh,
            'cosh': math.cosh,
            'tanh': math.tanh,
            'exp': math.exp,
            'log': math.log,
            'log10': math.log10,
            'log2': math.log2,
            'ceil': math.ceil,
            'floor': math.floor,
            'degrees': math.degrees,
            'radians': math.radians,
            'factorial': math.factorial,
        }
        
        # Mathematical constants
        self.constants = {
            'pi': math.pi,
            'e': math.e,
            'tau': math.tau,
            'inf': math.inf,
            'nan': math.nan,
        }
        
        self._initialized = False
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the calculator tool"""
        self.config = config
        self._initialized = True
        self.logger.info("Calculator tool initialized")
    
    async def cleanup(self) -> None:
        """Cleanup calculator resources"""
        self._initialized = False
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """
        Execute mathematical calculation
        
        Parameters:
            expression (str): Mathematical expression to evaluate
            precision (int): Number of decimal places for result (optional)
            format (str): Output format - 'number', 'string', 'scientific' (optional)
        """
        try:
            expression = parameters.get("expression", "")
            precision = parameters.get("precision", 10)
            output_format = parameters.get("format", "number")
            
            if not expression:
                return ToolResult(
                    success=False,
                    error="No expression provided",
                    tool_name="calculator"
                )
            
            # Clean and prepare expression
            cleaned_expression = self._clean_expression(expression)
            
            # Evaluate expression safely
            result = self._safe_eval(cleaned_expression)
            
            # Format result
            formatted_result = self._format_result(result, precision, output_format)
            
            return ToolResult(
                success=True,
                data={
                    "expression": expression,
                    "result": formatted_result,
                    "numeric_result": result,
                    "precision": precision,
                    "format": output_format
                },
                metadata={
                    "calculation_type": self._detect_calculation_type(expression),
                    "complexity": self._assess_complexity(expression)
                },
                tool_name="calculator"
            )
            
        except Exception as e:
            self.logger.error("Calculator execution failed", error=str(e), expression=parameters.get("expression", ""))
            return ToolResult(
                success=False,
                error=f"Calculation failed: {str(e)}",
                tool_name="calculator"
            )
    
    async def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate calculator parameters"""
        errors = []
        
        # Check required parameters
        if "expression" not in parameters:
            errors.append("Missing required parameter: expression")
        else:
            expression = parameters["expression"]
            if not isinstance(expression, str):
                errors.append("Expression must be a string")
            elif not expression.strip():
                errors.append("Expression cannot be empty")
            elif self._contains_unsafe_content(expression):
                errors.append("Expression contains unsafe content")
        
        # Check optional parameters
        if "precision" in parameters:
            precision = parameters["precision"]
            if not isinstance(precision, int) or precision < 0 or precision > 15:
                errors.append("Precision must be an integer between 0 and 15")
        
        if "format" in parameters:
            format_val = parameters["format"]
            if format_val not in ["number", "string", "scientific"]:
                errors.append("Format must be 'number', 'string', or 'scientific'")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    def get_capabilities(self) -> List[ToolCapability]:
        """Get calculator tool capabilities"""
        return [
            ToolCapability(
                name="calculator",
                description="Perform mathematical calculations and evaluate expressions safely",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate",
                            "examples": ["2 + 2", "sin(pi/2)", "sqrt(16)", "2**3 + 4*5"]
                        },
                        "precision": {
                            "type": "integer",
                            "description": "Number of decimal places for result",
                            "minimum": 0,
                            "maximum": 15,
                            "default": 10
                        },
                        "format": {
                            "type": "string",
                            "description": "Output format for the result",
                            "enum": ["number", "string", "scientific"],
                            "default": "number"
                        }
                    },
                    "required": ["expression"]
                },
                return_type="object",
                version="1.0.0",
                author="AI Agent Team",
                tags=["math", "calculation", "arithmetic", "trigonometry"],
                usage_examples=[
                    {
                        "description": "Basic arithmetic",
                        "parameters": {"expression": "15 + 25 * 2"},
                        "expected_result": 65
                    },
                    {
                        "description": "Trigonometric calculation",
                        "parameters": {"expression": "sin(pi/4)", "precision": 4},
                        "expected_result": 0.7071
                    },
                    {
                        "description": "Scientific notation",
                        "parameters": {"expression": "1e6 + 2e5", "format": "scientific"},
                        "expected_result": "1.2e+06"
                    }
                ]
            )
        ]
    
    def get_schema(self) -> Dict[str, Any]:
        """Get parameter schema"""
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate",
                    "minLength": 1,
                    "maxLength": 500
                },
                "precision": {
                    "type": "integer",
                    "description": "Number of decimal places",
                    "minimum": 0,
                    "maximum": 15,
                    "default": 10
                },
                "format": {
                    "type": "string",
                    "enum": ["number", "string", "scientific"],
                    "default": "number"
                }
            },
            "required": ["expression"],
            "additionalProperties": False
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for calculator tool"""
        try:
            # Test basic calculation
            test_result = self._safe_eval("2 + 2")
            
            return {
                "status": "healthy",
                "initialized": self._initialized,
                "test_calculation": "2 + 2 = 4",
                "test_passed": test_result == 4,
                "supported_functions": len(self.functions),
                "supported_constants": len(self.constants)
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "initialized": self._initialized
            }
    
    def _clean_expression(self, expression: str) -> str:
        """Clean and prepare mathematical expression"""
        # Remove whitespace
        cleaned = expression.strip()
        
        # Replace common text representations
        replacements = {
            '^': '**',  # Power operator
            'π': 'pi',  # Pi constant
            '÷': '/',   # Division
            '×': '*',   # Multiplication
        }
        
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        
        return cleaned
    
    def _safe_eval(self, expression: str) -> Union[int, float]:
        """Safely evaluate mathematical expression using AST"""
        try:
            # Parse expression into AST
            tree = ast.parse(expression, mode='eval')
            
            # Evaluate AST safely
            return self._eval_ast(tree.body)
            
        except (SyntaxError, ValueError) as e:
            raise ToolExecutionException(f"Invalid mathematical expression: {str(e)}")
    
    def _eval_ast(self, node):
        """Recursively evaluate AST nodes safely"""
        
        if isinstance(node, ast.Constant):  # Numbers
            return node.value
        
        elif isinstance(node, ast.Num):  # Numbers (Python < 3.8)
            return node.n
        
        elif isinstance(node, ast.Name):  # Variables/Constants
            if node.id in self.constants:
                return self.constants[node.id]
            else:
                raise ToolExecutionException(f"Unknown variable: {node.id}")
        
        elif isinstance(node, ast.BinOp):  # Binary operations
            left = self._eval_ast(node.left)
            right = self._eval_ast(node.right)
            op_type = type(node.op)
            
            if op_type in self.operators:
                try:
                    result = self.operators[op_type](left, right)
                    
                    # Check for division by zero
                    if math.isnan(result) or math.isinf(result):
                        raise ToolExecutionException("Mathematical error: division by zero or invalid operation")
                    
                    return result
                except ZeroDivisionError:
                    raise ToolExecutionException("Division by zero")
                except (OverflowError, ValueError) as e:
                    raise ToolExecutionException(f"Mathematical error: {str(e)}")
            else:
                raise ToolExecutionException(f"Unsupported operation: {op_type.__name__}")
        
        elif isinstance(node, ast.UnaryOp):  # Unary operations
            operand = self._eval_ast(node.operand)
            op_type = type(node.op)
            
            if op_type in self.operators:
                return self.operators[op_type](operand)
            else:
                raise ToolExecutionException(f"Unsupported unary operation: {op_type.__name__}")
        
        elif isinstance(node, ast.Call):  # Function calls
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                
                if func_name in self.functions:
                    # Evaluate arguments
                    args = [self._eval_ast(arg) for arg in node.args]
                    
                    try:
                        return self.functions[func_name](*args)
                    except (ValueError, TypeError, OverflowError) as e:
                        raise ToolExecutionException(f"Error in function {func_name}: {str(e)}")
                else:
                    raise ToolExecutionException(f"Unknown function: {func_name}")
            else:
                raise ToolExecutionException("Complex function calls not supported")
        
        elif isinstance(node, ast.List):  # Lists (for functions like min, max, sum)
            return [self._eval_ast(item) for item in node.elts]
        
        else:
            raise ToolExecutionException(f"Unsupported expression type: {type(node).__name__}")
    
    def _format_result(self, result: Union[int, float], precision: int, format_type: str) -> Any:
        """Format calculation result according to specified format"""
        
        if format_type == "number":
            if isinstance(result, int):
                return result
            else:
                return round(result, precision)
        
        elif format_type == "string":
            if isinstance(result, int):
                return str(result)
            else:
                return f"{result:.{precision}f}"
        
        elif format_type == "scientific":
            return f"{result:.{precision}e}"
        
        else:
            return result
    
    def _detect_calculation_type(self, expression: str) -> str:
        """Detect the type of calculation being performed"""
        expression_lower = expression.lower()
        
        if any(func in expression_lower for func in ['sin', 'cos', 'tan', 'asin', 'acos', 'atan']):
            return "trigonometric"
        elif any(func in expression_lower for func in ['log', 'exp', 'sqrt', 'pow']):
            return "logarithmic_exponential"
        elif any(func in expression_lower for func in ['factorial']):
            return "combinatorial"
        elif any(op in expression for op in ['+', '-', '*', '/', '**']):
            return "arithmetic"
        else:
            return "basic"
    
    def _assess_complexity(self, expression: str) -> str:
        """Assess the complexity of the mathematical expression"""
        
        # Count operations and functions
        operation_count = len(re.findall(r'[+\-*/^()]', expression))
        function_count = len(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*\s*\(', expression))
        
        total_complexity = operation_count + function_count * 2
        
        if total_complexity <= 2:
            return "simple"
        elif total_complexity <= 5:
            return "moderate"
        elif total_complexity <= 10:
            return "complex"
        else:
            return "very_complex"
    
    def _contains_unsafe_content(self, expression: str) -> bool:
        """Check if expression contains potentially unsafe content"""
        
        # Check for potentially dangerous keywords
        dangerous_keywords = [
            'import', 'exec', 'eval', 'open', 'file', 'input', 'raw_input',
            '__', 'getattr', 'setattr', 'delattr', 'globals', 'locals', 'vars'
        ]
        
        expression_lower = expression.lower()
        
        for keyword in dangerous_keywords:
            if keyword in expression_lower:
                return True
        
        # Check for excessive length
        if len(expression) > 500:
            return True
        
        # Check for excessive nesting
        if expression.count('(') > 20 or expression.count('[') > 10:
            return True
        
        return False


# Utility functions for mathematical operations

def calculate_statistics(numbers: List[Union[int, float]]) -> Dict[str, float]:
    """Calculate basic statistics for a list of numbers"""
    if not numbers:
        return {}
    
    sorted_numbers = sorted(numbers)
    n = len(numbers)
    
    mean = sum(numbers) / n
    
    # Median
    if n % 2 == 0:
        median = (sorted_numbers[n//2 - 1] + sorted_numbers[n//2]) / 2
    else:
        median = sorted_numbers[n//2]
    
    # Variance and standard deviation
    variance = sum((x - mean) ** 2 for x in numbers) / n
    std_dev = math.sqrt(variance)
    
    return {
        "count": n,
        "sum": sum(numbers),
        "mean": mean,
        "median": median,
        "min": min(numbers),
        "max": max(numbers),
        "range": max(numbers) - min(numbers),
        "variance": variance,
        "standard_deviation": std_dev
    }


def solve_quadratic(a: float, b: float, c: float) -> Dict[str, Any]:
    """Solve quadratic equation ax² + bx + c = 0"""
    
    if a == 0:
        if b == 0:
            return {"error": "Not a valid equation (both a and b are zero)"}
        else:
            # Linear equation
            return {"type": "linear", "solution": -c / b}
    
    # Calculate discriminant
    discriminant = b**2 - 4*a*c
    
    if discriminant > 0:
        # Two real solutions
        sqrt_discriminant = math.sqrt(discriminant)
        x1 = (-b + sqrt_discriminant) / (2*a)
        x2 = (-b - sqrt_discriminant) / (2*a)
        return {
            "type": "two_real_solutions",
            "solutions": [x1, x2],
            "discriminant": discriminant
        }
    
    elif discriminant == 0:
        # One real solution
        x = -b / (2*a)
        return {
            "type": "one_real_solution",
            "solution": x,
            "discriminant": discriminant
        }
    
    else:
        # Complex solutions
        real_part = -b / (2*a)
        imaginary_part = math.sqrt(-discriminant) / (2*a)
        return {
            "type": "complex_solutions",
            "solutions": [
                {"real": real_part, "imaginary": imaginary_part},
                {"real": real_part, "imaginary": -imaginary_part}
            ],
            "discriminant": discriminant
        }


def convert_units(value: float, from_unit: str, to_unit: str, unit_type: str) -> Dict[str, Any]:
    """Convert between different units"""
    
    # Length conversions (to meters)
    length_conversions = {
        "mm": 0.001,
        "cm": 0.01,
        "m": 1.0,
        "km": 1000.0,
        "in": 0.0254,
        "ft": 0.3048,
        "yd": 0.9144,
        "mi": 1609.34
    }
    
    # Weight conversions (to grams)
    weight_conversions = {
        "mg": 0.001,
        "g": 1.0,
        "kg": 1000.0,
        "lb": 453.592,
        "oz": 28.3495
    }
    
    # Temperature conversions
    def convert_temperature(value, from_unit, to_unit):
        # Convert to Celsius first
        if from_unit == "F":
            celsius = (value - 32) * 5/9
        elif from_unit == "K":
            celsius = value - 273.15
        else:  # Already Celsius
            celsius = value
        
        # Convert from Celsius to target
        if to_unit == "F":
            return celsius * 9/5 + 32
        elif to_unit == "K":
            return celsius + 273.15
        else:  # Already Celsius
            return celsius
    
    try:
        if unit_type == "length":
            if from_unit not in length_conversions or to_unit not in length_conversions:
                return {"error": f"Unsupported length units: {from_unit} or {to_unit}"}
            
            meters = value * length_conversions[from_unit]
            result = meters / length_conversions[to_unit]
            
        elif unit_type == "weight":
            if from_unit not in weight_conversions or to_unit not in weight_conversions:
                return {"error": f"Unsupported weight units: {from_unit} or {to_unit}"}
            
            grams = value * weight_conversions[from_unit]
            result = grams / weight_conversions[to_unit]
            
        elif unit_type == "temperature":
            if from_unit not in ["C", "F", "K"] or to_unit not in ["C", "F", "K"]:
                return {"error": f"Unsupported temperature units: {from_unit} or {to_unit}"}
            
            result = convert_temperature(value, from_unit, to_unit)
            
        else:
            return {"error": f"Unsupported unit type: {unit_type}"}
        
        return {
            "original_value": value,
            "original_unit": from_unit,
            "converted_value": result,
            "converted_unit": to_unit,
            "conversion_type": unit_type
        }
    
    except Exception as e:
        return {"error": f"Conversion failed: {str(e)}"}