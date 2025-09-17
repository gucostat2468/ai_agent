"""
Built-in Tools for AI Agent
Collection of essential tools that come pre-installed with the AI Agent.
"""

from .web_tools import WebScraperTool, HTTPClientTool
from .file_tools import FileReaderTool, FileWriterTool, FileManagerTool
from .system_tools import ShellExecutorTool, ProcessManagerTool
from .calculator import CalculatorTool
from .text_tools import TextProcessorTool, DocumentAnalyzerTool

__all__ = [
    "WebScraperTool",
    "HTTPClientTool", 
    "FileReaderTool",
    "FileWriterTool",
    "FileManagerTool",
    "ShellExecutorTool", 
    "ProcessManagerTool",
    "CalculatorTool",
    "TextProcessorTool",
    "DocumentAnalyzerTool"
]

# Register all built-in tools
BUILT_IN_TOOLS = [
    WebScraperTool,
    HTTPClientTool,
    FileReaderTool, 
    FileWriterTool,
    FileManagerTool,
    ShellExecutorTool,
    ProcessManagerTool,
    CalculatorTool,
    TextProcessorTool,
    DocumentAnalyzerTool
]