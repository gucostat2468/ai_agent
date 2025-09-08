#!/usr/bin/env python3
"""Setup configuration for AI Agent project."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="ai-agent",
    version="1.0.0",
    description="Robust and scalable AI Agent framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AI Agent Team",
    author_email="team@aiagent.com",
    url="https://github.com/aiagent/ai-agent",
    
    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    
    # Dependencies
    install_requires=read_requirements("requirements.txt"),
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.12.0",
            "pytest-cov>=4.1.0",
            "black>=23.10.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
            "pre-commit>=3.5.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "aws": [
            "boto3>=1.29.0",
            "botocore>=1.32.0",
        ],
        "gcp": [
            "google-cloud-aiplatform>=1.36.0",
        ],
        "azure": [
            "azure-cognitiveservices-language-textanalytics>=5.3.0",
        ],
    },
    
    # Entry points
    entry_points={
        "console_scripts": [
            "ai-agent=ai_agent.cli:main",
            "ai-agent-server=ai_agent.server:main",
        ],
        "ai_agent.plugins": [
            "web_scraper=ai_agent.plugins.examples.web_scraper:WebScraperPlugin",
            "file_manager=ai_agent.plugins.examples.file_manager:FileManagerPlugin",
            "calculator=ai_agent.plugins.examples.calculator:CalculatorPlugin",
        ],
    },
    
    # Package data
    include_package_data=True,
    package_data={
        "ai_agent": [
            "config/*.yaml",
            "config/*.json",
            "templates/*.txt",
            "templates/*.json",
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    # Keywords
    keywords="ai, artificial intelligence, agent, llm, automation, plugins",
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/aiagent/ai-agent/issues",
        "Source": "https://github.com/aiagent/ai-agent",
        "Documentation": "https://ai-agent.readthedocs.io/",
    },
)