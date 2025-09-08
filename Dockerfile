# Multi-stage build for AI Agent
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    AI_AGENT_ENV=production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libpq5 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create application user
RUN groupadd -r aiagent && \
    useradd -r -g aiagent -d /app -s /sbin/nologin aiagent

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=aiagent:aiagent . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/plugins /app/uploads && \
    chown -R aiagent:aiagent /app

# Install application
RUN pip install -e .

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Wait for database\n\
echo "Waiting for database..."\n\
while ! curl -f $DATABASE_URL 2>/dev/null; do\n\
    echo "Database not ready, waiting..."\n\
    sleep 2\n\
done\n\
\n\
# Run database migrations\n\
echo "Running database migrations..."\n\
alembic upgrade head\n\
\n\
# Start application\n\
echo "Starting AI Agent..."\n\
exec "$@"' > /app/entrypoint.sh && \
chmod +x /app/entrypoint.sh

# Switch to application user
USER aiagent

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 9090

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["uvicorn", "ai_agent.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Development stage (for local development)
FROM production as development

ENV AI_AGENT_ENV=development

# Switch back to root for development tools
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y \
    vim \
    git \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install \
    pytest \
    pytest-asyncio \
    pytest-mock \
    pytest-cov \
    black \
    isort \
    flake8 \
    mypy \
    ipdb \
    ipython

# Switch back to application user
USER aiagent

# Override command for development
CMD ["python", "-m", "ai_agent.main", "--dev"]