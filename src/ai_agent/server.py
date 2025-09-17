"""
AI Agent HTTP Server
FastAPI-based web server for AI Agent with REST API, WebSocket support, and monitoring endpoints.
"""

import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone
from pathlib import Path
import traceback

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from .core.agent import AIAgent, ProcessingContext
from .config.settings import AgentSettings
from .monitoring.logger import StructuredLogger, LoggingContext, get_audit_logger
from .monitoring.metrics import setup_prometheus_metrics, get_metrics_handler
from .utils.exceptions import AIAgentException, create_error_response
from .security.authentication import authenticate_request
from .security.authorization import authorize_request
from .security.input_validation import validate_input, sanitize_input


# Pydantic models for API

class ChatMessage(BaseModel):
    """Chat message model"""
    message: str = Field(..., min_length=1, max_length=10000, description="User message")
    session_id: Optional[str] = Field(None, description="Session identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Message metadata")


class ChatResponse(BaseModel):
    """Chat response model"""
    response: str = Field(..., description="Agent response")
    session_id: str = Field(..., description="Session identifier")
    message_id: str = Field(..., description="Message identifier")
    timestamp: datetime = Field(..., description="Response timestamp")
    processing_time: float = Field(..., description="Processing time in seconds")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Response metadata")


class TaskDefinition(BaseModel):
    """Task definition model"""
    name: str = Field(..., description="Task name")
    description: str = Field("", description="Task description")
    task_type: str = Field("generic", description="Task type")
    steps: List[Dict[str, Any]] = Field(default_factory=list, description="Task steps")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Task context")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Task metadata")
    priority: Optional[str] = Field("normal", description="Task priority")
    timeout: Optional[float] = Field(None, description="Task timeout in seconds")


class TaskResponse(BaseModel):
    """Task execution response model"""
    task_id: str = Field(..., description="Task identifier")
    status: str = Field(..., description="Task status")
    result: Optional[Any] = Field(None, description="Task result")
    error: Optional[str] = Field(None, description="Error message if failed")
    progress: float = Field(0.0, description="Task progress (0.0 to 1.0)")
    created_at: datetime = Field(..., description="Task creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Task start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Task completion timestamp")


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field("1.0.0", description="Application version")
    uptime: float = Field(..., description="Uptime in seconds")
    components: Dict[str, Any] = Field(..., description="Component health status")


# WebSocket connection manager

class WebSocketManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.logger = StructuredLogger(__name__)
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept and store WebSocket connection"""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.logger.info("WebSocket connection established", session_id=session_id)
    
    def disconnect(self, session_id: str):
        """Remove WebSocket connection"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            self.logger.info("WebSocket connection closed", session_id=session_id)
    
    async def send_message(self, session_id: str, message: Dict[str, Any]):
        """Send message to specific WebSocket"""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            try:
                await websocket.send_json(message)
            except Exception as e:
                self.logger.error("Failed to send WebSocket message", session_id=session_id, error=str(e))
                self.disconnect(session_id)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected WebSockets"""
        disconnected = []
        for session_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(message)
            except Exception as e:
                self.logger.error("Failed to broadcast to WebSocket", session_id=session_id, error=str(e))
                disconnected.append(session_id)
        
        # Remove disconnected sessions
        for session_id in disconnected:
            self.disconnect(session_id)


# Application lifespan management

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger = StructuredLogger(__name__)
    
    try:
        logger.info("Starting AI Agent server")
        
        # Initialize agent if not already done
        agent = app.state.agent
        if not agent.is_running:
            await agent.start()
        
        # Setup Prometheus metrics
        if app.state.settings.monitoring.enable_metrics:
            setup_prometheus_metrics(app)
        
        logger.info("AI Agent server started successfully")
        yield
        
    finally:
        logger.info("Shutting down AI Agent server")
        
        # Cleanup agent
        if hasattr(app.state, 'agent') and app.state.agent:
            await app.state.agent.cleanup()
        
        logger.info("AI Agent server shut down")


# Create FastAPI application

def create_app(agent: AIAgent, settings: AgentSettings) -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="AI Agent API",
        description="Intelligent AI Agent with plugin system and advanced capabilities",
        version="1.0.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan
    )
    
    # Store application state
    app.state.agent = agent
    app.state.settings = settings
    app.state.websocket_manager = WebSocketManager()
    app.state.startup_time = datetime.now(timezone.utc)
    
    # Setup middleware
    setup_middleware(app, settings)
    
    # Setup routes
    setup_routes(app)
    
    return app


def setup_middleware(app: FastAPI, settings: AgentSettings):
    """Setup application middleware"""
    
    # CORS middleware
    if settings.security.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.security.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )
    
    # Trusted host middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"] if settings.debug else ["localhost", "127.0.0.1", settings.host]
    )
    
    # Request logging middleware
    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        """Log all HTTP requests"""
        logger = StructuredLogger(__name__)
        audit_logger = get_audit_logger()
        
        start_time = asyncio.get_event_loop().time()
        request_id = str(uuid.uuid4())
        
        with LoggingContext(correlation_id_value=request_id):
            # Log request
            logger.info("HTTP request received",
                       method=request.method,
                       url=str(request.url),
                       user_agent=request.headers.get("user-agent"),
                       request_id=request_id)
            
            try:
                response = await call_next(request)
                
                # Calculate processing time
                processing_time = asyncio.get_event_loop().time() - start_time
                
                # Log response
                logger.info("HTTP request completed",
                           method=request.method,
                           url=str(request.url),
                           status_code=response.status_code,
                           processing_time=processing_time,
                           request_id=request_id)
                
                # Audit log for sensitive endpoints
                if any(path in str(request.url) for path in ["/chat", "/tasks", "/admin"]):
                    audit_logger.log_user_action(
                        user_id=request.headers.get("X-User-ID", "anonymous"),
                        action=f"{request.method} {request.url.path}",
                        resource=request.url.path,
                        result="success" if response.status_code < 400 else "failure"
                    )
                
                return response
                
            except Exception as e:
                processing_time = asyncio.get_event_loop().time() - start_time
                
                logger.error("HTTP request failed",
                           method=request.method,
                           url=str(request.url),
                           error=str(e),
                           processing_time=processing_time,
                           request_id=request_id)
                
                # Return error response
                return JSONResponse(
                    status_code=500,
                    content={"error": "Internal server error", "request_id": request_id}
                )


def setup_routes(app: FastAPI):
    """Setup application routes"""
    
    logger = StructuredLogger(__name__)
    security = HTTPBearer(auto_error=False)
    
    # Authentication dependency
    async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
        """Get current authenticated user"""
        if not credentials:
            return None
        
        try:
            user_info = await authenticate_request(credentials.credentials, app.state.settings.security)
            return user_info
        except Exception as e:
            logger.warning("Authentication failed", error=str(e))
            return None
    
    # Health check endpoint
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint"""
        try:
            agent_health = await app.state.agent.health_check()
            uptime = (datetime.now(timezone.utc) - app.state.startup_time).total_seconds()
            
            return HealthResponse(
                status=agent_health.get("status", "unknown"),
                timestamp=datetime.now(timezone.utc),
                uptime=uptime,
                components=agent_health.get("components", {})
            )
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            raise HTTPException(status_code=500, detail="Health check failed")
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "name": "AI Agent API",
            "version": "1.0.0",
            "status": "running",
            "docs_url": "/docs",
            "health_url": "/health"
        }
    
    # Chat endpoints
    @app.post("/api/v1/chat", response_model=ChatResponse)
    async def chat(
        message: ChatMessage,
        background_tasks: BackgroundTasks,
        user: dict = Depends(get_current_user)
    ):
        """Process chat message"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Validate and sanitize input
            validation_result = await validate_input(message.message, "text")
            if not validation_result.get("valid", True):
                raise HTTPException(status_code=400, detail="Invalid input")
            
            sanitized_message = await sanitize_input(message.message)
            
            # Create processing context
            session_id = message.session_id or str(uuid.uuid4())
            message_id = str(uuid.uuid4())
            
            context = ProcessingContext(
                session_id=session_id,
                user_id=message.user_id or (user.get("user_id") if user else None)
            )
            
            # Process message
            with LoggingContext(session_id_value=session_id, user_id_value=context.user_id):
                response = await app.state.agent.process_message(sanitized_message, context)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Create response
            chat_response = ChatResponse(
                response=response,
                session_id=session_id,
                message_id=message_id,
                timestamp=datetime.now(timezone.utc),
                processing_time=processing_time,
                metadata={"user_id": context.user_id}
            )
            
            # Send WebSocket notification if connected
            background_tasks.add_task(
                notify_websocket_clients,
                app.state.websocket_manager,
                session_id,
                {
                    "type": "chat_response",
                    "data": chat_response.dict()
                }
            )
            
            return chat_response
            
        except HTTPException:
            raise
        except AIAgentException as e:
            logger.error("Agent processing failed", error=str(e))
            raise HTTPException(status_code=422, detail=create_error_response(e))
        except Exception as e:
            logger.error("Chat endpoint failed", error=str(e))
            raise HTTPException(status_code=500, detail="Internal server error")
    
    # Task endpoints
    @app.post("/api/v1/tasks", response_model=TaskResponse)
    async def create_task(
        task_def: TaskDefinition,
        background_tasks: BackgroundTasks,
        user: dict = Depends(get_current_user)
    ):
        """Create and execute a task"""
        try:
            # Convert to dict for agent
            task_dict = task_def.dict()
            task_dict["id"] = str(uuid.uuid4())
            
            # Create processing context
            context = ProcessingContext(
                user_id=user.get("user_id") if user else None
            )
            
            # Execute task
            result = await app.state.agent.execute_task(task_dict, context)
            
            return TaskResponse(
                task_id=task_dict["id"],
                status="completed",
                result=result,
                progress=1.0,
                created_at=datetime.now(timezone.utc),
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error("Task creation failed", error=str(e))
            return TaskResponse(
                task_id=task_dict.get("id", "unknown"),
                status="failed",
                error=str(e),
                progress=0.0,
                created_at=datetime.now(timezone.utc)
            )
    
    @app.get("/api/v1/tasks/{task_id}")
    async def get_task_status(task_id: str, user: dict = Depends(get_current_user)):
        """Get task status"""
        # This would require task storage in the coordinator
        # For now, return a simple response
        return {"task_id": task_id, "status": "not_implemented"}
    
    # Agent information endpoints
    @app.get("/api/v1/agent/info")
    async def get_agent_info(user: dict = Depends(get_current_user)):
        """Get agent information"""
        try:
            health = await app.state.agent.health_check()
            return {
                "agent_id": health.get("agent_id"),
                "status": health.get("status"),
                "capabilities": {
                    "plugins": len(health.get("components", {}).get("plugins", {})),
                    "tools": len(health.get("components", {}).get("tools", {}))
                }
            }
        except Exception as e:
            logger.error("Agent info request failed", error=str(e))
            raise HTTPException(status_code=500, detail="Failed to get agent info")
    
    @app.get("/api/v1/agent/plugins")
    async def list_plugins(user: dict = Depends(get_current_user)):
        """List available plugins"""
        try:
            # This would require access to plugin registry
            return {"plugins": []}  # Placeholder
        except Exception as e:
            logger.error("Plugin list request failed", error=str(e))
            raise HTTPException(status_code=500, detail="Failed to list plugins")
    
    @app.get("/api/v1/agent/tools")
    async def list_tools(user: dict = Depends(get_current_user)):
        """List available tools"""
        try:
            # This would require access to tool registry
            return {"tools": []}  # Placeholder
        except Exception as e:
            logger.error("Tool list request failed", error=str(e))
            raise HTTPException(status_code=500, detail="Failed to list tools")
    
    # WebSocket endpoint
    @app.websocket("/ws/{session_id}")
    async def websocket_endpoint(websocket: WebSocket, session_id: str):
        """WebSocket endpoint for real-time communication"""
        manager = app.state.websocket_manager
        
        try:
            await manager.connect(websocket, session_id)
            
            # Send welcome message
            await manager.send_message(session_id, {
                "type": "connection",
                "message": "Connected to AI Agent WebSocket",
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            while True:
                # Receive message from client
                data = await websocket.receive_json()
                
                if data.get("type") == "chat":
                    # Process chat message via WebSocket
                    try:
                        message = data.get("message", "")
                        context = ProcessingContext(session_id=session_id)
                        
                        # Process with agent
                        response = await app.state.agent.process_message(message, context)
                        
                        # Send response
                        await manager.send_message(session_id, {
                            "type": "chat_response",
                            "response": response,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        })
                        
                    except Exception as e:
                        await manager.send_message(session_id, {
                            "type": "error",
                            "error": str(e),
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        })
                
                elif data.get("type") == "ping":
                    # Respond to ping
                    await manager.send_message(session_id, {
                        "type": "pong",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
        
        except WebSocketDisconnect:
            manager.disconnect(session_id)
            logger.info("WebSocket disconnected", session_id=session_id)
        
        except Exception as e:
            logger.error("WebSocket error", session_id=session_id, error=str(e))
            manager.disconnect(session_id)
    
    # Metrics endpoint
    if app.state.settings.monitoring.enable_metrics:
        @app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint"""
            try:
                metrics_handler = get_metrics_handler()
                return metrics_handler.generate_latest()
            except Exception as e:
                logger.error("Metrics endpoint failed", error=str(e))
                raise HTTPException(status_code=500, detail="Metrics unavailable")
    
    # Static files (if web UI is available)
    web_ui_path = Path("web")
    if web_ui_path.exists():
        app.mount("/static", StaticFiles(directory=web_ui_path / "static"), name="static")
        
        @app.get("/ui")
        async def web_ui():
            """Serve web UI"""
            ui_file = web_ui_path / "index.html"
            if ui_file.exists():
                return FileResponse(ui_file)
            else:
                return HTMLResponse("<h1>Web UI not available</h1>")
    
    # Admin endpoints (if enabled and authenticated)
    if app.state.settings.development.enable_dev_endpoints:
        
        @app.post("/admin/shutdown")
        async def admin_shutdown(user: dict = Depends(get_current_user)):
            """Shutdown the agent (admin only)"""
            if not user or not user.get("is_admin", False):
                raise HTTPException(status_code=403, detail="Admin access required")
            
            # Schedule shutdown
            async def shutdown_agent():
                await asyncio.sleep(1)  # Give time for response
                await app.state.agent.stop()
            
            asyncio.create_task(shutdown_agent())
            return {"message": "Shutdown initiated"}
        
        @app.get("/admin/logs")
        async def admin_logs(limit: int = 100, user: dict = Depends(get_current_user)):
            """Get recent logs (admin only)"""
            if not user or not user.get("is_admin", False):
                raise HTTPException(status_code=403, detail="Admin access required")
            
            # This would require log storage implementation
            return {"logs": [], "message": "Log retrieval not implemented"}
        
        @app.get("/admin/stats")
        async def admin_stats(user: dict = Depends(get_current_user)):
            """Get system statistics (admin only)"""
            if not user or not user.get("is_admin", False):
                raise HTTPException(status_code=403, detail="Admin access required")
            
            try:
                health = await app.state.agent.health_check()
                uptime = (datetime.now(timezone.utc) - app.state.startup_time).total_seconds()
                
                return {
                    "uptime": uptime,
                    "agent_status": health.get("status"),
                    "memory_usage": health.get("memory_usage", 0),
                    "session_count": health.get("session_count", 0),
                    "active_websockets": len(app.state.websocket_manager.active_connections)
                }
                
            except Exception as e:
                logger.error("Admin stats failed", error=str(e))
                raise HTTPException(status_code=500, detail="Failed to get stats")


async def notify_websocket_clients(manager: WebSocketManager, session_id: str, message: Dict[str, Any]):
    """Background task to notify WebSocket clients"""
    try:
        await manager.send_message(session_id, message)
    except Exception as e:
        logger = StructuredLogger(__name__)
        logger.error("WebSocket notification failed", session_id=session_id, error=str(e))


# Server management

class AIAgentServer:
    """AI Agent HTTP Server wrapper"""
    
    def __init__(self, agent: AIAgent, settings: AgentSettings):
        self.agent = agent
        self.settings = settings
        self.app = create_app(agent, settings)
        self.logger = StructuredLogger(__name__)
        self._server = None
    
    async def start(self):
        """Start the server"""
        try:
            config = uvicorn.Config(
                app=self.app,
                host=self.settings.host,
                port=self.settings.port,
                workers=1,  # Single worker for now to maintain state
                log_level="info" if not self.settings.debug else "debug",
                access_log=True,
                reload=self.settings.development.hot_reload and self.settings.debug
            )
            
            self._server = uvicorn.Server(config)
            
            self.logger.info("Starting AI Agent HTTP server",
                           host=self.settings.host,
                           port=self.settings.port)
            
            await self._server.serve()
            
        except Exception as e:
            self.logger.error("Failed to start server", error=str(e))
            raise
    
    async def stop(self):
        """Stop the server"""
        if self._server:
            try:
                self.logger.info("Stopping AI Agent HTTP server")
                self._server.should_exit = True
                await self._server.shutdown()
            except Exception as e:
                self.logger.error("Error stopping server", error=str(e))
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for server"""
        try:
            return {
                "server_status": "running" if self._server else "stopped",
                "host": self.settings.host,
                "port": self.settings.port,
                "debug": self.settings.debug,
                "websocket_connections": len(self.app.state.websocket_manager.active_connections) if hasattr(self.app.state, 'websocket_manager') else 0
            }
        except Exception as e:
            return {
                "server_status": "error",
                "error": str(e)
            }


def create_server(agent: AIAgent, settings: AgentSettings) -> AIAgentServer:
    """Create AI Agent server instance"""
    return AIAgentServer(agent, settings)


# Development server function
async def run_development_server(agent: AIAgent, settings: AgentSettings):
    """Run development server with hot reload"""
    server = create_server(agent, settings)
    await server.start()


if __name__ == "__main__":
    # This allows running the server directly for development
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from ai_agent.main import AIAgentApplication
    from ai_agent.config.settings import load_settings
    
    async def main():
        settings = load_settings()
        app = AIAgentApplication(settings)
        await app.initialize()
        
        server = create_server(app.agent, settings)
        await server.start()
    
    asyncio.run(main())