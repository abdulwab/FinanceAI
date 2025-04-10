# Backend Migration Plan: FastAPI as Unified Backend

## 1. Overview & Goals

This plan outlines the steps to consolidate all backend functionality into the FastAPI application, eliminating the need for Next.js API routes and creating a clear separation between frontend and backend.

### 1.1 Primary Goals

- Centralize all backend logic in the FastAPI application
- Implement comprehensive API endpoints for all features
- Create a robust database access layer using SQLAlchemy
- Establish secure authentication and authorization
- Support all features including community, agent management, and payment processing

## 2. Database Migration Strategy

### 2.1 Schema Definition

- Convert all Prisma models to SQLAlchemy models
- Implement all tables and relationships in SQLAlchemy
- Create Alembic migrations for schema management
- Set up database version tracking

```python
# Example User model in SQLAlchemy
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    name = Column(String)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    agents = relationship("Agent", back_populates="user")
    workflows = relationship("Workflow", back_populates="user")
    comments = relationship("Comment", back_populates="user")
    reactions = relationship("Reaction", back_populates="user")
    ratings = relationship("Rating", back_populates="user")
    followers = relationship("UserFollow", foreign_keys="UserFollow.followed_id", back_populates="followed")
    following = relationship("UserFollow", foreign_keys="UserFollow.follower_id", back_populates="follower")
```

### 2.2 Data Migration

- Create a data migration script to transfer data from existing database
- Implement validation logic to ensure data integrity
- Develop rollback procedures for failed migrations
- Set up a testing strategy for database migrations

### 2.3 Database Connection Management

- Implement connection pooling for efficient resource usage
- Create database session management middleware
- Implement transaction management with proper error handling
- Set up database logging and monitoring

## 3. Authentication System

### 3.1 JWT Authentication

- Implement JWT-based authentication system
- Create token generation and validation logic
- Set up refresh token rotation for improved security
- Implement token blacklisting for logout functionality

```python
# Example JWT authentication functions
from datetime import datetime, timedelta
from jose import jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Token settings
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = get_user_by_id(user_id)
    if user is None:
        raise credentials_exception
    return user
```

### 3.2 User Management

- Implement user registration, login, and logout endpoints
- Create password reset and email verification functionality
- Set up user profile management endpoints
- Implement role-based access control

### 3.3 OAuth Integration

- Add support for OAuth providers (Google, GitHub, etc.)
- Implement OAuth token exchange and validation
- Create user account merging functionality
- Set up proper error handling for authentication failures

## 4. Core API Endpoints

### 4.1 Agent Management API

- Implement CRUD endpoints for agent management
- Create agent execution endpoints with proper error handling
- Set up agent configuration validation
- Implement agent sharing and permissions

```python
# Example agent routes
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from .database import get_db
from .models import Agent, User
from .schemas import AgentCreate, AgentUpdate, AgentResponse
from .auth import get_current_user

router = APIRouter(prefix="/api/v1/agents", tags=["agents"])

@router.post("/", response_model=AgentResponse)
async def create_agent(
    agent: AgentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Create agent logic
    db_agent = Agent(
        name=agent.name,
        instructions=agent.instructions,
        model=agent.model,
        user_id=current_user.id
    )
    db.add(db_agent)
    db.commit()
    db.refresh(db_agent)
    return db_agent

@router.get("/", response_model=List[AgentResponse])
async def get_agents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    skip: int = 0,
    limit: int = 100
):
    # Get agents logic
    agents = db.query(Agent).filter(Agent.user_id == current_user.id).offset(skip).limit(limit).all()
    return agents

@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Get single agent logic
    agent = db.query(Agent).filter(Agent.id == agent_id, Agent.user_id == current_user.id).first()
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent

@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: int,
    agent: AgentUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Update agent logic
    db_agent = db.query(Agent).filter(Agent.id == agent_id, Agent.user_id == current_user.id).first()
    if db_agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    for key, value in agent.dict(exclude_unset=True).items():
        setattr(db_agent, key, value)
    
    db.commit()
    db.refresh(db_agent)
    return db_agent

@router.delete("/{agent_id}")
async def delete_agent(
    agent_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Delete agent logic
    db_agent = db.query(Agent).filter(Agent.id == agent_id, Agent.user_id == current_user.id).first()
    if db_agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    db.delete(db_agent)
    db.commit()
    return {"message": "Agent deleted successfully"}

@router.post("/{agent_id}/run")
async def run_agent(
    agent_id: int,
    input_data: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Run agent logic
    # 1. Get the agent
    # 2. Check permissions
    # 3. Charge credits
    # 4. Run the agent
    # 5. Log results
    # 6. Return output
    pass
```

### 4.2 Multi-Agent Workflow API

- Create endpoints for workflow definition and management
- Implement workflow execution with proper state management
- Set up handoff configuration and validation
- Create workflow monitoring and logging endpoints

### 4.3 Tool Management API

- Implement endpoints for available tools
- Create tool configuration and validation logic
- Set up tool execution endpoints with error handling
- Implement tool security measures

### 4.4 File Management API

- Create file upload and download endpoints
- Implement vector storage integration for file search
- Set up file processing pipelines
- Create file permission management

### 4.5 Community API

- Implement user following/followers endpoints
- Create agent commenting and rating API
- Set up agent discovery and search endpoints
- Implement notification system

### 4.6 Payment and Credit API

- Create Stripe integration for payment processing
- Implement credit balance management
- Set up subscription handling
- Create usage tracking and billing endpoints

## 5. OpenAI Agents SDK Integration

### 5.1 Agent Execution Engine

- Implement agent creation and configuration
- Set up agent execution with proper error handling
- Create tracing and debugging capabilities
- Implement rate limiting and resource management

```python
# Example agent execution logic
from agents import Agent, Runner
from pydantic import BaseModel

class AgentInput(BaseModel):
    message: str
    context: dict = None

async def execute_agent(agent_config, input_data: AgentInput):
    try:
        # Create agent from configuration
        agent = Agent(
            name=agent_config.name,
            instructions=agent_config.instructions,
            model=agent_config.model
        )
        
        # Add tools if configured
        if agent_config.tools:
            for tool_config in agent_config.tools:
                # Configure and add tool based on tool_config
                pass
        
        # Run the agent
        result = await Runner.run(agent, input_data.message, context=input_data.context)
        
        # Process and return result
        return {
            "output": result.final_output,
            "messages": result.messages,
            "usage": {
                "total_tokens": result.usage.total_tokens,
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens
            }
        }
    except Exception as e:
        # Log error and return appropriate response
        logger.error(f"Agent execution error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")
```

### 5.2 Multi-Provider Integration

- Implement provider selection and configuration
- Create API key management with proper encryption
- Set up provider fallback mechanisms
- Implement provider-specific optimizations

### 5.3 Tool Integration

- Implement WebSearchTool with provider selection
- Create FileSearchTool with vector store integration
- Set up ComputerTool with proper security measures
- Implement custom function tools with validation

## 6. Vector Store Implementation

### 6.1 Document Processing

- Implement file type support for various formats
- Create text extraction and chunking pipelines
- Set up metadata extraction and storage
- Implement document versioning

### 6.2 Vector Database Integration

- Set up connection to vector database (Pinecone/Qdrant)
- Implement embedding generation using provider APIs
- Create vector search with filtering capabilities
- Set up vector store management endpoints

## 7. Credit and Payment System

### 7.1 Credit Management

- Implement credit balance tracking
- Create credit usage logging
- Set up credit purchase workflow
- Implement credit allocation for subscriptions

### 7.2 Stripe Integration

- Create secure Stripe API integration
- Implement webhook handling for payment events
- Set up subscription management
- Create invoice and receipt generation

## 8. Real-time Communication

### 8.1 WebSockets Implementation

- Set up WebSocket server for real-time updates
- Implement authentication for WebSocket connections
- Create notification delivery system
- Set up connection management and error handling

```python
# Example WebSocket implementation
from fastapi import WebSocket, WebSocketDisconnect, Depends
from typing import Dict, List

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, user_id: str):
        if user_id in self.active_connections:
            self.active_connections[user_id].remove(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
    
    async def send_personal_message(self, message: str, user_id: str):
        if user_id in self.active_connections:
            for connection in self.active_connections[user_id]:
                await connection.send_text(message)

manager = ConnectionManager()

@router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    # Authenticate user (simplified example)
    # In production, use proper token validation
    
    await manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_text()
            # Process received data if needed
            
            # Echo back for demonstration
            await websocket.send_text(f"Message received: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
```

## 9. Security Implementation

### 9.1 API Security

- Implement CORS configuration for frontend access
- Set up rate limiting to prevent abuse
- Create API key authentication for programmatic access
- Implement input validation and sanitization

### 9.2 Data Protection

- Implement encryption for sensitive data
- Create secure storage for API keys
- Set up proper error handling to prevent information leakage
- Implement data access logging

## 10. Testing Strategy

### 10.1 Unit Testing

- Test SQLAlchemy models and relationships
- Validate API endpoint logic
- Test authentication and authorization
- Verify business logic functions

### 10.2 Integration Testing

- Test API endpoints with database integration
- Validate multi-provider functionality
- Test file processing and vector store integration
- Verify payment processing workflows

### 10.3 Load Testing

- Test API performance under load
- Validate database connection pooling
- Test WebSocket connection handling
- Verify resource usage and optimization

## 11. Migration Steps

### 11.1 Preparation Phase
1. Set up SQLAlchemy models for all entities
2. Create Alembic migration system
3. Implement authentication system
4. Develop core API endpoints

### 11.2 Implementation Phase
1. Migrate database data from existing system
2. Implement agent execution engine
3. Set up file processing and vector store
4. Create payment and credit system
5. Implement WebSocket for real-time features

### 11.3 Testing Phase
1. Validate all API endpoints
2. Test authentication flows
3. Verify database integrity
4. Test agent execution

### 11.4 Deployment Phase
1. Set up production environment
2. Configure monitoring and logging
3. Implement backup and recovery procedures
4. Launch with frontend integration

## 12. Potential Challenges and Solutions

### 12.1 Database Migration Complexity
- **Challenge**: Moving data from Prisma to SQLAlchemy
- **Solution**: Create thorough migration scripts with validation

### 12.2 Authentication Security
- **Challenge**: Ensuring secure token management
- **Solution**: Implement token rotation and blacklisting

### 12.3 Performance at Scale
- **Challenge**: Handling large numbers of concurrent requests
- **Solution**: Implement proper connection pooling and caching

### 12.4 AI Provider Integration
- **Challenge**: Managing multiple provider APIs
- **Solution**: Create a provider abstraction layer with fallbacks

## 13. Timeline and Milestones

### 13.1 Phase 1: Core Infrastructure (Weeks 1-3)
- Set up database models and migrations
- Implement authentication system
- Create core API endpoints
- Establish testing framework

### 13.2 Phase 2: Agent Features (Weeks 4-7)
- Implement OpenAI Agents SDK integration
- Create multi-provider support
- Set up tool integration
- Develop agent execution engine

### 13.3 Phase 3: Community Features (Weeks 8-10)
- Implement user social interactions
- Create agent sharing and discovery
- Set up commenting and rating system
- Develop notification system

### 13.4 Phase 4: Advanced Features (Weeks 11-14)
- Implement file processing and vector store
- Create payment and credit system
- Set up WebSockets for real-time updates
- Develop multi-agent workflows

### 13.5 Phase 5: Testing and Deployment (Weeks 15-16)
- Comprehensive testing
- Performance optimization
- Security auditing
- Production deployment

## 14. Conclusion

This migration will significantly improve our architecture by consolidating all backend functionality in the FastAPI application. This will create a clear separation of concerns, improve maintainability, and enable more efficient scaling. The FastAPI backend will provide a comprehensive API that supports all features required by the frontend, while handling all business logic and data management internally. 