# OpenAI Agents SDK Implementation Plan

## 1. Overview & Core SDK Components

The OpenAI Agents SDK will be integrated as the core agent execution engine within our FastAPI backend. We'll implement all major components:

- **Agents**: LLMs with instructions, tools, and capabilities
- **Handoffs**: Delegation between specialized agents 
- **Guardrails**: Input/output validation for security and quality
- **Tools**: Both built-in and custom function tools
- **Multi-Agent Orchestration**: Both LLM-based and code-based orchestration
- **Tracing**: Visualization and debugging of agent execution

## 2. Agent Management API

### 2.1 Agent Definition Endpoints

```python
@router.post("/api/v1/agents", response_model=AgentResponse)
async def create_agent(
    agent_data: AgentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new agent configuration."""
    # 1. Validate agent configuration
    # 2. Store in database
    # 3. Return agent data with ID
```

### 2.2 Agent Schema

```python
class AgentBase(BaseModel):
    name: str
    instructions: str
    model: Optional[str] = "gpt-4o"
    handoff_description: Optional[str] = None
    input_guardrails: Optional[List[GuardrailCreate]] = None
    output_guardrails: Optional[List[GuardrailCreate]] = None
    output_type: Optional[Dict[str, Any]] = None
    tools: Optional[List[ToolCreate]] = None
    metadata: Optional[Dict[str, Any]] = None

class AgentCreate(AgentBase):
    pass

class AgentResponse(AgentBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime
```

### 2.3 Agent Execution Endpoint

```python
@router.post("/api/v1/agents/{agent_id}/run")
async def run_agent(
    agent_id: int,
    input_data: AgentRunInput,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Run an agent with the provided input."""
    # 1. Load agent from database
    # 2. Validate user has permission & credits
    # 3. Build agent from configuration
    # 4. Execute agent
    # 5. Log result and usage
    # 6. Return response
```

## 3. Core Agent Implementation

### 3.1 Agent Building Function

```python
async def build_agent_from_config(agent_config, tools_config=None, context=None):
    """Build an OpenAI Agent object from stored configuration."""
    agent = Agent(
        name=agent_config.name,
        instructions=agent_config.instructions,
        model=agent_config.model or "gpt-4o",
    )
    
    # Add tools if configured
    if tools_config:
        for tool_config in tools_config:
            tool = await build_tool_from_config(tool_config)
            agent.tools.append(tool)
    
    # Configure output type if specified
    if agent_config.output_type:
        agent.output_type = create_dynamic_model_from_schema(agent_config.output_type)
    
    # Configure handoff description if specified
    if agent_config.handoff_description:
        agent.handoff_description = agent_config.handoff_description
    
    return agent
```

### 3.2 Agent Execution Function

```python
async def execute_agent(agent, input_text, context=None, stream=False):
    """Execute an agent and return the result."""
    try:
        # Start timing
        start_time = time.time()
        
        # Execute agent
        if stream:
            # Return streaming response
            return fastapi.responses.StreamingResponse(
                stream_agent_execution(agent, input_text, context),
                media_type="text/event-stream"
            )
        else:
            # Execute synchronously
            result = await Runner.run(agent, input_text, context=context)
            
            # Capture usage for billing
            usage = {
                "total_tokens": result.usage.total_tokens,
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "execution_time": time.time() - start_time
            }
            
            # Process result based on output type
            if agent.output_type:
                final_output = result.final_output_as(agent.output_type)
                formatted_output = final_output.model_dump()
            else:
                formatted_output = result.final_output
            
            return {
                "output": formatted_output,
                "messages": result.messages,
                "usage": usage
            }
            
    except Exception as e:
        logger.error(f"Agent execution error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")
```

## 4. Tool Implementation

### 4.1 Built-in Tool Integration

```python
class ToolType(str, Enum):
    WEB_SEARCH = "web_search"
    FILE_SEARCH = "file_search"
    COMPUTER = "computer"
    FUNCTION = "function"
    AGENT = "agent"

async def build_tool_from_config(tool_config):
    """Build the appropriate tool based on configuration."""
    if tool_config.type == ToolType.WEB_SEARCH:
        return WebSearchTool(
            max_search_results=tool_config.params.get("max_search_results", 5)
        )
    
    elif tool_config.type == ToolType.FILE_SEARCH:
        return FileSearchTool(
            vector_store_ids=[tool_config.params.get("vector_store_id")],
            max_num_results=tool_config.params.get("max_num_results", 3)
        )
    
    elif tool_config.type == ToolType.COMPUTER:
        return ComputerTool(
            allowed_commands=tool_config.params.get("allowed_commands"),
            blocked_paths=tool_config.params.get("blocked_paths"),
            read_only=tool_config.params.get("read_only", True)
        )
    
    elif tool_config.type == ToolType.FUNCTION:
        return await build_function_tool(tool_config)
    
    elif tool_config.type == ToolType.AGENT:
        agent_id = tool_config.params.get("agent_id")
        agent_config = await get_agent_config(agent_id)
        return await build_agent_from_config(agent_config)
```

### 4.2 Custom Function Tool Builder

```python
async def build_function_tool(tool_config):
    """Build a custom function tool from configuration."""
    
    # Define the function based on tool configuration
    async def dynamic_function(**kwargs):
        try:
            # If code execution is allowed and code is provided
            if tool_config.params.get("allow_code_execution", False) and tool_config.params.get("code"):
                # Execute the code in a sandboxed environment
                result = await execute_code_safely(
                    tool_config.params["code"], 
                    kwargs
                )
                return result
            # Otherwise use the predefined function logic
            elif tool_config.params.get("endpoint"):
                # Call the configured endpoint
                result = await call_function_endpoint(
                    tool_config.params["endpoint"], 
                    kwargs
                )
                return result
            else:
                raise ValueError("Function tool missing implementation")
        except Exception as e:
            logger.error(f"Function tool execution error: {str(e)}")
            return f"Error executing function: {str(e)}"
    
    # Set the function name and docstring
    dynamic_function.__name__ = tool_config.name
    dynamic_function.__doc__ = tool_config.description
    
    # Add parameter annotations
    if tool_config.params.get("parameters"):
        signature_params = []
        annotations = {}
        
        for param in tool_config.params["parameters"]:
            param_name = param["name"]
            param_type = eval(param["type"]) if param.get("type") else str
            default_value = param.get("default", Parameter.empty)
            
            signature_params.append(
                Parameter(
                    param_name, 
                    Parameter.POSITIONAL_OR_KEYWORD, 
                    default=default_value, 
                    annotation=param_type
                )
            )
            annotations[param_name] = param_type
        
        dynamic_function.__annotations__ = annotations
        dynamic_function.__signature__ = Signature(signature_params)
    
    return dynamic_function
```

## 5. Guardrails Implementation

### 5.1 Input Guardrails

```python
class GuardrailType(str, Enum):
    INPUT = "input"
    OUTPUT = "output"

class GuardrailBase(BaseModel):
    name: str
    description: str
    type: GuardrailType
    implementation: str  # "function", "agent", "regex"
    config: Dict[str, Any]

async def build_input_guardrail(guardrail_config):
    """Build an input guardrail from configuration."""
    
    async def guardrail_function(ctx, agent, input_data):
        if guardrail_config.implementation == "function":
            # Use a predefined function guardrail
            if guardrail_config.config.get("code"):
                result = await execute_code_safely(
                    guardrail_config.config["code"],
                    {"ctx": ctx, "agent": agent, "input_data": input_data}
                )
            elif guardrail_config.config.get("endpoint"):
                result = await call_guardrail_endpoint(
                    guardrail_config.config["endpoint"],
                    {"ctx": ctx, "agent_id": agent.id, "input_data": input_data}
                )
            return GuardrailFunctionOutput(
                output_info=result.get("output_info", {}),
                tripwire_triggered=result.get("tripwire_triggered", False),
                override_message=result.get("override_message")
            )
            
        elif guardrail_config.implementation == "agent":
            # Use another agent as a guardrail
            guardrail_agent_id = guardrail_config.config.get("agent_id")
            guardrail_agent_config = await get_agent_config(guardrail_agent_id)
            guardrail_agent = await build_agent_from_config(guardrail_agent_config)
            
            result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
            
            # Process the output to determine if tripwire triggered
            if guardrail_agent.output_type:
                output = result.final_output_as(guardrail_agent.output_type)
                return GuardrailFunctionOutput(
                    output_info=output.model_dump(),
                    tripwire_triggered=output.model_dump().get("tripwire_triggered", False),
                    override_message=output.model_dump().get("override_message")
                )
            else:
                # Default to passing the input if no structured output
                return GuardrailFunctionOutput(
                    output_info={"result": result.final_output},
                    tripwire_triggered=False
                )
                
        elif guardrail_config.implementation == "regex":
            # Use regex pattern matching as a guardrail
            pattern = guardrail_config.config.get("pattern", "")
            if pattern and re.search(pattern, input_data):
                return GuardrailFunctionOutput(
                    output_info={"pattern_matched": True},
                    tripwire_triggered=True,
                    override_message=guardrail_config.config.get("message", "Input not allowed")
                )
            return GuardrailFunctionOutput(
                output_info={"pattern_matched": False},
                tripwire_triggered=False
            )
    
    return InputGuardrail(guardrail_function=guardrail_function)
```

### 5.2 Output Guardrails

```python
async def build_output_guardrail(guardrail_config):
    """Build an output guardrail from configuration."""
    
    async def guardrail_function(ctx, agent, output_data):
        if guardrail_config.implementation == "function":
            # Similar to input guardrail but for output
            if guardrail_config.config.get("code"):
                result = await execute_code_safely(
                    guardrail_config.config["code"],
                    {"ctx": ctx, "agent": agent, "output_data": output_data}
                )
            elif guardrail_config.config.get("endpoint"):
                result = await call_guardrail_endpoint(
                    guardrail_config.config["endpoint"],
                    {"ctx": ctx, "agent_id": agent.id, "output_data": output_data}
                )
            return GuardrailFunctionOutput(
                output_info=result.get("output_info", {}),
                tripwire_triggered=result.get("tripwire_triggered", False),
                override_output=result.get("override_output")
            )
            
        elif guardrail_config.implementation == "agent":
            # Use another agent as a guardrail
            guardrail_agent_id = guardrail_config.config.get("agent_id")
            guardrail_agent_config = await get_agent_config(guardrail_agent_id)
            guardrail_agent = await build_agent_from_config(guardrail_agent_config)
            
            result = await Runner.run(
                guardrail_agent, 
                f"Review this output: {output_data}",
                context=ctx.context
            )
            
            # Process the output to determine if tripwire triggered
            if guardrail_agent.output_type:
                output = result.final_output_as(guardrail_agent.output_type)
                return GuardrailFunctionOutput(
                    output_info=output.model_dump(),
                    tripwire_triggered=output.model_dump().get("tripwire_triggered", False),
                    override_output=output.model_dump().get("override_output")
                )
            else:
                return GuardrailFunctionOutput(
                    output_info={"result": result.final_output},
                    tripwire_triggered=False
                )
                
        elif guardrail_config.implementation == "regex":
            # Use regex pattern matching as a guardrail
            pattern = guardrail_config.config.get("pattern", "")
            if pattern and re.search(pattern, output_data):
                return GuardrailFunctionOutput(
                    output_info={"pattern_matched": True},
                    tripwire_triggered=True,
                    override_output=guardrail_config.config.get("replacement", "Output not allowed")
                )
            return GuardrailFunctionOutput(
                output_info={"pattern_matched": False},
                tripwire_triggered=False
            )
    
    return OutputGuardrail(guardrail_function=guardrail_function)
```

## 6. Multi-Agent Orchestration

### 6.1 LLM-based Orchestration (Handoffs)

```python
@router.post("/api/v1/agents/{agent_id}/handoffs")
async def add_handoff(
    agent_id: int,
    handoff_data: HandoffCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Add a handoff target to an agent."""
    # 1. Verify agent exists and user has permission
    # 2. Verify target agent exists
    # 3. Create handoff relationship
    # 4. Return updated agent
```

### 6.2 Code-based Orchestration

```python
@router.post("/api/v1/workflows")
async def create_workflow(
    workflow_data: WorkflowCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new workflow of multiple agents."""
    # 1. Validate workflow configuration
    # 2. Store in database
    # 3. Return workflow with ID
```

```python
@router.post("/api/v1/workflows/{workflow_id}/run")
async def run_workflow(
    workflow_id: int,
    input_data: WorkflowRunInput,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Run a multi-agent workflow."""
    # 1. Load workflow from database
    # 2. Verify user has permission & credits
    # 3. Execute workflow
    # 4. Return results
```

```python
async def execute_workflow(workflow_config, input_data, context=None):
    """Execute a multi-agent workflow based on its configuration."""
    steps = workflow_config.steps
    results = {}
    
    # Process steps in defined order
    for step in steps:
        step_id = step.id
        agent_id = step.agent_id
        agent_config = await get_agent_config(agent_id)
        agent = await build_agent_from_config(agent_config)
        
        # Prepare step input
        if step.input_source == "user":
            step_input = input_data.message
        elif step.input_source == "previous_step":
            previous_step_id = step.input_source_id
            if previous_step_id not in results:
                raise ValueError(f"Previous step {previous_step_id} not executed yet")
            step_input = results[previous_step_id]["output"]
        
        # Execute agent
        step_context = context.copy() if context else {}
        # Add results from previous steps to context
        step_context["workflow_results"] = results
        
        result = await Runner.run(agent, step_input, context=step_context)
        
        # Store result for potential future steps
        results[step_id] = {
            "output": result.final_output,
            "messages": result.messages,
            "usage": {
                "total_tokens": result.usage.total_tokens,
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens
            }
        }
        
        # Check for conditional branching
        if step.conditions:
            for condition in step.conditions:
                if evaluate_condition(condition, results[step_id]):
                    # Jump to specified step
                    next_step_idx = find_step_index(steps, condition.target_step_id)
                    if next_step_idx > 0:
                        steps = steps[next_step_idx:]
                        break
    
    # Return final result and all intermediate results
    return results
```

## 7. Context Management

### 7.1 Context Objects

```python
class ContextCreate(BaseModel):
    agent_id: int
    name: str 
    content: Dict[str, Any]
    is_default: bool = False

@router.post("/api/v1/agents/{agent_id}/contexts")
async def add_context(
    agent_id: int,
    context_data: ContextCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Add a context object to an agent."""
    # 1. Verify agent exists and user has permission
    # 2. Create context object
    # 3. Return created context
```

### 7.2 Context Injection

```python
async def get_agent_context(agent_id, context_id=None):
    """Retrieve context for an agent."""
    if context_id:
        # Get specific context by ID
        context = await get_context_by_id(context_id)
    else:
        # Get default context if available
        context = await get_default_context_for_agent(agent_id)
    
    if not context:
        return {}
    
    return context.content
```

## 8. Model Provider Integration

### 8.1 Provider Configuration

```python
class ProviderType(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"

class ProviderConfig(BaseModel):
    type: ProviderType
    api_key: str
    base_url: Optional[str] = None
    models: List[str]
    default_model: str
    organization_id: Optional[str] = None
    additional_config: Optional[Dict[str, Any]] = None

@router.post("/api/v1/providers")
async def add_provider(
    provider_data: ProviderConfig,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Add a new provider configuration."""
    # 1. Encrypt API key
    # 2. Store provider config
    # 3. Return provider ID
```

### 8.2 Provider Selection

```python
async def get_openai_client(provider_id=None):
    """Get an OpenAI client for the specified provider."""
    if not provider_id:
        # Use default provider
        provider_config = await get_default_provider_config()
    else:
        provider_config = await get_provider_config(provider_id)
    
    if not provider_config:
        # Use system default
        return openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    if provider_config.type == ProviderType.OPENAI:
        return openai.OpenAI(
            api_key=decrypt_api_key(provider_config.api_key),
            organization=provider_config.organization_id,
            base_url=provider_config.base_url or "https://api.openai.com/v1"
        )
    elif provider_config.type == ProviderType.ANTHROPIC:
        # Return OpenAI-compatible client for Anthropic
        return openai.OpenAI(
            api_key=decrypt_api_key(provider_config.api_key),
            base_url=provider_config.base_url or "https://api.anthropic.com/v1"
        )
    elif provider_config.type == ProviderType.GOOGLE:
        # Return OpenAI-compatible client for Google
        return openai.OpenAI(
            api_key=decrypt_api_key(provider_config.api_key),
            base_url=provider_config.base_url or "https://generativelanguage.googleapis.com/v1"
        )
    elif provider_config.type == ProviderType.HUGGINGFACE:
        # Return OpenAI-compatible client for HuggingFace
        return openai.OpenAI(
            api_key=decrypt_api_key(provider_config.api_key),
            base_url=provider_config.base_url or "https://api-inference.huggingface.co/models"
        )
    elif provider_config.type == ProviderType.CUSTOM:
        # Return OpenAI-compatible client for custom provider
        return openai.OpenAI(
            api_key=decrypt_api_key(provider_config.api_key),
            base_url=provider_config.base_url
        )
```

### 8.3 Model Configuration for Agents

```python
@router.post("/api/v1/agents/{agent_id}/model")
async def set_agent_model(
    agent_id: int,
    model_data: ModelConfig,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Set the model configuration for an agent."""
    # 1. Verify agent exists and user has permission
    # 2. Update agent model configuration
    # 3. Return updated agent
```

## 9. Tracing & Debugging

### 9.1 Trace Storage

```python
@router.get("/api/v1/agents/{agent_id}/traces")
async def get_agent_traces(
    agent_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    limit: int = 10,
    offset: int = 0
):
    """Get execution traces for an agent."""
    # 1. Verify agent exists and user has permission
    # 2. Retrieve traces from database
    # 3. Return traces
```

### 9.2 Trace Viewer API

```python
@router.get("/api/v1/traces/{trace_id}")
async def get_trace(
    trace_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a specific execution trace."""
    # 1. Verify user has permission to view trace
    # 2. Retrieve trace from database
    # 3. Return trace data
```

## 10. File Processing & Vector Search

### 10.1 File Upload and Processing

```python
@router.post("/api/v1/files")
async def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Upload a file for use with agents."""
    # 1. Validate file type
    # 2. Store file in S3 or similar
    # 3. Process file for vector storage
    # 4. Return file metadata
```

### 10.2 Vector Store Integration

```python
async def initialize_vector_store():
    """Initialize connection to vector database."""
    # Connect to Pinecone, Qdrant, or other vector DB
    pass

async def store_file_embeddings(file_id, content, metadata):
    """Process file content and store embeddings."""
    # 1. Split content into chunks
    # 2. Generate embeddings
    # 3. Store in vector database
    pass

async def setup_file_search_tool(vector_store_id, max_results=3):
    """Create a FileSearchTool for the specified vector store."""
    return FileSearchTool(
        vector_store_ids=[vector_store_id],
        max_num_results=max_results
    )
```

## 11. API Schemas & Database Models

### 11.1 SQLAlchemy Models

```python
class Agent(Base):
    __tablename__ = "agents"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    instructions = Column(Text, nullable=False)
    model = Column(String, default="gpt-4o")
    handoff_description = Column(String, nullable=True)
    metadata = Column(JSON, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="agents")
    tools = relationship("Tool", back_populates="agent", cascade="all, delete-orphan")
    guardrails = relationship("Guardrail", back_populates="agent", cascade="all, delete-orphan")
    handoffs = relationship(
        "Agent", 
        secondary="agent_handoffs",
        primaryjoin="Agent.id == AgentHandoff.source_agent_id",
        secondaryjoin="Agent.id == AgentHandoff.target_agent_id",
        backref="handoff_sources"
    )
    contexts = relationship("AgentContext", back_populates="agent", cascade="all, delete-orphan")
    runs = relationship("AgentRun", back_populates="agent")
```

### 11.2 FastAPI Pydantic Models

```python
class AgentRunInput(BaseModel):
    message: str
    context_id: Optional[int] = None
    stream: bool = False
    provider_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

class AgentRunResponse(BaseModel):
    output: Union[str, Dict[str, Any]]
    messages: List[Dict[str, Any]]
    usage: Dict[str, Any]
    trace_id: Optional[str] = None
    run_id: str
    
class ToolCreate(BaseModel):
    name: str
    description: str
    type: ToolType
    params: Dict[str, Any]
    
class GuardrailCreate(BaseModel):
    name: str
    description: str
    type: GuardrailType
    implementation: str
    config: Dict[str, Any]
    
class ModelConfig(BaseModel):
    provider_id: Optional[int] = None
    model: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    additional_params: Optional[Dict[str, Any]] = None
```

## 12. Implementation Timeline

### Phase 1: Core Agent Infrastructure (Weeks 1-3)
- Set up SQLAlchemy models and database
- Implement basic agent CRUD
- Create agent execution engine
- Implement basic tools

### Phase 2: Advanced Agent Features (Weeks 4-6)
- Implement guardrails
- Add custom function tools
- Create handoff functionality
- Develop structured output support

### Phase 3: Multi-Agent Orchestration (Weeks 7-9)
- Implement LLM-based orchestration
- Create code-based workflow engine
- Add conditional branching
- Develop state management between agents

### Phase 4: Provider Integration & File Processing (Weeks 10-12)
- Add multi-provider support
- Implement file upload and processing
- Create vector store integration
- Add file search capability

### Phase 5: Tracing, Testing & Deployment (Weeks 13-14)
- Implement tracing and debugging
- Create comprehensive test suite
- Optimize performance
- Deploy to production

## 13. Conclusion

This implementation plan provides a comprehensive approach to integrating the OpenAI Agents SDK into our FastAPI backend. It covers all core features including agents, tools, guardrails, handoffs, and multi-agent orchestration. The modular design allows for flexibility and extensibility, while the clear API structure ensures easy integration with the frontend. 