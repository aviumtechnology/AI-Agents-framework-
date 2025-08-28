# Autonomous Coding Agent — Architecture

## Components
- **Planner** → uses LLM + RAG to break user goals into steps.
- **Executor** → executes plans using tools (compiling, testing, debugging).
- **ResearchAgent** → performs web search, ingests findings into RAG.
- **RAG (ChromaDB)** → persistent memory store, organized by `library/` namespaces.
- **Tools**:
  - Web Search (DuckDuckGo)
  - Filesystem (scoped to project workspaces)
  - Docker Sandbox (safe code execution)
  - MCP Adapter (exposes tools to MCP ecosystem)
- **CI/CD**:
  - Self-improvement guarded by `agent/ci/agent_ci.py`.
  - Projects validated via sandbox tests.

## Workflow
1. Research → results stored in RAG (namespace = `research.*` or `project.*`).
2. Planning → Planner uses LLM + RAG to generate a step plan.
3. Execution → Executor:
   - Writes code
   - Compiles/tests
   - Logs results in `compile_run.log` & `debug.log`
4. Self-improvement → Agent can modify its own code but must pass CI pipeline.
