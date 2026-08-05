# CodeSentry 🛡️

**Multi-Agent Codebase Intelligence & PR Review Copilot**

CodeSentry is an active engineering initiative building a production-grade, agentic code review platform powered by **LangGraph**. Designed to point at any public GitHub repository, CodeSentry constructs a high-fidelity semantic index of the codebase using AST-aware chunking, ingests commit history and past PR review discussions, and executes a multi-agent pipeline to review incoming Pull Requests with senior-engineer precision.

Unlike generic LLM diff summarizers that provide shallow feedback, CodeSentry is designed to understand codebase-specific architectural patterns, team conventions, and historical precedent—citing exact code references for every claim.

> 🚧 **Work in Progress**: This project is actively under development. Features, multi-agent graphs, and evaluation suites are being built according to our roadmap below.

---

## 🌟 Planned & Core Features

- **AST-Aware Code Ingestion**: Chunking code along syntactic boundaries (functions, classes, interfaces via Tree-Sitter) rather than arbitrary line counts.
- **Precedent-Aware Retrieval**: Hybrid RAG over codebase vectors (`pgvector`) combined with historical PR review comments.
- **Multi-Agent LangGraph Pipeline**: Dedicated agents independently reviewing bug risks, structural conventions, and test coverage gaps.
- **Strict Hallucination Guardrail (Critic Agent)**: Automated verification step that drops any agent claim that cannot be grounded directly in the retrieved codebase or PR diff.
- **Live GitHub Webhook Integration**: FastAPI event listener to post automated review comments directly on live PRs.
- **Golden Evaluation Suite**: Benchmark pipeline using Ragas against historical human-reviewed PRs.
- **LLMOps & Observability**: End-to-end execution tracing, prompt versioning, and cost tracking via Langfuse / LangSmith.

---

## 🏗️ Architecture & Multi-Agent Workflow

The system architecture is structured as a stateful graph orchestrated by **LangGraph**:

```
                           +-------------------+
                           |  GitHub Webhook   |
                           +---------+---------+
                                     |
                                     v
                           +-------------------+
                           |  Ingestion Agent  |
                           | (Tree-Sitter RAG) |
                           +---------+---------+
                                     |
                                     v
                           +-------------------+
                           |   Router Agent    |
                           +---------+---------+
                                     |
             +-----------------------+-----------------------+
             |                       |                       |
             v                       v                       v
   +------------------+    +-------------------+    +------------------+
   |  Bug-Risk Agent  |    | Convention Agent  |    |  Test-Gap Agent  |
   +---------+--------+    +---------+---------+    +--------+---------+
             |                       |                       |
             +-----------------------+-----------------------+
                                     |
                                     v
                           +-------------------+
                           |  Critic / Verifier|
                           |  (Guardrail Node) |
                           +---------+---------+
                                     |
                                     v
                           +-------------------+
                           | Review Writer     |
                           | (GitHub API Sync) |
                           +-------------------+
```

---

## 🛠️ Tech Stack

- **Orchestration**: LangGraph, LangChain
- **AST Parsing & Chunking**: `tree-sitter`
- **Vector Database**: PostgreSQL with `pgvector`
- **API & Webhook Gateway**: FastAPI, Uvicorn
- **LLMOps & Observability**: Langfuse / LangSmith
- **Evaluation**: Ragas, Pytest
- **Containerization**: Docker, Docker Compose

---

## 🗺️ Development Roadmap

- [x] **Phase 1: Ingestion Pipeline** — GitHub API ingestion, AST-aware chunking, vector embedding into `pgvector`.
- [x] **Phase 2: Baseline Retrieval Agent** — Single-agent RAG setup to sanity check retrieval quality over code diffs.
- [ ] **Phase 3: Multi-Agent Graph** — Full LangGraph implementation with Router, Specialist (Bug-Risk, Convention, Test-Gap), and Critic agents.
- [ ] **Phase 4: Golden Eval Suite** — Curating ~50 historical PRs and scoring agent performance via Ragas (precision/recall on bug-flags).
- [ ] **Phase 5: Webhooks & Observability** — FastAPI event handler for live GitHub comments and Langfuse tracing.
- [ ] **Phase 6: Deployment & Polish** — Docker containerization, performance tuning, and demo showcase.

---

## 🚀 Local Setup (Development)

### Prerequisites

- Python 3.10+
- PostgreSQL with `pgvector` extension enabled
- OpenAI / Anthropic API Key
- GitHub Personal Access Token

### Environment Configuration

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/CodeSentry.git
   cd CodeSentry
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and set your key values:
   ```bash
   cp .env.example .env
   ```
   Example `.env`:
   ```env
   DATABASE_URL=postgresql://user:password@localhost:5432/codesentry
   GITHUB_TOKEN=ghp_your_github_token
   OPENAI_API_KEY=sk-...
   LANGFUSE_PUBLIC_KEY=pk-...
   LANGFUSE_SECRET_KEY=sk-...
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```

---