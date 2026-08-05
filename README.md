# CodeSentry 🛡️

**Multi-Agent Codebase Intelligence & PR Review Copilot**

CodeSentry is an production-grade, agentic code review platform built with **LangGraph**. It points at any public GitHub repository, builds a high-fidelity semantic index of the full codebase (via AST-aware chunking), ingests commit history and past PR review discussions, and executes a multi-agent pipeline to review incoming Pull Requests with senior-engineer precision.

Unlike generic LLM diff summarizers that offer surface-level feedback, CodeSentry understands your codebase's unique architectural patterns, team conventions, and historical precedent—citing exact code references and past PR decisions for every claim it makes.

---

## 🌟 Key Features & Highlights

- **AST-Aware Code Ingestion**: Splitting code along meaningful boundaries (functions, classes, interfaces using Tree-Sitter) rather than arbitrary line counts.
- **Precedent-Aware Retrieval**: Combines hybrid RAG over the codebase with a historical memory of closed PR comments to reflect "how this team usually reviews code."
- **Multi-Agent LangGraph Pipeline**: Specialized agents independently evaluate bug risks, structural conventions, and test coverage gaps.
- **Strict Hallucination Guardrail (Critic Agent)**: Automatically verifies and drops any agent claim that cannot be anchored directly to an actual line in the retrieved codebase or PR diff.
- **Live GitHub Webhook Integration**: Event-driven architecture using FastAPI to automatically post review comments directly onto live PRs in real time.
- **Golden Evaluation Suite**: Evaluated against historical human-reviewed PRs using Ragas and custom metrics (precision/recall on bug flags).
- **Full Observability & LLMOps**: End-to-end tracing, prompt versioning, and per-review cost tracking via Langfuse / LangSmith.

---

## 🏗️ Architecture & Multi-Agent Workflow

CodeSentry's core workflow is modeled as a stateful, directed graph orchestrated by **LangGraph**.

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

### Agent Roles

1. **Ingestion Agent**: Clones the target repo, extracts syntax trees via Tree-Sitter, chunks code cleanly along AST boundaries, and embeds vectors into `pgvector`. It also ingests closed PR discussions and commit history.
2. **Router Agent**: Analyzes incoming PR diffs to classify scope and dynamically dispatch tasks to specialist agents (e.g., bug risk analysis, style/convention checking, architecture review, test coverage gaps).
3. **Retrieval Agent**: Performs hybrid RAG (dense vector similarity + keyword search) over both the codebase vectors and the team's historical review comment database.
4. **Bug-Risk Specialist**: Performs static-pattern recognition combined with LLM reasoning over the diff against retrieved similar implementations.
5. **Convention Specialist**: Verifies new changes against established project idioms (naming conventions, error handling strategies, modular structure).
6. **Critic / Verifier Agent**: Acts as an automated hallucination guardrail. Every claim made by specialist agents must explicitly reference a valid line in the retrieved context or diff; unverified claims are filtered out.
7. **Review Writer Agent**: Synthesizes verified feedback into a clean, markdown-formatted PR comment and posts it directly back to GitHub via API.

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

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- PostgreSQL with `pgvector` extension enabled
- Docker & Docker Compose (optional)
- GitHub Personal Access Token (or GitHub App credentials)
- OpenAI / Anthropic API Key / Groq 

### Installation

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
   Copy `.env.example` to `.env` and fill in your keys:
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

4. **Initialize Database**:
   ```bash
   python db.py
   ```

5. **Run FastAPI Webhook Server**:
   ```bash
   uvicorn main:app --reload --port 8000
   ```
