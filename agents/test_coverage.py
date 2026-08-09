from pydantic import BaseModel
from langchain_groq import ChatGroq
from graph_state import ReviewState, Finding
from agents.utils import build_code_context, build_precedent_context

class TestCoverageOutput(BaseModel):
    findings: list[Finding]

def test_coverage_node(state: ReviewState) -> dict:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
    structured_llm = llm.with_structured_output(TestCoverageOutput)
    
    code_ctx = build_code_context(state.get("code_chunks", []))
    precedent_ctx = build_precedent_context(state.get("precedents", []))
    
    prompt = f"""You are an expert code reviewer checking for missing test coverage.
    
## Diff under review
```diff
{state['diff']}
```

## Retrieved code chunks
{code_ctx or 'None retrieved.'}

## Retrieved PR precedents
{precedent_ctx or 'None retrieved.'}

Your task: Return a list of findings representing missing test coverage or testing issues.
Rules:
- Every finding MUST cite a specific [code:ID] or [precedent:ID] in source_chunk_ids.
- Do not invent line numbers or PR numbers.
- If there are no test coverage issues, return an empty list of findings.
"""
    result = structured_llm.invoke(prompt)
    return {"test_coverage_findings": result.findings}
