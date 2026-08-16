from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from graph_state import ReviewState, Finding
from agents.utils import build_code_context, build_precedent_context
from llm_utils import invoke_llm

class BugRiskOutput(BaseModel):
    findings: list[Finding]

def bug_risk_node(state: ReviewState) -> dict:
    llm = ChatGoogleGenerativeAI(model="models/gemini-3.5-flash", temperature=0.1)
    structured_llm = llm.with_structured_output(BugRiskOutput)
    
    code_ctx = build_code_context(state.get("code_chunks", []))
    precedent_ctx = build_precedent_context(state.get("precedents", []))
    
    prompt = f"""You are an expert code reviewer checking for bug risks, logic errors, security issues, or race conditions.
    
## Diff under review
```diff
{state['diff']}
```

## Retrieved code chunks
{code_ctx or 'None retrieved.'}

## Retrieved PR precedents
{precedent_ctx or 'None retrieved.'}

Your task: Return a list of findings representing bug risks. 
Rules:
- If the issue is in newly added/changed code from the diff itself, cite it as [diff] in source_chunk_ids.
- If the issue relates to existing code you retrieved, cite the specific [code:ID] or [precedent:ID].
- A finding can cite both [diff] and a [code:ID]/[precedent:ID] if both are relevant.
- Do not invent chunk IDs that weren't given to you above.
- If there are no bug risks, return an empty list of findings.
"""
    # Instruct the model to format output according to the Pydantic schema.
    result = invoke_llm(structured_llm, prompt)
    return {"bug_risk_findings": result.findings}
