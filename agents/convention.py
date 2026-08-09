from pydantic import BaseModel
from langchain_groq import ChatGroq
from graph_state import ReviewState, Finding
from agents.utils import build_code_context, build_precedent_context

class ConventionOutput(BaseModel):
    findings: list[Finding]

def convention_node(state: ReviewState) -> dict:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
    structured_llm = llm.with_structured_output(ConventionOutput)
    
    code_ctx = build_code_context(state.get("code_chunks", []))
    precedent_ctx = build_precedent_context(state.get("precedents", []))
    
    prompt = f"""You are an expert code reviewer checking for codebase convention consistency and style deviations.
    
## Diff under review
```diff
{state['diff']}
```

## Retrieved code chunks
{code_ctx or 'None retrieved.'}

## Retrieved PR precedents
{precedent_ctx or 'None retrieved.'}

Your task: Return a list of findings representing convention deviations.
Rules:
- Only flag a convention issue if the retrieved code chunks or precedents genuinely
  establish a pattern that the diff's code deviates from.
- If the retrieved context is not meaningfully related to the code being reviewed
  (different domain, unrelated functionality), do NOT force a comparison — return
  an empty list of findings instead.
- Every finding MUST cite a specific [code:ID] or [precedent:ID], and that cited
  chunk's content must actually demonstrate the pattern you're claiming exists.
- It is correct and expected to return zero findings when there's nothing genuinely
  comparable in the retrieved context — do not manufacture a finding just to have
  output.
"""
    result = structured_llm.invoke(prompt)
    return {"convention_findings": result.findings}
