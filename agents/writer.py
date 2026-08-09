import json
from langchain_groq import ChatGroq
from graph_state import ReviewState

def writer_node(state: ReviewState) -> dict:
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)
    
    verified_findings = state.get("verified_findings", [])
    
    if not verified_findings:
        return {"final_review": "No significant issues found."}
        
    findings_json = json.dumps([f.model_dump() for f in verified_findings], indent=2)
    
    prompt = f"""You are an expert code reviewer. Write a final markdown PR review based on the following verified findings.

## Verified Findings
```json
{findings_json}
```

Your review should have exactly these sections, if applicable:
## Bug Risk
## Convention Consistency
## Test Coverage

Under each section, present the findings as clear, concise bullet points.
Include the citations (e.g., [code:1], [precedent:2]) directly in your bullet points.
Do NOT invent any findings that are not in the JSON.
Do NOT add any sections beyond the three listed above — no "Additional Recommendations",
no "Approval Status", no summary commentary, no suggested next steps. Only restate
what is in the verified findings JSON, organized into those three sections.
If a section has no matching findings in the JSON, write "No findings in this category."
under that heading and nothing else.
"""
    
    result = llm.invoke(prompt)
    return {"final_review": result.content}
