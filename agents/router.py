from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from graph_state import ReviewState
from llm_utils import invoke_llm

class RouterOutput(BaseModel):
    category: str = Field(description="One of: 'bug_risk', 'style', 'test_gap', 'mixed'")

def router_node(state: ReviewState) -> dict:
    llm = ChatGoogleGenerativeAI(model="models/gemini-3.5-flash", temperature=0.0)
    structured_llm = llm.with_structured_output(RouterOutput)
    
    prompt = f"""You are a PR routing assistant. Classify the following diff into one of the following categories:
- 'bug_risk': if the diff mainly fixes bugs or introduces complex logic changes.
- 'style': if the diff mainly changes conventions, formatting, or style.
- 'test_gap': if the diff mainly adds or modifies tests.
- 'mixed': if the diff does multiple things or doesn't neatly fit one category.

Diff:
{state['diff']}
"""
    
    result = invoke_llm(structured_llm, prompt)
    return {"category": result.category}
#this is a tradeoff the router isnt on the function rn changed back to using all three agents