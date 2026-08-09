from langgraph.graph import StateGraph, END
from graph_state import ReviewState

from agents.router import router_node
from agents.retrieval import retrieval_node
from agents.bug_risk import bug_risk_node
from agents.convention import convention_node
from agents.test_coverage import test_coverage_node
from agents.critic import critic_node
from agents.writer import writer_node

def route_specialists(state: ReviewState):
     # Always run all three specialists. category is still available in state
    # for logging/future use, but gating specialist execution on the
    # router's single-label guess caused false negatives — a diff classified
    # "bug_risk" still had a genuine convention issue that got silently
    # skipped. convention/test_coverage already degrade to zero findings
    # when nothing applies, so always running them just costs a couple
    # extra LLM calls, not a correctness risk.Reverted back to using all three agents will think of a solution to this later..
        return ["bug_risk", "convention", "test_coverage"]

workflow = StateGraph(ReviewState)

workflow.add_node("router", router_node)
workflow.add_node("retrieval", retrieval_node)
workflow.add_node("bug_risk", bug_risk_node)
workflow.add_node("convention", convention_node)
workflow.add_node("test_coverage", test_coverage_node)
workflow.add_node("critic", critic_node)
workflow.add_node("writer", writer_node)

workflow.set_entry_point("router")

workflow.add_edge("router", "retrieval")

workflow.add_conditional_edges(
    "retrieval", 
    route_specialists, 
    ["bug_risk", "convention", "test_coverage"]
)

workflow.add_edge("bug_risk", "critic")
workflow.add_edge("convention", "critic")
workflow.add_edge("test_coverage", "critic")
workflow.add_edge("critic", "writer")
workflow.add_edge("writer", END)

compiled_graph = workflow.compile()
