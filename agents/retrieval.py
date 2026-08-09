import embedder
import db
from graph_state import ReviewState

CODE_TOP_K = 5
PRECEDENT_TOP_K = 3

def retrieval_node(state: ReviewState) -> dict:
    diff_text = state["diff"]
    query_embedding = embedder.embed_one(diff_text[:4096])
    
    code_chunks = db.hybrid_search_code(
        query_embedding=query_embedding,
        query_text=diff_text,
        top_k=CODE_TOP_K,
    )
    precedents = db.hybrid_search_precedents(
        query_embedding=query_embedding,
        query_text=diff_text,
        top_k=PRECEDENT_TOP_K,
    )
    
    def _strip(row: dict) -> dict:
        return {k: v for k, v in row.items() if k != "embedding"}
        
    return {
        "code_chunks": [_strip(c) for c in code_chunks],
        "precedents": [_strip(p) for p in precedents]
    }
