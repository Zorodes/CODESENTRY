import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field

class Finding(BaseModel):
    category: str  # "bug_risk" | "convention" | "test_coverage"
    severity: str = Field(description="Severity of the finding (e.g., High, Medium, Low)")
    description: str = Field(description="Detailed description of the issue")
    source_chunk_ids: list[str] = Field(
        description="List of citation IDs supporting this finding, e.g., ['[code:1]', '[precedent:4]']"
    )

class ReviewState(TypedDict):
    diff: str
    repo_owner: str
    repo_name: str
    
    category: str
    
    code_chunks: list[dict]
    precedents: list[dict]
    
    bug_risk_findings: Annotated[list[Finding], operator.add]
    convention_findings: Annotated[list[Finding], operator.add]
    test_coverage_findings: Annotated[list[Finding], operator.add]
    
    verified_findings: list[Finding]
    final_review: str
