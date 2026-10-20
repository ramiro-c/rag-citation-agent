"""Data models shared by the RAG citation agent."""

from pydantic import BaseModel, Field


class Source(BaseModel):
    """A retrieved document fragment used as grounding context."""

    doc_id: str
    text: str
    score: float = Field(ge=0.0, le=1.0)


class Citation(BaseModel):
    """A claim's grounding: the source doc and the exact supporting quote."""

    doc_id: str
    quote: str


class CitedAnswer(BaseModel):
    """The agent's final output: always carries sources it was grounded on."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    low_confidence: bool = False
    insufficient_evidence: bool = False
