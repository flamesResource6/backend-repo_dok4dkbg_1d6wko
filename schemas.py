from pydantic import BaseModel, Field
from typing import Optional, List

# Collections
class Corpus(BaseModel):
    title: str = Field(..., description="Display name for this corpus")
    text: str = Field(..., min_length=1, description="Raw text used to train the generator")

# Requests
class GenerationRequest(BaseModel):
    text: Optional[str] = Field(None, description="Raw text to generate from. Mutually exclusive with corpus_id")
    corpus_id: Optional[str] = Field(None, description="Existing corpus to generate from")
    length: int = Field(200, ge=1, le=2000)
    order: int = Field(3, ge=1, le=10)
    temperature: float = Field(1.0, gt=0.0, le=2.5)
    seed: Optional[str] = Field(None, description="Optional starting string")

class GenerationResponse(BaseModel):
    output: str
    used_corpus_id: Optional[str] = None
    meta: dict = {}

# Responses
class CorpusSummary(BaseModel):
    id: str
    title: str
    created_at: Optional[str] = None
