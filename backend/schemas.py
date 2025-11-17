from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Corpus(BaseModel):
    title: str
    content: str
    type: Optional[str] = Field(default="generic", description="Category of the corpus: song, poem, lyrics, etc.")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class GenerationRequest(BaseModel):
    corpus_id: Optional[str] = Field(default=None, description="Use an existing saved corpus by id")
    raw_text: Optional[str] = Field(default=None, description="Alternatively, generate from this raw text without saving")
    length: int = Field(default=200, ge=1, le=5000)
    temperature: float = Field(default=0.9, gt=0.01, le=2.5)
    order: int = Field(default=3, ge=1, le=8, description="Markov order (n-gram size)")
    seed: Optional[str] = Field(default=None, description="Optional starting seed text")
