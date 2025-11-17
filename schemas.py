from pydantic import BaseModel, Field
from typing import Optional, List, Literal

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
    # Style controls (used for hinting and TTS defaults)
    genre: Optional[str] = Field(None, description="Genre hint, e.g., hip-hop, pop, jazz")
    flow: Optional[str] = Field(None, description="Flow hint, e.g., smooth, rapid, storytelling")
    bpm: Optional[int] = Field(None, ge=40, le=240, description="Beats per minute hint")
    mood: Optional[str] = Field(None, description="Mood hint, e.g., happy, moody, epic")
    voice: Optional[Literal['female','male']] = Field('female', description="Preferred voice for TTS if used")
    language: Optional[str] = Field('en', description="Language code for TTS, e.g., en, en-uk, hi, es")

class GenerationResponse(BaseModel):
    output: str
    used_corpus_id: Optional[str] = None
    meta: dict = {}

# Responses
class CorpusSummary(BaseModel):
    id: str
    title: str
    created_at: Optional[str] = None

# TTS
class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: Optional[Literal['female','male']] = Field('female')
    language: Optional[str] = Field('en')
    slow: Optional[bool] = Field(False)

class TTSResponse(BaseModel):
    audio_base64: str
    mime_type: str = 'audio/mpeg'
