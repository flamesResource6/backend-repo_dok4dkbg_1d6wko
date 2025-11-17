import os
import random
import base64
from io import BytesIO
from typing import Optional, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from database import db, create_document, get_documents
from bson import ObjectId

from schemas import (
    Corpus,
    GenerationRequest,
    GenerationResponse,
    CorpusSummary,
    TTSRequest,
    TTSResponse,
)

app = FastAPI(title="Creative N-gram Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# Utility: Char-level N-gram model
# --------------------
class CharNGramModel:
    def __init__(self, order: int = 3):
        self.order = order
        self.model: Dict[str, Dict[str, int]] = {}
        self.starts: List[str] = []

    def train(self, text: str):
        text = text.replace("\r", "")
        if len(text) < self.order + 1:
            raise ValueError("Text too short for the chosen order")
        self.model.clear()
        self.starts.clear()
        for i in range(len(text) - self.order):
            gram = text[i : i + self.order]
            nxt = text[i + self.order]
            self.model.setdefault(gram, {})
            self.model[gram][nxt] = self.model[gram].get(nxt, 0) + 1
            if i == 0 or text[i - 1] in "\n.!?":
                self.starts.append(gram)
        if not self.starts:
            # default fallbacks
            self.starts = list(self.model.keys())[:20]

    def _sample_next(self, gram: str, temperature: float) -> Optional[str]:
        dist = self.model.get(gram)
        if not dist:
            return None
        # temperature sampling
        chars = list(dist.keys())
        counts = [dist[c] for c in chars]
        # apply temperature
        weights = [pow(c, 1.0 / max(1e-6, temperature)) for c in counts]
        total = sum(weights)
        r = random.random() * total
        upto = 0.0
        for ch, w in zip(chars, weights):
            upto += w
            if r <= upto:
                return ch
        return chars[-1]

    def generate(self, length: int, temperature: float = 1.0, seed: Optional[str] = None) -> str:
        if not self.model:
            return ""
        if seed and len(seed) >= self.order:
            current = seed[-self.order :]
            out = seed
        else:
            current = random.choice(self.starts)
            out = current
        while len(out) < length:
            nxt = self._sample_next(current, temperature)
            if nxt is None:
                current = random.choice(self.starts)
                out += current
                continue
            out += nxt
            current = out[-self.order :]
        return out[:length]


# --------------------
# Routes
# --------------------
@app.get("/")
def root():
    return {"message": "Creative N-gram API running"}

@app.get("/test")
def test():
    info = {"backend": "ok", "db": False, "db_name": None}
    try:
        if db is not None:
            info["db"] = True
            info["db_name"] = db.name
            # attempt list
            info["collections"] = db.list_collection_names()
    except Exception as e:
        info["db_error"] = str(e)
    return info

@app.post("/corpus", response_model=CorpusSummary)
def save_corpus(corpus: Corpus):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    _id = create_document("corpus", corpus)
    return CorpusSummary(id=_id, title=corpus.title)

@app.get("/corpus", response_model=List[CorpusSummary])
def list_corpora(limit: int = 50):
    if db is None:
        return []
    docs = get_documents("corpus", {}, limit)
    summaries: List[CorpusSummary] = []
    for d in docs:
        summaries.append(CorpusSummary(id=str(d.get("_id")), title=d.get("title", "Untitled")))
    return summaries

@app.post("/generate", response_model=GenerationResponse)
def generate_text(req: GenerationRequest):
    # Resolve source text
    source_text: Optional[str] = None
    used_corpus_id: Optional[str] = None

    if req.text:
        source_text = req.text
    elif req.corpus_id:
        if db is None:
            raise HTTPException(status_code=400, detail="corpus_id provided but database is not configured")
        try:
            oid = ObjectId(req.corpus_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid corpus_id")
        doc = db["corpus"].find_one({"_id": oid})
        if not doc:
            raise HTTPException(status_code=404, detail="Corpus not found")
        source_text = doc.get("text", "")
        used_corpus_id = req.corpus_id

    if not source_text:
        raise HTTPException(status_code=400, detail="Provide either text or corpus_id")

    try:
        model = CharNGramModel(order=req.order)
        model.train(source_text)
        output = model.generate(length=req.length, temperature=req.temperature, seed=req.seed)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Generation failed")

    # include stylistic hints in meta for transparency
    return GenerationResponse(
        output=output,
        used_corpus_id=used_corpus_id,
        meta={
            "order": req.order,
            "length": req.length,
            "temperature": req.temperature,
            "seed": req.seed or None,
            "genre": getattr(req, "genre", None),
            "flow": getattr(req, "flow", None),
            "bpm": getattr(req, "bpm", None),
            "mood": getattr(req, "mood", None),
            "voice": getattr(req, "voice", None),
            "language": getattr(req, "language", None),
        },
    )


@app.post("/tts", response_model=TTSResponse)
def text_to_speech(tts: TTSRequest):
    try:
        # Lazy import to speed cold start
        from gtts import gTTS  # type: ignore

        # Note: gTTS doesn't support gender selection; we pass language and slow.
        # The 'voice' field is accepted for API symmetry but not used by gTTS.
        mp3_buffer = BytesIO()
        gTTS(text=tts.text, lang=tts.language or "en", slow=tts.slow).write_to_fp(mp3_buffer)
        mp3_buffer.seek(0)
        b64 = base64.b64encode(mp3_buffer.read()).decode("utf-8")
        return TTSResponse(audio_base64=b64, mime_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
