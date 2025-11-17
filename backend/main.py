from datetime import datetime
import hashlib
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bson import ObjectId
from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection
from dotenv import load_dotenv

from schemas import Corpus, GenerationRequest

load_dotenv()

MONGO_URL = os.getenv("DATABASE_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("DATABASE_NAME", "appdb")

client = MongoClient(MONGO_URL)
db = client[DB_NAME]

app = FastAPI(title="Music & Poetry Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def collection(name: str) -> Collection:
    coll = db[name]
    # Helpful index for lookups
    coll.create_index([("created_at", ASCENDING)], background=True)
    return coll


# Utilities

def oid_str(oid: Any) -> str:
    return str(oid) if isinstance(oid, ObjectId) else oid


def now_ts() -> datetime:
    return datetime.utcnow()


# Simple character-level n-gram model (Markov chain)
class CharNGramModel:
    def __init__(self, order: int = 3):
        self.order = order
        self.table: Dict[str, Dict[str, int]] = {}

    def fit(self, text: str):
        if len(text) < self.order + 1:
            return
        for i in range(len(text) - self.order):
            gram = text[i : i + self.order]
            nxt = text[i + self.order]
            bucket = self.table.setdefault(gram, {})
            bucket[nxt] = bucket.get(nxt, 0) + 1

    def sample_next(self, gram: str, temperature: float = 1.0) -> Optional[str]:
        import math, random

        dist = self.table.get(gram)
        if not dist:
            return None
        # Convert counts to logits then apply temperature
        chars = list(dist.keys())
        counts = [dist[c] for c in chars]
        total = sum(counts)
        probs = [c / total for c in counts]
        # Gumbel-softmax style sampling
        g = []
        for p in probs:
            if p <= 0:
                g.append(float("-inf"))
            else:
                import random
                u = max(1e-8, random.random())
                g.append(math.log(p) - math.log(-math.log(u)))
        # temperature scaling
        g = [x / max(1e-6, temperature) for x in g]
        idx = max(range(len(g)), key=lambda i: g[i])
        return chars[idx]

    def generate(self, length: int, seed: Optional[str] = None, temperature: float = 1.0) -> str:
        import random

        if not self.table:
            return seed or ""
        if seed is None or len(seed) < self.order or seed[: self.order] not in self.table:
            seed_candidates = list(self.table.keys())
            seed = random.choice(seed_candidates)
        out = seed
        gram = out[-self.order :]
        while len(out) < length:
            nxt = self.sample_next(gram, temperature=temperature)
            if nxt is None:
                # reset with random gram
                seed_candidates = list(self.table.keys())
                gram = random.choice(seed_candidates)
                out += " "
                continue
            out += nxt
            gram = out[-self.order :]
        return out


# Routes

@app.get("/")
def root():
    return {"message": "Music & Poetry Generator API running"}


@app.get("/test")
def test_db():
    try:
        db.command("ping")
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


class CorpusOut(BaseModel):
    id: str
    title: str
    type: str
    created_at: datetime
    updated_at: Optional[datetime]


@app.post("/corpus", response_model=CorpusOut)
def add_corpus(item: Corpus):
    doc = item.model_dump()
    ts = now_ts()
    doc["created_at"] = ts
    doc["updated_at"] = ts
    res = collection("corpus").insert_one(doc)
    return CorpusOut(
        id=str(res.inserted_id),
        title=doc["title"],
        type=doc.get("type", "generic"),
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
    )


@app.get("/corpus", response_model=List[CorpusOut])
def list_corpus(limit: int = 50):
    cur = collection("corpus").find({}, {"content": 0}).sort("created_at", -1).limit(limit)
    return [
        CorpusOut(
            id=str(d["_id"]),
            title=d.get("title", "Untitled"),
            type=d.get("type", "generic"),
            created_at=d.get("created_at", now_ts()),
            updated_at=d.get("updated_at"),
        )
        for d in cur
    ]


@app.get("/corpus/{corpus_id}")
def get_corpus(corpus_id: str):
    d = collection("corpus").find_one({"_id": ObjectId(corpus_id)})
    if not d:
        raise HTTPException(status_code=404, detail="Not found")
    d["id"] = str(d["_id"]) 
    del d["_id"]
    return d


@app.post("/generate")
def generate_text(req: GenerationRequest):
    # Fetch corpus text
    text_source: Optional[str] = None
    if req.corpus_id:
        d = collection("corpus").find_one({"_id": ObjectId(req.corpus_id)})
        if not d:
            raise HTTPException(status_code=404, detail="Corpus not found")
        text_source = d.get("content", "")
    elif req.raw_text:
        text_source = req.raw_text
    else:
        raise HTTPException(status_code=400, detail="Provide corpus_id or raw_text")

    # Build model
    order = req.order
    model = CharNGramModel(order=order)
    model.fit(text_source)

    seed = req.seed if req.seed else None
    out = model.generate(length=req.length, seed=seed, temperature=req.temperature)

    # Save generation record
    gen_doc = {
        "source_type": "corpus_id" if req.corpus_id else "raw_text",
        "corpus_id": req.corpus_id,
        "length": req.length,
        "temperature": req.temperature,
        "order": req.order,
        "seed": req.seed,
        "created_at": now_ts(),
        "preview": out[:200],
    }
    collection("generation").insert_one(gen_doc)

    return {"result": out}
