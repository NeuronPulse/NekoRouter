#!/usr/bin/env python3
"""Local OpenAI-compatible embedding server for NekoRouter testing.

This script starts a tiny FastAPI service that mimics the OpenAI embeddings
endpoint, backed by a local sentence-transformers model. It is useful for
running NekoRouter without calling a paid embedding API.

Dependencies:
    pip install fastapi uvicorn sentence-transformers

Usage:
    python scripts/local_embedding_server.py

Then configure NekoRouter in config/local.toml:

    [embedding]
    base_url = "http://127.0.0.1:8000/v1"
    model = "local"
    api_key = "local"

    [qdrant]
    url = "http://127.0.0.1:6333"
    vector_dim = 512   # must match the model output dimension

The default model is `BAAI/bge-small-zh-v1.5` (512 dims). You can override it
with `--model`.
"""

import argparse
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

app = FastAPI(title="NekoRouter Local Embedding Server")

_DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
_model: SentenceTransformer | None = None


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: dict


class EmbeddingsRequest(BaseModel):
    model: str
    input: List[str]


@app.post("/v1/embeddings")
async def embeddings(req: EmbeddingsRequest) -> EmbeddingsResponse:
    assert _model is not None, "model not loaded"
    texts = req.input
    vectors = _model.encode(texts, normalize_embeddings=True)
    data = [
        EmbeddingData(
            embedding=vectors[i].tolist(),
            index=i,
        )
        for i in range(len(texts))
    ]
    return EmbeddingsResponse(
        data=data,
        model=req.model,
        usage={"prompt_tokens": 0, "total_tokens": 0},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a local OpenAI-compatible embedding server"
    )
    parser.add_argument("--host", default="0.0.0.0", help="bind host")
    parser.add_argument("--port", type=int, default=8000, help="bind port")
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help="sentence-transformers model name or path",
    )
    args = parser.parse_args()

    print(f"Loading embedding model: {args.model}")
    _model = SentenceTransformer(args.model)
    print(f"Model loaded, dimension: {_model.get_sentence_embedding_dimension()}")
    print(f"Server listening on http://{args.host}:{args.port}/v1/embeddings")
    uvicorn.run(app, host=args.host, port=args.port)
