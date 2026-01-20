import logging
import os
import sys
import asyncio
from collections import OrderedDict
from typing import List, Optional, Tuple
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
DEEPWIKI_OPEN_DIR = os.path.join(REPO_ROOT, "tools", "rag", "deepwiki_server", "deepwiki-open")

load_dotenv(dotenv_path=os.path.join(DEEPWIKI_OPEN_DIR, ".env"))

for p in [REPO_ROOT, DEEPWIKI_OPEN_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from api.rag import RAG

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Retrieve API",
    description="Non-intrusive API exposing RAG retrieval only",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


RAG_CACHE_MAX_SIZE = int(os.getenv("RAG_CACHE_MAX_SIZE", "8"))
_rag_cache_lock: "asyncio.Lock" = asyncio.Lock()
_rag_cache: "OrderedDict[str, RAG]" = OrderedDict()


def _normalize_list(items: Optional[List[str]]) -> Tuple[str, ...]:
    if not items:
        return tuple()
    return tuple(sorted([s.strip() for s in items if isinstance(s, str) and s.strip()]))


def _make_cache_key(
    provider: str,
    repo_url: str,
    repo_type: Optional[str],
    excluded_dirs: Optional[List[str]],
    excluded_files: Optional[List[str]],
    included_dirs: Optional[List[str]],
    included_files: Optional[List[str]],
) -> str:
    nd = _normalize_list(excluded_dirs)
    nf = _normalize_list(excluded_files)
    idirs = _normalize_list(included_dirs)
    ifiles = _normalize_list(included_files)
    parts = [provider or "", repo_type or "", repo_url or "", "|d|", *nd, "|f|", *nf, "|id|", *idirs, "|if|", *ifiles]
    return "\u241F".join(parts)  # use a rarely used separator


async def get_or_create_prepared_rag(
    provider: str,
    repo_url: str,
    repo_type: Optional[str],
    excluded_dirs: Optional[List[str]],
    excluded_files: Optional[List[str]],
    included_dirs: Optional[List[str]],
    included_files: Optional[List[str]],
) -> RAG:
    key = _make_cache_key(provider, repo_url, repo_type, excluded_dirs, excluded_files, included_dirs, included_files)
    async with _rag_cache_lock:
        cached = _rag_cache.get(key)
        if cached is not None:
            _rag_cache.move_to_end(key)
            return cached

        rag = RAG(provider=provider)
        rag.prepare_retriever(repo_url, repo_type, None, excluded_dirs, excluded_files, included_dirs, included_files)

        _rag_cache[key] = rag
        # Simple LRU eviction
        while len(_rag_cache) > RAG_CACHE_MAX_SIZE:
            _rag_cache.popitem(last=False)
        return rag


class RagRetrieveRequest(BaseModel):
    repo_url: str = Field(..., description="URL of the repository to query (affected by 'type')")
    query: str = Field(..., description="Natural-language query. Ignored if 'filePath' is provided.")
    type: Optional[str] = Field("github", description="Repository type: github | gitlab | bitbucket")
    filePath: Optional[str] = Field(None, description="If provided, overrides 'query' with 'Contexts related to {filePath}'. This steers retrieval but is not a hard filter.")

    excluded_dirs: Optional[str] = Field(None, description="Newline-separated directories to exclude (URL-decoded)")
    excluded_files: Optional[str] = Field(None, description="Newline-separated file patterns to exclude (URL-decoded)")
    included_dirs: Optional[str] = Field(None, description="Newline-separated directories to include exclusively (whitelist; URL-decoded)")
    included_files: Optional[str] = Field(None, description="Newline-separated file patterns to include exclusively (whitelist; URL-decoded)")

    top_k: Optional[int] = Field(8, description="Max number of documents to return after deduplication and snippet truncation")
    min_score: Optional[float] = Field(None, description="Keep docs with score >= threshold when scores exist; not normalized to [0,1]; distance metrics not supported.")
    deduplicate: Optional[bool] = Field(True, description="Deduplicate results by (file_path, text)")
    style: Optional[str] = Field("fastrag", description="Rendering style for context_text only: fastrag | slowrag | simple")
    return_format: Optional[str] = Field("context_text", description="Which fields to return: documents | context_text | both")
    snippet_chars: Optional[int] = Field(2048, description="Max characters per document text in response")


class RagRetrieveResponse(BaseModel):
    documents: Optional[List[dict]] = None
    context_text: Optional[str] = None
    meta: Optional[dict] = None


@app.post("/rag/retrieve")
async def rag_retrieve(request: RagRetrieveRequest) -> RagRetrieveResponse:
    try:
        # Use openai provider to align with current generator.json (google removed)
        provider = "openai"

        excluded_dirs = [unquote(d) for d in request.excluded_dirs.split('\n') if d.strip()] if request.excluded_dirs else None
        excluded_files = [unquote(f) for f in request.excluded_files.split('\n') if f.strip()] if request.excluded_files else None
        included_dirs = [unquote(d) for d in request.included_dirs.split('\n') if d.strip()] if request.included_dirs else None
        included_files = [unquote(f) for f in request.included_files.split('\n') if f.strip()] if request.included_files else None

        request_rag = await get_or_create_prepared_rag(
            provider,
            request.repo_url,
            request.type,
            excluded_dirs,
            excluded_files,
            included_dirs,
            included_files,
        )

        effective_query = request.query
        if request.filePath:
            effective_query = f"Contexts related to {request.filePath}"

        retrieved = request_rag(effective_query, language="en")

        documents_list: List[dict] = []
        context_text = ""

        if retrieved and getattr(retrieved[0], 'documents', None):
            ret0 = retrieved[0]
            docs = ret0.documents
            score_array = None
            for attr in ["scores", "similarities", "similarity_scores", "distances"]:
                arr = getattr(ret0, attr, None)
                if arr is not None:
                    score_array = arr
                    break

            temp_items = []
            for idx, doc in enumerate(docs):
                file_path = getattr(doc, 'meta_data', {}).get('file_path', 'unknown') if hasattr(doc, 'meta_data') else 'unknown'
                text = getattr(doc, 'text', '')
                score = None
                if score_array is not None and idx < len(score_array):
                    score = score_array[idx]
                temp_items.append({"file_path": file_path, "text": text, "score": score})

            if request.min_score is not None and temp_items and any(i["score"] is not None for i in temp_items):
                temp_items = [i for i in temp_items if i["score"] is not None and i["score"] >= request.min_score]

            if request.deduplicate:
                seen = set()
                deduped = []
                for i in temp_items:
                    key = (i["file_path"], i["text"]) if isinstance(i["text"], str) else (i["file_path"], str(i["text"]))
                    if key in seen:
                        continue
                    seen.add(key)
                    deduped.append(i)
                temp_items = deduped

            if isinstance(request.snippet_chars, int) and request.snippet_chars > 0:
                for i in temp_items:
                    t = i.get("text", "")
                    if isinstance(t, str) and len(t) > request.snippet_chars:
                        i["text"] = t[:request.snippet_chars]

            if isinstance(request.top_k, int) and request.top_k > 0:
                temp_items = temp_items[:request.top_k]

            documents_list = temp_items

            if request.style == "slowrag":
                docs_by_file = {}
                for i in temp_items:
                    docs_by_file.setdefault(i["file_path"], []).append(i["text"]) 
                parts = []
                for file_path, texts in docs_by_file.items():
                    header = f"## File Path: {file_path}\n\n"
                    content = "\n\n".join(texts)
                    parts.append(f"{header}{content}")
                if parts:
                    context_text = "\n\n" + "-" * 10 + "\n\n".join(parts)
            elif request.style == "fastrag":
                lines = []
                for idx, i in enumerate(temp_items):
                    source = i.get("file_path", "unknown")
                    content = i.get("text", "")
                    lines.append(f"DOC {idx + 1}(Source: {source}):\n")
                    lines.append(f"{content}\n")
                context_text = ("".join([line + ("\n" if not line.endswith("\n") else "") for line in lines])).strip()
            else:  # simple
                context_text = "\n\n".join([i["text"] for i in temp_items])

        result: dict = {}
        if request.return_format in ("both", "documents"):
            result["documents"] = documents_list
        if request.return_format in ("both", "context_text"):
            result["context_text"] = context_text
        result["meta"] = {
            "total_docs": len(documents_list),
            "return_format": request.return_format,
            "style": request.style,
        }

        return RagRetrieveResponse(**result)

    except ValueError as e:
        if "No valid documents with embeddings found" in str(e):
            raise HTTPException(status_code=500, detail="No valid document embeddings found. Please check repository content or retry.")
        raise HTTPException(status_code=500, detail=f"Error preparing retriever: {str(e)}")
    except Exception as e:
        if "All embeddings should be of the same size" in str(e):
            raise HTTPException(status_code=500, detail="Inconsistent embedding sizes detected. Please try again.")
        raise HTTPException(status_code=500, detail=f"Error in RAG retrieval: {str(e)}")


@app.get("/")
async def root():
    return {"status": "RAG retrieve API running"}


