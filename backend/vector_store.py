"""
Vector Store — RAG Memory over Past Analyses
──────────────────────────────────────────────
Uses ChromaDB for local vector storage and Gemini embeddings
to enable semantic search over past search results.
"""

import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

_collection = None
_embed_model = None


def _get_collection():
    """Lazy-init ChromaDB collection."""
    global _collection
    if _collection is None:
        try:
            import chromadb
            db_path = os.path.join(os.path.dirname(__file__), "chroma_data")
            client = chromadb.PersistentClient(path=db_path)
            _collection = client.get_or_create_collection(
                name="news_memory",
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            print(f"[VectorStore] ChromaDB init failed: {e}")
            return None
    return _collection


def _embed_text(text: str) -> list[float]:
    """Generate embedding using Gemini's embedding model."""
    global _embed_model
    try:
        import google.generativeai as genai
        if _embed_model is None:
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            _embed_model = True  # flag that we've configured
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text[:8000],
        )
        return result["embedding"]
    except Exception as e:
        print(f"[VectorStore] Embedding failed: {e}")
        return None


def index_search(search_id: int, query: str, analysis: str, metadata: dict = None):
    """
    Index a search result into the vector store.
    Combines query + analysis for the embedding.
    """
    collection = _get_collection()
    if collection is None:
        return False

    text = f"Query: {query}\n\nAnalysis: {analysis[:3000]}"
    embedding = _embed_text(text)
    if embedding is None:
        return False

    doc_id = f"search_{search_id}"
    meta = {
        "search_id": search_id,
        "query": query[:200],
        "confidence": metadata.get("confidence", 0) if metadata else 0,
        "sentiment": metadata.get("sentiment", "neutral") if metadata else "neutral",
    }

    try:
        # Upsert to handle re-indexing
        collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text[:5000]],
            metadatas=[meta],
        )
        return True
    except Exception as e:
        print(f"[VectorStore] Index failed: {e}")
        return False


def query_memory(question: str, top_k: int = 5) -> list[dict]:
    """
    Semantic search over past analyses.
    Returns top_k most relevant past results.
    """
    collection = _get_collection()
    if collection is None:
        return []

    embedding = _embed_text(question)
    if embedding is None:
        return []

    try:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, 10),
        )

        items = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0
                items.append({
                    "document": doc[:500],
                    "search_id": meta.get("search_id"),
                    "query": meta.get("query", ""),
                    "confidence": meta.get("confidence", 0),
                    "sentiment": meta.get("sentiment", "neutral"),
                    "similarity": round(1 - distance, 3),  # cosine distance → similarity
                })
        return items
    except Exception as e:
        print(f"[VectorStore] Query failed: {e}")
        return []


def get_memory_stats() -> dict:
    """Return stats about the vector store."""
    collection = _get_collection()
    if collection is None:
        return {"count": 0, "status": "unavailable"}
    try:
        return {"count": collection.count(), "status": "active"}
    except Exception:
        return {"count": 0, "status": "error"}
