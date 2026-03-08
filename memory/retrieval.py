import numpy as np
from .db import get_connection
from .models import SearchResult, DocType
from .encoder import model

# Recherche FTS5
def _fts_search(conn, query: str, limit: int) -> list[SearchResult]:
    clean_query = " ".join(query.split())
    try:
        rows = conn.execute("""
            SELECT d.id, d.content, d.source, d.doc_type,
                   bm25(docs_fts) AS score
            FROM docs_fts
            JOIN documents d ON d.id = docs_fts.rowid
            WHERE docs_fts MATCH ?
            ORDER BY score
            LIMIT ?
        """, (clean_query, limit)).fetchall()
    except Exception:
        return []

    return [
        SearchResult(
            id=r["id"], content=r["content"],
            source=r["source"], doc_type=DocType(r["doc_type"]),
            score=-r["score"],
            matched_by="fts"
        ) for r in rows
    ]

# Recherche vectorielle avec SQLite-Vec
def _vec_search(conn, query: str, limit: int) -> list[SearchResult]:
    embedding = model.encode(query, normalize_embeddings=True)

    rows = conn.execute("""
        SELECT dv.rowid, d.content, d.source, d.doc_type, dv.distance
        FROM docs_vec dv
        JOIN documents d ON d.id = dv.rowid
        WHERE embedding MATCH ?
          AND k = ?
    """, (embedding.astype(np.float32).tobytes(), limit)).fetchall()

    return [
        SearchResult(
            id=r["rowid"], content=r["content"],
            source=r["source"], doc_type=DocType(r["doc_type"]),
            score=round(1 - r["distance"] / 2, 4),  # distance → similarité
            matched_by="vec"
        ) for r in rows
    ]

# Fusion des résultats FTS5 et vectoriels avec RRF
def _rrf(fts: list[SearchResult], vec: list[SearchResult], k=60) -> list[SearchResult]:
    scores: dict[int, float] = {}
    index: dict[int, SearchResult] = {}

    for rank, r in enumerate(fts):
        scores[r.id] = scores.get(r.id, 0) + 1 / (k + rank + 1)
        index[r.id] = r

    for rank, r in enumerate(vec):
        scores[r.id] = scores.get(r.id, 0) + 1 / (k + rank + 1)
        if r.id not in index:
            index[r.id] = r

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for doc_id, score in ranked:
        r = index[doc_id]
        results.append(SearchResult(
            id=r.id, content=r.content, source=r.source,
            doc_type=r.doc_type, score=round(score, 6),
            matched_by="hybrid"
        ))
    return results

# Fonction de recherche publique combinant FTS5 et recherche vectorielle, avec option de filtrage par type de document
def search(query: str, limit: int = 5, doc_type_filter: DocType = None) -> list[SearchResult]:
    conn = get_connection()

    fts_results = _fts_search(conn, query, limit=limit * 2)
    vec_results = _vec_search(conn, query, limit=limit * 2)

    results = _rrf(fts_results, vec_results)

    # Filtre optionnel par type
    if doc_type_filter:
        results = [r for r in results if r.doc_type == doc_type_filter]

    conn.close()
    return results[:limit]