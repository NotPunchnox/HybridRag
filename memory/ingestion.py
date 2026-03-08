import json
import numpy as np
from pathlib import Path
from .db import get_connection, init_db
from .models import Document, DocType
from .encoder import model

def _insert(conn, doc: Document) -> int:
    cur = conn.execute(
        """INSERT INTO documents (content, doc_type, source, metadata)
           VALUES (?, ?, ?, ?)""",
        (doc.content, doc.doc_type.value, doc.source, json.dumps(doc.metadata))
    )
    doc_id = cur.lastrowid

    # Requête convertie en Embedding
    # TODO: inférence avec rkllama
    embedding = model.encode(doc.content, normalize_embeddings=True)
    conn.execute(
        "INSERT INTO docs_vec (rowid, embedding) VALUES (?, ?)",
        (doc_id, embedding.astype(np.float32).tobytes())
    )
    return doc_id


def ingest_text_file(filepath: str | Path):
    """Charge un fichier .txt, le découpe en paragraphes et l'indexe."""
    path = Path(filepath)
    text = path.read_text(encoding="utf-8")

    # Découpe par paragraphes (séparés par ligne vide) (chunk)
    chunks = [p.strip() for p in text.split("\n\n") if p.strip()]

    conn = get_connection()
    init_db(conn)

    for i, chunk in enumerate(chunks):
        doc = Document(
            content=chunk,
            doc_type=DocType.DOCUMENT,
            source=path.name,
            metadata={"chunk_index": i, "total_chunks": len(chunks)}
        )
        _insert(conn, doc)

    conn.commit()
    conn.close()
    print(f"[ingestion] {path.name} → {len(chunks)} chunks indexés")


def ingest_conversation(filepath: str | Path):
    """
    Charge un historique de conversation JSON.
    """
    path = Path(filepath)
    turns = json.loads(path.read_text(encoding="utf-8"))

    conn = get_connection()
    init_db(conn)

    # En cas d'ajout de nouvelles conversations avec des clés JSON différentes, faire attention à modifier les clés ci-dessous (speaker, message, id)
    for turn in turns:
        doc = Document(
            content=turn["message"],
            doc_type=DocType.CONVERSATION,
            source=path.name,
            metadata={"speaker": turn["speaker"], "turn": turn["id"]}
        )
        _insert(conn, doc)

    conn.commit()
    conn.close()
    print(f"[ingestion] {path.name} → {len(turns)} tours indexés")


def ingest_single_message(speaker: str, text: str, turn: int):
    """Indexe un seul message en temps réel (pendant une conversation live)."""
    conn = get_connection()
    init_db(conn)

    doc = Document(
        content=text,
        doc_type=DocType.CONVERSATION,
        source="live_conversation",
        metadata={"speaker": speaker, "turn": turn}
    )
    doc_id = _insert(conn, doc)
    conn.commit()
    conn.close()
    return doc_id