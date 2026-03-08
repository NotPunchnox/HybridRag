import sqlite3
import sqlite_vec
from pathlib import Path

EMBEDDING_DIM = 384
DB_PATH = Path(__file__).parent.parent / "db" / "memory.db"

def get_connection() -> sqlite3.Connection:
    # Se connecter à la base de données SQLite
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # Charger l'extension vec pour les recherches vectorielles
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    
    # Retourner la connexion pour être utilisée dans les autres fonctions
    return conn

def init_db(conn: sqlite3.Connection):
    # Crée les tables nécessaires si elles n'existent pas déjà
    conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS documents (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            content    TEXT NOT NULL,
            doc_type   TEXT NOT NULL,
            source     TEXT NOT NULL,
            metadata   TEXT DEFAULT '{{}}',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
            content,
            source,
            content='documents',
            content_rowid='id',
            tokenize='unicode61'   -- gère les accents
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS docs_vec USING vec0(
            embedding FLOAT[{EMBEDDING_DIM}]
        );

        -- Sync automatique FTS5
        CREATE TRIGGER IF NOT EXISTS docs_ai AFTER INSERT ON documents BEGIN
            INSERT INTO docs_fts(rowid, content, source)
            VALUES (new.id, new.content, new.source);
        END;
        CREATE TRIGGER IF NOT EXISTS docs_ad AFTER DELETE ON documents BEGIN
            INSERT INTO docs_fts(docs_fts, rowid, content, source)
            VALUES ('delete', old.id, old.content, old.source);
        END;
    """)
    conn.commit()