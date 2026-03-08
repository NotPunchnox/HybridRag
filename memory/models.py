from dataclasses import dataclass, field
from enum import Enum

# Types de documents gérés par le système de mémoire (document/conversation)
class DocType(str, Enum):
    DOCUMENT   = "document"        # fichier texte, manuel, note
    CONVERSATION = "conversation"  # tour de dialogue passé

@dataclass
class Document:
    content: str
    doc_type: DocType
    source: str
    metadata: dict = field(default_factory=dict)

@dataclass
class SearchResult:
    id: int
    content: str
    source: str
    doc_type: DocType
    score: float
    matched_by: str # "fts", "vec", ou "hybrid"