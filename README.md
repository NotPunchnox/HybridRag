# Embedding Memory

Système de mémoire vectorielle local pour indexer et retrouver des documents et historiques de conversation via une recherche hybride (sémantique + plein texte).

## Stack technique

| Composant | Technologie |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) |
| Base de données | SQLite + extension `sqlite-vec` |
| Recherche plein texte | FTS5 (SQLite, tokenizer `unicode61`) |
| Recherche vectorielle | Similarité cosinus (vecteurs normalisés) |
| Fusion des résultats | RRF — Reciprocal Rank Fusion |
| Runtime | Python 3.13 |

## Installation

```bash
git clone https://github.com/notpunchnox/HybridRag.git
cd HybridRag/
pip install -r requirements.txt
```
*(**Attention** : dans les dernières versions de Linux, vous devez créer un environnement virtuel Python avec `python3 -m venv env` puis activer l'environnement avec `source env/bin/activate` avant d'installer les dépendances, sinon vous risquez d'avoir des problèmes de permissions ou de conflits de versions)*


## Structure
```
Embedding/
├─ data/
│  └─ docs/
│     ├─ conversation_history.json
│     └─ exemple_connaissances.txt
├─ db/
│  ├─ .gitkeep
│  └─ memory.db
├─ memory/
│  ├─ __init__.py
│  ├─ db.py
│  ├─ encoder.py
│  ├─ ingestion.py
│  ├─ models.py
│  └─ retrieval.py
├─ main.py
├─ README.md
└─ requirements.txt
```

## Utilisation

Pour lancer le script il suffit d'exécuter la commande suivante :
```bash
python main.py
```

Code d'exemple d'utilisation :

```py
from memory.ingestion import ingest_text_file, ingest_conversation
from memory.retrieval import search
from memory.models import DocType

# Indexation (à faire une seule fois — provoque des doublons si répété)
ingest_text_file("data/docs/exemple_connaissances.txt")
ingest_conversation("data/docs/conversation_history.json")

# Recherche
results = search("créer une API Python", limit=5)
results = search("créer une API Python", limit=5, doc_type_filter=DocType.DOCUMENT)
results = search("créer une API Python", limit=5, doc_type_filter=DocType.CONVERSATION)
```

#### Exemple de résultat de recherche
```bash
[ingestion] exemple_connaissances.txt → 7 chunks indexés
[ingestion] conversation_history.json → 8 tours indexés

============================================================
PROMPT : « Je veux créer une api avec Python. Par commencer ? »
============================================================

[Tous types]
  [conversation]   score=0.0164  source=conversation_history.json
  → Je dois créer une API REST en Python. Par où je commence?...

  [conversation]   score=0.0161  source=conversation_history.json
  → Bonjour! Je peux vous aider avec des questions de programmation, expliquer des concepts techniques, ...

  [conversation]   score=0.0159  source=conversation_history.json
  → Je recommande d utiliser Flask ou FastAPI. FastAPI est plus moderne et performant. Vous auriez besoi...

[Documents uniquement]
  [document]       score=0.0156  source=exemple_connaissances.txt
  → # Connaissances Python...

[Conversations uniquement]
  [conversation]   score=0.0164  source=conversation_history.json
  → Je dois créer une API REST en Python. Par où je commence?...

  [conversation]   score=0.0161  source=conversation_history.json
  → Bonjour! Je peux vous aider avec des questions de programmation, expliquer des concepts techniques, ...
```

*/!\ Dans le cas d'une conversation le logiciel aura plus de chance de retrouver la requête que la réponse, c'est pour ça qu'un système d'id est mis en place, si vous souhaitez retrouvez la réponse en + de la requête (ce qui est plus pertinent pour aider le modèle à répondre convenablement) vous pouvez faire en sorte de récupérer **(si speaker == "user" alors: l'id du document trouvé +1)**, afin de récupérer automatiquement la réponse du modèle à la question*

## Format des données sources
**Fichier texte** : `(.txt)` : découpé en chunks par double saut de ligne `(\n\n)`.

**Fichier conversation** : `(.json)` : format JSON avec des objets représentant les tours de conversation.
```json
[
    { "id": 1, "speaker": "user", "message": "..." },
    { "id": 2, "speaker": "ai",   "message": "..." }
]
```

## Base de données SQLite

La base de données `memory.db` contient trois tables principales :
- `documents` : contenu, type (`document` ou `conversation`), source, métadonnées.
- `docs_fts` : index FTS5 pour la recherche plein texte.(synchronisé automatiquement via trigger*)*.
- `docs_vec` : index vectoriel *(embeddings float32, 384 dim)*.

## Recherche Hybride
1. **FTS5** : recherche plein texte BM25 pour trouver les documents les plus pertinents.
2. **Vectorielle** : similarité cosinus sur embeddings pour trouver les documents les plus proches sémantiquement.
3. **RRF** : fusion des deux listes de résultats pour obtenir un classement final.

*Dans ce projet, la recherche vectorielle se fait à partir de **distance angulaire** (cosinus) sur des vecteurs normalisés, ce qui est plus pertinent pour ce type de données que la **distance euclidienne**.*

$\cos\theta = \frac{\vec{u} \cdot \vec{v}}{||\vec{u}|| \cdot ||\vec{v}||}$


## Copyright & License
Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails

Auteur: [NotPunchnox](https://github.com/notpunchnox)