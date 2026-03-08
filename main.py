from memory.ingestion import ingest_text_file, ingest_conversation
from memory.retrieval import search
from memory.models import DocType

def display_results(results):
    if not results:
        print("  (aucun résultat)")
        return
    for r in results:
        tag = f"[{r.doc_type.value}]"
        print(f"  {tag:<16} score={r.score:.4f}  source={r.source}")
        print(f"  → {r.content[:100]}...")
        print()

if __name__ == "__main__":
    # Ajout des données dans la db vectorielle (à faire une seule fois, sinon doublons)
    ingest_text_file("data/docs/exemple_connaissances.txt")
    ingest_conversation("data/docs/conversation_history.json")

    # Simulation de prompts entrants
    prompts = [
        "Je veux créer une api avec Python. Par commencer ?"
    ]

    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"PROMPT : « {prompt} »")
        print(f"{'='*60}")

        print("\n[Tous types]")
        display_results(search(prompt, limit=3))

        print("[Documents uniquement]")
        display_results(search(prompt, limit=2, doc_type_filter=DocType.DOCUMENT))

        print("[Conversations uniquement]")
        display_results(search(prompt, limit=2, doc_type_filter=DocType.CONVERSATION))