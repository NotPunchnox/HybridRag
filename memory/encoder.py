from sentence_transformers import SentenceTransformer

# Déclarer le modèle d'embeddings au niveau du module pour éviter de le recharger à chaque insertion
# Modèle à charger pour les embeddings ("all-MiniLM-L6-v2" : https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

# /!\ Changer device="cpu" en device="cuda" si vous avez un GPU compatible.
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")