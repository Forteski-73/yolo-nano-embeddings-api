# vector_db.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

COLLECTION_NAME = "decoracoes_pratos"

# =========================================================================
# CONFIGURAÇÃO SEM DOCKER (MODO LOCAL)
# O Qdrant criará automaticamente uma pasta chamada "qdrant_local" no seu 
# projeto. Os dados do treino ficarão salvos de forma persistente ali dentro.
# =========================================================================
qdrant_client = QdrantClient(path="./qdrant_local")

def inicializar_banco_vetorial():
    """
    Garante que a coleção de vetores existe com a configuração correta de distância.
    """
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        print(f"[QDRANT LOCAL]: Criando coleção '{COLLECTION_NAME}'...")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            # Mantemos o tamanho do DINOv2 (768) e Distância de Cosseno para alta precisão
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )