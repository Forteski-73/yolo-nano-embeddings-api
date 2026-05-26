# app.py (Trechos principais refatorados)
from fastapi import FastAPI
import uuid
import random
from datetime import datetime
from fastapi import Body
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue
from PIL import Image, ImageEnhance

from detector import detectar_prato
from utils import load_image_from_base64, load_image_from_url
from model import gerar_embedding_dinov2
from vector_db import qdrant_client, COLLECTION_NAME, inicializar_banco_vetorial
from models import TrainRequest, ImageRequest

from fastapi import Query
from datetime import datetime, timedelta
import traceback
from db import get_conn

import cv2
import base64
from io import BytesIO
from PIL import Image
import numpy as np

app = FastAPI(root_path="/AI")

@app.on_event("startup")
def startup_event():
    # Garante que o banco vetorial está pronto ao iniciar a API
    inicializar_banco_vetorial()

# =====================================================
# 🎨 FUNÇÃO DE DATA AUGMENTATION
# =====================================================
def aplicar_augmentations(img: Image.Image, variacao_index: int) -> Image.Image:
    """
    Aplica rotações e variações de iluminação para simular fotos reais.
    O 'variacao_index 0' sempre retorna a imagem original intacta.
    """
    if variacao_index == 0:
        return img # A primeira imagem salva é sempre a original perfeita

    # 1. Rotação aleatória (Crucial para pratos redondos)
    angulo = random.uniform(0, 360)
    img_aug = img.rotate(angulo, resample=Image.BICUBIC, expand=False, fillcolor="white")
    
    # 2. Variação leve de Brilho
    fator_brilho = random.uniform(0.8, 1.2)
    img_aug = ImageEnhance.Brightness(img_aug).enhance(fator_brilho)
    
    # 3. Variação leve de Contraste
    fator_contraste = random.uniform(0.85, 1.15)
    img_aug = ImageEnhance.Contrast(img_aug).enhance(fator_contraste)
    
    return img_aug

# =====================================================
# 🧠 TREINAMENTO (SALVAMENTO VETORIAL DIRETO)
# =====================================================
@app.post("/treinar")
def treinar(req: TrainRequest):
    try:
        # [Mantém a sua lógica original de carregar a imagem e passar pelo YOLO]
        img_original = load_image_from_base64(req.image_base64) if req.image_base64 else load_image_from_url(req.image_url)
        if img_original is None:
            return {"success": False, "message": "Imagem inválida."}
            
        img_recortada = detectar_prato(img_original)
        if img_recortada is None:
            return {"success": False, "message": "YOLO não detectou o objeto."}

        total_augmentations = req.augmentations or 10
        pontos_para_inserir = []

        # Loop de Data Augmentation (Mantendo suas rotações e filtros simulando ambiente real)
        for i in range(total_augmentations):
            nova_img = aplicar_augmentations(img_recortada, variacao_index=i) 
            
            # Extração profissional com DINOv2
            embedding = gerar_embedding_dinov2(nova_img)
            
            # Prepara o "ponto" para o Qdrant
            pontos_para_inserir.append(
                PointStruct(
                    id=str(uuid.uuid4()), # ID único para cada variação treinada
                    vector=embedding.tolist(),
                    payload={
                        "categoria": req.categoria,
                        "data_treino": datetime.utcnow().isoformat(),
                        "model_version": "dinov2_base_v1"
                    }
                )
            )

        # Inserção em lote (Bulk Upsert) -> Ultra rápido
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=pontos_para_inserir
        )

        return {
            "success": True,
            "message": f"Treinamento concluído. {total_augmentations} vetores indexados.",
            "categoria": req.categoria
        }

    except Exception as e:
        return {"success": False, "error_code": "TRAINING_ERROR", "message": str(e)}

# =====================================================
# 🔍 RECONHECIMENTO COM ALTA PRECISÃO
# =====================================================

# Função auxiliar para converter a imagem processada para Base64
def converter_para_base64(img_processada) -> str:
    # Se a imagem for do OpenCV (numpy array)
    if isinstance(img_processada, np.ndarray):
        _, buffer = cv2.imencode('.jpg', img_processada)
        return base64.b64encode(buffer).decode('utf-8')
    
    # Se a imagem for do PIL Image
    elif isinstance(img_processada, Image.Image):
        buffered = BytesIO()
        img_processada.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    raise ValueError("Formato de imagem não suportado para conversão em Base64")

@app.post("/analisarBase64")
def analisar_base64(req: ImageRequest):
    if not req.image_base64:
        return {"success": False, "message": "Base64 vazio."}

    try:
        img_original = load_image_from_base64(req.image_base64)
        img_processada = detectar_prato(img_original)
        
        if img_processada is None:
            return {"success": False, "message": "Nenhum prato/peça detectado pelo YOLO."}

        # 🔄 CONVERSÃO: Transforma o recorte do YOLO de volta para string Base64
        base64_processada = converter_para_base64(img_original)

        # 1. Gera o embedding da imagem enviada pelo cliente
        embedding_teste = gerar_embedding_dinov2(img_processada)

        # 1ª Tentativa: Busca estrita (Padrão Ouro)
        resposta_qdrant = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding_teste.tolist(),
            limit=3,
            score_threshold=0.80  # Rígido
        )

        # Se não retornou nada (lista vazia), entra o plano B: busca mais frouxa
        if not resposta_qdrant.points:
            print("Nenhum resultado com score >= 0.80. Tentando busca frouxa...")
            
            resposta_qdrant = qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=embedding_teste.tolist(),
                limit=3,              # Mantém os 3 melhores
                score_threshold=0.65  # Relaxa o limiar (pega decorações parecidas, mas não idênticas)
            )
        
        # Extrai a lista de pontos de dentro do objeto de resposta do Qdrant
        resultados_busca = resposta_qdrant.points

        if not resultados_busca:
            return {
                "success": True,
                "reconhecido": False,
                "imagem_processada": base64_processada,
                "message": "A imagem não pertence a nenhuma decoração catalogada."
            }

        # O primeiro elemento é o de maior score (similaridade mais próxima de 1.0)
        melhor_match = resultados_busca[0]

        return {
            "success": True,
            "reconhecido": True,
            "imagem_processada": base64_processada,
            "data": {
                "categoria_detectada": melhor_match.payload["categoria"],
                "porcentagem_similaridade": f"{round(melhor_match.score * 100, 2)}%", # Ex: "96.42%"
                "confianca": round(melhor_match.score * 100, 2), # Ex: 94.55%
                "ranking_proximidade": [
                    {"categoria": r.payload["categoria"], "confianca": round(r.score * 100, 2)}
                    for r in resultados_busca
                ]
            }
        }

    except Exception as e:
        return {"success": False, "error_code": "PROCESSING_ERROR", "message": str(e)}
    



# =====================================================
# 📋 LISTAR TODAS AS CATEGORIAS CADASTRADAS (QDRANT)
# =====================================================

@app.get("/listarCategorias")
def listar_categorias():
    try:
        categorias_unicas = set()
        offset = None

        # Faz um loop usando paginação (scroll) para garantir que lerá 
        # todos os pontos, mesmo se a base crescer muito
        while True:
            valores_scroll, proximo_offset = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                with_payload=True,     # Precisamos ler o payload para pegar o nome da categoria
                with_vectors=False,    # IMPORTANTE: False ignora os vetores (deixa a busca ultra rápida)
                limit=100,             # Lê de 100 em 100 pontos por lote
                offset=offset
            )

            # Extrai o nome da categoria do payload de cada ponto encontrado
            for ponto in valores_scroll:
                if ponto.payload and "categoria" in ponto.payload:
                    categorias_unicas.add(ponto.payload["categoria"])

            # Se não houver mais páginas (offset), encerra o loop
            if proximo_offset is None:
                break
                
            offset = proximo_offset

        # Converte o 'set' (que remove duplicados automaticamente) em uma lista ordenada
        lista_final = sorted(list(categorias_unicas))

        return {
            "success": True,
            "total_categorias": len(lista_final),
            "data": lista_final
        }

    except Exception as e:
        return {
            "success": False,
            "error_code": "FETCH_CATEGORIES_ERROR",
            "message": f"Erro ao listar categorias: {str(e)}"
        }

# =====================================================
# 🗑️ APAGAR CATEGORIA DO BANCO VETORIAL
# =====================================================
@app.delete("/excluirCategoria")
def excluir_categoria(categoria: str = Body(..., embed=True)):
    try:
        # Executa a deleção baseada em um filtro de payload
        resultado = qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="categoria",
                        match=MatchValue(value=categoria)
                    )
                ]
            )
        )
        
        return {
            "success": True,
            "message": f"Todos os vetores da categoria '{categoria}' foram removidos com sucesso.",
            "details": str(resultado)
        }

    except Exception as e:
        return {
            "success": False,
            "error_code": "DELETE_ERROR",
            "message": f"Erro ao excluir a categoria: {str(e)}"
        }
  
# Coloque a função de parse antes do endpoint
def parse_br_date(date_str: str) -> datetime:
    """Converte string no formato dd/mm/yyyy para objeto datetime."""
    return datetime.strptime(date_str, "%d/%m/%Y")

# =====================================================
# 📦 LISTAGEM DE IMAGENS RECENTES
# =====================================================
@app.get("/imagensRecentes")
def listar_imagens_recentes(
    start_date: str = Query(None),
    end_date: str = Query(None)
):
    try:
        # Certifique-se de que a função get_conn() está importada e configurada
        conn = get_conn()
        cursor = conn.cursor()

        # =================================================
        # DEFAULT: últimas 24h
        # =================================================
        if end_date:
            end_dt = parse_br_date(end_date)
        else:
            end_dt = datetime.now()

        if start_date:
            start_dt = parse_br_date(start_date)
        else:
            start_dt = end_dt - timedelta(days=1)

        # =================================================
        # Garante o range completo do dia para o SQL
        # =================================================
        start_dt = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end_dt = end_dt.replace(hour=23, minute=59, second=59, microsecond=999999)

        sql = """
        SELECT 
            product_id, 
            CONCAT('https://oxfordtec.com.br/Imagens/', image_path) AS full_image_url
        FROM product_image 
        WHERE updated_at BETWEEN %s AND %s
          AND finalidade = 'PRODUTO' 
          AND image_main = 1;
        """

        cursor.execute(sql, (start_dt, end_dt))
        rows = cursor.fetchall()

        lista_produtos = []

        for row in rows:
            lista_produtos.append({
                "product_id": row[0],
                "full_image_url": row[1]
            })

        cursor.close()
        conn.close()

        return {
            "success": True,
            "total_items": len(lista_produtos),
            "data": lista_produtos
        }

    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "error_code": "DATABASE_ERROR",
            "message": str(e)
        }