# =====================================================
# app.py
# =====================================================

from fastapi import FastAPI

from detector import detectar_prato

from db import get_conn

from model import gerar_embedding

from utils import (
    load_image_from_url,
    load_image_from_base64,
    image_to_base64
)

from features import (
    hsv_histogram,
    texture_lbp,
    edge_density,
    symmetry_score
)

from similarity import calcular_similaridade

from models import (
    TrainRequest,
    ImageRequest,
    UrlImageRequest
)

from PIL import (
    Image,
    ImageEnhance,
    ImageFilter
)

from fastapi import Query
from datetime import datetime, timedelta

import numpy as np
import random
import json
import traceback

app = FastAPI(
    root_path="/AI"
)

# =====================================================
# 🔍 ANÁLISE BASE64
# =====================================================
@app.post("/analisarBase64")
def analisar_base64(req: ImageRequest):

    print("===================================================")
    print("BASE64 RECEBIDO (TAMANHO):", len(req.image_base64))
    print("BASE64 INICIO:", req.image_base64[:200])
    print("BASE64 FIM:", req.image_base64[-200:])
    print("===================================================")

    if not req.image_base64:

        return {
            "success": False,
            "message": "Base64 vazio."
        }

    img_original = load_image_from_base64(
        req.image_base64
    )

    if img_original is None:

        return {
            "success": False,
            "error_code": "INVALID_BASE64",
            "message": "Imagem inválida ou corrompida."
        }

    try:

        # ============================================
        # SEGMENTAÇÃO YOLO
        # ============================================
        img_processada = detectar_prato(
            img_original
        )

        # ============================================
        # OBJETO INVÁLIDO
        # ============================================
        if img_processada is None:

            return {
                "success": False,
                "error_code": "INVALID_OBJECT",
                "message": (
                    "Nenhum prato, bowl, cup, mug "
                    "ou vase foi encontrado."
                )
            }

        # ============================================
        # ANALISA SIMILARIDADE
        # ============================================
        result = calcular_similaridade(
            img_processada
        )

        # ============================================
        # BASE64 PROCESSADA
        # ============================================
        img_base64 = image_to_base64(
            img_processada
        )

        # ============================================
        # RETORNO
        # ============================================
        #result["imagem_processada"] = req.image_base64
        result["imagem_processada"] = img_base64

        return {
            "success": True,
            "data": result
        }

    except Exception as e:

        traceback.print_exc()

        return {
            "success": False,
            "error_code": "PROCESSING_ERROR",
            "message": str(e)
        }

# =====================================================
# 🔍 ANÁLISE URL
# =====================================================
@app.post("/analisarUrl")
def analisar_url(req: UrlImageRequest):

    img = load_image_from_url(
        req.image_url
    )

    if img is None:

        return {
            "success": False,
            "error_code": "IMAGE_NOT_ACCESSIBLE",
            "message": (
                "Não foi possível acessar a imagem. "
                "Verifique o link ou permissões."
            )
        }

    result = calcular_similaridade(img)

    return {
        "success": True,
        "data": result
    }

# =====================================================
# 🧠 TREINAMENTO
# =====================================================
@app.post("/treinar")
def treinar(req: TrainRequest):

    try:

        # =================================================
        # CARREGA IMAGEM
        # =================================================
        img = None

        # ================================================
        # URL
        # ================================================
        if req.image_url:

            print("[TREINAMENTO]: carregando via URL")

            img = load_image_from_url(
                req.image_url
            )

        # ================================================
        # BASE64
        # ================================================
        elif req.image_base64:

            print("[TREINAMENTO]: carregando via BASE64")

            img = load_image_from_base64(
                req.image_base64
            )

        # ================================================
        # NENHUMA IMAGEM
        # ================================================
        else:

            return {
                "success": False,
                "error_code": "IMAGE_REQUIRED",
                "message": (
                    "Informe image_url ou image_base64."
                )
            }

        # ================================================
        # FALHA AO CARREGAR
        # ================================================
        if img is None:

            return {
                "success": False,
                "error_code": "IMAGE_NOT_ACCESSIBLE",
                "message": (
                    "Não foi possível carregar a imagem."
                )
            }
        # =================================================
        # YOLO SEGMENTATION
        # =================================================
        img = detectar_prato(img)

        # 👇 COLE ESTAS LINHAS AQUI ANTES DA LINHA 295 👇
        if img is None:
            print("[ERRO CRÍTICO]: Cancelando o treino porque o YOLO não gerou máscara.")
            return {
                "success": False,
                "error_code": "OBJECT_NOT_DETECTED",
                "message": "Não foi possível detectar um prato nesta imagem. Treinamento cancelado."
            }
        # 👆 SE VIER NONE, O CÓDIGO PARA AQUI E NÃO QUEBRA O SERVIDOR 👆

        # =================================================
        # BANCO
        # =================================================
        conn = get_conn()

        cursor = conn.cursor()

        # =================================================
        # AUGMENTATIONS
        # =================================================
        total_augmentations = (
            req.augmentations or 20
        )

        for i in range(total_augmentations):

            # =============================================
            # CÓPIA
            # =============================================
            nova = img.copy()

            # =============================================
            # ROTAÇÃO 
            # =============================================
            nova = nova.rotate(
                random.uniform(-12, 12),
                expand=True,
                fillcolor="black" 
            )

            # =============================================
            # BRIGHTNESS
            # =============================================
            nova = ImageEnhance.Brightness(
                nova
            ).enhance(
                random.uniform(0.85, 1.15)
            )

            # =============================================
            # CONTRASTE
            # =============================================
            nova = ImageEnhance.Contrast(
                nova
            ).enhance(
                random.uniform(0.90, 1.10)
            )

            # =============================================
            # SATURAÇÃO
            # =============================================
            nova = ImageEnhance.Color(
                nova
            ).enhance(
                random.uniform(0.90, 1.10)
            )

            # =============================================
            # BLUR LEVE
            # =============================================
            if random.random() < 0.30:

                nova = nova.filter(
                    ImageFilter.GaussianBlur(
                        radius=0.5
                    )
                )

            # =============================================
            # RUÍDO LEVE
            # =============================================
            if random.random() < 0.30:

                arr = np.array(nova)

                noise = np.random.normal(
                    0,
                    5,
                    arr.shape
                )

                arr = np.clip(
                    arr + noise,
                    0,
                    255
                ).astype(np.uint8)

                nova = Image.fromarray(arr)

            # =============================================
            # REDIMENSIONA
            # =============================================
            nova.thumbnail((640, 640))

            # =============================================
            # FUNDO BRANCO agora é PRETO
            # =============================================
            
            """fundo = Image.new(
                "RGB",
                (640, 640),
                (255, 255, 255)
            )"""
            fundo = Image.new(
                "RGB",
                (640, 640),
                (0, 0, 0) 
            )

            x = (
                640 - nova.width
            ) // 2

            y = (
                640 - nova.height
            ) // 2

            fundo.paste(
                nova,
                (x, y)
            )

            nova = fundo

            # =============================================
            # FEATURES
            # =============================================
            emb = gerar_embedding(nova)

            hsv = hsv_histogram(nova)

            lbp = texture_lbp(nova)

            edge = edge_density(nova)

            sym = symmetry_score(nova)

            # =============================================
            # INSERT
            # =============================================
            sql = """
            INSERT INTO image_embeddings
            (
                categoria,
                embedding_model,
                embedding,
                hsv_histogram,
                texture_lbp,
                edge_density,
                symmetry_score
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            """

            cursor.execute(

                sql,

                (

                    req.categoria,

                    req.embedding_model,

                    json.dumps(
                        emb.tolist()
                    ),

                    json.dumps(hsv),

                    json.dumps(lbp),

                    float(edge),

                    float(sym)

                )
            )

            print(
                f"[{req.embedding_model}] "
                f"TREINAMENTO: ⚙️ -> "
                f"{i+1}/{total_augmentations}"
            )

        # =================================================
        # COMMIT
        # =================================================
        conn.commit()

        cursor.close()

        conn.close()

        return {
            "success": True,
            "status": "ok",
            "msg": "Treinamento concluído",
            "categoria": req.categoria,
            "embedding_model": req.embedding_model,
            "imagens_geradas": total_augmentations
        }

    except Exception as e:

        traceback.print_exc()

        return {
            "success": False,
            "error_code": "TRAINING_ERROR",
            "message": str(e)
        }
    

# =====================================================
# 📦 LISTAGEM DE IMAGENS RECENTES
# =====================================================
@app.get("/imagensRecentes")
def listar_imagens_recentes(
    start_date: str = Query(None),
    end_date: str = Query(None)
):
    try:
        conn = get_conn()
        cursor = conn.cursor()

        # =================================================
        # DEFAULT: últimas 24h
        # =================================================
        if not end_date:
            end_date = datetime.now()

        if not start_date:
            start_date = end_date - timedelta(days=1)

        # =================================================
        # garante formato SQL
        # =================================================
        if isinstance(end_date, str):
            end_date = parse_br_date(end_date)

        if isinstance(start_date, str):
            start_date = parse_br_date(start_date)

        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date   = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)

        sql = """
        SELECT 
            product_id, 
            CONCAT('https://oxfordtec.com.br/Imagens/', image_path) AS full_image_url
        FROM product_image 
        WHERE updated_at BETWEEN %s AND %s
          AND finalidade = 'PRODUTO' 
          AND image_main = 1;
        """

        cursor.execute(sql, (start_date, end_date))
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
    
def parse_br_date(date_str: str):
    # formato: dd/mm/yyyy
    return datetime.strptime(date_str, "%d/%m/%Y")